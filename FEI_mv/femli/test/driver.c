#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "util/mli_utils.h"
#include "base/mli_defs.h"
#include "cintface/cmli.h"
#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "krylov/krylov.h"
#define habs(x) ((x)>=0 ? x : -x)

HYPRE_ParCSRMatrix GenerateConvectionDiffusion(MPI_Comm,
                      int nx, int ny, int nz, int P, int Q, int R,
                      int p, int q, int r, double alpha, double beta,
                      double  *value );

int main(int argc, char **argv)
{

   int                j, nx=7, ny=7, nz=3, P, Q, R, p, q, r, nprocs;
   int                mypid, startRow;
   int                *partition, globalSize, localSize, nsweeps, rowSize;
   int                *colInd, k, *procCnts, *offsets, *rowCnts, ftype;
   int                ndofs=3, nullDim=6, testProb=0, solver=2, scaleFlag=0;
   int                fleng, rleng, status;
   char               *targv[10], fname[100], rhsFname[100], methodName[10];
   double             *values, *nullVecs, *scaleVec, *colVal, *gscaleVec;
   double             *rhsVector=NULL, alpha, beta, *weights;
   HYPRE_IJMatrix     newIJA;
   HYPRE_IJVector     IJrhs;
   HYPRE_ParCSRMatrix HYPREA;
   hypre_ParCSRMatrix *hypreA;
   hypre_ParVector    *sol, *rhs;
   CMLI               *cmli;
   CMLI_Matrix        *cmliMat;
   CMLI_Method        *cmliMethod;
   CMLI_Vector        *csol, *crhs;
   MLI_Function       *funcPtr;

   /* ------------------------------------------------------------- *
    * machine setup
    * ------------------------------------------------------------- */

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

   if ( testProb == 0 )
   {
      P = 1;
      Q = nprocs;
      R = 1;
      p = mypid % P;
      q = (( mypid - p)/P) % Q;
      r = ( mypid - p - P*q)/( P*Q );
      values = (double *) calloc(4, sizeof(double));
      values[3] = -0.0;
      values[2] = -1.0;
      values[1] = -2.0;
      values[0] = 6.0;
      alpha     = 240.0;
      beta      = 120.0;
      HYPREA = (HYPRE_ParCSRMatrix) GenerateConvectionDiffusion(MPI_COMM_WORLD, 
                           nx, ny, nz, P, Q, R, p, q, r, alpha, beta, values); 
      free( values );
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
      if ( scaleFlag ) 
      {
         scaleVec  = (double *) malloc(localSize*sizeof(double));
         gscaleVec = (double *) malloc(globalSize*sizeof(double));
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            scaleVec[j-startRow] = 0.0;
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for (k = 0; k < rowSize; k++)
               if ( colInd[k] == j ) scaleVec[j-startRow] = colVal[k]; 
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         for (j = 0; j < localSize; j++) 
         {
            if ( scaleVec[j] <= 0.0 ) 
            {
               printf("Proc %d : diag %d = %e <= 0 \n",mypid,j,scaleVec[j]); 
               exit(1);
            }
            scaleVec[j] = 1.0/sqrt(scaleVec[j]); 
         }
         procCnts = (int *) malloc( nprocs * sizeof(int) );
         offsets   = (int *) malloc( nprocs * sizeof(int) );
         MPI_Allgather(&localSize,1,MPI_INT,procCnts,1,MPI_INT,MPI_COMM_WORLD);
         offsets[0] = 0;
         for ( j = 1; j < nprocs; j++ )
            offsets[j] = offsets[j-1] + procCnts[j-1];
         MPI_Allgatherv(scaleVec, localSize, MPI_DOUBLE, gscaleVec,
                        procCnts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         free( procCnts );
         free( offsets );
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, startRow, startRow+localSize-1,
                              startRow, startRow+localSize-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         rowCnts = (int *) malloc( localSize * sizeof(int) );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,NULL);
            rowCnts[j-startRow] = rowSize;
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, rowCnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( rowCnts );
         for ( j = startRow; j < startRow+localSize; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPREA,j,&rowSize,&colInd,&colVal);
            for ( k = 0; k < rowSize; k++ ) 
            {
               colVal[k] = colVal[k] * gscaleVec[colInd[k]] * gscaleVec[j];
               if ( colInd[k] == j && habs(colVal[k]-1.0) > 1.0e-8 )
                  printf("Proc %d : diag %d(%d) = %e != 1.0\n",mypid,j,k,
                         colVal[k]);
            }
            HYPRE_IJMatrixSetValues(newIJA, 1, &rowSize, (const int *) &j,
                   (const int *) colInd, (const double *) colVal);
            HYPRE_ParCSRMatrixRestoreRow(HYPREA,j,&rowSize,&colInd,&colVal);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPREA);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPREA);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( gscaleVec );
         nullVecs = ( double *) malloc(localSize * sizeof(double));
         for ( j = 0; j < localSize; j++ ) nullVecs[j] = 1.0 / scaleVec[j]; 
         free( scaleVec );
      } else nullVecs = NULL;
   }
   else if ( testProb == 1 )
   {
      if ( mypid == 0 )
      {
         printf("Matrix file name : ");
         scanf("%s", fname);
         fleng = strlen(fname);
         fleng++;
         printf("\nMatrix file type (0 - Tumin, 1 - IJA) : ");
         scanf("%d", &ftype);
         printf("rhs file name : ");
         scanf("%s", rhsFname);
         rleng = strlen(rhsFname);
         rleng++;
      }
      MPI_Bcast(&fleng, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ftype, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&rleng, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(fname, fleng, MPI_CHAR, 0, MPI_COMM_WORLD);
      MPI_Bcast(rhsFname, rleng, MPI_CHAR, 0, MPI_COMM_WORLD);
      if ( ftype == 0 )
      {
         MLI_Utils_HypreMatrixReadTuminFormat(fname,MPI_COMM_WORLD,ndofs,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
      }
      else
      {
/*
         MLI_Utils_HypreMatrixReadIJAFormat(fname,MPI_COMM_WORLD,ndofs,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
*/
         MLI_Utils_HypreParMatrixReadIJAFormat(fname,MPI_COMM_WORLD,
                     (void **) &HYPREA, scaleFlag, &scaleVec);
      }
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
      globalSize = partition[nprocs];
      startRow   = partition[mypid];
      localSize  = partition[mypid+1] - startRow;
      free( partition );
      rhsVector = ( double *) malloc(localSize * sizeof(double));
      status = MLI_Utils_DoubleParVectorRead(rhsFname, MPI_COMM_WORLD, 
                               localSize, startRow, rhsVector);
      if ( status < 0 )
      {
         free( rhsVector );
         rhsVector = NULL;
      }
      nullVecs = ( double *) malloc(localSize * nullDim * sizeof(double));
/*
      MLI_Utils_DoubleParVectorRead("rigid_body_mode01",MPI_COMM_WORLD,
                              localSize, startRow, nullVecs);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode02",MPI_COMM_WORLD,
                              localSize, startRow, &nullVecs[localSize]);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode03",MPI_COMM_WORLD,
                              localSize, startRow, &nullVecs[localSize*2]);
*/
      MLI_Utils_DoubleParVectorRead("rigid_body_mode01",MPI_COMM_WORLD,
                              localSize, 0, nullVecs);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode02",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize]);
      MLI_Utils_DoubleParVectorRead("rigid_body_mode03",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*2]);
      if ( scaleFlag )
      {
         for ( j = 0; j < localSize; j++ ) 
            scaleVec[j] = sqrt(scaleVec[j]); 
         for ( j = 0; j < localSize; j++ ) nullVecs[j] *= scaleVec[j]; 
         for ( j = 0; j < localSize; j++ ) 
            nullVecs[localSize+j] *= scaleVec[j]; 
         for ( j = 0; j < localSize; j++ ) 
            nullVecs[2*localSize+j] *= scaleVec[j]; 
      }
      if ( nullDim > 3 )
      {
         MLI_Utils_DoubleParVectorRead("rigid_body_mode04",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*3]);
         MLI_Utils_DoubleParVectorRead("rigid_body_mode05",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*4]);
         MLI_Utils_DoubleParVectorRead("rigid_body_mode06",MPI_COMM_WORLD,
                              localSize, 0, &nullVecs[localSize*5]);
         if ( scaleFlag )
         {
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[3*localSize+j] *= scaleVec[j]; 
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[4*localSize+j] *= scaleVec[j]; 
            for ( j = 0; j < localSize; j++ ) 
               nullVecs[5*localSize+j] *= scaleVec[j]; 
         }
      }
   }

   hypreA = (hypre_ParCSRMatrix *) HYPREA;
   HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
   sol = hypre_ParVectorCreate(MPI_COMM_WORLD, globalSize, partition);
   hypre_ParVectorInitialize( sol );
   hypre_ParVectorSetConstantValues( sol, 0.0 );

   HYPRE_ParCSRMatrixGetRowPartitioning(HYPREA, &partition);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, partition[mypid],
                               partition[mypid+1]-1, &IJrhs);
   free( partition );
   HYPRE_IJVectorSetObjectType(IJrhs, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(IJrhs);
   HYPRE_IJVectorAssemble(IJrhs);
   if ( rhsVector != NULL )
   {
      colInd = (int *) malloc(localSize * sizeof(double));
      for (j = 0; j < localSize; j++) colInd[j] = startRow + j;
      HYPRE_IJVectorSetValues(IJrhs, localSize, (const int *) colInd,
                              (const double *) rhsVector);
   }
   HYPRE_IJVectorGetObject(IJrhs, (void*) &rhs);
   HYPRE_IJVectorSetObjectType(IJrhs, -1);
   HYPRE_IJVectorDestroy(IJrhs);
   if ( rhsVector == NULL )
      hypre_ParVectorSetConstantValues( rhs, 1.0 );
   else
      free( rhsVector );

   funcPtr = (MLI_Function *) malloc( sizeof( MLI_Function ) );
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   csol = MLI_VectorCreate(sol, "HYPRE_ParVector", funcPtr);
   crhs = MLI_VectorCreate(rhs, "HYPRE_ParVector", funcPtr);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   cmliMat = MLI_MatrixCreate((void*) hypreA,"HYPRE_ParCSR",funcPtr);
   free( funcPtr );
   cmli = MLI_Create( MPI_COMM_WORLD );
   strcpy( methodName, "AMGRS" );
   cmliMethod = MLI_MethodCreate( methodName, MPI_COMM_WORLD );
   MLI_MethodSetParams( cmliMethod, "setOutputLevel 2", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setNumLevels 2", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setMinCoarseSize 5", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setCoarseSolver SuperLU", 2, targv );
   if ( ! strcmp(methodName, "AMGRS") )
   {
      nsweeps = 2;
      weights = (double *) malloc( sizeof(double)*nsweeps );
      for ( j = 0; j < nsweeps; j++ ) weights[j] = 0.05;
      targv[0] = (char *) &nsweeps;
      targv[1] = (char *) weights;
      MLI_MethodSetParams( cmliMethod, "setPreSmoother SGS", 2, targv );
      MLI_MethodSetParams( cmliMethod, "setPostSmoother SGS", 2, targv );
      free(weights);
      MLI_MethodSetParams( cmliMethod, "setCoarsenScheme ruge", 0, NULL );
      MLI_MethodSetParams( cmliMethod, "setStrengthThreshold 0.0", 0, NULL );
/*
      MLI_MethodSetParams( cmliMethod, "setSmootherPrintRNorm", 0, NULL );
*/
      MLI_MethodSetParams( cmliMethod, "setSmootherFindOmega", 0, NULL );
/*
*/
/*
*/
/*
      MLI_MethodSetParams( cmliMethod, "useInjectionForR", 0, NULL );
MLI_MethodSetParams( cmliMethod, "nonsymmetric", 0, NULL );
*/
   }
   else
   {
      nsweeps = 2;
      weights = (double *) malloc( sizeof(double)*nsweeps );
      for ( j = 0; j < nsweeps; j++ ) weights[j] = 0.1;
      targv[0] = (char *) &nsweeps;
      targv[1] = (char *) weights;
      MLI_MethodSetParams( cmliMethod, "setPreSmoother SGS", 2, targv );
      MLI_MethodSetParams( cmliMethod, "setPostSmoother SGS", 2, targv );
      free(weights);
      MLI_MethodSetParams( cmliMethod, "setPweight 0.0", 0, NULL );
      MLI_MethodSetParams( cmliMethod, "setStrengthThreshold 0.08", 0, NULL );
      MLI_MethodSetParams( cmliMethod, "setCalibrationSize 0", 0, NULL );
   }
   nsweeps = 1;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) NULL;
/*
   MLI_MethodSetParams( cmliMethod, "setSmootherPrintRNorm", 0, NULL );
*/
   if ( testProb == 0 )
   {
      ndofs    = 1;
      nullDim  = 1;
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
      if ( ! strcmp(methodName, "AMGSA") )
         MLI_MethodSetParams( cmliMethod, "setNullSpace", 4, targv );
      free( nullVecs );
   }
   if ( testProb == 1 )
   {
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
      if ( ! strcmp(methodName, "AMGSA") )
         MLI_MethodSetParams( cmliMethod, "setNullSpace", 4, targv );
      free( nullVecs );
   }
   MLI_MethodSetParams( cmliMethod, "print", 0, NULL );
   MLI_SetMethod( cmli, cmliMethod );
   MLI_SetSystemMatrix( cmli, 0, cmliMat );
   MLI_SetOutputLevel( cmli, 2 );

   if ( solver == 0 )
   {
      MLI_Setup( cmli );
      MLI_Solve( cmli, csol, crhs );
   } 
   else if ( solver == 1 )
   {
      MLI_Utils_HyprePCGSolve(cmli, (HYPRE_Matrix) HYPREA, 
                             (HYPRE_Vector) rhs, (HYPRE_Vector) sol);
   }
   else
   {
      MLI_Utils_HypreGMRESSolve(cmli, (HYPRE_Matrix) HYPREA, 
                              (HYPRE_Vector) rhs, (HYPRE_Vector) sol);
   }
   MLI_Print( cmli );
   MLI_Destroy( cmli );
   MLI_MatrixDestroy( cmliMat );
   MLI_VectorDestroy( csol );
   MLI_VectorDestroy( crhs );
   MLI_MethodDestroy( cmliMethod );
   MPI_Finalize();
}

/* *************************************************************** *
 * problem generation
 * ---------------------------------------------------------------- */

int hypre_GeneratePartitioning(int, int, int**);

int hypre_mapCD( int  ix, int  iy, int  iz, int  p, int  q, int  r,
     int  P, int  Q, int  R, int *nx_part, int *ny_part, int *nz_part,
     int *global_part )
{
   int nx_local, ny_local, ix_local, iy_local, iz_local;
   int global_index, proc_num;
 
   proc_num = r*P*Q + q*P + p;
   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   ix_local = ix - nx_part[p];
   iy_local = iy - ny_part[q];
   iz_local = iz - nz_part[r];
   global_index = global_part[proc_num] 
      + (iz_local*ny_local+iy_local)*nx_local + ix_local;

   return global_index;
}

HYPRE_ParCSRMatrix GenerateConvectionDiffusion( MPI_Comm comm,
                      int nx, int ny, int nz, int P, int Q, int R,
                      int p, int q, int r, double alpha, double beta,
                      double  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag, *offd;

   int    *diag_i, *diag_j, *offd_i, *offd_j;
   double *diag_data, *offd_data;

   int *global_part, ix, iy, iz, cnt, o_cnt, local_num_rows; 
   int *col_map_offd, row_index, i,j;

   int nx_local, ny_local, nz_local;
   int nx_size, ny_size, nz_size, num_cols_offd, grid_size;

   int *nx_part, *ny_part, *nz_part;

   int num_procs, my_id, P_busy, Q_busy, R_busy;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   grid_size = nx*ny*nz;

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   global_part = hypre_CTAlloc(int,P*Q*R+1);

   global_part[0] = 0;
   cnt = 1;
   for (iz = 0; iz < R; iz++)
   {
      nz_size = nz_part[iz+1]-nz_part[iz];
      for (iy = 0; iy < Q; iy++)
      {
         ny_size = ny_part[iy+1]-ny_part[iy];
         for (ix = 0; ix < P; ix++)
         {
            nx_size = nx_part[ix+1] - nx_part[ix];
            global_part[cnt] = global_part[cnt-1];
            global_part[cnt++] += nx_size*ny_size*nz_size;
         }
      }
   }

   nx_local = nx_part[p+1] - nx_part[p];
   ny_local = ny_part[q+1] - ny_part[q];
   nz_local = nz_part[r+1] - nz_part[r];

   my_id = r*(P*Q) + q*P + p;
   num_procs = P*Q*R;

   local_num_rows = nx_local*ny_local*nz_local;
   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);
   R_busy = hypre_min(nz,R);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local*nz_local;
   if (p < P_busy-1) num_cols_offd += ny_local*nz_local;
   if (q) num_cols_offd += nx_local*nz_local;
   if (q < Q_busy-1) num_cols_offd += nx_local*nz_local;
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R_busy-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;

   col_map_offd = hypre_CTAlloc(int, num_cols_offd);

   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz > nz_part[r]) 
               diag_i[cnt]++;
            else
            {
               if (iz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy > ny_part[q]) 
               diag_i[cnt]++;
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix > nx_part[p]) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_part[p+1]) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_part[q+1]) 
               diag_i[cnt]++;
            else
            {
               if (iy+1 < ny) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz+1 < nz_part[r+1]) 
               diag_i[cnt]++;
            else
            {
               if (iz+1 < nz) 
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }

   diag_j = hypre_CTAlloc(int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_part[r]; iz < nz_part[r+1]; iz++)
   {
      for (iy = ny_part[q];  iy < ny_part[q+1]; iy++)
      {
         for (ix = nx_part[p]; ix < nx_part[p+1]; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_part[r]) 
            {
               diag_j[cnt] = row_index-nx_local*ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy,iz-1,p,q,r-1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            if (iy > ny_part[q]) 
            {
               diag_j[cnt] = row_index-nx_local;
               diag_data[cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
            }
            else
            {
               if (iy) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy-1,iz,p,q-1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2] - 0.5 * beta / (double)(ny - 1);
               }
            }
            if (ix > nx_part[p]) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
if ( row_index == 0 ) printf("convection = %e\n", diag_data[cnt-1]);
            }
            else
            {
               if (ix) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix-1,iy,iz,p-1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1] - 0.5 * alpha / (double) (nx-1);
               }
            }
            if (ix+1 < nx_part[p+1]) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[1] + 0.5 * alpha / (double) (nx-1);
if ( row_index == 0 ) printf("convection = %e\n", diag_data[cnt-1]);
            }
            else
            {
               if (ix+1 < nx) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix+1,iy,iz,p+1,q,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[1] + 0.5 * alpha / (double) (nx-1);
               }
            }
            if (iy+1 < ny_part[q+1]) 
            {
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[2] + 0.5 * beta / (double) (ny-1);
            }
            else
            {
               if (iy+1 < ny) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy+1,iz,p,q+1,r,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[2] + 0.5 * beta / (double) (ny-1);
               }
            }
            if (iz+1 < nz_part[r+1]) 
            {
               diag_j[cnt] = row_index+nx_local*ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (iz+1 < nz) 
               {
                  offd_j[o_cnt] = hypre_mapCD(ix,iy,iz+1,p,q,r+1,P,Q,R,
                                      nx_part,ny_part,nz_part,global_part);
                  offd_data[o_cnt++] = value[3];
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      for (i=0; i < num_cols_offd; i++)
         col_map_offd[i] = offd_j[i];
   	
      qsort0(col_map_offd, 0, num_cols_offd-1);

      for (i=0; i < num_cols_offd; i++)
         for (j=0; j < num_cols_offd; j++)
            if (offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
   }

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   return (HYPRE_ParCSRMatrix) A;
}

