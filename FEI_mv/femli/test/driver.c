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

int main(int argc, char **argv)
{

   int                j, nx=256, ny=256, P, Q, p, q, nprocs, mypid, startRow;
   int                *partition, globalSize, localSize, nsweeps, rowSize;
   int                *colInd, k, *procCnts, *offsets, *rowCnts, ftype;
   int                ndofs=3, nullDim=6, testProb=1, solver=1, scaleFlag=0;
   int                fleng, rleng, status;
   char               *targv[10], fname[100], rhsFname[100];
   double             *values, *nullVecs, *scaleVec, *colVal, *gscaleVec;
   double             *rhsVector=NULL;
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
      p = mypid % P;
      q = ( mypid - p)/P;
      values = (double *) calloc(2, sizeof(double));
      values[1] = -1.0;
      values[0] = 0.0;
      if (nx > 1) values[0] += 2.0;
      if (ny > 1) values[0] += 2.0;
      if (nx > 1 && ny > 1) values[0] += 4.0;
      HYPREA = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD, 
                                             nx, ny, P, Q, p, q, values); 
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
   cmliMethod = MLI_MethodCreate( "AMGSA", MPI_COMM_WORLD );
   nsweeps = 1;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) NULL;
   MLI_MethodSetParams( cmliMethod, "setNumLevels 3", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setPreSmoother ParaSails", 2, targv );
   MLI_MethodSetParams( cmliMethod, "setPostSmoother ParaSails", 2, targv );
   MLI_MethodSetParams( cmliMethod, "setOutputLevel 2", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setMinCoarseSize 10", 0, NULL );
   MLI_MethodSetParams( cmliMethod, "setPweight 0.0", 0, NULL );
   nsweeps = 1;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) NULL;
   MLI_MethodSetParams( cmliMethod, "setCoarseSolver SuperLU", 2, targv );
   MLI_MethodSetParams( cmliMethod, "setCalibrationSize 0", 0, NULL );
   if ( testProb == 0 )
   {
      ndofs    = 1;
      nullDim  = 1;
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
      MLI_MethodSetParams( cmliMethod, "setNullSpace", 4, targv );
      free( nullVecs );
   }
   if ( testProb == 1 )
   {
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &nullDim;
      targv[2] = (char *) nullVecs;
      targv[3] = (char *) &localSize;
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


