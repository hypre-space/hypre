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

int main(int argc, char **argv)
{

   int                j, nx=64, ny=512, P, Q, p, q, nprocs, mypid, start_row;
   int                *partition, global_size, local_size, nsweeps, row_size;
   int                *col_ind, k, *proc_cnts, *offsets, *row_cnts;
   int                ndofs=3, null_dim=6, test_prob=0, solver=1, scale_flag=0;
   char               *targv[10];
   double             *values, *null_vects, *scale_vec, *col_val, *gscale_vec;
   HYPRE_IJMatrix     newIJA;
   HYPRE_ParCSRMatrix HYPRE_A;
   hypre_ParCSRMatrix *hypre_A;
   hypre_ParVector    *sol, *rhs;
   CMLI               *cmli;
   CMLI_Matrix        *cmli_mat;
   CMLI_Method        *cmli_method;
   CMLI_Vector        *csol, *crhs;
   MLI_Function       *func_ptr;

   /* ------------------------------------------------------------- *
    * machine setup
    * ------------------------------------------------------------- */

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

   if ( test_prob == 0 )
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
      HYPRE_A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD, 
                                             nx, ny, P, Q, p, q, values); 
      free( values );
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPRE_A, &partition);
      global_size = partition[nprocs];
      start_row   = partition[mypid];
      local_size  = partition[mypid+1] - start_row;
      free( partition );
      if ( scale_flag ) 
      {
         scale_vec  = (double *) malloc(local_size*sizeof(double));
         gscale_vec = (double *) malloc(global_size*sizeof(double));
         for ( j = start_row; j < start_row+local_size; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPRE_A,j,&row_size,&col_ind,&col_val);
            for (k = 0; k < row_size; k++)
               if ( col_ind[k] == j ) scale_vec[j-start_row] = col_val[k]; 
            HYPRE_ParCSRMatrixRestoreRow(HYPRE_A,j,&row_size,&col_ind,&col_val);
         }
         for (j = 0; j < local_size; j++) scale_vec[j] = 1.0/sqrt(scale_vec[j]); 
         proc_cnts = (int *) malloc( nprocs * sizeof(int) );
         offsets   = (int *) malloc( nprocs * sizeof(int) );
         MPI_Allgather(&local_size,1,MPI_INT,proc_cnts,1,MPI_INT,MPI_COMM_WORLD);
         offsets[0] = 0;
         for ( j = 1; j < nprocs; j++ )
            offsets[j] = offsets[j-1] + proc_cnts[j-1];
         MPI_Allgatherv(scale_vec, local_size, MPI_DOUBLE, gscale_vec,
                        proc_cnts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         free( proc_cnts );
         free( offsets );
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, start_row, start_row+local_size-1,
                              start_row, start_row+local_size-1, &newIJA);
         HYPRE_IJMatrixSetObjectType(newIJA, HYPRE_PARCSR);
         row_cnts = (int *) malloc( local_size * sizeof(int) );
         for ( j = start_row; j < start_row+local_size; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPRE_A,j,&row_size,&col_ind,NULL);
            row_cnts[j-start_row] = row_size;
            HYPRE_ParCSRMatrixRestoreRow(HYPRE_A,j,&row_size,&col_ind,NULL);
         }
         HYPRE_IJMatrixSetRowSizes(newIJA, row_cnts);
         HYPRE_IJMatrixInitialize(newIJA);
         free( row_cnts );
         for ( j = start_row; j < start_row+local_size; j++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(HYPRE_A,j,&row_size,&col_ind,&col_val);
            for ( k = 0; k < row_size; k++ ) 
               col_val[k] = col_val[k] * gscale_vec[col_ind[k]] * gscale_vec[j];
            HYPRE_IJMatrixSetValues(newIJA, 1, &row_size, (const int *) &j,
                   (const int *) col_ind, (const double *) col_val);
            HYPRE_ParCSRMatrixRestoreRow(HYPRE_A,j,&row_size,&col_ind,&col_val);
         }
         HYPRE_IJMatrixAssemble(newIJA);
         HYPRE_ParCSRMatrixDestroy(HYPRE_A);
         HYPRE_IJMatrixGetObject(newIJA, (void **) &HYPRE_A);
         HYPRE_IJMatrixSetObjectType(newIJA, -1);
         HYPRE_IJMatrixDestroy(newIJA);
         free( gscale_vec );
         null_vects = ( double *) malloc(local_size * sizeof(double));
         for ( j = 0; j < local_size; j++ ) null_vects[j] = 1.0 / scale_vec[j]; 
         free( scale_vec );
      } else null_vects = NULL;
   }
   else if ( test_prob == 1 )
   {
      MLI_Utils_HypreMatrixRead(".data",MPI_COMM_WORLD,ndofs,(void **) &HYPRE_A,
                                scale_flag, &scale_vec);
      HYPRE_ParCSRMatrixGetRowPartitioning(HYPRE_A, &partition);
      global_size = partition[nprocs];
      start_row   = partition[mypid];
      local_size  = partition[mypid+1] - start_row;
      free( partition );
      null_vects = ( double *) malloc(local_size * null_dim * sizeof(double));
      MLI_Utils_DoubleVectorRead("rigid_body_mode01",MPI_COMM_WORLD,
                              local_size, start_row, null_vects);
      MLI_Utils_DoubleVectorRead("rigid_body_mode02",MPI_COMM_WORLD,
                              local_size, start_row, &null_vects[local_size]);
      MLI_Utils_DoubleVectorRead("rigid_body_mode03",MPI_COMM_WORLD,
                              local_size, start_row, &null_vects[local_size*2]);
      if ( scale_flag )
      {
         for ( j = 0; j < local_size; j++ ) 
            scale_vec[j] = sqrt(scale_vec[j]); 
         for ( j = 0; j < local_size; j++ ) null_vects[j] *= scale_vec[j]; 
         for ( j = 0; j < local_size; j++ ) 
            null_vects[local_size+j] *= scale_vec[j]; 
         for ( j = 0; j < local_size; j++ ) 
            null_vects[2*local_size+j] *= scale_vec[j]; 
      }
      if ( null_dim > 3 )
      {
         MLI_Utils_DoubleVectorRead("rigid_body_mode04",MPI_COMM_WORLD,
                              local_size, start_row, &null_vects[local_size*3]);
         MLI_Utils_DoubleVectorRead("rigid_body_mode05",MPI_COMM_WORLD,
                              local_size, start_row, &null_vects[local_size*4]);
         MLI_Utils_DoubleVectorRead("rigid_body_mode06",MPI_COMM_WORLD,
                              local_size, start_row, &null_vects[local_size*5]);
         if ( scale_flag )
         {
            for ( j = 0; j < local_size; j++ ) 
               null_vects[3*local_size+j] *= scale_vec[j]; 
            for ( j = 0; j < local_size; j++ ) 
               null_vects[4*local_size+j] *= scale_vec[j]; 
            for ( j = 0; j < local_size; j++ ) 
               null_vects[5*local_size+j] *= scale_vec[j]; 
         }
      }
   }

   hypre_A = (hypre_ParCSRMatrix *) HYPRE_A;
   HYPRE_ParCSRMatrixGetRowPartitioning(HYPRE_A, &partition);
   sol = hypre_ParVectorCreate(MPI_COMM_WORLD, global_size, partition);
   hypre_ParVectorInitialize( sol );
   hypre_ParVectorSetConstantValues( sol, 0.0 );

   HYPRE_ParCSRMatrixGetRowPartitioning(HYPRE_A, &partition);
   rhs = hypre_ParVectorCreate(MPI_COMM_WORLD, global_size, partition);
   hypre_ParVectorInitialize( rhs );
/*
   hypre_ParVectorSetRandomValues( rhs, 13984 );
*/
   hypre_ParVectorSetConstantValues( rhs, 1.0 );

   func_ptr = (MLI_Function *) malloc( sizeof( MLI_Function ) );
   MLI_Utils_HypreVectorGetDestroyFunc(func_ptr);
   csol = MLI_VectorCreate(sol, "HYPRE_ParVector", func_ptr);
   crhs = MLI_VectorCreate(rhs, "HYPRE_ParVector", func_ptr);

   /* ------------------------------------------------------------- *
    * problem setup
    * ------------------------------------------------------------- */

   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   cmli_mat = MLI_MatrixCreate((void*) hypre_A,"HYPRE_ParCSR",func_ptr);
   free( func_ptr );
   cmli = MLI_Create( MPI_COMM_WORLD );
   cmli_method = MLI_MethodCreate( "AMGSA", MPI_COMM_WORLD );
   nsweeps = 4;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) NULL;
   MLI_MethodSetParams( cmli_method, "setNumLevels 20", 0, NULL );
   MLI_MethodSetParams( cmli_method, "setPreSmoother CG", 2, targv );
   MLI_MethodSetParams( cmli_method, "setPostSmoother CG", 2, targv );
   MLI_MethodSetParams( cmli_method, "setOutputLevel 2", 0, NULL );
   nsweeps = 20;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) NULL;
   MLI_MethodSetParams( cmli_method, "setCoarseSolver SuperLU", 2, targv );
   MLI_MethodSetParams( cmli_method, "setCalibrationSize 0", 0, NULL );
   if ( test_prob == 0 )
   {
      ndofs      = 1;
      null_dim   = 1;
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &null_dim;
      targv[2] = (char *) null_vects;
      targv[3] = (char *) &local_size;
      MLI_MethodSetParams( cmli_method, "setNullSpace", 4, targv );
      free( null_vects );
   }
   if ( test_prob == 1 )
   {
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &null_dim;
      targv[2] = (char *) null_vects;
      targv[3] = (char *) &local_size;
      MLI_MethodSetParams( cmli_method, "setNullSpace", 4, targv );
      free( null_vects );
   }
   MLI_MethodSetParams( cmli_method, "print", 0, NULL );
   MLI_SetMethod( cmli, cmli_method );
   MLI_SetSystemMatrix( cmli, 0, cmli_mat );
   MLI_SetOutputLevel( cmli, 2 );

   if ( solver == 0 )
   {
      MLI_Setup( cmli );
      MLI_Solve( cmli, csol, crhs );
   } 
   else if ( solver == 1 )
   {
      MLI_Utils_HyprePCGSolve(cmli, (HYPRE_Matrix) HYPRE_A, 
                             (HYPRE_Vector) rhs, (HYPRE_Vector) sol);
   }
   else
   {
      MLI_Utils_HypreGMRESSolve(cmli, (HYPRE_Matrix) HYPRE_A, 
                              (HYPRE_Vector) rhs, (HYPRE_Vector) sol);
   }
   MLI_Print( cmli );
   MLI_Destroy( cmli );
   MLI_MatrixDestroy( cmli_mat );
   MLI_VectorDestroy( csol );
   MLI_VectorDestroy( crhs );
   MLI_MethodDestroy( cmli_method );
   MPI_Finalize();
}


