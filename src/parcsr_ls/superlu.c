#include "_hypre_parcsr_ls.h"
#include <math.h>

#ifdef HAVE_DSUPERLU
/*#include "superlu.h"*/

#include <math.h>
#include "superlu_ddefs.h"

#ifndef hypre_DSLU_DATA_HEADER
#define hypre_DSLU_DATA_HEADER

typedef struct 
{
   HYPRE_Int global_num_rows;
   SuperMatrix A_dslu;
   HYPRE_Real *berr;
   LUstruct_t dslu_data_LU;
   SuperLUStat_t dslu_data_stat;
   superlu_dist_options_t dslu_options;
   gridinfo_t dslu_data_grid;
   ScalePermstruct_t dslu_ScalePermstruct;
   SOLVEstruct_t dslu_solve;
} 
hypre_DSLUData; 

#endif

HYPRE_Int hypre_SLUDistSetup( HYPRE_Solver *solver, hypre_ParCSRMatrix *A, HYPRE_Int print_level)
{
      /* Par Data Structure variables */
   HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_local;
   HYPRE_Int num_rows;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int pcols=1, prows=1;
   hypre_DSLUData *dslu_data = NULL;

   HYPRE_Int info = 0;
   HYPRE_Int nrhs = 0;
   
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   dslu_data = hypre_CTAlloc(hypre_DSLUData, 1, HYPRE_MEMORY_HOST);

   /* Merge diag and offd into one matrix (global ids) */
   A_local = hypre_MergeDiagAndOffd(A);

   num_rows = hypre_CSRMatrixNumRows(A_local);
   /* Now convert hypre matrix to a SuperMatrix */
   dCreate_CompRowLoc_Matrix_dist(
            &(dslu_data->A_dslu),global_num_rows,global_num_rows,
            hypre_CSRMatrixNumNonzeros(A_local),
            num_rows,
            hypre_ParCSRMatrixFirstRowIndex(A),
            hypre_CSRMatrixData(A_local),
            hypre_CSRMatrixJ(A_local),hypre_CSRMatrixI(A_local),
            SLU_NR_loc, SLU_D, SLU_GE);

   hypre_CSRMatrixData(A_local) = NULL;
   hypre_CSRMatrixI(A_local) = NULL;
   hypre_CSRMatrixJ(A_local) = NULL;
   hypre_CSRMatrixDestroy(A_local);

   /*Create process grid */
   while (prows*pcols <= num_procs) ++prows;
   --prows;
   pcols = num_procs/prows;
   while (prows*pcols != num_procs)
   {
      prows -= 1;
      pcols = num_procs/prows;
   }
   //hypre_printf(" prows %d pcols %d\n", prows, pcols);

   superlu_gridinit(comm, prows, pcols, &(dslu_data->dslu_data_grid));

   set_default_options_dist(&(dslu_data->dslu_options));

   dslu_data->dslu_options.Fact = DOFACT;
   if (print_level == 0 || print_level == 2) dslu_data->dslu_options.PrintStat = NO;
   /*dslu_data->dslu_options.IterRefine = SLU_DOUBLE;
   dslu_data->dslu_options.ColPerm = MMD_AT_PLUS_A;
   dslu_data->dslu_options.DiagPivotThresh = 1.0;
   dslu_data->dslu_options.ReplaceTinyPivot = NO; */

   ScalePermstructInit(global_num_rows, global_num_rows, &(dslu_data->dslu_ScalePermstruct));
 
   LUstructInit(global_num_rows, &(dslu_data->dslu_data_LU));

   PStatInit(&(dslu_data->dslu_data_stat));

   dslu_data->global_num_rows = global_num_rows;

   dslu_data->berr = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_HOST);
   dslu_data->berr[0] = 0.0;

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu), 
      &(dslu_data->dslu_ScalePermstruct), NULL, num_rows, nrhs, 
      &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU), 
      &(dslu_data->dslu_solve), dslu_data->berr, &(dslu_data->dslu_data_stat), &info);

   dslu_data->dslu_options.Fact = FACTORED;

   *solver = (HYPRE_Solver) dslu_data;
   return hypre_error_flag;
}

HYPRE_Int hypre_SLUDistSolve( void* solver, hypre_ParVector *b, hypre_ParVector *x)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;
   HYPRE_Int info = 0;
   HYPRE_Real *B = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Int size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   HYPRE_Int nrhs = 1;

   hypre_ParVectorCopy(b,x);

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu), 
      &(dslu_data->dslu_ScalePermstruct), B, size, nrhs, 
      &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU), 
      &(dslu_data->dslu_solve), dslu_data->berr, &(dslu_data->dslu_data_stat), &info);

   return hypre_error_flag;
}

HYPRE_Int hypre_SLUDistDestroy( void* solver)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;

   PStatFree(&(dslu_data->dslu_data_stat));
   Destroy_CompRowLoc_Matrix_dist(&(dslu_data->A_dslu));
   ScalePermstructFree(&(dslu_data->dslu_ScalePermstruct));
   Destroy_LU(dslu_data->global_num_rows, &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU));
   LUstructFree(&(dslu_data->dslu_data_LU));
   if (dslu_data->dslu_options.SolveInitialized)
      dSolveFinalize(&(dslu_data->dslu_options), &(dslu_data->dslu_solve));
   superlu_gridexit(&(dslu_data->dslu_data_grid));
   hypre_TFree(dslu_data->berr, HYPRE_MEMORY_HOST); 
   hypre_TFree(dslu_data, HYPRE_MEMORY_HOST); 
   return hypre_error_flag;
}
#endif
