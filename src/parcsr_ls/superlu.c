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
   LUstruct_t dslu_data_LU;
   SuperLUStat_t dslu_data_stat;
   superlu_dist_options_t dslu_options;
   gridinfo_t dslu_data_grid;
   ScalePermstruct_t dslu_ScalePermstruct;
   SOLVEstruct_t dslu_solve;

} 
hypre_DSLUData; 

#endif

HYPRE_Int hypre_SLUDistSetup( HYPRE_Solver *solver, hypre_ParCSRMatrix *A)
{
      /* Par Data Structure variables */
   HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   /*SuperMatrix super_A;
   Stype_t stype = SLU_NR;
   Dtype_t dtype = SLU_D;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t data_solve;
   Mtype_t mtype = SLU_GE;
   HYPRE_Real anorm;*/
   hypre_CSRMatrix *A_local;
   HYPRE_Int num_rows;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int pcols=1, prows=1;
   hypre_DSLUData *dslu_data = NULL;

   HYPRE_Int info = 0;
   HYPRE_Int nrhs = 1;
   HYPRE_Real *B = NULL;
   HYPRE_Real *berr = NULL;
   
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   dslu_data = hypre_CTAlloc(hypre_DSLUData, 1);
   /*Create process grid */
   while (prows*pcols != num_procs) ++prows;
   --prows;
   pcols = num_procs/prows;
   while (prows*pcols != my_id)
   {
      prows -= 1;
      pcols = num_procs/prows;
   }

   superlu_gridinit(comm, prows, pcols, &(dslu_data->dslu_data_grid));

   set_default_options_dist(&(dslu_data->dslu_options));

   dslu_data->dslu_options.ColPerm = NATURAL;
   dslu_data->dslu_options.RowPerm = NATURAL;

   ScalePermstructInit(global_num_rows, global_num_rows, &(dslu_data->dslu_ScalePermstruct));
 
   LUstructInit(global_num_rows, &(dslu_data->dslu_data_LU));

   PStatInit(&(dslu_data->dslu_data_stat));

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
            SLU_NR, SLU_D, SLU_GE);

   hypre_CSRMatrixDestroy (A_local);

   berr = hypre_CTAlloc(HYPRE_Real, nrhs);
   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu), 
      &(dslu_data->dslu_ScalePermstruct), B, num_rows, nrhs, 
      &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU), 
      &(dslu_data->dslu_solve), berr, &(dslu_data->dslu_data_stat), &info);


   /*anorm = dplangs((char *)"I", &super_A, &data_grid);

   pdgstrf(&options, global_num_rows, global_num_cols, anorm, &data_LU, &data_grid, &data_stat, info_int);

   dSolveInit(&options, &super_A, &(ScalePermstruct->perm_r), &(ScalePermstruct->perm_c), 1, 
      &data_LU, &data_grid, &data_solve); */

   /*dslu_data = hypre_CTAlloc(hypre_DSLUData, 1);
   dslu_data->A_DSLU = super_A;
   dslu_data->DSLU_options = options;
   dslu_data->DSLU_data_grid = data_grid;
   dslu_data->DSLU_data_LU = data_LU;
   dslu_data->DSLU_data_stat = data_stat;
   dslu_data->DSLU_ScalePermstruct = ScalePermstruct;
   dslu_data->DSLU_data_solve = data_solve;
   dslu_data->global_num_cols = global_num_cols;*/

   hypre_TFree(berr);
   *solver = (HYPRE_Solver) dslu_data;
   return hypre_error_flag;
}

HYPRE_Int hypre_SLUDistSolve( void* solver, hypre_ParVector *b, hypre_ParVector *x)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;
   /*SuperMatrix *super_A;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t data_solve;*/
   HYPRE_Int info = 0;
   HYPRE_Real *B = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Int size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   /*super_A = DSLU_data->A_DSLU;
   options = DSLU_data->DSLU_options;
   data_grid = DSLU_data->DSLU_data_grid;
   data_LU = DSLU_data->DSLU_data_LU;
   data_stat = DSLU_data->DSLU_data_stat;
   ScalePermstruct = DSLU_data->DSLU_ScalePermstruct;
   data_solve = DSLU_data->DSLU_data_solve;*/
   HYPRE_Int nrhs = 1;
   HYPRE_Real *berr;
   berr = hypre_CTAlloc(HYPRE_Real, nrhs);

   (dslu_data->dslu_options).Fact = FACTORED;

   hypre_ParVectorCopy(b,x);

   pdgssvx(&(dslu_data->dslu_options), &(dslu_data->A_dslu), 
      &(dslu_data->dslu_ScalePermstruct), B, size, nrhs, 
      &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU), 
      &(dslu_data->dslu_solve), berr, &(dslu_data->dslu_data_stat), &info);

   hypre_TFree(berr);
   return hypre_error_flag;
}

HYPRE_Int hypre_SLUDistDestroy( void* solver)
{
   hypre_DSLUData *dslu_data = (hypre_DSLUData *) solver;
   /*SuperMatrix *super_A;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t data_solve;

   super_A = DSLU_data->A_DSLU;
   options = DSLU_data->DSLU_options;
   data_grid = DSLU_data->DSLU_data_grid;
   data_LU = DSLU_data->DSLU_data_LU;
   data_stat = DSLU_data->DSLU_data_stat;
   ScalePermstruct = DSLU_data->DSLU_ScalePermstruct;
   data_solve = DSLU_data->DSLU_data_solve; */

   PStatFree(&(dslu_data->dslu_data_stat));
   Destroy_CompRowLoc_Matrix_dist(&(dslu_data->A_dslu));
   ScalePermstructFree(&(dslu_data->dslu_ScalePermstruct));
   Destroy_LU(dslu_data->global_num_rows, &(dslu_data->dslu_data_grid), &(dslu_data->dslu_data_LU));
   LUstructFree(&(dslu_data->dslu_data_LU));
   if (dslu_data->dslu_options.SolveInitialized)
      dSolveFinalize(&(dslu_data->dslu_options), &(dslu_data->dslu_solve));
   superlu_gridexit(&(dslu_data->dslu_data_grid));
   hypre_TFree(dslu_data); 
   return hypre_error_flag;
}
#endif
