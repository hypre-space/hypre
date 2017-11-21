#ifdef HYPRE_USE_DSLU
#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include <math.h>
#include "superlu_ddefs.h"

typedef struct
{
   HYPRE_Int global_num_cols;
   SuperMatrix *A_DSLU;
   LUstruct_t DSLU_data_LU;
   SuperLUStat_t DSLU_data_stat;
   superlu_dist_options_t DSLU_options;
   gridinfo_t DSLU_data_grid;
   ScalePermstruct_t DSLU_ScalePermstruct;
   SOLVEstruct_t DSLU_solve;

} hypre_DSLUData;

HYPRE_Int hypre_SLUDistSetup( hypre_ParAMGData *amg_data, 
				HYPRE_Int level)
{
      /* Par Data Structure variables */
   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   MPI_Comm           comm = hypre_ParCSRMatrixComm(A);
   SuperMatrix *super_A;
   Stype_t stype = SLU_NR;
   Dtype_t dtype = SLU_D;
   Mtype_t mtype = SLU_GE;
   hypre_CSRMatrix *A_local;
   HYPRE_Int nprocs, my_id;
   HYPRE_Int pcols, prows=1;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t data_solve;
   hypre_DSLUData DSLU_data = NULL;

   HYPRE_Int info = 0;
   HYPRE_Real *B = NULL;
   HYPRE_Real *berr = NULL;
   /*HYPRE_Real anorm;*/
   
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*Create process grid */
   while (prows*pcols != num_procs) ++prows;
   --prows;
   pcols = num_procs/prows;
   while (prows*pcols != mysize)
   {
      prows -= 1;
      pcols = num_procs/prows;
   }

   superlu_gridinit(comm, prows, pcols, &data_grid);

   set_default_options_dist(&options);

   options.colperm = NATURAL;
   options.rowperm = NATURAL;

   ScalePermstructInit(global_num_rows, global_num_cols, &ScalePermstruct);
 
   LUstructInit(global_num_cols, &data_LU);

   PStatInit(&data_stat);

   /* Merge diag and offd into one matrix (global ids) */
   A_local = hypre_MergeDiagAndOffd(A);

   /* Now convert hypre matrix to a SuperMatrix */
   dCreate_CompRowLoc_Matrix_dist(
            &super_A,global_num_rows,global_num_cols,
            hypre_CSRMatrixNumNonzeros(A_local),
            hypre_CSRMatrixNumRows(A_local),
            hypre_ParCSRMatrixFirstRowIndex(A),
            hypre_CSRMatrixData(A_local),
            hypre_CSRMatrixJ(A_local),hypre_CSRMatrixI(A_local),
            stype, dtype, mtype);

   /*anorm = dplangs((char *)"I", &super_A, &data_grid);

   pdgstrf(&options, global_num_rows, global_num_cols, anorm, &data_LU, &data_grid, &data_stat, info_int);

   dSolveInit(&options, &super_A, &(ScalePermstruct->perm_r), &(ScalePermstruct->perm_c), 1, 
      &data_LU, &data_grid, &data_solve); */

   pdgssvx(&options, &super_A, &ScalePermstruct, B, num_rows, 1, &data_grid,
         &data_LU, &data_solve, berr, &data_stat, &info);

   DSLU_data = hypre_TAlloc(hypre_DSLUData, 1);
   DSLU_data->A_DSLU = super_A;
   DSLU_data->DSLU_options = options;
   DSLU_data->DSLU_data_grid = data_grid;
   DSLU_data->DSLU_data_LU = data_LU;
   DSLU_data->DSLU_data_stat = data_stat;
   DSLU_data->DSLU_ScalePermstruct = ScalePermstruct;
   DSLU_data->DSLU_data_solve = data_solve;
   DSLU_data->global_num_cols = global_num_cols;

   hypre_ParAMGDataSLUData(amg_data) = DSLU_data;
}

HYPRE_Int hypre_SLUDistSolve( hypre_ParAMGData *amg_data, hypre_ParVector *b, hypre_ParVector *x)
{
   SuperMatrix *super_A;
   Stype_t stype = SLU_NR;
   Dtype_t dtype = SLU_D;
   Mtype_t mtype = SLU_GE;
   hypre_CSRMatrix *A_local;
   HYPRE_Int nprocs, my_id;
   HYPRE_Int pcols, prows=1;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t DSLU_solve;
   HYPRE_int info = 0;
   hypre_DSLUData DSLU_data = hypre_ParAMGDataDSLUData(amg_data);
   HYPRE_Real *B = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Int size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   super_A = DSLU_data->A_DSLU;
   options = DSLU_data->DSLU_options;
   data_grid = DSLU_data->DSLU_data_grid;
   data_LU = DSLU_data->DSLU_data_LU;
   data_stat = DSLU_data->DSLU_data_stat;
   ScalePermstruct = DSLU_data->DSLU_ScalePermstruct;
   data_solve = DSLU_data->DSLU_data_solve;
   HYPRE_Int nrhs = 1;

   options.fact = FACTORED;

   hypre_ParVectorCopy(b,x);

   pdgssvx(&options, &super_A, &ScalePermstruct, B, size, nrhs, &data_grid,
         &data_LU, &data_solve, berr, &data_stat, &info);

}

HYPRE_Int hypre_SLUDistDestroy( hypre_ParAMGData *amg_data)
{
   SuperMatrix *super_A;
   gridinfo_t data_grid;
   superlu_dist_options_t options;
   ScalePermstruct_t ScalePermstruct;
   LUstruct_t data_LU;
   SuperLUStat_t data_stat;
   SOLVEstruct_t DSLU_solve;
   hypre_DSLUData DSLU_data = hypre_ParAMGDataDSLUData(amg_data);
   HYPRE_Int global_num_cols = DSLU_data->global_num_cols;

   super_A = DSLU_data->A_DSLU;
   options = DSLU_data->DSLU_options;
   data_grid = DSLU_data->DSLU_data_grid;
   data_LU = DSLU_data->DSLU_data_LU;
   data_stat = DSLU_data->DSLU_data_stat;
   ScalePermstruct = DSLU_data->DSLU_ScalePermstruct;
   data_solve = DSLU_data->DSLU_data_solve;

   PStatFree(&data_stat);
   Destroy_CompRowLoc_Matrix_dist(&super_A);
   ScalePermstructFree(&ScalePermstruct);
   Destroy_LU(global_num_cols, &data_grid, &data_LU);
   LUStructFree(&LUstruct);
   if (options.SolveInitialized)
      dSolveFinalize(&options, &data_solve);
   superlu_gridexit(&data_grid);
   HYPRE_TFree(DSLU_data); 
}
#endif     
