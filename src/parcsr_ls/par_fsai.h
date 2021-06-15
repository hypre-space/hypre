/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParFSAI_DATA_HEADER
#define hypre_ParFSAI_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParFSAIData
 *--------------------------------------------------------------------------*/
typedef struct 
{
   
   HYPRE_MemoryLocation memory_location;     /* memory location of matrices/vectors in FSAIData */

   // HYPRE_Int            global_solver;

   /* FSAI Problem data */
   hypre_ParCSRMatrix   *A_mat;
   HYPRE_Int            max_steps;           /* Maximum iterations run per row */
   HYPRE_Int            max_step_size;       /* Maximum number of nonzero elements added to a row of G per step */
   HYPRE_Real           tolerance;           /* Minimum amount of change between two steps */ 

   /* Solver Problem Data */
   HYPRE_Int            min_iterations;      /* Minimum iterations run for the solver */
   HYPRE_Int            max_iterations;      /* Maximum iterations run for the solver */
   HYPRE_Real           S_tolerance;         /* Tolerance for the solver */
    
   
   /* Data generated in the setup phase */
   hypre_ParCSRMatrix   *G_mat;              /* Matrix holding FSAI factor. M^(-1) = G'G */
   hypre_ParCSRMatrix   *S_Pattern;          /* Sparsity Pattern */
   hypre_ParVector      *kaporin_gradient;
   hypre_ParVector      *nnz_per_row;        /* How many nonzeros each row has (for CUDA Gather) */
   hypre_ParVector      *nnz_cum_sum;        /* Cumulative sum of number of elements per row (for CUDA Gather) */

   /* log info */
   HYPRE_Int            logging;
   HYPRE_Int            num_iterations;
   HYPRE_Real           rel_resid_norm;
   hypre_ParVector      *residual;        /* available if logging > 1 */

   /* output params */
   HYPRE_Int            print_level;
   char                 log_file_name[256];
   HYPRE_Int            debug_flag;    

} hypre_ParFSAIData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_ParFSAIData structure
 *--------------------------------------------------------------------------*/

#define hypre_ParFSAIDataMemoryLocation(fsai_data)          ((fsai_data) -> memory_location)

/* Problem data */
#define hypre_ParFSAIDataAmat(fsai_data)                    ((fsai_data) -> A_mat)
#define hypre_ParFSAIDataTolerance(fsai_data)               ((fsai_data) -> tolerance)
#define hypre_ParFSAIDataMaxSteps(fsai_data)                ((fsai_data) -> max_steps)
#define hypre_ParFSAIDataMaxStepSize(fsai_data)             ((fsai_data) -> max_step_size)

#define hypre_ParFSAIDataMinIterations(fsai_data)           ((fsai_data) -> min_iterations)
#define hypre_ParFSAIDataMaxIterations(fsai_data)           ((fsai_data) -> max_iterations)
#define hypre_ParFSAIDataSTolerance(fsai_data)              ((fsai_data) -> S_tolerance)
   
/* Data generated in the setup phase */
#define hypre_ParFSAIDataGmat(fsai_data)                    ((fsai_data) -> G_mat)
#define hypre_ParFSAIDataSPattern(fsai_data)                ((fsai_data) -> S_Pattern)
#define hypre_ParFSAIDataKaporinGradient(fsai_data)         ((fsai_data) -> kaporin_gradient)
#define hypre_ParFSAIDataNnzPerRow(fsai_data)               ((fsai_data) -> nnz_per_row)
#define hypre_ParFSAIDataNnzCumSum(fsai_data)               ((fsai_data) -> nnz_cum_sum)

/* log info data */
#define hypre_ParFSAIDataLogging(fsai_data)                 ((fsai_data) -> logging)
#define hypre_ParFSAIDataNumIterations(fsai_data)           ((fsai_data) -> num_iterations)
#define hypre_ParFSAIDataRelativeResidualNorm(fsai_data)    ((fsai_data) -> rel_resid_norm)
#define hypre_ParFSAIDataResidual(fsai_data)                ((fsai_data) -> residual)

/* output parameters */
#define hypre_ParFSAIDataPrintLevel(fsai_data)              ((fsai_data) -> print_level)
#define hypre_ParFSAIDataLogFileName(fsai_data)             ((fsai_data) -> log_file_name)
#define hypre_ParFSAIDataDebugFlag(fsai_data)               ((fsai_data) -> debug_flag)

#endif
