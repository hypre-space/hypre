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

   /* Problem data */
   // HYPRE_Int            global_solver;
   hypre_ParCSRMatrix   *A_mat;
   HYPRE_Int            max_iterations;      /* Maximum iterations run per row */
   HYPRE_Real           tolerance;           /* Minimum amount of change between two iterations */ 
   HYPRE_Int            max_s;               /* Maximum number of nonzero elements added to a row of G per iteration */
    
   
   /* Data generated in the setup phase */
   hypre_ParCSRMatrix   *G_mat;               /* Matrix holding FSAI factor. M^(-1) = G'G */
   hypre_ParCSRMatrix   *S_Pattern;          /* Sparsity Pattern */
   hypre_ParVector      **kaporin_gradient;
   hypre_ParVector      **num_elements;      /* How many nonzeros each row has (for CUDA Gather) */
   hypre_ParVector      **cum_sum;           /* Cumulative sum of number of elements per row (for CUDA Gather) */
   

} hypre_ParFSAIData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_ParFSAIData structure
 *--------------------------------------------------------------------------*/

#define hypre_ParFSAIDataMemoryLocation(fsai_data)    ((fsai_data) -> memory_location)

/* Problem data */
#define hypre_ParFSAIDataAmat(fsai_data)              ((fsai_data) -> A_mat)
#define hypre_ParFSAIDataMaxIterations(fsai_data)     ((fsai_data) -> max_iterations)
#define hypre_ParFSAIDataTolerance(fsai_data)         ((fsai_data) -> tolerance)
#define hypre_ParFSAIDataMaxs(fsai_data)              ((fsai_data) -> max_s)
   
/* Data generated in the setup phase */
#define hypre_ParFSAIDataGmat(fsai_data)              ((fsai_data) -> G_mat)
#define hypre_ParFSAIDataSPattern(fsai_data)          ((fsai_data) -> S_Pattern)
#define hypre_ParFSAIDataKaporinGradient(fsai_data)   ((fsai_data) -> kaporin_gradient)
#define hypre_ParFSAIDataNumElements(fsai_data)       ((fsai_data) -> num_elements)
#define hypre_ParFSAIDataCumSum(fsai_data)            ((fsai_data) -> cum_sum)
