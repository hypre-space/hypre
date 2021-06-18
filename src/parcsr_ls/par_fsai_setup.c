/******************************************************************************
 *  Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *  
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

#define DEBUG 0
#define PRINT_CF 0

#define DEBUG_SAVE_ALL_OPS 0

/*****************************************************************************
 *  
 * Routine for driving the setup phase of FSAI
 *
 ******************************************************************************/

/*****************************************************************************
 * hypre_FSAISetup
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                      hypre_ParCSRMatrix *A  )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData       *fsai_data = (hypre_ParFSAIData*) fsai_vdata;
   hypre_MemoryLocation    memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* Data structure variables */

   hypre_ParCSRMatrix      *G_array;
   hypre_ParCSRMatrix      *P_array;                
   hypre_ParCSRMatrix      *S_Pattern;              
   hypre_ParVector         *kaporin_gradient;       
   hypre_ParVector         *nnz_per_row;            
   hypre_ParVector         *nnz_cum_sum;            
   HYPRE_Real              kap_tolerance           = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps               = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size           = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               logging                 = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Int               print_level             = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int               debug_flag;             = hypre_ParFSAIDataDebugFlag(fsai_data);


   /* Local variables */

   HYPRE_Int num_rows      = hypre_ParCSRMatrixGlobalNumRows(A);

   HYPRE_Int               num_procs, my_id, num_threads;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   num_threads = hypre_NumThreads;

   G_array           = hypre_ParFSAIDataGArray(fsai_data);
   P_array           = hypre_ParFSAIDataPArray(fsai_data);
   S_Pattern         = hypre_ParFSAIDataSPattern(fsai_data);
   kaporin_gradient  = hypre_ParFSAIDataKaporinGradient(fsai_data);
   nnz_per_row       = hypre_ParFSAIDataNnzPerRow(fsai_data);
   nnz_cum_sum       = hypre_ParFSAIDataNnzCumSum(fsai_data);

   /* free up storage in case of new setup without previous destroy */

   if( G_array || P_array || S_Pattern || kaporin_gradient || nnz_per_row || nnz_cum_sum )
   {
      if( G_array != NULL )
         hypre_ParCSRMatrixDestroy(G_array);
      
      if( P_array != NULL )
         hypre_ParCSRMatrixDestroy(P_array);
      
      if( S_Pattern != NULL )
         hypre_ParCSRMatrixDestroy(S_Pattern);
       
      if( kaporin_gradient != NULL )
         hypre_ParCSRVectorDestroy(kaporin_gradient);
      
      if( nnz_per_row != NULL )
         hypre_ParCSRVectorDestroy(nnz_per_row);
      
      if( nnz_cum_sum != NULL )
         hypre_ParCSRVectorDestroy(nnz_cum_sum);
   }

   G_array = hypre_CTAlloc(hypreCSRMatrix*, num_rows, HYPRE_MEMORY_HOST); 
   P_array = hypre_CTAlloc(hypreCSRMatrix*, num_rows, HYPRE_MEMORY_HOST);     //XXX: Can probably delete P_array, store new nonzero space in temp vector, then do a row merge with S_Pattern - will change later 
   S_Pattern = hypre_CTAlloc(hypreCSRMatrix*, num_rows, HYPRE_MEMORY_HOST); 
   kaporin_gradient = hypre_CTAlloc(hypreCSRVector*, num_rows, HYPRE_MEMORY_HOST); 
   nnz_per_row = hypre_CTAlloc(hypreCSRVector*, num_rows, HYPRE_MEMORY_HOST); 
   nnz_cum_sum = hypre_CTAlloc(hypreCSRVector*, num_rows, HYPRE_MEMORY_HOST); 

   return(hypre_error_flag);

}
