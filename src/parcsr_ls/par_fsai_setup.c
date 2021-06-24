/******************************************************************************
 *  Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *  
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
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

   HYPRE_Real              kap_tolerance           = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps               = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size           = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               logging                 = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Int               print_level             = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int               debug_flag;             = hypre_ParFSAIDataDebugFlag(fsai_data);

   /* Local variables */

   HYPRE_Int               num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_Int               *row_partition
   HYPRE_Int               global_start;
   HYPRE_Int               local_size;
   HYPRE_Real              psi_old, psi_new;
   HYPRE_Real              row_scale;
 
   HYPRE_Int               row_size;
   HYPRE_BigInt            *col_ind;
   HYPRE_BigInt            *values;
   
   HYPRE_Int               i, j, k;       /* Loop variables */

   HYPRE_ParCSRMatrixGetRowPartitioning(A, &row_partition);
   global_start = row_partition[my_id];
   local_size = row_partition[my_id+1] - row_partition[my_id];
   hypre_TFree(row_partition, HYPRE_MEMORY_HOST);


   for( i = 0; i < local_size; i++ ){    /* Cycle through each of the local rows */

      HYPRE_Real           *kaporin_gradient = (HYPRE_Real*) calloc(global_start+i-1, sizeof(HYPRE_Real));
      HYPRE_Real           *G_temp           = (HYPRE_Real*) calloc(global_start+i, sizeof(HYPRE_Real));
      HYPRE_BigInt         *S_Pattern        = (HYPRE_BigInt*) calloc(min(max_steps*max_step_size, row_start+i-1), sizeof(HYPRE_BigInt));

      G_temp[global_start+i-1] = 1.0;

      HYPRE_ParCSRMatrixGetRow(A, i, &row_size, &col_ind, &values);

      for( k = 0; k < max_steps; k++ ){      /* Cycle through each iteration for that row */
         
         /* Steps:
         * Compute Kaporin Gradient
         *  1) How to get a row of A not owned by current process? HYPRE_ParCSRMatrixGetRow only uses local indexing, correct?
         *  2) kaporin_gradient[j] = 2*( InnerProd(A[j], G_temp[i]) + A[j][i])
         *     kaporin_gradient = 2 * MatVec(A[0:j], G_temp[i]') + 2*A[i] simplified right?
         *  3) To use hypre_ParCSRMatrixMatVec, does G_temp need to be a ParVector?
         
         * Grab max_step_size UNIQUE positions from kaporian gradient
         *  - Need to write my own function. A binary array can be used to mark with locations have already been added to the pattern.
         *
         * Gather A[P, P], G[i, P], and -A[P, i]
         *  - Adapt the hypre_ParCSRMatrixExtractBExt function. Don't want to return a CSR matrix because we're looking for a dense matrix.
         *
         * Determine psi_{k} = G_temp[i]*A*G_temp[i]'
         *
         * Solve A[P, P]G[i, P]' = -A[P, i]
         *
         * Determine psi_{k+1} = G_temp[i]*A*G_temp[i]'
         */
         if(abs( psi_new - psi_old )/psi_old < kaporin_tol)
            break;

      }

      /* row_scale = 1/sqrt(A[i, i] -  abs( InnerProd(G_temp, A[i])) )
      *  G[i] = row_scale * G_temp  
      */

      free(kaporin_gradient);
      free(G_temp);
      free(S_Pattern);

   }

   return(hypre_error_flag);

}
