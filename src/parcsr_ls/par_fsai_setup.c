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

   HYPRE_BigInt            row_start, row_end, col_start, col_end;
   hypre_ParCSRMatrixGetLocalRange(A, &row_start, &row_end, &col_start, &col_end);
   
   HYPRE_BigInt            num_local_rows = row_end - row_start + 1;
   HYPRE_BigInt            i;
   HYPRE_Int               k;

   for( i = 0; i < num_local_rows; i++ ){    /* Cycle through each of the local rows */

      HYPRE_Real           *kaporin_gradient = (HYPRE_Real*) calloc(row_start+i-1, sizeof(HYPRE_Real));
      HYPRE_BigInt         *S_Pattern = (HYPRE_BigInt*) calloc(min(max_steps*max_step_size, row_start+i-1), sizeof(HYPRE_BigInt));

      for( k = 0; k < max_steps; k++ ){      /* Cycle through each iteration for that row */
         
         /* Steps:
         * Compute Kaporin Gradient
         * Grab max_step_size UNIQUE positions from kaporian gradient
         * Gather P (TODO)
         * Solve A[P, P]G[i, P]' = -A[P, i]
         * Check psi
         */

      }

      /* Scale row */

      free(kaporin_gradient);
      free(S_Pattern);

   }

   return(hypre_error_flag);

}
