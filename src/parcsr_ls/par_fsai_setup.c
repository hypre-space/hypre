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

   /* Data structure variables */

   hypre_ParCSRMatrix      *A_array;
   hypre_ParCSRMatrix      *G_array;
   hypre_ParCSRMatrix      *P_array;
   hypre_ParCSRMatrix      *S_Pattern;
   hypre_ParVector         *Residual_array;
   hypre_ParVector         *fsai_kaporin_gradient;
   hypre_ParVector         *fsai_nnz_per_row;
   hypre_ParVector         *fsai_nnz_cum_sum;
   HYPRE_Real              fsai_tolerance;
   HYPRE_Int               fsai_max_steps;
   HYPRE_Int               fsai_max_step_size;
   HYPRE_Int               fsai_logging;
   HYPRE_Int               fsai_print_level;
   HYPRE_Int               debug_flag;

   hypre_MemoryLocation    memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* Local variables */

   HYPRE_Int               num_procs, my_id, num_threads;
   hypre_ParCSRMatrix      *Gtemp;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   num_threads = hypre_NumThreads;

   fsai_tolerance       = hypre_ParFSAIDataTolerance(fsai_data);
   fsai_max_steps       = hypre_ParFSAIDataMaxSteps(fsai_data);
   fsai_max_steps_size  = hypre_ParFSAIDataMaxStepSize(fsai_data);
   fsai_logging         = hypre_ParFSAIDataLogging(fsai_data);
   fsai_print_level     = hypre_ParFSAIDataPrintLevel(fsai_data);
   debug_flag           = hypre_ParFSAIDataDebugFlag(fsai_data);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixSetDNumNonzeros(A);

   A_array = hypre_ParFSAIDataAArray(fsai_data);
   G_array = hypre_ParFSAIDataGArray(fsai_data);
   P_array = hypre_ParFSAIDataPArray(fsai_data);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   HYPRE_FUNC_ANNOTATE_END;

   

}
