/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *    
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *  
 * AMG solve routine
 * 
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void               *fsai_vdata,
                   hypre_ParCSRMatrix *A         )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   HYPRE_Int      fsai_print_level;
   HYPRE_Int      fsai_logging;
   HYPRE_Real     tolerance;
   HYPRE_Int      max_steps;
   HYPRE_Int      max_step_size;
   

   /* Local variables */


   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   fsai_print_level     = hypre_ParFSAIDataPrintLevel(fsai_data);
   fsai_logging         = hypre_ParFSAIDataLogging(fsai_data);
   tolerance            = hypre_ParFSAIDataTolerence(fsai_data);
   max_steps            = hypre_ParFSAIDataMaxSteps(fsai_data);
   max_step_size        = hypre_ParFSAIDataMaxStepSize(fsai_data);


   HYPRE_ANNOTATE_FUNC_END;

}
