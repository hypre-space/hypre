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
hypre_FSAISolve( void   *fsai_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector    *b,
                        hypre_ParVector    *x         )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   HYPRE_Int            fsai_print_level;
   HYPRE_Int            fsai_logging;
   HYPRE_ParCSRMatrix   *G_mat;

   /* Local variables */

   HYPRE_Int            num_procs, my_id;
   HYPRE_Int            rows_per_proc;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   fsai_print_level     = hypre_ParFSAIDataPrintLevel(fsai_data);
   fsai_logging         = hypre_ParFSAIDataLogging(fsai_data);
   tolerance            = hypre_ParFSAIDataTolerence(fsai_data);
   G_mat                = hypre_ParFSAIDataGmat(fsai_data);

   HYPRE_ANNOTATE_FUNC_END;

}
