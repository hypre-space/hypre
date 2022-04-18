/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_fsai.h"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_FSAISetupDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupDevice( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   hypre_ParFSAIData       *fsai_data     = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int                max_steps     = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int                max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int                algo_type     = hypre_ParFSAIDataAlgoType(fsai_data);

   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt             num_rows_A    = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols_A    = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt            *row_starts_A  = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt            *col_starts_A  = hypre_ParCSRMatrixColStarts(A);

   hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);

   hypre_ParCSRMatrix      *G;
   hypre_ParCSRMatrix      *h_G;
   hypre_ParCSRMatrix      *h_A;
   HYPRE_Int                max_nnzrow_diag_G;
   HYPRE_Int                max_nonzeros_diag_G;

   hypre_GpuProfilingPushRange("FSAISetup");

   /* Destroy matrix G */
   HYPRE_ParCSRMatrixDestroy(hypre_ParFSAIDataGmat(fsai_data));

   /* Create matrix G on host */
   max_nnzrow_diag_G   = max_steps * max_step_size + 1;
   max_nonzeros_diag_G = num_rows_diag_A * max_nnzrow_diag_G;
   h_G = hypre_ParCSRMatrixCreate(comm, num_rows_A, num_cols_A,
                                 row_starts_A, col_starts_A,
                                 0, max_nonzeros_diag_G, 0);
   hypre_ParCSRMatrixInitialize_v2(h_G, HYPRE_MEMORY_HOST);
   hypre_ParFSAIDataGmat(fsai_data) = h_G;

   /* Clone matrix to host */
   h_A = hypre_ParCSRMatrixClone_v2(A, 1, HYPRE_MEMORY_HOST);

   /* Compute FSAI factor on host */
   switch (algo_type)
   {
      case 1:
         hypre_FSAISetupNative(fsai_vdata, h_A, f, u);
         break;

      case 2:
         hypre_FSAISetupOMPDyn(fsai_vdata, h_A, f, u);
         break;

      default:
         hypre_FSAISetupNative(fsai_vdata, h_A, f, u);
         break;
   }

   /* Move FSAI factor G to device */
   G = hypre_ParCSRMatrixClone_v2(h_G, 1, HYPRE_MEMORY_DEVICE);
   hypre_ParFSAIDataGmat(fsai_data) = G;

   /* Destroy temporary data on host */
   HYPRE_ParCSRMatrixDestroy(h_G);
   HYPRE_ParCSRMatrixDestroy(h_A);

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */
