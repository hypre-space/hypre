/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for matrix statistics specialized to ParCSRMatrix types
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassOneLocalDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputePassOneLocalDevice(hypre_ParCSRMatrix   *A,
                                                 hypre_MatrixStats    *stats)
{
#if 0
   /* Diag matrix data */
   hypre_CSRMatrix     *diag;
   HYPRE_Int           *diag_i;
   HYPRE_Complex       *diag_a;

   /* Offd matrix data */
   hypre_CSRMatrix     *offd;
   HYPRE_Int           *offd_i;
   HYPRE_Complex       *offd_a;

   /* Local arrays */
   hypre_ulonglongint  *actual_nonzeros;
   HYPRE_Int           *nnzrow_min;
   HYPRE_Int           *nnzrow_max;
   HYPRE_Real          *rowsum_min;
   HYPRE_Real          *rowsum_max;
   HYPRE_Real          *rowsum_avg;

   /* Local variables */
   HYPRE_Int            num_rows;

#endif
   return hypre_error_flag;
}
