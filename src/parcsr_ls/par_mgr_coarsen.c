/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"

/*--------------------------------------------------------------------------
 * hypre_MGRCoarseParms
 *
 * Computes the fine and coarse partitioning arrays at once.
 *
 * TODO: Generate the dof_func array as in hypre_BoomerAMGCoarseParms
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRCoarseParms(MPI_Comm          comm,
                     HYPRE_Int         num_rows,
                     hypre_IntArray   *CF_marker,
                     HYPRE_BigInt     *row_starts_cpts,
                     HYPRE_BigInt     *row_starts_fpts)
{
   HYPRE_UNUSED_VAR(num_rows);

   HYPRE_Int     num_cpts;
   HYPRE_Int     num_fpts;

   HYPRE_BigInt  sbuffer_recv[2];
   HYPRE_BigInt  sbuffer_send[2];

   /* Count number of Coarse points */
   hypre_IntArrayCount(CF_marker, 1, &num_cpts);

   /* Count number of Fine points */
   hypre_IntArrayCount(CF_marker, -1, &num_fpts);

   /* Scan global starts */
   sbuffer_send[0] = (HYPRE_BigInt) num_cpts;
   sbuffer_send[1] = (HYPRE_BigInt) num_fpts;
   hypre_MPI_Scan(&sbuffer_send, &sbuffer_recv, 2, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* First points in next processor's range */
   row_starts_cpts[1] = sbuffer_recv[0];
   row_starts_fpts[1] = sbuffer_recv[1];

   /* First points in current processor's range */
   row_starts_cpts[0] = row_starts_cpts[1] - sbuffer_send[0];
   row_starts_fpts[0] = row_starts_fpts[1] - sbuffer_send[1];

   return hypre_error_flag;
}
