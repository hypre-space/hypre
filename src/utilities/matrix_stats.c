/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/******************************************************************************
 *
 * Member functions for hypre_MatrixStats class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsCreate
 *--------------------------------------------------------------------------*/

hypre_MatrixStats*
hypre_MatrixStatsCreate(void)
{
   hypre_MatrixStats  *stats;

   stats = hypre_CTAlloc(hypre_MatrixStats, 1, HYPRE_MEMORY_HOST);

   hypre_MatrixStatsNumRows(stats)         = 0;
   hypre_MatrixStatsNumCols(stats)         = 0;
   hypre_MatrixStatsNumNonzeros(stats)     = 0ULL;

   hypre_MatrixStatsActualNonzeros(stats)  = 0ULL;
   hypre_MatrixStatsActualThreshold(stats) = HYPRE_REAL_EPSILON;
   hypre_MatrixStatsSparsity(stats)        = 0.0;

   hypre_MatrixStatsNnzrowMin(stats)       = 0;
   hypre_MatrixStatsNnzrowMax(stats)       = 0;
   hypre_MatrixStatsNnzrowAvg(stats)       = 0.0;
   hypre_MatrixStatsNnzrowStDev(stats)     = 0.0;
   hypre_MatrixStatsNnzrowSqsum(stats)     = 0.0;

   hypre_MatrixStatsRowsumMin(stats)       = 0.0;
   hypre_MatrixStatsRowsumMax(stats)       = 0.0;
   hypre_MatrixStatsRowsumAvg(stats)       = 0.0;
   hypre_MatrixStatsRowsumStDev(stats)     = 0.0;
   hypre_MatrixStatsRowsumSqsum(stats)     = 0.0;

   hypre_MatrixStatsAbsrowsumMin(stats)    = 0.0;
   hypre_MatrixStatsAbsrowsumMax(stats)    = 0.0;
   hypre_MatrixStatsAbsrowsumAvg(stats)    = 0.0;
   hypre_MatrixStatsAbsrowsumStDev(stats)  = 0.0;
   hypre_MatrixStatsAbsrowsumSqsum(stats)  = 0.0;

   return stats;
}

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MatrixStatsDestroy(hypre_MatrixStats *stats)
{
   if (stats)
   {
      hypre_TFree(stats, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * Member functions for hypre_MatrixStatsArray class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsArrayCreate
 *--------------------------------------------------------------------------*/

hypre_MatrixStatsArray*
hypre_MatrixStatsArrayCreate(HYPRE_Int capacity)
{
   hypre_MatrixStatsArray  *stats_array;
   HYPRE_Int                i;

   stats_array = hypre_CTAlloc(hypre_MatrixStatsArray, 1, HYPRE_MEMORY_HOST);

   hypre_MatrixStatsArrayCapacity(stats_array) = capacity;
   hypre_MatrixStatsArrayEntries(stats_array)  = hypre_TAlloc(hypre_MatrixStats *,
                                                              capacity,
                                                              HYPRE_MEMORY_HOST);
   for (i = 0; i < capacity; i++)
   {
      hypre_MatrixStatsArrayEntry(stats_array, i) = hypre_MatrixStatsCreate();
   }

   return stats_array;
}

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsArrayDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MatrixStatsArrayDestroy(hypre_MatrixStatsArray *stats_array)
{
   HYPRE_Int   i;
   HYPRE_Int   capacity;

   if (stats_array)
   {
      capacity = hypre_MatrixStatsArrayCapacity(stats_array);

      for (i = 0; i < capacity; i++)
      {
         hypre_MatrixStatsDestroy(hypre_MatrixStatsArrayEntry(stats_array, i));
      }
      hypre_TFree(hypre_MatrixStatsArrayEntries(stats_array), HYPRE_MEMORY_HOST);
      hypre_TFree(stats_array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsReduce
 *
 * Reduce local matrix statistics across MPI processes.
 *
 * Expects local_stats populated by hypre_ParCSRMatrixStatsComputeLocal (or equivalent),
 * with field semantics:
 *   - {Nnzrow,Rowsum,Absrowsum}{Min,Max,Avg} are local per-row statistics
 *   - {Nnzrow,Rowsum,Absrowsum}Sqsum is the local sum of squared deviations
 *     from the local mean
 *
 * On output, global_stats holds the corresponding global statistics, with
 * Sqsum re-centered on the global mean (Chan/Pebay parallel-variance combine):
 *   global_sqsum = sum_p ( local_sqsum_p + n_p * (local_mean_p - global_mean)^2 )
 *
 * Uses 5 Allreduces:
 *   1. MAX over packed [-min_nnz, max_nnz, -min_rowsum, max_rowsum,
 *                       -min_absrowsum, max_absrowsum]
 *   2. SUM over nonzero counts as HYPRE_Real to preflight BigInt range
 *   3. SUM over [num_rows, num_nonzeros, actual_nonzeros] as HYPRE_BigInt
 *   4. SUM over local row-sum totals to derive global means
 *   5. SUM over each rank's mean-adjusted sqsums
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MatrixStatsReduce(hypre_MatrixStats *local_stats,
                        hypre_MatrixStats *global_stats,
                        MPI_Comm           comm)
{
   HYPRE_BigInt   local_num_rows;
   HYPRE_BigInt   bigint_send[3], bigint_recv[3];
   HYPRE_BigInt   global_num_rows;
   hypre_ulonglongint global_num_nonzeros;
   hypre_ulonglongint global_actual_nonzeros;
   HYPRE_Real     global_size;
   HYPRE_Real     real_send[6], real_recv[6];
   HYPRE_Real     extrema_recv[6];
   HYPRE_Real     local_avg_nnz, local_avg_rowsum, local_avg_absrowsum;
   HYPRE_Real     global_avg_nnz, global_avg_rowsum, global_avg_absrowsum;
   HYPRE_Real     global_sum_rows[2];
   HYPRE_Real     global_dev_sqsum_nnz, global_dev_sqsum_rowsum, global_dev_sqsum_absrowsum;
   HYPRE_Real     inv_global_num_rows;
   HYPRE_Int      i;
   HYPRE_Int      local_overflow;
   HYPRE_Int      global_overflow;

   if (!local_stats)
   {
      hypre_error_in_arg(1);
   }
   if (!global_stats)
   {
      hypre_error_in_arg(2);
   }
   if (!local_stats || !global_stats)
   {
      return hypre_error_flag;
   }

   local_num_rows   = hypre_MatrixStatsNumRows(local_stats);
   local_avg_nnz    = hypre_MatrixStatsNnzrowAvg(local_stats);
   local_avg_rowsum = hypre_MatrixStatsRowsumAvg(local_stats);
   local_avg_absrowsum = hypre_MatrixStatsAbsrowsumAvg(local_stats);

   /* (1) MAX-reduce packed mins (negated) and maxes. Empty ranks supply
    * sentinel values that cannot win the max, so they don't pollute the
    * global min/max. */
   if (local_num_rows > 0)
   {
      real_send[0] = -(HYPRE_Real) hypre_MatrixStatsNnzrowMin(local_stats);
      real_send[1] =  (HYPRE_Real) hypre_MatrixStatsNnzrowMax(local_stats);
      real_send[2] = -hypre_MatrixStatsRowsumMin(local_stats);
      real_send[3] =  hypre_MatrixStatsRowsumMax(local_stats);
      real_send[4] = -hypre_MatrixStatsAbsrowsumMin(local_stats);
      real_send[5] =  hypre_MatrixStatsAbsrowsumMax(local_stats);
   }
   else
   {
      real_send[0] = -HYPRE_REAL_MAX;
      real_send[1] = -HYPRE_REAL_MAX;
      real_send[2] = -HYPRE_REAL_MAX;
      real_send[3] = -HYPRE_REAL_MAX;
      real_send[4] = -HYPRE_REAL_MAX;
      real_send[5] = -HYPRE_REAL_MAX;
   }
   hypre_MPI_Allreduce(real_send, real_recv, 6, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);
   for (i = 0; i < 6; i++)
   {
      extrema_recv[i] = real_recv[i];
   }

   /* (2) Preflight unsigned count ranges before the signed BigInt reduction. */
   local_overflow = (hypre_MatrixStatsNumNonzeros(local_stats) >
                     (hypre_ulonglongint) HYPRE_BIG_INT_MAX) ||
                    (hypre_MatrixStatsActualNonzeros(local_stats) >
                     (hypre_ulonglongint) HYPRE_BIG_INT_MAX);
   hypre_MPI_Allreduce(&local_overflow, &global_overflow, 1, HYPRE_MPI_INT,
                       hypre_MPI_MAX, comm);
   if (global_overflow)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Matrix statistics count exceeds HYPRE_BigInt range\n");
      return hypre_error_flag;
   }

   real_send[0] = (HYPRE_Real) hypre_MatrixStatsNumNonzeros(local_stats);
   real_send[1] = (HYPRE_Real) hypre_MatrixStatsActualNonzeros(local_stats);
   hypre_MPI_Allreduce(real_send, real_recv, 2, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   if (real_recv[0] > (HYPRE_Real) HYPRE_BIG_INT_MAX ||
       real_recv[1] > (HYPRE_Real) HYPRE_BIG_INT_MAX)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Global matrix statistics count exceeds HYPRE_BigInt range\n");
      return hypre_error_flag;
   }

   /* (3) SUM-reduce row and nonzero counts as BigInt. */
   bigint_send[0] = local_num_rows;
   bigint_send[1] = (HYPRE_BigInt) hypre_MatrixStatsNumNonzeros(local_stats);
   bigint_send[2] = (HYPRE_BigInt) hypre_MatrixStatsActualNonzeros(local_stats);
   hypre_MPI_Allreduce(bigint_send, bigint_recv, 3, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   global_num_rows        = bigint_recv[0];
   global_num_nonzeros    = (hypre_ulonglongint) bigint_recv[1];
   global_actual_nonzeros = (hypre_ulonglongint) bigint_recv[2];

   hypre_MatrixStatsNumRows(global_stats)         = global_num_rows;
   hypre_MatrixStatsNumCols(global_stats)         = hypre_MatrixStatsNumCols(local_stats);
   hypre_MatrixStatsNumNonzeros(global_stats)     = global_num_nonzeros;
   hypre_MatrixStatsActualNonzeros(global_stats)  = global_actual_nonzeros;
   hypre_MatrixStatsActualThreshold(global_stats) = hypre_MatrixStatsActualThreshold(local_stats);

   if (global_num_rows == 0)
   {
      hypre_MatrixStatsSparsity(global_stats)    = 0.0;
      hypre_MatrixStatsNnzrowMin(global_stats)   = 0;
      hypre_MatrixStatsNnzrowMax(global_stats)   = 0;
      hypre_MatrixStatsNnzrowAvg(global_stats)   = 0.0;
      hypre_MatrixStatsNnzrowStDev(global_stats) = 0.0;
      hypre_MatrixStatsNnzrowSqsum(global_stats) = 0.0;
      hypre_MatrixStatsRowsumMin(global_stats)   = 0.0;
      hypre_MatrixStatsRowsumMax(global_stats)   = 0.0;
      hypre_MatrixStatsRowsumAvg(global_stats)   = 0.0;
      hypre_MatrixStatsRowsumStDev(global_stats) = 0.0;
      hypre_MatrixStatsRowsumSqsum(global_stats) = 0.0;
      hypre_MatrixStatsAbsrowsumMin(global_stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumMax(global_stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumAvg(global_stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumStDev(global_stats) = 0.0;
      hypre_MatrixStatsAbsrowsumSqsum(global_stats) = 0.0;
      return hypre_error_flag;
   }

   global_size = (HYPRE_Real) global_num_rows *
                 (HYPRE_Real) hypre_MatrixStatsNumCols(global_stats);
   hypre_MatrixStatsSparsity(global_stats) = global_size > 0.0 ?
                                             100.0 * (1.0 - (HYPRE_Real) global_num_nonzeros /
                                                      global_size) :
                                             0.0;

   hypre_MatrixStatsNnzrowMin(global_stats) = (HYPRE_Int) - extrema_recv[0];
   hypre_MatrixStatsNnzrowMax(global_stats) = (HYPRE_Int)   extrema_recv[1];
   hypre_MatrixStatsRowsumMin(global_stats) =              -extrema_recv[2];
   hypre_MatrixStatsRowsumMax(global_stats) =               extrema_recv[3];
   hypre_MatrixStatsAbsrowsumMin(global_stats) =           -extrema_recv[4];
   hypre_MatrixStatsAbsrowsumMax(global_stats) =            extrema_recv[5];

   inv_global_num_rows = 1.0 / (HYPRE_Real) global_num_rows;

   /* (4) Global mean for nnz comes for free from global_num_nonzeros.
    * For row sums we need the global sums of local row sums. */
   real_send[0] = local_avg_rowsum    * (HYPRE_Real) local_num_rows;
   real_send[1] = local_avg_absrowsum * (HYPRE_Real) local_num_rows;
   hypre_MPI_Allreduce(real_send, global_sum_rows, 2, HYPRE_MPI_REAL,
                       hypre_MPI_SUM, comm);

   global_avg_nnz       = (HYPRE_Real) global_num_nonzeros * inv_global_num_rows;
   global_avg_rowsum    = global_sum_rows[0] * inv_global_num_rows;
   global_avg_absrowsum = global_sum_rows[1] * inv_global_num_rows;

   /* (5) Combine sqsums to a common (global) mean:
    *   sqsum_global = sum_p ( sqsum_p + n_p * (mean_p - global_mean)^2 ) */
   {
      HYPRE_Real n_local = (HYPRE_Real) local_num_rows;
      HYPRE_Real d_nnz       = local_avg_nnz       - global_avg_nnz;
      HYPRE_Real d_rowsum    = local_avg_rowsum    - global_avg_rowsum;
      HYPRE_Real d_absrowsum = local_avg_absrowsum - global_avg_absrowsum;

      real_send[0] = hypre_MatrixStatsNnzrowSqsum(local_stats)     + n_local * d_nnz       * d_nnz;
      real_send[1] = hypre_MatrixStatsRowsumSqsum(local_stats)     + n_local * d_rowsum    * d_rowsum;
      real_send[2] = hypre_MatrixStatsAbsrowsumSqsum(local_stats)  + n_local * d_absrowsum * d_absrowsum;
   }
   hypre_MPI_Allreduce(real_send, real_recv, 3, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   global_dev_sqsum_nnz       = real_recv[0];
   global_dev_sqsum_rowsum    = real_recv[1];
   global_dev_sqsum_absrowsum = real_recv[2];
   if (global_dev_sqsum_nnz       < 0.0) { global_dev_sqsum_nnz       = 0.0; }
   if (global_dev_sqsum_rowsum    < 0.0) { global_dev_sqsum_rowsum    = 0.0; }
   if (global_dev_sqsum_absrowsum < 0.0) { global_dev_sqsum_absrowsum = 0.0; }

   hypre_MatrixStatsNnzrowAvg(global_stats)   = global_avg_nnz;
   hypre_MatrixStatsNnzrowSqsum(global_stats) = global_dev_sqsum_nnz;
   hypre_MatrixStatsNnzrowStDev(global_stats) = hypre_sqrt(global_dev_sqsum_nnz *
                                                           inv_global_num_rows);

   hypre_MatrixStatsRowsumAvg(global_stats)   = global_avg_rowsum;
   hypre_MatrixStatsRowsumSqsum(global_stats) = global_dev_sqsum_rowsum;
   hypre_MatrixStatsRowsumStDev(global_stats) = hypre_sqrt(global_dev_sqsum_rowsum *
                                                           inv_global_num_rows);

   hypre_MatrixStatsAbsrowsumAvg(global_stats)   = global_avg_absrowsum;
   hypre_MatrixStatsAbsrowsumSqsum(global_stats) = global_dev_sqsum_absrowsum;
   hypre_MatrixStatsAbsrowsumStDev(global_stats) = hypre_sqrt(global_dev_sqsum_absrowsum *
                                                              inv_global_num_rows);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MatrixStatsArrayPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MatrixStatsArrayPrint(HYPRE_Int                num_hierarchies,
                            HYPRE_Int               *num_levels,
                            HYPRE_Int                use_divisors,
                            HYPRE_Int                shift,
                            const char             **messages,
                            hypre_MatrixStatsArray  *stats_array)
{
   HYPRE_Int            capacity = hypre_MatrixStatsArrayCapacity(stats_array);

   hypre_MatrixStats   *stats;
   hypre_MatrixStats   *stats_finest;
   hypre_MatrixStats   *stats_next;

   HYPRE_Int            ndigits[HYPRE_NDIGITS_SIZE];
   HYPRE_Int            offsets[8];
   HYPRE_Int            divisors[4];
   HYPRE_Int            i, square;
   HYPRE_Int            square_count;
   HYPRE_Int            num_levels_total;
   HYPRE_BigInt         fine_num_rows;
   HYPRE_BigInt         coarse_num_rows;
   HYPRE_BigInt         total_num_rows;

   /* Compute total number of levels */
   num_levels_total = 0;
   for (i = 0; i < num_hierarchies; i++)
   {
      num_levels_total += num_levels[i];
   }

   /* Sanity check */
   if (capacity < num_levels_total)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Matrix statistics array does not have enough capacity!");
      return hypre_error_flag;
   }

   /* Check if all matrices are square or rectangular */
   square_count = 0;
   for (i = 0; i < num_levels_total; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      if (hypre_MatrixStatsNumRows(stats) ==
          hypre_MatrixStatsNumCols(stats))
      {
         square_count += 1;
      }
   }

   if (square_count == 0)
   {
      square = 0;
   }
   else if (square_count == num_levels_total)
   {
      square = 1;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Cannot process square and rectangular matrices at the same time!");
      return hypre_error_flag;
   }

   /* Set some shortcuts */
   stats_finest = hypre_MatrixStatsArrayEntry(stats_array, 0);

   /* Digits computation */
   ndigits[0]  = hypre_max(7, hypre_ndigits(hypre_MatrixStatsNumRows(stats_finest)));
   ndigits[1]  = 7;
   ndigits[2]  = hypre_max(8, hypre_ndigits_ull(hypre_MatrixStatsNumNonzeros(stats_finest)));
   ndigits[3]  = 7;
   ndigits[4]  = 4;
   ndigits[5]  = 4;
   ndigits[6]  = 4;
   ndigits[7]  = 5;
   ndigits[8]  = 8;
   ndigits[9]  = 8;
   ndigits[10] = 8;
   ndigits[11] = 8;
   ndigits[12] = 8;
   ndigits[13] = 8;
   ndigits[14] = 8;
   ndigits[15] = 8;

   for (i = 0; i < num_levels_total; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      stats_next = hypre_MatrixStatsArrayEntry(stats_array, i + 1);

      total_num_rows = hypre_MatrixStatsNumRows(stats);
      if (square)
      {
         if (i < num_levels_total - 1)
         {
            coarse_num_rows = hypre_MatrixStatsNumRows(stats_next);
         }
         else
         {
            coarse_num_rows = 0;
         }
      }
      else
      {
         coarse_num_rows = hypre_MatrixStatsNumCols(stats);
      }
      fine_num_rows = total_num_rows - coarse_num_rows;

      ndigits[1] = hypre_max(ndigits[1], 1 + hypre_ndigits(fine_num_rows));
      ndigits[3] = hypre_max(ndigits[3],
                             4 + hypre_ndigits((HYPRE_Int) hypre_MatrixStatsSparsity(stats)));
      ndigits[4] = hypre_max(ndigits[4],
                             1 + hypre_ndigits(hypre_MatrixStatsNnzrowMin(stats)));
      ndigits[5] = hypre_max(ndigits[5],
                             1 + hypre_ndigits(hypre_MatrixStatsNnzrowMax(stats)));
      ndigits[6] = hypre_max(ndigits[6],
                             4 + hypre_ndigits((HYPRE_Int) hypre_MatrixStatsNnzrowAvg(stats)));
      ndigits[7] = hypre_max(ndigits[7],
                             4 + hypre_ndigits((HYPRE_Int) hypre_MatrixStatsNnzrowStDev(stats)));
   }

   /* Column offsets calculation */
   offsets[0] = 6 + ndigits[0] + ndigits[1] + ndigits[2];
   offsets[1] = 0 + ndigits[2];
   offsets[2] = 2 + ndigits[3];
   offsets[3] = 7 + (ndigits[4] + ndigits[5] + ndigits[6] + ndigits[7]) / 2;
   offsets[4] = (1 + ndigits[4] + ndigits[5] + ndigits[6] + ndigits[7]) / 2 - 3;
   offsets[5] = 4 + (ndigits[8] + ndigits[9] + ndigits[10] + ndigits[11]) / 2;
   offsets[6] = (1 + ndigits[8] + ndigits[9] + ndigits[10] + ndigits[11]) / 2 - 3;
   offsets[7] = 4 + (ndigits[12] + ndigits[13] + ndigits[14] + ndigits[15]) / 2;
   if (!square)
   {
      offsets[0] += 2;
   }

   /* Table divisors */
   if (use_divisors)
   {
      divisors[0] = 9 + ndigits[0] + ndigits[1] +  ndigits[3] + 2 * ndigits[2];
      divisors[1] = 5 + ndigits[4] + ndigits[5] +  ndigits[6] +     ndigits[7];
      divisors[2] = 5 + ndigits[8] + ndigits[9] + ndigits[10] +    ndigits[11];
      divisors[3] = 5 + ndigits[12] + ndigits[13] + ndigits[14] + ndigits[15];
      if (!square)
      {
         divisors[0] += 2;
      }
   }
   else
   {
      divisors[0] = 4 + ndigits[2]; /* Note: ndigits[2] happens twice */
      for (i = 0; i < HYPRE_NDIGITS_SIZE; i++)
      {
         divisors[0] += ndigits[i] + 1;
      }
   }

   /* Header first line */
   HYPRE_PRINT_INDENT(shift);
   {
      hypre_printf("\n%s",   messages[0]);
   }
   HYPRE_PRINT_INDENT(shift);
   {
      hypre_printf("%*s ", offsets[0], "nonzero");
      hypre_printf("%*s ", offsets[1], "actual");
   }
   if (use_divisors)
   {
      hypre_printf("%*s ", offsets[2], "|");
   }
   {
      hypre_printf("%*s ", offsets[3], "entries/row");
   }
   if (use_divisors)
   {
      hypre_printf("%*s ", offsets[4], "|");
   }
   {
      hypre_printf("%*s ", offsets[5], "rowsums");
   }
   if (use_divisors)
   {
      hypre_printf("%*s ", offsets[6], "|");
   }
   {
      hypre_printf("%*s ", offsets[7], "absrowsums");
      hypre_printf("\n");
   }

   /* Header second line */
   HYPRE_PRINT_INDENT(shift);
   {
      hypre_printf("%s ", "lev");
   }
   if (square)
   {
      hypre_printf("%*s ",  ndigits[0], "rows");
      hypre_printf("%*s ",  ndigits[1], "fine");
      hypre_printf("%*s ",  ndigits[2], "entries");
   }
   else
   {
      hypre_printf("%*s ",   ndigits[0], "rows");
      hypre_printf("%-*s ",  ndigits[1], "x cols");
      hypre_printf("%*s ",   ndigits[2], "   entries");
   }
   {
      hypre_printf("%*s ",  ndigits[2], "entries");
      hypre_printf("%*s ",  ndigits[3], "sparse");
   }
   if (use_divisors)
   {
      hypre_printf("| ");
   }
   {
      /* entries per row */
      hypre_printf("%*s ",  ndigits[4], "min");
      hypre_printf("%*s ",  ndigits[5], "max");
      hypre_printf("%*s ",  ndigits[6], "avg");
      hypre_printf("%*s ",  ndigits[7], "stdev");
   }
   if (use_divisors)
   {
      hypre_printf("| ");
   }
   {
      /* rowsums */
      hypre_printf("%*s ",  ndigits[8], "min");
      hypre_printf("%*s ",  ndigits[9], "max");
      hypre_printf("%*s ", ndigits[10], "avg");
      hypre_printf("%*s ", ndigits[11], "stdev");
   }
   if (use_divisors)
   {
      hypre_printf("| ");
   }
   {
      /* absrowsums */
      hypre_printf("%*s ", ndigits[12], "min");
      hypre_printf("%*s ", ndigits[13], "max");
      hypre_printf("%*s ", ndigits[14], "avg");
      hypre_printf("%*s ", ndigits[15], "stdev");
   }
   {
      hypre_printf("\n");
   }
   HYPRE_PRINT_INDENT(shift);
   if (use_divisors)
   {
      HYPRE_PRINT_TOP_DIVISOR(4, divisors);
   }
   else
   {
      HYPRE_PRINT_TOP_DIVISOR(1, divisors);
   }

   /* Values */
   for (i = 0; i < num_levels_total; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      stats_next = hypre_MatrixStatsArrayEntry(stats_array, i + 1);

      total_num_rows = hypre_MatrixStatsNumRows(stats);
      if (square)
      {
         if (i < num_levels_total - 1)
         {
            coarse_num_rows = hypre_MatrixStatsNumRows(stats_next);
         }
         else
         {
            coarse_num_rows = 0;
         }
      }
      else
      {
         coarse_num_rows = hypre_MatrixStatsNumCols(stats);
      }
      fine_num_rows = total_num_rows - coarse_num_rows;

      /* General info */
      HYPRE_PRINT_INDENT(shift);
      if (square)
      {
         hypre_printf("%3d %*b %*b %*b %*b %*.3f ",
                      i,
                      ndigits[0], hypre_MatrixStatsNumRows(stats),
                      ndigits[1], fine_num_rows,
                      ndigits[2], hypre_MatrixStatsNumNonzeros(stats),
                      ndigits[2], hypre_MatrixStatsActualNonzeros(stats),
                      ndigits[3], hypre_MatrixStatsSparsity(stats));
      }
      else
      {
         hypre_printf("%3d %*b x %-*b %*b %*b %*.3f ",
                      i,
                      ndigits[0], hypre_MatrixStatsNumRows(stats),
                      ndigits[1], coarse_num_rows,
                      ndigits[2], hypre_MatrixStatsNumNonzeros(stats),
                      ndigits[2], hypre_MatrixStatsActualNonzeros(stats),
                      ndigits[3], hypre_MatrixStatsSparsity(stats));
      }
      if (use_divisors)
      {
         hypre_printf("| ");
      }

      /* Entries per row info */
      hypre_printf("%*d %*d %*.1f %*.2f ",
                   ndigits[4], hypre_MatrixStatsNnzrowMin(stats),
                   ndigits[5], hypre_MatrixStatsNnzrowMax(stats),
                   ndigits[6], hypre_MatrixStatsNnzrowAvg(stats),
                   ndigits[7], hypre_MatrixStatsNnzrowStDev(stats));
      if (use_divisors)
      {
         hypre_printf("| ");
      }

      /* Row sum info */
      hypre_printf("%*.1e %*.1e %*.1e %*.1e",
                   ndigits[8], hypre_MatrixStatsRowsumMin(stats),
                   ndigits[9], hypre_MatrixStatsRowsumMax(stats),
                   ndigits[10], hypre_MatrixStatsRowsumAvg(stats),
                   ndigits[11], hypre_MatrixStatsRowsumStDev(stats));
      if (use_divisors)
      {
         hypre_printf(" | ");
      }
      else
      {
         hypre_printf(" ");
      }
      hypre_printf("%*.1e %*.1e %*.1e %*.1e",
                   ndigits[12], hypre_MatrixStatsAbsrowsumMin(stats),
                   ndigits[13], hypre_MatrixStatsAbsrowsumMax(stats),
                   ndigits[14], hypre_MatrixStatsAbsrowsumAvg(stats),
                   ndigits[15], hypre_MatrixStatsAbsrowsumStDev(stats));

      if (use_divisors)
      {
         if (num_hierarchies == 1)
         {
            hypre_printf("\n");
         }
         else if (num_hierarchies == 2)
         {
            if (i == num_levels[0] / 2)
            {
               hypre_printf(messages[2]);
            }

            if ((num_levels[1] > 1) &&
                (i == num_levels[0] + num_levels[1] / 2))
            {
               hypre_printf(messages[3]);
            }

            hypre_printf("\n");
            HYPRE_PRINT_INDENT(shift);
            if (square)
            {
               if (i == num_levels[0])
               {
                  HYPRE_PRINT_MID_DIVISOR(4, divisors, messages[1]);
               }
            }
            else
            {
               if (i == (num_levels[0] - 1))
               {
                  HYPRE_PRINT_MID_DIVISOR(4, divisors, messages[1]);
               }
            }
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_hierarchies > 2 not implemented!");
            return hypre_error_flag;
         }
      }
      else
      {
         hypre_printf("\n");
      }
   }
   hypre_printf("\n\n");

   return hypre_error_flag;
}
