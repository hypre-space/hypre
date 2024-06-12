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
   HYPRE_Int            offsets[6];
   HYPRE_Int            divisors[3];
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
   ndigits[2]  = hypre_max(8, hypre_ndigits(hypre_MatrixStatsNumNonzeros(stats_finest)));
   ndigits[3]  = 7;
   ndigits[4]  = 4;
   ndigits[5]  = 4;
   ndigits[6]  = 4;
   ndigits[7]  = 4;
   ndigits[8]  = 8;
   ndigits[9]  = 8;
   ndigits[10] = 8;
   ndigits[11] = 8;

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
   offsets[4] = (ndigits[4] + ndigits[5] + ndigits[6] + ndigits[7]) / 2 +
                (ndigits[4] + ndigits[5] + ndigits[6] + ndigits[7]) % 2 - 3;
   offsets[5] = 4 + (ndigits[8] + ndigits[9] + ndigits[10] + ndigits[11]) / 2;
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
      // entries per row
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
      // rowsums
      hypre_printf("%*s ",  ndigits[8], "min");
      hypre_printf("%*s ",  ndigits[9], "max");
      hypre_printf("%*s ", ndigits[10], "avg");
      hypre_printf("%*s ", ndigits[11], "stdev");
   }
   {
      hypre_printf("\n");
   }
   HYPRE_PRINT_INDENT(shift);
   if (use_divisors)
   {
      HYPRE_PRINT_TOP_DIVISOR(3, divisors);
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
                  HYPRE_PRINT_MID_DIVISOR(3, divisors, messages[1]);
               }
            }
            else
            {
               if (i == (num_levels[0] - 1))
               {
                  HYPRE_PRINT_MID_DIVISOR(3, divisors, messages[1]);
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
