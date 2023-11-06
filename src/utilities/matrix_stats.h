/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_MATRIX_STATS_HEADER
#define hypre_MATRIX_STATS_HEADER

/******************************************************************************
 *
 * Header info for (generic) matrix statistics data structure
 *
 *****************************************************************************/

typedef struct hypre_MatrixStats_struct
{
   /* General info */
   HYPRE_BigInt        num_rows;
   HYPRE_BigInt        num_cols;
   hypre_ulonglongint  num_nonzeros;

   /* Actual nonzeros statistics */
   hypre_ulonglongint  actual_nonzeros;
   HYPRE_Real          actual_threshold;
   HYPRE_Real          sparsity;

   /* Nonzeros per row statistics */
   HYPRE_Int           nnzrow_min;
   HYPRE_Int           nnzrow_max;
   HYPRE_Real          nnzrow_avg;
   HYPRE_Real          nnzrow_stdev;
   HYPRE_Real          nnzrow_sqsum;

   /* Row sum statistics */
   HYPRE_Real          rowsum_min;
   HYPRE_Real          rowsum_max;
   HYPRE_Real          rowsum_avg;
   HYPRE_Real          rowsum_stdev;
   HYPRE_Real          rowsum_sqsum;
} hypre_MatrixStats;

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define hypre_MatrixStatsNumRows(data)               ((data) -> num_rows)
#define hypre_MatrixStatsNumCols(data)               ((data) -> num_cols)
#define hypre_MatrixStatsNumNonzeros(data)           ((data) -> num_nonzeros)

#define hypre_MatrixStatsSparsity(data)              ((data) -> sparsity)
#define hypre_MatrixStatsActualNonzeros(data)        ((data) -> actual_nonzeros)
#define hypre_MatrixStatsActualThreshold(data)       ((data) -> actual_threshold)

#define hypre_MatrixStatsNnzrowMin(data)             ((data) -> nnzrow_min)
#define hypre_MatrixStatsNnzrowMax(data)             ((data) -> nnzrow_max)
#define hypre_MatrixStatsNnzrowAvg(data)             ((data) -> nnzrow_avg)
#define hypre_MatrixStatsNnzrowStDev(data)           ((data) -> nnzrow_stdev)
#define hypre_MatrixStatsNnzrowSqsum(data)           ((data) -> nnzrow_sqsum)

#define hypre_MatrixStatsRowsumMin(data)             ((data) -> rowsum_min)
#define hypre_MatrixStatsRowsumMax(data)             ((data) -> rowsum_max)
#define hypre_MatrixStatsRowsumAvg(data)             ((data) -> rowsum_avg)
#define hypre_MatrixStatsRowsumStDev(data)           ((data) -> rowsum_stdev)
#define hypre_MatrixStatsRowsumSqsum(data)           ((data) -> rowsum_sqsum)

/******************************************************************************
 *
 * Header info for array of (generic) matrix statistics data structure
 *
 *****************************************************************************/

typedef struct hypre_MatrixStatsArray_struct
{
   HYPRE_Int             capacity;
   hypre_MatrixStats   **entries;
} hypre_MatrixStatsArray;

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define hypre_MatrixStatsArrayCapacity(data)         ((data) -> capacity)
#define hypre_MatrixStatsArrayEntries(data)          ((data) -> entries)
#define hypre_MatrixStatsArrayEntry(data, i)         ((data) -> entries[i])

/*--------------------------------------------------------------------------
 * Helper macros for table formatting
 *--------------------------------------------------------------------------*/

#define HYPRE_PRINT_TOP_DIVISOR(m, d)               \
   for (HYPRE_Int __i = 0; __i < m; __i++)          \
   {                                                \
      for (HYPRE_Int __j = 0; __j < d[__i]; __j++)  \
      {                                             \
         hypre_printf("%s", "=");                   \
      }                                             \
      if (__i < m - 1)                              \
      {                                             \
         hypre_printf("+");                         \
      }                                             \
   }                                                \
   hypre_printf("\n");

#define HYPRE_PRINT_MID_DIVISOR(m, d, msg)          \
   for (HYPRE_Int __i = 0; __i < m; __i++)          \
   {                                                \
      for (HYPRE_Int __j = 0; __j < d[__i]; __j++)  \
      {                                             \
         hypre_printf("%s", "-");                   \
      }                                             \
      if (__i < m - 1)                              \
      {                                             \
         hypre_printf("+");                         \
      }                                             \
   }                                                \
   hypre_printf(" %s\n", msg);

#define HYPRE_PRINT_INDENT(n)                       \
   hypre_printf("%*s", (n > 0) ? n : 0, "");


#define HYPRE_PRINT_SHIFTED_PARAM(n, ...)           \
   HYPRE_PRINT_INDENT(n)                            \
   hypre_printf(__VA_ARGS__)

#define HYPRE_NDIGITS_SIZE 12

#endif /* hypre_MATRIX_STATS_HEADER */
