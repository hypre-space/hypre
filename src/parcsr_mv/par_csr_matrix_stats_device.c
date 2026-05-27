/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Device functions for ParCSRMatrix statistics
 *
 *****************************************************************************/

#if defined(HYPRE_USING_GPU)
#include "_hypre_onedpl.hpp"
#endif

#include "_hypre_parcsr_mv.h"

#if defined(HYPRE_USING_GPU)
#include "_hypre_utilities.hpp"
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#include <thrust/transform_reduce.h>
#endif

#if defined(HYPRE_USING_SYCL)
namespace thrust = std;
#endif

typedef struct
{
   hypre_ulonglongint actual_nonzeros;
   HYPRE_Int          nnzrow_min;
   HYPRE_Int          nnzrow_max;
   HYPRE_Real         rowsum_min;
   HYPRE_Real         rowsum_max;
   HYPRE_Real         rowsum_sum;
   HYPRE_Real         absrowsum_min;
   HYPRE_Real         absrowsum_max;
   HYPRE_Real         absrowsum_sum;
} hypre_ParCSRMatrixStatsPassOneData;

typedef struct
{
   HYPRE_Real nnzrow_sqsum;
   HYPRE_Real rowsum_sqsum;
   HYPRE_Real absrowsum_sqsum;
} hypre_ParCSRMatrixStatsPassTwoData;

struct hypre_FunctorParCSRMatrixStatsPassOneRow
{
   HYPRE_Int     *diag_i;
   HYPRE_Complex *diag_a;
   HYPRE_Int     *offd_i;
   HYPRE_Complex *offd_a;
   HYPRE_Real     threshold;

   __host__ __device__
   hypre_ParCSRMatrixStatsPassOneData operator()(HYPRE_Int row) const
   {
      HYPRE_Int  diag_start = diag_i[row];
      HYPRE_Int  diag_end   = diag_i[row + 1];
      HYPRE_Int  offd_start = offd_i[row];
      HYPRE_Int  offd_end   = offd_i[row + 1];
      HYPRE_Int  row_nnz    = (diag_end - diag_start) + (offd_end - offd_start);
      HYPRE_Real rowsum     = 0.0;
      HYPRE_Real absrowsum  = 0.0;
      hypre_ulonglongint actual_nonzeros = 0ULL;

      for (HYPRE_Int j = diag_start; j < diag_end; j++)
      {
         HYPRE_Real absvalue = hypre_cabs(diag_a[j]);
         actual_nonzeros += (absvalue > threshold) ? 1ULL : 0ULL;
         rowsum          += diag_a[j];
         absrowsum       += absvalue;
      }

      for (HYPRE_Int j = offd_start; j < offd_end; j++)
      {
         HYPRE_Real absvalue = hypre_cabs(offd_a[j]);
         actual_nonzeros += (absvalue > threshold) ? 1ULL : 0ULL;
         rowsum          += offd_a[j];
         absrowsum       += absvalue;
      }

      hypre_ParCSRMatrixStatsPassOneData result;
      result.actual_nonzeros = actual_nonzeros;
      result.nnzrow_min      = row_nnz;
      result.nnzrow_max      = row_nnz;
      result.rowsum_min      = rowsum;
      result.rowsum_max      = rowsum;
      result.rowsum_sum      = rowsum;
      result.absrowsum_min   = absrowsum;
      result.absrowsum_max   = absrowsum;
      result.absrowsum_sum   = absrowsum;

      return result;
   }
};

struct hypre_FunctorParCSRMatrixStatsPassOneReduce
{
   __host__ __device__
   hypre_ParCSRMatrixStatsPassOneData
   operator()(const hypre_ParCSRMatrixStatsPassOneData &a,
              const hypre_ParCSRMatrixStatsPassOneData &b) const
   {
      hypre_ParCSRMatrixStatsPassOneData result;

      result.actual_nonzeros = a.actual_nonzeros + b.actual_nonzeros;
      result.nnzrow_min      = hypre_min(a.nnzrow_min, b.nnzrow_min);
      result.nnzrow_max      = hypre_max(a.nnzrow_max, b.nnzrow_max);
      result.rowsum_min      = hypre_min(a.rowsum_min, b.rowsum_min);
      result.rowsum_max      = hypre_max(a.rowsum_max, b.rowsum_max);
      result.rowsum_sum      = a.rowsum_sum + b.rowsum_sum;
      result.absrowsum_min   = hypre_min(a.absrowsum_min, b.absrowsum_min);
      result.absrowsum_max   = hypre_max(a.absrowsum_max, b.absrowsum_max);
      result.absrowsum_sum   = a.absrowsum_sum + b.absrowsum_sum;

      return result;
   }
};

struct hypre_FunctorParCSRMatrixStatsPassTwoRow
{
   HYPRE_Int     *diag_i;
   HYPRE_Complex *diag_a;
   HYPRE_Int     *offd_i;
   HYPRE_Complex *offd_a;
   HYPRE_Real     nnzrow_avg;
   HYPRE_Real     rowsum_avg;
   HYPRE_Real     absrowsum_avg;

   __host__ __device__
   hypre_ParCSRMatrixStatsPassTwoData operator()(HYPRE_Int row) const
   {
      HYPRE_Int  diag_start = diag_i[row];
      HYPRE_Int  diag_end   = diag_i[row + 1];
      HYPRE_Int  offd_start = offd_i[row];
      HYPRE_Int  offd_end   = offd_i[row + 1];
      HYPRE_Int  row_nnz    = (diag_end - diag_start) + (offd_end - offd_start);
      HYPRE_Real rowsum     = 0.0;
      HYPRE_Real absrowsum  = 0.0;

      for (HYPRE_Int j = diag_start; j < diag_end; j++)
      {
         rowsum    += diag_a[j];
         absrowsum += hypre_cabs(diag_a[j]);
      }

      for (HYPRE_Int j = offd_start; j < offd_end; j++)
      {
         rowsum    += offd_a[j];
         absrowsum += hypre_cabs(offd_a[j]);
      }

      hypre_ParCSRMatrixStatsPassTwoData result;
      result.nnzrow_sqsum    = hypre_squared((HYPRE_Real) row_nnz - nnzrow_avg);
      result.rowsum_sqsum    = hypre_squared(rowsum - rowsum_avg);
      result.absrowsum_sqsum = hypre_squared(absrowsum - absrowsum_avg);

      return result;
   }
};

struct hypre_FunctorParCSRMatrixStatsPassTwoReduce
{
   __host__ __device__
   hypre_ParCSRMatrixStatsPassTwoData
   operator()(const hypre_ParCSRMatrixStatsPassTwoData &a,
              const hypre_ParCSRMatrixStatsPassTwoData &b) const
   {
      hypre_ParCSRMatrixStatsPassTwoData result;

      result.nnzrow_sqsum    = a.nnzrow_sqsum    + b.nnzrow_sqsum;
      result.rowsum_sqsum    = a.rowsum_sqsum    + b.rowsum_sqsum;
      result.absrowsum_sqsum = a.absrowsum_sqsum + b.absrowsum_sqsum;

      return result;
   }
};

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassOneLocalDevice
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
#endif
HYPRE_Int
hypre_ParCSRMatrixStatsComputePassOneLocalDevice(hypre_ParCSRMatrix *A,
                                                 hypre_MatrixStats  *stats)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_ParCSRMatrixStatsPassOneData init;
   init.actual_nonzeros = 0ULL;
   init.nnzrow_min      = HYPRE_INT_MAX;
   init.nnzrow_max      = 0;
   init.rowsum_min      = HYPRE_REAL_MAX;
   init.rowsum_max      = -HYPRE_REAL_MAX;
   init.rowsum_sum      = 0.0;
   init.absrowsum_min   = HYPRE_REAL_MAX;
   init.absrowsum_max   = 0.0;
   init.absrowsum_sum   = 0.0;

   if (num_rows <= 0)
   {
      hypre_MatrixStatsActualNonzeros(stats) = init.actual_nonzeros;
      hypre_MatrixStatsNnzrowMin(stats)      = init.nnzrow_min;
      hypre_MatrixStatsNnzrowMax(stats)      = init.nnzrow_max;
      hypre_MatrixStatsRowsumMin(stats)      = init.rowsum_min;
      hypre_MatrixStatsRowsumMax(stats)      = init.rowsum_max;
      hypre_MatrixStatsRowsumAvg(stats)      = init.rowsum_sum;
      hypre_MatrixStatsAbsrowsumMin(stats)   = init.absrowsum_min;
      hypre_MatrixStatsAbsrowsumMax(stats)   = init.absrowsum_max;
      hypre_MatrixStatsAbsrowsumAvg(stats)   = init.absrowsum_sum;
      return hypre_error_flag;
   }

   hypre_FunctorParCSRMatrixStatsPassOneRow row_op =
   {
      hypre_CSRMatrixI(diag),
      hypre_CSRMatrixData(diag),
      hypre_CSRMatrixI(offd),
      hypre_CSRMatrixData(offd),
      hypre_MatrixStatsActualThreshold(stats)
   };

#if defined(HYPRE_USING_SYCL)
   auto begin = oneapi::dpl::counting_iterator<HYPRE_Int>(0);
   hypre_ParCSRMatrixStatsPassOneData result =
      HYPRE_ONEDPL_CALL(std::transform_reduce,
                        begin,
                        begin + num_rows,
                        init,
                        hypre_FunctorParCSRMatrixStatsPassOneReduce(),
                        row_op);
#else
   auto begin = thrust::make_counting_iterator<HYPRE_Int>(0);
   hypre_ParCSRMatrixStatsPassOneData result =
      HYPRE_THRUST_CALL(transform_reduce,
                        begin,
                        begin + num_rows,
                        row_op,
                        init,
                        hypre_FunctorParCSRMatrixStatsPassOneReduce());
#endif

   hypre_MatrixStatsActualNonzeros(stats) = result.actual_nonzeros;
   hypre_MatrixStatsNnzrowMin(stats)      = result.nnzrow_min;
   hypre_MatrixStatsNnzrowMax(stats)      = result.nnzrow_max;
   hypre_MatrixStatsRowsumMin(stats)      = result.rowsum_min;
   hypre_MatrixStatsRowsumMax(stats)      = result.rowsum_max;
   hypre_MatrixStatsRowsumAvg(stats)      = result.rowsum_sum;
   hypre_MatrixStatsAbsrowsumMin(stats)   = result.absrowsum_min;
   hypre_MatrixStatsAbsrowsumMax(stats)   = result.absrowsum_max;
   hypre_MatrixStatsAbsrowsumAvg(stats)   = result.absrowsum_sum;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassTwoLocalDevice
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
#endif
HYPRE_Int
hypre_ParCSRMatrixStatsComputePassTwoLocalDevice(hypre_ParCSRMatrix *A,
                                                 hypre_MatrixStats  *stats)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_ParCSRMatrixStatsPassTwoData init;
   init.nnzrow_sqsum    = 0.0;
   init.rowsum_sqsum    = 0.0;
   init.absrowsum_sqsum = 0.0;

   if (num_rows <= 0)
   {
      hypre_MatrixStatsNnzrowSqsum(stats)    = init.nnzrow_sqsum;
      hypre_MatrixStatsRowsumSqsum(stats)    = init.rowsum_sqsum;
      hypre_MatrixStatsAbsrowsumSqsum(stats) = init.absrowsum_sqsum;
      return hypre_error_flag;
   }

   hypre_FunctorParCSRMatrixStatsPassTwoRow row_op =
   {
      hypre_CSRMatrixI(diag),
      hypre_CSRMatrixData(diag),
      hypre_CSRMatrixI(offd),
      hypre_CSRMatrixData(offd),
      hypre_MatrixStatsNnzrowAvg(stats),
      hypre_MatrixStatsRowsumAvg(stats),
      hypre_MatrixStatsAbsrowsumAvg(stats)
   };

#if defined(HYPRE_USING_SYCL)
   auto begin = oneapi::dpl::counting_iterator<HYPRE_Int>(0);
   hypre_ParCSRMatrixStatsPassTwoData result =
      HYPRE_ONEDPL_CALL(std::transform_reduce,
                        begin,
                        begin + num_rows,
                        init,
                        hypre_FunctorParCSRMatrixStatsPassTwoReduce(),
                        row_op);
#else
   auto begin = thrust::make_counting_iterator<HYPRE_Int>(0);
   hypre_ParCSRMatrixStatsPassTwoData result =
      HYPRE_THRUST_CALL(transform_reduce,
                        begin,
                        begin + num_rows,
                        row_op,
                        init,
                        hypre_FunctorParCSRMatrixStatsPassTwoReduce());
#endif

   hypre_MatrixStatsNnzrowSqsum(stats)    = result.nnzrow_sqsum;
   hypre_MatrixStatsRowsumSqsum(stats)    = result.rowsum_sqsum;
   hypre_MatrixStatsAbsrowsumSqsum(stats) = result.absrowsum_sqsum;

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
