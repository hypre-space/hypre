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

#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_ParCSRMatrixStatsComputePassOneLocalDevice(hypre_ParCSRMatrix *A,
                                                           hypre_MatrixStats  *stats);
HYPRE_Int hypre_ParCSRMatrixStatsComputePassTwoLocalDevice(hypre_ParCSRMatrix *A,
                                                           hypre_MatrixStats  *stats);
#endif

/* Shortcuts */
#define sendbuffer(i, j, lda) sendbuffer[i * lda + j]
#define recvbuffer(i, j, lda) recvbuffer[i * lda + j]

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassOneLocalHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputePassOneLocalHost(hypre_ParCSRMatrix   *A,
                                               hypre_MatrixStats    *stats)
{
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
   HYPRE_Real          *absrowsum_min;
   HYPRE_Real          *absrowsum_max;
   HYPRE_Real          *absrowsum_avg;

   /* Local variables */
   HYPRE_Int            i, j;
   HYPRE_Int            num_rows;
   HYPRE_Int            num_threads = hypre_NumThreads();

   /* Allocate memory */
   actual_nonzeros = hypre_TAlloc(hypre_ulonglongint, num_threads, HYPRE_MEMORY_HOST);
   nnzrow_min      = hypre_TAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   nnzrow_max      = hypre_TAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   rowsum_min      = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   rowsum_max      = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   rowsum_avg      = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   absrowsum_min   = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   absrowsum_max   = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   absrowsum_avg   = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);

   /* ParCSRMatrix info */
   num_rows = hypre_ParCSRMatrixNumRows(A);

   /* Diag matrix data */
   diag   = hypre_ParCSRMatrixDiag(A);
   diag_i = hypre_CSRMatrixI(diag);
   diag_a = hypre_CSRMatrixData(diag);

   /* Offd matrix data */
   offd   = hypre_ParCSRMatrixOffd(A);
   offd_i = hypre_CSRMatrixI(offd);
   offd_a = hypre_CSRMatrixData(offd);

   /* Initialize local thread arrays */
   for (i = 0; i < num_threads; i++)
   {
      actual_nonzeros[i] = 0ULL;
      nnzrow_min[i]      = hypre_pow2(30);
      nnzrow_max[i]      = 0;
      rowsum_min[i]      = hypre_pow(2, 100);
      rowsum_max[i]      = - hypre_pow(2, 100);
      rowsum_avg[i]      = 0.0;
      absrowsum_min[i]   = hypre_pow(2, 100);
      absrowsum_max[i]   = 0.0;
      absrowsum_avg[i]   = 0.0;
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i, j)
#endif
   {
      HYPRE_Int   nnzrow, ns, ne;
      HYPRE_Int   mytid = hypre_GetThreadNum();
      HYPRE_Real  threshold = hypre_MatrixStatsActualThreshold(stats);
      HYPRE_Real  rowsum;
      HYPRE_Real  absrowsum;

      hypre_partition1D(num_rows, hypre_NumActiveThreads(), mytid, &ns, &ne);

      for (i = ns; i < ne; i++)
      {
         nnzrow = (diag_i[i + 1] - diag_i[i]) + (offd_i[i + 1] - offd_i[i]);

         rowsum    = 0.0;
         absrowsum = 0.0;
         for (j = diag_i[i]; j < diag_i[i + 1]; j++)
         {
            actual_nonzeros[mytid] += (hypre_cabs(diag_a[j]) > threshold) ? 1 : 0;
            rowsum    += diag_a[j];
            absrowsum += hypre_cabs(diag_a[j]);
         }

         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            actual_nonzeros[mytid] += (hypre_cabs(offd_a[j]) > threshold) ? 1 : 0;
            rowsum    += offd_a[j];
            absrowsum += hypre_cabs(offd_a[j]);
         }

         /* Update sum quantities */
         rowsum_avg[mytid]    += rowsum;
         absrowsum_avg[mytid] += absrowsum;

         /* Update min quantities */
         nnzrow_min[mytid]    = (nnzrow_min[mytid]    > nnzrow)    ? nnzrow    : nnzrow_min[mytid];
         rowsum_min[mytid]    = (rowsum_min[mytid]    > rowsum)    ? rowsum    : rowsum_min[mytid];
         absrowsum_min[mytid] = (absrowsum_min[mytid] > absrowsum) ? absrowsum : absrowsum_min[mytid];

         /* Update max quantities */
         nnzrow_max[mytid]    = (nnzrow_max[mytid]    < nnzrow)    ? nnzrow    : nnzrow_max[mytid];
         rowsum_max[mytid]    = (rowsum_max[mytid]    < rowsum)    ? rowsum    : rowsum_max[mytid];
         absrowsum_max[mytid] = (absrowsum_max[mytid] < absrowsum) ? absrowsum : absrowsum_max[mytid];
      }
   } /* end of parallel region */

   /* Reduce along threads */
   for (i = 1; i < num_threads; i++)
   {
      actual_nonzeros[0] += actual_nonzeros[i];
      rowsum_avg[0]      += rowsum_avg[i];
      absrowsum_avg[0]   += absrowsum_avg[i];

      nnzrow_min[0]       = hypre_min(nnzrow_min[0], nnzrow_min[i]);
      nnzrow_max[0]       = hypre_max(nnzrow_max[0], nnzrow_max[i]);

      rowsum_min[0]       = hypre_min(rowsum_min[0], rowsum_min[i]);
      rowsum_max[0]       = hypre_max(rowsum_max[0], rowsum_max[i]);
      absrowsum_min[0]    = hypre_min(absrowsum_min[0], absrowsum_min[i]);
      absrowsum_max[0]    = hypre_max(absrowsum_max[0], absrowsum_max[i]);
   }

   /* Set output values */
   hypre_MatrixStatsActualNonzeros(stats) = actual_nonzeros[0];

   hypre_MatrixStatsNnzrowMin(stats)      = nnzrow_min[0];
   hypre_MatrixStatsNnzrowMax(stats)      = nnzrow_max[0];

   hypre_MatrixStatsRowsumMin(stats)      = rowsum_min[0];
   hypre_MatrixStatsRowsumMax(stats)      = rowsum_max[0];
   hypre_MatrixStatsRowsumAvg(stats)      = rowsum_avg[0];
   hypre_MatrixStatsAbsrowsumMin(stats)   = absrowsum_min[0];
   hypre_MatrixStatsAbsrowsumMax(stats)   = absrowsum_max[0];
   hypre_MatrixStatsAbsrowsumAvg(stats)   = absrowsum_avg[0];

   /* Free memory */
   hypre_TFree(actual_nonzeros, HYPRE_MEMORY_HOST);
   hypre_TFree(nnzrow_min, HYPRE_MEMORY_HOST);
   hypre_TFree(nnzrow_max, HYPRE_MEMORY_HOST);
   hypre_TFree(rowsum_min, HYPRE_MEMORY_HOST);
   hypre_TFree(rowsum_max, HYPRE_MEMORY_HOST);
   hypre_TFree(rowsum_avg, HYPRE_MEMORY_HOST);
   hypre_TFree(absrowsum_min, HYPRE_MEMORY_HOST);
   hypre_TFree(absrowsum_max, HYPRE_MEMORY_HOST);
   hypre_TFree(absrowsum_avg, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassTwoLocalHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputePassTwoLocalHost(hypre_ParCSRMatrix  *A,
                                               hypre_MatrixStats   *stats)
{
   /* Diag matrix data */
   hypre_CSRMatrix     *diag;
   HYPRE_Int           *diag_i;
   HYPRE_Complex       *diag_a;

   /* Offd matrix data */
   hypre_CSRMatrix     *offd;
   HYPRE_Int           *offd_i;
   HYPRE_Complex       *offd_a;

   /* Local arrays */
   HYPRE_Real          *nnzrow_avg;
   HYPRE_Real          *rowsum_avg;
   HYPRE_Real          *absrowsum_avg;
   HYPRE_Real          *nnzrow_sqsum;
   HYPRE_Real          *rowsum_sqsum;
   HYPRE_Real          *absrowsum_sqsum;

   /* Local variables */
   HYPRE_Int            i, j;
   HYPRE_Int            num_rows;
   HYPRE_Int            num_threads = hypre_NumThreads();
   HYPRE_Int            nnzrow;

   /* Allocate memory */
   nnzrow_avg   = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   rowsum_avg   = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   absrowsum_avg = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   nnzrow_sqsum = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   rowsum_sqsum = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);
   absrowsum_sqsum = hypre_TAlloc(HYPRE_Real, num_threads, HYPRE_MEMORY_HOST);

   /* Initialize matrix variables */
   diag     = hypre_ParCSRMatrixDiag(A);
   offd     = hypre_ParCSRMatrixOffd(A);
   diag_i   = hypre_CSRMatrixI(diag);
   offd_i   = hypre_CSRMatrixI(offd);
   diag_a   = hypre_CSRMatrixData(diag);
   offd_a   = hypre_CSRMatrixData(offd);
   num_rows = hypre_CSRMatrixNumRows(diag);

   /* Initialize local thread variables */
   for (i = 0; i < num_threads; i++)
   {
      rowsum_avg[i]   = hypre_MatrixStatsRowsumAvg(stats);
      absrowsum_avg[i] = hypre_MatrixStatsAbsrowsumAvg(stats);
      nnzrow_avg[i]   = hypre_MatrixStatsNnzrowAvg(stats);
      nnzrow_sqsum[i] = 0.0;
      rowsum_sqsum[i] = 0.0;
      absrowsum_sqsum[i] = 0.0;
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i, j)
#endif
   {
      HYPRE_Int   ns, ne;
      HYPRE_Int   mytid = hypre_GetThreadNum();
      HYPRE_Real  rowsum;
      HYPRE_Real  absrowsum;

      hypre_partition1D(num_rows, hypre_NumActiveThreads(), mytid, &ns, &ne);

      for (i = ns; i < ne; i++)
      {
         nnzrow = (diag_i[i + 1] - diag_i[i]) + (offd_i[i + 1] - offd_i[i]);

         rowsum    = 0.0;
         absrowsum = 0.0;
         for (j = diag_i[i]; j < diag_i[i + 1]; j++)
         {
            rowsum    += diag_a[j];
            absrowsum += hypre_cabs(diag_a[j]);
         }

         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            rowsum    += offd_a[j];
            absrowsum += hypre_cabs(offd_a[j]);
         }

         /* Update sum quantities */
         nnzrow_sqsum[mytid] += hypre_squared((HYPRE_Real) nnzrow - nnzrow_avg[mytid]);
         rowsum_sqsum[mytid] += hypre_squared(rowsum - rowsum_avg[mytid]);
         absrowsum_sqsum[mytid] += hypre_squared(absrowsum - absrowsum_avg[mytid]);
      }
   } /* end of parallel region */

   /* Reduce along threads */
   for (i = 1; i < num_threads; i++)
   {
      nnzrow_sqsum[0] += nnzrow_sqsum[i];
      rowsum_sqsum[0] += rowsum_sqsum[i];
      absrowsum_sqsum[0] += absrowsum_sqsum[i];
   }

   /* Set output values */
   hypre_MatrixStatsNnzrowSqsum(stats) = nnzrow_sqsum[0];
   hypre_MatrixStatsRowsumSqsum(stats) = rowsum_sqsum[0];
   hypre_MatrixStatsAbsrowsumSqsum(stats) = absrowsum_sqsum[0];

   /* Free memory */
   hypre_TFree(nnzrow_sqsum, HYPRE_MEMORY_HOST);
   hypre_TFree(rowsum_sqsum, HYPRE_MEMORY_HOST);
   hypre_TFree(absrowsum_sqsum, HYPRE_MEMORY_HOST);
   hypre_TFree(nnzrow_avg, HYPRE_MEMORY_HOST);
   hypre_TFree(rowsum_avg, HYPRE_MEMORY_HOST);
   hypre_TFree(absrowsum_avg, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassOneLocal
 *
 * Compute the first pass of matrix statistics locally on a rank consisting
 * of average (avg), minimum (min) and maximum (max) quantities.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputePassOneLocal(hypre_ParCSRMatrix *A,
                                           hypre_MatrixStats  *stats)
{
   /* Call backend implementations */
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixStatsComputePassOneLocalDevice(A, stats);
   }
   else
#endif
   {
      hypre_ParCSRMatrixStatsComputePassOneLocalHost(A, stats);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputePassTwoLocal
 *
 * Compute the second pass of matrix statistics locally on a rank consisting
 * of squared sum (sqsum) and standard deviation (stdev) quantities.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputePassTwoLocal(hypre_ParCSRMatrix *A,
                                           hypre_MatrixStats  *stats)
{
   /* Call backend implementations */
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixStatsComputePassTwoLocalDevice(A, stats);
   }
   else
#endif
   {
      hypre_ParCSRMatrixStatsComputePassTwoLocalHost(A, stats);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsComputeLocal
 *
 * Compute complete rank-local matrix statistics without any MPI reduction.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsComputeLocal(hypre_ParCSRMatrix *A,
                                    hypre_MatrixStats  *stats)
{
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   HYPRE_Int        local_num_rows;
   HYPRE_BigInt     local_num_cols;
   HYPRE_Int        local_num_nonzeros;
   HYPRE_Real       local_size;

   if (!A)
   {
      hypre_error_in_arg(1);
   }
   if (!stats)
   {
      hypre_error_in_arg(2);
   }
   if (!A || !stats)
   {
      return hypre_error_flag;
   }

   diag = hypre_ParCSRMatrixDiag(A);
   offd = hypre_ParCSRMatrixOffd(A);

   local_num_rows     = hypre_ParCSRMatrixNumRows(A);
   local_num_cols     = hypre_ParCSRMatrixGlobalNumCols(A);
   local_num_nonzeros = hypre_CSRMatrixNumNonzeros(diag) + hypre_CSRMatrixNumNonzeros(offd);

   hypre_MatrixStatsNumRows(stats)     = local_num_rows;
   hypre_MatrixStatsNumCols(stats)     = local_num_cols;
   hypre_MatrixStatsNumNonzeros(stats) = local_num_nonzeros;

   if (local_num_rows == 0)
   {
      hypre_MatrixStatsActualNonzeros(stats) = 0ULL;
      hypre_MatrixStatsSparsity(stats)       = 0.0;

      hypre_MatrixStatsNnzrowMin(stats)      = 0;
      hypre_MatrixStatsNnzrowMax(stats)      = 0;
      hypre_MatrixStatsNnzrowAvg(stats)      = 0.0;
      hypre_MatrixStatsNnzrowStDev(stats)    = 0.0;
      hypre_MatrixStatsNnzrowSqsum(stats)    = 0.0;

      hypre_MatrixStatsRowsumMin(stats)      = 0.0;
      hypre_MatrixStatsRowsumMax(stats)      = 0.0;
      hypre_MatrixStatsRowsumAvg(stats)      = 0.0;
      hypre_MatrixStatsRowsumStDev(stats)    = 0.0;
      hypre_MatrixStatsRowsumSqsum(stats)    = 0.0;

      hypre_MatrixStatsAbsrowsumMin(stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumMax(stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumAvg(stats)   = 0.0;
      hypre_MatrixStatsAbsrowsumStDev(stats) = 0.0;
      hypre_MatrixStatsAbsrowsumSqsum(stats) = 0.0;

      return hypre_error_flag;
   }

   hypre_ParCSRMatrixStatsComputePassOneLocal(A, stats);

   hypre_MatrixStatsNnzrowAvg(stats)    = (HYPRE_Real) local_num_nonzeros /
                                          (HYPRE_Real) local_num_rows;
   hypre_MatrixStatsRowsumAvg(stats)    = hypre_MatrixStatsRowsumAvg(stats) /
                                          (HYPRE_Real) local_num_rows;
   hypre_MatrixStatsAbsrowsumAvg(stats) = hypre_MatrixStatsAbsrowsumAvg(stats) /
                                          (HYPRE_Real) local_num_rows;

   local_size = (HYPRE_Real) local_num_rows * (HYPRE_Real) local_num_cols;
   hypre_MatrixStatsSparsity(stats) = local_size > 0.0 ?
                                      100.0 * (1.0 - (HYPRE_Real) local_num_nonzeros / local_size) :
                                      0.0;

   hypre_ParCSRMatrixStatsComputePassTwoLocal(A, stats);

   hypre_MatrixStatsNnzrowStDev(stats) = hypre_sqrt(hypre_MatrixStatsNnzrowSqsum(stats) /
                                                    (HYPRE_Real) local_num_rows);
   hypre_MatrixStatsRowsumStDev(stats) = hypre_sqrt(hypre_MatrixStatsRowsumSqsum(stats) /
                                                    (HYPRE_Real) local_num_rows);
   hypre_MatrixStatsAbsrowsumStDev(stats) = hypre_sqrt(hypre_MatrixStatsAbsrowsumSqsum(stats) /
                                                       (HYPRE_Real) local_num_rows);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixStatsArrayCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixStatsArrayCompute(HYPRE_Int                num_matrices,
                                    hypre_ParCSRMatrix     **matrices,
                                    hypre_MatrixStatsArray  *stats_array)
{
   hypre_MatrixStats     *stats;

   /* MPI buffers */
   HYPRE_Real            *recvbuffer;
   HYPRE_Real            *sendbuffer;

   /* Local variables */
   MPI_Comm               comm;
   hypre_CSRMatrix       *diag;
   hypre_CSRMatrix       *offd;
   HYPRE_Int              i;
   HYPRE_Int              local_num_rows;
   HYPRE_BigInt           global_num_rows;
   HYPRE_Real             global_size;

   /* Sanity check */
   if (num_matrices < 1)
   {
      return hypre_error_flag;
   }

   /* We assume all MPI communicators are equal */
   comm = hypre_ParCSRMatrixComm(matrices[0]);

   /* Allocate MPI buffers */
   recvbuffer = hypre_CTAlloc(HYPRE_Real, 6 * num_matrices, HYPRE_MEMORY_HOST);
   sendbuffer = hypre_CTAlloc(HYPRE_Real, 6 * num_matrices, HYPRE_MEMORY_HOST);

   /* Set matrix dimensions */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      hypre_MatrixStatsNumRows(stats) = hypre_ParCSRMatrixGlobalNumRows(matrices[i]);
      hypre_MatrixStatsNumCols(stats) = hypre_ParCSRMatrixGlobalNumCols(matrices[i]);
   }

   /*-------------------------------------------------
    *  First pass for computing statistics
    *-------------------------------------------------*/

   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      hypre_ParCSRMatrixStatsComputePassOneLocal(matrices[i], stats);
   }

   /*-------------------------------------------------
    *  Global reduce for min/max quantities
    *-------------------------------------------------*/

   /* Pack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      local_num_rows = hypre_ParCSRMatrixNumRows(matrices[i]);

      if (local_num_rows > 0)
      {
         sendbuffer(i, 0, 6) = (HYPRE_Real) - hypre_MatrixStatsNnzrowMin(stats);
         sendbuffer(i, 1, 6) = (HYPRE_Real)   hypre_MatrixStatsNnzrowMax(stats);
         sendbuffer(i, 2, 6) = (HYPRE_Real) - hypre_MatrixStatsRowsumMin(stats);
         sendbuffer(i, 3, 6) = (HYPRE_Real)   hypre_MatrixStatsRowsumMax(stats);
         sendbuffer(i, 4, 6) = (HYPRE_Real) - hypre_MatrixStatsAbsrowsumMin(stats);
         sendbuffer(i, 5, 6) = (HYPRE_Real)   hypre_MatrixStatsAbsrowsumMax(stats);
      }
      else
      {
         sendbuffer(i, 0, 6) = -HYPRE_REAL_MAX;
         sendbuffer(i, 1, 6) = -HYPRE_REAL_MAX;
         sendbuffer(i, 2, 6) = -HYPRE_REAL_MAX;
         sendbuffer(i, 3, 6) = -HYPRE_REAL_MAX;
         sendbuffer(i, 4, 6) = -HYPRE_REAL_MAX;
         sendbuffer(i, 5, 6) = -HYPRE_REAL_MAX;
      }
   }

   hypre_MPI_Reduce(sendbuffer, recvbuffer, 6 * num_matrices,
                    HYPRE_MPI_REAL, hypre_MPI_MAX, 0, comm);

   /* Unpack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      hypre_MatrixStatsNnzrowMin(stats)    = (HYPRE_Int)  - recvbuffer(i, 0, 6);
      hypre_MatrixStatsNnzrowMax(stats)    = (HYPRE_Int)    recvbuffer(i, 1, 6);
      hypre_MatrixStatsRowsumMin(stats)    = (HYPRE_Real) - recvbuffer(i, 2, 6);
      hypre_MatrixStatsRowsumMax(stats)    = (HYPRE_Real)   recvbuffer(i, 3, 6);
      hypre_MatrixStatsAbsrowsumMin(stats) = (HYPRE_Real) - recvbuffer(i, 4, 6);
      hypre_MatrixStatsAbsrowsumMax(stats) = (HYPRE_Real)   recvbuffer(i, 5, 6);
   }

   /*-------------------------------------------------
    *  Global reduce for summation quantities
    *-------------------------------------------------*/

   /* Pack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      diag  = hypre_ParCSRMatrixDiag(matrices[i]);
      offd  = hypre_ParCSRMatrixOffd(matrices[i]);

      sendbuffer(i, 0, 4) = (HYPRE_Real) (hypre_CSRMatrixNumNonzeros(diag) +
                                          hypre_CSRMatrixNumNonzeros(offd));
      sendbuffer(i, 1, 4) = (HYPRE_Real)  hypre_MatrixStatsActualNonzeros(stats);
      sendbuffer(i, 2, 4) = (HYPRE_Real)  hypre_MatrixStatsRowsumAvg(stats);
      sendbuffer(i, 3, 4) = (HYPRE_Real)  hypre_MatrixStatsAbsrowsumAvg(stats);
   }

   hypre_MPI_Reduce(sendbuffer, recvbuffer, 4 * num_matrices,
                    HYPRE_MPI_REAL, hypre_MPI_SUM, 0, comm);

   /* Unpack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrices[i]);
      global_size     = hypre_squared((HYPRE_Real) global_num_rows);

      hypre_MatrixStatsNumNonzeros(stats)    = (hypre_ulonglongint) recvbuffer(i, 0, 4);
      hypre_MatrixStatsActualNonzeros(stats) = (hypre_ulonglongint) recvbuffer(i, 1, 4);
      hypre_MatrixStatsRowsumAvg(stats)      = (HYPRE_Real)         recvbuffer(i, 2, 4) /
                                               (HYPRE_Real)         global_num_rows;
      hypre_MatrixStatsAbsrowsumAvg(stats)   = (HYPRE_Real)         recvbuffer(i, 3, 4) /
                                               (HYPRE_Real)         global_num_rows;
      hypre_MatrixStatsNnzrowAvg(stats)      = (HYPRE_Real)         recvbuffer(i, 0, 4) /
                                               (HYPRE_Real)         global_num_rows;

      hypre_MatrixStatsSparsity(stats)       = 100.0 * (1.0 - recvbuffer(i, 0, 4) / global_size);

      hypre_ParCSRMatrixNumNonzeros(matrices[i]) = (HYPRE_Int) recvbuffer(i, 0, 4);
      hypre_ParCSRMatrixDNumNonzeros(matrices[i]) = (hypre_double) recvbuffer(i, 0, 4);
   }

   /*-------------------------------------------------
    *  Second pass for computing statistics
    *-------------------------------------------------*/

   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      hypre_ParCSRMatrixStatsComputePassTwoLocal(matrices[i], stats);
   }

   /*-------------------------------------------------
    *  Global reduce for summation quantities
    *-------------------------------------------------*/

   /* Pack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);

      sendbuffer(i, 0, 3) = hypre_MatrixStatsNnzrowSqsum(stats);
      sendbuffer(i, 1, 3) = hypre_MatrixStatsRowsumSqsum(stats);
      sendbuffer(i, 2, 3) = hypre_MatrixStatsAbsrowsumSqsum(stats);
   }

   hypre_MPI_Reduce(sendbuffer, recvbuffer, 3 * num_matrices,
                    HYPRE_MPI_REAL, hypre_MPI_SUM, 0, comm);

   /* Unpack MPI buffers */
   for (i = 0; i < num_matrices; i++)
   {
      stats = hypre_MatrixStatsArrayEntry(stats_array, i);
      global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrices[i]);

      hypre_MatrixStatsNnzrowSqsum(stats)    = recvbuffer(i, 0, 3);
      hypre_MatrixStatsRowsumSqsum(stats)    = recvbuffer(i, 1, 3);
      hypre_MatrixStatsAbsrowsumSqsum(stats) = recvbuffer(i, 2, 3);

      hypre_MatrixStatsNnzrowStDev(stats) = hypre_sqrt(recvbuffer(i, 0, 3) /
                                                       (HYPRE_Real) global_num_rows);
      hypre_MatrixStatsRowsumStDev(stats) = hypre_sqrt(recvbuffer(i, 1, 3) /
                                                       (HYPRE_Real) global_num_rows);
      hypre_MatrixStatsAbsrowsumStDev(stats) = hypre_sqrt(recvbuffer(i, 2, 3) /
                                                          (HYPRE_Real) global_num_rows);
   }

   /* Free MPI buffers */
   hypre_TFree(recvbuffer, HYPRE_MEMORY_HOST);
   hypre_TFree(sendbuffer, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
