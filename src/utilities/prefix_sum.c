/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * This is a helper routine to compute a prefix sum of integer values.
 *
 * The current implementation is okay for modest numbers of threads.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PrefixSumInt(HYPRE_Int   nvals,
                   HYPRE_Int  *vals,
                   HYPRE_Int  *sums)
{
   HYPRE_Int  j, nthreads, bsize;

   nthreads = hypre_NumThreads();
   bsize = (nvals + nthreads - 1) / nthreads; /* This distributes the remainder */

   if (nvals < nthreads || bsize == 1)
   {
      sums[0] = 0;
      for (j = 1; j < nvals; j++)
      {
         sums[j] += sums[j - 1] + vals[j - 1];
      }
   }
   else
   {
      /* Compute preliminary partial sums (in parallel) within each interval */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j + bsize), nvals);

         sums[j] = 0;
         for (i = j + 1; i < n; i++)
         {
            sums[i] = sums[i - 1] + vals[i - 1];
         }
      }

      /* Compute final partial sums (in serial) for the first entry of every interval */
      for (j = bsize; j < nvals; j += bsize)
      {
         sums[j] = sums[j - bsize] + sums[j - 1] + vals[j - 1];
      }

      /* Compute final partial sums (in parallel) for the remaining entries */
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
#endif
      for (j = bsize; j < nvals; j += bsize)
      {
         HYPRE_Int  i, n = hypre_min((j + bsize), nvals);

         for (i = j + 1; i < n; i++)
         {
            sums[i] += sums[j];
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Assumed to be called within an omp region.
 * Let x_i be the input of ith thread.
 * The output of ith thread y_i = x_0 + x_1 + ... + x_{i-1}
 * Additionally, sum = x_0 + x_1 + ... + x_{nthreads - 1}
 * Note that always y_0 = 0
 *
 * @param workspace at least with length (nthreads+1)
 *        workspace[tid] will contain result for tid
 *        workspace[nthreads] will contain sum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_prefix_sum( HYPRE_Int *in_out,
                  HYPRE_Int *sum,
                  HYPRE_Int *workspace )
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[my_thread_num + 1] = *in_out;

   #pragma omp barrier
   #pragma omp master
   {
      HYPRE_Int i;
      workspace[0] = 0;
      for (i = 1; i < num_threads; i++)
      {
         workspace[i + 1] += workspace[i];
      }
      *sum = workspace[num_threads];
   }
   #pragma omp barrier

   *in_out = workspace[my_thread_num];
#else /* !HYPRE_USING_OPENMP */
   *sum = *in_out;
   *in_out = 0;

   workspace[0] = 0;
   workspace[1] = *sum;
#endif /* !HYPRE_USING_OPENMP */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This version does prefix sum in pair.
 * Useful when we prefix sum of diag and offd in tandem.
 *
 * @param worksapce at least with length 2*(nthreads+1)
 *        workspace[2*tid] and workspace[2*tid+1] will contain results for tid
 *        workspace[3*nthreads] and workspace[3*nthreads + 1] will contain sums
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_prefix_sum_pair( HYPRE_Int *in_out1,
                       HYPRE_Int *sum1,
                       HYPRE_Int *in_out2,
                       HYPRE_Int *sum2,
                       HYPRE_Int *workspace )
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1) * 2] = *in_out1;
   workspace[(my_thread_num + 1) * 2 + 1] = *in_out2;

   #pragma omp barrier
   #pragma omp master
   {
      HYPRE_Int i;
      workspace[0] = 0;
      workspace[1] = 0;

      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1) * 2] += workspace[i * 2];
         workspace[(i + 1) * 2 + 1] += workspace[i * 2 + 1];
      }
      *sum1 = workspace[num_threads * 2];
      *sum2 = workspace[num_threads * 2 + 1];
   }
   #pragma omp barrier

   *in_out1 = workspace[my_thread_num * 2];
   *in_out2 = workspace[my_thread_num * 2 + 1];
#else /* !HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *in_out1 = 0;
   *in_out2 = 0;

   workspace[0] = 0;
   workspace[1] = 0;
   workspace[2] = *sum1;
   workspace[3] = *sum2;
#endif /* !HYPRE_USING_OPENMP */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * @param workspace at least with length 3*(nthreads+1)
 *        workspace[3*tid:3*tid+3) will contain results for tid
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_prefix_sum_triple( HYPRE_Int *in_out1,
                         HYPRE_Int *sum1,
                         HYPRE_Int *in_out2,
                         HYPRE_Int *sum2,
                         HYPRE_Int *in_out3,
                         HYPRE_Int *sum3,
                         HYPRE_Int *workspace )
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1) * 3] = *in_out1;
   workspace[(my_thread_num + 1) * 3 + 1] = *in_out2;
   workspace[(my_thread_num + 1) * 3 + 2] = *in_out3;

   #pragma omp barrier
   #pragma omp master
   {
      HYPRE_Int i;
      workspace[0] = 0;
      workspace[1] = 0;
      workspace[2] = 0;

      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1) * 3] += workspace[i * 3];
         workspace[(i + 1) * 3 + 1] += workspace[i * 3 + 1];
         workspace[(i + 1) * 3 + 2] += workspace[i * 3 + 2];
      }
      *sum1 = workspace[num_threads * 3];
      *sum2 = workspace[num_threads * 3 + 1];
      *sum3 = workspace[num_threads * 3 + 2];
   }
   #pragma omp barrier

   *in_out1 = workspace[my_thread_num * 3];
   *in_out2 = workspace[my_thread_num * 3 + 1];
   *in_out3 = workspace[my_thread_num * 3 + 2];
#else /* !HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *sum3 = *in_out3;
   *in_out1 = 0;
   *in_out2 = 0;
   *in_out3 = 0;

   workspace[0] = 0;
   workspace[1] = 0;
   workspace[2] = 0;
   workspace[3] = *sum1;
   workspace[4] = *sum2;
   workspace[5] = *sum3;
#endif /* !HYPRE_USING_OPENMP */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * N prefix-sums together.
 * workspace[n*tid:n*(tid+1)) will contain results for tid
 * workspace[nthreads*tid:nthreads*(tid+1)) will contain sums
 *
 * @param workspace at least with length n*(nthreads+1)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_prefix_sum_multiple( HYPRE_Int *in_out,
                           HYPRE_Int *sum,
                           HYPRE_Int  n,
                           HYPRE_Int *workspace )
{
   HYPRE_Int i;
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   for (i = 0; i < n; i++)
   {
      workspace[(my_thread_num + 1)*n + i] = in_out[i];
   }

   #pragma omp barrier
   #pragma omp master
   {
      HYPRE_Int t;
      for (i = 0; i < n; i++)
      {
         workspace[i] = 0;
      }

      // assuming n is not so big, we don't parallelize this loop
      for (t = 1; t < num_threads; t++)
      {
         for (i = 0; i < n; i++)
         {
            workspace[(t + 1)*n + i] += workspace[t * n + i];
         }
      }

      for (i = 0; i < n; i++)
      {
         sum[i] = workspace[num_threads * n + i];
      }
   }
   #pragma omp barrier

   for (i = 0; i < n; i++)
   {
      in_out[i] = workspace[my_thread_num * n + i];
   }
#else /* !HYPRE_USING_OPENMP */
   for (i = 0; i < n; i++)
   {
      sum[i] = in_out[i];
      in_out[i] = 0;

      workspace[i] = 0;
      workspace[n + i] = sum[i];
   }
#endif /* !HYPRE_USING_OPENMP */

   return hypre_error_flag;
}
