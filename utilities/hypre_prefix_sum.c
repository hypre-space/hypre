#include "_hypre_utilities.h"

static HYPRE_Int prefix_sum[2048];

void hypre_prefix_sum(HYPRE_Int *in_out, HYPRE_Int *sum)
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   prefix_sum[my_thread_num + 1] = *in_out;

#pragma omp barrier
#pragma omp master
   {
      prefix_sum[0] = 0;
      HYPRE_Int i;
      for (i = 1; i < num_threads; i++)
      {
         prefix_sum[i + 1] += prefix_sum[i];
      }
      *sum = prefix_sum[num_threads];
   }
#pragma omp barrier

   *in_out = prefix_sum[my_thread_num];
#else /* !HYPRE_USING_OPENMP */
   *sum = *in_out;
   *in_out = 0;
#endif /* !HYPRE_USING_OPENMP */
}
