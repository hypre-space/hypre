#include "_hypre_utilities.h"

void hypre_prefix_sum(HYPRE_Int *in_out, HYPRE_Int *sum, HYPRE_Int *workspace)
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[my_thread_num + 1] = *in_out;

#pragma omp barrier
#pragma omp master
   {
      workspace[0] = 0;
      HYPRE_Int i;
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
#endif /* !HYPRE_USING_OPENMP */
}

void hypre_prefix_sum_pair(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *workspace)
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1)*2] = *in_out1;
   workspace[(my_thread_num + 1)*2 + 1] = *in_out2;

#pragma omp barrier
#pragma omp master
   {
      workspace[0] = 0;
      workspace[1] = 0;

      HYPRE_Int i;
      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1)*2] += workspace[i*2];
         workspace[(i + 1)*2 + 1] += workspace[i*2 + 1];
      }
      *sum1 = workspace[num_threads*2];
      *sum2 = workspace[num_threads*2 + 1];
   }
#pragma omp barrier

   *in_out1 = workspace[my_thread_num*2];
   *in_out2 = workspace[my_thread_num*2 + 1];
#else /* !HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *in_out1 = 0;
   *in_out2 = 0;
#endif /* !HYPRE_USING_OPENMP */
}

void hypre_prefix_sum_triple(HYPRE_Int *in_out1, HYPRE_Int *sum1, HYPRE_Int *in_out2, HYPRE_Int *sum2, HYPRE_Int *in_out3, HYPRE_Int *sum3, HYPRE_Int *workspace)
{
#ifdef HYPRE_USING_OPENMP
   HYPRE_Int my_thread_num = hypre_GetThreadNum();
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   hypre_assert(1 == num_threads || omp_in_parallel());

   workspace[(my_thread_num + 1)*3] = *in_out1;
   workspace[(my_thread_num + 1)*3 + 1] = *in_out2;
   workspace[(my_thread_num + 1)*3 + 2] = *in_out3;

#pragma omp barrier
#pragma omp master
   {
      workspace[0] = 0;
      workspace[1] = 0;
      workspace[2] = 0;

      HYPRE_Int i;
      for (i = 1; i < num_threads; i++)
      {
         workspace[(i + 1)*3] += workspace[i*3];
         workspace[(i + 1)*3 + 1] += workspace[i*3 + 1];
         workspace[(i + 1)*3 + 2] += workspace[i*3 + 2];
      }
      *sum1 = workspace[num_threads*3];
      *sum2 = workspace[num_threads*3 + 1];
      *sum3 = workspace[num_threads*3 + 2];
   }
#pragma omp barrier

   *in_out1 = workspace[my_thread_num*3];
   *in_out2 = workspace[my_thread_num*3 + 1];
   *in_out3 = workspace[my_thread_num*3 + 2];
#else /* !HYPRE_USING_OPENMP */
   *sum1 = *in_out1;
   *sum2 = *in_out2;
   *sum3 = *in_out3;
   *in_out1 = 0;
   *in_out2 = 0;
   *in_out3 = 0;
#endif /* !HYPRE_USING_OPENMP */
}
