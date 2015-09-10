#include "_hypre_utilities.h"
//#define DBG_MERGE_SORT
#ifdef DBG_MERGE_SORT
#include <assert.h>
#include <algorithm>
#endif

#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

/**
 * merge two sorted (w/o duplication) sequences and eliminate
 * duplicates.
 *
 * @return length of merged output
 */
HYPRE_Int *hypre_merge_unique(HYPRE_Int *first1, HYPRE_Int *last1, HYPRE_Int *first2, HYPRE_Int *last2, HYPRE_Int *out)
{
   for ( ; first1 != last1; ++out)
   {
      if (first2 == last2)
      {
         for ( ; first1 != last1; ++out)
         {
            *out = *first1;
            ++first1;
            for ( ; first1 != last1 && *first1 == *out; ++first1);
         }
         return out;
      }
      if (*first1 == *first2)
      {
         *out = *first1;
         ++first1; ++first2;
         for ( ; first1 != last1 && *first1 == *out; ++first1);
         for ( ; first2 != last2 && *first2 == *out; ++first2);
      }
      else if (*first2 < *first1)
      {
         *out = *first2;
         ++first2;
         for ( ; first2 != last2 && *first2 == *out; ++first2);
      }
      else
      {
         *out = *first1;
         ++first1;
         for ( ; first1 != last1 && *first1 == *out; ++first1);
      }
   }
   for ( ; first2 != last2; ++out)
   {
      *out = *first2;
      ++first2;
      for ( ; first2 != last2 && *first2 == *out; ++first2);
   }
   return out;
}

void hypre_merge(HYPRE_Int *first1, HYPRE_Int *last1, HYPRE_Int *first2, HYPRE_Int *last2, HYPRE_Int *out)
{
   for ( ; first1 != last1; ++out)
   {
      if (first2 == last2)
      {
         for ( ; first1 != last1; ++first1, ++out)
         {
            *out = *first1;
         }
         return;
      }
      if (*first2 < *first1)
      {
         *out = *first2;
         ++first2;
      }
      else
      {
         *out = *first1;
         ++first1;
      }
   }
   for ( ; first2 != last2; ++first2, ++out)
   {
      *out = *first2;
   }
}


void kth_element_(
   HYPRE_Int *out1, HYPRE_Int *out2,
   HYPRE_Int *a1, HYPRE_Int *a2,
   HYPRE_Int left, HYPRE_Int right, HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
{
   while (1)
   {
      HYPRE_Int i = (left + right)/2; // right < k -> i < k
      HYPRE_Int j = k - i - 1;
#ifdef DBG_MERGE_SORT
      assert(left <= right && right <= k);
      assert(i < k); // i == k implies left == right == k that can never happen
      assert(j >= 0 && j < n2);
#endif

      if ((j == -1 || a1[i] >= a2[j]) && (j == n2 - 1 || a1[i] <= a2[j + 1]))
      {
         *out1 = i; *out2 = j + 1;
         return;
      }
      else if (j >= 0 && a2[j] >= a1[i] && (i == n1 - 1 || a2[j] <= a1[i + 1]))
      {
         *out1 = i + 1; *out2 = j;
         return;
      }
      else if (a1[i] > a2[j] && j != n2 - 1 && a1[i] > a2[j+1])
      {
         // search in left half of a1
         right = i - 1;
      }
      else
      {
         // search in right half of a1
         left = i + 1;
      }
   }
}

/**
 * Partition the input so that
 * a1[0:*out1) and a2[0:*out2) contain the smallest k elements
 */
void kth_element(
   HYPRE_Int *out1, HYPRE_Int *out2,
   HYPRE_Int *a1, HYPRE_Int *a2, HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
{
   // either of the inputs is empty
   if (n1 == 0)
   {
      *out1 = 0; *out2 = k;
      return;
   }
   if (n2 == 0)
   {
      *out1 = k; *out2 = 0;
      return;
   }
   if (k >= n1 + n2)
   {
      *out1 = n1; *out2 = n2;
      return;
   }

   // one is greater than the other
   if (k < n1 && a1[k] <= a2[0])
   {
      *out1 = k; *out2 = 0;
      return;
   }
   if (k - n1 >= 0 && a2[k - n1] >= a1[n1 - 1])
   {
      *out1 = n1; *out2 = k - n1;
      return;
   }
   if (k < n2 && a2[k] <= a1[0])
   {
      *out1 = 0; *out2 = k;
      return;
   }
   if (k - n2 >= 0 && a1[k - n2] >= a2[n2 - 1])
   {
      *out1 = k - n2; *out2 = n2;
      return;
   }
   // now k > 0

   // faster to do binary search on the shorter sequence
   if (n1 > n2)
   {
      SWAP(HYPRE_Int, n1, n2);
      SWAP(HYPRE_Int *, a1, a2);
      SWAP(HYPRE_Int *, out1, out2);
   }

   if (k < (n1 + n2)/2)
   {
      kth_element_(out1, out2, a1, a2, 0, hypre_min(n1 - 1, k), n1, n2, k);
   }
   else
   {
      // when k is big, faster to find (n1 + n2 - k)th biggest element
      HYPRE_Int offset1 = hypre_max(k - n2, 0), offset2 = hypre_max(k - n1, 0);
      HYPRE_Int new_k = k - offset1 - offset2;

      HYPRE_Int new_n1 = hypre_min(n1 - offset1, new_k + 1);
      HYPRE_Int new_n2 = hypre_min(n2 - offset2, new_k + 1);
      kth_element_(out1, out2, a1 + offset1, a2 + offset2, 0, new_n1 - 1, new_n1, new_n2, new_k);

      *out1 += offset1;
      *out2 += offset2;
   }
#ifdef DBG_MERGE_SORT
   assert(*out1 + *out2 == k);
#endif
}

/**
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zeor-based) among the threads that participate in this merge
 *
 * It is assumed that [first1, last1) and [first2, last2) are sorted
 * without dulication.
 *
 * @return length of merged output
 */
HYPRE_Int hypre_parallel_merge_unique(
   HYPRE_Int *first1, HYPRE_Int *last1, HYPRE_Int *first2, HYPRE_Int *last2,
   HYPRE_Int *temp, HYPRE_Int *out,
   HYPRE_Int num_threads, HYPRE_Int my_thread_num,
   HYPRE_Int *prefix_sum_workspace)
{
   HYPRE_Int n1 = last1 - first1;
   HYPRE_Int n2 = last2 - first2;
   HYPRE_Int n = n1 + n2;
   HYPRE_Int n_per_thread = (n + num_threads - 1)/num_threads;
   HYPRE_Int begin_rank = hypre_min(n_per_thread*my_thread_num, n);
   HYPRE_Int end_rank = hypre_min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   int *dbg_buf = NULL;
   if (my_thread_num == 0)
   {
      dbg_buf = new int[n];
      std::merge(first1, last1, first2, last2, dbg_buf);
   }
#endif

   HYPRE_Int begin1, begin2, end1, end2;
   kth_element(&begin1, &begin2, first1, first2, n1, n2, begin_rank);
   kth_element(&end1, &end2, first1, first2, n1, n2, end_rank);

   while (begin1 > end1 && begin1 > 0 && begin2 < n2 && first1[begin1 - 1] == first2[begin2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      begin1--; begin2++; 
   }
   while (begin2 > end2 && end1 > 0 && end2 < n2 && first1[end1 - 1] == first2[end2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      end1--; end2++;
   }

#ifdef DBG_MERGE_SORT
   assert(begin1 <= end1);
   assert(begin2 <= end2);
#endif

   HYPRE_Int out_len = hypre_merge_unique(
      first1 + begin1, first1 + end1,
      first2 + begin2, first2 + end2,
      temp + begin1 + begin2) - (temp + begin1 + begin2);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(temp + begin1 + begin2, temp + begin1 + begin2 + out_len));
   assert(std::adjacent_find(temp + begin1 + begin2, temp + begin1 + begin2 + out_len) == temp + begin1 + begin2 + out_len);
#endif

   prefix_sum_workspace[my_thread_num] = out_len;

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   HYPRE_Int i;
   if (0 == my_thread_num)
   {
      // take care of duplicates at boundary
      HYPRE_Int prev = temp[out_len - 1];
      HYPRE_Int t;
      for (t = 1; t < num_threads; t++)
      {
         HYPRE_Int begin_rank = hypre_min(n_per_thread*t, n);
         HYPRE_Int end_rank = begin_rank + prefix_sum_workspace[t];
         for (i = begin_rank; i < end_rank && temp[i] == prev; i++);
         prefix_sum_workspace[t] -= i - begin_rank;
         prefix_sum_workspace[t] += prefix_sum_workspace[t - 1];

         if (prefix_sum_workspace[t] > 0)
         {
            prev = temp[end_rank - 1];
         }
      }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   HYPRE_Int out_begin = my_thread_num == 0 ? 0 : prefix_sum_workspace[my_thread_num - 1];
   HYPRE_Int out_end = prefix_sum_workspace[my_thread_num];

   HYPRE_Int num_duplicates = out_len - (out_end - out_begin);
   begin_rank += num_duplicates;

   for (i = 0; i < out_end - out_begin; i++)
   {
      out[i + out_begin] = temp[i + begin_rank];
   }

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(out + out_begin, out + out_end));
   assert(std::adjacent_find(out + out_begin, out + out_end) == out + out_end);

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (0 == my_thread_num)
   {
      assert(std::is_sorted(out, out + prefix_sum_workspace[num_threads - 1]));
      assert(std::adjacent_find(out, out + prefix_sum_workspace[num_threads - 1]) == out + prefix_sum_workspace[num_threads - 1]);
      int ref_len = std::unique(dbg_buf, dbg_buf + n) - dbg_buf;
      assert(ref_len == prefix_sum_workspace[num_threads - 1]);
      assert(std::equal(out, out + prefix_sum_workspace[num_threads - 1], dbg_buf));

      delete[] dbg_buf;
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
#endif

   return prefix_sum_workspace[num_threads - 1];
}

/**
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zeor-based) among the threads that participate in this merge
 */
void hypre_parallel_merge(
   HYPRE_Int *first1, HYPRE_Int *last1, HYPRE_Int *first2, HYPRE_Int *last2,
   HYPRE_Int *out,
   HYPRE_Int num_threads, HYPRE_Int my_thread_num)
{
   HYPRE_Int n1 = last1 - first1;
   HYPRE_Int n2 = last2 - first2;
   HYPRE_Int n = n1 + n2;
   HYPRE_Int n_per_thread = (n + num_threads - 1)/num_threads;
   HYPRE_Int begin_rank = hypre_min(n_per_thread*my_thread_num, n);
   HYPRE_Int end_rank = hypre_min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(first1, last1));
   assert(std::is_sorted(first2, last2));
#endif

   HYPRE_Int begin1, begin2, end1, end2;
   kth_element(&begin1, &begin2, first1, first2, n1, n2, begin_rank);
   kth_element(&end1, &end2, first1, first2, n1, n2, end_rank);

   while (begin1 > end1 && begin1 > 0 && begin2 < n2 && first1[begin1 - 1] == first2[begin2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      begin1--; begin2++; 
   }
   while (begin2 > end2 && end1 > 0 && end2 < n2 && first1[end1 - 1] == first2[end2])
   {
#ifdef DBG_MERGE_SORT
      printf("%s:%d\n", __FILE__, __LINE__);
#endif
      end1--; end2++;
   }

#ifdef DBG_MERGE_SORT
   assert(begin1 <= end1);
   assert(begin2 <= end2);
#endif

   hypre_merge(
      first1 + begin1, first1 + end1,
      first2 + begin2, first2 + end2,
      out + begin1 + begin2);

#ifdef DBG_MERGE_SORT
   assert(std::is_sorted(out + begin1 + begin2, out + end1 + end2));
#endif
}

/**
 * @params in contents can change
 */
HYPRE_Int hypre_merge_sort_unique2(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **out)
{
   if (0 == len) return 0;

#ifdef DBG_MERGE_SORT
   int *dbg_buf = new int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   HYPRE_Int thread_private_len[hypre_NumThreads()];
   HYPRE_Int out_len = 0;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
      HYPRE_Int num_threads = hypre_NumActiveThreads();
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      // thread-private sort
      HYPRE_Int i_per_thread = (len + num_threads - 1)/num_threads;
      HYPRE_Int i_begin = hypre_min(i_per_thread*my_thread_num, len);
      HYPRE_Int i_end = hypre_min(i_begin + i_per_thread, len);

      qsort0(in, i_begin, i_end - 1);

      if (1 == num_threads)
      {
         // when there's only one thread, we need to eliminate duplicates separately
         if (len)
         {
            out_len = 1;

            HYPRE_Int i;
            for (i = 1; i < len; i++)
            {
               if (in[i] != in[i - 1]) in[out_len++] = in[i];
            }
         }

         *out = in;
      }
      else
      {
         // merge sorted sequences
         HYPRE_Int in_group_size;
         HYPRE_Int *in_buf = in;
         HYPRE_Int *out_buf = temp;
         for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

            // merge 2 in-groups into 1 out-group
            HYPRE_Int out_group_size = in_group_size*2;
            HYPRE_Int group_leader = my_thread_num/out_group_size*out_group_size;
            HYPRE_Int group_sub_leader = hypre_min(group_leader + in_group_size, num_threads - 1);
            HYPRE_Int id_in_group = my_thread_num%out_group_size;
            HYPRE_Int num_threads_in_group =
               hypre_min(group_leader + out_group_size, num_threads) - group_leader;

            HYPRE_Int in_group1_begin = hypre_min(i_per_thread*group_leader, len);
            HYPRE_Int in_group1_end = hypre_min(in_group1_begin + i_per_thread*in_group_size, len);

            HYPRE_Int in_group2_begin = hypre_min(in_group1_begin + i_per_thread*in_group_size, len);
            HYPRE_Int in_group2_end = hypre_min(in_group2_begin + i_per_thread*in_group_size, len);

            if (out_group_size < num_threads)
            {
               hypre_parallel_merge(
                  in_buf + in_group1_begin, in_buf + in_group1_end,
                  in_buf + in_group2_begin, in_buf + in_group2_end,
                  out_buf + in_group1_begin,
                  num_threads_in_group,
                  id_in_group);
            }
            else
            {
               HYPRE_Int len = hypre_parallel_merge_unique(
                  in_buf + in_group1_begin, in_buf + in_group1_end,
                  in_buf + in_group2_begin, in_buf + in_group2_end,
                  out_buf + in_group1_begin,
                  in_buf + in_group1_begin,
                  num_threads_in_group,
                  id_in_group,
                  thread_private_len + group_leader);

               if (0 == my_thread_num) out_len = len;
            }

            HYPRE_Int *temp = in_buf;
            in_buf = out_buf;
            out_buf = temp;
         }

         *out = out_buf;
      }
   } /* omp parallel */

#ifdef DBG_MERGE_SORT
   int ref_len = std::unique(dbg_buf, dbg_buf + len) - dbg_buf;
   assert(ref_len == out_len);
   assert(std::equal(*out, *out + out_len, dbg_buf));

   delete[] dbg_buf;
#endif

   return out_len;
}

HYPRE_Int hypre_merge_sort_unique(HYPRE_Int *in, HYPRE_Int *out, HYPRE_Int len)
{
   HYPRE_Int *out_buf;
   HYPRE_Int out_len = hypre_merge_sort_unique2(in, out, len, &out_buf);
   if (out_buf != out)
   {
      HYPRE_Int i;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for
#endif
      for (i = 0; i < out_len; i++)
      {
         out[i] = in[i];
      }
   }
   return out_len;
}

void hypre_merge_sort(HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **out)
{
   if (0 == len) return;

#ifdef DBG_MERGE_SORT
   int *dbg_buf = new int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   HYPRE_Int thread_private_len[hypre_NumThreads()];
   HYPRE_Int out_len = 0;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
      HYPRE_Int num_threads = hypre_NumActiveThreads();
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      // thread-private sort
      HYPRE_Int i_per_thread = (len + num_threads - 1)/num_threads;
      HYPRE_Int i_begin = hypre_min(i_per_thread*my_thread_num, len);
      HYPRE_Int i_end = hypre_min(i_begin + i_per_thread, len);

      qsort0(in, i_begin, i_end - 1);

      // merge sorted sequences
      HYPRE_Int in_group_size;
      HYPRE_Int *in_buf = in;
      HYPRE_Int *out_buf = temp;
      for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

         // merge 2 in-groups into 1 out-group
         HYPRE_Int out_group_size = in_group_size*2;
         HYPRE_Int group_leader = my_thread_num/out_group_size*out_group_size;
         HYPRE_Int group_sub_leader = hypre_min(group_leader + in_group_size, num_threads - 1);
         HYPRE_Int id_in_group = my_thread_num%out_group_size;
         HYPRE_Int num_threads_in_group =
            hypre_min(group_leader + out_group_size, num_threads) - group_leader;

         HYPRE_Int in_group1_begin = hypre_min(i_per_thread*group_leader, len);
         HYPRE_Int in_group1_end = hypre_min(in_group1_begin + i_per_thread*in_group_size, len);

         HYPRE_Int in_group2_begin = hypre_min(in_group1_begin + i_per_thread*in_group_size, len);
         HYPRE_Int in_group2_end = hypre_min(in_group2_begin + i_per_thread*in_group_size, len);

         hypre_parallel_merge(
            in_buf + in_group1_begin, in_buf + in_group1_end,
            in_buf + in_group2_begin, in_buf + in_group2_end,
            out_buf + in_group1_begin,
            num_threads_in_group,
            id_in_group);

         HYPRE_Int *temp = in_buf;
         in_buf = out_buf;
         out_buf = temp;
      }

      *out = in_buf;
   } /* omp parallel */

#ifdef DBG_MERGE_SORT
   assert(std::equal(*out, *out + len, dbg_buf));

   delete[] dbg_buf;
#endif
}

/* vim: set tabstop=8 softtabstop=3 sw=3 expandtab: */
