/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "../seq_mv/HYPRE_seq_mv.h"
//#define DBG_MERGE_SORT
#ifdef DBG_MERGE_SORT
#include <algorithm>
#include <unordered_map>
#endif

#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

/*--------------------------------------------------------------------------
 * hypre_IntArrayMergeOrdered: merge two ordered arrays
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayMergeOrdered( hypre_IntArray *array1,
                            hypre_IntArray *array2,
                            hypre_IntArray *array3 )
{
   HYPRE_Int i = 0, j = 0, k = 0;
   const HYPRE_Int size1 = hypre_IntArraySize(array1);
   const HYPRE_Int size2 = hypre_IntArraySize(array2);

   HYPRE_MemoryLocation memory_location = hypre_IntArrayMemoryLocation(array3);

   HYPRE_Int *array1_data = hypre_IntArrayData(array1);
   HYPRE_Int *array2_data = hypre_IntArrayData(array2);
   HYPRE_Int *array3_data = hypre_TAlloc(HYPRE_Int, size1 + size2, memory_location);

   while (i < size1 && j < size2)
   {
      if (array1_data[i] > array2_data[j])
      {
         array3_data[k++] = array2_data[j++];
      }
      else if (array1_data[i] < array2_data[j])
      {
         array3_data[k++] = array1_data[i++];
      }
      else
      {
         array3_data[k++] = array1_data[i++];
         j++;
      }
   }

   while (i < size1)
   {
      array3_data[k++] = array1_data[i++];
   }

   while (j < size2)
   {
      array3_data[k++] = array2_data[j++];
   }

   array3_data = hypre_TReAlloc_v2(array3_data, HYPRE_Int, size1 + size2, HYPRE_Int, k,
                                   memory_location);

   hypre_IntArraySize(array3) = k;
   hypre_IntArrayData(array3) = array3_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_union2
 *
 * Union of two sorted (in ascending order) array arr1 and arr2 into arr3
 *
 * Assumptions:
 *              1) no duplicate entries in arr1 and arr2. But an entry is
 *                 allowed to appear in both arr1 and arr2
 *              2) arr3 should have enough space on entry
 *              3) map1 and map2 map arr1 and arr2 to arr3
 *--------------------------------------------------------------------------*/
void hypre_union2( HYPRE_Int n1,  HYPRE_BigInt *arr1,
                   HYPRE_Int n2,  HYPRE_BigInt *arr2,
                   HYPRE_Int *n3, HYPRE_BigInt *arr3,
                   HYPRE_Int *map1, HYPRE_Int *map2 )
{
   HYPRE_Int i = 0, j = 0, k = 0;
   while (i < n1 && j < n2)
   {
      if (arr1[i] < arr2[j])
      {
         if (map1) { map1[i] = k; }
         arr3[k++] = arr1[i++];
      }
      else if (arr1[i] > arr2[j])
      {
         if (map2) { map2[j] = k; }
         arr3[k++] = arr2[j++];
      }
      else /* == */
      {
         if (map1) { map1[i] = k; }
         if (map2) { map2[j] = k; }
         arr3[k++] = arr1[i++];
         j++;
      }
   }
   while (i < n1)
   {
      if (map1) { map1[i] = k; }
      arr3[k++] = arr1[i++];
   }
   while (j < n2)
   {
      if (map2) { map2[j] = k; }
      arr3[k++] = arr2[j++];
   }
   *n3 = k;
}

/*--------------------------------------------------------------------------
 * hypre_merge
 *--------------------------------------------------------------------------*/
static void hypre_merge( HYPRE_Int *first1, HYPRE_Int *last1,
                         HYPRE_Int *first2, HYPRE_Int *last2,
                         HYPRE_Int *out )
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

/*--------------------------------------------------------------------------
 * hypre_big_merge
 *--------------------------------------------------------------------------*/

static void hypre_big_merge( HYPRE_BigInt *first1, HYPRE_BigInt *last1,
                             HYPRE_BigInt *first2, HYPRE_BigInt *last2,
                             HYPRE_BigInt *out )
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

/*--------------------------------------------------------------------------
 * kth_element_
 *--------------------------------------------------------------------------*/

static void kth_element_( HYPRE_Int *out1, HYPRE_Int *out2,
                          HYPRE_Int *a1, HYPRE_Int *a2,
                          HYPRE_Int left, HYPRE_Int right,
                          HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
{
   while (1)
   {
      HYPRE_Int i = (left + right) / 2; // right < k -> i < k
      HYPRE_Int j = k - i - 1;
#ifdef DBG_MERGE_SORT
      hypre_assert(left <= right && right <= k);
      hypre_assert(i < k); // i == k implies left == right == k that can never happen
      hypre_assert(j >= 0 && j < n2);
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
      else if (a1[i] > a2[j] && j != n2 - 1 && a1[i] > a2[j + 1])
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

/*--------------------------------------------------------------------------
 * kth_element
 *
 * Partition the input so that
 * a1[0:*out1) and a2[0:*out2) contain the smallest k elements
 *--------------------------------------------------------------------------*/

static void kth_element( HYPRE_Int *out1, HYPRE_Int *out2,
                         HYPRE_Int *a1, HYPRE_Int *a2,
                         HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
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

   if (k < (n1 + n2) / 2)
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
   hypre_assert(*out1 + *out2 == k);
#endif
}

/*--------------------------------------------------------------------------
 * big_kth_element_
 *--------------------------------------------------------------------------*/

static void big_kth_element_( HYPRE_Int *out1, HYPRE_Int *out2,
                              HYPRE_BigInt *a1, HYPRE_BigInt *a2,
                              HYPRE_Int left, HYPRE_Int right,
                              HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
{
   while (1)
   {
      HYPRE_Int i = (left + right) / 2; // right < k -> i < k
      HYPRE_Int j = k - i - 1;
#ifdef DBG_MERGE_SORT
      hypre_assert(left <= right && right <= k);
      hypre_assert(i < k); // i == k implies left == right == k that can never happen
      hypre_assert(j >= 0 && j < n2);
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
      else if (a1[i] > a2[j] && j != n2 - 1 && a1[i] > a2[j + 1])
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

/*--------------------------------------------------------------------------
 * big_kth_element
 *
 * Partition the input so that
 * a1[0:*out1) and a2[0:*out2) contain the smallest k elements
 *--------------------------------------------------------------------------*/

static void big_kth_element( HYPRE_Int *out1, HYPRE_Int *out2,
                             HYPRE_BigInt *a1, HYPRE_BigInt *a2,
                             HYPRE_Int n1, HYPRE_Int n2, HYPRE_Int k)
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
      SWAP(HYPRE_BigInt *, a1, a2);
      SWAP(HYPRE_Int *, out1, out2);
   }

   if (k < (n1 + n2) / 2)
   {
      big_kth_element_(out1, out2, a1, a2, 0, hypre_min(n1 - 1, k), n1, n2, k);
   }
   else
   {
      // when k is big, faster to find (n1 + n2 - k)th biggest element
      HYPRE_Int offset1 = hypre_max(k - n2, 0), offset2 = hypre_max(k - n1, 0);
      HYPRE_Int new_k = k - offset1 - offset2;

      HYPRE_Int new_n1 = hypre_min(n1 - offset1, new_k + 1);
      HYPRE_Int new_n2 = hypre_min(n2 - offset2, new_k + 1);
      big_kth_element_(out1, out2, a1 + (HYPRE_BigInt)offset1, a2 + (HYPRE_BigInt)offset2, 0, new_n1 - 1,
                       new_n1, new_n2, new_k);

      *out1 += offset1;
      *out2 += offset2;
   }
#ifdef DBG_MERGE_SORT
   hypre_assert(*out1 + *out2 == k);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_parallel_merge
 *
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zer0-based) among the threads that
 *                      participate in this merge
 *--------------------------------------------------------------------------*/

static void hypre_parallel_merge( HYPRE_Int *first1, HYPRE_Int *last1,
                                  HYPRE_Int *first2, HYPRE_Int *last2,
                                  HYPRE_Int *out, HYPRE_Int num_threads,
                                  HYPRE_Int my_thread_num )
{
   HYPRE_Int n1 = last1 - first1;
   HYPRE_Int n2 = last2 - first2;
   HYPRE_Int n = n1 + n2;
   HYPRE_Int n_per_thread = (n + num_threads - 1) / num_threads;
   HYPRE_Int begin_rank = hypre_min(n_per_thread * my_thread_num, n);
   HYPRE_Int end_rank = hypre_min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   hypre_assert(std::is_sorted(first1, last1));
   hypre_assert(std::is_sorted(first2, last2));
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
   hypre_assert(begin1 <= end1);
   hypre_assert(begin2 <= end2);
#endif

   hypre_merge(
      first1 + begin1, first1 + end1,
      first2 + begin2, first2 + end2,
      out + begin1 + begin2);

#ifdef DBG_MERGE_SORT
   hypre_assert(std::is_sorted(out + begin1 + begin2, out + end1 + end2));
#endif
}

/*--------------------------------------------------------------------------
 * hypre_big_parallel_merge
 *
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zero-based) among the threads that
 *                      participate in this merge
 *--------------------------------------------------------------------------*/

static void hypre_big_parallel_merge(
   HYPRE_BigInt *first1, HYPRE_BigInt *last1, HYPRE_BigInt *first2, HYPRE_BigInt *last2,
   HYPRE_BigInt *out,
   HYPRE_Int num_threads, HYPRE_Int my_thread_num)
{
   HYPRE_Int n1 = (HYPRE_Int)(last1 - first1);
   HYPRE_Int n2 = (HYPRE_Int)(last2 - first2);
   HYPRE_Int n = n1 + n2;
   HYPRE_Int n_per_thread = (n + num_threads - 1) / num_threads;
   HYPRE_Int begin_rank = hypre_min(n_per_thread * my_thread_num, n);
   HYPRE_Int end_rank = hypre_min(begin_rank + n_per_thread, n);

#ifdef DBG_MERGE_SORT
   hypre_assert(std::is_sorted(first1, last1));
   hypre_assert(std::is_sorted(first2, last2));
#endif

   HYPRE_Int begin1, begin2, end1, end2;
   big_kth_element(&begin1, &begin2, first1, first2, n1, n2, begin_rank);
   big_kth_element(&end1, &end2, first1, first2, n1, n2, end_rank);

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
   hypre_assert(begin1 <= end1);
   hypre_assert(begin2 <= end2);
#endif

   hypre_big_merge(
      first1 + (HYPRE_BigInt)begin1, first1 + (HYPRE_BigInt)end1,
      first2 + (HYPRE_BigInt)begin2, first2 + (HYPRE_BigInt)end2,
      out + (HYPRE_BigInt)(begin1 + begin2));

#ifdef DBG_MERGE_SORT
   hypre_assert(std::is_sorted(out + begin1 + begin2, out + end1 + end2));
#endif
}

/*--------------------------------------------------------------------------
 * hypre_merge_sort
 *--------------------------------------------------------------------------*/

void hypre_merge_sort( HYPRE_Int *in, HYPRE_Int *temp, HYPRE_Int len, HYPRE_Int **out )
{
   if (0 == len) { return; }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

#ifdef DBG_MERGE_SORT
   HYPRE_Int *dbg_buf = new HYPRE_Int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   // HYPRE_Int thread_private_len[hypre_NumThreads()];
   // HYPRE_Int out_len = 0;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int num_threads = hypre_NumActiveThreads();
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      // thread-private sort
      HYPRE_Int i_per_thread = (len + num_threads - 1) / num_threads;
      HYPRE_Int i_begin = hypre_min(i_per_thread * my_thread_num, len);
      HYPRE_Int i_end = hypre_min(i_begin + i_per_thread, len);

      hypre_qsort0(in, i_begin, i_end - 1);

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
         HYPRE_Int out_group_size = in_group_size * 2;
         HYPRE_Int group_leader = my_thread_num / out_group_size * out_group_size;
         // HYPRE_Int group_sub_leader = hypre_min(group_leader + in_group_size, num_threads - 1);
         HYPRE_Int id_in_group = my_thread_num % out_group_size;
         HYPRE_Int num_threads_in_group =
            hypre_min(group_leader + out_group_size, num_threads) - group_leader;

         HYPRE_Int in_group1_begin = hypre_min(i_per_thread * group_leader, len);
         HYPRE_Int in_group1_end = hypre_min(in_group1_begin + i_per_thread * in_group_size, len);

         HYPRE_Int in_group2_begin = hypre_min(in_group1_begin + i_per_thread * in_group_size, len);
         HYPRE_Int in_group2_end = hypre_min(in_group2_begin + i_per_thread * in_group_size, len);

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
   hypre_assert(std::equal(*out, *out + len, dbg_buf));

   delete[] dbg_buf;
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif
}

/*--------------------------------------------------------------------------
 * hypre_sort_and_create_inverse_map
 *
 * Sort array "in" with length len and put result in array "out"
 *   "in" will be deallocated unless in == *out
 *   inverse_map is an inverse hash table s.t.
 *      inverse_map[i] = j iff (*out)[j] = i
 *--------------------------------------------------------------------------*/

void hypre_sort_and_create_inverse_map(HYPRE_Int *in, HYPRE_Int len, HYPRE_Int **out,
                                       hypre_UnorderedIntMap *inverse_map)
{
   if (len == 0)
   {
      return;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int *temp = hypre_TAlloc(HYPRE_Int,  len, HYPRE_MEMORY_HOST);
   hypre_merge_sort(in, temp, len, out);
   hypre_UnorderedIntMapCreate(inverse_map, 2 * len, 16 * hypre_NumThreads());
   HYPRE_Int i;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < len; i++)
   {
      HYPRE_Int old = hypre_UnorderedIntMapPutIfAbsent(inverse_map, (*out)[i], i);
      hypre_assert(old == HYPRE_HOPSCOTCH_HASH_EMPTY);
#ifdef DBG_MERGE_SORT
      if (hypre_UnorderedIntMapGet(inverse_map, (*out)[i]) != i)
      {
         fprintf(stderr, "%d %d\n", i, (*out)[i]);
         hypre_assert(false);
      }
#endif
   }

#ifdef DBG_MERGE_SORT
   std::unordered_map<HYPRE_Int, HYPRE_Int> inverse_map2(len);
   for (HYPRE_Int i = 0; i < len; ++i)
   {
      inverse_map2[(*out)[i]] = i;
      if (hypre_UnorderedIntMapGet(inverse_map, (*out)[i]) != i)
      {
         fprintf(stderr, "%d %d\n", i, (*out)[i]);
         hypre_assert(false);
      }
   }
   hypre_assert(hypre_UnorderedIntMapSize(inverse_map) == len);
#endif

   if (*out == in)
   {
      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_TFree(in, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif
}

/*--------------------------------------------------------------------------
 * hypre_big_merge_sort
 *--------------------------------------------------------------------------*/

void hypre_big_merge_sort(HYPRE_BigInt *in, HYPRE_BigInt *temp, HYPRE_Int len,
                          HYPRE_BigInt **out)
{
   if (0 == len) { return; }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

#ifdef DBG_MERGE_SORT
   HYPRE_Int *dbg_buf = new HYPRE_Int[len];
   std::copy(in, in + len, dbg_buf);
   std::sort(dbg_buf, dbg_buf + len);
#endif

   // HYPRE_Int thread_private_len[hypre_NumThreads()];
   // HYPRE_Int out_len = 0;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int num_threads = hypre_NumActiveThreads();
      HYPRE_Int my_thread_num = hypre_GetThreadNum();

      // thread-private sort
      HYPRE_Int i_per_thread = (len + num_threads - 1) / num_threads;
      HYPRE_Int i_begin = hypre_min(i_per_thread * my_thread_num, len);
      HYPRE_Int i_end = hypre_min(i_begin + i_per_thread, len);

      hypre_BigQsort0(in, i_begin, i_end - 1);

      // merge sorted sequences
      HYPRE_Int in_group_size;
      HYPRE_BigInt *in_buf = in;
      HYPRE_BigInt *out_buf = temp;
      for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         // merge 2 in-groups into 1 out-group
         HYPRE_Int out_group_size = in_group_size * 2;
         HYPRE_Int group_leader = my_thread_num / out_group_size * out_group_size;
         // HYPRE_Int group_sub_leader = hypre_min(group_leader + in_group_size, num_threads - 1);
         HYPRE_Int id_in_group = my_thread_num % out_group_size;
         HYPRE_Int num_threads_in_group =
            hypre_min(group_leader + out_group_size, num_threads) - group_leader;

         HYPRE_Int in_group1_begin = hypre_min(i_per_thread * group_leader, len);
         HYPRE_Int in_group1_end = hypre_min(in_group1_begin + i_per_thread * in_group_size, len);

         HYPRE_Int in_group2_begin = hypre_min(in_group1_begin + i_per_thread * in_group_size, len);
         HYPRE_Int in_group2_end = hypre_min(in_group2_begin + i_per_thread * in_group_size, len);

         hypre_big_parallel_merge(
            in_buf + (HYPRE_BigInt)in_group1_begin, in_buf + (HYPRE_BigInt)in_group1_end,
            in_buf + (HYPRE_BigInt)in_group2_begin, in_buf + (HYPRE_BigInt)in_group2_end,
            out_buf + (HYPRE_BigInt)in_group1_begin,
            num_threads_in_group,
            id_in_group);

         HYPRE_BigInt *temp = in_buf;
         in_buf = out_buf;
         out_buf = temp;
      }

      *out = in_buf;
   } /* omp parallel */

#ifdef DBG_MERGE_SORT
   hypre_assert(std::equal(*out, *out + len, dbg_buf));

   delete[] dbg_buf;
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif
}

/*--------------------------------------------------------------------------
 * hypre_big_sort_and_create_inverse_map
 *--------------------------------------------------------------------------*/

void hypre_big_sort_and_create_inverse_map(HYPRE_BigInt *in, HYPRE_Int len, HYPRE_BigInt **out,
                                           hypre_UnorderedBigIntMap *inverse_map)
{
   if (len == 0)
   {
      return;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

   HYPRE_BigInt *temp = hypre_TAlloc(HYPRE_BigInt,  len, HYPRE_MEMORY_HOST);
   hypre_big_merge_sort(in, temp, len, out);
   hypre_UnorderedBigIntMapCreate(inverse_map, 2 * len, 16 * hypre_NumThreads());
   HYPRE_Int i;
#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < len; i++)
   {
      HYPRE_Int old = hypre_UnorderedBigIntMapPutIfAbsent(inverse_map, (*out)[i], i);
      hypre_assert(old == HYPRE_HOPSCOTCH_HASH_EMPTY);
#ifdef DBG_MERGE_SORT
      if (hypre_UnorderedBigIntMapGet(inverse_map, (*out)[i]) != i)
      {
         fprintf(stderr, "%d %d\n", i, (*out)[i]);
         hypre_assert(false);
      }
#endif
   }

#ifdef DBG_MERGE_SORT
   std::unordered_map<HYPRE_Int, HYPRE_Int> inverse_map2(len);
   for (HYPRE_Int i = 0; i < len; ++i)
   {
      inverse_map2[(*out)[i]] = i;
      if (hypre_UnorderedBigIntMapGet(inverse_map, (*out)[i]) != i)
      {
         fprintf(stderr, "%d %d\n", i, (*out)[i]);
         hypre_assert(false);
      }
   }
   hypre_assert(hypre_UnorderedBigIntMapSize(inverse_map) == len);
#endif

   if (*out == in)
   {
      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_TFree(in, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif
}

/* vim: set tabstop=8 softtabstop=3 sw=3 expandtab: */
