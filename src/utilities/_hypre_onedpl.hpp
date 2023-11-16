/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_ONEDPL_H
#define HYPRE_ONEDPL_H

#include "HYPRE_config.h"

#if defined(HYPRE_USING_SYCL)

/* oneAPI DPL headers */
/* NOTE: these must be included before standard C++ headers */

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

//[pred, op](Ref1 a, Ref2 s) { return pred(s) ? op(a) : a; });
template <typename T, typename Predicate, typename Operator>
struct transform_if_unary_zip_mask_fun
{
   transform_if_unary_zip_mask_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
   template <typename _T>
   void operator()(_T&& t) const
   {
      using std::get;
      if (pred(get<1>(t)))
      {
         get<2>(t) = op(get<0>(t));
      }
   }

private:
   Predicate pred;
   Operator op;
};

template <class Iter1, class Iter2, class Iter3,
          class UnaryOperation, class Pred>
Iter3 hypreSycl_transform_if(Iter1 first, Iter1 last, Iter2 mask,
                             Iter3 result, UnaryOperation unary_op, Pred pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   using T = typename std::iterator_traits<Iter1>::value_type;
   const auto n = std::distance(first, last);
   auto begin_for_each = oneapi::dpl::make_zip_iterator(first, mask, result);
   HYPRE_ONEDPL_CALL( std::for_each,
                      begin_for_each, begin_for_each + n,
                      transform_if_unary_zip_mask_fun<T, Pred, UnaryOperation>(pred, unary_op) );
   return result + n;
}

// Functor evaluates second element of tied sequence with predicate.
// Used by: copy_if
template <typename Predicate> struct predicate_key_fun
{
   predicate_key_fun(Predicate _pred) : pred(_pred) {}

   template <typename _T1> bool operator()(_T1 &&a) const
   {
      return pred(std::get<1>(a));
   }

private:
   Predicate pred;
};

// Need custom version of copy_if when predicate operates on a mask
// instead of the data being copied (natively suppored in thrust, but
// not supported in oneDPL)
template <typename Iter1, typename Iter2, typename Iter3, typename Pred>
Iter3 hypreSycl_copy_if(Iter1 first, Iter1 last, Iter2 mask,
                        Iter3 result, Pred pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   auto ret_val = HYPRE_ONEDPL_CALL( std::copy_if,
                                     oneapi::dpl::make_zip_iterator(first, mask),
                                     oneapi::dpl::make_zip_iterator(last, mask + std::distance(first, last)),
                                     oneapi::dpl::make_zip_iterator(result, oneapi::dpl::discard_iterator()),
                                     predicate_key_fun<Pred>(pred));
   return std::get<0>(ret_val.base());
}

// Similar to above, need mask version of remove_if
// NOTE: We copy the mask below because this implementation also
// remove elements from the mask in addition to the input.
template <typename Iter1, typename Iter2, typename Pred>
Iter1 hypreSycl_remove_if(Iter1 first, Iter1 last, Iter2 mask, Pred pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   using ValueType = typename std::iterator_traits<Iter2>::value_type;
   Iter2 mask_cpy = hypre_CTAlloc(ValueType, std::distance(first, last), HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(mask_cpy, mask, ValueType, std::distance(first, last), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   auto ret_val = HYPRE_ONEDPL_CALL( std::remove_if,
                                     oneapi::dpl::make_zip_iterator(first, mask_cpy),
                                     oneapi::dpl::make_zip_iterator(last, mask_cpy + std::distance(first, last)),
                                     predicate_key_fun<Pred>(pred));
   hypre_TFree(mask_cpy, HYPRE_MEMORY_DEVICE);
   return std::get<0>(ret_val.base());
}

// Similar to above, need mask version of remove_copy_if
template <typename Iter1, typename Iter2, typename Iter3, typename Pred>
Iter3 hypreSycl_remove_copy_if(Iter1 first, Iter1 last, Iter2 mask, Iter3 result, Pred pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   auto ret_val = HYPRE_ONEDPL_CALL( std::remove_copy_if,
                                     oneapi::dpl::make_zip_iterator(first, mask),
                                     oneapi::dpl::make_zip_iterator(last, mask + std::distance(first, last)),
                                     oneapi::dpl::make_zip_iterator(result, oneapi::dpl::discard_iterator()),
                                     predicate_key_fun<Pred>(pred));
   return std::get<0>(ret_val.base());
}

// Equivalent of thrust::scatter_if
template <typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter, typename Predicate>
void hypreSycl_scatter_if(InputIter1 first, InputIter1 last,
                          InputIter2 map, InputIter3 mask, OutputIter result,
                          Predicate pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same <
      typename std::iterator_traits<InputIter2>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<InputIter3>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<OutputIter>::iterator_category,
      std::random_access_iterator_tag >::value,
      "Iterators passed to algorithms must be random-access iterators.");
   hypreSycl_transform_if(first, last, mask,
                          oneapi::dpl::make_permutation_iterator(result, map),
   [ = ](auto &&v) { return v; }, [ = ](auto &&m) { return pred(m); });
}

// Equivalent of thrust::scatter
template <typename InputIter1, typename InputIter2,
          typename OutputIter>
void hypreSycl_scatter(InputIter1 first, InputIter1 last,
                             InputIter2 map, OutputIter result)
{
   static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same <
      typename std::iterator_traits<InputIter2>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<OutputIter>::iterator_category,
      std::random_access_iterator_tag >::value,
      "Iterators passed to algorithms must be random-access iterators.");
   auto perm_result =
      oneapi::dpl::make_permutation_iterator(result, map);
   HYPRE_ONEDPL_CALL( oneapi::dpl::copy, first, last, perm_result);
}

// Equivalent of thrust::gather_if
template <typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter, typename Predicate>
void hypreSycl_gather_if(InputIter1 map_first, InputIter1 map_last,
                          InputIter2 mask, InputIter3 input_first, OutputIter result,
                          Predicate pred)
{
   static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same <
      typename std::iterator_traits<InputIter2>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<InputIter3>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<OutputIter>::iterator_category,
      std::random_access_iterator_tag >::value,
      "Iterators passed to algorithms must be random-access iterators.");
   auto perm_begin =
      oneapi::dpl::make_permutation_iterator(input_first, map_first);
   const auto n = std::distance(map_first, map_last);
   hypreSycl_copy_if(perm_begin, perm_begin + n, mask, result,
   [ = ](auto &&m) { return pred(m); } );
}

// Equivalent of thrust::gather
template <typename InputIter1, typename InputIter2,
          typename OutputIter>
OutputIter hypreSycl_gather(InputIter1 map_first, InputIter1 map_last,
                            InputIter2 input_first, OutputIter result)
{
   static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same <
      typename std::iterator_traits<InputIter2>::iterator_category,
      std::random_access_iterator_tag >::value &&
      std::is_same <
      typename std::iterator_traits<OutputIter>::iterator_category,
      std::random_access_iterator_tag >::value,
      "Iterators passed to algorithms must be random-access iterators.");
   auto perm_begin =
      oneapi::dpl::make_permutation_iterator(input_first, map_first);
   const auto n = std::distance(map_first, map_last);
   return HYPRE_ONEDPL_CALL( oneapi::dpl::copy, perm_begin, perm_begin + n, result);
}

// Equivalent of thrust::sequence (with step=1)
template <class Iter, class T>
void hypreSycl_sequence(Iter first, Iter last, T init = 0)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   using DiffType = typename std::iterator_traits<Iter>::difference_type;
   HYPRE_ONEDPL_CALL( std::transform,
                      oneapi::dpl::counting_iterator<DiffType>(init),
                      oneapi::dpl::counting_iterator<DiffType>(init + std::distance(first, last)),
                      first,
                      [](auto i) { return i; });
}

// Equivalent of thrust::stable_sort_by_key
template <class Iter1, class Iter2>
void hypreSycl_stable_sort_by_key(Iter1 keys_first, Iter1 keys_last, Iter2 values_first)
{
   static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
      std::random_access_iterator_tag>::value &&
      std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
      std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
   const auto n = std::distance(keys_first, keys_last);
   auto zipped_begin = oneapi::dpl::make_zip_iterator(keys_first, values_first);
   HYPRE_ONEDPL_CALL( std::stable_sort,
                      zipped_begin,
                      zipped_begin + n,
   [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );
}

#endif

#endif
