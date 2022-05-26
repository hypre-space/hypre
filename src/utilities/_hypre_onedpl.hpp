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



// Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a;
// });
template <typename T, typename Predicate> struct replace_if_fun
{
public:
   typedef T result_of;
   replace_if_fun(Predicate _pred, T _new_value)
      : pred(_pred), new_value(_new_value) {}

   template <typename _T1, typename _T2> T operator()(_T1 &&a, _T2 &&s) const
   {
      return pred(s) ? new_value : a;
   }

private:
   Predicate pred;
   const T new_value;
};

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

// Functor evaluates second element of tied sequence with predicate.
// Used by: copy_if, remove_copy_if, stable_partition_copy
// Lambda:
template <typename Predicate> struct predicate_key_fun
{
   typedef bool result_of;
   predicate_key_fun(Predicate _pred) : pred(_pred) {}

   template <typename _T1> result_of operator()(_T1 &&a) const
   {
      return pred(std::get<1>(a));
   }

private:
   Predicate pred;
};

// Used by: remove_if
template <typename Predicate> struct negate_predicate_key_fun
{
   typedef bool result_of;
   negate_predicate_key_fun(Predicate _pred) : pred(_pred) {}

   template <typename _T1> result_of operator()(_T1 &&a) const
   {
      return !pred(std::get<1>(a));
   }

private:
   Predicate pred;
};

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
   const auto n = ::std::distance(map_first, map_last);
   return HYPRE_ONEDPL_CALL( oneapi::dpl::copy, perm_begin, perm_begin + n, result);
}

#endif

#endif
