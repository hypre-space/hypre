/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_FUNCTORS_H
#define HYPRE_FUNCTORS_H

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * hypreFunctor_DenseMatrixIdentity
 *
 * Functor for generating a dense identity matrix.
 * This assumes that the input array "a" is zeros everywhere
 *--------------------------------------------------------------------------*/

struct hypreFunctor_DenseMatrixIdentity
{
   HYPRE_Int   n_;
   HYPRE_Real *a_;

   hypreFunctor_DenseMatrixIdentity(HYPRE_Int n, HYPRE_Real *a)
   {
      n_ = n;
      a_ = a;
   }

   __host__ __device__ void operator()(HYPRE_Int i)
   {
      a_[i * n_ + i] = 1.0;
   }
};

/*--------------------------------------------------------------------------
 * hypreFunctor_ArrayStridedAccess
 *
 * Functor for performing strided data access on a templated array.
 *
 * The stride interval "s_" is used to access every "s_"-th element
 * from the source array "a_".
 *
 * It is templated to support various data types for the array.
 *--------------------------------------------------------------------------*/

template <typename T>
struct hypreFunctor_ArrayStridedAccess
{
   HYPRE_Int  s_;
   T         *a_;

   hypreFunctor_ArrayStridedAccess(HYPRE_Int s, T *a) : s_(s), a_(a) {}

   __host__ __device__ T operator()(HYPRE_Int i)
   {
      return a_[i * s_];
   }
};

/*--------------------------------------------------------------------------
 * hypreFunctor_IndexStrided
 *
 * This functor multiplies a given index "i" by a specified stride "s_".
 *
 * It is templated to support various data types for the index and stride.
 *--------------------------------------------------------------------------*/

template <typename T>
struct hypreFunctor_IndexStrided
{
   T s_;

   hypreFunctor_IndexStrided(T s) : s_(s) {}

   __host__ __device__ T operator()(const T i) const
   {
      return i * s_;
   }
};

/*--------------------------------------------------------------------------
 * hypreFunctor_IndexCycle
 *--------------------------------------------------------------------------*/

struct hypreFunctor_IndexCycle
{
   HYPRE_Int cycle_length;

   hypreFunctor_IndexCycle(HYPRE_Int _cycle_length) : cycle_length(_cycle_length) {}

   __host__ __device__ HYPRE_Int operator()(HYPRE_Int i) const
   {
      return i % cycle_length;
   }
};

#endif /* if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */
#endif /* ifndef HYPRE_FUNCTORS_H */
