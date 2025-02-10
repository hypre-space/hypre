/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_PREDICATES_H
#define HYPRE_PREDICATES_H

/******************************************************************************
 *
 * Header file defining predicates for thrust used throughout hypre
 *
 *****************************************************************************/

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * hyprePred_StridedAccess
 *
 * This struct defines a predicate for strided access in array-like data.
 *
 * It is used to determine if an element at a given index should be processed
 * or not, based on a specified stride. The operator() returns true when the
 * index is a multiple of the stride, indicating the element at that index
 * is part of the strided subset.
 *--------------------------------------------------------------------------*/

struct hyprePred_StridedAccess
{
   HYPRE_Int  s_;

   hyprePred_StridedAccess(HYPRE_Int s) : s_(s) {}

   __host__ __device__ HYPRE_Int operator()(const HYPRE_Int i) const
   {
      return (!(i % s_));
   }
};

#endif /* if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */
#endif /* ifndef HYPRE_PREDICATES_H */
