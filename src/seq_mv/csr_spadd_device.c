/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/
#include "seq_mv.h"
#include "csr_sparse_device.h"

#if defined(HYPRE_USING_CUDA)

/* in Matlab notation: if num_b != NULL, for each num_b[i], A(num_b[i], :) += B(i,:).
 *                                       A is ma x n and B is mb x n. len(num_b) == mb.
 *                                       NOTE: all numbers in num_b must be in [0,...,ma-1]
 *                     if num_b == NULL, C = A + B. ma == mb
 *                     C is ma x n in both cases
 */
HYPRE_Int
hypreDevice_CSRSpAdd(HYPRE_Int  ma,       HYPRE_Int   mb,        HYPRE_Int   n,
                     HYPRE_Int  nnzA,     HYPRE_Int   nnzB,
                     HYPRE_Int *d_ia,     HYPRE_Int  *d_ja,      HYPRE_Complex *d_aa,
                     HYPRE_Int *d_ib,     HYPRE_Int  *d_jb,      HYPRE_Complex *d_ab,  HYPRE_Int      *d_num_b,
                     HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,  HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out)
{
   hypre_double tt = 0.0, tm;

   HYPRE_Int do_timing = hypre_device_sparse_opts->do_timing;

   if (do_timing)
   {
      cudaThreadSynchronize();
      tm = tt = time_getWallclockSeconds();
   }

   /* expand */
   HYPRE_Int nnzT = nnzA + nnzB, nnzC;
   HYPRE_Int *d_it, *d_jt, *d_pm, *d_it_cp, *d_jt_cp, *d_ic, *d_jc;
   HYPRE_Complex *d_at, *d_at_cp, *d_ac;

   d_it = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   d_jt = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   d_at = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);

   thrust::device_ptr<HYPRE_Int>     d_it_ptr = thrust::device_pointer_cast(d_it);
   thrust::device_ptr<HYPRE_Int>     d_jt_ptr = thrust::device_pointer_cast(d_jt);
   thrust::device_ptr<HYPRE_Complex> d_at_ptr = thrust::device_pointer_cast(d_at);

   /* expansion */
   hypre_TMemcpy(d_jt,        d_ja, HYPRE_Int,     nnzA,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jt + nnzA, d_jb, HYPRE_Int,     nnzB,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at,        d_aa, HYPRE_Complex, nnzA,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at + nnzA, d_ab, HYPRE_Complex, nnzB,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(ma, d_ia, d_it);
   if (d_num_b)
   {
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(mb, d_ib, d_num_b, d_it + nnzA);
   }
   else
   {
      hypre_assert(ma == mb);
      hypreDevice_CsrRowPtrsToIndices_v2(mb, d_ib, d_it + nnzA);
   }

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->spadd_expansion_time = tm - tm_old;
   }

   /* permutation vector */
   d_pm = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   thrust::device_ptr<HYPRE_Int>d_pm_ptr = thrust::device_pointer_cast(d_pm);

   /* make copy of (it, jt, at), since gather cannot be done in-place */
   d_it_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   d_jt_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   d_at_cp = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);

   thrust::device_ptr<HYPRE_Int>     d_it_cp_ptr = thrust::device_pointer_cast(d_it_cp);
   thrust::device_ptr<HYPRE_Int>     d_jt_cp_ptr = thrust::device_pointer_cast(d_jt_cp);
   thrust::device_ptr<HYPRE_Complex> d_at_cp_ptr = thrust::device_pointer_cast(d_at_cp);

   hypre_TMemcpy(d_it_cp, d_it, HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jt_cp, d_jt, HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at_cp, d_at, HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort: lexicographical order (row, col) */
   thrust::sequence(d_pm_ptr, d_pm_ptr + nnzT);
   thrust::stable_sort_by_key(d_jt_ptr, d_jt_ptr + nnzT, d_pm_ptr);
   thrust::gather(d_pm_ptr, d_pm_ptr + nnzT, d_it_cp_ptr, d_it_ptr);

   thrust::stable_sort_by_key(d_it_ptr, d_it_ptr + nnzT, d_pm_ptr);
   thrust::gather(d_pm_ptr, d_pm_ptr + nnzT, d_jt_cp_ptr, d_jt_ptr);
   thrust::gather(d_pm_ptr, d_pm_ptr + nnzT, d_at_cp_ptr, d_at_ptr);

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->spadd_sorting_time = tm - tm_old;
   }

   /* compress */
   typedef thrust::tuple< thrust::device_ptr<HYPRE_Int>, thrust::device_ptr<HYPRE_Int> > IteratorTuple;
   typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

   thrust::pair< ZipIterator, thrust::device_ptr<HYPRE_Complex> > new_end =
      thrust::reduce_by_key (
      thrust::make_zip_iterator(thrust::make_tuple(d_it_ptr       , d_jt_ptr       )),
      thrust::make_zip_iterator(thrust::make_tuple(d_it_ptr + nnzT, d_jt_ptr + nnzT)),
      d_at_ptr,
      thrust::make_zip_iterator(thrust::make_tuple(d_it_cp_ptr,     d_jt_cp_ptr)),
      d_at_cp_ptr,
      thrust::equal_to< thrust::tuple<HYPRE_Int, HYPRE_Int> >()
   );

   /* returns end: so nnz = end - start */
   nnzC = new_end.second - d_at_cp_ptr;

   /* allocate final C */
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_ac = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_jc, d_jt_cp, HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_ac, d_at_cp, HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->spadd_compression_time = tm - tm_old;
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(ma, nnzC, d_it_cp);

#if DEBUG_MODE
   HYPRE_Int tmp_nnzC;
   hypre_TMemcpy(&tmp_nnzC, &d_ic[ma], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   assert(nnzC == tmp_nnzC);
#endif

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->spadd_convert_ptr_time = tm - tm_old;
   }

   hypre_TFree(d_it,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_pm,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_it_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at_cp, HYPRE_MEMORY_DEVICE);

   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

   if (do_timing)
   {
      cudaThreadSynchronize();
      tt = time_getWallclockSeconds() - tt;
      hypre_device_sparse_handle->spadd_time = tt;
   }

   return hypre_error_flag;
}

#endif

