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

HYPRE_Int
hypreDevice_CSRSpTrans(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                       HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                       HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                       HYPRE_Int   want_data)
{
   HYPRE_Int do_timing = hypre_device_sparse_opts->do_timing;
   hypre_double tt = 0.0, tm;
   HYPRE_Int *d_jt, *d_it, *d_pm, *d_ic, *d_jc;
   HYPRE_Complex *d_ac = NULL;

   /* allocate C */
   d_jc = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   }

   /* permutation vector */
   d_pm = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);

   if (do_timing)
   {
      cudaThreadSynchronize();
      tm = tt = time_getWallclockSeconds();
   }

   /* expansion: A's row idx */
   d_it = hypreDevice_CsrRowPtrsToIndices(m, nnzA, d_ia);

   /* a copy of col idx of A */
   d_jt = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jt, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->sptrans_expansion_time = tm - tm_old;
   }

   /* sort: by col */
   thrust::sequence(thrust::device, d_pm, d_pm + nnzA);
   thrust::stable_sort_by_key(thrust::device, d_jt, d_jt + nnzA, d_pm);
   thrust::gather(thrust::device, d_pm, d_pm + nnzA, d_it, d_jc);
   if (want_data)
   {
      thrust::gather(thrust::device, d_pm, d_pm + nnzA, d_aa, d_ac);
   }

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->sptrans_sorting_time = tm - tm_old;
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(n, nnzA, d_jt);

#if DEBUG_MODE
   HYPRE_Int nnzC;
   hypre_TMemcpy(&nnzC, &d_ic[n], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_assert(nnzC == nnzA);
#endif

   if (do_timing)
   {
      cudaThreadSynchronize();
      hypre_double tm_old = tm;
      tm = time_getWallclockSeconds();
      hypre_device_sparse_handle->sptrans_rowptr_time = tm - tm_old;
   }

   hypre_TFree(d_jt, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_it, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_pm, HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

   if (do_timing)
   {
      cudaThreadSynchronize();
      tt = time_getWallclockSeconds() - tt;
      hypre_device_sparse_handle->sptrans_time = tt;
   }

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA */

