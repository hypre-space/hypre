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

#if defined(HYPRE_USING_CUDA)

/* size of work = want_data ? 3*nnzA : nnzA */
HYPRE_Int
hypreDevice_CSRSpTrans_v2(HYPRE_Int   m,    HYPRE_Int  n,    HYPRE_Int      nnzA,
                          HYPRE_Int  *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa,
                          HYPRE_Int  *d_ic, HYPRE_Int *d_jc, HYPRE_Complex *d_ac,
                          HYPRE_Int   want_data, HYPRE_Int *work)
{
   /* trivial case */
   if (nnzA == 0)
   {
      hypre_Memset(d_ic, 0, sizeof(HYPRE_Int)*(n+1), HYPRE_MEMORY_DEVICE);

      return hypre_error_flag;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int *d_jt, *d_it, *d_pm;

   /* d_jt: a copy of col idx of A, d_ja */
   d_jt = work;
   hypre_TMemcpy(d_jt, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* d_it: row idx of A, expanded from row ptrs */
   d_it = want_data ? d_jt + nnzA : d_jc;
   hypreDevice_CsrRowPtrsToIndices_v2(m, d_ia, d_it);

   if (want_data)
   {
      /* permutation vector */
      //d_pm = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
      d_pm = d_it + nnzA;
      thrust::sequence(thrust::device, d_pm, d_pm + nnzA);
      /* sort: by col */
      thrust::stable_sort_by_key(thrust::device, d_jt, d_jt + nnzA, d_pm);
      thrust::gather(thrust::device, d_pm, d_pm + nnzA, d_it, d_jc);
      thrust::gather(thrust::device, d_pm, d_pm + nnzA, d_aa, d_ac);
   }
   else
   {
      /* sort: by col */
      thrust::stable_sort_by_key(thrust::device, d_jt, d_jt + nnzA, d_jc);
   }

   /* convert into ic: col idx --> col ptrs */
   hypreDevice_CsrRowIndicesToPtrs_v2(n, nnzA, d_jt, d_ic);

#if DEBUG_MODE
   HYPRE_Int nnzC;
   hypre_TMemcpy(&nnzC, &d_ic[n], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_assert(nnzC == nnzA);
#endif

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpTrans(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                       HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                       HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                       HYPRE_Int   want_data)
{
   HYPRE_Int *d_ic, *d_jc, ierr;
   HYPRE_Complex *d_ac;

   d_ic = hypre_TAlloc(HYPRE_Int, n+1,  HYPRE_MEMORY_DEVICE);
   d_jc = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   }

   size_t mem_size = want_data ? 3*nnzA : nnzA;
   HYPRE_Int *work = hypre_TAlloc(HYPRE_Int, mem_size, HYPRE_MEMORY_DEVICE);

   ierr = hypreDevice_CSRSpTrans_v2(m, n, nnzA, d_ia, d_ja, d_aa, d_ic, d_jc, d_ac,
                                    want_data, work);

   hypre_TFree(work, HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

   return ierr;
}

#endif /* HYPRE_USING_CUDA */

