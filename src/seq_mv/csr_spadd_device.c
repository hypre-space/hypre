/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

#if defined(HYPRE_USING_CUDA)

/* in Matlab notation: if num_b != NULL, C = A, and
 *                                       for each num_b[i], A(num_b[i], :) += B(i,:).
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
   /* trivial case */
   if (nnzA == 0 && nnzB == 0)
   {
      *d_ic_out = hypre_CTAlloc(HYPRE_Int, ma + 1, HYPRE_MEMORY_DEVICE);
      *d_jc_out = hypre_CTAlloc(HYPRE_Int,      0, HYPRE_MEMORY_DEVICE);
      *d_ac_out = hypre_CTAlloc(HYPRE_Complex,  0, HYPRE_MEMORY_DEVICE);
      *nnzC_out = 0;

      return hypre_error_flag;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPADD] -= hypre_MPI_Wtime();
#endif

   /* expand */
   HYPRE_Int nnzT = nnzA + nnzB, nnzC;
   HYPRE_Int *d_it, *d_jt, *d_pm, *d_it_cp, *d_jt_cp, *d_ic, *d_jc;
   HYPRE_Complex *d_at, *d_at_cp, *d_ac;

   HYPRE_Int align = 32;
   HYPRE_Int nnzT2 = (nnzT + align - 1) / align * align;
   char *work_mem = hypre_TAlloc(char, (5*sizeof(HYPRE_Int)+2*sizeof(HYPRE_Complex))*nnzT2, HYPRE_MEMORY_DEVICE);
   char *work_mem_saved = work_mem;

   //d_it = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   //d_jt = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   //d_at = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);
   d_it = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_jt = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_at = (HYPRE_Complex *) work_mem;
   work_mem += sizeof(HYPRE_Complex) * nnzT2;

   /* expansion */
   hypre_TMemcpy(d_jt,        d_ja, HYPRE_Int,     nnzA,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jt + nnzA, d_jb, HYPRE_Int,     nnzB,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at,        d_aa, HYPRE_Complex, nnzA,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at + nnzA, d_ab, HYPRE_Complex, nnzB,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(ma, d_ia, d_it);
   if (d_num_b || mb <= 0)
   {
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(mb, d_ib, d_num_b, d_it + nnzA);
   }
   else
   {
      hypre_assert(ma == mb);
      hypreDevice_CsrRowPtrsToIndices_v2(mb, d_ib, d_it + nnzA);
   }

   /* permutation vector */
   //d_pm = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   d_pm = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;

   /* make copy of (it, jt, at), since gather cannot be done in-place */
   //d_it_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   //d_jt_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   //d_at_cp = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);
   d_it_cp = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_jt_cp = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_at_cp = (HYPRE_Complex *) work_mem;
   work_mem += sizeof(HYPRE_Complex) * nnzT2;

   hypre_assert(work_mem - work_mem_saved == (5*sizeof(HYPRE_Int)+2*sizeof(HYPRE_Complex))*nnzT2);

   hypre_TMemcpy(d_it_cp, d_it, HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jt_cp, d_jt, HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_at_cp, d_at, HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort: lexicographical order (row, col) */
   HYPRE_THRUST_CALL(sequence, d_pm, d_pm + nnzT);
   HYPRE_THRUST_CALL(stable_sort_by_key, d_jt, d_jt + nnzT, d_pm);
   HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzT, d_it_cp, d_it);

   HYPRE_THRUST_CALL(stable_sort_by_key, d_it, d_it + nnzT, d_pm);
   HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzT, d_jt_cp, d_jt);
   HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzT, d_at_cp, d_at);

   /* compress */
   typedef thrust::tuple< thrust::device_ptr<HYPRE_Int>, thrust::device_ptr<HYPRE_Int> > IteratorTuple;
   typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

   thrust::pair< ZipIterator, thrust::device_ptr<HYPRE_Complex> > new_end =
      HYPRE_THRUST_CALL(reduce_by_key,
      thrust::make_zip_iterator(thrust::make_tuple(d_it       , d_jt       )),
      thrust::make_zip_iterator(thrust::make_tuple(d_it + nnzT, d_jt + nnzT)),
      d_at,
      thrust::make_zip_iterator(thrust::make_tuple(d_it_cp,     d_jt_cp)),
      d_at_cp,
      thrust::equal_to< thrust::tuple<HYPRE_Int, HYPRE_Int> >()
   );

   /* returns end: so nnz = end - start */
   nnzC = new_end.second - thrust::device_pointer_cast(d_at_cp);

   /* allocate final C */
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_ac = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_jc, d_jt_cp, HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_ac, d_at_cp, HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(ma, nnzC, d_it_cp);

#if DEBUG_MODE
   HYPRE_Int tmp_nnzC;
   hypre_TMemcpy(&tmp_nnzC, &d_ic[ma], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   assert(nnzC == tmp_nnzC);
#endif

   /*
   hypre_TFree(d_it,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_pm,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_it_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at_cp, HYPRE_MEMORY_DEVICE);
   */
   hypre_TFree(work_mem_saved, HYPRE_MEMORY_DEVICE);

   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPADD] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif

