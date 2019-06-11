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
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA)

HYPRE_Int
hypreDevice_CSRSpGemm(HYPRE_Int   m,        HYPRE_Int   k,        HYPRE_Int       n,
                      HYPRE_Int   nnza,     HYPRE_Int   nnzb,
                      HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_a,
                      HYPRE_Int  *d_ib,     HYPRE_Int  *d_jb,     HYPRE_Complex  *d_b,
                      HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out,
                      HYPRE_Int  *nnzC)
{
   /* trivial case */
   if (nnza == 0 || nnzb == 0)
   {
      *d_ic_out = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      *d_jc_out = hypre_CTAlloc(HYPRE_Int,     0, HYPRE_MEMORY_DEVICE);
      *d_c_out  = hypre_CTAlloc(HYPRE_Complex, 0, HYPRE_MEMORY_DEVICE);
      *nnzC = 0;

      return hypre_error_flag;
   }

   /* use CUSPARSE */
   if (hypre_device_csr_handle->use_cusparse_spgemm)
   {
      hypreDevice_CSRSpGemmCusparse(m, k, n, nnza, d_ia, d_ja, d_a, nnzb, d_ib, d_jb, d_b,
                                    nnzC, d_ic_out, d_jc_out, d_c_out);

      return hypre_error_flag;
   }

   HYPRE_Int m2 = hypre_device_csr_handle->spgemm_num_passes < 3 ? m : 2*m;
   HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m2, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);

   if (hypre_device_csr_handle->spgemm_num_passes < 3)
   {
      hypreDevice_CSRSpGemmWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                              d_ic_out, d_jc_out, d_c_out, nnzC);
   }
   else
   {
      HYPRE_Int rownnz_exact;
      /* a binary array to indicate if row nnz counting is failed for a row */
      //HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *d_rf = d_rc + m;

      hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

      /* row nnz is exact if no row failed */
      rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf) == 0;

      //hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

      hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact,
                                                d_ic_out, d_jc_out, d_c_out, nnzC);
   }

   hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA */

