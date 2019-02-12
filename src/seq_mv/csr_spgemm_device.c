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

hypre_DeviceCSRSparseOpts   *hypre_device_sparse_opts = NULL;
hypre_DeviceCSRSparseHandle *hypre_device_sparse_handle = NULL;

HYPRE_Int
hypreDevice_CSRSpGemm(HYPRE_Int   m,        HYPRE_Int   k,        HYPRE_Int       n,
                      HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_a,
                      HYPRE_Int  *d_ib,     HYPRE_Int  *d_jb,     HYPRE_Complex  *d_b,
                      HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out,
                      HYPRE_Int  *nnzC)
{
   HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc,
                                       hypre_device_sparse_opts, hypre_device_sparse_handle);

   HYPRE_Int rownnz_exact;
   /* a binary array to indicate if row nnz counting is failed for a row */
   HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf,
                                         hypre_device_sparse_opts, hypre_device_sparse_handle);

   /* row nnz is exact if no row failed */
   rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf) == 0;

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact,
                                             d_ic_out, d_jc_out, d_c_out, nnzC,
                                             hypre_device_sparse_opts, hypre_device_sparse_handle);

   hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA */

