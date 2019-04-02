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

hypre_DeviceCSRSparseHandle *
hypre_DeviceCSRSparseHandleCreate(hypre_DeviceCSRSparseOpts *opts)
{
   hypre_DeviceCSRSparseHandle *handle = hypre_CTAlloc(hypre_DeviceCSRSparseHandle, 1, HYPRE_MEMORY_HOST);
   if (opts->rownnz_estimate_method == 3)
   {
      /* Create pseudo-random number generator */
      CURAND_CALL(curandCreateGenerator(&handle->gen, CURAND_RNG_PSEUDO_DEFAULT));
      /* Set seed */
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed(handle->gen, 1234ULL));
   }

   return handle;
}

HYPRE_Int
hypre_DeviceCSRSparseHandleDestroy(hypre_DeviceCSRSparseHandle *handle)
{
   if (handle->gen)
   {
      CURAND_CALL(curandDestroyGenerator(handle->gen));
   }
   hypre_TFree(handle, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

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
   if (hypre_device_sparse_opts->use_cusparse_spgemm)
   {
      hypreDevice_CSRSpGemmCusparse(m, k, n, nnza, d_ia, d_ja, d_a, nnzb, d_ib, d_jb, d_b,
                                    nnzC, d_ic_out, d_jc_out, d_c_out,
                                    hypre_device_sparse_opts, hypre_device_sparse_handle);

      return hypre_error_flag;
   }

   HYPRE_Int m2 = hypre_device_sparse_opts->spgemm_num_passes < 3 ? m : 2*m;
   HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m2, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc,
                                       hypre_device_sparse_opts, hypre_device_sparse_handle);

   if (hypre_device_sparse_opts->spgemm_num_passes < 3)
   {
      hypreDevice_CSRSpGemmWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                              d_ic_out, d_jc_out, d_c_out, nnzC,
                                              hypre_device_sparse_opts, hypre_device_sparse_handle);
   }
   else
   {
      HYPRE_Int rownnz_exact;
      /* a binary array to indicate if row nnz counting is failed for a row */
      //HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *d_rf = d_rc + m;

      hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf,
                                            hypre_device_sparse_opts, hypre_device_sparse_handle);

      /* row nnz is exact if no row failed */
      rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf) == 0;

      //hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

      hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact,
                                                d_ic_out, d_jc_out, d_c_out, nnzC,
                                                hypre_device_sparse_opts, hypre_device_sparse_handle);
   }

   hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSparseHandlePrint()
{
   hypre_printf("ghash_size                   %ld\n",  hypre_device_sparse_handle->ghash_size);
   hypre_printf("ghash2_size                  %ld\n",  hypre_device_sparse_handle->ghash2_size);
   hypre_printf("nnzC_gpu                     %ld\n",  hypre_device_sparse_handle->nnzC_gpu);
   hypre_printf("rownnz_estimate_time         %.2f\n", hypre_device_sparse_handle->rownnz_estimate_time);
   hypre_printf("rownnz_estimate_curand_time  %.2f\n", hypre_device_sparse_handle->rownnz_estimate_curand_time);
   hypre_printf("rownnz_estimate_mem          %ld\n",  hypre_device_sparse_handle->rownnz_estimate_mem);
   hypre_printf("spmm_create_hashtable_time   %.2f\n", hypre_device_sparse_handle->spmm_create_hashtable_time);
   hypre_printf("spmm_attempt1_time           %.2f\n", hypre_device_sparse_handle->spmm_attempt1_time);
   hypre_printf("spmm_post_attempt1_time      %.2f\n", hypre_device_sparse_handle->spmm_post_attempt1_time);
   hypre_printf("spmm_attempt2_time           %.2f\n", hypre_device_sparse_handle->spmm_attempt2_time);
   hypre_printf("spmm_post_attempt2_time      %.2f\n", hypre_device_sparse_handle->spmm_post_attempt2_time);
   hypre_printf("spmm_attempt_mem             %ld\n",  hypre_device_sparse_handle->spmm_attempt_mem);
   hypre_printf("spmm_symbolic_time           %.2f\n", hypre_device_sparse_handle->spmm_symbolic_time);
   hypre_printf("spmm_symbolic_mem            %ld\n",  hypre_device_sparse_handle->spmm_symbolic_mem);
   hypre_printf("spmm_post_symbolic_time      %.2f\n", hypre_device_sparse_handle->spmm_post_symbolic_time);
   hypre_printf("spmm_numeric_time            %.2f\n", hypre_device_sparse_handle->spmm_numeric_time);
   hypre_printf("spmm_numeric_mem             %ld\n",  hypre_device_sparse_handle->spmm_numeric_mem);
   hypre_printf("spmm_post_numeric_time       %.2f\n", hypre_device_sparse_handle->spmm_post_numeric_time);
   hypre_printf("spadd_expansion_time         %.2f\n", hypre_device_sparse_handle->spadd_expansion_time);
   hypre_printf("spadd_sorting_time           %.2f\n", hypre_device_sparse_handle->spadd_sorting_time);
   hypre_printf("spadd_compression_time       %.2f\n", hypre_device_sparse_handle->spadd_compression_time);
   hypre_printf("spadd_convert_ptr_time       %.2f\n", hypre_device_sparse_handle->spadd_convert_ptr_time);
   hypre_printf("spadd_time                   %.2f\n", hypre_device_sparse_handle->spadd_time);
   hypre_printf("sptrans_expansion_time       %.2f\n", hypre_device_sparse_handle->sptrans_expansion_time);
   hypre_printf("sptrans_sorting_time         %.2f\n", hypre_device_sparse_handle->sptrans_sorting_time);
   hypre_printf("sptrans_rowptr_time          %.2f\n", hypre_device_sparse_handle->sptrans_rowptr_time);
   hypre_printf("sptrans_time                 %.2f\n", hypre_device_sparse_handle->sptrans_time);
   hypre_printf("spmm_cusparse_time           %.2f\n", hypre_device_sparse_handle->spmm_cusparse_time);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSparseHandleClearStats()
{
   curandGenerator_t tmp_gen = hypre_device_sparse_handle->gen;
   memset(hypre_device_sparse_handle, 0, sizeof(hypre_DeviceCSRSparseHandle));
   hypre_device_sparse_handle->gen = tmp_gen;

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA */

