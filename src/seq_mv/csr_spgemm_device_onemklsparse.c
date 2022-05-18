/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_SYCL) && defined(HYPRE_USING_ONEMKLSPARSE)

HYPRE_Int
hypreDevice_CSRSpGemmOnemklsparse(HYPRE_Int                            m,
                                  HYPRE_Int                            k,
                                  HYPRE_Int                            n,
                                  oneapi::mkl::sparse::matrix_handle_t handle_A,
                                  HYPRE_Int                            nnzA,
                                  HYPRE_Int                           *d_ia,
                                  HYPRE_Int                           *d_ja,
                                  HYPRE_Complex                       *d_a,
                                  oneapi::mkl::sparse::matrix_handle_t handle_B,
                                  HYPRE_Int                            nnzB,
                                  HYPRE_Int                           *d_ib,
                                  HYPRE_Int                           *d_jb,
                                  HYPRE_Complex                       *d_b,
                                  oneapi::mkl::sparse::matrix_handle_t handle_C,
                                  HYPRE_Int                           *nnzC_out,
                                  HYPRE_Int                          **d_ic_out,
                                  HYPRE_Int                          **d_jc_out,
                                  HYPRE_Complex                      **d_c_out)
{
   std::int64_t *tmp_size1 = NULL, *tmp_size2, *nnzC = NULL;
   void *tmp_buffer1 = NULL;
   void *tmp_buffer2 = NULL;
   HYPRE_Int *d_ic, *d_jc = NULL;
   HYPRE_Complex *d_c = NULL;

   oneapi::mkl::sparse::matmat_descr_t descr = NULL;
   oneapi::mkl::sparse::matmat_request req;

   d_ic = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   oneapi::mkl::sparse::set_csr_data(handle_C, m, n, oneapi::mkl::index_base::zero, d_ic, d_jc, d_c);

   oneapi::mkl::sparse::init_matmat_descr(&descr);
   oneapi::mkl::sparse::set_matmat_data(descr,
                                        oneapi::mkl::sparse::matrix_view_descr::general,
                                        oneapi::mkl::transpose::nontrans,
                                        oneapi::mkl::sparse::matrix_view_descr::general,
                                        oneapi::mkl::transpose::nontrans,
                                        oneapi::mkl::sparse::matrix_view_descr::general);

   /* get tmp_buffer1 size for work estimation */
   req = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
   tmp_size1 = hypre_CTAlloc(std::int64_t, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size1,
                                                  NULL,
                                                  {}).wait() );

   /* allocate tmp_buffer1 for work estimation */
   tmp_buffer1 = (void*) hypre_CTAlloc(std::uint8_t, *tmp_size1, HYPRE_MEMORY_DEVICE);

   /* do work_estimation */
   req = oneapi::mkl::sparse::matmat_request::work_estimation;
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size1,
                                                  tmp_buffer1,
                                                  {}).wait() );

   /* get tmp_buffer2 size for computation */
   req = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
   tmp_size2 = hypre_CTAlloc(std::int64_t, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size2,
                                                  NULL,
                                                  {}).wait() );

   /* allocate tmp_buffer2 for computation */
   tmp_buffer2 = (void*) hypre_CTAlloc(std::uint8_t, *tmp_size2, HYPRE_MEMORY_DEVICE);

   /* do the computation */
   req = oneapi::mkl::sparse::matmat_request::compute;
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size2,
                                                  tmp_buffer2,
                                                  {}).wait() );

   /* get nnzC */
   req = oneapi::mkl::sparse::matmat_request::get_nnz;
   nnzC = hypre_CTAlloc(std::int64_t, 1, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  nnzC,
                                                  NULL,
                                                  {}).wait() );

   /* allocate col index and data arrays */
   d_jc = hypre_CTAlloc(HYPRE_Int, *nnzC, HYPRE_MEMORY_DEVICE);
   d_c = hypre_CTAlloc(HYPRE_Complex, *nnzC, HYPRE_MEMORY_DEVICE);
   oneapi::mkl::sparse::set_csr_data(handle_C, m, n, oneapi::mkl::index_base::zero, d_ic, d_jc, d_c);

   /* finalize C */
   req = oneapi::mkl::sparse::matmat_request::finalize;
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*hypre_HandleComputeStream(hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  NULL,
                                                  NULL,
                                                  {}).wait() );

   /* release the matmat descr */
   oneapi::mkl::sparse::release_matmat_descr(&descr);

   /* assign the output */
   *nnzC_out = (HYPRE_Int) * nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out = d_c;

   return hypre_error_flag;
}
#endif
