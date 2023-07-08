/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef SEQ_MV_HPP
#define SEQ_MV_HPP

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
cusparseSpMatDescr_t hypre_CSRMatrixToCusparseSpMat(const hypre_CSRMatrix *A, HYPRE_Int offset);

cusparseSpMatDescr_t hypre_CSRMatrixToCusparseSpMat_core( HYPRE_Int n, HYPRE_Int m,
                                                          HYPRE_Int offset, HYPRE_Int nnz, HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data);

cusparseDnVecDescr_t hypre_VectorToCusparseDnVec_core(HYPRE_Complex *x_data, HYPRE_Int n);

cusparseDnVecDescr_t hypre_VectorToCusparseDnVec(const hypre_Vector *x, HYPRE_Int offset,
                                                 HYPRE_Int size_override);

cusparseDnMatDescr_t hypre_VectorToCusparseDnMat_core(HYPRE_Complex *x_data, HYPRE_Int nrow,
                                                      HYPRE_Int ncol, HYPRE_Int order);

cusparseDnMatDescr_t hypre_VectorToCusparseDnMat(const hypre_Vector *x);
#endif

HYPRE_Int hypreDevice_CSRSpGemmCusparseOldAPI(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                              cusparseMatDescr_t descr_A, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                              cusparseMatDescr_t descr_B, HYPRE_Int nnzB, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                                              cusparseMatDescr_t descr_C, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                              HYPRE_Complex **d_c_out);

HYPRE_Int hypreDevice_CSRSpGemmCusparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                        cusparseMatDescr_t descr_A, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                        cusparseMatDescr_t descr_B, HYPRE_Int nnzB, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                                        cusparseMatDescr_t descr_C, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out,
                                        HYPRE_Complex **d_c_out);

HYPRE_Int hypre_SortCSRCusparse( HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnzA,
                                 cusparseMatDescr_t descrA,
                                 const HYPRE_Int *d_ia, HYPRE_Int *d_ja_sorted, HYPRE_Complex *d_a_sorted );
#endif

#if defined(HYPRE_USING_ROCSPARSE)
HYPRE_Int hypreDevice_CSRSpGemmRocsparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                         rocsparse_mat_descr descrA, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                         rocsparse_mat_descr descrB, HYPRE_Int nnzB, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                                         rocsparse_mat_descr descrC, rocsparse_mat_info infoC, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out,
                                         HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);

HYPRE_Int hypre_SortCSRRocsparse( HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnzA,
                                  rocsparse_mat_descr descrA,
                                  const HYPRE_Int *d_ia, HYPRE_Int *d_ja_sorted, HYPRE_Complex *d_a_sorted );
#endif

#if defined(HYPRE_USING_ONEMKLSPARSE)
HYPRE_Int hypreDevice_CSRSpGemmOnemklsparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                            oneapi::mkl::sparse::matrix_handle_t handle_A,
                                            HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                            oneapi::mkl::sparse::matrix_handle_t handle_B,
                                            HYPRE_Int nnzB, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                                            oneapi::mkl::sparse::matrix_handle_t handle_C,
                                            HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);
#endif

#ifdef __cplusplus
}
#endif

#endif
