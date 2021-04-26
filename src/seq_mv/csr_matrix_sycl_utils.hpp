/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_matrix_sycl_utils.cxx */
#if defined(HYPRE_USING_ONEMKLSPARSE)

void hypre_CSRMatrixToOnemklsparseSpMat(const hypre_CSRMatrix *A, HYPRE_Int offset, oneapi::mkl::sparse::matrix_handle_t& matA_handle);

void hypre_CSRMatrixToOnemklsparseSpMat_core( HYPRE_Int n, HYPRE_Int m, HYPRE_Int offset, HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data, oneapi::mkl::sparse::matrix_handle_t& matA_handle);

//cusparseDnVecDescr_t hypre_VectorToOnemklsparseDnVec(const hypre_Vector *x, HYPRE_Int offset, HYPRE_Int size_override);

#endif

