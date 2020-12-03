/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* csr_matrix_cuda_utils.c */
#if defined(HYPRE_USING_CUSPARSE)
cusparseSpMatDescr_t hypre_CSRMatrixToCusparseSpMat(const hypre_CSRMatrix *A, HYPRE_Int offset);
cusparseSpMatDescr_t hypre_CSRMatrixToCusparseSpMat_core( HYPRE_Int n, HYPRE_Int m, HYPRE_Int offset, HYPRE_Int nnz, HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data);
cusparseDnVecDescr_t hypre_VectorToCusparseDnVec(const hypre_Vector *x, HYPRE_Int offset, HYPRE_Int size_override);
#endif

