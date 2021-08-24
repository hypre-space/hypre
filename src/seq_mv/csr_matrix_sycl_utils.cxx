/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#define PSTL_USE_PARALLEL_POLICIES 0 // for GCC 9
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0 // for GCC 10

#include "seq_mv.hpp"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_ONEMKLSPARSE)

/*
 * @brief Creates a SYCL(onemkl) csr handle for a raw CSR matrix
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] offset the first row considered
 * @param[in] nnz Number of nonzeroes [SYCL onemkl doesn't need it]
 * @param[in] *i Row indices
 * @param[in] *j Colmn indices
 * @param[in] *data Values
 * @return oneMKL sparse matrix handle
 */
void
hypre_CSRMatrixToOnemklsparseSpMat_core( HYPRE_Int      n,
                                         HYPRE_Int      m,
                                         HYPRE_Int      offset,
                                         HYPRE_Int     *i,
                                         HYPRE_Int     *j,
                                         HYPRE_Complex *data,
					 oneapi::mkl::sparse::matrix_handle_t& matA_handle)
{
   /*
   assert( (hypre_CSRMatrixNumRows(A) - offset != 0) &&
           (hypre_CSRMatrixNumCols(A) != 0) &&
           (hypre_CSRMatrixNumNonzeros(A) != 0) &&
           "Matrix has no nonzeros");
   */

   HYPRE_SYCL_CALL( oneapi::mkl::sparse::init_matrix_handle(&matA_handle) );
   HYPRE_SYCL_CALL( oneapi::mkl::sparse::set_csr_data(matA_handle,
                                                      n - offset,
                                                      m,
                                                      oneapi::mkl::index_base::zero,
                                                      i + offset,
                                                      j,
                                                      data) );

}

/*
 * @brief Creates a SYCL(onemkl) csr handle from a hypre_CSRMatrix
 * @param[in] *A Pointer to hypre_CSRMatrix
 * @param[in] offset Row offset
 * @return oneMKL sparse matrix handle
 * @warning Assumes CSRMatrix has base 0
 */
void
hypre_CSRMatrixToOnemklsparseSpMat(const hypre_CSRMatrix *A,
                                   HYPRE_Int        offset,
				   oneapi::mkl::sparse::matrix_handle_t& matA_handle)
{
  return hypre_CSRMatrixToOnemklsparseSpMat_core( hypre_CSRMatrixNumRows(A),
                                                  hypre_CSRMatrixNumCols(A),
                                                  offset,
                                                  hypre_CSRMatrixI(A),
                                                  hypre_CSRMatrixJ(A),
                                                  hypre_CSRMatrixData(A),
						  matA_handle);
}

// todo: abagusetty: not needed for onemkl sparse
// /*
//  * @brief Creates a oneMKLsparse dense vector handle from a hypre_Vector
//  * @param[in] *x Pointer to a hypre_Vector
//  * @param[in] offset Row offset
//  * @return onemklsparse dense vector handle
//  * @warning Assumes CSRMatrix uses doubles for values
//  */
// cusparseDnVecDescr_t
// hypre_VectorToOnemklsparseDnVec(const hypre_Vector *x,
//                                 HYPRE_Int     offset,
//                                 HYPRE_Int     size_override)
// {
//   size_override >= 0 ? size_override : hypre_VectorSize(x) - offset,
//     hypre_VectorData(x) + offset,

//    return vecX;
// }

#endif // #if defined(HYPRE_USING_ONEMKLSPARSE)
