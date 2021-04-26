/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
/*
 * @brief Creates a cuda csr descriptor for a raw CSR matrix
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] offset the first row considered
 * @param[in] nnz Number of nonzeroes
 * @param[in] *i Row indices
 * @param[in] *j Colmn indices
 * @param[in] *data Values
 * @return Descriptor
 */
cusparseSpMatDescr_t
hypre_CSRMatrixToCusparseSpMat_core( HYPRE_Int      n,
                                     HYPRE_Int      m,
                                     HYPRE_Int      offset,
                                     HYPRE_Int      nnz,
                                     HYPRE_Int     *i,
                                     HYPRE_Int     *j,
                                     HYPRE_Complex *data)
{
   const cudaDataType        data_type  = hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

   cusparseSpMatDescr_t matA;

   /*
   hypre_assert( (hypre_CSRMatrixNumRows(A) - offset != 0) &&
                 (hypre_CSRMatrixNumCols(A) != 0) &&
                 (hypre_CSRMatrixNumNonzeros(A) != 0) &&
                 "Matrix has no nonzeros");
   */

   HYPRE_CUSPARSE_CALL( cusparseCreateCsr(&matA,
                                          n - offset,
                                          m,
                                          nnz,
                                          i + offset,
                                          j,
                                          data,
                                          index_type,
                                          index_type,
                                          index_base,
                                          data_type) );

   return matA;
}

/*
 * @brief Creates a cuSPARSE CSR descriptor from a hypre_CSRMatrix
 * @param[in] *A Pointer to hypre_CSRMatrix
 * @param[in] offset Row offset
 * @return cuSPARSE CSR Descriptor
 * @warning Assumes CSRMatrix has base 0
 */
cusparseSpMatDescr_t
hypre_CSRMatrixToCusparseSpMat(const hypre_CSRMatrix *A,
                                     HYPRE_Int        offset)
{
   return hypre_CSRMatrixToCusparseSpMat_core( hypre_CSRMatrixNumRows(A),
                                               hypre_CSRMatrixNumCols(A),
                                               offset,
                                               hypre_CSRMatrixNumNonzeros(A),
                                               hypre_CSRMatrixI(A),
                                               hypre_CSRMatrixJ(A),
                                               hypre_CSRMatrixData(A) );
}

/*
 * @brief Creates a cuSPARSE dense vector descriptor from a hypre_Vector
 * @param[in] *x Pointer to a hypre_Vector
 * @param[in] offset Row offset
 * @return cuSPARSE dense vector descriptor
 * @warning Assumes CSRMatrix uses doubles for values
 */
cusparseDnVecDescr_t
hypre_VectorToCusparseDnVec(const hypre_Vector *x,
                                  HYPRE_Int     offset,
                                  HYPRE_Int     size_override)
{
   const cudaDataType data_type = hypre_HYPREComplexToCudaDataType();

   cusparseDnVecDescr_t vecX;

   HYPRE_CUSPARSE_CALL( cusparseCreateDnVec(&vecX,
                                            size_override >= 0 ? size_override : hypre_VectorSize(x) - offset,
                                            hypre_VectorData(x) + offset,
                                            data_type) );
   return vecX;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#endif // #if defined(HYPRE_USING_CUSPARSE)

