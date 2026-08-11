/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef SEQ_MV_HPP
#define SEQ_MV_HPP

#include "_hypre_utilities.hpp"

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_seq_mv_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
static inline cusparseSpMatDescr_t
hypre_CSRMatrixToCusparseSpMat_core( HYPRE_Int      n,
                                     HYPRE_Int      m,
                                     HYPRE_Int      offset,
                                     HYPRE_Int      nnz,
                                     HYPRE_Int     *i,
                                     HYPRE_Int     *j,
                                     HYPRE_Complex *data )
{
   const cudaDataType        data_type  = hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;
   cusparseSpMatDescr_t      matA;

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

static inline cusparseSpMatDescr_t
hypre_CSRMatrixToCusparseSpMat(const hypre_CSRMatrix *A,
                               HYPRE_Int              offset)
{
   return hypre_CSRMatrixToCusparseSpMat_core(hypre_CSRMatrixNumRows(A),
                                              hypre_CSRMatrixNumCols(A),
                                              offset,
                                              hypre_CSRMatrixNumNonzeros(A),
                                              hypre_CSRMatrixI(A),
                                              hypre_CSRMatrixJ(A),
                                              hypre_CSRMatrixData(A));
}

static inline cusparseDnVecDescr_t
hypre_VectorToCusparseDnVec_core(HYPRE_Complex *x_data,
                                 HYPRE_Int      n)
{
   const cudaDataType   data_type = hypre_HYPREComplexToCudaDataType();
   cusparseDnVecDescr_t vecX;

   HYPRE_CUSPARSE_CALL( cusparseCreateDnVec(&vecX,
                                            n,
                                            x_data,
                                            data_type) );
   return vecX;
}

static inline cusparseDnVecDescr_t
hypre_VectorToCusparseDnVec(const hypre_Vector *x,
                            HYPRE_Int           offset,
                            HYPRE_Int           size_override)
{
   return hypre_VectorToCusparseDnVec_core(hypre_VectorData(x) + offset,
                                           size_override >= 0 ? size_override : hypre_VectorSize(x) - offset);
}

static inline cusparseDnMatDescr_t
hypre_VectorToCusparseDnMat_core(HYPRE_Complex *x_data,
                                 HYPRE_Int      nrow,
                                 HYPRE_Int      ncol,
                                 HYPRE_Int      order)
{
   const cudaDataType  data_type = hypre_HYPREComplexToCudaDataType();
   cusparseDnMatDescr_t matX;

   HYPRE_CUSPARSE_CALL( cusparseCreateDnMat(&matX,
                                            nrow,
                                            ncol,
                                            (order == 0) ? nrow : ncol,
                                            x_data,
                                            data_type,
                                            (order == 0) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW) );
   return matX;
}

static inline cusparseDnMatDescr_t
hypre_VectorToCusparseDnMat(const hypre_Vector *x)
{
   return hypre_VectorToCusparseDnMat_core(hypre_VectorData(x),
                                           hypre_VectorSize(x),
                                           hypre_VectorNumVectors(x),
                                           hypre_VectorMultiVecStorageMethod(x));
}
#endif

#ifdef __cplusplus
}
#endif

#endif
