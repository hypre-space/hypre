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

#if defined(HYPRE_USING_CUSPARSE) ||\
    defined(HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * hypre_GpuVecDataCreate
 *--------------------------------------------------------------------------*/

static inline hypre_GpuVecData *
hypre_GpuVecDataCreate(void)
{
   hypre_GpuVecData *data = hypre_CTAlloc(hypre_GpuVecData, 1, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   hypre_GpuVecDataDnVecDescr(data) = NULL;
   hypre_GpuVecDataCachedPtr(data) = NULL;
   hypre_GpuVecDataCachedSize(data) = 0;
   hypre_GpuVecDataCachedType(data) = hypre_HYPREComplexToCudaDataType();
#endif

#if defined(HYPRE_USING_ROCSPARSE) && (ROCSPARSE_VERSION >= 200000)
   hypre_GpuVecDataDnVecDescr(data) = NULL;
   hypre_GpuVecDataCachedPtr(data) = NULL;
   hypre_GpuVecDataCachedSize(data) = 0;
#endif

   return data;
}

/*--------------------------------------------------------------------------
 * hypre_GpuVecDataInvalidate
 *--------------------------------------------------------------------------*/

static inline HYPRE_Int
hypre_GpuVecDataInvalidate(hypre_GpuVecData *data)
{
   if (data)
   {
#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
      if (hypre_GpuVecDataDnVecDescr(data))
      {
         HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(hypre_GpuVecDataDnVecDescr(data)) );
         hypre_GpuVecDataDnVecDescr(data) = NULL;
      }
      hypre_GpuVecDataCachedPtr(data) = NULL;
      hypre_GpuVecDataCachedSize(data) = 0;
      hypre_GpuVecDataCachedType(data) = hypre_HYPREComplexToCudaDataType();
#endif

#if defined(HYPRE_USING_ROCSPARSE) && (ROCSPARSE_VERSION >= 200000)
      if (hypre_GpuVecDataDnVecDescr(data))
      {
         HYPRE_ROCSPARSE_CALL( rocsparse_destroy_dnvec_descr(hypre_GpuVecDataDnVecDescr(data)) );
         hypre_GpuVecDataDnVecDescr(data) = NULL;
      }
      hypre_GpuVecDataCachedPtr(data) = NULL;
      hypre_GpuVecDataCachedSize(data) = 0;
#endif
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_VectorGetGPUVecData
 *--------------------------------------------------------------------------*/

static inline hypre_GpuVecData *
hypre_VectorGetGPUVecData(hypre_Vector *vector)
{
   if (!hypre_VectorGPUVecData(vector))
   {
      hypre_VectorGPUVecData(vector) = hypre_GpuVecDataCreate();
   }

   return hypre_VectorGPUVecData(vector);
}

#endif /* HYPRE_USING_CUSPARSE || HYPRE_USING_ROCSPARSE */

#if defined(HYPRE_USING_ROCSPARSE) && (ROCSPARSE_VERSION >= 200000)

/*--------------------------------------------------------------------------
 * hypre_VectorGetRocsparseDnVecDescr
 *--------------------------------------------------------------------------*/

static inline rocsparse_dnvec_descr
hypre_VectorGetRocsparseDnVecDescr(hypre_Vector       *vector,
                                   int64_t             size,
                                   void               *data,
                                   rocsparse_datatype  type)
{
   hypre_GpuVecData *vec = hypre_VectorGetGPUVecData(vector);

   if (hypre_GpuVecDataDnVecDescr(vec) &&
       hypre_GpuVecDataCachedSize(vec) == size &&
       hypre_GpuVecDataCachedType(vec) == type)
   {
      if (hypre_GpuVecDataCachedPtr(vec) != data)
      {
         HYPRE_ROCSPARSE_CALL( rocsparse_dnvec_set_values(hypre_GpuVecDataDnVecDescr(vec),
                                                          data) );
         hypre_GpuVecDataCachedPtr(vec) = data;
      }

      return hypre_GpuVecDataDnVecDescr(vec);
   }

   if (hypre_GpuVecDataDnVecDescr(vec))
   {
      HYPRE_ROCSPARSE_CALL( rocsparse_destroy_dnvec_descr(hypre_GpuVecDataDnVecDescr(vec)) );
   }

   HYPRE_ROCSPARSE_CALL( rocsparse_create_dnvec_descr(&hypre_GpuVecDataDnVecDescr(vec),
                                                      size,
                                                      data,
                                                      type) );
   hypre_GpuVecDataCachedPtr(vec) = data;
   hypre_GpuVecDataCachedSize(vec) = size;
   hypre_GpuVecDataCachedType(vec) = type;

   return hypre_GpuVecDataDnVecDescr(vec);
}

#endif /* HYPRE_USING_ROCSPARSE && ROCSPARSE_VERSION >= 200000 */

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

/*--------------------------------------------------------------------------
 * hypre_VectorGetCusparseDnVecDescr
 *--------------------------------------------------------------------------*/

static inline cusparseDnVecDescr_t
hypre_VectorGetCusparseDnVecDescr(hypre_Vector *vector,
                                  HYPRE_Int     offset,
                                  HYPRE_Int     size)
{
   hypre_GpuVecData *vec = hypre_VectorGetGPUVecData(vector);
   void             *ptr = hypre_VectorData(vector) + offset;
   cudaDataType      type = hypre_HYPREComplexToCudaDataType();

   if (hypre_GpuVecDataDnVecDescr(vec) &&
       hypre_GpuVecDataCachedSize(vec) == size &&
       hypre_GpuVecDataCachedType(vec) == type)
   {
      if (hypre_GpuVecDataCachedPtr(vec) != ptr)
      {
         HYPRE_CUSPARSE_CALL( cusparseDnVecSetValues(hypre_GpuVecDataDnVecDescr(vec), ptr) );
         hypre_GpuVecDataCachedPtr(vec) = ptr;
      }

      return hypre_GpuVecDataDnVecDescr(vec);
   }

   if (hypre_GpuVecDataDnVecDescr(vec))
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(hypre_GpuVecDataDnVecDescr(vec)) );
   }

   HYPRE_CUSPARSE_CALL( cusparseCreateDnVec(&hypre_GpuVecDataDnVecDescr(vec),
                                            size,
                                            ptr,
                                            type) );
   hypre_GpuVecDataCachedPtr(vec) = ptr;
   hypre_GpuVecDataCachedSize(vec) = size;
   hypre_GpuVecDataCachedType(vec) = type;

   return hypre_GpuVecDataDnVecDescr(vec);
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
hypre_VectorToCusparseDnVec(hypre_Vector *x,
                            HYPRE_Int     offset,
                            HYPRE_Int     size_override)
{
   HYPRE_Int n = size_override >= 0 ? size_override : hypre_VectorSize(x) - offset;
   return hypre_VectorGetCusparseDnVecDescr(x, offset, n);
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
