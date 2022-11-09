/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetupVendor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetupVendor( hypre_DirectSolverData  *data,
                               hypre_DenseMatrix       *A,
                               hypre_Vector            *f,
                               hypre_Vector            *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);
   HYPRE_Int                 num_batches     = hypre_DenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_DenseMatrixUBatchNumRows(A);
   HYPRE_Complex           **data_aop        = hypre_DenseMatrixDataAOP(A);
   HYPRE_Int                *pivots;

   /*-----------------------------------------------------
    * Select appropiate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL(cublasDgetrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  num_rows,
                                                  diag_aop,
                                                  num_rows,
                                                  NULL,
                                                  info,
                                                  num_batches));
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Missing vendor library for LU factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      case HYPRE_DIRECT_SOLVER_LUPIV:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUBLAS)
            pivots = hypre_CTAlloc(HYPRE_Int, num_rows * num_batches, memory_location);

            HYPRE_CUBLAS_CALL(cublasDgetrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  num_rows,
                                                  diag_aop,
                                                  num_rows,
                                                  pivots,
                                                  info,
                                                  num_batches));

            hypre_DirectSolverDataPivots(data) = pivots;
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Missing vendor library for LU factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUSOLVER)
            HYPRE_CUSOLVER_CALL(cusolverDnDpotrfBatched(
                                   hypre_HandleVendorSolverHandle(hypre_handle()),
                                   CUBLAS_FILL_MODE_LOWER,
                                   num_rows,
                                   mat_aop,
                                   num_rows,
                                   info,
                                   num_batches));
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Missing vendor for Cholesky factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolveVendor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolveVendor( hypre_DirectSolverData  *data,
                               hypre_DenseMatrix       *A,
                               hypre_Vector            *f,
                               hypre_Vector            *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);
   HYPRE_Int                 num_batches     = hypre_DenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_DenseMatrixUBatchNumRows(A);
   HYPRE_Complex           **data_aop        = hypre_DenseMatrixDataAOP(A);

   /* Vector variables */
   HYPRE_Complex            *rhs_data        = hypre_VectorData(f);
   HYPRE_Complex            *sol_data        = hypre_VectorData(u);
   HYPRE_Complex           **sol_aop;

   /*-----------------------------------------------------
    * Create array of pointers for the solution data
    *-----------------------------------------------------*/

   sol_aop = hypre_TAlloc(HYPRE_Complex *, num_batches, HYPRE_MEMORY_DEVICE);
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_ComplexArrayToArrayOfPtrs, gDim, bDim,
                        num_rows * num_batches, num_rows, sol_data, sol_aop );
   }
   hypre_TMemcpy(sol_data, rhs_data, HYPRE_Complex, num_rows * num_batches,
                 memory_location, memory_location);

   /*-----------------------------------------------------
    * Select appropiate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL(cublasDgetrsBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  CUBLAS_OP_N,
                                                  num_rows,
                                                  1,
                                                  data_aop,
                                                  num_rows,
                                                  pivots,
                                                  sol_aop,
                                                  num_rows,
                                                  infos,
                                                  num_batches));
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Missing vendor library for LU factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUSOLVER)
            HYPRE_CUSOLVER_CALL(cusolverDnDpotrsBatched(
                                   hypre_HandleVendorSolverHandle(hypre_handle()),
                                   CUBLAS_FILL_MODE_LOWER,
                                   num_rows,
                                   1,
                                   data_aop,
                                   num_rows,
                                   sol_aop,
                                   num_rows,
                                   info,
                                   num_batches));
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Missing vendor for Cholesky factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   /* Free memory */
   hypre_TFree(sol_aop, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverInvertVendor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverInvertVendor( hypre_DirectSolverData  *data,
                                hypre_DenseMatrix       *A,
                                hypre_DenseMatrix       *Ainv )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);
   HYPRE_Int                 num_batches     = hypre_DenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_DenseMatrixUBatchNumRows(A);
   HYPRE_Complex           **mat_aop         = hypre_DenseMatrixDataAOP(A);
   HYPRE_Complex           **matinv_aop      = hypre_DenseMatrixDataAOP(Ainv);

   /*-----------------------------------------------------
    * Select appropiate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
         if (type == HYPRE_DENSE_MATRIX_STANDARD)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         else if (type == HYPRE_DENSE_MATRIX_UBATCHED)
         {
#if defined (HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL(cublasDgetriBatched(hypre_HandleCublasHandle(hypre_handle()),
                                                  num_rows,
                                                  mat_aop,
                                                  num_rows,
                                                  pivots,
                                                  matinv_aop,
                                                  num_rows,
                                                  info,
                                                  num_batches));
#else
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "Missing vendor library for LU factorization!\n");
            return hypre_error_flag;
#endif
         }
         else if (type == HYPRE_DENSE_MATRIX_VBATCHED)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
         }
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
         return hypre_error_flag;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}

#endif
