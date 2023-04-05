/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectVendorSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectVendorSetup( hypre_DirectSolverData     *data,
                                      hypre_UBatchedDenseMatrix  *A,
                                      hypre_Vector               *f,
                                      hypre_Vector               *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_Int                *pivots          = hypre_DirectSolverDataPivots(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Complex           **data_aop        = hypre_UBatchedDenseMatrixDataAOP(A);

   /*-----------------------------------------------------
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
#if defined(HYPRE_USING_CUDA) && defined (HYPRE_USING_CUBLAS)
         HYPRE_CUBLAS_CALL(cublasDgetrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                               num_rows,
                                               diag_aop,
                                               num_rows,
                                               NULL,
                                               info,
                                               num_batches));
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCBLAS)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "LU factorization call from rocBLAS not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for LU factorization!\n");
         return hypre_error_flag;
#endif

      case HYPRE_DIRECT_SOLVER_LUPIV:
#if defined(HYPRE_USING_CUDA) && defined (HYPRE_USING_CUBLAS)
         HYPRE_CUBLAS_CALL(cublasDgetrfBatched(hypre_HandleCublasHandle(hypre_handle()),
                                               num_rows,
                                               diag_aop,
                                               num_rows,
                                               pivots,
                                               info,
                                               num_batches));
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCBLAS)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "LU with pivoting call from rocBLAS not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for LU factorization with pivoting!\n");
         return hypre_error_flag;
#endif

      case HYPRE_DIRECT_SOLVER_CHOL:
#if defined (HYPRE_USING_CUDA) && defined (HYPRE_USING_CUSOLVER)
         HYPRE_CUSOLVER_CALL(cusolverDnDpotrfBatched(
                                hypre_HandleVendorSolverHandle(hypre_handle()),
                                CUBLAS_FILL_MODE_LOWER,
                                num_rows,
                                mat_aop,
                                num_rows,
                                info,
                                num_batches));
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCSOLVER)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Cholesky call from rocBLAS not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for Cholesky factorization!\n");
         return hypre_error_flag;
#endif

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectVendorSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectVendorSolve( hypre_DirectSolverData     *data,
                                      hypre_UBatchedDenseMatrix  *A,
                                      hypre_Vector               *f,
                                      hypre_Vector               *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Complex           **data_aop        = hypre_UBatchedDenseMatrixDataAOP(A);

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
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
      case HYPRE_DIRECT_SOLVER_LUPIV:
#if defined(HYPRE_USING_CUDA) && defined (HYPRE_USING_CUBLAS)
         HYPRE_CUBLAS_CALL(cublasDgetrsBatched(hypre_HandleCublasHandle(hypre_handle()),
                                               CUBLAS_OP_N,
                                               num_rows,
                                               1,
                                               data_aop,
                                               num_rows,
                                               pivots,
                                               sol_aop,
                                               num_rows,
                                               info,
                                               num_batches));
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCBLAS)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "LU with pivoting call from rocBLAS not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for LU factorization with pivoting!\n");
         return hypre_error_flag;
#endif

      case HYPRE_DIRECT_SOLVER_CHOL:
#if defined(HYPRE_USING_CUDA) && defined (HYPRE_USING_CUSOLVER)
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
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCSOLVER)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Cholesky call from rocSOLVER not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for Cholesky solve!\n");
         return hypre_error_flag;
#endif

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   /* Free memory */
   hypre_TFree(sol_aop, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectVendorInvert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectVendorInvert( hypre_DirectSolverData      *data,
                                       hypre_UBatchedDenseMatrix   *A,
                                       hypre_UBatchedDenseMatrix  **Ainv_ptr )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Complex           **mat_aop         = hypre_UBatchedDenseMatrixDataAOP(A);
   HYPRE_Complex           **matinv_aop      = hypre_UBatchedDenseMatrixDataAOP(Ainv);

   /*-----------------------------------------------------
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
      case HYPRE_DIRECT_SOLVER_LUPIV:
#if defined(HYPRE_USING_CUDA) && defined (HYPRE_USING_CUBLAS)
         HYPRE_CUBLAS_CALL(cublasDgetriBatched(hypre_HandleCublasHandle(hypre_handle()),
                                               num_rows,
                                               mat_aop,
                                               num_rows,
                                               pivots,
                                               matinv_aop,
                                               num_rows,
                                               info,
                                               num_batches));
         break;
#elif defined(HYPRE_USING_HIP) && defined (HYPRE_USING_ROCBLAS)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Matrix inversion call from rocBLAS not implemented!\n");
         return hypre_error_flag;
#else
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Missing vendor library for matrix inversion!\n");
         return hypre_error_flag;
#endif

      case HYPRE_DIRECT_SOLVER_CHOL:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
         return hypre_error_flag;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}

#endif
