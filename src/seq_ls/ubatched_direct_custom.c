/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectCustomSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectCustomSetup( hypre_DirectSolverData     *data,
                                      hypre_UBatchedDenseMatrix  *A,
                                      hypre_Vector               *f,
                                      hypre_Vector               *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
#if 0
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_Int                *pivots          = hypre_DirectSolverDataPivots(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Complex           **data_aop        = hypre_UBatchedDenseMatrixDataAOP(A);
#endif

   /*-----------------------------------------------------
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      case HYPRE_DIRECT_SOLVER_LUPIV:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectCustomSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectCustomSolve( hypre_DirectSolverData     *data,
                                      hypre_UBatchedDenseMatrix  *A,
                                      hypre_Vector               *f,
                                      hypre_Vector               *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
   //HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   //HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Int                 num_rows_total  = hypre_UBatchedDenseMatrixNumRowsTotal(A);
   //HYPRE_Complex           **data_aop        = hypre_UBatchedDenseMatrixDataAOP(A);

   /* Vector variables */
   HYPRE_Complex            *rhs_data        = hypre_VectorData(f);
   HYPRE_Complex            *sol_data        = hypre_VectorData(u);
   HYPRE_Complex           **sol_aop;

   /*-----------------------------------------------------
    * Create array of pointers for the solution data
    *-----------------------------------------------------*/

   /* Allocate */
   sol_aop = hypre_TAlloc(HYPRE_Complex *, num_batches, memory_location);

   /* Build array of pointers to solution vector */
   hypre_SeqVectorDataToArrayOfPointers(u, num_batches, sol_aop);

   /* Copy rhs -> sol */
   hypre_TMemcpy(sol_data, rhs_data, HYPRE_Complex, num_rows_total,
                 memory_location, memory_location);

   /*-----------------------------------------------------
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
      case HYPRE_DIRECT_SOLVER_LUPIV:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   /* Free memory */
   hypre_TFree(sol_aop, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseDirectCustomInvert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseDirectCustomInvert( hypre_DirectSolverData      *data,
                                       hypre_UBatchedDenseMatrix   *A,
                                       hypre_UBatchedDenseMatrix  **Ainv_ptr )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);
#if 0
   HYPRE_Int                *info            = hypre_DirectSolverDataInfo(data);
   HYPRE_MemoryLocation      memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /* Matrix variables */
   HYPRE_Int                 num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int                 num_rows        = hypre_UBatchedDenseMatrixNumRows(A);
   HYPRE_Complex           **mat_aop         = hypre_UBatchedDenseMatrixDataAOP(A);
   HYPRE_Complex           **matinv_aop      = hypre_UBatchedDenseMatrixDataAOP(Ainv_ptr);
#endif

   /*-----------------------------------------------------
    * Select appropriate method
    *-----------------------------------------------------*/

   switch (method)
   {
      case HYPRE_DIRECT_SOLVER_LU:
      case HYPRE_DIRECT_SOLVER_LUPIV:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      case HYPRE_DIRECT_SOLVER_CHOL:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver method!\n");
   }

   return hypre_error_flag;
}
