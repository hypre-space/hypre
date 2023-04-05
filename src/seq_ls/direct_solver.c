/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_DirectSolverCreate
 *--------------------------------------------------------------------------*/

void*
hypre_DirectSolverCreate( hypre_DirectSolverBackend  backend,
                          hypre_DirectSolverMethod   method,
                          hypre_MatrixType           mat_type,
                          HYPRE_Int                  size,
                          HYPRE_MemoryLocation       memory_location )
{
   hypre_DirectSolverData  *data;

   data = hypre_TAlloc(hypre_DirectSolverData, 1, HYPRE_MEMORY_HOST);

   /* Set default values */
   hypre_DirectSolverDataBackend(data)        = backend;
   hypre_DirectSolverDataMethod(data)         = method;
   hypre_DirectSolverDataMatType(data)        = mat_type;
   hypre_DirectSolverDataSize(data)           = size;
   hypre_DirectSolverDataPivots(data)         = NULL;
   hypre_DirectSolverDataInfo(data)           = NULL;
   hypre_DirectSolverDataMemoryLocation(data) = memory_location;

   /* TODO: Call specific create routines for each solver option */
   switch (backend)
   {
      case HYPRE_DIRECT_SOLVER_VENDOR:
         hypre_printf("Call Create for option 1\n");
         break;

      case HYPRE_DIRECT_SOLVER_CUSTOM:
         hypre_printf("Call Create for option 2\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
   }

   return (void*) data;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverDestroy(void* vdata)
{
   hypre_DirectSolverData  *data = (hypre_DirectSolverData*) vdata;
   HYPRE_MemoryLocation     memory_location;

   if (data)
   {
      memory_location = hypre_DirectSolverDataMemoryLocation(data);

      hypre_TFree(hypre_DirectSolverDataInfo(data), memory_location);
      hypre_TFree(hypre_DirectSolverDataPivots(data), memory_location);
      hypre_TFree(data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverInitialize(void* vdata)
{
   hypre_DirectSolverData    *data   = (hypre_DirectSolverData*) vdata;
   hypre_DirectSolverMethod   method = hypre_DirectSolverDataMethod(data);
   HYPRE_Int                  size   = hypre_DirectSolverDataSize(data);
   HYPRE_MemoryLocation       memory_location = hypre_DirectSolverDataMemoryLocation(data);

   hypre_DirectSolverDataInfo(data) = hypre_CTAlloc(HYPRE_Int, size, memory_location);

   if (method == HYPRE_DIRECT_SOLVER_LUPIV)
   {
      hypre_DirectSolverDataPivots(data) = hypre_CTAlloc(HYPRE_Int, size, memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetup( void         *vdata,
                         void         *vA,
                         hypre_Vector *f,
                         hypre_Vector *u )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;

   hypre_DirectSolverBackend   backend  = hypre_DirectSolverDataBackend(data);
   hypre_MatrixType            mat_type = hypre_DirectSolverDataMatType(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Select matrix type
    *-----------------------------------------------------*/

   switch (mat_type)
   {
      case HYPRE_MATRIX_TYPE_UBATCHED_DENSE:
         hypre_UBatchedDenseMatrix *A = (hypre_UBatchedDenseMatrix*) vA;

         /*-----------------------------------------------------
          * Sanity check
          *-----------------------------------------------------*/

         if (memory_location != hypre_UBatchedDenseMatrixMemoryLocation(A))
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
            return hypre_error_flag;
         }

         /*-----------------------------------------------------
          * Select appropriate backend
          *-----------------------------------------------------*/

         switch (backend)
         {
#if defined (HYPRE_USING_CUDA)
            case HYPRE_DIRECT_SOLVER_VENDOR:
               hypre_UBatchedDenseDirectVendorSetup(data, A, f, u);
               break;
#endif

            case HYPRE_DIRECT_SOLVER_CUSTOM:
               hypre_UBatchedDenseDirectCustomSetup(data, A, f, u);
               break;

            default:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
               return hypre_error_flag;
         }
         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_DENSE:
         //hypre_VBatchedDenseMatrix *A = (hypre_VBatchedDenseMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           variable batched dense matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_UBATCHED_SPARSE:
         //hypre_UBatchedCSRMatrix *A = (hypre_UBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           uniform batched sparse matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_SPARSE:
         //hypre_VBatchedCSRMatrix *A = (hypre_VBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           variable batched sparse matrices!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown matrix type!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolve( void         *vdata,
                         void         *vA,
                         hypre_Vector *f,
                         hypre_Vector *u )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;

   hypre_DirectSolverBackend   backend  = hypre_DirectSolverDataBackend(data);
   hypre_MatrixType            mat_type = hypre_DirectSolverDataMatType(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Select matrix type
    *-----------------------------------------------------*/

   switch (mat_type)
   {
      case HYPRE_MATRIX_TYPE_UBATCHED_DENSE:
         hypre_UBatchedDenseMatrix *A = (hypre_UBatchedDenseMatrix*) vA;

         /*-----------------------------------------------------
          * Sanity check
          *-----------------------------------------------------*/

         if (memory_location != hypre_UBatchedDenseMatrixMemoryLocation(A))
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
            return hypre_error_flag;
         }

         /*-----------------------------------------------------
          * Select appropriate backend
          *-----------------------------------------------------*/

         switch (backend)
         {
#if defined (HYPRE_USING_CUDA)
            case HYPRE_DIRECT_SOLVER_VENDOR:
               hypre_UBatchedDenseDirectVendorSolve(data, A, f, u);
               break;
#endif

            case HYPRE_DIRECT_SOLVER_CUSTOM:
               hypre_UBatchedDenseDirectCustomSolve(data, A, f, u);
               break;

            default:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
               return hypre_error_flag;
         }
         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_DENSE:
         //hypre_VBatchedDenseMatrix *A = (hypre_VBatchedDenseMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           variable batched dense matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_UBATCHED_SPARSE:
         //hypre_UBatchedCSRMatrix *A = (hypre_UBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           uniform batched sparse matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_SPARSE:
         //hypre_VBatchedCSRMatrix *A = (hypre_VBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Direct solvers not implemented for\
                           variable batched sparse matrices!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown matrix type!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverInvert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverInvert( void  *vdata,
                          void  *vA,
                          void **vAinv_ptr )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;

   hypre_DirectSolverBackend   backend  = hypre_DirectSolverDataBackend(data);
   hypre_MatrixType            mat_type = hypre_DirectSolverDataMatType(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Select matrix type
    *-----------------------------------------------------*/

   switch (mat_type)
   {
      case HYPRE_MATRIX_TYPE_UBATCHED_DENSE:
         hypre_UBatchedDenseMatrix *A = (hypre_UBatchedDenseMatrix*) vA;
         hypre_UBatchedDenseMatrix *Ainv;

         /*-----------------------------------------------------
          * Sanity check
          *-----------------------------------------------------*/

         if (memory_location != hypre_UBatchedDenseMatrixMemoryLocation(A))
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
            return hypre_error_flag;
         }

         /*-----------------------------------------------------
          * Select appropriate backend
          *-----------------------------------------------------*/

         switch (backend)
         {
#if defined (HYPRE_USING_CUDA)
            case HYPRE_DIRECT_SOLVER_VENDOR:
               hypre_UBatchedDenseDirectVendorInvert(data, A, &Ainv);
               break;
#endif

            case HYPRE_DIRECT_SOLVER_CUSTOM:
               hypre_UBatchedDenseDirectCustomInvert(data, A, &Ainv);
               break;

            default:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
               return hypre_error_flag;
         }

         /* Set output pointer */
         *vAinv_ptr = (void*) Ainv;

         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_DENSE:
         //hypre_VBatchedDenseMatrix *A = (hypre_VBatchedDenseMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Matrix inversion implemented for\
                           variable batched dense matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_UBATCHED_SPARSE:
         //hypre_UBatchedCSRMatrix *A = (hypre_UBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Matrix inversion not implemented for\
                           uniform batched sparse matrices!\n");
         break;

      case HYPRE_MATRIX_TYPE_VBATCHED_SPARSE:
         //hypre_VBatchedCSRMatrix *A = (hypre_VBatchedCSRMatrix*) vA;

         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Matrix inversion not implemented for\
                           variable batched sparse matrices!\n");
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown matrix type!\n");
   }

   return hypre_error_flag;
}
