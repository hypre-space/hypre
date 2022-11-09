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
hypre_DirectSolverCreate( hypre_DirectSolverBackend backend,
                          hypre_DirectSolverMethod  method,
                          HYPRE_Int                 info_size,
                          HYPRE_MemoryLocation      memory_location )
{
   hypre_DirectSolverData  *data;

   data = hypre_TAlloc(hypre_DirectSolverData, 1, HYPRE_MEMORY_HOST);

   /* Set default values */
   hypre_DirectSolverDataBackend(data)        = backend;
   hypre_DirectSolverDataMethod(data)         = method;
   hypre_DirectSolverDataInfoSize(data)       = info_size;
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
   HYPRE_MemoryLocation     memory_location = hypre_DirectSolverDataMemoryLocation(data);

   if (data)
   {
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
   hypre_DirectSolverData  *data = (hypre_DirectSolverData*) vdata;
   HYPRE_Int                size = hypre_DirectSolverDataInfoSize(data);
   HYPRE_MemoryLocation     memory_location = hypre_DirectSolverDataMemoryLocation(data);

   hypre_DirectSolverDataInfo(data) = hypre_CTAlloc(HYPRE_Int, size, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetup( void               *vdata,
                         hypre_DenseMatrix  *A,
                         hypre_Vector       *f,
                         hypre_Vector       *u )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;
   hypre_DirectSolverBackend   backend = hypre_DirectSolverDataBackend(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Sanity check
    *-----------------------------------------------------*/

   if (memory_location != hypre_DenseMatrixMemoryLocation(A))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Select appropiate backend
    *-----------------------------------------------------*/

   switch (backend)
   {
#if defined (HYPRE_USING_CUDA)
      case HYPRE_DIRECT_SOLVER_VENDOR:
         hypre_DirectSolverSetupVendor(data, A, f, u);
         break;
#endif

      case HYPRE_DIRECT_SOLVER_CUSTOM:
         hypre_DirectSolverSetupCustom(data, A, f, u);
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolve( void               *vdata,
                         hypre_DenseMatrix  *A,
                         hypre_Vector       *f,
                         hypre_Vector       *u )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;
   hypre_DirectSolverBackend   backend = hypre_DirectSolverDataBackend(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Sanity checks
    *-----------------------------------------------------*/

   if (memory_location != hypre_DenseMatrixMemoryLocation(A))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
      return hypre_error_flag;
   }

   if (memory_location != hypre_VectorMemoryLocation(f))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for f!\n");
      return hypre_error_flag;
   }

   if (memory_location != hypre_VectorMemoryLocation(u))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for u!\n");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Select appropiate backend
    *-----------------------------------------------------*/

   switch (backend)
   {
#if defined (HYPRE_USING_CUDA)
      case HYPRE_DIRECT_SOLVER_VENDOR:
         hypre_DirectSolverSolveVendor(data, A, f, u);
         break;
#endif

      case HYPRE_DIRECT_SOLVER_CUSTOM:
         hypre_DirectSolverSolveCustom(data, A, f, u);
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverInvert
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverInvert( void               *vdata,
                          hypre_DenseMatrix  *A,
                          hypre_DenseMatrix  *Ainv )
{
   hypre_DirectSolverData     *data = (hypre_DirectSolverData*) vdata;
   hypre_DirectSolverBackend   backend = hypre_DirectSolverDataBackend(data);
   HYPRE_MemoryLocation        memory_location = hypre_DirectSolverDataMemoryLocation(data);

   /*-----------------------------------------------------
    * Sanity check
    *-----------------------------------------------------*/

   if (memory_location != hypre_DenseMatrixMemoryLocation(A))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for A!\n");
      return hypre_error_flag;
   }

   if (memory_location != hypre_DenseMatrixMemoryLocation(Ainv))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unexpected memory location for Ainv!\n");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------
    * Select appropiate backend
    *-----------------------------------------------------*/

   switch (backend)
   {
#if defined (HYPRE_USING_CUDA)
      case HYPRE_DIRECT_SOLVER_VENDOR:
         hypre_DirectSolverInvertVendor(data, A, Ainv);
         break;
#endif

      case HYPRE_DIRECT_SOLVER_CUSTOM:
         hypre_DirectSolverInvertCustom(data, A, Ainv);
         break;

      default:
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unknown direct solver backend!\n");
   }

   return hypre_error_flag;
}
