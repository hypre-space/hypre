/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetupCustom
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetupCustom( hypre_DirectSolverData  *data,
                               hypre_DenseMatrix       *A,
                               hypre_Vector            *f,
                               hypre_Vector            *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);

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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
 * hypre_DirectSolverSolveCustom
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolveCustom( hypre_DirectSolverData  *data,
                               hypre_DenseMatrix       *A,
                               hypre_Vector            *f,
                               hypre_Vector            *u )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);

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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
 * hypre_DirectSolverInvertCustom
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverInvertCustom( hypre_DirectSolverData  *data,
                                hypre_DenseMatrix       *A,
                                hypre_DenseMatrix       *Ainv )
{
   hypre_DirectSolverMethod  method          = hypre_DirectSolverDataMethod(data);

   /* Matrix variables */
   hypre_DenseMatrixType     type            = hypre_DenseMatrixType(A);

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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented!\n");
            return hypre_error_flag;
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
