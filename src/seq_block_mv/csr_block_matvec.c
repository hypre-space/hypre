/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRBlockMatrix class.
 *
 *****************************************************************************/

#include "csr_block_matrix.h"
#include "../seq_mv/seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRBlockMatrixMatvec(HYPRE_Complex alpha, hypre_CSRBlockMatrix *A,
                           hypre_Vector *x, HYPRE_Complex beta, hypre_Vector *y)
{
   HYPRE_Complex    *A_data   = hypre_CSRBlockMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRBlockMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRBlockMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRBlockMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRBlockMatrixNumCols(A);
   HYPRE_Int         blk_size = hypre_CSRBlockMatrixBlockSize(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);

   HYPRE_Int         i, b1, b2, jj, bnnz = blk_size * blk_size;
   HYPRE_Int         ierr = 0;
   HYPRE_Complex     temp;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   if (num_cols * blk_size != x_size) { ierr = 1; }
   if (num_rows * blk_size != y_size) { ierr = 2; }
   if (num_cols * blk_size != x_size && num_rows * blk_size != y_size) { ierr = 3; }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * blk_size; i++) { y_data[i] *= beta; }

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * blk_size; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * blk_size; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,jj,b1,b2,temp) HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
      {
         for (b1 = 0; b1 < blk_size; b1++)
         {
            temp = y_data[i * blk_size + b1];
            for (b2 = 0; b2 < blk_size; b2++)
            {
               temp += A_data[jj * bnnz + b1 * blk_size + b2] * x_data[A_j[jj] * blk_size + b2];
            }
            y_data[i * blk_size + b1] = temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * blk_size; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRBlockMatrixMatvecT( HYPRE_Complex         alpha,
                             hypre_CSRBlockMatrix *A,
                             hypre_Vector         *x,
                             HYPRE_Complex         beta,
                             hypre_Vector          *y     )
{
   HYPRE_Complex    *A_data    = hypre_CSRBlockMatrixData(A);
   HYPRE_Int        *A_i       = hypre_CSRBlockMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRBlockMatrixJ(A);
   HYPRE_Int         num_rows  = hypre_CSRBlockMatrixNumRows(A);
   HYPRE_Int         num_cols  = hypre_CSRBlockMatrixNumCols(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);

   HYPRE_Complex     temp;

   HYPRE_Int         i, j, jj;
   HYPRE_Int         ierr  = 0;
   HYPRE_Int         b1, b2;

   HYPRE_Int         blk_size = hypre_CSRBlockMatrixBlockSize(A);
   HYPRE_Int         bnnz = blk_size * blk_size;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   if (num_rows * blk_size != x_size)
   {
      ierr = 1;
   }

   if (num_cols * blk_size != y_size)
   {
      ierr = 2;
   }

   if (num_rows * blk_size != x_size && num_cols * blk_size != y_size)
   {
      ierr = 3;
   }
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * blk_size; i++)
      {
         y_data[i] *= beta;
      }

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * blk_size; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * blk_size; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, jj,j, b1, b2) HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i + 1]; jj++) /*each nonzero in that row*/
      {
         for (b1 = 0; b1 < blk_size; b1++) /*row */
         {
            for (b2 = 0; b2 < blk_size; b2++) /*col*/
            {
               j = A_j[jj]; /*col */
               y_data[j * blk_size + b2] +=
                  A_data[jj * bnnz + b1 * blk_size + b2] * x_data[i * blk_size + b1];
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * blk_size; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}

