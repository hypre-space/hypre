/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "csr_multimatvec.h"
#include "seq_mv.h"
#include "seq_multivector.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatMultivec(HYPRE_Complex alpha, hypre_CSRMatrix *A,
                           hypre_Multivector *x, HYPRE_Complex beta,
                           hypre_Multivector *y)
{
   HYPRE_Complex *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int    *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int    *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int    num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int    num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex *x_data = hypre_MultivectorData(x);
   HYPRE_Complex *y_data = hypre_MultivectorData(y);
   HYPRE_Int    x_size = hypre_MultivectorSize(x);
   HYPRE_Int    y_size = hypre_MultivectorSize(y);
   HYPRE_Int    num_vectors = hypre_MultivectorNumVectors(x);
   HYPRE_Int    *x_active_ind = x->active_indices;
   HYPRE_Int    *y_active_ind = y->active_indices;
   HYPRE_Int    num_active_vectors = x->num_active_vectors;
   HYPRE_Int    i, j, jj, m, ierr = 0, optimize;
   HYPRE_Complex temp, tempx, xpar = 0.7, *xptr, *yptr;

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

   hypre_assert(num_active_vectors == y->num_active_vectors);
   if (num_cols != x_size) { ierr = 1; }
   if (num_rows != y_size) { ierr = 2; }
   if (num_cols != x_size && num_rows != y_size) { ierr = 3; }
   optimize = 0;
   if (num_active_vectors == num_vectors && num_vectors == y->num_vectors)
   {
      optimize = 1;
   }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] *= beta; }

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
         for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] = 0.0; }
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] *= temp; }
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

   if ( num_vectors == 1 )
   {
      for (i = 0; i < num_rows; i++)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            temp += A_data[jj] * x_data[A_j[jj]];
         }
         y_data[i] = temp;
      }
   }
   else
   {
      if (optimize == 0)
      {
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_active_vectors; ++j)
            {
               xptr = x_data[x_active_ind[j] * x_size];
               temp = y_data[y_active_ind[j] * y_size + i];
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  temp += A_data[jj] * xptr[A_j[jj]];
               }
               y_data[y_active_ind[j]*y_size + i] = temp;
            }
         }
      }
      else
      {
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_vectors; ++j)
            {
               xptr = x_data[j * x_size];
               temp = y_data[j * y_size + i];
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  temp += A_data[jj] * xptr[A_j[jj]];
               }
               y_data[j * y_size + i] = temp;
            }
         }
         /* different version
         for (j=0; j<num_vectors; ++j)
         {
            xptr = x_data[j*x_size];
            for (i = 0; i < num_rows; i++)
            {
               temp = y_data[j*y_size+i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  temp += A_data[jj] * xptr[A_j[jj]];
               y_data[j*y_size+i] = temp;
            }
         }
         */
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
      for (i = 0; i < num_rows * num_vectors; i++)
      {
         y_data[i] *= alpha;
      }
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatMultivecT(HYPRE_Complex alpha, hypre_CSRMatrix *A,
                            hypre_Multivector *x, HYPRE_Complex beta,
                            hypre_Multivector *y)
{
   HYPRE_Complex *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int    *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int    *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int    num_rows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int    num_cols  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex *x_data = hypre_MultivectorData(x);
   HYPRE_Complex *y_data = hypre_MultivectorData(y);
   HYPRE_Int    x_size = hypre_MultivectorSize(x);
   HYPRE_Int    y_size = hypre_MultivectorSize(y);
   HYPRE_Int    num_vectors = hypre_MultivectorNumVectors(x);
   HYPRE_Int    *x_active_ind = x->active_indices;
   HYPRE_Int    *y_active_ind = y->active_indices;
   HYPRE_Int    num_active_vectors = x->num_active_vectors;
   HYPRE_Complex temp;
   HYPRE_Int    i, jv, jj, size, ierr = 0;

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

   hypre_assert(num_active_vectors == y->num_active_vectors);
   if (num_rows != x_size) { ierr = 1; }
   if (num_cols != y_size) { ierr = 2; }
   if (num_rows != x_size && num_cols != y_size) { ierr = 3; }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] *= beta; }
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
         for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] = 0.0; }
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] *= temp; }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/

   if ( num_vectors == 1 )
   {
      for (i = 0; i < num_rows; i++)
      {
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            y_data[A_j[jj]] += A_data[jj] * x_data[i];
         }
      }
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            y_data[A_j[jj] + jv * y_size] += A_data[jj] * x_data[i + jv * x_size];
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
      for (i = 0; i < num_cols * num_vectors; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}

