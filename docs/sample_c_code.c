/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * This is an example of a C code illustrating the indentation used
 * for Hypre.  This code does not illustrate issues related to
 * documentation, error handling, efficiency of implementation or 
 * naming conventions.
 * 
 * The most important item here is consistent indentation of the following
 * structures:
 *    - for loops
 *    - if statements
 *
 * Note that this code does something nonsensical - it is mainly for the
 * illustration of indentation.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Matvec - matrix-vector function.
 * 
 * Calculates y = alpha * A * x + beta * y
 * where A is a matrix stored in compressed sparse row format, x and y
 * are n vectors, and alpha and beta are scalars.
 *
 *--------------------------------------------------------------------------*/

void
Matvec( double  alpha, 
        Matrix *A, 
        Vector *x, 
        double  beta, 
        Vector *y     )
{
   double     *a  = MatrixData(A);  /* element values for matrix A */
   int        *ia = MatrixIA(A);    /* pointer to start of each row */
   int        *ja = MatrixJA(A);    /* column values for matrix elements */
   int         n  = MatrixSize(A);  /* size of matrix */

   double     *xp = VectorData(x);
   double     *yp = VectorData(y);

   double      temp;

   int         i, j, jj;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation 
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      for (i = 0; i < n; i++)
         yp[i] *= beta;

      return;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
   
   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
         for (i = 0; i < n; i++)
            yp[i] = 0.0;
      }
      else
      {
         for (i = 0; i < n; i++)
            yp[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

   for (i = 0; i < n; i++)
   {
      for (jj = ia[i]-1; jj < ia[i+1]-1; jj++)
      {
         j = ja[jj]-1;
         yp[i] += a[jj] * xp[j];
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
      for (i = 0; i < n; i++)
         yp[i] *= alpha;
   }
}
