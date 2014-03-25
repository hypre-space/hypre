/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





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
   HYPRE_Int        *ia = MatrixIA(A);    /* pointer to start of each row */
   HYPRE_Int        *ja = MatrixJA(A);    /* column values for matrix elements */
   HYPRE_Int         n  = MatrixSize(A);  /* size of matrix */

   double     *xp = VectorData(x);
   double     *yp = VectorData(y);

   double      temp;

   HYPRE_Int         i, j, jj;

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
