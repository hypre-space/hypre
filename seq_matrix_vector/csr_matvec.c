/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Matvec
 *--------------------------------------------------------------------------*/

int
hypre_Matvec( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y     )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         num_rows = hypre_CSRMatrixNumRows(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);

   double      temp;

   int         i, j, jj;

   int         ierr = 0;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      for (i = 0; i < num_rows; i++)
	 y_data[i] *= beta;

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
	 for (i = 0; i < num_rows; i++)
	    y_data[i] = 0.0;
      }
      else
      {
	 for (i = 0; i < num_rows; i++)
	    y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
	 j = A_j[jj];
         y_data[i] += A_data[jj] * x_data[j];
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
      for (i = 0; i < num_rows; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_Matvec.
 *--------------------------------------------------------------------------*/

int
hypre_MatvecT( double           alpha,
               hypre_CSRMatrix *A,
               hypre_Vector    *x,
               double           beta,
               hypre_Vector    *y     )
{
   double     *A_data    = hypre_CSRMatrixData(A);
   int        *A_i       = hypre_CSRMatrixI(A);
   int        *A_j       = hypre_CSRMatrixJ(A);
   int         num_rows  = hypre_CSRMatrixNumRows(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);

   double      temp;

   int         i, j, jj;

   int         ierr  = 0;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      for (i = 0; i < num_rows; i++)
	 y_data[i] *= beta;

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
	 for (i = 0; i < num_rows; i++)
	    y_data[i] = 0.0;
      }
      else
      {
	 for (i = 0; i < num_rows; i++)
	    y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
	 j = A_j[jj];
         y_data[j] += A_data[jj] * x_data[i];
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
      for (i = 0; i < num_rows; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}

