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
 * Constructors and destructors for matrix structure.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewMatrix
 *--------------------------------------------------------------------------*/

Matrix  *NewMatrix(data, ia, ja, size)
double  *data;
int     *ia;
int     *ja;
int      size;
{
   Matrix     *new;


   new = ctalloc(Matrix, 1);

   MatrixData(new) = data;
   MatrixIA(new)   = ia;
   MatrixJA(new)   = ja;
   MatrixSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * FreeMatrix
 *--------------------------------------------------------------------------*/

void     FreeMatrix(matrix)
Matrix  *matrix;
{
   if (matrix)
   {
      tfree(MatrixData(matrix));
      tfree(MatrixIA(matrix));
      tfree(MatrixJA(matrix));
      tfree(matrix);
   }
}

/*--------------------------------------------------------------------------
 * Matvec
 *--------------------------------------------------------------------------*/

void            Matvec(alpha, A, x, beta, y)
double          alpha;
Matrix         *A;
Vector         *x;
double          beta;
Vector         *y;
{
   double     *a  = MatrixData(A);
   int        *ia = MatrixIA(A);
   int        *ja = MatrixJA(A);
   int         n  = MatrixSize(A);

   double     *xp = VectorData(x);
   double     *yp = VectorData(y);

   double      temp;

   int         i, j, jj;


   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
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
