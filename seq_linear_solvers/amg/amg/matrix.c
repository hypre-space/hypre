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
 * hypre_NewMatrix
 *--------------------------------------------------------------------------*/

hypre_Matrix  *hypre_NewMatrix(data, ia, ja, size)
double  *data;
int     *ia;
int     *ja;
int      size;
{
   hypre_Matrix     *new;


   new = hypre_CTAlloc(hypre_Matrix, 1);

   hypre_MatrixData(new) = data;
   hypre_MatrixIA(new)   = ia;
   hypre_MatrixJA(new)   = ja;
   hypre_MatrixSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * hypre_FreeMatrix
 *--------------------------------------------------------------------------*/

void     hypre_FreeMatrix(matrix)
hypre_Matrix  *matrix;
{
   if (matrix)
   {
      hypre_TFree(hypre_MatrixData(matrix));
      hypre_TFree(hypre_MatrixIA(matrix));
      hypre_TFree(hypre_MatrixJA(matrix));
      hypre_TFree(matrix);
   }
}

/*--------------------------------------------------------------------------
 * hypre_Matvec
 *--------------------------------------------------------------------------*/

void            hypre_Matvec(alpha, A, x, beta, y)
double          alpha;
hypre_Matrix         *A;
hypre_Vector         *x;
double          beta;
hypre_Vector         *y;
{
   double     *a  = hypre_MatrixData(A);
   int        *ia = hypre_MatrixIA(A);
   int        *ja = hypre_MatrixJA(A);
   int         n  = hypre_MatrixSize(A);

   double     *xp = hypre_VectorData(x);
   double     *yp = hypre_VectorData(y);

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
         if (j >= 0)        
         {                        
                /* temporary alteration, until P_array
                   has excess entries removed.  veh, 1/98 */

	    yp[i] += a[jj] * xp[j];
         }
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



/*--------------------------------------------------------------------------
 * hypre_MatvecT
 *
 *             Performs y <- alpha * A^T * x + beta * y
 *
 *             Modified from hypre_Matvec, Jan 7 1998, by Van Henson
 *
 *--------------------------------------------------------------------------*/

void            hypre_MatvecT(alpha, A, x, beta, y)
double          alpha;
hypre_Matrix         *A;
hypre_Vector         *x;
double          beta;
hypre_Vector         *y;
{
   double     *a  = hypre_MatrixData(A);
   int        *ia = hypre_MatrixIA(A);
   int        *ja = hypre_MatrixJA(A);
   int         n  = hypre_MatrixSize(A);

   double     *xp = hypre_VectorData(x);
   double     *yp = hypre_VectorData(y);

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
    * y += A^T*x
    *-----------------------------------------------------------------*/

   for (i = 0; i < n; i++)
   {
      for (jj = ia[i]-1; jj < ia[i+1]-1; jj++)
      {
	 j = ja[jj]-1;
         if (j >= 0)        
         {                        
                /* temporary alteration, until P_array
                   has excess entries removed.  veh, 1/98 */

	    yp[j] += a[jj] * xp[i];
         }
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
