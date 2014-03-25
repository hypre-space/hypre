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
 * Constructors and destructors for matrix structure.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_NewMatrix
 *--------------------------------------------------------------------------*/

hypre_Matrix  *hypre_NewMatrix(data, ia, ja, size)
HYPRE_Real  *data;
HYPRE_Int     *ia;
HYPRE_Int     *ja;
HYPRE_Int      size;
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
HYPRE_Real      alpha;
hypre_Matrix         *A;
hypre_Vector         *x;
HYPRE_Real      beta;
hypre_Vector         *y;
{
   HYPRE_Real *a  = hypre_MatrixData(A);
   HYPRE_Int        *ia = hypre_MatrixIA(A);
   HYPRE_Int        *ja = hypre_MatrixJA(A);
   HYPRE_Int         n  = hypre_MatrixSize(A);

   HYPRE_Real *xp = hypre_VectorData(x);
   HYPRE_Real *yp = hypre_VectorData(y);

   HYPRE_Real  temp;

   HYPRE_Int         i, j, jj;


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
HYPRE_Real      alpha;
hypre_Matrix         *A;
hypre_Vector         *x;
HYPRE_Real      beta;
hypre_Vector         *y;
{
   HYPRE_Real *a  = hypre_MatrixData(A);
   HYPRE_Int        *ia = hypre_MatrixIA(A);
   HYPRE_Int        *ja = hypre_MatrixJA(A);
   HYPRE_Int         n  = hypre_MatrixSize(A);

   HYPRE_Real *xp = hypre_VectorData(x);
   HYPRE_Real *yp = hypre_VectorData(y);

   HYPRE_Real  temp;

   HYPRE_Int         xlen = hypre_VectorSize(x);
   HYPRE_Int         ylen = hypre_VectorSize(y);

   HYPRE_Int         i, j, jj;


   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      for (i = 0; i < ylen; i++)
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
	 for (i = 0; i < ylen; i++)
	    yp[i] = 0.0;
      }
      else
      {
	 for (i = 0; i < ylen; i++)
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
