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
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixMatvec( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y     )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         num_rows = hypre_CSRMatrixNumRows(A);
   int         num_cols = hypre_CSRMatrixNumCols(A);

   int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);
   int         num_vectors = hypre_VectorNumVectors(x);
   int         idxstride_y = hypre_VectorIndexStride(y);
   int         vecstride_y = hypre_VectorVectorStride(y);
   int         idxstride_x = hypre_VectorIndexStride(x);
   int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp, tempx;

   int         i, j, jj;

   int         m;

   double     xpar=0.7;

   int         ierr = 0;


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
 
    hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

    if (num_cols != x_size)
              ierr = 1;

    if (num_rows != y_size)
              ierr = 2;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
       for (i = 0; i < num_rows*num_vectors; i++)
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i++)
	    y_data[i] = 0.0;
      }
      else
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i++)
	    y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
#define HYPRE_SMP_PRIVATE i,jj,m,tempx
#include "../utilities/hypre_smp_forloop.h"

      for (i = 0; i < num_rownnz; i++)
      {
         m = A_rownnz[i];

         /*
          * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
          * {
          *         j = A_j[jj];   
          *  y_data[m] += A_data[jj] * x_data[j];
          * } */
         if ( num_vectors==1 )
         {
            tempx = y_data[m];
            for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
               tempx +=  A_data[jj] * x_data[A_j[jj]];
            y_data[m] = tempx;
         }
         else
            for ( j=0; j<num_vectors; ++j )
            {
               tempx = y_data[ j*vecstride_y + m*idxstride_y ];
               for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
                  tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               y_data[ j*vecstride_y + m*idxstride_y] = tempx;
            }
      }

   }
   else
   {
#define HYPRE_SMP_PRIVATE i,jj,temp
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_rows; i++)
      {
         if ( num_vectors==1 )
         {
            temp = y_data[i];
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               temp += A_data[jj] * x_data[A_j[jj]];
            y_data[i] = temp;
         }
         else
            for ( j=0; j<num_vectors; ++j )
            {
               temp = y_data[ j*vecstride_y + i*idxstride_y ];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] = temp;
            }
      }
   }


   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_rows*num_vectors; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixMatvecT( double           alpha,
               hypre_CSRMatrix *A,
               hypre_Vector    *x,
               double           beta,
               hypre_Vector    *y     )
{
   double     *A_data    = hypre_CSRMatrixData(A);
   int        *A_i       = hypre_CSRMatrixI(A);
   int        *A_j       = hypre_CSRMatrixJ(A);
   int         num_rows  = hypre_CSRMatrixNumRows(A);
   int         num_cols  = hypre_CSRMatrixNumCols(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);
   int         num_vectors = hypre_VectorNumVectors(x);
   int         idxstride_y = hypre_VectorIndexStride(y);
   int         vecstride_y = hypre_VectorVectorStride(y);
   int         idxstride_x = hypre_VectorIndexStride(x);
   int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp;

   int         i, i1, j, jv, jj, ns, ne, size, rest;
   int         num_threads;

   int         ierr  = 0;

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

    hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i++)
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i++)
	    y_data[i] = 0.0;
      }
      else
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i++)
	    y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = hypre_NumThreads();
   if (num_threads > 1)
   {

#define HYPRE_SMP_PRIVATE i, i1,jj,j,ns,ne,size,rest
#include "../utilities/hypre_smp_forloop.h"
      for (i1 = 0; i1 < num_threads; i1++)
      {
         size = num_cols/num_threads;
         rest = num_cols - size*num_threads;
         if (i1 < rest)
         {
            ns = i1*size+i1-1;
            ne = (i1+1)*size+i1+1;
         }
         else
         {
            ns = i1*size+rest-1;
            ne = (i1+1)*size+rest;
         }
         if ( num_vectors==1 )
         {
            for (i = 0; i < num_rows; i++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  if (j > ns && j < ne)
                     y_data[j] += A_data[jj] * x_data[i];
               }
            }
         }
         else
         {
            for (i = 0; i < num_rows; i++)
            {
               for ( jv=0; jv<num_vectors; ++jv )
               {
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     j = A_j[jj];
                     if (j > ns && j < ne)
                        y_data[ j*idxstride_y + jv*vecstride_y ] +=
                           A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
                  }
               }
            }
         }

      }
   }
   else 
   {
      for (i = 0; i < num_rows; i++)
      {
         if ( num_vectors==1 )
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               j = A_j[jj];
               y_data[j] += A_data[jj] * x_data[i];
            }
         }
         else
         {
            for ( jv=0; jv<num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
               }
            }
         }
      }
   }
   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i++)
	 y_data[i] *= alpha;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
                                                                                                              
int
hypre_CSRMatrixMatvec_FF( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y,
              int             *CF_marker_x,
              int             *CF_marker_y,
              int fpt )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         num_rows = hypre_CSRMatrixNumRows(A);
   int         num_cols = hypre_CSRMatrixNumCols(A);
                                                                                                              
   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   int         x_size = hypre_VectorSize(x);
   int         y_size = hypre_VectorSize(y);
                                                                                                              
   double      temp;
                                                                                                              
   int         i, jj;
                                                                                                              
   int         ierr = 0;
                                                                                                              
                                                                                                              
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
                                                                                                              
    if (num_cols != x_size)
              ierr = 1;
                                                                                                              
    if (num_rows != y_size)
              ierr = 2;
                                                                                                              
    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;
                                                                                                              
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/
                                                                                                              
    if (alpha == 0.0)
    {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
       for (i = 0; i < num_rows; i++)
          if (CF_marker_x[i] == fpt) y_data[i] *= beta;
                                                                                                              
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
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
      }
      else
      {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] *= temp;
      }
   }
                                                                                                              
   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/
                                                                                                              
#define HYPRE_SMP_PRIVATE i,jj
#include "../utilities/hypre_smp_forloop.h"
                                                                                                              
   for (i = 0; i < num_rows; i++)
   {
      if (CF_marker_x[i] == fpt)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            if (CF_marker_y[A_j[jj]] == fpt) temp += A_data[jj] * x_data[A_j[jj]];
         y_data[i] = temp;
      }
   }
                                                                                                              
                                                                                                              
   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
                                                                                                              
   if (alpha != 1.0)
   {
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
   }
                                                                                                              
   return ierr;
}
