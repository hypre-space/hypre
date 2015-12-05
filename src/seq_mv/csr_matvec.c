/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.16 $
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

HYPRE_Int
hypre_CSRMatrixMatvec( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y     )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   double     xpar=0.7;

   HYPRE_Int         ierr = 0;


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
            tempx = 0.0;
            for (jj = A_i[m]; jj < A_i[m+1]; jj++)
               tempx +=  A_data[jj] * x_data[A_j[jj]];
            y_data[m] += tempx;
         }
         else
            for ( j=0; j<num_vectors; ++j )
            {
               tempx = 0.0;
               for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
                  tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               y_data[ j*vecstride_y + m*idxstride_y] += tempx;
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
            temp = 0.0;
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               temp += A_data[jj] * x_data[A_j[jj]];
            y_data[i] += temp;
         }
         else
            for ( j=0; j<num_vectors; ++j )
            {
               temp = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] += temp;
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

#if 0
/* AHB: 10/2009- not using this one anymore because of threading scheme */
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvecT( double           alpha,
               hypre_CSRMatrix *A,
               hypre_Vector    *x,
               double           beta,
               hypre_Vector    *y     )
{
   double     *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp;

   HYPRE_Int         i, i1, j, jv, jj, ns, ne, size, rest;
   HYPRE_Int         num_threads;

   HYPRE_Int         ierr  = 0;

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
#else

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvecT( double           alpha,
               hypre_CSRMatrix *A,
               hypre_Vector    *x,
               double           beta,
               hypre_Vector    *y     )
{
   double     *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   double      temp;

   double      *y_data_expand;
   HYPRE_Int         my_thread_num = 0, offset = 0;
   
   HYPRE_Int         i, j, jv, jj;
   HYPRE_Int         num_threads;

   HYPRE_Int         ierr  = 0;

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
      y_data_expand = hypre_CTAlloc(double, num_threads*y_size);

      if ( num_vectors==1 )
      {

#define HYPRE_SMP_PRIVATE i,jj,j,my_thread_num,offset
#define HYPRE_SMP_PAR_REGION
#include "../utilities/hypre_smp_forloop.h"
         {                                      
            my_thread_num = hypre_GetThreadNum();
            offset =  y_size*my_thread_num;
#define HYPRE_SMP_FOR
#include "../utilities/hypre_smp_forloop.h"
            for (i = 0; i < num_rows; i++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data_expand[offset + j] += A_data[jj] * x_data[i];
               }
            }

            /* implied barrier (for threads)*/           
#define HYPRE_SMP_FOR
#include "../utilities/hypre_smp_forloop.h"
            for (i = 0; i < y_size; i++)
            {
               for (j = 0; j < num_threads; j++)
               {
                  y_data[i] += y_data_expand[j*y_size + i];
                  
               }
            }

         } /* end parallel threaded region */
      }
      else
      {
         /* multiple vector case is not threaded */
         for (i = 0; i < num_rows; i++)
         {
            for ( jv=0; jv<num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
               }
            }
         }
      }

      hypre_TFree(y_data_expand);

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


#endif
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
                                                                                                              
HYPRE_Int
hypre_CSRMatrixMatvec_FF( double           alpha,
              hypre_CSRMatrix *A,
              hypre_Vector    *x,
              double           beta,
              hypre_Vector    *y,
              HYPRE_Int             *CF_marker_x,
              HYPRE_Int             *CF_marker_y,
              HYPRE_Int fpt )
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
                                                                                                              
   double     *x_data = hypre_VectorData(x);
   double     *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
                                                                                                              
   double      temp;
                                                                                                              
   HYPRE_Int         i, jj;
                                                                                                              
   HYPRE_Int         ierr = 0;
                                                                                                              
                                                                                                              
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
