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

#include "seq_mv.h"
#include <assert.h>


/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlace( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset )
{
#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) /* CUDA */
   PUSH_RANGE_PAYLOAD("MATVEC",0, hypre_CSRMatrixNumRows(A));
   HYPRE_Int ierr = hypre_CSRMatrixMatvecDevice( alpha,A,x,beta,b,y,offset );
   POP_RANGE;
#elif defined(HYPRE_USING_OPENMP_OFFLOAD) /* OMP 4.5 */
   PUSH_RANGE_PAYLOAD("MATVEC-OMP",0, hypre_CSRMatrixNumRows(A));
   HYPRE_Int ierr = hypre_CSRMatrixMatvecOutOfPlaceOOMP( alpha,A,x,beta,b,y,offset );
   POP_RANGE;
#else /* CPU */
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);
   HYPRE_Complex     temp, tempx;
   HYPRE_Int         i, j, jj, m, ierr=0;
   HYPRE_Real        xpar=0.7;
   hypre_Vector     *x_tmp = NULL;

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
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows*num_vectors; i++)
         y_data[i] = beta*b_data[i];

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows) || num_vectors > 1)
   {
      /*-----------------------------------------------------------------------
       * y = (beta/alpha)*y
       *-----------------------------------------------------------------------*/

      if (temp != 1.0)
      {
         if (temp == 0.0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = 0.0;
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i]*temp;
         }
      }
      else
      {
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i];
      }


      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      if (num_rownnz < xpar*(num_rows))
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,m,tempx) HYPRE_SMP_SCHEDULE
#endif

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
               tempx = 0;
               for (jj = A_i[m]; jj < A_i[m+1]; jj++)
                  tempx +=  A_data[jj] * x_data[A_j[jj]];
               y_data[m] += tempx;
            }
            else
               for ( j=0; j<num_vectors; ++j )
               {
                  tempx = 0;
                  for (jj = A_i[m]; jj < A_i[m+1]; jj++)
                     tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
                  y_data[ j*vecstride_y + m*idxstride_y] += tempx;
               }
         }
      }
      else // num_vectors > 1
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_vectors; ++j)
            {
               tempx = 0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] += tempx;
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
         for (i = 0; i < num_rows*num_vectors; i++)
            y_data[i] *= alpha;
      }
   }
   else
   { // JSP: this is currently the only path optimized
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,tempx)
#endif
      {
      HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
      HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
      hypre_assert(iBegin <= iEnd);
      hypre_assert(iBegin >= 0 && iBegin <= num_rows);
      hypre_assert(iEnd >= 0 && iEnd <= num_rows);

      if (0 == temp)
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*A*x
      } // temp == 0
      else if (-1 == temp) // beta == -alpha
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x - y
         else if (-1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x + y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x - y)
      } // temp == -1
      else if (1 == temp)
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x + y
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x - y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x + y)
      }
      else
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x + temp*y
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x - temp*y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x + temp*y)
      } // temp != 0 && temp != -1 && temp != 1
      } // omp parallel
   }

   if (x == y) hypre_SeqVectorDestroy(x_tmp);

#endif /* CPU */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

   return ierr;
}

HYPRE_Int
hypre_CSRMatrixMatvec( HYPRE_Complex    alpha,
                       hypre_CSRMatrix *A,
                       hypre_Vector    *x,
                       HYPRE_Complex    beta,
                       hypre_Vector    *y     )
{
   return hypre_CSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y, 0);
}

#if defined (HYPRE_USING_UNIFIED_MEMORY)
HYPRE_Int
hypre_CSRMatrixMatvec3( HYPRE_Complex    alpha,
                       hypre_CSRMatrix *A,
                       hypre_Vector    *x,
                       HYPRE_Complex    beta,
                       hypre_Vector    *y     )
{
   return hypre_CSRMatrixMatvecOutOfPlaceOOMP3(alpha, A, x, beta, y, y, 0);
}
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvecT( HYPRE_Complex    alpha,
                        hypre_CSRMatrix *A,
                        hypre_Vector    *x,
                        HYPRE_Complex    beta,
                        hypre_Vector    *y     )
{
   HYPRE_Complex    *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp;

   HYPRE_Complex    *y_data_expand;
   HYPRE_Int         my_thread_num = 0, offset = 0;

   HYPRE_Int         i, j, jv, jj;
   HYPRE_Int         num_threads;

   HYPRE_Int         ierr  = 0;

   hypre_Vector     *x_tmp = NULL;

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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
         y_data[i] *= beta;

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
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
         for (i = 0; i < num_cols*num_vectors; i++)
            y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
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
      y_data_expand = hypre_CTAlloc(HYPRE_Complex,  num_threads*y_size, HYPRE_MEMORY_HOST);

      if ( num_vectors==1 )
      {

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,j,my_thread_num,offset)
#endif
         {
            my_thread_num = hypre_GetThreadNum();
            offset =  y_size*my_thread_num;
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data_expand[offset + j] += A_data[jj] * x_data[i];
               }
            }

            /* implied barrier (for threads)*/
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
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

      hypre_TFree(y_data_expand, HYPRE_MEMORY_HOST);

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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
         y_data[i] *= alpha;
   }

   if (x == y) hypre_SeqVectorDestroy(x_tmp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvec_FF( HYPRE_Complex    alpha,
                          hypre_CSRMatrix *A,
                          hypre_Vector    *x,
                          HYPRE_Complex    beta,
                          hypre_Vector    *y,
                          HYPRE_Int       *CF_marker_x,
                          HYPRE_Int       *CF_marker_y,
                          HYPRE_Int        fpt )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);

   HYPRE_Complex      temp;

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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj) HYPRE_SMP_SCHEDULE
#endif

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
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
   }

   return ierr;
}
#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
HYPRE_Int
hypre_CSRMatrixMatvecDevice( HYPRE_Complex    alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector    *x,
                             HYPRE_Complex    beta,
                             hypre_Vector    *b,
                             hypre_Vector    *y,
                             HYPRE_Int        offset )
{

  static cusparseHandle_t handle;
  static cusparseMatDescr_t descr;
  static HYPRE_Int FirstCall=1;
  cusparseStatus_t status;
  static cudaStream_t s[10];
  static HYPRE_Int myid;

  if (b!=y){

    PUSH_RANGE_PAYLOAD("MEMCPY",1,y->size-offset);
    VecCopy(y->data,b->data,(y->size-offset),HYPRE_STREAM(4));
    POP_RANGE
  }

  if (x==y) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");

  if (FirstCall){
    PUSH_RANGE("FIRST_CALL",4);

    handle=getCusparseHandle();

    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: Matrix descriptor initialization failed\n");
      return hypre_error_flag;
    }

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    FirstCall=0;
    hypre_int jj;
    for(jj=0;jj<5;jj++)
      s[jj]=HYPRE_STREAM(jj);
    nvtxNameCudaStreamA(s[4], "HYPRE_COMPUTE_STREAM");
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
    myid++;
    POP_RANGE;
  }

  PUSH_RANGE("PREFETCH+SPMV",2);

  hypre_CSRMatrixPrefetchToDevice(A);
  hypre_SeqVectorPrefetchToDevice(x);
  hypre_SeqVectorPrefetchToDevice(y);

  //if (offset!=0) hypre_printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: \n");
  cusparseErrchk(cusparseDcsrmv(handle ,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 A->num_rows-offset, A->num_cols, A->num_nonzeros,
                 &alpha, descr,
                 A->data ,A->i+offset,A->j,
                 x->data, &beta, y->data+offset));

  if (!GetAsyncMode()){
  hypre_CheckErrorDevice(cudaStreamSynchronize(s[4]));
  }
  POP_RANGE;

  return 0;

}
#endif

