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
                                 HYPRE_Int        offset     )
{
#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

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

   HYPRE_Int         i, ii, j, jj, k;
   HYPRE_Real        xpar=0.7;
   HYPRE_Complex     tempx;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

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
   {
      ierr = 1;
   }

   if (num_rows != y_size || num_rows != b_size)
   {
      ierr = 2;
   }

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
   {
      ierr = 3;
   }

   if ((x == y) && (alpha != 0.0))
   {
      x_tmp  = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
   }

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows*num_vectors; i++)
      {
         y_data[i] = beta*b_data[i];
      }
   }
   else if (alpha == 1.0)
   {
      if (beta == 0.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx;
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx;
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx;
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx;
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == 1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx + b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx + b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == -1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx - b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx - b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx - b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx - b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx + beta*b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx + beta*b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + beta*b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + beta*b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
   }
   else if (alpha == -1.0)
   {
      if (beta == 0.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx;
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx;
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx;
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx;
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == 1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx + b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx + b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == -1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx - b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx - b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx - b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx - b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = tempx + beta*b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx + beta*b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + beta*b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx -= A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = tempx + beta*b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
   }
   else
   {
      if (beta == 0.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = alpha*tempx;
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = alpha*tempx;
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx;
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx;
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == 1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = alpha*tempx + b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = alpha*tempx + b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx + b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx + b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else if (beta == -1.0)
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = alpha*tempx - b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = alpha*tempx - b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx - b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx - b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      }
      else
      {
         if (num_vectors == 1)
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  ii = A_rownnz[i];
                  tempx = 0;
                  for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[ii] = alpha*tempx + beta*b_data[ii];
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  tempx = 0;
                  for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = alpha*tempx + beta*b_data[i];
               }
            }
         }
         else
         {
            if (num_rownnz < xpar*(num_rows))
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     ii = A_rownnz[i];
                     k  = j*vecstride_y + ii*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[ii]; jj < A_i[ii+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx + beta*b_data[k];
                  }
               }
            }
            else
            {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,k,tempx) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; j++)
                  {
                     k = j*vecstride_y + i*idxstride_y;
                     tempx = 0;
                     for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j*vecstride_x + A_j[jj]*idxstride_x];
                     }
                     y_data[k] = alpha*tempx + beta*b_data[k];
                  }
               }
            } /* if (num_rownnz < xpar*(num_rows)) */
         } /* if (num_vectors == 1) */
      } /* beta */
   } /* alpha */

   if ((x == y) && (alpha != 0.0))
   {
      hypre_SeqVectorDestroy(x_tmp);
   }

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
      y_data_expand = hypre_CTAlloc(HYPRE_Complex, num_threads*y_size);

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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecDiagScale
 *
 * y = alpha*inv(A_D)*x + beta*y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CSRMatrixMatvecDiagScale( HYPRE_Complex     alpha,
                                hypre_CSRMatrix  *A,
                                hypre_Vector     *x,
                                HYPRE_Complex     beta,
                                hypre_Vector     *y )
{
   HYPRE_Int          num_rows    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int          num_cols    = hypre_CSRMatrixNumCols(A);
   HYPRE_Int          x_size      = hypre_VectorSize(x);
   HYPRE_Int          y_size      = hypre_VectorSize(y);
   HYPRE_Int          num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int          x_idxstride = hypre_VectorIndexStride(x);
   HYPRE_Int          x_vecstride = hypre_VectorVectorStride(x);
   HYPRE_Int          y_idxstride = hypre_VectorIndexStride(y);
   HYPRE_Int          y_vecstride = hypre_VectorVectorStride(y);
   HYPRE_Complex     *x_data      = hypre_VectorData(x);
   HYPRE_Complex     *y_data      = hypre_VectorData(y);
   HYPRE_Complex     *A_data      = hypre_CSRMatrixData(A);
   HYPRE_Int         *A_i         = hypre_CSRMatrixI(A);

   HYPRE_Int          i, j, k0, k1;

   /* Safety checks */
   if (num_vectors != hypre_VectorNumVectors(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "multivec x and y do not match");
      return hypre_error_flag;
   }

   if (num_rows != y_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "A and y do not match");
      return hypre_error_flag;
   }

   if (num_cols != x_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "A and x do not match");
      return hypre_error_flag;
   }

   if (num_rows != num_cols)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "A is not square");
      return hypre_error_flag;
   }

   /* y = alpha*inv(A_D)*x + beta*y */
   if (beta == 0.0)
   {
      if (alpha == 1.0)
      {
         if (num_vectors == 1)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = x_data[i] / A_data[A_i[i]];
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k0,k1) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               k0 = i*y_idxstride;
               k1 = i*x_idxstride;
               for (j = 0; j < num_vectors; ++j)
               {
                  y_data[j*y_vecstride + k0] = x_data[j*x_vecstride + k1] /
                                               A_data[A_i[i]];
               }
            }
         }
      }
      else
      {
         if (num_vectors == 1)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = alpha * x_data[i] / A_data[A_i[i]];
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k0,k1) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               k0 = i*y_idxstride;
               k1 = i*x_idxstride;
               for (j = 0; j < num_vectors; ++j)
               {
                  y_data[j*y_vecstride + k0] = alpha * x_data[j*x_vecstride + k1] /
                                               A_data[A_i[i]];
               }
            }
         }
      } /* if (alpha == 1.0) */
   }
   else
   {
      if (alpha == 1.0)
      {
         if (num_vectors == 1)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = x_data[i] / A_data[A_i[i]] + beta * y_data[i];
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k0,k1) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               k0 = i*y_idxstride;
               k1 = i*x_idxstride;
               for (j = 0; j < num_vectors; ++j)
               {
                  y_data[j*y_vecstride + k0] = beta * y_data[j*y_vecstride + k0] +
                                               x_data[j*x_vecstride + k1] /
                                               A_data[A_i[i]];
               }
            }
         }
      }
      else
      {
         if (num_vectors == 1)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = alpha * x_data[i] / A_data[A_i[i]] + beta * y_data[i];
            }
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k0,k1) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               k0 = i*y_idxstride;
               k1 = i*x_idxstride;
               for (j = 0; j < num_vectors; ++j)
               {
                  y_data[j*y_vecstride + k0] =  beta * y_data[j*y_vecstride + k0] +
                                               alpha * x_data[j*x_vecstride + k1] /
                                               A_data[A_i[i]];
               }
            }
         }
      } /* if (alpha == 1.0) */
   } /* if (beta == 0.0) */

   return hypre_error_flag;
}
