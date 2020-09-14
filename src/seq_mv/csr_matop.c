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
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include <assert.h>

#include "seq_mv.h"
#include "csr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 *
 * Adds two CSR Matrices A and B and returns a CSR Matrix C;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixAdd( hypre_CSRMatrix *A,
                    hypre_CSRMatrix *B )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix  *C;
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, jcol, num_nonzeros;
   HYPRE_Int         pos;
   HYPRE_Int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   marker = hypre_CTAlloc(HYPRE_Int, ncols_A);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   num_nonzeros = 0;
   C_i[0] = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] != ic)
         {
            marker[jcol] = ic;
            num_nonzeros++;
         }
      }
      C_i[ic+1] = num_nonzeros;
   }

   C = hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
      marker[ia] = -1;

   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         C_data[pos] = A_data[ia];
         marker[jcol] = pos;
         pos++;
      }
      for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] < C_i[ic])
         {
            C_j[pos] = jcol;
            C_data[pos] = B_data[ib];
            marker[jcol] = pos;
            pos++;
         }
         else
         {
            C_data[marker[jcol]] += B_data[ib];
         }
      }
   }

   hypre_TFree(marker);
   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiply
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixMultiply( hypre_CSRMatrix *A,
                         hypre_CSRMatrix *B)
{
   HYPRE_Complex    *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int        *rownnz_A  = hypre_CSRMatrixRownnz(A);
   HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A   = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         nnzrows_A = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int         ncols_A   = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data    = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i       = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j       = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B   = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B   = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         num_nonzeros_B = hypre_CSRMatrixNumNonzeros(B);
   hypre_CSRMatrix  *C;
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, ja, jb, num_nonzeros=0;
   HYPRE_Int         row_start, counter;
   HYPRE_Complex     a_entry, b_entry;
   HYPRE_Int         allsquare = 0;
   HYPRE_Int         max_num_threads;
   HYPRE_Int         *jj_count;

   if (ncols_A != nrows_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   if (nrows_A == ncols_B) allsquare = 1;

   if (num_nonzeros_B == 0)
   {
      C = hypre_CSRMatrixCreate(nrows_A, ncols_B, 0);
      hypre_CSRMatrixInitialize(C);

      return C;
   }

   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A + 1);

   max_num_threads = hypre_NumThreads();
   jj_count = hypre_CTAlloc(HYPRE_Int, max_num_threads);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(ia, ib, ic, ja, jb, num_nonzeros, row_start, counter, a_entry, b_entry)
#endif
   {
      HYPRE_Int *B_marker = NULL;
      HYPRE_Int ns, ne, ii, jj;
      HYPRE_Int size, rest, num_threads;
      HYPRE_Int i1, iic;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();

      size = nnzrows_A/num_threads;
      rest = nnzrows_A - size*num_threads;
      if (ii < rest)
      {
         ns = ii*size+ii;
         ne = (ii+1)*size+ii+1;
      }
      else
      {
         ns = ii*size+rest;
         ne = (ii+1)*size+rest;
      }

      B_marker = hypre_CTAlloc(HYPRE_Int, ncols_B);

      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      /* Set B_marker */
      num_nonzeros = 0;
      if (rownnz_A == NULL)
      {
         for (ic = ns; ic < ne; ic++)
         {
            C_i[ic] = num_nonzeros;
            if (allsquare)
            {
               B_marker[ic] = ic;
               num_nonzeros++;
            }
            for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
            {
               ja = A_j[ia];
               for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
               {
                  jb = B_j[ib];
                  if (B_marker[jb] != ic)
                  {
                     B_marker[jb] = ic;
                     num_nonzeros++;
                  }
               }
            }
         }
      }
      else
      {
         if (allsquare)
         {
            for (ic = ns; ic < ne; ic++)
            {
               iic = rownnz_A[ic];
               C_i[iic] = num_nonzeros;
               B_marker[iic] = iic;
               num_nonzeros++;
               for (ia = A_i[iic]; ia < A_i[iic+1]; ia++)
               {
                  ja = A_j[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     if (B_marker[jb] != iic)
                     {
                        B_marker[jb] = iic;
                        num_nonzeros++;
                     }
                  }
               }
            }
         }
         else
         {
            for (ic = ns; ic < ne; ic++)
            {
               iic = rownnz_A[ic];
               C_i[iic] = num_nonzeros;
               for (ia = A_i[iic]; ia < A_i[iic+1]; ia++)
               {
                  ja = A_j[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     if (B_marker[jb] != iic)
                     {
                        B_marker[jb] = iic;
                        num_nonzeros++;
                     }
                  }
               }
            }
         }
      }
      jj_count[ii] = num_nonzeros;

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      if (ii)
      {
         jj = jj_count[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            jj += jj_count[i1];
         }

         if (rownnz_A == NULL)
         {
            for (i1 = ns; i1 < ne; i1++)
            {
               C_i[i1] += jj;
            }
         }
         else
         {
            for (i1 = ns; i1 < ne; i1++)
            {
               iic = rownnz_A[i1];
               C_i[iic] += jj;
            }
         }
      }

      if (rownnz_A != NULL)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
         for (ic = 0; ic < nrows_A; ic++)
         {
            if (C_i[ic+1] < C_i[ic])
            {
               C_i[ic+1] = C_i[ic];
            }
         }
      }

      if (ii == 0)
      {
         C_i[nrows_A] = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            C_i[nrows_A] += jj_count[i1];
         }

         C = hypre_CSRMatrixCreate(nrows_A, ncols_B, C_i[nrows_A]);
         hypre_CSRMatrixI(C) = C_i;
         hypre_CSRMatrixInitialize(C);
         C_j = hypre_CSRMatrixJ(C);
         C_data = hypre_CSRMatrixData(C);
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      if (rownnz_A == NULL)
      {
         counter = C_i[ns];
         if (allsquare)
         {
            for (ic = ns; ic < ne; ic++)
            {
               row_start = C_i[ic];
               B_marker[ic] = counter;
               C_data[counter] = 0;
               C_j[counter] = ic;
               counter++;
               for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
               {
                  ja = A_j[ia];
                  a_entry = A_data[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     b_entry = B_data[ib];
                     if (B_marker[jb] < row_start)
                     {
                        B_marker[jb] = counter;
                        C_j[B_marker[jb]] = jb;
                        C_data[B_marker[jb]] = a_entry*b_entry;
                        counter++;
                     }
                     else
                     {
                        C_data[B_marker[jb]] += a_entry*b_entry;
                     }
                  }
               }
            }
         }
         else
         {
            for (ic = ns; ic < ne; ic++)
            {
               row_start = C_i[ic];
               for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
               {
                  ja = A_j[ia];
                  a_entry = A_data[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     b_entry = B_data[ib];
                     if (B_marker[jb] < row_start)
                     {
                        B_marker[jb] = counter;
                        C_j[B_marker[jb]] = jb;
                        C_data[B_marker[jb]] = a_entry*b_entry;
                        counter++;
                     }
                     else
                     {
                        C_data[B_marker[jb]] += a_entry*b_entry;
                     }
                  }
               }
            }
         } /* if (allsquare) */
      }
      else
      {
         counter = C_i[rownnz_A[ns]];
         if (allsquare)
         {
            for (ic = ns; ic < ne; ic++)
            {
               iic = rownnz_A[ic];
               row_start = C_i[iic];
               B_marker[iic] = counter;
               C_data[counter] = 0;
               C_j[counter] = iic;
               counter++;
               for (ia = A_i[iic]; ia < A_i[iic+1]; ia++)
               {
                  ja = A_j[ia];
                  a_entry = A_data[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     b_entry = B_data[ib];
                     if (B_marker[jb] < row_start)
                     {
                        B_marker[jb] = counter;
                        C_j[B_marker[jb]] = jb;
                        C_data[B_marker[jb]] = a_entry*b_entry;
                        counter++;
                     }
                     else
                     {
                        C_data[B_marker[jb]] += a_entry*b_entry;
                     }
                  }
               }
            }
         }
         else
         {
            for (ic = ns; ic < ne; ic++)
            {
               iic = rownnz_A[ic];
               row_start = C_i[iic];
               for (ia = A_i[iic]; ia < A_i[iic+1]; ia++)
               {
                  ja = A_j[ia];
                  a_entry = A_data[ia];
                  for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
                  {
                     jb = B_j[ib];
                     b_entry = B_data[ib];
                     if (B_marker[jb] < row_start)
                     {
                        B_marker[jb] = counter;
                        C_j[B_marker[jb]] = jb;
                        C_data[B_marker[jb]] = a_entry*b_entry;
                        counter++;
                     }
                     else
                     {
                        C_data[B_marker[jb]] += a_entry*b_entry;
                     }
                  }
               }
            }
         } /* if (allsquare) */
      } /* if (rownnz_A == NULL) */

      hypre_TFree(B_marker);
   } /*end parallel region */

   // Set rownnz and num_rownnz
   hypre_CSRMatrixSetRownnz(C);

   // Free memory
   hypre_TFree(jj_count);

   return C;
}

hypre_CSRMatrix *
hypre_CSRMatrixDeleteZeros( hypre_CSRMatrix *A, HYPRE_Real tol)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         num_nonzeros  = hypre_CSRMatrixNumNonzeros(A);

   hypre_CSRMatrix  *B;
   HYPRE_Complex    *B_data;
   HYPRE_Int        *B_i;
   HYPRE_Int        *B_j;

   HYPRE_Int         zeros;
   HYPRE_Int         i, j;
   HYPRE_Int         pos_A, pos_B;

   zeros = 0;
   for (i=0; i < num_nonzeros; i++)
      if (hypre_cabs(A_data[i]) <= tol)
         zeros++;

   if (zeros)
   {
      B = hypre_CSRMatrixCreate(nrows_A,ncols_A,num_nonzeros-zeros);
      hypre_CSRMatrixInitialize(B);
      B_i = hypre_CSRMatrixI(B);
      B_j = hypre_CSRMatrixJ(B);
      B_data = hypre_CSRMatrixData(B);
      B_i[0] = 0;
      pos_A = 0;
      pos_B = 0;
      for (i=0; i < nrows_A; i++)
      {
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            if (hypre_cabs(A_data[j]) <= tol)
            {
               pos_A++;
            }
            else
            {
               B_data[pos_B] = A_data[pos_A];
               B_j[pos_B] = A_j[pos_A];
               pos_B++;
               pos_A++;
            }
         }
         B_i[i+1] = pos_B;
      }
      return B;
   }
   else
      return NULL;
}


/******************************************************************************
 *
 * Finds transpose of a hypre_CSRMatrix
 *
 *****************************************************************************/

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline HYPRE_Int transpose_idx(HYPRE_Int idx, HYPRE_Int dim1, HYPRE_Int dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_CSRMatrixTranspose( hypre_CSRMatrix  *A,
                                    hypre_CSRMatrix **AT,
                                    HYPRE_Int         data)

{
   HYPRE_Complex      *A_data         = hypre_CSRMatrixData(A);
   HYPRE_Int          *A_i            = hypre_CSRMatrixI(A);
   HYPRE_Int          *A_j            = hypre_CSRMatrixJ(A);
   HYPRE_Int          *rownnz_A       = hypre_CSRMatrixRownnz(A);
   HYPRE_Int           nnzrows_A      = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int           num_rows_A     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int           num_cols_A     = hypre_CSRMatrixNumCols(A);
   HYPRE_Int           num_nonzeros_A = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Complex      *AT_data;
   HYPRE_Int          *AT_i;
   HYPRE_Int          *AT_j;
   HYPRE_Int           num_rows_AT;
   HYPRE_Int           num_cols_AT;
   HYPRE_Int           num_nonzeros_AT;

   HYPRE_Int           max_col;
   HYPRE_Int           i, j;

   /*--------------------------------------------------------------
    * First, ascertain that num_cols and num_nonzeros has been set.
    * If not, set them.
    *--------------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (!num_nonzeros_A)
   {
      num_nonzeros_A = A_i[num_rows_A];
   }

   if (num_rows_A && num_nonzeros_A && ! num_cols_A)
   {
      max_col = -1;
      for (i = 0; i < num_rows_A; ++i)
      {
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            if (A_j[j] > max_col)
            {
               max_col = A_j[j];
            }
         }
      }
      num_cols_A = max_col+1;
   }

   num_rows_AT = num_cols_A;
   num_cols_AT = num_rows_A;
   num_nonzeros_AT = num_nonzeros_A;

   *AT = hypre_CSRMatrixCreate(num_rows_AT, num_cols_AT, num_nonzeros_AT);

   if (0 == num_cols_A)
   {
      // JSP: parallel counting sorting breaks down
      // when A has no columns
      hypre_CSRMatrixInitialize(*AT);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   AT_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros_AT);
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data)
   {
      AT_data = hypre_CTAlloc(HYPRE_Complex, num_nonzeros_AT);
      hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Parallel count sort
    *-----------------------------------------------------------------*/

   AT_i = hypre_CTAlloc(HYPRE_Int, (num_cols_A + 1)*hypre_NumThreads());

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
      HYPRE_Int   num_threads = hypre_NumActiveThreads();
      HYPRE_Int   ii = hypre_GetThreadNum();
      HYPRE_Int   ns, ne, size, rest;
      HYPRE_Int   offset;
      HYPRE_Int   i, j, idx;

      size = nnzrows_A/num_threads;
      rest = nnzrows_A - size*num_threads;
      if (ii < rest)
      {
         ns = ii*size + ii;
         ne = (ii + 1)*size + ii + 1;
      }
      else
      {
         ns = ii*size + rest;
         ne = (ii + 1)*size + rest;
      }

      /*-----------------------------------------------------------------
       * Count the number of entries that will go into each bucket
       * AT_i is used as HYPRE_Int[num_threads][num_cols_A] 2D array
       *-----------------------------------------------------------------*/
      if (rownnz_A == NULL)
      {
         for (j = A_i[ns]; j < A_i[ne]; ++j)
         {
            AT_i[ii*num_cols_A + A_j[j]]++;
         }
      }
      else
      {
         for (j = A_i[rownnz_A[ns]]; j < A_i[rownnz_A[ne]]; ++j)
         {
            AT_i[ii*num_cols_A + A_j[j]]++;
         }
      }

      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Prefix Sum");
      /*-----------------------------------------------------------------
       * Parallel prefix sum of AT_i with length num_cols_A * num_threads
       * accessed as if it is transposed as HYPRE_Int[num_cols_A][num_threads]
       *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
      for (i = ii*num_cols_A + 1; i < (ii + 1)*num_cols_A; ++i)
      {
         HYPRE_Int transpose_i = transpose_idx(i, num_threads, num_cols_A);
         HYPRE_Int transpose_i_minus_1 = transpose_idx(i - 1, num_threads, num_cols_A);

         AT_i[transpose_i] += AT_i[transpose_i_minus_1];
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#pragma omp master
#endif
      {
         for (i = 1; i < num_threads; ++i)
         {
            HYPRE_Int j0 = num_cols_A*i - 1, j1 = num_cols_A*(i + 1) - 1;
            HYPRE_Int transpose_j0 = transpose_idx(j0, num_threads, num_cols_A);
            HYPRE_Int transpose_j1 = transpose_idx(j1, num_threads, num_cols_A);

            AT_i[transpose_j1] += AT_i[transpose_j0];
         }
      }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      if (ii > 0)
      {
         HYPRE_Int transpose_i0 = transpose_idx(num_cols_A*ii - 1,
                                                num_threads, num_cols_A);
         HYPRE_Int offset = AT_i[transpose_i0];

         for (i = ii*num_cols_A; i < (ii + 1)*num_cols_A - 1; ++i)
         {
            HYPRE_Int transpose_i = transpose_idx(i, num_threads, num_cols_A);

            AT_i[transpose_i] += offset;
         }
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif
      HYPRE_ANNOTATE_REGION_END("%s", "Prefix Sum");

      /*----------------------------------------------------------------
       * Load the data and column numbers of AT
       *----------------------------------------------------------------*/

      if (data)
      {
         if (rownnz_A == NULL)
         {
            for (i = ne - 1; i >= ns; --i)
            {
               for (j = A_i[i + 1] - 1; j >= A_i[i]; --j)
               {
                  idx = A_j[j];
                  --AT_i[ii*num_cols_A + idx];

                  offset = AT_i[ii*num_cols_A + idx];
                  AT_data[offset] = A_data[j];
                  AT_j[offset] = i;
               }
            }
         }
         else
         {
            for (i = ne - 1; i >= ns; --i)
            {
               for (j = A_i[rownnz_A[i] + 1] - 1; j >= A_i[rownnz_A[i]]; --j)
               {
                  idx = A_j[j];
                  --AT_i[ii*num_cols_A + idx];

                  offset = AT_i[ii*num_cols_A + idx];
                  AT_data[offset] = A_data[j];
                  AT_j[offset] = rownnz_A[i];
               }
            }
         }
      }
      else
      {
         if (rownnz_A == NULL)
         {
            for (i = ne - 1; i >= ns; --i)
            {
               for (j = A_i[i + 1] - 1; j >= A_i[i]; --j)
               {
                  idx = A_j[j];
                  --AT_i[ii*num_cols_A + idx];

                  offset = AT_i[ii*num_cols_A + idx];
                  AT_j[offset] = i;
               }
            }
         }
         else
         {
            for (i = ne - 1; i >= ns; --i)
            {
               for (j = A_i[rownnz_A[i] + 1] - 1; j >= A_i[rownnz_A[i]]; --j)
               {
                  idx = A_j[j];
                  --AT_i[ii*num_cols_A + idx];

                  offset = AT_i[ii*num_cols_A + idx];
                  AT_j[offset] = rownnz_A[i];
               }
            }
         }
      }
   } /*end parallel region */

   hypre_CSRMatrixI(*AT) = hypre_TReAlloc(AT_i, HYPRE_Int, (num_cols_A + 1));
   hypre_CSRMatrixI(*AT)[num_cols_A] = num_nonzeros_A;

   // Set rownnz and num_rownnz
   if (hypre_CSRMatrixNumRownnz(A) < num_rows_A)
   {
      hypre_CSRMatrixSetRownnz(*AT);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixReorder:
 * Reorders the column and data arrays of a square CSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixReorder( hypre_CSRMatrix *A )
{
   HYPRE_Complex *A_data     = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i        = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j        = hypre_CSRMatrixJ(A);
   HYPRE_Int     *rownnz_A   = hypre_CSRMatrixRownnz(A);
   HYPRE_Int      nnzrows_A  = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int      num_rows_A = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      num_cols_A = hypre_CSRMatrixNumCols(A);

   HYPRE_Int      i, ii, j;

   /* the matrix should be square */
   if (num_rows_A != num_cols_A)
   {
      return -1;
   }

   if (rownnz_A == NULL)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, j) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows_A; i++)
      {
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            if (A_j[j] == i)
            {
               if (j != A_i[i])
               {
                  hypre_swap(A_j, A_i[i], j);
                  hypre_swap_c(A_data, A_i[i], j);
               }
               break;
            }
         }
      }
   }
   else
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i, ii, j) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nnzrows_A; i++)
      {
         ii = A_i[rownnz_A[i]];
         for (j = A_i[ii]; j < A_i[ii+1]; j++)
         {
            if (A_j[j] == ii)
            {
               if (j != A_i[ii])
               {
                  hypre_swap(A_j, A_i[ii], j);
                  hypre_swap_c(A_data, A_i[ii], j);
               }
               break;
            }
         }
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSumElts:
 * Returns the sum of all matrix elements.
 *--------------------------------------------------------------------------*/

HYPRE_Complex hypre_CSRMatrixSumElts( hypre_CSRMatrix *A )
{
   HYPRE_Complex  sum = 0;
   HYPRE_Complex *data = hypre_CSRMatrixData( A );
   HYPRE_Int      num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      i;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; ++i)
   {
      sum += data[i];
   }

   return sum;
}
