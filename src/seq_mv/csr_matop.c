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
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros 
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
      marker[ia] = -1;

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

   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1);

   max_num_threads = hypre_NumThreads();

   jj_count = hypre_CTAlloc(HYPRE_Int, max_num_threads);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(ia, ib, ic, ja, jb, num_nonzeros, row_start, counter, a_entry, b_entry)
#endif
   {
    HYPRE_Int *B_marker = NULL;
    HYPRE_Int ns, ne, ii, jj;
    HYPRE_Int size, rest, num_threads;
    HYPRE_Int i1;
    ii = hypre_GetThreadNum();
    num_threads = hypre_NumActiveThreads();

   size = nrows_A/num_threads;
   rest = nrows_A - size*num_threads;
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
      B_marker[ib] = -1;

    num_nonzeros = 0;
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
   jj_count[ii] = num_nonzeros;

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

    if (ii)
    {
       jj = jj_count[0];
       for (i1 = 1; i1 < ii; i1++)
          jj += jj_count[i1];

       for (i1 = ns; i1 < ne; i1++)
          C_i[i1] += jj;
    }
    else
    {
       C_i[nrows_A] = 0;
       for (i1 = 0; i1 < num_threads; i1++)
          C_i[nrows_A] += jj_count[i1];

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
      B_marker[ib] = -1;

   counter = C_i[ns];
   for (ic = ns; ic < ne; ic++)
   {
      row_start = C_i[ic];
      if (allsquare) 
      {
         B_marker[ic] = counter;
         C_data[counter] = 0;
         C_j[counter] = ic;
         counter++;
      }
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
		C_data[B_marker[jb]] += a_entry*b_entry;
	}
      }
   }
   hypre_TFree(B_marker);
  } /*end parallel region */
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


HYPRE_Int hypre_CSRMatrixTranspose(hypre_CSRMatrix   *A, hypre_CSRMatrix   **AT,
                                   HYPRE_Int data)

{
   HYPRE_Complex      *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int          *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int          *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int           num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int           num_colsA = hypre_CSRMatrixNumCols(A);
   HYPRE_Int           num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Complex      *AT_data;
   /*HYPRE_Int          *AT_i;*/
   HYPRE_Int          *AT_j;
   HYPRE_Int           num_rowsAT;
   HYPRE_Int           num_colsAT;
   HYPRE_Int           num_nonzerosAT;

   HYPRE_Int           max_col;
   HYPRE_Int           i, j;

   /*-------------------------------------------------------------- 
    * First, ascertain that num_cols and num_nonzeros has been set. 
    * If not, set them.
    *--------------------------------------------------------------*/

   if (! num_nonzerosA)
   {
      num_nonzerosA = A_i[num_rowsA];
   }

   if (num_rowsA && num_nonzerosA && ! num_colsA)
   {
      max_col = -1;
      for (i = 0; i < num_rowsA; ++i)
      {
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            if (A_j[j] > max_col)
               max_col = A_j[j];
         }
      }
      num_colsA = max_col+1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = hypre_CSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT);

   if (0 == num_colsA)
   {
      // JSP: parallel counting sorting breaks down
      // when A has no columns
      hypre_CSRMatrixInitialize(*AT);
      return 0;
   }

   AT_j = hypre_CTAlloc(HYPRE_Int, num_nonzerosAT);
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data) 
   {
      AT_data = hypre_CTAlloc(HYPRE_Complex, num_nonzerosAT);
      hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Parallel count sort
    *-----------------------------------------------------------------*/

   HYPRE_Int *bucket = hypre_TAlloc(
    HYPRE_Int, (num_colsA + 1)*hypre_NumThreads());

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   HYPRE_Int my_thread_num = hypre_GetThreadNum();

   HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
   HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
   hypre_assert(iBegin <= iEnd);
   hypre_assert(iBegin >= 0 && iBegin <= num_rowsA);
   hypre_assert(iEnd >= 0 && iEnd <= num_rowsA);

   HYPRE_Int i, j;
   memset(bucket + my_thread_num*num_colsA, 0, sizeof(HYPRE_Int)*num_colsA);

   /*-----------------------------------------------------------------
    * Count the number of entries that will go into each bucket
    * bucket is used as HYPRE_Int[num_threads][num_colsA] 2D array
    *-----------------------------------------------------------------*/

   for (j = A_i[iBegin]; j < A_i[iEnd]; ++j) {
     HYPRE_Int idx = A_j[j];
     bucket[my_thread_num*num_colsA + idx]++;
   }

   /*-----------------------------------------------------------------
    * Parallel prefix sum of bucket with length num_colsA * num_threads
    * accessed as if it is transposed as HYPRE_Int[num_colsA][num_threads]
    *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   for (i = my_thread_num*num_colsA + 1; i < (my_thread_num + 1)*num_colsA; ++i) {
     HYPRE_Int transpose_i = transpose_idx(i, num_threads, num_colsA);
     HYPRE_Int transpose_i_minus_1 = transpose_idx(i - 1, num_threads, num_colsA);

     bucket[transpose_i] += bucket[transpose_i_minus_1];
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#pragma omp master
#endif
   {
     for (i = 1; i < num_threads; ++i) {
       HYPRE_Int j0 = num_colsA*i - 1, j1 = num_colsA*(i + 1) - 1;
       HYPRE_Int transpose_j0 = transpose_idx(j0, num_threads, num_colsA);
       HYPRE_Int transpose_j1 = transpose_idx(j1, num_threads, num_colsA);

       bucket[transpose_j1] += bucket[transpose_j0];
     }
   }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (my_thread_num > 0) {
     HYPRE_Int transpose_i0 = transpose_idx(num_colsA*my_thread_num - 1, num_threads, num_colsA);
     HYPRE_Int offset = bucket[transpose_i0];

     for (i = my_thread_num*num_colsA; i < (my_thread_num + 1)*num_colsA - 1; ++i) {
       HYPRE_Int transpose_i = transpose_idx(i, num_threads, num_colsA);

       bucket[transpose_i] += offset;
     }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   /*----------------------------------------------------------------
    * Load the data and column numbers of AT
    *----------------------------------------------------------------*/

   if (data) {
      for (i = iEnd - 1; i >= iBegin; --i) {
        for (j = A_i[i + 1] - 1; j >= A_i[i]; --j) {
          HYPRE_Int idx = A_j[j];
          --bucket[my_thread_num*num_colsA + idx];

          HYPRE_Int offset = bucket[my_thread_num*num_colsA + idx];

          AT_data[offset] = A_data[j];
          AT_j[offset] = i;
        }
      }
   }
   else {
      for (i = iEnd - 1; i >= iBegin; --i) {
        for (j = A_i[i + 1] - 1; j >= A_i[i]; --j) {
          HYPRE_Int idx = A_j[j];
          --bucket[my_thread_num*num_colsA + idx];

          HYPRE_Int offset = bucket[my_thread_num*num_colsA + idx];

          AT_j[offset] = i;
        }
      }
   }
   } /*end parallel region */

   hypre_CSRMatrixI(*AT) = bucket; 
      // JSP: bucket is hypre_NumThreads() times longer than
      // the size needed for AT_i, but this should be OK.
      // If the memory size is a concern, we can allocate
      // a new memory for AT_i and copy from bucket.
   hypre_CSRMatrixI(*AT)[num_colsA] = num_nonzerosA;

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_CSRMatrixReorder:
 * Reorders the column and data arrays of a square CSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_CSRMatrixReorder(hypre_CSRMatrix *A)
{
   HYPRE_Int     i, j, tempi, row_size;
   HYPRE_Complex tempd;

   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int      num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      num_colsA = hypre_CSRMatrixNumCols(A);

   /* the matrix should be square */
   if (num_rowsA != num_colsA)
      return -1;

   for (i = 0; i < num_rowsA; i++)
   {
      row_size = A_i[i+1]-A_i[i];

      for (j = 0; j < row_size; j++)
      {
         if (A_j[j] == i)
         {
            if (j != 0)
            {
               tempi = A_j[0];
               A_j[0] = A_j[j];
               A_j[j] = tempi;

               tempd = A_data[0];
               A_data[0] = A_data[j];
               A_data[j] = tempd;
            }
            break;
         }

         /* diagonal element is missing */
         if (j == row_size-1)
            return -2;
      }

      A_j    += row_size;
      A_data += row_size;
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

   for ( i=0; i<num_nonzeros; ++i ) sum += data[i];

   return sum;
}
