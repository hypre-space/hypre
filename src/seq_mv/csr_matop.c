/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "csr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixAddHost ( hypre_CSRMatrix *A,
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

   HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1, memory_location_C);

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
   hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }
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

   hypre_TFree(marker, HYPRE_MEMORY_HOST);

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixAdd( hypre_CSRMatrix *A,
                    hypre_CSRMatrix *B)
{
   hypre_CSRMatrix *C = NULL;

#if defined(HYPRE_USING_CUDA)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_CSRMatrixMemoryLocation(A),
                                                       hypre_CSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_CSRMatrixAddDevice(A, B);
   }
   else
#endif
   {
      C = hypre_CSRMatrixAddHost(A, B);
   }

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixBigAdd( hypre_CSRMatrix *A,
                       hypre_CSRMatrix *B )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_BigInt     *A_j      = hypre_CSRMatrixBigJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_BigInt     *B_j      = hypre_CSRMatrixBigJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix  *C;
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_BigInt     *C_j;

   HYPRE_Int         ia, ib, ic, num_nonzeros;
   HYPRE_BigInt      jcol;
   HYPRE_Int         pos;
   HYPRE_Int         *marker;

   HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1, memory_location_C);

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
   hypre_CSRMatrixInitialize_v2(C, 1, memory_location_C);
   C_j = hypre_CSRMatrixBigJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

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

   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiply
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyHost( hypre_CSRMatrix *A,
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

   HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   if (ncols_A != nrows_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   if (nrows_A == ncols_B)
   {
      allsquare = 1;
   }

   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1, memory_location_C);

   max_num_threads = hypre_NumThreads();

   jj_count = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);

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

      B_marker = hypre_CTAlloc(HYPRE_Int, ncols_B, HYPRE_MEMORY_HOST);

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
         hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
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
      hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   } /*end parallel region */
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixMultiply( hypre_CSRMatrix *A,
                         hypre_CSRMatrix *B)
{
   hypre_CSRMatrix *C = NULL;

#if defined(HYPRE_USING_CUDA)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_CSRMatrixMemoryLocation(A),
                                                       hypre_CSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_CSRMatrixMultiplyDevice(A,B);
   }
   else
#endif
   {
      C = hypre_CSRMatrixMultiplyHost(A,B);
   }

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

HYPRE_Int
hypre_CSRMatrixTransposeHost(hypre_CSRMatrix  *A,
                             hypre_CSRMatrix **AT,
                             HYPRE_Int         data)

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

   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);

   /*--------------------------------------------------------------
    * First, ascertain that num_cols and num_nonzeros has been set.
    * If not, set them.
    *--------------------------------------------------------------*/

   if (!num_nonzerosA)
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
            {
               max_col = A_j[j];
            }
         }
      }
      num_colsA = max_col+1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = hypre_CSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT);
   hypre_CSRMatrixMemoryLocation(*AT) = memory_location;

   if (0 == num_colsA)
   {
      // JSP: parallel counting sorting breaks down
      // when A has no columns
      hypre_CSRMatrixInitialize(*AT);
      return 0;
   }

   AT_j = hypre_CTAlloc(HYPRE_Int, num_nonzerosAT, memory_location);
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data)
   {
      AT_data = hypre_CTAlloc(HYPRE_Complex, num_nonzerosAT, memory_location);
      hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Parallel count sort
    *-----------------------------------------------------------------*/
   HYPRE_Int *bucket = hypre_TAlloc(HYPRE_Int, (num_colsA + 1)*hypre_NumThreads(), HYPRE_MEMORY_HOST);

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

      for (j = A_i[iBegin]; j < A_i[iEnd]; ++j)
      {
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

      for (i = my_thread_num*num_colsA + 1; i < (my_thread_num + 1)*num_colsA; ++i)
      {
         HYPRE_Int transpose_i = transpose_idx(i, num_threads, num_colsA);
         HYPRE_Int transpose_i_minus_1 = transpose_idx(i - 1, num_threads, num_colsA);

         bucket[transpose_i] += bucket[transpose_i_minus_1];
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#pragma omp master
#endif
      {
         for (i = 1; i < num_threads; ++i)
         {
            HYPRE_Int j0 = num_colsA*i - 1, j1 = num_colsA*(i + 1) - 1;
            HYPRE_Int transpose_j0 = transpose_idx(j0, num_threads, num_colsA);
            HYPRE_Int transpose_j1 = transpose_idx(j1, num_threads, num_colsA);

            bucket[transpose_j1] += bucket[transpose_j0];
         }
      }
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      if (my_thread_num > 0)
      {
         HYPRE_Int transpose_i0 = transpose_idx(num_colsA*my_thread_num - 1, num_threads, num_colsA);
         HYPRE_Int offset = bucket[transpose_i0];

         for (i = my_thread_num*num_colsA; i < (my_thread_num + 1)*num_colsA - 1; ++i)
         {
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

      if (data)
      {
         for (i = iEnd - 1; i >= iBegin; --i)
         {
            for (j = A_i[i + 1] - 1; j >= A_i[i]; --j)
            {
               HYPRE_Int idx = A_j[j];
               --bucket[my_thread_num*num_colsA + idx];

               HYPRE_Int offset = bucket[my_thread_num*num_colsA + idx];

               AT_data[offset] = A_data[j];
               AT_j[offset] = i;
            }
         }
      }
      else
      {
         for (i = iEnd - 1; i >= iBegin; --i)
         {
            for (j = A_i[i + 1] - 1; j >= A_i[i]; --j)
            {
               HYPRE_Int idx = A_j[j];
               --bucket[my_thread_num*num_colsA + idx];

               HYPRE_Int offset = bucket[my_thread_num*num_colsA + idx];

               AT_j[offset] = i;
            }
         }
      }
   } /*end parallel region */

   hypre_CSRMatrixI(*AT) = hypre_TAlloc(HYPRE_Int, num_colsA + 1, memory_location);
   hypre_TMemcpy(hypre_CSRMatrixI(*AT), bucket, HYPRE_Int, num_colsA + 1, memory_location, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixI(*AT)[num_colsA] = num_nonzerosA;

   hypre_TFree(bucket, HYPRE_MEMORY_HOST);

   return (0);
}


HYPRE_Int
hypre_CSRMatrixTranspose(hypre_CSRMatrix  *A,
                         hypre_CSRMatrix **AT,
                         HYPRE_Int         data)
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_CUDA)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_CSRMatrixTransposeDevice(A, AT, data);
   }
   else
#endif
   {
      ierr = hypre_CSRMatrixTransposeHost(A, AT, data);
   }

   return ierr;
}


/* RL: TODO add memory locations */
HYPRE_Int hypre_CSRMatrixSplit(hypre_CSRMatrix  *Bs_ext,
                               HYPRE_BigInt      first_col_diag_B,
                               HYPRE_BigInt      last_col_diag_B,
                               HYPRE_Int         num_cols_offd_B,
                               HYPRE_BigInt     *col_map_offd_B,
                               HYPRE_Int        *num_cols_offd_C_ptr,
                               HYPRE_BigInt    **col_map_offd_C_ptr,
                               hypre_CSRMatrix **Bext_diag_ptr,
                               hypre_CSRMatrix **Bext_offd_ptr)
{
   HYPRE_Complex   *Bs_ext_data = hypre_CSRMatrixData(Bs_ext);
   HYPRE_Int       *Bs_ext_i    = hypre_CSRMatrixI(Bs_ext);
   HYPRE_BigInt    *Bs_ext_j    = hypre_CSRMatrixBigJ(Bs_ext);
   HYPRE_Int        num_rows_Bext = hypre_CSRMatrixNumRows(Bs_ext);
   HYPRE_Int        B_ext_diag_size = 0;
   HYPRE_Int        B_ext_offd_size = 0;
   HYPRE_Int       *B_ext_diag_i = NULL;
   HYPRE_Int       *B_ext_diag_j = NULL;
   HYPRE_Complex   *B_ext_diag_data = NULL;
   HYPRE_Int       *B_ext_offd_i = NULL;
   HYPRE_Int       *B_ext_offd_j = NULL;
   HYPRE_Complex   *B_ext_offd_data = NULL;
   HYPRE_Int       *my_diag_array;
   HYPRE_Int       *my_offd_array;
   HYPRE_BigInt    *temp;
   HYPRE_Int        max_num_threads;
   HYPRE_Int        cnt = 0;
   hypre_CSRMatrix *Bext_diag = NULL;
   hypre_CSRMatrix *Bext_offd = NULL;
   HYPRE_BigInt    *col_map_offd_C = NULL;
   HYPRE_Int        num_cols_offd_C = 0;

   B_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_Bext+1, HYPRE_MEMORY_HOST);
   B_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_Bext+1, HYPRE_MEMORY_HOST);

   max_num_threads = hypre_NumThreads();
   my_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);
   my_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
      HYPRE_Int size, rest, ii;
      HYPRE_Int ns, ne;
      HYPRE_Int i1, i, j;
      HYPRE_Int my_offd_size, my_diag_size;
      HYPRE_Int cnt_offd, cnt_diag;
      HYPRE_Int num_threads = hypre_NumActiveThreads();

      size = num_rows_Bext/num_threads;
      rest = num_rows_Bext - size*num_threads;
      ii = hypre_GetThreadNum();
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

      my_diag_size = 0;
      my_offd_size = 0;
      for (i=ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            {
               my_offd_size++;
            }
            else
            {
               my_diag_size++;
            }
         }
      }
      my_diag_array[ii] = my_diag_size;
      my_offd_array[ii] = my_offd_size;

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      if (ii)
      {
         my_diag_size = my_diag_array[0];
         my_offd_size = my_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            my_diag_size += my_diag_array[i1];
            my_offd_size += my_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            B_ext_diag_i[i1] += my_diag_size;
            B_ext_offd_i[i1] += my_offd_size;
         }
      }
      else
      {
         B_ext_diag_size = 0;
         B_ext_offd_size = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            B_ext_diag_size += my_diag_array[i1];
            B_ext_offd_size += my_offd_array[i1];
         }
         B_ext_diag_i[num_rows_Bext] = B_ext_diag_size;
         B_ext_offd_i[num_rows_Bext] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j    = hypre_CTAlloc(HYPRE_Int,     B_ext_diag_size, HYPRE_MEMORY_HOST);
            B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size, HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size)
         {
            B_ext_offd_j    = hypre_CTAlloc(HYPRE_Int,     B_ext_offd_size, HYPRE_MEMORY_HOST);
            B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size, HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size || num_cols_offd_B)
         {
            temp = hypre_CTAlloc(HYPRE_BigInt, B_ext_offd_size + num_cols_offd_B, HYPRE_MEMORY_HOST);
         }
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            {
               temp[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

      /* This computes the mappings */
#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      if (ii == 0)
      {
         cnt = 0;
         if (B_ext_offd_size || num_cols_offd_B)
         {
            cnt = B_ext_offd_size;
            for (i=0; i < num_cols_offd_B; i++)
            {
               temp[cnt++] = col_map_offd_B[i];
            }
            if (cnt)
            {
               hypre_BigQsort0(temp, 0, cnt-1);
               num_cols_offd_C = 1;
               HYPRE_BigInt value = temp[0];
               for (i = 1; i < cnt; i++)
               {
                  if (temp[i] > value)
                  {
                     value = temp[i];
                     temp[num_cols_offd_C++] = value;
                  }
               }
            }

            if (num_cols_offd_C)
            {
               col_map_offd_C = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
            }

            for (i = 0; i < num_cols_offd_C; i++)
            {
               col_map_offd_C[i] = temp[i];
            }

            hypre_TFree(temp, HYPRE_MEMORY_HOST);
         }
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

      for (i = ns; i < ne; i++)
      {
         for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i+1]; j++)
         {
            B_ext_offd_j[j] = hypre_BigBinarySearch(col_map_offd_C, B_ext_offd_j[j], num_cols_offd_C);
         }
      }
   } /* end parallel region */

   hypre_TFree(my_diag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(my_offd_array, HYPRE_MEMORY_HOST);

   Bext_diag = hypre_CSRMatrixCreate(num_rows_Bext, last_col_diag_B-first_col_diag_B+1, B_ext_diag_size);
   hypre_CSRMatrixMemoryLocation(Bext_diag) = HYPRE_MEMORY_HOST;
   Bext_offd = hypre_CSRMatrixCreate(num_rows_Bext, num_cols_offd_C, B_ext_offd_size);
   hypre_CSRMatrixMemoryLocation(Bext_offd) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(Bext_diag)    = B_ext_diag_i;
   hypre_CSRMatrixJ(Bext_diag)    = B_ext_diag_j;
   hypre_CSRMatrixData(Bext_diag) = B_ext_diag_data;
   hypre_CSRMatrixI(Bext_offd)    = B_ext_offd_i;
   hypre_CSRMatrixJ(Bext_offd)    = B_ext_offd_j;
   hypre_CSRMatrixData(Bext_offd) = B_ext_offd_data;

   *col_map_offd_C_ptr = col_map_offd_C;
   *Bext_diag_ptr = Bext_diag;
   *Bext_offd_ptr = Bext_offd;
   *num_cols_offd_C_ptr = num_cols_offd_C;

   return hypre_error_flag;
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
 * hypre_CSRMatrixAddPartial:
 * adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/
hypre_CSRMatrix *
hypre_CSRMatrixAddPartial( hypre_CSRMatrix *A,
                           hypre_CSRMatrix *B,
                           HYPRE_Int *row_nums)
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
   HYPRE_Int         pos, i, i2, j, cnt;
   HYPRE_Int         *marker;
   HYPRE_Int         *map;
   HYPRE_Int         *temp;

   HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   if (ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   map = hypre_CTAlloc(HYPRE_Int, nrows_B, HYPRE_MEMORY_HOST);
   temp = hypre_CTAlloc(HYPRE_Int, nrows_B, HYPRE_MEMORY_HOST);
   for (i=0; i < nrows_B; i++)
   {
      map[i] = i;
      temp[i] = row_nums[i];
   }

   hypre_qsort2i(temp,map,0,nrows_B-1);

   marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1, memory_location_C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   num_nonzeros = 0;
   C_i[0] = 0;
   cnt = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      if (cnt < nrows_B && temp[cnt] == ic)
      {
         for (j = cnt; j < nrows_B; j++)
         {
            if (temp[j] == ic)
            {
               i2 = map[cnt++];
               for (ib = B_i[i2]; ib < B_i[i2+1]; ib++)
               {
                  jcol = B_j[ib];
                  if (marker[jcol] != ic)
                  {
                     marker[jcol] = ic;
                     num_nonzeros++;
                  }
               }
            }
            else
            {
               break;
            }
         }
      }
      C_i[ic+1] = num_nonzeros;
   }

   C = hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   cnt = 0;
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
      if (cnt < nrows_B && temp[cnt] == ic)
      {
         for (j = cnt; j < nrows_B; j++)
         {
            if (temp[j] == ic)
            {
               i2 = map[cnt++];
               for (ib = B_i[i2]; ib < B_i[i2+1]; ib++)
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
            else
            {
               break;
            }
         }
      }
   }

   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(map, HYPRE_MEMORY_HOST);
   hypre_TFree(temp, HYPRE_MEMORY_HOST);

   return C;
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

   for ( i = 0; i < num_nonzeros; ++i )
   {
      sum += data[i];
   }

   return sum;
}

HYPRE_Real hypre_CSRMatrixFnorm( hypre_CSRMatrix *A )
{
   HYPRE_Complex  sum = 0;
   HYPRE_Complex *data = hypre_CSRMatrixData( A );
   HYPRE_Int      num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      i, nrows, *A_i;

   nrows = hypre_CSRMatrixNumRows(A);
   A_i = hypre_CSRMatrixI(A);

   hypre_assert(num_nonzeros == A_i[nrows]);

   for ( i = 0; i < num_nonzeros; ++i )
   {
      HYPRE_Complex v = data[i];
      sum += v * v;
   }

   return sqrt(sum);
}

/* type == 0, sum,
 *         1, abs sum
 *         2, square sum
 */
void
hypre_CSRMatrixComputeRowSumHost( hypre_CSRMatrix *A,
                                  HYPRE_Int       *CF_i,
                                  HYPRE_Int       *CF_j,
                                  HYPRE_Complex   *row_sum,
                                  HYPRE_Int        type,
                                  HYPRE_Complex    scal,
                                  const char      *set_or_add)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);

   HYPRE_Int i, j;

   for (i = 0; i < nrows; i++)
   {
      HYPRE_Complex row_sum_i = set_or_add[0] == 's' ? 0.0 : row_sum[i];

      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         if (CF_i && CF_j && CF_i[i] != CF_j[A_j[j]])
         {
            continue;
         }

         if (type == 0)
         {
            row_sum_i += scal * A_data[j];
         }
         else if (type == 1)
         {
            row_sum_i += scal * fabs(A_data[j]);
         }
         else if (type == 2)
         {
            row_sum_i += scal * A_data[j] * A_data[j];
         }
      }

      row_sum[i] = row_sum_i;
   }
}

void
hypre_CSRMatrixComputeRowSum( hypre_CSRMatrix *A,
                              HYPRE_Int       *CF_i,
                              HYPRE_Int       *CF_j,
                              HYPRE_Complex   *row_sum,
                              HYPRE_Int        type,
                              HYPRE_Complex    scal,
                              const char      *set_or_add)
{
   hypre_assert( (CF_i && CF_j) || (!CF_i && !CF_j) );

#if defined(HYPRE_USING_CUDA)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_CSRMatrixComputeRowSumDevice(A, CF_i, CF_j, row_sum, type, scal, set_or_add);
   }
   else
#endif
   {
      hypre_CSRMatrixComputeRowSumHost(A, CF_i, CF_j, row_sum, type, scal, set_or_add);
   }
}

void
hypre_CSRMatrixExtractDiagonalHost( hypre_CSRMatrix *A,
                                    HYPRE_Complex   *d,
                                    HYPRE_Int        type)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   HYPRE_Int      i, j;
   HYPRE_Complex  d_i;

   for (i = 0; i < nrows; i++)
   {
      d_i = 0.0;
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         if (A_j[j] == i)
         {
            if (type == 0)
            {
               d_i = A_data[j];
            }
            else if (type == 1)
            {
               d_i = fabs(A_data[j]);
            }
            break;
         }
      }
      d[i] = d_i;
   }
}

/* type 0: diag
 *      1: abs diag
 */
void
hypre_CSRMatrixExtractDiagonal( hypre_CSRMatrix *A,
                                HYPRE_Complex   *d,
                                HYPRE_Int        type)
{
#if defined(HYPRE_USING_CUDA)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_CSRMatrixExtractDiagonalDevice(A, d, type);
   }
   else
#endif
   {
      hypre_CSRMatrixExtractDiagonalHost(A, d, type);
   }
}

