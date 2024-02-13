/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
 * hypre_CSRMatrixAddFirstPass:
 *
 * Performs the first pass needed for Matrix/Matrix addition (C = A + B).
 * This function:
 *    1) Computes the row pointer of the resulting matrix C_i
 *    2) Allocates memory for the matrix C and returns it to the user
 *
 * Notes: 1) It can be used safely inside OpenMP parallel regions.
 *        2) firstrow, lastrow and marker are private variables.
 *        3) The remaining arguments are shared variables.
 *        4) twspace (thread workspace) must be allocated outside the
 *           parallel region.
 *        5) The mapping arrays map_A2C and map_B2C are used when adding
 *           off-diagonal matrices. They can be set to NULL pointer when
 *           adding diagonal matrices.
 *        6) Assumes that the elements of C_i are initialized to zero.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CSRMatrixAddFirstPass( HYPRE_Int              firstrow,
                             HYPRE_Int              lastrow,
                             HYPRE_Int             *twspace,
                             HYPRE_Int             *marker,
                             HYPRE_Int             *map_A2C,
                             HYPRE_Int             *map_B2C,
                             hypre_CSRMatrix       *A,
                             hypre_CSRMatrix       *B,
                             HYPRE_Int              nrows_C,
                             HYPRE_Int              nnzrows_C,
                             HYPRE_Int              ncols_C,
                             HYPRE_Int             *rownnz_C,
                             HYPRE_MemoryLocation   memory_location_C,
                             HYPRE_Int             *C_i,
                             hypre_CSRMatrix      **C_ptr )
{
   HYPRE_Int   *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int   *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int   *B_i = hypre_CSRMatrixI(B);
   HYPRE_Int   *B_j = hypre_CSRMatrixJ(B);

   HYPRE_Int    i, ia, ib, ic, iic, ii, i1;
   HYPRE_Int    jcol, jj;
   HYPRE_Int    num_threads = hypre_NumActiveThreads();
   HYPRE_Int    num_nonzeros;

   /* Initialize marker array */
   for (i = 0; i < ncols_C; i++)
   {
      marker[i] = -1;
   }

   ii = hypre_GetThreadNum();
   num_nonzeros = 0;
   for (ic = firstrow; ic < lastrow; ic++)
   {
      iic = rownnz_C ? rownnz_C[ic] : ic;

      if (map_A2C)
      {
         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = map_A2C[A_j[ia]];
            marker[jcol] = iic;
            num_nonzeros++;
         }
      }
      else
      {
         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = A_j[ia];
            marker[jcol] = iic;
            num_nonzeros++;
         }
      }

      if (map_B2C)
      {
         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = map_B2C[B_j[ib]];
            if (marker[jcol] != iic)
            {
               marker[jcol] = iic;
               num_nonzeros++;
            }
         }
      }
      else
      {
         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] != iic)
            {
               marker[jcol] = iic;
               num_nonzeros++;
            }
         }
      }
      C_i[iic + 1] = num_nonzeros;
   }
   twspace[ii] = num_nonzeros;

#ifdef HYPRE_USING_OPENMP
   #pragma omp barrier
#endif

   /* Correct C_i - phase 1 */
   if (ii)
   {
      jj = twspace[0];
      for (i1 = 1; i1 < ii; i1++)
      {
         jj += twspace[i1];
      }

      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;
         C_i[iic + 1] += jj;
      }
   }
   else
   {
      num_nonzeros = 0;
      for (i1 = 0; i1 < num_threads; i1++)
      {
         num_nonzeros += twspace[i1];
      }

      *C_ptr = hypre_CSRMatrixCreate(nrows_C, ncols_C, num_nonzeros);
      hypre_CSRMatrixI(*C_ptr) = C_i;
      hypre_CSRMatrixRownnz(*C_ptr) = rownnz_C;
      hypre_CSRMatrixNumRownnz(*C_ptr) = nnzrows_C;
      hypre_CSRMatrixInitialize_v2(*C_ptr, 0, memory_location_C);
   }

   /* Correct C_i - phase 2 */
   if (rownnz_C != NULL)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      for (ic = firstrow; ic < (lastrow - 1); ic++)
      {
         for (iic = rownnz_C[ic] + 1; iic < rownnz_C[ic + 1]; iic++)
         {
            hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[ic] + 1];
         }
      }

      if (ii < (num_threads - 1))
      {
         for (iic = rownnz_C[lastrow - 1] + 1; iic < rownnz_C[lastrow]; iic++)
         {
            hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[lastrow - 1] + 1];
         }
      }
      else
      {
         for (iic = rownnz_C[lastrow - 1] + 1; iic < nrows_C; iic++)
         {
            hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[lastrow - 1] + 1];
         }
      }
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp barrier
#endif

#ifdef HYPRE_DEBUG
   if (!ii)
   {
      for (i = 0; i < nrows_C; i++)
      {
         hypre_assert(C_i[i] <= C_i[i + 1]);
         hypre_assert(((A_i[i + 1] - A_i[i]) +
                       (B_i[i + 1] - B_i[i])) >=
                      (C_i[i + 1] - C_i[i]));
         hypre_assert((C_i[i + 1] - C_i[i]) >= (A_i[i + 1] - A_i[i]));
         hypre_assert((C_i[i + 1] - C_i[i]) >= (B_i[i + 1] - B_i[i]));
      }
      hypre_assert((C_i[nrows_C] - C_i[0]) == num_nonzeros);
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAddSecondPass:
 *
 * Performs the second pass needed for Matrix/Matrix addition (C = A + B).
 * This function computes C_j and C_data.
 *
 * Notes: see notes for hypre_CSRMatrixAddFirstPass
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_CSRMatrixAddSecondPass( HYPRE_Int          firstrow,
                              HYPRE_Int          lastrow,
                              HYPRE_Int         *marker,
                              HYPRE_Int         *map_A2C,
                              HYPRE_Int         *map_B2C,
                              HYPRE_Int         *rownnz_C,
                              HYPRE_Complex      alpha,
                              HYPRE_Complex      beta,
                              hypre_CSRMatrix   *A,
                              hypre_CSRMatrix   *B,
                              hypre_CSRMatrix   *C )
{
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int         nnzs_A   = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int         nnzs_B   = hypre_CSRMatrixNumNonzeros(B);

   HYPRE_Int        *C_i      = hypre_CSRMatrixI(C);
   HYPRE_Int        *C_j      = hypre_CSRMatrixJ(C);
   HYPRE_Complex    *C_data   = hypre_CSRMatrixData(C);
   HYPRE_Int         ncols_C  = hypre_CSRMatrixNumCols(C);

   HYPRE_Int         ia, ib, ic, iic;
   HYPRE_Int         jcol, pos;

   hypre_assert(( map_A2C &&  map_B2C) ||
                (!map_A2C && !map_B2C) ||
                ( map_A2C && (nnzs_B == 0)) ||
                ( map_B2C && (nnzs_A == 0)));

   /* Initialize marker vector */
   for (ia = 0; ia < ncols_C; ia++)
   {
      marker[ia] = -1;
   }

   pos = C_i[rownnz_C ? rownnz_C[firstrow] : firstrow];
   if ((map_A2C && map_B2C) || ( map_A2C && (nnzs_B == 0)) || ( map_B2C && (nnzs_A == 0)))
   {
      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = map_A2C[A_j[ia]];
            C_j[pos] = jcol;
            C_data[pos] = alpha * A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = map_B2C[B_j[ib]];
            if (marker[jcol] < C_i[iic])
            {
               C_j[pos] = jcol;
               C_data[pos] = beta * B_data[ib];
               marker[jcol] = pos;
               pos++;
            }
            else
            {
               hypre_assert(C_j[marker[jcol]] == jcol);
               C_data[marker[jcol]] += beta * B_data[ib];
            }
         }
         hypre_assert(pos == C_i[iic + 1]);
      } /* end for loop */
   }
   else
   {
      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = alpha * A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] < C_i[iic])
            {
               C_j[pos] = jcol;
               C_data[pos] = beta * B_data[ib];
               marker[jcol] = pos;
               pos++;
            }
            else
            {
               hypre_assert(C_j[marker[jcol]] == jcol);
               C_data[marker[jcol]] += beta * B_data[ib];
            }
         }
         hypre_assert(pos == C_i[iic + 1]);
      } /* end for loop */
   }
   hypre_assert(pos == C_i[rownnz_C ? rownnz_C[lastrow - 1] + 1 : lastrow]);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 *
 * Adds two CSR Matrices A and B and returns a CSR Matrix C = alpha*A + beta*B;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixAddHost ( HYPRE_Complex    alpha,
                         hypre_CSRMatrix *A,
                         HYPRE_Complex    beta,
                         hypre_CSRMatrix *B )
{
   /* CSRMatrix A */
   HYPRE_Int        *rownnz_A  = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         nrows_A   = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         nnzrows_A = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int         ncols_A   = hypre_CSRMatrixNumCols(A);

   /* CSRMatrix B */
   HYPRE_Int        *rownnz_B  = hypre_CSRMatrixRownnz(B);
   HYPRE_Int         nrows_B   = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         nnzrows_B = hypre_CSRMatrixNumRownnz(B);
   HYPRE_Int         ncols_B   = hypre_CSRMatrixNumCols(B);

   /* CSRMatrix C */
   hypre_CSRMatrix  *C;
   HYPRE_Int        *C_i;
   HYPRE_Int        *rownnz_C;
   HYPRE_Int         nnzrows_C;

   HYPRE_Int        *twspace;

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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   /* Allocate memory */
   twspace = hypre_TAlloc(HYPRE_Int, hypre_NumThreads(), HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A + 1, memory_location_C);

   /* Set nonzero rows data of diag_C */
   nnzrows_C = nrows_A;
   if ((nnzrows_A < nrows_A) && (nnzrows_B < nrows_B))
   {
      hypre_IntArray arr_A;
      hypre_IntArray arr_B;
      hypre_IntArray arr_C;

      hypre_IntArrayData(&arr_A) = rownnz_A;
      hypre_IntArrayData(&arr_B) = rownnz_B;
      hypre_IntArraySize(&arr_A) = nnzrows_A;
      hypre_IntArraySize(&arr_B) = nnzrows_B;
      hypre_IntArrayMemoryLocation(&arr_C) = memory_location_C;

      hypre_IntArrayMergeOrdered(&arr_A, &arr_B, &arr_C);

      nnzrows_C = hypre_IntArraySize(&arr_C);
      rownnz_C  = hypre_IntArrayData(&arr_C);
   }
   else
   {
      rownnz_C = NULL;
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int   ns, ne;
      HYPRE_Int  *marker = NULL;

      hypre_partition1D(nnzrows_C, hypre_NumActiveThreads(), hypre_GetThreadNum(), &ns, &ne);

      marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);

      hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker, NULL, NULL,
                                  A, B, nrows_A, nnzrows_C, ncols_A, rownnz_C,
                                  memory_location_C, C_i, &C);

      hypre_CSRMatrixAddSecondPass(ns, ne, marker, NULL, NULL,
                                   rownnz_C, alpha, beta, A, B, C);

      hypre_TFree(marker, HYPRE_MEMORY_HOST);
   } /* end of parallel region */

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixAdd( HYPRE_Complex    alpha,
                    hypre_CSRMatrix *A,
                    HYPRE_Complex    beta,
                    hypre_CSRMatrix *B)
{
   hypre_CSRMatrix *C = NULL;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_CSRMatrixMemoryLocation(A),
                                                      hypre_CSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_CSRMatrixAddDevice(alpha, A, beta, B);
   }
   else
#endif
   {
      C = hypre_CSRMatrixAddHost(alpha, A, beta, B);
   }

   return C;
}

#if 0
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigAdd:
 *
 * RL: comment it out which was used in ams.c. Should be combined with
 *     above hypre_CSRMatrixAddHost whenever it is needed again
 *
 * Adds two CSR Matrices A and B with column indices stored as HYPRE_BigInt
 * and returns a CSR Matrix C;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
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
   HYPRE_Int        *twspace;

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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   /* Allocate memory */
   twspace = hypre_TAlloc(HYPRE_Int, hypre_NumThreads(), HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A + 1, memory_location_C);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int     ia, ib, ic, num_nonzeros;
      HYPRE_Int     ns, ne, pos;
      HYPRE_BigInt  jcol;
      HYPRE_Int     ii, num_threads;
      HYPRE_Int     jj;
      HYPRE_Int    *marker = NULL;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(nrows_A, num_threads, ii, &ns, &ne);

      marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);
      for (ia = 0; ia < ncols_A; ia++)
      {
         marker[ia] = -1;
      }

      /* First pass */
      num_nonzeros = 0;
      for (ic = ns; ic < ne; ic++)
      {
         C_i[ic] = num_nonzeros;
         for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
         {
            jcol = A_j[ia];
            marker[jcol] = ic;
            num_nonzeros++;
         }

         for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] != ic)
            {
               marker[jcol] = ic;
               num_nonzeros++;
            }
         }
         C_i[ic + 1] = num_nonzeros;
      }
      twspace[ii] = num_nonzeros;

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct row pointer */
      if (ii)
      {
         jj = twspace[0];
         for (ic = 1; ic < ii; ic++)
         {
            jj += twspace[ia];
         }

         for (ic = ns; ic < ne; ic++)
         {
            C_i[ic] += jj;
         }
      }
      else
      {
         C_i[nrows_A] = 0;
         for (ic = 0; ic < num_threads; ic++)
         {
            C_i[nrows_A] += twspace[ic];
         }

         C = hypre_CSRMatrixCreate(nrows_A, ncols_A, C_i[nrows_A]);
         hypre_CSRMatrixI(C) = C_i;
         hypre_CSRMatrixInitialize_v2(C, 1, memory_location_C);
         C_j = hypre_CSRMatrixBigJ(C);
         C_data = hypre_CSRMatrixData(C);
      }

      /* Second pass */
      for (ia = 0; ia < ncols_A; ia++)
      {
         marker[ia] = -1;
      }

      pos = C_i[ns];
      for (ic = ns; ic < ne; ic++)
      {
         for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
         {
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
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
   } /* end of parallel region */

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);

   return C;
}

#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiplyHost
 *
 * Multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyHost( hypre_CSRMatrix *A,
                             hypre_CSRMatrix *B )
{
   HYPRE_Complex        *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int            *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int            *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int            *rownnz_A  = hypre_CSRMatrixRownnz(A);
   HYPRE_Int             nrows_A   = hypre_CSRMatrixNumRows(A);
   HYPRE_Int             ncols_A   = hypre_CSRMatrixNumCols(A);
   HYPRE_Int             nnzrows_A = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int             num_nnz_A = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Complex        *B_data    = hypre_CSRMatrixData(B);
   HYPRE_Int            *B_i       = hypre_CSRMatrixI(B);
   HYPRE_Int            *B_j       = hypre_CSRMatrixJ(B);
   HYPRE_Int             nrows_B   = hypre_CSRMatrixNumRows(B);
   HYPRE_Int             ncols_B   = hypre_CSRMatrixNumCols(B);
   HYPRE_Int             num_nnz_B = hypre_CSRMatrixNumNonzeros(B);

   HYPRE_MemoryLocation  memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation  memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   hypre_CSRMatrix      *C;
   HYPRE_Complex        *C_data;
   HYPRE_Int            *C_i;
   HYPRE_Int            *C_j;

   HYPRE_Int             ia, ib, ic, ja, jb, num_nonzeros;
   HYPRE_Int             counter;
   HYPRE_Complex         a_entry, b_entry;
   HYPRE_Int             allsquare = 0;
   HYPRE_Int            *twspace;

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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   if (nrows_A == ncols_B)
   {
      allsquare = 1;
   }

   if ((num_nnz_A == 0) || (num_nnz_B == 0))
   {
      C = hypre_CSRMatrixCreate(nrows_A, ncols_B, 0);
      hypre_CSRMatrixNumRownnz(C) = 0;
      hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);

      return C;
   }

   /* Allocate memory */
   twspace = hypre_TAlloc(HYPRE_Int, hypre_NumThreads(), HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A + 1, memory_location_C);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(ia, ib, ic, ja, jb, num_nonzeros, counter, a_entry, b_entry)
#endif
   {
      HYPRE_Int  *B_marker = NULL;
      HYPRE_Int   ns, ne, ii, jj;
      HYPRE_Int   num_threads;
      HYPRE_Int   i1, iic;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(nnzrows_A, num_threads, ii, &ns, &ne);

      B_marker = hypre_CTAlloc(HYPRE_Int, ncols_B, HYPRE_MEMORY_HOST);
      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      HYPRE_ANNOTATE_REGION_BEGIN("%s", "First pass");

      /* First pass: compute sizes of C rows. */
      num_nonzeros = 0;
      for (ic = ns; ic < ne; ic++)
      {
         if (rownnz_A)
         {
            iic = rownnz_A[ic];
            C_i[iic] = num_nonzeros;
         }
         else
         {
            iic = ic;
            C_i[iic] = num_nonzeros;
            if (allsquare)
            {
               B_marker[iic] = iic;
               num_nonzeros++;
            }
         }

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
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
      twspace[ii] = num_nonzeros;

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct C_i - phase 1 */
      if (ii)
      {
         jj = twspace[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            jj += twspace[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            iic = rownnz_A ? rownnz_A[i1] : i1;
            C_i[iic] += jj;
         }
      }
      else
      {
         C_i[nrows_A] = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            C_i[nrows_A] += twspace[i1];
         }

         C = hypre_CSRMatrixCreate(nrows_A, ncols_B, C_i[nrows_A]);
         hypre_CSRMatrixI(C) = C_i;
         hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
         C_j = hypre_CSRMatrixJ(C);
         C_data = hypre_CSRMatrixData(C);
      }

      /* Correct C_i - phase 2 */
      if (rownnz_A != NULL)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         for (ic = ns; ic < (ne - 1); ic++)
         {
            for (iic = rownnz_A[ic] + 1; iic < rownnz_A[ic + 1]; iic++)
            {
               C_i[iic] = C_i[rownnz_A[ic + 1]];
            }
         }

         if (ii < (num_threads - 1))
         {
            for (iic = rownnz_A[ne - 1] + 1; iic < rownnz_A[ne]; iic++)
            {
               C_i[iic] = C_i[rownnz_A[ne]];
            }
         }
         else
         {
            for (iic = rownnz_A[ne - 1] + 1; iic < nrows_A; iic++)
            {
               C_i[iic] = C_i[nrows_A];
            }
         }
      }
      /* End of First Pass */
      HYPRE_ANNOTATE_REGION_END("%s", "First pass");

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Second pass: Fill in C_data and C_j. */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Second pass");
      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      counter = rownnz_A ? C_i[rownnz_A[ns]] : C_i[ns];
      for (ic = ns; ic < ne; ic++)
      {
         if (rownnz_A)
         {
            iic = rownnz_A[ic];
         }
         else
         {
            iic = ic;
            if (allsquare)
            {
               B_marker[ic] = counter;
               C_data[counter] = 0;
               C_j[counter] = ic;
               counter++;
            }
         }

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            ja = A_j[ia];
            a_entry = A_data[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
            {
               jb = B_j[ib];
               b_entry = B_data[ib];
               if (B_marker[jb] < C_i[iic])
               {
                  B_marker[jb] = counter;
                  C_j[B_marker[jb]] = jb;
                  C_data[B_marker[jb]] = a_entry * b_entry;
                  counter++;
               }
               else
               {
                  C_data[B_marker[jb]] += a_entry * b_entry;
               }
            }
         }
      }
      HYPRE_ANNOTATE_REGION_END("%s", "Second pass");

      /* End of Second Pass */
      hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   } /*end parallel region */

#ifdef HYPRE_DEBUG
   for (ic = 0; ic < nrows_A; ic++)
   {
      hypre_assert(C_i[ic] <= C_i[ic + 1]);
   }
#endif

   // Set rownnz and num_rownnz
   hypre_CSRMatrixSetRownnz(C);

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixMultiply( hypre_CSRMatrix *A,
                         hypre_CSRMatrix *B)
{
   hypre_CSRMatrix *C = NULL;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_CSRMatrixMemoryLocation(A),
                                                      hypre_CSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_CSRMatrixMultiplyDevice(A, B);
   }
   else
#endif
   {
      C = hypre_CSRMatrixMultiplyHost(A, B);
   }

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixDeleteZeros( hypre_CSRMatrix *A,
                            HYPRE_Real       tol )
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
   for (i = 0; i < num_nonzeros; i++)
   {
      if (hypre_cabs(A_data[i]) <= tol)
      {
         zeros++;
      }
   }

   if (zeros)
   {
      B = hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros - zeros);
      hypre_CSRMatrixInitialize(B);
      B_i = hypre_CSRMatrixI(B);
      B_j = hypre_CSRMatrixJ(B);
      B_data = hypre_CSRMatrixData(B);
      B_i[0] = 0;
      pos_A = pos_B = 0;
      for (i = 0; i < nrows_A; i++)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
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
         B_i[i + 1] = pos_B;
      }

      return B;
   }
   else
   {
      return NULL;
   }
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
static inline HYPRE_Int
transpose_idx (HYPRE_Int idx, HYPRE_Int dim1, HYPRE_Int dim2)
{
   return idx % dim1 * dim2 + idx / dim1;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTransposeHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTransposeHost(hypre_CSRMatrix  *A,
                             hypre_CSRMatrix **AT,
                             HYPRE_Int         data)

{
   HYPRE_Complex        *A_data     = hypre_CSRMatrixData(A);
   HYPRE_Int            *A_i        = hypre_CSRMatrixI(A);
   HYPRE_Int            *A_j        = hypre_CSRMatrixJ(A);
   HYPRE_Int            *rownnz_A   = hypre_CSRMatrixRownnz(A);
   HYPRE_Int             nnzrows_A  = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Int             num_rows_A = hypre_CSRMatrixNumRows(A);
   HYPRE_Int             num_cols_A = hypre_CSRMatrixNumCols(A);
   HYPRE_Int             num_nnzs_A = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_MemoryLocation  memory_location = hypre_CSRMatrixMemoryLocation(A);

   HYPRE_Complex        *AT_data;
   HYPRE_Int            *AT_j;
   HYPRE_Int             num_rows_AT;
   HYPRE_Int             num_cols_AT;
   HYPRE_Int             num_nnzs_AT;

   HYPRE_Int             max_col;
   HYPRE_Int             i, j;

   /*--------------------------------------------------------------
    * First, ascertain that num_cols and num_nonzeros has been set.
    * If not, set them.
    *--------------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (!num_nnzs_A && A_i)
   {
      num_nnzs_A = A_i[num_rows_A];
   }

   if (num_rows_A && num_nnzs_A && ! num_cols_A)
   {
      max_col = -1;
      for (i = 0; i < num_rows_A; ++i)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            if (A_j[j] > max_col)
            {
               max_col = A_j[j];
            }
         }
      }
      num_cols_A = max_col + 1;
   }

   num_rows_AT = num_cols_A;
   num_cols_AT = num_rows_A;
   num_nnzs_AT = num_nnzs_A;

   *AT = hypre_CSRMatrixCreate(num_rows_AT, num_cols_AT, num_nnzs_AT);
   hypre_CSRMatrixMemoryLocation(*AT) = memory_location;

   if (num_cols_A == 0)
   {
      // JSP: parallel counting sorting breaks down
      // when A has no columns
      hypre_CSRMatrixInitialize(*AT);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   AT_j = hypre_CTAlloc(HYPRE_Int, num_nnzs_AT, memory_location);
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data)
   {
      AT_data = hypre_CTAlloc(HYPRE_Complex, num_nnzs_AT, memory_location);
      hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Parallel count sort
    *-----------------------------------------------------------------*/
   HYPRE_Int *bucket = hypre_CTAlloc(HYPRE_Int, (num_cols_A + 1) * hypre_NumThreads(),
                                     HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int   ii, num_threads, ns, ne;
      HYPRE_Int   i, j, j0, j1, ir;
      HYPRE_Int   idx, offset;
      HYPRE_Int   transpose_i;
      HYPRE_Int   transpose_i_minus_1;
      HYPRE_Int   transpose_i0;
      HYPRE_Int   transpose_j0;
      HYPRE_Int   transpose_j1;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(nnzrows_A, num_threads, ii, &ns, &ne);

      /*-----------------------------------------------------------------
       * Count the number of entries that will go into each bucket
       * bucket is used as HYPRE_Int[num_threads][num_colsA] 2D array
       *-----------------------------------------------------------------*/
      if (rownnz_A == NULL)
      {
         for (j = A_i[ns]; j < A_i[ne]; ++j)
         {
            bucket[ii * num_cols_A + A_j[j]]++;
         }
      }
      else
      {
         for (i = ns; i < ne; i++)
         {
            ir = rownnz_A[i];
            for (j = A_i[ir]; j < A_i[ir + 1]; ++j)
            {
               bucket[ii * num_cols_A + A_j[j]]++;
            }
         }
      }

      /*-----------------------------------------------------------------
       * Parallel prefix sum of bucket with length num_colsA * num_threads
       * accessed as if it is transposed as HYPRE_Int[num_colsA][num_threads]
       *-----------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = ii * num_cols_A + 1; i < (ii + 1)*num_cols_A; ++i)
      {
         transpose_i = transpose_idx(i, num_threads, num_cols_A);
         transpose_i_minus_1 = transpose_idx(i - 1, num_threads, num_cols_A);

         bucket[transpose_i] += bucket[transpose_i_minus_1];
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp master
#endif
      {
         for (i = 1; i < num_threads; ++i)
         {
            j0 = num_cols_A * i - 1;
            j1 = num_cols_A * (i + 1) - 1;
            transpose_j0 = transpose_idx(j0, num_threads, num_cols_A);
            transpose_j1 = transpose_idx(j1, num_threads, num_cols_A);

            bucket[transpose_j1] += bucket[transpose_j0];
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii > 0)
      {
         transpose_i0 = transpose_idx(num_cols_A * ii - 1, num_threads, num_cols_A);
         offset = bucket[transpose_i0];

         for (i = ii * num_cols_A; i < (ii + 1)*num_cols_A - 1; ++i)
         {
            transpose_i = transpose_idx(i, num_threads, num_cols_A);

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
         for (i = ne - 1; i >= ns; --i)
         {
            ir = rownnz_A ? rownnz_A[i] : i;
            for (j = A_i[ir + 1] - 1; j >= A_i[ir]; --j)
            {
               idx = A_j[j];
               --bucket[ii * num_cols_A + idx];

               offset = bucket[ii * num_cols_A + idx];
               AT_data[offset] = A_data[j];
               AT_j[offset] = ir;
            }
         }
      }
      else
      {
         for (i = ne - 1; i >= ns; --i)
         {
            ir = rownnz_A ? rownnz_A[i] : i;
            for (j = A_i[ir + 1] - 1; j >= A_i[ir]; --j)
            {
               idx = A_j[j];
               --bucket[ii * num_cols_A + idx];

               offset = bucket[ii * num_cols_A + idx];
               AT_j[offset] = ir;
            }
         }
      }
   } /* end parallel region */

   hypre_CSRMatrixI(*AT) = hypre_TAlloc(HYPRE_Int, num_cols_A + 1, memory_location);
   hypre_TMemcpy(hypre_CSRMatrixI(*AT), bucket, HYPRE_Int, num_cols_A + 1, memory_location,
                 HYPRE_MEMORY_HOST);
   hypre_CSRMatrixI(*AT)[num_cols_A] = num_nnzs_A;
   hypre_TFree(bucket, HYPRE_MEMORY_HOST);

   // Set rownnz and num_rownnz
   if (hypre_CSRMatrixNumRownnz(A) < num_rows_A)
   {
      hypre_CSRMatrixSetRownnz(*AT);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTranspose(hypre_CSRMatrix  *A,
                         hypre_CSRMatrix **AT,
                         HYPRE_Int         data)
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
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

   hypre_CSRMatrixSetPatternOnly(*AT, hypre_CSRMatrixPatternOnly(A));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSplit
 *--------------------------------------------------------------------------*/

/* RL: TODO add memory locations */
HYPRE_Int
hypre_CSRMatrixSplit(hypre_CSRMatrix  *Bs_ext,
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
   HYPRE_BigInt    *B_ext_offd_bigj = NULL;
   HYPRE_Complex   *B_ext_offd_data = NULL;
   HYPRE_Int       *my_diag_array;
   HYPRE_Int       *my_offd_array;
   HYPRE_BigInt    *temp = NULL;
   HYPRE_Int        max_num_threads;
   HYPRE_Int        cnt = 0;
   hypre_CSRMatrix *Bext_diag = NULL;
   hypre_CSRMatrix *Bext_offd = NULL;
   HYPRE_BigInt    *col_map_offd_C = NULL;
   HYPRE_Int        num_cols_offd_C = 0;

   B_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_Bext + 1, HYPRE_MEMORY_HOST);
   B_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_Bext + 1, HYPRE_MEMORY_HOST);

   max_num_threads = hypre_NumThreads();
   my_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);
   my_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int ns, ne, ii, num_threads;
      HYPRE_Int i1, i, j;
      HYPRE_Int my_offd_size, my_diag_size;
      HYPRE_Int cnt_offd, cnt_diag;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(num_rows_Bext, num_threads, ii, &ns, &ne);

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
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

         B_ext_diag_j    = hypre_CTAlloc(HYPRE_Int,     B_ext_diag_size, HYPRE_MEMORY_HOST);
         B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size, HYPRE_MEMORY_HOST);
         B_ext_offd_j    = hypre_CTAlloc(HYPRE_Int,     B_ext_offd_size, HYPRE_MEMORY_HOST);
         B_ext_offd_bigj = hypre_CTAlloc(HYPRE_BigInt,  B_ext_offd_size, HYPRE_MEMORY_HOST);
         B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size, HYPRE_MEMORY_HOST);
         if (B_ext_offd_size || num_cols_offd_B)
         {
            temp = hypre_CTAlloc(HYPRE_BigInt, B_ext_offd_size + num_cols_offd_B,
                                 HYPRE_MEMORY_HOST);
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            {
               temp[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_bigj[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = (HYPRE_Int) (Bs_ext_j[j] - first_col_diag_B);
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
            for (i = 0; i < num_cols_offd_B; i++)
            {
               temp[cnt++] = col_map_offd_B[i];
            }
            if (cnt)
            {
               hypre_BigQsort0(temp, 0, cnt - 1);
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
         for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
         {
            B_ext_offd_j[j] = hypre_BigBinarySearch(col_map_offd_C,
                                                    B_ext_offd_bigj[j],
                                                    num_cols_offd_C);
         }
      }
   } /* end parallel region */

   hypre_TFree(my_diag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(my_offd_array, HYPRE_MEMORY_HOST);
   hypre_TFree(B_ext_offd_bigj, HYPRE_MEMORY_HOST);

   Bext_diag = hypre_CSRMatrixCreate(num_rows_Bext,
                                     (HYPRE_Int) (last_col_diag_B - first_col_diag_B + 1),
                                     B_ext_diag_size);
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
 * hypre_CSRMatrixReorderHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixReorderHost(hypre_CSRMatrix *A)
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

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, ii, j) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nnzrows_A; i++)
   {
      ii = rownnz_A ? rownnz_A[i] : i;
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixReorder:
 *
 * Reorders the column and data arrays of a square CSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixReorder(hypre_CSRMatrix *A)
{
   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_CSRMatrixMoveDiagFirstDevice(A);
   }
   else
#endif
   {
      ierr = hypre_CSRMatrixReorderHost(A);
   }

   return ierr;
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   map = hypre_CTAlloc(HYPRE_Int, nrows_B, HYPRE_MEMORY_HOST);
   temp = hypre_CTAlloc(HYPRE_Int, nrows_B, HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_B; i++)
   {
      map[i] = i;
      temp[i] = row_nums[i];
   }

   hypre_qsort2i(temp, map, 0, nrows_B - 1);

   marker = hypre_CTAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A + 1, memory_location_C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   num_nonzeros = 0;
   C_i[0] = 0;
   cnt = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
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
               for (ib = B_i[i2]; ib < B_i[i2 + 1]; ib++)
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
      C_i[ic + 1] = num_nonzeros;
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
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
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
               for (ib = B_i[i2]; ib < B_i[i2 + 1]; ib++)
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

HYPRE_Complex
hypre_CSRMatrixSumElts( hypre_CSRMatrix *A )
{
   HYPRE_Complex  sum = 0;
   HYPRE_Complex *data = hypre_CSRMatrixData(A);
   HYPRE_Int      num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      i;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; i++)
   {
      sum += data[i];
   }

   return sum;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixFnorm
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_CSRMatrixFnorm( hypre_CSRMatrix *A )
{
   HYPRE_Int       nrows        = hypre_CSRMatrixNumRows(A);
   HYPRE_Int       num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int      *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Complex  *A_data       = hypre_CSRMatrixData(A);
   HYPRE_Int       i;
   HYPRE_Complex   sum = 0;

   hypre_assert(num_nonzeros == A_i[nrows]);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; ++i)
   {
      HYPRE_Complex v = A_data[i];
      sum += v * v;
   }

   return hypre_sqrt(sum);
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixComputeRowSumHost
 *
 * type == 0, sum,
 *         1, abs sum
 *         2, square sum
 *--------------------------------------------------------------------------*/

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

      for (j = A_i[i]; j < A_i[i + 1]; j++)
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
            row_sum_i += scal * hypre_cabs(A_data[j]);
         }
         else if (type == 2)
         {
            row_sum_i += scal * A_data[j] * A_data[j];
         }
      }

      row_sum[i] = row_sum_i;
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixComputeRowSum
 *--------------------------------------------------------------------------*/

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

#if defined(HYPRE_USING_GPU)
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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDiagonalHost
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *      4: abs diag inverse sqrt
 *--------------------------------------------------------------------------*/

HYPRE_Int
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
   char           msg[HYPRE_MAX_MSG_LEN];

   for (i = 0; i < nrows; i++)
   {
      d_i = 0.0;
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if (A_j[j] == i)
         {
            if (type == 0)
            {
               d_i = A_data[j];
            }
            else if (type == 1)
            {
               d_i = hypre_cabs(A_data[j]);
            }
            else
            {
               if (A_data[j] == 0.0)
               {
                  hypre_sprintf(msg, "Zero diagonal found at row %i!", i);
                  hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
               }
               else if (type == 2)
               {
                  d_i = 1.0 / A_data[j];
               }
               else if (type == 3)
               {
                  d_i = 1.0 / hypre_sqrt(A_data[j]);
               }
               else if (type == 4)
               {
                  d_i = 1.0 / hypre_sqrt(hypre_cabs(A_data[j]));
               }
            }
            break;
         }
      }
      d[i] = d_i;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDiagonal
 *
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDiagonal( hypre_CSRMatrix *A,
                                HYPRE_Complex   *d,
                                HYPRE_Int        type)
{
#if defined(HYPRE_USING_GPU)
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixScale
 *
 * Scales CSR matrix: A = scalar * A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixScale( hypre_CSRMatrix *A,
                      HYPRE_Complex    scalar)
{
   HYPRE_Complex *data = hypre_CSRMatrixData(A);
   HYPRE_Int      i;
   HYPRE_Int      k = hypre_CSRMatrixNumNonzeros(A);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypreDevice_ComplexScalen(data, k, data, scalar);
   }
   else
#endif
   {
      for (i = 0; i < k; i++)
      {
         data[i] *= scalar;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDiagScaleHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixDiagScaleHost( hypre_CSRMatrix *A,
                              hypre_Vector    *ld,
                              hypre_Vector    *rd)
{

   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);

   HYPRE_Complex *ldata  = ld ? hypre_VectorData(ld) : NULL;
   HYPRE_Complex *rdata  = rd ? hypre_VectorData(rd) : NULL;
   HYPRE_Int      lsize  = ld ? hypre_VectorSize(ld) : 0;
   HYPRE_Int      rsize  = rd ? hypre_VectorSize(rd) : 0;

   HYPRE_Int      i, j;
   HYPRE_Complex  sl;
   HYPRE_Complex  sr;

   if (ldata && rdata)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sl, sr) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         sl = ldata[i];
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            sr = rdata[A_j[j]];
            A_data[j] = sl * A_data[j] * sr;
         }
      }
   }
   else if (ldata && !rdata)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sl) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         sl = ldata[i];
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            A_data[j] = sl * A_data[j];
         }
      }
   }
   else if (!ldata && rdata)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sr) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            sr = rdata[A_j[j]];
            A_data[j] = A_data[j] * sr;
         }
      }
   }
   else
   {
      /* Throw an error if the scaling factors should have a size different than zero */
      if (lsize || rsize)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Scaling matrices are not set!\n");
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDiagScale
 *
 * Computes A = diag(ld) * A * diag(rd), where the diagonal matrices
 * "diag(ld)" and "diag(rd)" are stored as local vectors.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixDiagScale( hypre_CSRMatrix *A,
                          hypre_Vector    *ld,
                          hypre_Vector    *rd)
{
   /* Sanity checks */
   if (ld && hypre_VectorSize(ld) && !hypre_VectorData(ld))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ld scaling coefficients are not set\n");
      return hypre_error_flag;
   }

   if (rd && hypre_VectorSize(rd) && !hypre_VectorData(rd))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "rd scaling coefficients are not set\n");
      return hypre_error_flag;
   }

   if (!rd && !ld)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec;

   if (ld && rd)
   {
      /* TODO (VPM): replace with GetExecPolicy3 */
      exec = hypre_GetExecPolicy2(hypre_CSRMatrixMemoryLocation(A),
                                  hypre_VectorMemoryLocation(ld));
   }
   else if (ld)
   {
      exec = hypre_GetExecPolicy2(hypre_CSRMatrixMemoryLocation(A),
                                  hypre_VectorMemoryLocation(ld));
   }
   else
   {
      exec = hypre_GetExecPolicy2(hypre_CSRMatrixMemoryLocation(A),
                                  hypre_VectorMemoryLocation(rd));
   }

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_CSRMatrixDiagScaleDevice(A, ld, rd);
   }
   else
#endif
   {
      hypre_CSRMatrixDiagScaleHost(A, ld, rd);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetConstantValues( hypre_CSRMatrix *A,
                                  HYPRE_Complex    value)
{
   HYPRE_Int i;
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);

   if (!hypre_CSRMatrixData(A))
   {
      hypre_CSRMatrixData(A) = hypre_TAlloc(HYPRE_Complex, nnz, hypre_CSRMatrixMemoryLocation(A));
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypreDevice_ComplexFilln(hypre_CSRMatrixData(A), nnz, value);
   }
   else
#endif
   {
      for (i = 0; i < nnz; i++)
      {
         hypre_CSRMatrixData(A)[i] = value;
      }
   }

   return hypre_error_flag;
}
