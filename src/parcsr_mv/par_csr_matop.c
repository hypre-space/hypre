/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_lapack.h"
#include "_hypre_blas.h"

/*--------------------------------------------------------------------------
 * hypre_ParMatmul_RowSizes:
 *
 * Computes sizes of C rows. Formerly part of hypre_ParMatmul but removed
 * so it can also be used for multiplication of Boolean matrices.
 *
 * Arrays computed: C_diag_i, C_offd_i.
 *
 * Arrays needed: (17, all HYPRE_Int*)
 *   rownnz_A,
 *   A_diag_i, A_diag_j,
 *   A_offd_i, A_offd_j,
 *   B_diag_i, B_diag_j,
 *   B_offd_i, B_offd_j,
 *   B_ext_i, B_ext_j,
 *   col_map_offd_B, col_map_offd_B,
 *   B_offd_i, B_offd_j,
 *   B_ext_i, B_ext_j.
 *
 * Scalars computed: C_diag_size, C_offd_size.
 *
 * Scalars needed:
 *   num_rownnz_A, num_rows_diag_A, num_cols_offd_A, allsquare,
 *   first_col_diag_B, num_cols_diag_B, num_cols_offd_B, num_cols_offd_C
 *--------------------------------------------------------------------------*/

void
hypre_ParMatmul_RowSizes( HYPRE_MemoryLocation memory_location,
                          HYPRE_Int **C_diag_i,
                          HYPRE_Int **C_offd_i,
                          HYPRE_Int  *rownnz_A,
                          HYPRE_Int  *A_diag_i,
                          HYPRE_Int  *A_diag_j,
                          HYPRE_Int  *A_offd_i,
                          HYPRE_Int  *A_offd_j,
                          HYPRE_Int  *B_diag_i,
                          HYPRE_Int  *B_diag_j,
                          HYPRE_Int  *B_offd_i,
                          HYPRE_Int  *B_offd_j,
                          HYPRE_Int  *B_ext_diag_i,
                          HYPRE_Int  *B_ext_diag_j,
                          HYPRE_Int  *B_ext_offd_i,
                          HYPRE_Int  *B_ext_offd_j,
                          HYPRE_Int  *map_B_to_C,
                          HYPRE_Int  *C_diag_size,
                          HYPRE_Int  *C_offd_size,
                          HYPRE_Int   num_rownnz_A,
                          HYPRE_Int   num_rows_diag_A,
                          HYPRE_Int   num_cols_offd_A,
                          HYPRE_Int   allsquare,
                          HYPRE_Int   num_cols_diag_B,
                          HYPRE_Int   num_cols_offd_B,
                          HYPRE_Int   num_cols_offd_C )
{
   HYPRE_Int *jj_count_diag_array;
   HYPRE_Int *jj_count_offd_array;

   HYPRE_Int  start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_Int  num_threads = hypre_NumThreads();

   *C_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A + 1, memory_location);
   *C_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A + 1, memory_location);

   jj_count_diag_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int  *B_marker = NULL;
      HYPRE_Int   jj_row_begin_diag, jj_count_diag;
      HYPRE_Int   jj_row_begin_offd, jj_count_offd;
      HYPRE_Int   i1, ii1, i2, i3, jj2, jj3;
      HYPRE_Int   size, rest, num_threads;
      HYPRE_Int   ii, ns, ne;

      num_threads = hypre_NumActiveThreads();
      size = num_rownnz_A / num_threads;
      rest = num_rownnz_A - size * num_threads;

      ii = hypre_GetThreadNum();
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }
      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;

      if (num_cols_diag_B || num_cols_offd_C)
      {
         B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B + num_cols_offd_C, HYPRE_MEMORY_HOST);
      }

      for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
      {
         B_marker[i1] = -1;
      }

      for (i1 = ns; i1 < ne; i1++)
      {
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         if (rownnz_A)
         {
            ii1 = rownnz_A[i1];
         }
         else
         {
            ii1 = i1;

            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            if (allsquare)
            {
               B_marker[i1] = jj_count_diag;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
            for (jj2 = A_offd_i[ii1]; jj2 < A_offd_i[ii1 + 1]; jj2++)
            {
               i2 = A_offd_j[jj2];

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + B_ext_offd_j[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
               }

               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[ii1]; jj2 < A_diag_i[ii1 + 1]; jj2++)
         {
            i2 = A_diag_j[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_diag.
             *-----------------------------------------------------------*/

            for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_diag_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{ii1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  jj_count_diag++;
               }
            }

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_offd.
             *-----------------------------------------------------------*/

            if (num_cols_offd_B)
            {
               for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
               }
            }
         }

         /*--------------------------------------------------------------------
          * Set C_diag_i and C_offd_i for this row.
          *--------------------------------------------------------------------*/

         (*C_diag_i)[ii1] = jj_row_begin_diag;
         (*C_offd_i)[ii1] = jj_row_begin_offd;
      }

      jj_count_diag_array[ii] = jj_count_diag;
      jj_count_offd_array[ii] = jj_count_offd;

      hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct diag_i and offd_i - phase 1 */
      if (ii)
      {
         jj_count_diag = jj_count_diag_array[0];
         jj_count_offd = jj_count_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            jj_count_diag += jj_count_diag_array[i1];
            jj_count_offd += jj_count_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            ii1 = rownnz_A ? rownnz_A[i1] : i1;
            (*C_diag_i)[ii1] += jj_count_diag;
            (*C_offd_i)[ii1] += jj_count_offd;
         }
      }
      else
      {
         (*C_diag_i)[num_rows_diag_A] = 0;
         (*C_offd_i)[num_rows_diag_A] = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            (*C_diag_i)[num_rows_diag_A] += jj_count_diag_array[i1];
            (*C_offd_i)[num_rows_diag_A] += jj_count_offd_array[i1];
         }
      }

      /* Correct diag_i and offd_i - phase 2 */
      if (rownnz_A != NULL)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         for (i1 = ns; i1 < (ne - 1); i1++)
         {
            for (ii1 = rownnz_A[i1] + 1; ii1 < rownnz_A[i1 + 1]; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[rownnz_A[i1 + 1]];
               (*C_offd_i)[ii1] = (*C_offd_i)[rownnz_A[i1 + 1]];
            }
         }

         if (ii < (num_threads - 1))
         {
            for (ii1 = rownnz_A[ne - 1] + 1; ii1 < rownnz_A[ne]; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[rownnz_A[ne]];
               (*C_offd_i)[ii1] = (*C_offd_i)[rownnz_A[ne]];
            }
         }
         else
         {
            for (ii1 = rownnz_A[ne - 1] + 1; ii1 < num_rows_diag_A; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[num_rows_diag_A];
               (*C_offd_i)[ii1] = (*C_offd_i)[num_rows_diag_A];
            }
         }
      }
   } /* end parallel loop */

   *C_diag_size = (*C_diag_i)[num_rows_diag_A];
   *C_offd_size = (*C_offd_i)[num_rows_diag_A];

#ifdef HYPRE_DEBUG
   HYPRE_Int i;

   for (i = 0; i < num_rows_diag_A; i++)
   {
      hypre_assert((*C_diag_i)[i] <= (*C_diag_i)[i + 1]);
      hypre_assert((*C_offd_i)[i] <= (*C_offd_i)[i + 1]);
   }
#endif

   hypre_TFree(jj_count_diag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd_array, HYPRE_MEMORY_HOST);

   /* End of First Pass */
}

/*--------------------------------------------------------------------------
 * hypre_ParMatmul:
 *
 * Multiplies two ParCSRMatrices A and B and returns the product in
 * ParCSRMatrix C.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParMatmul( hypre_ParCSRMatrix  *A,
                 hypre_ParCSRMatrix  *B )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATMUL] -= hypre_MPI_Wtime();
#endif

   /* ParCSRMatrix A */
   MPI_Comm            comm              = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt        nrows_A           = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt        ncols_A           = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt       *row_starts_A      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_rownnz_A;
   HYPRE_Int          *rownnz_A = NULL;

   /* ParCSRMatrix B */
   HYPRE_BigInt        nrows_B           = hypre_ParCSRMatrixGlobalNumRows(B);
   HYPRE_BigInt        ncols_B           = hypre_ParCSRMatrixGlobalNumCols(B);
   HYPRE_BigInt        first_col_diag_B  = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt       *col_starts_B      = hypre_ParCSRMatrixColStarts(B);
   HYPRE_BigInt        last_col_diag_B;

   /* A_diag */
   hypre_CSRMatrix    *A_diag            = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data       = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i          = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j          = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_diag_ir         = hypre_CSRMatrixRownnz(A_diag);
   HYPRE_Int           num_rownnz_diag_A = hypre_CSRMatrixNumRownnz(A_diag);
   HYPRE_Int           num_rows_diag_A   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int           num_cols_diag_A   = hypre_CSRMatrixNumCols(A_diag);

   /* A_offd */
   hypre_CSRMatrix    *A_offd            = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data       = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i          = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j          = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int          *A_offd_ir         = hypre_CSRMatrixRownnz(A_offd);
   HYPRE_Int           num_rownnz_offd_A = hypre_CSRMatrixNumRownnz(A_offd);
   HYPRE_Int           num_rows_offd_A   = hypre_CSRMatrixNumRows(A_offd);
   HYPRE_Int           num_cols_offd_A   = hypre_CSRMatrixNumCols(A_offd);

   /* B_diag */
   hypre_CSRMatrix    *B_diag            = hypre_ParCSRMatrixDiag(B);
   HYPRE_Complex      *B_diag_data       = hypre_CSRMatrixData(B_diag);
   HYPRE_Int          *B_diag_i          = hypre_CSRMatrixI(B_diag);
   HYPRE_Int          *B_diag_j          = hypre_CSRMatrixJ(B_diag);
   HYPRE_Int           num_rows_diag_B   = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int           num_cols_diag_B   = hypre_CSRMatrixNumCols(B_diag);

   /* B_offd */
   hypre_CSRMatrix    *B_offd            = hypre_ParCSRMatrixOffd(B);
   HYPRE_BigInt       *col_map_offd_B    = hypre_ParCSRMatrixColMapOffd(B);
   HYPRE_Complex      *B_offd_data       = hypre_CSRMatrixData(B_offd);
   HYPRE_Int          *B_offd_i          = hypre_CSRMatrixI(B_offd);
   HYPRE_Int          *B_offd_j          = hypre_CSRMatrixJ(B_offd);
   HYPRE_Int           num_cols_offd_B   = hypre_CSRMatrixNumCols(B_offd);

   /* ParCSRMatrix C */
   hypre_ParCSRMatrix *C;
   HYPRE_BigInt       *col_map_offd_C = NULL;
   HYPRE_Int          *map_B_to_C = NULL;

   /* C_diag */
   hypre_CSRMatrix    *C_diag;
   HYPRE_Complex      *C_diag_data;
   HYPRE_Int          *C_diag_i;
   HYPRE_Int          *C_diag_j;
   HYPRE_Int           C_offd_size;
   HYPRE_Int           num_cols_offd_C = 0;

   /* C_offd */
   hypre_CSRMatrix    *C_offd;
   HYPRE_Complex      *C_offd_data = NULL;
   HYPRE_Int          *C_offd_i = NULL;
   HYPRE_Int          *C_offd_j = NULL;
   HYPRE_Int           C_diag_size;

   /* Bs_ext */
   hypre_CSRMatrix    *Bs_ext = NULL;
   HYPRE_Complex      *Bs_ext_data = NULL;
   HYPRE_Int          *Bs_ext_i = NULL;
   HYPRE_BigInt       *Bs_ext_j = NULL;
   HYPRE_Complex      *B_ext_diag_data = NULL;
   HYPRE_Int          *B_ext_diag_i;
   HYPRE_Int          *B_ext_diag_j = NULL;
   HYPRE_Int           B_ext_diag_size;
   HYPRE_Complex      *B_ext_offd_data = NULL;
   HYPRE_Int          *B_ext_offd_i;
   HYPRE_Int          *B_ext_offd_j = NULL;
   HYPRE_BigInt       *B_big_offd_j = NULL;
   HYPRE_Int           B_ext_offd_size;

   HYPRE_Int           allsquare = 0;
   HYPRE_Int           num_procs;
   HYPRE_Int          *my_diag_array;
   HYPRE_Int          *my_offd_array;
   HYPRE_Int           max_num_threads;

   HYPRE_Complex       zero = 0.0;

   HYPRE_MemoryLocation memory_location_A = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   max_num_threads = hypre_NumThreads();
   my_diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);
   my_offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);

   if (ncols_A != nrows_B || num_cols_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");

      HYPRE_ANNOTATE_FUNC_END;
      return NULL;
   }

   /* if C=A*B is square globally and locally, then C_diag should be square also */
   if ( num_rows_diag_A == num_cols_diag_B && nrows_A == ncols_B )
   {
      allsquare = 1;
   }

   /* Set rownnz of A */
   if (num_rownnz_diag_A != num_rows_diag_A &&
       num_rownnz_offd_A != num_rows_offd_A )
   {
      hypre_IntArray arr_diag;
      hypre_IntArray arr_offd;
      hypre_IntArray arr_rownnz;

      hypre_IntArrayData(&arr_diag) = A_diag_ir;
      hypre_IntArrayData(&arr_offd) = A_offd_ir;
      hypre_IntArraySize(&arr_diag) = num_rownnz_diag_A;
      hypre_IntArraySize(&arr_offd) = num_rownnz_offd_A;
      hypre_IntArrayMemoryLocation(&arr_rownnz) = memory_location_A;

      hypre_IntArrayMergeOrdered(&arr_diag, &arr_offd, &arr_rownnz);

      num_rownnz_A = hypre_IntArraySize(&arr_rownnz);
      rownnz_A     = hypre_IntArrayData(&arr_rownnz);
   }
   else
   {
      num_rownnz_A = hypre_max(num_rows_diag_A, num_rows_offd_A);
   }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext      = hypre_ParCSRMatrixExtractBExt(B, A, 1);
      Bs_ext_data = hypre_CSRMatrixData(Bs_ext);
      Bs_ext_i    = hypre_CSRMatrixI(Bs_ext);
      Bs_ext_j    = hypre_CSRMatrixBigJ(Bs_ext);
   }
   B_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   B_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + (HYPRE_BigInt) num_cols_diag_B - 1;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_UnorderedBigIntSet set;

   #pragma omp parallel
   {
      HYPRE_Int size, rest, ii;
      HYPRE_Int ns, ne;
      HYPRE_Int i1, i, j;
      HYPRE_Int my_offd_size, my_diag_size;
      HYPRE_Int cnt_offd, cnt_diag;
      HYPRE_Int num_threads = hypre_NumActiveThreads();

      size = num_cols_offd_A / num_threads;
      rest = num_cols_offd_A - size * num_threads;
      ii = hypre_GetThreadNum();
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
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

      #pragma omp barrier

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
         B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
         B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  B_ext_diag_size, HYPRE_MEMORY_HOST);
            B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size, HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size)
         {
            B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size, HYPRE_MEMORY_HOST);
            B_big_offd_j = hypre_CTAlloc(HYPRE_BigInt, B_ext_offd_size, HYPRE_MEMORY_HOST);
            B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size, HYPRE_MEMORY_HOST);
         }
         hypre_UnorderedBigIntSetCreate(&set, B_ext_offd_size + num_cols_offd_B, 16 * hypre_NumThreads());
      }


      #pragma omp barrier

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               hypre_UnorderedBigIntSetPut(&set, Bs_ext_j[j]);
               B_big_offd_j[cnt_offd] = Bs_ext_j[j];
               //Bs_ext_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = (HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

      HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_B);
      for (i = i_begin; i < i_end; i++)
      {
         hypre_UnorderedBigIntSetPut(&set, col_map_offd_B[i]);
      }
   } /* omp parallel */

   col_map_offd_C = hypre_UnorderedBigIntSetCopyToArray(&set, &num_cols_offd_C);
   hypre_UnorderedBigIntSetDestroy(&set);
   hypre_UnorderedBigIntMap col_map_offd_C_inverse;
   hypre_big_sort_and_create_inverse_map(col_map_offd_C,
                                         num_cols_offd_C,
                                         &col_map_offd_C,
                                         &col_map_offd_C_inverse);

   HYPRE_Int i, j;
   #pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
      {
         //B_ext_offd_j[j] = hypre_UnorderedIntMapGet(&col_map_offd_C_inverse, B_ext_offd_j[j]);
         B_ext_offd_j[j] = hypre_UnorderedBigIntMapGet(&col_map_offd_C_inverse, B_big_offd_j[j]);
      }
   }

   if (num_cols_offd_C)
   {
      hypre_UnorderedBigIntMapDestroy(&col_map_offd_C_inverse);
   }

   hypre_TFree(my_diag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(my_offd_array, HYPRE_MEMORY_HOST);

   if (num_cols_offd_B)
   {
      HYPRE_Int i;
      map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

      #pragma omp parallel private(i)
      {
         HYPRE_Int i_begin, i_end;
         hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_C);

         HYPRE_Int cnt;
         if (i_end > i_begin)
         {
            cnt = hypre_BigLowerBound(col_map_offd_B,
                                      col_map_offd_B + (HYPRE_BigInt)num_cols_offd_B,
                                      col_map_offd_C[i_begin]) - col_map_offd_B;
         }

         for (i = i_begin; i < i_end && cnt < num_cols_offd_B; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
            }
         }
      }
   }
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Bs_ext);
      Bs_ext = NULL;
   }

#else /* !HYPRE_CONCURRENT_HOPSCOTCH */

   HYPRE_BigInt *temp = NULL;
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

      size = num_cols_offd_A / num_threads;
      rest = num_cols_offd_A - size * num_threads;
      ii = hypre_GetThreadNum();
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
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
         B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
         B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j = hypre_CTAlloc(HYPRE_Int, B_ext_diag_size, HYPRE_MEMORY_HOST);
            B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size, HYPRE_MEMORY_HOST);
         }

         if (B_ext_offd_size)
         {
            B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size, HYPRE_MEMORY_HOST);
            B_big_offd_j = hypre_CTAlloc(HYPRE_BigInt, B_ext_offd_size, HYPRE_MEMORY_HOST);
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
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               temp[cnt_offd] = Bs_ext_j[j];
               B_big_offd_j[cnt_offd] = Bs_ext_j[j];
               //Bs_ext_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = (HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii == 0)
      {
         HYPRE_Int cnt;

         if (num_procs > 1)
         {
            hypre_CSRMatrixDestroy(Bs_ext);
            Bs_ext = NULL;
         }

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
               HYPRE_BigInt value;

               hypre_BigQsort0(temp, 0, cnt - 1);
               num_cols_offd_C = 1;
               value = temp[0];
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
            B_ext_offd_j[j] = hypre_BigBinarySearch(col_map_offd_C, B_big_offd_j[j],
                                                    //B_ext_offd_j[j] = hypre_BigBinarySearch(col_map_offd_C, Bs_ext_j[j],
                                                    num_cols_offd_C);
         }
      }

   } /* end parallel region */
   hypre_TFree(B_big_offd_j, HYPRE_MEMORY_HOST);

   hypre_TFree(my_diag_array, HYPRE_MEMORY_HOST);
   hypre_TFree(my_offd_array, HYPRE_MEMORY_HOST);

   if (num_cols_offd_B)
   {
      HYPRE_Int i, cnt;
      map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
      {
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
      }
   }

#endif /* !HYPRE_CONCURRENT_HOPSCOTCH */

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_REGION_BEGIN("%s", "First pass");
   hypre_ParMatmul_RowSizes(memory_location_C, &C_diag_i, &C_offd_i,
                            rownnz_A, A_diag_i, A_diag_j,
                            A_offd_i, A_offd_j,
                            B_diag_i, B_diag_j,
                            B_offd_i, B_offd_j,
                            B_ext_diag_i, B_ext_diag_j,
                            B_ext_offd_i, B_ext_offd_j, map_B_to_C,
                            &C_diag_size, &C_offd_size,
                            num_rownnz_A, num_rows_diag_A, num_cols_offd_A,
                            allsquare, num_cols_diag_B, num_cols_offd_B,
                            num_cols_offd_C);
   HYPRE_ANNOTATE_REGION_END("%s", "First pass");

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_B = first_col_diag_B + (HYPRE_BigInt)num_cols_diag_B - 1;
   C_diag_data = hypre_CTAlloc(HYPRE_Complex, C_diag_size, memory_location_C);
   C_diag_j    = hypre_CTAlloc(HYPRE_Int, C_diag_size, memory_location_C);
   if (C_offd_size)
   {
      C_offd_data = hypre_CTAlloc(HYPRE_Complex, C_offd_size, memory_location_C);
      C_offd_j    = hypre_CTAlloc(HYPRE_Int, C_offd_size, memory_location_C);
   }

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Second pass");

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int     *B_marker = NULL;
      HYPRE_Int      ns, ne, size, rest, ii;
      HYPRE_Int      i1, ii1, i2, i3, jj2, jj3;
      HYPRE_Int      jj_row_begin_diag, jj_count_diag;
      HYPRE_Int      jj_row_begin_offd, jj_count_offd;
      HYPRE_Int      num_threads;
      HYPRE_Complex  a_entry; /*, a_b_product;*/

      num_threads = hypre_NumActiveThreads();
      size = num_rownnz_A / num_threads;
      rest = num_rownnz_A - size * num_threads;

      ii = hypre_GetThreadNum();
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }
      jj_count_diag = C_diag_i[rownnz_A ? rownnz_A[ns] : ns];
      jj_count_offd = C_offd_i[rownnz_A ? rownnz_A[ns] : ns];

      if (num_cols_diag_B || num_cols_offd_C)
      {
         B_marker = hypre_CTAlloc(HYPRE_Int, num_cols_diag_B + num_cols_offd_C,
                                  HYPRE_MEMORY_HOST);
         for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
         {
            B_marker[i1] = -1;
         }
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/
      for (i1 = ns; i1 < ne; i1++)
      {
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         if (rownnz_A)
         {
            ii1 = rownnz_A[i1];
         }
         else
         {
            ii1 = i1;

            /*--------------------------------------------------------------------
             *  Create diagonal entry, C_{i1,i1}
             *--------------------------------------------------------------------*/

            if (allsquare)
            {
               B_marker[i1] = jj_count_diag;
               C_diag_data[jj_count_diag] = zero;
               C_diag_j[jj_count_diag] = i1;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
            for (jj2 = A_offd_i[ii1]; jj2 < A_offd_i[ii1 + 1]; jj2++)
            {
               i2 = A_offd_j[jj2];
               a_entry = A_offd_data[jj2];

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + B_ext_offd_j[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_entry * B_ext_offd_data[jj3];
                     C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_entry * B_ext_offd_data[jj3];
                  }
               }
               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_entry * B_ext_diag_data[jj3];
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[B_marker[i3]] += a_entry * B_ext_diag_data[jj3];
                  }
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[ii1]; jj2 < A_diag_i[ii1 + 1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            a_entry = A_diag_data[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_diag.
             *-----------------------------------------------------------*/

            for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_diag_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{ii1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_data[jj_count_diag] = a_entry * B_diag_data[jj3];
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
               }
               else
               {
                  C_diag_data[B_marker[i3]] += a_entry * B_diag_data[jj3];
               }
            }
            if (num_cols_offd_B)
            {
               for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_entry * B_offd_data[jj3];
                     C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_entry * B_offd_data[jj3];
                  }
               }
            }
         }
      }

      hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   } /*end parallel region */
   HYPRE_ANNOTATE_REGION_END("%s", "Second pass");

   C = hypre_ParCSRMatrixCreate(comm, nrows_A, ncols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag_size, C_offd_size);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data;
   hypre_CSRMatrixI(C_diag)    = C_diag_i;
   hypre_CSRMatrixJ(C_diag)    = C_diag_j;
   hypre_CSRMatrixMemoryLocation(C_diag) = memory_location_C;
   hypre_CSRMatrixSetRownnz(C_diag);

   C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrixI(C_offd)  = C_offd_i;
   hypre_ParCSRMatrixOffd(C) = C_offd;
   if (num_cols_offd_C)
   {
      hypre_CSRMatrixData(C_offd)     = C_offd_data;
      hypre_CSRMatrixJ(C_offd)        = C_offd_j;
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }
   hypre_CSRMatrixMemoryLocation(C_offd) = memory_location_C;
   hypre_CSRMatrixSetRownnz(C_offd);


   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/
   hypre_TFree(B_ext_diag_i, HYPRE_MEMORY_HOST);
   if (B_ext_diag_size)
   {
      hypre_TFree(B_ext_diag_j, HYPRE_MEMORY_HOST);
      hypre_TFree(B_ext_diag_data, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(B_ext_offd_i, HYPRE_MEMORY_HOST);
   if (B_ext_offd_size)
   {
      hypre_TFree(B_ext_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(B_ext_offd_data, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_B)
   {
      hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(rownnz_A, memory_location_A);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATMUL] += hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBExt_Arrays_Overlap
 *
 * The following function was formerly part of hypre_ParCSRMatrixExtractBExt
 * but the code was removed so it can be used for a corresponding function
 * for Boolean matrices
 *
 * JSP: to allow communication overlapping, it returns comm_handle_idx and
 * comm_handle_data. Before accessing B, they should be destroyed (including
 * send_data contained in the comm_handle).
 *--------------------------------------------------------------------------*/

void hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
   HYPRE_Int ** pB_ext_i,
   HYPRE_BigInt ** pB_ext_j,
   HYPRE_Complex ** pB_ext_data,
   HYPRE_BigInt ** pB_ext_row_map,
   HYPRE_Int * num_nonzeros,
   HYPRE_Int data,
   HYPRE_Int find_row_map,
   MPI_Comm comm,
   hypre_ParCSRCommPkg * comm_pkg,
   HYPRE_Int num_cols_B,
   HYPRE_Int num_recvs,
   HYPRE_Int num_sends,
   HYPRE_BigInt first_col_diag,
   HYPRE_BigInt * row_starts,
   HYPRE_Int * recv_vec_starts,
   HYPRE_Int * send_map_starts,
   HYPRE_Int * send_map_elmts,
   HYPRE_Int * diag_i,
   HYPRE_Int * diag_j,
   HYPRE_Int * offd_i,
   HYPRE_Int * offd_j,
   HYPRE_BigInt * col_map_offd,
   HYPRE_Real * diag_data,
   HYPRE_Real * offd_data,
   hypre_ParCSRCommHandle **comm_handle_idx,
   hypre_ParCSRCommHandle **comm_handle_data,
   HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
   HYPRE_Int skip_fine, /* 1 if only coarse points are needed */
   HYPRE_Int skip_same_sign /* 1 if only points that have the same sign are needed */
   // extended based long range interpolation: skip_fine = 1, skip_same_sign = 0 for S matrix, skip_fine = 1, skip_same_sign = 1 for A matrix
   // other interpolation: skip_fine = 0, skip_same_sign = 0
)
{
   HYPRE_UNUSED_VAR(num_cols_B);

   hypre_ParCSRCommHandle *comm_handle, *row_map_comm_handle = NULL;
   hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;
   HYPRE_Int *B_int_i;
   HYPRE_BigInt *B_int_j;
   HYPRE_Int *B_ext_i;
   HYPRE_BigInt * B_ext_j;
   HYPRE_Complex * B_ext_data;
   HYPRE_Complex * B_int_data = NULL;
   HYPRE_BigInt * B_int_row_map;
   HYPRE_BigInt * B_ext_row_map;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int i, j, k;
   HYPRE_Int start_index;
   /*HYPRE_Int jrow;*/
   HYPRE_Int num_rows_B_ext;
   HYPRE_Int *prefix_sum_workspace;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_BigInt first_row_index = row_starts[0];

   num_rows_B_ext = recv_vec_starts[num_recvs];
   if ( num_rows_B_ext < 0 )    /* no B_ext, no communication */
   {
      *pB_ext_i = NULL;
      *pB_ext_j = NULL;
      if ( data ) { *pB_ext_data = NULL; }
      if ( find_row_map ) { *pB_ext_row_map = NULL; }
      *num_nonzeros = 0;
      return;
   };
   B_int_i = hypre_CTAlloc(HYPRE_Int,  send_map_starts[num_sends] + 1, HYPRE_MEMORY_HOST);
   B_ext_i = hypre_CTAlloc(HYPRE_Int,  num_rows_B_ext + 1, HYPRE_MEMORY_HOST);
   *pB_ext_i = B_ext_i;
   if ( find_row_map )
   {
      B_int_row_map = hypre_CTAlloc( HYPRE_BigInt,  send_map_starts[num_sends] + 1, HYPRE_MEMORY_HOST);
      B_ext_row_map = hypre_CTAlloc( HYPRE_BigInt,  num_rows_B_ext + 1, HYPRE_MEMORY_HOST);
      *pB_ext_row_map = B_ext_row_map;
   };

   /*--------------------------------------------------------------------------
    * generate B_int_i through adding number of row-elements of offd and diag
    * for corresponding rows. B_int_i[j+1] contains the number of elements of
    * a row j (which is determined through send_map_elmts)
    *--------------------------------------------------------------------------*/

   jdata_send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_recvs + 1, HYPRE_MEMORY_HOST);
   jdata_send_map_starts[0] = B_int_i[0] = 0;

   /*HYPRE_Int prefix_sum_workspace[(hypre_NumThreads() + 1)*num_sends];*/
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int,  (hypre_NumThreads() + 1) * num_sends,
                                       HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,k)
#endif
   {
      /*HYPRE_Int counts[num_sends];*/
      HYPRE_Int *counts;
      counts = hypre_TAlloc(HYPRE_Int,  num_sends, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         HYPRE_Int j_begin, j_end;
         hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
         j_begin += send_map_starts[i];
         j_end += send_map_starts[i];

         HYPRE_Int count = 0;
         if (skip_fine && skip_same_sign)
         {
            for (j = j_begin; j < j_end; j++)
            {
               HYPRE_Int jrow = send_map_elmts[j];
               HYPRE_Int len = 0;

               if (diag_data[diag_i[jrow]] >= 0)
               {
                  for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                  {
                     if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0) { len++; }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (offd_data[k] < 0) { len++; }
                  }
               }
               else
               {
                  for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                  {
                     if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0) { len++; }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (offd_data[k] > 0) { len++; }
                  }
               }

               B_int_i[j + 1] = len;
               count += len;
            }
         }
         else if (skip_fine)
         {
            for (j = j_begin; j < j_end; j++)
            {
               HYPRE_Int jrow = send_map_elmts[j];
               HYPRE_Int len = 0;

               for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
               {
                  if (CF_marker[diag_j[k]] >= 0) { len++; }
               }
               for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
               {
                  if (CF_marker_offd[offd_j[k]] >= 0) { len++; }
               }

               B_int_i[j + 1] = len;
               count += len;
            }
         }
         else
         {
            for (j = j_begin; j < j_end; j++)
            {
               HYPRE_Int jrow = send_map_elmts[j];
               HYPRE_Int len = diag_i[jrow + 1] - diag_i[jrow];
               len += offd_i[jrow + 1] - offd_i[jrow];
               B_int_i[j + 1] = len;
               count += len;
            }
         }

         if (find_row_map)
         {
            for (j = j_begin; j < j_end; j++)
            {
               HYPRE_Int jrow = send_map_elmts[j];
               B_int_row_map[j] = (HYPRE_BigInt)jrow + first_row_index;
            }
         }

         counts[i] = count;
      }

      hypre_prefix_sum_multiple(counts, jdata_send_map_starts + 1, num_sends, prefix_sum_workspace);

#ifdef HYPRE_USING_OPENMP
      #pragma omp master
#endif
      {
         for (i = 1; i < num_sends; i++)
         {
            jdata_send_map_starts[i + 1] += jdata_send_map_starts[i];
         }

         /*--------------------------------------------------------------------------
          * initialize communication
          *--------------------------------------------------------------------------*/

         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                    &B_int_i[1], &(B_ext_i[1]) );
         if ( find_row_map )
         {
            /* scatter/gather B_int row numbers to form array of B_ext row numbers */
            row_map_comm_handle = hypre_ParCSRCommHandleCreate
                                  (21, comm_pkg, B_int_row_map, B_ext_row_map );
         }

         B_int_j = hypre_TAlloc(HYPRE_BigInt,  jdata_send_map_starts[num_sends], HYPRE_MEMORY_HOST);
         if (data) { B_int_data = hypre_TAlloc(HYPRE_Complex,  jdata_send_map_starts[num_sends], HYPRE_MEMORY_HOST); }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = 0; i < num_sends; i++)
      {
         HYPRE_Int j_begin, j_end;
         hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
         j_begin += send_map_starts[i];
         j_end += send_map_starts[i];

         HYPRE_Int count = counts[i] + jdata_send_map_starts[i];

         if (data)
         {
            if (skip_same_sign && skip_fine)
            {
               for (j = j_begin; j < j_end; j++)
               {
                  HYPRE_Int jrow = send_map_elmts[j];
                  /*HYPRE_Int count_begin = count;*/

                  if (diag_data[diag_i[jrow]] >= 0)
                  {
                     for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                     {
                        if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0)
                        {
                           B_int_j[count] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
                           B_int_data[count] = diag_data[k];
                           count++;
                        }
                     }
                     for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                     {
                        HYPRE_Int c = offd_j[k];
                        HYPRE_BigInt c_global = col_map_offd[c];
                        if (offd_data[k] < 0)
                        {
                           B_int_j[count] = c_global;
                           B_int_data[count] = offd_data[k];
                           count++;
                        }
                     }
                  }
                  else
                  {
                     for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                     {
                        if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0)
                        {
                           B_int_j[count] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
                           B_int_data[count] = diag_data[k];
                           count++;
                        }
                     }
                     for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                     {
                        HYPRE_Int c = offd_j[k];
                        HYPRE_BigInt c_global = col_map_offd[c];
                        if (offd_data[k] > 0)
                        {
                           B_int_j[count] = c_global;
                           B_int_data[count] = offd_data[k];
                           count++;
                        }
                     }
                  }
               }
            }
            else
            {
               for (j = j_begin; j < j_end; ++j)
               {
                  HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
                     B_int_data[count] = diag_data[k];
                     count++;
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = col_map_offd[offd_j[k]];
                     B_int_data[count] = offd_data[k];
                     count++;
                  }
               }
            }
         } // data
         else
         {
            if (skip_fine)
            {
               for (j = j_begin; j < j_end; j++)
               {
                  HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     if (CF_marker[diag_j[k]] >= 0)
                     {
                        B_int_j[count] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
                        count++;
                     }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (CF_marker_offd[offd_j[k]] >= 0)
                     {
                        B_int_j[count] = col_map_offd[offd_j[k]];
                        count++;
                     }
                  }
               }
            }
            else
            {
               for (j = j_begin; j < j_end; ++j)
               {
                  HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
                     count++;
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = col_map_offd[offd_j[k]];
                     count++;
                  }
               }
            }
         } // !data
      } /* for each send target */
      hypre_TFree(counts, HYPRE_MEMORY_HOST);
   } /* omp parallel. JSP: this takes most of time in this function */
   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);

   /* Create temporary communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    jdata_recv_vec_starts,
                                    num_sends,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    jdata_send_map_starts,
                                    NULL,
                                    &tmp_comm_pkg);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * after communication exchange B_ext_i[j+1] contains the number of elements
    * of a row j !
    * evaluate B_ext_i and compute *num_nonzeros for B_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
      {
         B_ext_i[j + 1] += B_ext_i[j];
      }
   }

   *num_nonzeros = B_ext_i[num_rows_B_ext];

   *pB_ext_j = hypre_TAlloc(HYPRE_BigInt,  *num_nonzeros, HYPRE_MEMORY_HOST);
   B_ext_j = *pB_ext_j;
   if (data)
   {
      *pB_ext_data = hypre_TAlloc(HYPRE_Complex,  *num_nonzeros, HYPRE_MEMORY_HOST);
      B_ext_data = *pB_ext_data;
   }

   for (i = 0; i < num_recvs; i++)
   {
      start_index = B_ext_i[recv_vec_starts[i]];
      *num_nonzeros = B_ext_i[recv_vec_starts[i + 1]] - start_index;
      jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
   }

   *comm_handle_idx = hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
   if (data)
   {
      *comm_handle_data = hypre_ParCSRCommHandleCreate(1, tmp_comm_pkg, B_int_data,
                                                       B_ext_data);
   }

   /* Free memory */
   hypre_TFree(jdata_send_map_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(jdata_recv_vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);
   if (row_map_comm_handle)
   {
      hypre_ParCSRCommHandleDestroy(row_map_comm_handle);
      row_map_comm_handle = NULL;
   }
   if (find_row_map)
   {
      hypre_TFree(B_int_row_map, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(B_int_i, HYPRE_MEMORY_HOST);

   /* end generic part */
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBExt_Arrays
 *--------------------------------------------------------------------------*/

void hypre_ParCSRMatrixExtractBExt_Arrays(
   HYPRE_Int ** pB_ext_i,
   HYPRE_BigInt ** pB_ext_j,
   HYPRE_Complex ** pB_ext_data,
   HYPRE_BigInt ** pB_ext_row_map,
   HYPRE_Int * num_nonzeros,
   HYPRE_Int data,
   HYPRE_Int find_row_map,
   MPI_Comm comm,
   hypre_ParCSRCommPkg * comm_pkg,
   HYPRE_Int num_cols_B,
   HYPRE_Int num_recvs,
   HYPRE_Int num_sends,
   HYPRE_BigInt first_col_diag,
   HYPRE_BigInt * row_starts,
   HYPRE_Int * recv_vec_starts,
   HYPRE_Int * send_map_starts,
   HYPRE_Int * send_map_elmts,
   HYPRE_Int * diag_i,
   HYPRE_Int * diag_j,
   HYPRE_Int * offd_i,
   HYPRE_Int * offd_j,
   HYPRE_BigInt * col_map_offd,
   HYPRE_Real * diag_data,
   HYPRE_Real * offd_data
)
{
   hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
      pB_ext_i, pB_ext_j, pB_ext_data, pB_ext_row_map, num_nonzeros,
      data, find_row_map, comm, comm_pkg, num_cols_B, num_recvs, num_sends,
      first_col_diag, row_starts, recv_vec_starts, send_map_starts, send_map_elmts,
      diag_i, diag_j, offd_i, offd_j, col_map_offd, diag_data, offd_data,
      &comm_handle_idx, &comm_handle_data,
      NULL, NULL,
      0, 0);

   HYPRE_Int *send_idx = (HYPRE_Int *)comm_handle_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   hypre_TFree(send_idx, HYPRE_MEMORY_HOST);

   if (data)
   {
      HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_data->send_data;
      hypre_ParCSRCommHandleDestroy(comm_handle_data);
      hypre_TFree(send_data, HYPRE_MEMORY_HOST);
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBExt : extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_ParCSRMatrixExtractBExt_Overlap( hypre_ParCSRMatrix *B,
                                       hypre_ParCSRMatrix *A,
                                       HYPRE_Int data,
                                       hypre_ParCSRCommHandle **comm_handle_idx,
                                       hypre_ParCSRCommHandle **comm_handle_data,
                                       HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd,
                                       HYPRE_Int skip_fine, HYPRE_Int skip_same_sign )
{
   MPI_Comm  comm = hypre_ParCSRMatrixComm(B);
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   /*HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(B);*/
   HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int num_recvs;
   HYPRE_Int *recv_vec_starts;
   HYPRE_Int num_sends;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Real *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Real *offd_data = hypre_CSRMatrixData(offd);

   HYPRE_Int num_cols_B, num_nonzeros;
   HYPRE_Int num_rows_B_ext;

   hypre_CSRMatrix *B_ext;

   HYPRE_Int *B_ext_i;
   HYPRE_BigInt *B_ext_j;
   HYPRE_Complex *B_ext_data;
   HYPRE_BigInt *idummy;

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays_Overlap
   ( &B_ext_i, &B_ext_j, &B_ext_data, &idummy,
     &num_nonzeros,
     data, 0, comm, comm_pkg,
     num_cols_B, num_recvs, num_sends,
     first_col_diag, B->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data,
     comm_handle_idx, comm_handle_data,
     CF_marker, CF_marker_offd,
     skip_fine, skip_same_sign
   );

   B_ext = hypre_CSRMatrixCreate(num_rows_B_ext, num_cols_B, num_nonzeros);
   hypre_CSRMatrixMemoryLocation(B_ext) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixBigJ(B_ext) = B_ext_j;
   if (data) { hypre_CSRMatrixData(B_ext) = B_ext_data; }

   return B_ext;
}

hypre_CSRMatrix *
hypre_ParCSRMatrixExtractBExt( hypre_ParCSRMatrix *B,
                               hypre_ParCSRMatrix *A,
                               HYPRE_Int want_data )
{
#if 0
   hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   hypre_CSRMatrix *B_ext = hypre_ParCSRMatrixExtractBExt_Overlap(B, A, want_data, &comm_handle_idx,
                                                                  &comm_handle_data, NULL, NULL, 0, 0);

   HYPRE_Int *send_idx = (HYPRE_Int *)comm_handle_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   hypre_TFree(send_idx, HYPRE_MEMORY_HOST);

   if (want_data)
   {
      HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_data->send_data;
      hypre_ParCSRCommHandleDestroy(comm_handle_data);
      hypre_TFree(send_data, HYPRE_MEMORY_HOST);
   }
#else
   hypre_assert( hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B)) ==
                 hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(B)) );

   hypre_CSRMatrix *B_ext;
   void            *request;

   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   hypre_ParcsrGetExternalRowsInit(B,
                                   hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)),
                                   hypre_ParCSRMatrixColMapOffd(A),
                                   hypre_ParCSRMatrixCommPkg(A),
                                   want_data,
                                   &request);

   B_ext = hypre_ParcsrGetExternalRowsWait(request);
#endif

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTransposeHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTransposeHost( hypre_ParCSRMatrix  *A,
                                 hypre_ParCSRMatrix **AT_ptr,
                                 HYPRE_Int            data )
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix         *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                num_cols = hypre_ParCSRMatrixNumCols(A);
   HYPRE_BigInt             first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt            *row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt            *col_starts = hypre_ParCSRMatrixColStarts(A);

   HYPRE_Int                num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                num_sends = 0, num_recvs = 0, num_cols_offd_AT;
   HYPRE_Int                i, j, k, index, counter, j_row;
   HYPRE_BigInt             value;

   hypre_ParCSRMatrix      *AT;
   hypre_CSRMatrix         *AT_diag;
   hypre_CSRMatrix         *AT_offd;
   hypre_CSRMatrix         *AT_tmp;

   HYPRE_BigInt             first_row_index_AT, first_col_diag_AT;
   HYPRE_Int                local_num_rows_AT, local_num_cols_AT;

   HYPRE_Int               *AT_tmp_i;
   HYPRE_Int               *AT_tmp_j;
   HYPRE_BigInt            *AT_big_j = NULL;
   HYPRE_Complex           *AT_tmp_data = NULL;

   HYPRE_Int               *AT_buf_i = NULL;
   HYPRE_BigInt            *AT_buf_j = NULL;
   HYPRE_Complex           *AT_buf_data = NULL;

   HYPRE_Int               *AT_offd_i;
   HYPRE_Int               *AT_offd_j;
   HYPRE_Complex           *AT_offd_data;
   HYPRE_BigInt            *col_map_offd_AT;
   HYPRE_BigInt             row_starts_AT[2];
   HYPRE_BigInt             col_starts_AT[2];

   HYPRE_Int                num_procs, my_id;

   HYPRE_Int               *recv_procs = NULL;
   HYPRE_Int               *send_procs = NULL;
   HYPRE_Int               *recv_vec_starts = NULL;
   HYPRE_Int               *send_map_starts = NULL;
   HYPRE_Int               *send_map_elmts = NULL;
   HYPRE_Int               *tmp_recv_vec_starts;
   HYPRE_Int               *tmp_send_map_starts;
   hypre_ParCSRCommPkg     *tmp_comm_pkg = NULL;
   hypre_ParCSRCommHandle  *comm_handle = NULL;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   num_cols_offd_AT = 0;
   counter = 0;
   AT_offd_j = NULL;
   AT_offd_data = NULL;
   col_map_offd_AT = NULL;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      hypre_CSRMatrixTranspose (A_offd, &AT_tmp, data);

      AT_tmp_i = hypre_CSRMatrixI(AT_tmp);
      AT_tmp_j = hypre_CSRMatrixJ(AT_tmp);
      if (data)
      {
         AT_tmp_data = hypre_CSRMatrixData(AT_tmp);
      }

      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      AT_buf_i = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends], HYPRE_MEMORY_HOST);
      if (AT_tmp_i[num_cols_offd])
      {
         AT_big_j = hypre_CTAlloc(HYPRE_BigInt, AT_tmp_i[num_cols_offd], HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < AT_tmp_i[num_cols_offd]; i++)
      {
         //AT_tmp_j[i] += first_row_index;
         AT_big_j[i] = (HYPRE_BigInt)AT_tmp_j[i] + first_row_index;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         AT_tmp_i[i] = AT_tmp_i[i + 1] - AT_tmp_i[i];
      }

      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, AT_tmp_i, AT_buf_i);
   }

   hypre_CSRMatrixTranspose(A_diag, &AT_diag, data);

   AT_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols + 1, memory_location);

   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      tmp_send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
      tmp_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);

      tmp_send_map_starts[0] = send_map_starts[0];
      for (i = 0; i < num_sends; i++)
      {
         tmp_send_map_starts[i + 1] = tmp_send_map_starts[i];
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            tmp_send_map_starts[i + 1] += AT_buf_i[j];
            AT_offd_i[send_map_elmts[j] + 1] += AT_buf_i[j];
         }
      }
      for (i = 0; i < num_cols; i++)
      {
         AT_offd_i[i + 1] += AT_offd_i[i];
      }

      tmp_recv_vec_starts[0] = recv_vec_starts[0];
      for (i = 0; i < num_recvs; i++)
      {
         tmp_recv_vec_starts[i + 1] = tmp_recv_vec_starts[i];
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
         {
            tmp_recv_vec_starts[i + 1] +=  AT_tmp_i[j];
         }
      }

      /* Create temporary communication package */
      hypre_ParCSRCommPkgCreateAndFill(comm,
                                       num_recvs, recv_procs, tmp_recv_vec_starts,
                                       num_sends, send_procs, tmp_send_map_starts,
                                       NULL,
                                       &tmp_comm_pkg);

      AT_buf_j = hypre_CTAlloc(HYPRE_BigInt, tmp_send_map_starts[num_sends], HYPRE_MEMORY_HOST);
      comm_handle = hypre_ParCSRCommHandleCreate(22, tmp_comm_pkg, AT_big_j,
                                                 AT_buf_j);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      hypre_TFree(AT_big_j, HYPRE_MEMORY_HOST);

      if (data)
      {
         AT_buf_data = hypre_CTAlloc(HYPRE_Complex, tmp_send_map_starts[num_sends], HYPRE_MEMORY_HOST);
         comm_handle = hypre_ParCSRCommHandleCreate(2, tmp_comm_pkg, AT_tmp_data,
                                                    AT_buf_data);
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
      }

      hypre_TFree(tmp_recv_vec_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_send_map_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixDestroy(AT_tmp);

      if (AT_offd_i[num_cols])
      {
         AT_offd_j = hypre_CTAlloc(HYPRE_Int, AT_offd_i[num_cols], memory_location);
         AT_big_j = hypre_CTAlloc(HYPRE_BigInt, AT_offd_i[num_cols], HYPRE_MEMORY_HOST);
         if (data)
         {
            AT_offd_data = hypre_CTAlloc(HYPRE_Complex,  AT_offd_i[num_cols], memory_location);
         }
      }
      else
      {
         AT_offd_j = NULL;
         AT_offd_data = NULL;
      }

      counter = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            j_row = send_map_elmts[j];
            index = AT_offd_i[j_row];
            for (k = 0; k < AT_buf_i[j]; k++)
            {
               if (data)
               {
                  AT_offd_data[index] = AT_buf_data[counter];
               }
               AT_big_j[index++] = AT_buf_j[counter++];
            }
            AT_offd_i[j_row] = index;
         }
      }
      for (i = num_cols; i > 0; i--)
      {
         AT_offd_i[i] = AT_offd_i[i - 1];
      }
      AT_offd_i[0] = 0;

      if (counter)
      {
         hypre_BigQsort0(AT_buf_j, 0, counter - 1);
         num_cols_offd_AT = 1;
         value = AT_buf_j[0];
         for (i = 1; i < counter; i++)
         {
            if (value < AT_buf_j[i])
            {
               AT_buf_j[num_cols_offd_AT++] = AT_buf_j[i];
               value = AT_buf_j[i];
            }
         }
      }

      if (num_cols_offd_AT)
      {
         col_map_offd_AT = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_HOST);
      }
      else
      {
         col_map_offd_AT = NULL;
      }

      for (i = 0; i < num_cols_offd_AT; i++)
      {
         col_map_offd_AT[i] = AT_buf_j[i];
      }
      hypre_TFree(AT_buf_i, HYPRE_MEMORY_HOST);
      hypre_TFree(AT_buf_j, HYPRE_MEMORY_HOST);
      if (data)
      {
         hypre_TFree(AT_buf_data, HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < counter; i++)
      {
         AT_offd_j[i] = hypre_BigBinarySearch(col_map_offd_AT, AT_big_j[i],
                                              num_cols_offd_AT);
      }
      hypre_TFree(AT_big_j, HYPRE_MEMORY_HOST);
   }

   AT_offd = hypre_CSRMatrixCreate(num_cols, num_cols_offd_AT, counter);
   hypre_CSRMatrixMemoryLocation(AT_offd) = memory_location;
   hypre_CSRMatrixI(AT_offd) = AT_offd_i;
   hypre_CSRMatrixJ(AT_offd) = AT_offd_j;
   hypre_CSRMatrixData(AT_offd) = AT_offd_data;

   for (i = 0; i < 2; i++)
   {
      row_starts_AT[i] = col_starts[i];
      col_starts_AT[i] = row_starts[i];
   }

   first_row_index_AT = row_starts_AT[0];
   first_col_diag_AT  = col_starts_AT[0];

   local_num_rows_AT = (HYPRE_Int)(row_starts_AT[1] - first_row_index_AT );
   local_num_cols_AT = (HYPRE_Int)(col_starts_AT[1] - first_col_diag_AT);

   AT = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixComm(AT) = comm;
   hypre_ParCSRMatrixDiag(AT) = AT_diag;
   hypre_ParCSRMatrixOffd(AT) = AT_offd;
   hypre_ParCSRMatrixGlobalNumRows(AT) = hypre_ParCSRMatrixGlobalNumCols(A);
   hypre_ParCSRMatrixGlobalNumCols(AT) = hypre_ParCSRMatrixGlobalNumRows(A);
   hypre_ParCSRMatrixRowStarts(AT)[0]  = row_starts_AT[0];
   hypre_ParCSRMatrixRowStarts(AT)[1]  = row_starts_AT[1];
   hypre_ParCSRMatrixColStarts(AT)[0]  = col_starts_AT[0];
   hypre_ParCSRMatrixColStarts(AT)[1]  = col_starts_AT[1];
   hypre_ParCSRMatrixColMapOffd(AT)    = col_map_offd_AT;

   hypre_ParCSRMatrixFirstRowIndex(AT) = first_row_index_AT;
   hypre_ParCSRMatrixFirstColDiag(AT)  = first_col_diag_AT;

   hypre_ParCSRMatrixLastRowIndex(AT) = first_row_index_AT + local_num_rows_AT - 1;
   hypre_ParCSRMatrixLastColDiag(AT)  = first_col_diag_AT + local_num_cols_AT - 1;

   hypre_ParCSRMatrixOwnsData(AT) = 1;
   hypre_ParCSRMatrixCommPkg(AT)  = NULL;
   hypre_ParCSRMatrixCommPkgT(AT) = NULL;

   hypre_ParCSRMatrixRowindices(AT) = NULL;
   hypre_ParCSRMatrixRowvalues(AT)  = NULL;
   hypre_ParCSRMatrixGetrowactive(AT) = 0;

   hypre_ParCSRMatrixOwnsAssumedPartition(AT) = 1;

   *AT_ptr = AT;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTranspose( hypre_ParCSRMatrix  *A,
                             hypre_ParCSRMatrix **AT_ptr,
                             HYPRE_Int            data )
{
   hypre_GpuProfilingPushRange("ParCSRMatrixTranspose");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixTransposeDevice(A, AT_ptr, data);
   }
   else
#endif
   {
      hypre_ParCSRMatrixTransposeHost(A, AT_ptr, data);
   }

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixLocalTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixLocalTranspose( hypre_ParCSRMatrix  *A )
{
   if (!hypre_ParCSRMatrixDiagT(A))
   {
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      if (A_diag)
      {
         hypre_CSRMatrix *AT_diag = NULL;
         hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
         hypre_ParCSRMatrixDiagT(A) = AT_diag;
      }
   }

   if (!hypre_ParCSRMatrixOffdT(A))
   {
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      if (A_offd)
      {
         hypre_CSRMatrix *AT_offd = NULL;
         hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
         hypre_ParCSRMatrixOffdT(A) = AT_offd;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGenSpanningTree
 *
 * generate a parallel spanning tree (for Maxwell Equation)
 * G_csr is the node to edge connectivity matrix
 *--------------------------------------------------------------------------*/

void
hypre_ParCSRMatrixGenSpanningTree( hypre_ParCSRMatrix *G_csr,
                                   HYPRE_Int         **indices,
                                   HYPRE_Int           G_type )
{
   HYPRE_BigInt nrows_G, ncols_G;
   HYPRE_Int *G_diag_i, *G_diag_j, *GT_diag_mat, i, j, k, edge;
   HYPRE_Int *nodes_marked, *edges_marked, *queue, queue_tail, queue_head, node;
   HYPRE_Int mypid, nprocs, n_children, *children, nsends, *send_procs, *recv_cnts;
   HYPRE_Int nrecvs, *recv_procs, n_proc_array, *proc_array, *pgraph_i, *pgraph_j;
   HYPRE_Int parent, proc, proc2, node2, found, *t_indices, tree_size, *T_diag_i;
   HYPRE_Int *T_diag_j, *counts, offset;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *comm_pkg;
   hypre_CSRMatrix     *G_diag;

   /* fetch G matrix (G_type = 0 ==> node to edge) */

   if (G_type == 0)
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      G_diag_i = hypre_CSRMatrixI(G_diag);
      G_diag_j = hypre_CSRMatrixJ(G_diag);
   }
   else
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      T_diag_i = hypre_CSRMatrixI(G_diag);
      T_diag_j = hypre_CSRMatrixJ(G_diag);
      counts = hypre_TAlloc(HYPRE_Int, nrows_G, HYPRE_MEMORY_HOST);
      for (i = 0; i < nrows_G; i++) { counts[i] = 0; }
      for (i = 0; i < T_diag_i[ncols_G]; i++) { counts[T_diag_j[i]]++; }
      G_diag_i = hypre_TAlloc(HYPRE_Int, (nrows_G + 1), HYPRE_MEMORY_HOST);
      G_diag_j = hypre_TAlloc(HYPRE_Int, T_diag_i[ncols_G], HYPRE_MEMORY_HOST);
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) { G_diag_i[i] = G_diag_i[i - 1] + counts[i - 1]; }
      for (i = 0; i < ncols_G; i++)
      {
         for (j = T_diag_i[i]; j < T_diag_i[i + 1]; j++)
         {
            k = T_diag_j[j];
            offset = G_diag_i[k]++;
            G_diag_j[offset] = i;
         }
      }
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++)
      {
         G_diag_i[i] = G_diag_i[i - 1] + counts[i - 1];
      }
      hypre_TFree(counts, HYPRE_MEMORY_HOST);
   }

   /* form G transpose in special form (2 nodes per edge max) */

   GT_diag_mat = hypre_TAlloc(HYPRE_Int, 2 * ncols_G, HYPRE_MEMORY_HOST);
   for (i = 0; i < 2 * ncols_G; i++) { GT_diag_mat[i] = -1; }
   for (i = 0; i < nrows_G; i++)
   {
      for (j = G_diag_i[i]; j < G_diag_i[i + 1]; j++)
      {
         edge = G_diag_j[j];
         if (GT_diag_mat[edge * 2] == -1) { GT_diag_mat[edge * 2] = i; }
         else { GT_diag_mat[edge * 2 + 1] = i; }
      }
   }

   /* BFS on the local matrix graph to find tree */

   nodes_marked = hypre_TAlloc(HYPRE_Int, nrows_G, HYPRE_MEMORY_HOST);
   edges_marked = hypre_TAlloc(HYPRE_Int, ncols_G, HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_G; i++) { nodes_marked[i] = 0; }
   for (i = 0; i < ncols_G; i++) { edges_marked[i] = 0; }
   queue = hypre_TAlloc(HYPRE_Int, nrows_G, HYPRE_MEMORY_HOST);
   queue_head = 0;
   queue_tail = 1;
   queue[0] = 0;
   nodes_marked[0] = 1;
   while ((queue_tail - queue_head) > 0)
   {
      node = queue[queue_tail - 1];
      queue_tail--;
      for (i = G_diag_i[node]; i < G_diag_i[node + 1]; i++)
      {
         edge = G_diag_j[i];
         if (edges_marked[edge] == 0)
         {
            if (GT_diag_mat[2 * edge + 1] != -1)
            {
               node2 = GT_diag_mat[2 * edge];
               if (node2 == node) { node2 = GT_diag_mat[2 * edge + 1]; }
               if (nodes_marked[node2] == 0)
               {
                  nodes_marked[node2] = 1;
                  edges_marked[edge] = 1;
                  queue[queue_tail] = node2;
                  queue_tail++;
               }
            }
         }
      }
   }
   hypre_TFree(nodes_marked, HYPRE_MEMORY_HOST);
   hypre_TFree(queue, HYPRE_MEMORY_HOST);
   hypre_TFree(GT_diag_mat, HYPRE_MEMORY_HOST);

   /* fetch the communication information from */

   comm = hypre_ParCSRMatrixComm(G_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   if (nprocs == 1 && comm_pkg == NULL)
   {

      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) G_csr);

      comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   }

   /* construct processor graph based on node-edge connection */
   /* (local edges connected to neighbor processor nodes)     */

   n_children = 0;
   nrecvs = nsends = 0;
   if (nprocs > 1)
   {
      nsends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      nrecvs     = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      proc_array = NULL;
      if ((nsends + nrecvs) > 0)
      {
         n_proc_array = 0;
         proc_array = hypre_TAlloc(HYPRE_Int, (nsends + nrecvs), HYPRE_MEMORY_HOST);
         for (i = 0; i < nsends; i++) { proc_array[i] = send_procs[i]; }
         for (i = 0; i < nrecvs; i++) { proc_array[nsends + i] = recv_procs[i]; }
         hypre_qsort0(proc_array, 0, nsends + nrecvs - 1);
         n_proc_array = 1;
         for (i = 1; i < nrecvs + nsends; i++)
            if (proc_array[i] != proc_array[n_proc_array])
            {
               proc_array[n_proc_array++] = proc_array[i];
            }
      }
      pgraph_i = hypre_TAlloc(HYPRE_Int, (nprocs + 1), HYPRE_MEMORY_HOST);
      recv_cnts = hypre_TAlloc(HYPRE_Int, nprocs, HYPRE_MEMORY_HOST);
      hypre_MPI_Allgather(&n_proc_array, 1, HYPRE_MPI_INT, recv_cnts, 1,
                          HYPRE_MPI_INT, comm);
      pgraph_i[0] = 0;
      for (i = 1; i <= nprocs; i++)
      {
         pgraph_i[i] = pgraph_i[i - 1] + recv_cnts[i - 1];
      }
      pgraph_j = hypre_TAlloc(HYPRE_Int, pgraph_i[nprocs], HYPRE_MEMORY_HOST);
      hypre_MPI_Allgatherv(proc_array, n_proc_array, HYPRE_MPI_INT, pgraph_j,
                           recv_cnts, pgraph_i, HYPRE_MPI_INT, comm);
      hypre_TFree(recv_cnts, HYPRE_MEMORY_HOST);

      /* BFS on the processor graph to determine parent and children */

      nodes_marked = hypre_TAlloc(HYPRE_Int, nprocs, HYPRE_MEMORY_HOST);
      for (i = 0; i < nprocs; i++) { nodes_marked[i] = -1; }
      queue = hypre_TAlloc(HYPRE_Int, nprocs, HYPRE_MEMORY_HOST);
      queue_head = 0;
      queue_tail = 1;
      node = 0;
      queue[0] = node;
      while ((queue_tail - queue_head) > 0)
      {
         proc = queue[queue_tail - 1];
         queue_tail--;
         for (i = pgraph_i[proc]; i < pgraph_i[proc + 1]; i++)
         {
            proc2 = pgraph_j[i];
            if (nodes_marked[proc2] < 0)
            {
               nodes_marked[proc2] = proc;
               queue[queue_tail] = proc2;
               queue_tail++;
            }
         }
      }
      parent = nodes_marked[mypid];
      n_children = 0;
      for (i = 0; i < nprocs; i++) if (nodes_marked[i] == mypid) { n_children++; }
      if (n_children == 0) {n_children = 0; children = NULL;}
      else
      {
         children = hypre_TAlloc(HYPRE_Int, n_children, HYPRE_MEMORY_HOST);
         n_children = 0;
         for (i = 0; i < nprocs; i++)
            if (nodes_marked[i] == mypid) { children[n_children++] = i; }
      }
      hypre_TFree(nodes_marked, HYPRE_MEMORY_HOST);
      hypre_TFree(queue, HYPRE_MEMORY_HOST);
      hypre_TFree(pgraph_i, HYPRE_MEMORY_HOST);
      hypre_TFree(pgraph_j, HYPRE_MEMORY_HOST);
   }

   /* first, connection with my parent : if the edge in my parent *
    * is incident to one of my nodes, then my parent will mark it */

   found = 0;
   for (i = 0; i < nrecvs; i++)
   {
      proc = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (proc == parent)
      {
         found = 1;
         break;
      }
   }

   /* but if all the edges connected to my parent are on my side, *
    * then I will just pick one of them as tree edge              */

   if (found == 0)
   {
      for (i = 0; i < nsends; i++)
      {
         proc = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == parent)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }

   /* next, if my processor has an edge incident on one node in my *
    * child, put this edge on the tree. But if there is no such    *
    * edge, then I will assume my child will pick up an edge       */

   for (j = 0; j < n_children; j++)
   {
      proc = children[j];
      for (i = 0; i < nsends; i++)
      {
         proc2 = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == proc2)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   if (n_children > 0)
   {
      hypre_TFree(children, HYPRE_MEMORY_HOST);
   }

   /* count the size of the tree */

   tree_size = 0;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) { tree_size++; }
   t_indices = hypre_TAlloc(HYPRE_Int, (tree_size + 1), HYPRE_MEMORY_HOST);
   t_indices[0] = tree_size;
   tree_size = 1;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) { t_indices[tree_size++] = i; }
   (*indices) = t_indices;
   hypre_TFree(edges_marked, HYPRE_MEMORY_HOST);
   if (G_type != 0)
   {
      hypre_TFree(G_diag_i, HYPRE_MEMORY_HOST);
      hypre_TFree(G_diag_j, HYPRE_MEMORY_HOST);
   }
}

/* -----------------------------------------------------------------------------
 * extract submatrices based on given indices
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractSubmatrices( hypre_ParCSRMatrix *A_csr,
                                           HYPRE_Int *indices2,
                                           hypre_ParCSRMatrix ***submatrices )
{
   HYPRE_Int    nrows_A, nindices, *indices, *A_diag_i, *A_diag_j, mypid, nprocs;
   HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *exp_indices;
   HYPRE_BigInt *itmp_array;
   HYPRE_Int    nnz11, nnz12, nnz21, nnz22, col, ncols_offd, nnz_offd, nnz_diag;
   HYPRE_Int    nrows, nnz;
   HYPRE_BigInt global_nrows, global_ncols, *row_starts, *col_starts;
   HYPRE_Int    *diag_i, *diag_j, row, *offd_i;
   HYPRE_Complex *A_diag_a, *diag_a;
   hypre_ParCSRMatrix *A11_csr, *A12_csr, *A21_csr, *A22_csr;
   hypre_CSRMatrix    *A_diag, *diag, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   hypre_qsort0(indices, 0, nindices - 1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   comm = hypre_ParCSRMatrixComm(A_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);
   if (nprocs > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ExtractSubmatrices: cannot handle nprocs > 1 yet.\n");
      exit(1);
   }

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = hypre_TAlloc(HYPRE_Int, (nprocs + 1), HYPRE_MEMORY_HOST);
   proc_offsets2 = hypre_TAlloc(HYPRE_Int, (nprocs + 1), HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(&nindices, 1, HYPRE_MPI_INT, proc_offsets1, 1,
                       HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++)
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   }
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++)
   {
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];
   }

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = hypre_TAlloc(HYPRE_Int, nrows_A, HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_A; i++) { exp_indices[i] = -1; }
   for (i = 0; i < nindices; i++)
   {
      if (exp_indices[indices[i]] == -1) { exp_indices[indices[i]] = i; }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ExtractSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz12 = nnz21 = nnz22 = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz11++; }
            else { nnz12++; }
         }
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz21++; }
            else { nnz22++; }
         }
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz11;
   /* This case is not yet implemented! */
   global_nrows = 0;
   global_ncols = 0;
   row_starts = NULL;
   col_starts = NULL;
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A12 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz12;
   global_nrows = (HYPRE_BigInt)proc_offsets1[nprocs];
   global_ncols = (HYPRE_BigInt)proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   col_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (HYPRE_BigInt)proc_offsets1[i];
      col_starts[i] = (HYPRE_BigInt)proc_offsets2[i];
   }
   A12_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }

   if (nnz > nnz_diag)
   {
      hypre_assert(0);
      hypre_error(HYPRE_ERROR_GENERIC);
   }

   diag = hypre_ParCSRMatrixDiag(A12_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = hypre_ParCSRMatrixOffd(A12_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;
   hypre_TFree(row_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A21 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz21;
   global_nrows = (HYPRE_BigInt)proc_offsets2[nprocs];
   global_ncols = (HYPRE_BigInt)proc_offsets1[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   col_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = (HYPRE_BigInt)proc_offsets1[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;
   hypre_TFree(row_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A22 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz22;
   global_nrows = (HYPRE_BigInt)proc_offsets2[nprocs];
   global_ncols = (HYPRE_BigInt)proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   col_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = (HYPRE_BigInt)proc_offsets2[i];
   }
   A22_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A22_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = hypre_ParCSRMatrixOffd(A22_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;
   hypre_TFree(row_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A12_csr;
   (*submatrices)[2] = A21_csr;
   (*submatrices)[3] = A22_csr;
   hypre_TFree(proc_offsets1, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_offsets2, HYPRE_MEMORY_HOST);
   hypre_TFree(exp_indices, HYPRE_MEMORY_HOST);
}

/* -----------------------------------------------------------------------------
 * extract submatrices of a rectangular matrix
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractRowSubmatrices( hypre_ParCSRMatrix *A_csr,
                                              HYPRE_Int *indices2,
                                              hypre_ParCSRMatrix ***submatrices )
{
   HYPRE_Int    nrows_A, nindices, *indices, *A_diag_i, *A_diag_j, mypid, nprocs;
   HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *exp_indices;
   HYPRE_Int    nnz11, nnz21, col, ncols_offd, nnz_offd, nnz_diag;
   HYPRE_Int    *A_offd_i, *A_offd_j;
   HYPRE_Int    nrows, nnz;
   HYPRE_BigInt global_nrows, global_ncols, *row_starts, *col_starts, *itmp_array;
   HYPRE_Int    *diag_i, *diag_j, row, *offd_i, *offd_j, nnz11_offd, nnz21_offd;
   HYPRE_Complex *A_diag_a, *diag_a, *offd_a;
   hypre_ParCSRMatrix *A11_csr, *A21_csr;
   hypre_CSRMatrix    *A_diag, *diag, *A_offd, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   hypre_qsort0(indices, 0, nindices - 1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = (HYPRE_Int)hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   A_offd = hypre_ParCSRMatrixOffd(A_csr);
   A_offd_i = hypre_CSRMatrixI(A_offd);
   A_offd_j = hypre_CSRMatrixJ(A_offd);
   comm = hypre_ParCSRMatrixComm(A_csr);
   hypre_MPI_Comm_rank(comm, &mypid);
   hypre_MPI_Comm_size(comm, &nprocs);

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = hypre_TAlloc(HYPRE_Int, (nprocs + 1), HYPRE_MEMORY_HOST);
   proc_offsets2 = hypre_TAlloc(HYPRE_Int, (nprocs + 1), HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(&nindices, 1, HYPRE_MPI_INT, proc_offsets1, 1,
                       HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++)
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   }
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++)
   {
      proc_offsets2[i] = (HYPRE_Int)(itmp_array[i] - proc_offsets1[i]);
   }

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = hypre_TAlloc(HYPRE_Int, nrows_A, HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_A; i++) { exp_indices[i] = -1; }
   for (i = 0; i < nindices; i++)
   {
      if (exp_indices[indices[i]] == -1) { exp_indices[indices[i]] = i; }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ExtractRowSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz21 = nnz11_offd = nnz21_offd = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz11++; }
         }
         nnz11_offd += A_offd_i[i + 1] - A_offd_i[i];
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0) { nnz21++; }
         }
         nnz21_offd += A_offd_i[i + 1] - A_offd_i[i];
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_diag   = nnz11;
   nnz_offd   = nnz11_offd;

   global_nrows = (HYPRE_BigInt)proc_offsets1[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   col_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (HYPRE_BigInt)proc_offsets1[i];
      col_starts[i] = itmp_array[i];
   }
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   offd_j = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   offd_a = hypre_CTAlloc(HYPRE_Complex,  nnz_offd, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;
   hypre_TFree(row_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A21 matrix
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_offd   = nnz21_offd;
   nnz_diag   = nnz21;
   global_nrows = (HYPRE_BigInt)proc_offsets2[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   col_starts = hypre_CTAlloc(HYPRE_BigInt,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = itmp_array[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   diag_a = hypre_CTAlloc(HYPRE_Complex,  nnz_diag, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            diag_j[nnz] = A_diag_j[j];
            diag_a[nnz++] = A_diag_a[j];
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_HOST);
   offd_j = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   offd_a = hypre_CTAlloc(HYPRE_Complex,  nnz_offd, HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;
   hypre_TFree(row_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A21_csr;
   hypre_TFree(proc_offsets1, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_offsets2, HYPRE_MEMORY_HOST);
   hypre_TFree(exp_indices, HYPRE_MEMORY_HOST);
}

/* -----------------------------------------------------------------------------
 * return the sum of all local elements of the matrix
 * ----------------------------------------------------------------------------- */

HYPRE_Complex hypre_ParCSRMatrixLocalSumElts( hypre_ParCSRMatrix * A )
{
   hypre_CSRMatrix * A_diag = hypre_ParCSRMatrixDiag( A );
   hypre_CSRMatrix * A_offd = hypre_ParCSRMatrixOffd( A );

   return hypre_CSRMatrixSumElts(A_diag) + hypre_CSRMatrixSumElts(A_offd);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatAminvDB
 * computes C = (A - inv(D)B) where D is a diagonal matrix
 * Note: Data structure of A is expected to be a subset of data structure of B!
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixAminvDB( hypre_ParCSRMatrix  *A,
                           hypre_ParCSRMatrix  *B,
                           HYPRE_Complex       *d,
                           hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm              comm            = hypre_ParCSRMatrixComm(B);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int             num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_ParCSRCommPkg  *comm_pkg_B      = hypre_ParCSRMatrixCommPkg(B);
   hypre_CSRMatrix      *B_diag          = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix      *B_offd          = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int             num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_Int             num_sends_B;
   HYPRE_Int             num_recvs_B;
   HYPRE_Int             i, j, cnt;

   HYPRE_Int            *A_diag_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex        *A_diag_data    = hypre_CSRMatrixData(A_diag);

   HYPRE_Int            *A_offd_i       = hypre_CSRMatrixI(A_offd);
   HYPRE_Int            *A_offd_j       = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex        *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_BigInt         *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_Int             num_rows       = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int            *B_diag_i       = hypre_CSRMatrixI(B_diag);
   HYPRE_Int            *B_diag_j       = hypre_CSRMatrixJ(B_diag);
   HYPRE_Complex        *B_diag_data    = hypre_CSRMatrixData(B_diag);

   HYPRE_Int            *B_offd_i       = hypre_CSRMatrixI(B_offd);
   HYPRE_Int            *B_offd_j       = hypre_CSRMatrixJ(B_offd);
   HYPRE_Complex        *B_offd_data    = hypre_CSRMatrixData(B_offd);
   HYPRE_BigInt         *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRMatrix   *C           = NULL;
   hypre_CSRMatrix      *C_diag      = NULL;
   hypre_CSRMatrix      *C_offd      = NULL;
   HYPRE_Int            *C_diag_i    = NULL;
   HYPRE_Int            *C_diag_j    = NULL;
   HYPRE_Complex        *C_diag_data = NULL;
   HYPRE_Int            *C_offd_i    = NULL;
   HYPRE_Int            *C_offd_j    = NULL;
   HYPRE_Complex        *C_offd_data = NULL;

   HYPRE_Int             num_procs, my_id;
   HYPRE_Int            *recv_procs_B;
   HYPRE_Int            *send_procs_B;
   HYPRE_Int            *recv_vec_starts_B;
   HYPRE_Int            *send_map_starts_B;
   HYPRE_Int            *send_map_elmts_B;
   hypre_ParCSRCommPkg  *comm_pkg_C = NULL;
   HYPRE_Int            *recv_procs_C;
   HYPRE_Int            *send_procs_C;
   HYPRE_Int            *recv_vec_starts_C;
   HYPRE_Int            *send_map_starts_C;
   HYPRE_Int            *send_map_elmts_C;
   HYPRE_Int            *map_to_B = NULL;
   HYPRE_Complex        *D_tmp;
   HYPRE_Int             size, rest, num_threads, ii;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   num_threads = hypre_NumThreads();

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for B, a CommPkg is generated
    *--------------------------------------------------------------------*/

   if (!comm_pkg_B)
   {
      hypre_MatvecCommPkgCreate(B);
      comm_pkg_B = hypre_ParCSRMatrixCommPkg(B);
   }

   C = hypre_ParCSRMatrixClone(B, 0);
   /*hypre_ParCSRMatrixInitialize(C);*/

   C_diag = hypre_ParCSRMatrixDiag(C);
   C_diag_i = hypre_CSRMatrixI(C_diag);
   C_diag_j = hypre_CSRMatrixJ(C_diag);
   C_diag_data = hypre_CSRMatrixData(C_diag);
   C_offd = hypre_ParCSRMatrixOffd(C);
   C_offd_i = hypre_CSRMatrixI(C_offd);
   C_offd_j = hypre_CSRMatrixJ(C_offd);
   C_offd_data = hypre_CSRMatrixData(C_offd);

   size = num_rows / num_threads;
   rest = num_rows - size * num_threads;

   D_tmp = hypre_CTAlloc(HYPRE_Complex, num_rows, HYPRE_MEMORY_HOST);

   if (num_cols_offd_A)
   {
      map_to_B = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < num_cols_offd_A; i++)
      {
         while (col_map_offd_B[cnt] < col_map_offd_A[i])
         {
            cnt++;
         }
         map_to_B[i] = cnt;
         cnt++;
      }
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii, i, j)
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      HYPRE_Int *A_marker = NULL;
      HYPRE_Int ns, ne, A_col, num_cols, nmax;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }
      nmax = hypre_max(num_rows, num_cols_offd_B);
      A_marker = hypre_CTAlloc(HYPRE_Int,  nmax, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows; i++)
      {
         A_marker[i] = -1;
      }

      for (i = ns; i < ne; i++)
      {
         D_tmp[i] = 1.0 / d[i];
      }

      num_cols = C_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            A_col = A_diag_j[j];
            if (A_marker[A_col] < C_diag_i[i])
            {
               A_marker[A_col] = num_cols;
               C_diag_j[num_cols] = A_col;
               C_diag_data[num_cols] = A_diag_data[j];
               num_cols++;
            }
            else
            {
               C_diag_data[A_marker[A_col]] += A_diag_data[j];
            }
         }
         for (j = B_diag_i[i]; j < B_diag_i[i + 1]; j++)
         {
            A_col = B_diag_j[j];
            if (A_marker[A_col] < C_diag_i[i])
            {
               A_marker[A_col] = num_cols;
               C_diag_j[num_cols] = A_col;
               C_diag_data[num_cols] = -D_tmp[i] * B_diag_data[j];
               num_cols++;
            }
            else
            {
               C_diag_data[A_marker[A_col]] -= D_tmp[i] * B_diag_data[j];
            }
         }
      }

      for (i = 0; i < num_cols_offd_B; i++)
      {
         A_marker[i] = -1;
      }

      num_cols = C_offd_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            A_col = map_to_B[A_offd_j[j]];
            if (A_marker[A_col] < B_offd_i[i])
            {
               A_marker[A_col] = num_cols;
               C_offd_j[num_cols] = A_col;
               C_offd_data[num_cols] = A_offd_data[j];
               num_cols++;
            }
            else
            {
               C_offd_data[A_marker[A_col]] += A_offd_data[j];
            }
         }
         for (j = B_offd_i[i]; j < B_offd_i[i + 1]; j++)
         {
            A_col = B_offd_j[j];
            if (A_marker[A_col] < B_offd_i[i])
            {
               A_marker[A_col] = num_cols;
               C_offd_j[num_cols] = A_col;
               C_offd_data[num_cols] = -D_tmp[i] * B_offd_data[j];
               num_cols++;
            }
            else
            {
               C_offd_data[A_marker[A_col]] -= D_tmp[i] * B_offd_data[j];
            }
         }
      }
      hypre_TFree(A_marker, HYPRE_MEMORY_HOST);

   } /* end parallel region */

   /*for (i=0; i < num_cols_offd_B; i++)
     col_map_offd_C[i] = col_map_offd_B[i]; */

   num_sends_B       = hypre_ParCSRCommPkgNumSends(comm_pkg_B);
   num_recvs_B       = hypre_ParCSRCommPkgNumRecvs(comm_pkg_B);
   recv_procs_B      = hypre_ParCSRCommPkgRecvProcs(comm_pkg_B);
   recv_vec_starts_B = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_B);
   send_procs_B      = hypre_ParCSRCommPkgSendProcs(comm_pkg_B);
   send_map_starts_B = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_B);
   send_map_elmts_B  = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_B);

   recv_procs_C      = hypre_CTAlloc(HYPRE_Int, num_recvs_B, HYPRE_MEMORY_HOST);
   recv_vec_starts_C = hypre_CTAlloc(HYPRE_Int, num_recvs_B + 1, HYPRE_MEMORY_HOST);
   send_procs_C      = hypre_CTAlloc(HYPRE_Int, num_sends_B, HYPRE_MEMORY_HOST);
   send_map_starts_C = hypre_CTAlloc(HYPRE_Int, num_sends_B + 1, HYPRE_MEMORY_HOST);
   send_map_elmts_C  = hypre_CTAlloc(HYPRE_Int, send_map_starts_B[num_sends_B], HYPRE_MEMORY_HOST);

   for (i = 0; i < num_recvs_B; i++)
   {
      recv_procs_C[i] = recv_procs_B[i];
   }
   for (i = 0; i < num_recvs_B + 1; i++)
   {
      recv_vec_starts_C[i] = recv_vec_starts_B[i];
   }
   for (i = 0; i < num_sends_B; i++)
   {
      send_procs_C[i] = send_procs_B[i];
   }
   for (i = 0; i < num_sends_B + 1; i++)
   {
      send_map_starts_C[i] = send_map_starts_B[i];
   }
   for (i = 0; i < send_map_starts_B[num_sends_B]; i++)
   {
      send_map_elmts_C[i] = send_map_elmts_B[i];
   }

   /* Create communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_B, recv_procs_C, recv_vec_starts_C,
                                    num_sends_B, send_procs_C, send_map_starts_C,
                                    send_map_elmts_C,
                                    &comm_pkg_C);

   hypre_ParCSRMatrixCommPkg(C) = comm_pkg_C;

   hypre_TFree(D_tmp, HYPRE_MEMORY_HOST);
   if (num_cols_offd_A)
   {
      hypre_TFree(map_to_B, HYPRE_MEMORY_HOST);
   }

   *C_ptr = C;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParTMatmul:
 *
 * Multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C
 *
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParTMatmul( hypre_ParCSRMatrix  *A,
                  hypre_ParCSRMatrix  *B)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *AT_diag = NULL;

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *AT_offd = NULL;

   HYPRE_Int    num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int    num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);

   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_BigInt    *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   HYPRE_BigInt    first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt *col_starts_A = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int    num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int    num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int    num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_BigInt       *col_map_offd_C = NULL;
   HYPRE_Int          *map_B_to_C;

   hypre_CSRMatrix *C_diag = NULL;
   hypre_CSRMatrix *C_tmp_diag = NULL;

   HYPRE_Complex   *C_diag_data = NULL;
   HYPRE_Int       *C_diag_i = NULL;
   HYPRE_Int       *C_diag_j = NULL;
   HYPRE_BigInt    first_col_diag_C;
   HYPRE_BigInt    last_col_diag_C;

   hypre_CSRMatrix *C_offd = NULL;
   hypre_CSRMatrix *C_tmp_offd = NULL;
   hypre_CSRMatrix *C_int = NULL;
   hypre_CSRMatrix *C_ext = NULL;
   HYPRE_Int   *C_ext_i = NULL;
   HYPRE_BigInt   *C_ext_j = NULL;
   HYPRE_Complex   *C_ext_data = NULL;
   HYPRE_Int   *C_ext_diag_i = NULL;
   HYPRE_Int   *C_ext_diag_j = NULL;
   HYPRE_Complex   *C_ext_diag_data = NULL;
   HYPRE_Int   *C_ext_offd_i = NULL;
   HYPRE_Int   *C_ext_offd_j = NULL;
   HYPRE_Complex   *C_ext_offd_data = NULL;
   HYPRE_Int    C_ext_size = 0;
   HYPRE_Int    C_ext_diag_size = 0;
   HYPRE_Int    C_ext_offd_size = 0;

   HYPRE_Int   *C_tmp_diag_i;
   HYPRE_Int   *C_tmp_diag_j = NULL;
   HYPRE_Complex   *C_tmp_diag_data = NULL;
   HYPRE_Int   *C_tmp_offd_i = NULL;
   HYPRE_Int   *C_tmp_offd_j = NULL;
   HYPRE_Complex   *C_tmp_offd_data = NULL;

   HYPRE_Complex   *C_offd_data = NULL;
   HYPRE_Int       *C_offd_i = NULL;
   HYPRE_Int       *C_offd_j = NULL;

   HYPRE_BigInt    *temp;
   HYPRE_Int       *send_map_starts_A = NULL;
   HYPRE_Int       *send_map_elmts_A;
   HYPRE_Int        num_sends_A = 0;

   HYPRE_Int        num_cols_offd_C = 0;

   HYPRE_Int       *P_marker;

   HYPRE_Int        i, j;
   HYPRE_Int        i1, j_indx;

   HYPRE_BigInt     nrows_A, ncols_A;
   HYPRE_BigInt     nrows_B, ncols_B;
   /*HYPRE_Int              allsquare = 0;*/
   HYPRE_Int        cnt, cnt_offd, cnt_diag;
   HYPRE_BigInt     value;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Int        max_num_threads;
   HYPRE_Int       *C_diag_array = NULL;
   HYPRE_Int       *C_offd_array = NULL;

   HYPRE_BigInt first_row_index, first_col_diag;
   HYPRE_Int local_num_rows, local_num_cols;

   nrows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   ncols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   nrows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   ncols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   max_num_threads = hypre_NumThreads();

   if (nrows_A != nrows_B || num_rows_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   HYPRE_MemoryLocation memory_location_A = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation memory_location_B = hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation memory_location_C = hypre_max(memory_location_A, memory_location_B);

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (!comm_pkg_A)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   }

   hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
   hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);

   C_tmp_diag = hypre_CSRMatrixMultiply(AT_diag, B_diag);
   C_ext_size = 0;
   if (num_procs > 1)
   {
      hypre_CSRMatrix *C_int_diag;
      hypre_CSRMatrix *C_int_offd;
      void            *request;

      C_tmp_offd = hypre_CSRMatrixMultiply(AT_diag, B_offd);
      C_int_diag = hypre_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = hypre_CSRMatrixMultiply(AT_offd, B_offd);
      hypre_ParCSRMatrixDiag(B) = C_int_diag;
      hypre_ParCSRMatrixOffd(B) = C_int_offd;
      C_int = hypre_MergeDiagAndOffd(B);
      hypre_ParCSRMatrixDiag(B) = B_diag;
      hypre_ParCSRMatrixOffd(B) = B_offd;
      hypre_ExchangeExternalRowsInit(C_int, comm_pkg_A, &request);
      C_ext = hypre_ExchangeExternalRowsWait(request);
      C_ext_i = hypre_CSRMatrixI(C_ext);
      C_ext_j = hypre_CSRMatrixBigJ(C_ext);
      C_ext_data = hypre_CSRMatrixData(C_ext);
      C_ext_size = C_ext_i[hypre_CSRMatrixNumRows(C_ext)];

      hypre_CSRMatrixDestroy(C_int);
      hypre_CSRMatrixDestroy(C_int_diag);
      hypre_CSRMatrixDestroy(C_int_offd);
   }
   else
   {
      C_tmp_offd = hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      hypre_CSRMatrixInitialize(C_tmp_offd);
      hypre_CSRMatrixNumRownnz(C_tmp_offd) = 0;
   }
   hypre_CSRMatrixDestroy(AT_diag);
   hypre_CSRMatrixDestroy(AT_offd);

   /*-----------------------------------------------------------------------
    *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
    *  to obtain C_diag and C_offd
    *-----------------------------------------------------------------------*/

   /* check for new nonzero columns in C_offd generated through C_ext */

   first_col_diag_C = first_col_diag_B;
   last_col_diag_C = first_col_diag_B + (HYPRE_BigInt)num_cols_diag_B - 1;

   C_tmp_diag_i = hypre_CSRMatrixI(C_tmp_diag);
   if (C_ext_size || num_cols_offd_B)
   {
      HYPRE_Int C_ext_num_rows;

      num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
      send_map_starts_A = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
      send_map_elmts_A = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_ext_num_rows =  send_map_starts_A[num_sends_A];

      C_ext_diag_i = hypre_CTAlloc(HYPRE_Int,  C_ext_num_rows + 1, HYPRE_MEMORY_HOST);
      C_ext_offd_i = hypre_CTAlloc(HYPRE_Int,  C_ext_num_rows + 1, HYPRE_MEMORY_HOST);
      temp = hypre_CTAlloc(HYPRE_BigInt,  C_ext_size + num_cols_offd_B, HYPRE_MEMORY_HOST);
      C_ext_diag_size = 0;
      C_ext_offd_size = 0;
      for (i = 0; i < C_ext_num_rows; i++)
      {
         for (j = C_ext_i[i]; j < C_ext_i[i + 1]; j++)
         {
            if (C_ext_j[j] < first_col_diag_C ||
                C_ext_j[j] > last_col_diag_C)
            {
               temp[C_ext_offd_size++] = C_ext_j[j];
            }
            else
            {
               C_ext_diag_size++;
            }
         }
         C_ext_diag_i[i + 1] = C_ext_diag_size;
         C_ext_offd_i[i + 1] = C_ext_offd_size;
      }
      cnt = C_ext_offd_size;
      for (i = 0; i < num_cols_offd_B; i++)
      {
         temp[cnt++] = col_map_offd_B[i];
      }

      if (cnt)
      {
         hypre_BigQsort0(temp, 0, cnt - 1);
         value = temp[0];
         num_cols_offd_C = 1;
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

      if (C_ext_diag_size)
      {
         C_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  C_ext_diag_size, HYPRE_MEMORY_HOST);
         C_ext_diag_data = hypre_CTAlloc(HYPRE_Complex,  C_ext_diag_size, HYPRE_MEMORY_HOST);
      }
      if (C_ext_offd_size)
      {
         C_ext_offd_j = hypre_CTAlloc(HYPRE_Int,  C_ext_offd_size, HYPRE_MEMORY_HOST);
         C_ext_offd_data = hypre_CTAlloc(HYPRE_Complex,  C_ext_offd_size, HYPRE_MEMORY_HOST);
      }

      C_tmp_diag_j = hypre_CSRMatrixJ(C_tmp_diag);
      C_tmp_diag_data = hypre_CSRMatrixData(C_tmp_diag);

      C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);
      C_tmp_offd_data = hypre_CSRMatrixData(C_tmp_offd);

      cnt_offd = 0;
      cnt_diag = 0;
      for (i = 0; i < C_ext_num_rows; i++)
      {
         for (j = C_ext_i[i]; j < C_ext_i[i + 1]; j++)
         {
            if (C_ext_j[j] < first_col_diag_C ||
                C_ext_j[j] > last_col_diag_C)
            {
               C_ext_offd_j[cnt_offd] = hypre_BigBinarySearch(col_map_offd_C,
                                                              C_ext_j[j],
                                                              num_cols_offd_C);
               C_ext_offd_data[cnt_offd++] = C_ext_data[j];
            }
            else
            {
               C_ext_diag_j[cnt_diag] = (HYPRE_Int)(C_ext_j[j] - first_col_diag_C);
               C_ext_diag_data[cnt_diag++] = C_ext_data[j];
            }
         }
      }
   }

   if (C_ext)
   {
      hypre_CSRMatrixDestroy(C_ext);
      C_ext = NULL;
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
      {
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
      }
      for (i = 0; i < hypre_CSRMatrixI(C_tmp_offd)[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
      {
         j_indx = C_tmp_offd_j[i];
         C_tmp_offd_j[i] = map_B_to_C[j_indx];
      }
   }

   /*-----------------------------------------------------------------------
    *  Need to compute:
    *    C_diag = C_tmp_diag + C_ext_diag
    *    C_offd = C_tmp_offd + C_ext_offd
    *
    *  First generate structure
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
      C_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_A + 1, memory_location_C);
      C_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_A + 1, memory_location_C);

      C_diag_array = hypre_CTAlloc(HYPRE_Int,  max_num_threads, HYPRE_MEMORY_HOST);
      C_offd_array = hypre_CTAlloc(HYPRE_Int,  max_num_threads, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         HYPRE_Int *B_marker = NULL;
         HYPRE_Int *B_marker_offd = NULL;
         HYPRE_Int ik, jk, j1, j2, jcol;
         HYPRE_Int ns, ne, ii, nnz_d, nnz_o;
         HYPRE_Int rest, size;
         HYPRE_Int num_threads = hypre_NumActiveThreads();

         size = num_cols_diag_A / num_threads;
         rest = num_cols_diag_A - size * num_threads;
         ii = hypre_GetThreadNum();
         if (ii < rest)
         {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
         }
         else
         {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
         }

         B_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_diag_B, HYPRE_MEMORY_HOST);
         B_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_C, HYPRE_MEMORY_HOST);

         for (ik = 0; ik < num_cols_diag_B; ik++)
         {
            B_marker[ik] = -1;
         }

         for (ik = 0; ik < num_cols_offd_C; ik++)
         {
            B_marker_offd[ik] = -1;
         }

         nnz_d = 0;
         nnz_o = 0;
         for (ik = ns; ik < ne; ik++)
         {
            for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik + 1]; jk++)
            {
               jcol = C_tmp_diag_j[jk];
               B_marker[jcol] = ik;
               nnz_d++;
            }

            for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik + 1]; jk++)
            {
               jcol = C_tmp_offd_j[jk];
               B_marker_offd[jcol] = ik;
               nnz_o++;
            }

            for (jk = 0; jk < num_sends_A; jk++)
            {
               for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk + 1]; j1++)
               {
                  if (send_map_elmts_A[j1] == ik)
                  {
                     for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_diag_j[j2];
                        if (B_marker[jcol] < ik)
                        {
                           B_marker[jcol] = ik;
                           nnz_d++;
                        }
                     }
                     for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_offd_j[j2];
                        if (B_marker_offd[jcol] < ik)
                        {
                           B_marker_offd[jcol] = ik;
                           nnz_o++;
                        }
                     }
                     break;
                  }
               }
            }
            C_diag_array[ii] = nnz_d;
            C_offd_array[ii] = nnz_o;
         }
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         if (ii == 0)
         {
            nnz_d = 0;
            nnz_o = 0;
            for (ik = 0; ik < num_threads - 1; ik++)
            {
               C_diag_array[ik + 1] += C_diag_array[ik];
               C_offd_array[ik + 1] += C_offd_array[ik];
            }
            nnz_d = C_diag_array[num_threads - 1];
            nnz_o = C_offd_array[num_threads - 1];
            C_diag_i[num_cols_diag_A] = nnz_d;
            C_offd_i[num_cols_diag_A] = nnz_o;

            C_diag = hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_diag_A, nnz_d);
            C_offd = hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_offd_C, nnz_o);
            hypre_CSRMatrixI(C_diag) = C_diag_i;
            hypre_CSRMatrixInitialize_v2(C_diag, 0, memory_location_C);
            C_diag_j = hypre_CSRMatrixJ(C_diag);
            C_diag_data = hypre_CSRMatrixData(C_diag);
            hypre_CSRMatrixI(C_offd) = C_offd_i;
            hypre_CSRMatrixInitialize_v2(C_offd, 0, memory_location_C);
            C_offd_j = hypre_CSRMatrixJ(C_offd);
            C_offd_data = hypre_CSRMatrixData(C_offd);
         }
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /*-----------------------------------------------------------------------
          *  Need to compute C_diag = C_tmp_diag + C_ext_diag
          *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
          *  Now fill in values
          *-----------------------------------------------------------------------*/

         for (ik = 0; ik < num_cols_diag_B; ik++)
         {
            B_marker[ik] = -1;
         }

         for (ik = 0; ik < num_cols_offd_C; ik++)
         {
            B_marker_offd[ik] = -1;
         }

         /*-----------------------------------------------------------------------
          *  Populate matrices
          *-----------------------------------------------------------------------*/

         nnz_d = 0;
         nnz_o = 0;
         if (ii)
         {
            nnz_d = C_diag_array[ii - 1];
            nnz_o = C_offd_array[ii - 1];
         }
         for (ik = ns; ik < ne; ik++)
         {
            C_diag_i[ik] = nnz_d;
            C_offd_i[ik] = nnz_o;
            for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik + 1]; jk++)
            {
               jcol = C_tmp_diag_j[jk];
               C_diag_j[nnz_d] = jcol;
               C_diag_data[nnz_d] = C_tmp_diag_data[jk];
               B_marker[jcol] = nnz_d;
               nnz_d++;
            }

            for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik + 1]; jk++)
            {
               jcol = C_tmp_offd_j[jk];
               C_offd_j[nnz_o] = jcol;
               C_offd_data[nnz_o] = C_tmp_offd_data[jk];
               B_marker_offd[jcol] = nnz_o;
               nnz_o++;
            }

            for (jk = 0; jk < num_sends_A; jk++)
            {
               for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk + 1]; j1++)
               {
                  if (send_map_elmts_A[j1] == ik)
                  {
                     for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_diag_j[j2];
                        if (B_marker[jcol] < C_diag_i[ik])
                        {
                           C_diag_j[nnz_d] = jcol;
                           C_diag_data[nnz_d] = C_ext_diag_data[j2];
                           B_marker[jcol] = nnz_d;
                           nnz_d++;
                        }
                        else
                        {
                           C_diag_data[B_marker[jcol]] += C_ext_diag_data[j2];
                        }
                     }
                     for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_offd_j[j2];
                        if (B_marker_offd[jcol] < C_offd_i[ik])
                        {
                           C_offd_j[nnz_o] = jcol;
                           C_offd_data[nnz_o] = C_ext_offd_data[j2];
                           B_marker_offd[jcol] = nnz_o;
                           nnz_o++;
                        }
                        else
                        {
                           C_offd_data[B_marker_offd[jcol]] += C_ext_offd_data[j2];
                        }
                     }
                     break;
                  }
               }
            }
         }
         hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
         hypre_TFree(B_marker_offd, HYPRE_MEMORY_HOST);

      } /*end parallel region */

      hypre_TFree(C_diag_array, HYPRE_MEMORY_HOST);
      hypre_TFree(C_offd_array, HYPRE_MEMORY_HOST);
   }

   /*C = hypre_ParCSRMatrixCreate(comm, ncols_A, ncols_B, col_starts_A,
     col_starts_B, num_cols_offd_C, nnz_diag, nnz_offd);

     hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
     hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C)); */
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next
      processor's rows */
   first_row_index = col_starts_A[0];
   local_num_rows = (HYPRE_Int)(col_starts_A[1] - first_row_index );
   first_col_diag = col_starts_B[0];
   local_num_cols = (HYPRE_Int)(col_starts_B[1] - first_col_diag);

   C = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixComm(C) = comm;
   hypre_ParCSRMatrixGlobalNumRows(C) = ncols_A;
   hypre_ParCSRMatrixGlobalNumCols(C) = ncols_B;
   hypre_ParCSRMatrixFirstRowIndex(C) = first_row_index;
   hypre_ParCSRMatrixFirstColDiag(C) = first_col_diag;
   hypre_ParCSRMatrixLastRowIndex(C) = first_row_index + (HYPRE_BigInt)local_num_rows - 1;
   hypre_ParCSRMatrixLastColDiag(C) = first_col_diag + (HYPRE_BigInt)local_num_cols - 1;
   hypre_ParCSRMatrixColMapOffd(C) = NULL;
   hypre_ParCSRMatrixAssumedPartition(C) = NULL;
   hypre_ParCSRMatrixCommPkg(C) = NULL;
   hypre_ParCSRMatrixCommPkgT(C) = NULL;

   /* C row/col starts*/
   hypre_ParCSRMatrixRowStarts(C)[0] = col_starts_A[0];
   hypre_ParCSRMatrixRowStarts(C)[1] = col_starts_A[1];
   hypre_ParCSRMatrixColStarts(C)[0] = col_starts_B[0];
   hypre_ParCSRMatrixColStarts(C)[1] = col_starts_B[1];

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(C) = 1;
   hypre_ParCSRMatrixRowindices(C) = NULL;
   hypre_ParCSRMatrixRowvalues(C) = NULL;
   hypre_ParCSRMatrixGetrowactive(C) = 0;

   if (C_diag)
   {
      hypre_CSRMatrixSetRownnz(C_diag);
      hypre_ParCSRMatrixDiag(C) = C_diag;
   }
   else
   {
      hypre_ParCSRMatrixDiag(C) = C_tmp_diag;
   }

   if (C_offd)
   {
      hypre_CSRMatrixSetRownnz(C_offd);
      hypre_ParCSRMatrixOffd(C) = C_offd;
   }
   else
   {
      hypre_ParCSRMatrixOffd(C) = C_tmp_offd;
   }

   hypre_assert(hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(C)) == memory_location_C);
   hypre_assert(hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(C)) == memory_location_C);

   if (num_cols_offd_C)
   {
      HYPRE_Int jj_count_offd, nnz_offd;
      HYPRE_BigInt *new_col_map_offd_C = NULL;

      P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_offd_C, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd_C; i++)
      {
         P_marker[i] = -1;
      }

      jj_count_offd = 0;
      nnz_offd = C_offd_i[num_cols_diag_A];
      for (i = 0; i < nnz_offd; i++)
      {
         i1 = C_offd_j[i];
         if (P_marker[i1])
         {
            P_marker[i1] = 0;
            jj_count_offd++;
         }
      }

      if (jj_count_offd < num_cols_offd_C)
      {
         new_col_map_offd_C = hypre_CTAlloc(HYPRE_BigInt, jj_count_offd, HYPRE_MEMORY_HOST);
         jj_count_offd = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (!P_marker[i])
            {
               P_marker[i] = jj_count_offd;
               new_col_map_offd_C[jj_count_offd++] = col_map_offd_C[i];
            }
         }

         for (i = 0; i < nnz_offd; i++)
         {
            i1 = C_offd_j[i];
            C_offd_j[i] = P_marker[i1];
         }

         num_cols_offd_C = jj_count_offd;
         hypre_TFree(col_map_offd_C, HYPRE_MEMORY_HOST);
         col_map_offd_C = new_col_map_offd_C;
         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(C)) = num_cols_offd_C;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/
   if (C_ext_size || num_cols_offd_B)
   {
      hypre_TFree(C_ext_diag_i, HYPRE_MEMORY_HOST);
      hypre_TFree(C_ext_offd_i, HYPRE_MEMORY_HOST);
   }

   if (C_ext_diag_size)
   {
      hypre_TFree(C_ext_diag_j, HYPRE_MEMORY_HOST);
      hypre_TFree(C_ext_diag_data, HYPRE_MEMORY_HOST);
   }

   if (C_ext_offd_size)
   {
      hypre_TFree(C_ext_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(C_ext_offd_data, HYPRE_MEMORY_HOST);
   }

   if (num_cols_offd_B)
   {
      hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
   }

   if (C_diag)
   {
      hypre_CSRMatrixDestroy(C_tmp_diag);
   }

   if (C_offd)
   {
      hypre_CSRMatrixDestroy(C_tmp_offd);
   }

#if defined(HYPRE_USING_GPU)
   if ( hypre_GetExecPolicy2(memory_location_A, memory_location_B) == HYPRE_EXEC_DEVICE )
   {
      hypre_CSRMatrixMoveDiagFirstDevice(hypre_ParCSRMatrixDiag(C));
      hypre_SyncComputeStream(hypre_handle());
   }
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParvecBdiagInvScal
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParvecBdiagInvScal( hypre_ParVector     *b,
                          HYPRE_Int            blockSize,
                          hypre_ParVector    **bs,
                          hypre_ParCSRMatrix  *A)
{
   MPI_Comm         comm     = hypre_ParCSRMatrixComm(b);
   HYPRE_Int        num_procs, my_id;
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   HYPRE_Int    i, j, s;
   HYPRE_BigInt block_start, block_end;
   HYPRE_BigInt nrow_global = hypre_ParVectorGlobalSize(b);
   HYPRE_BigInt first_row   = hypre_ParVectorFirstIndex(b);
   HYPRE_BigInt last_row    = hypre_ParVectorLastIndex(b);
   HYPRE_BigInt end_row     = last_row + 1; /* one past-the-last */
   HYPRE_BigInt first_row_block = first_row / (HYPRE_BigInt)(blockSize) * (HYPRE_BigInt)blockSize;
   HYPRE_BigInt end_row_block   = hypre_min( (last_row / (HYPRE_BigInt)blockSize + 1) *
                                             (HYPRE_BigInt)blockSize, nrow_global );

   hypre_assert(blockSize == A->bdiag_size);
   HYPRE_Complex *bdiaginv = A->bdiaginv;
   hypre_ParCSRCommPkg *comm_pkg = A->bdiaginv_comm_pkg;

   HYPRE_Complex *dense = bdiaginv;

   //for (i=first_row_block; i < end_row; i+=blockSize) ;
   //printf("===[%d %d), [ %d %d ) %d === \n", first_row, end_row, first_row_block, end_row_block, i);

   /* local vector of b */
   hypre_Vector    *b_local      = hypre_ParVectorLocalVector(b);
   HYPRE_Complex   *b_local_data = hypre_VectorData(b_local);
   /* number of sends (#procs) */
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   HYPRE_Int num_rows_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   HYPRE_Int num_rows_recv = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_ParVector *bnew = hypre_ParVectorCreate( hypre_ParVectorComm(b),
                                                  hypre_ParVectorGlobalSize(b),
                                                  hypre_ParVectorPartitioning(b) );
   hypre_ParVectorInitialize(bnew);
   hypre_Vector    *bnew_local      = hypre_ParVectorLocalVector(bnew);
   HYPRE_Complex   *bnew_local_data = hypre_VectorData(bnew_local);

   /* send and recv b */
   HYPRE_Complex *send_b = hypre_TAlloc(HYPRE_Complex, num_rows_send, HYPRE_MEMORY_HOST);
   HYPRE_Complex *recv_b = hypre_TAlloc(HYPRE_Complex, num_rows_recv, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_rows_send; i++)
   {
      j = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      send_b[i] = b_local_data[j];
   }
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, send_b, recv_b);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (block_start = first_row_block; block_start < end_row_block; block_start += blockSize)
   {
      HYPRE_BigInt big_i;
      block_end = hypre_min(block_start + (HYPRE_BigInt)blockSize, nrow_global);
      s = (HYPRE_Int)(block_end - block_start);
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         if (big_i < first_row || big_i >= end_row)
         {
            continue;
         }

         HYPRE_Int local_i = (HYPRE_Int)(big_i - first_row);
         HYPRE_Int block_i = (HYPRE_Int)(big_i - block_start);

         bnew_local_data[local_i] = 0.0;

         for (j = 0; j < s; j++)
         {
            HYPRE_BigInt global_rid = block_start + (HYPRE_BigInt)j;
            HYPRE_Complex val = dense[block_i + j * blockSize];
            if (val == 0.0)
            {
               continue;
            }
            if (global_rid >= first_row && global_rid < end_row)
            {
               HYPRE_Int rid = (HYPRE_Int)(global_rid - first_row);
               bnew_local_data[local_i] += val * b_local_data[rid];
            }
            else
            {
               HYPRE_Int rid;

               if (global_rid < first_row)
               {
                  rid = (HYPRE_Int)(global_rid - first_row_block);
               }
               else
               {
                  rid = (HYPRE_Int)(first_row - first_row_block + global_rid - end_row);
               }
               bnew_local_data[local_i] += val * recv_b[rid];
            }
         }
      }
      dense += blockSize * blockSize;
   }

   hypre_TFree(send_b, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_b, HYPRE_MEMORY_HOST);
   *bs = bnew;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParcsrBdiagInvScal
 *
 * Compute As = B^{-1}*A, where B is the block diagonal of A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParcsrBdiagInvScal( hypre_ParCSRMatrix   *A,
                          HYPRE_Int             blockSize,
                          hypre_ParCSRMatrix  **As)
{
   MPI_Comm         comm     = hypre_ParCSRMatrixComm(A);
   HYPRE_Int        num_procs, my_id;
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   HYPRE_Int i, j, k, s;
   HYPRE_BigInt block_start, block_end;
   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);


   HYPRE_Int nrow_local = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt first_row  = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt last_row   = hypre_ParCSRMatrixLastRowIndex(A);
   HYPRE_BigInt end_row    = first_row + (HYPRE_BigInt)nrow_local; /* one past-the-last */

   HYPRE_Int ncol_local = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_BigInt first_col  = hypre_ParCSRMatrixFirstColDiag(A);
   /* HYPRE_Int last_col   = hypre_ParCSRMatrixLastColDiag(A); */
   HYPRE_BigInt end_col    = first_col + (HYPRE_BigInt)ncol_local;

   HYPRE_BigInt nrow_global = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt ncol_global = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(A);
   void *request;

   /* if square globally and locally */
   HYPRE_Int square2 = (nrow_global == ncol_global) && (nrow_local == ncol_local) &&
                       (first_row == first_col);

   if (nrow_global != ncol_global)
   {
      hypre_printf("hypre_ParcsrBdiagInvScal: only support N_ROW == N_COL\n");
      return hypre_error_flag;
   }

   /* in block diagonals, row range of the blocks this proc span */
   HYPRE_BigInt first_row_block = first_row / (HYPRE_BigInt)blockSize * (HYPRE_BigInt)blockSize;
   HYPRE_BigInt end_row_block   = hypre_min( (last_row / (HYPRE_BigInt)blockSize + 1) *
                                             (HYPRE_BigInt)blockSize, nrow_global );
   HYPRE_Int num_blocks = (HYPRE_Int)(last_row / (HYPRE_BigInt)blockSize + 1 - first_row /
                                      (HYPRE_BigInt)blockSize);

   //for (i=first_row_block; i < end_row; i+=blockSize) ;
   //printf("===[%d %d), [ %d %d ) %d === \n", first_row, end_row, first_row_block, end_row_block, i);
   //return 0;

   /* number of external rows */
   HYPRE_Int num_ext_rows = (HYPRE_Int)(end_row_block - first_row_block - (end_row - first_row));
   HYPRE_BigInt *ext_indices;
   HYPRE_Int A_ext_nnz;

   hypre_CSRMatrix *A_ext   = NULL;
   HYPRE_Complex   *A_ext_a = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Real *dense_all = hypre_CTAlloc(HYPRE_Complex, num_blocks * blockSize * blockSize,
                                         HYPRE_MEMORY_HOST);
   HYPRE_Real *dense = dense_all;
   HYPRE_Int *IPIV  = hypre_TAlloc(HYPRE_Int, blockSize, HYPRE_MEMORY_HOST);
   HYPRE_Complex *dgetri_work = NULL;
   HYPRE_Int      dgetri_lwork = -1, lapack_info;

   HYPRE_Int  num_cols_A_offd_new;
   HYPRE_BigInt *col_map_offd_A_new;
   HYPRE_BigInt big_i;
   HYPRE_Int *offd2new = NULL;
   HYPRE_Int *marker_diag, *marker_newoffd;

   HYPRE_Int nnz_diag = A_diag_i[nrow_local];
   HYPRE_Int nnz_offd = A_offd_i[nrow_local];
   HYPRE_Int nnz_diag_new = 0, nnz_offd_new = 0;
   HYPRE_Int *A_diag_i_new, *A_diag_j_new, *A_offd_i_new, *A_offd_j_new;
   HYPRE_Complex *A_diag_a_new, *A_offd_a_new;
   /* heuristic */
   HYPRE_Int nnz_diag_alloc = 2 * nnz_diag;
   HYPRE_Int nnz_offd_alloc = 2 * nnz_offd;

   A_diag_i_new = hypre_CTAlloc(HYPRE_Int,     nrow_local + 1, HYPRE_MEMORY_HOST);
   A_diag_j_new = hypre_CTAlloc(HYPRE_Int,     nnz_diag_alloc, HYPRE_MEMORY_HOST);
   A_diag_a_new = hypre_CTAlloc(HYPRE_Complex, nnz_diag_alloc, HYPRE_MEMORY_HOST);
   A_offd_i_new = hypre_CTAlloc(HYPRE_Int,     nrow_local + 1, HYPRE_MEMORY_HOST);
   A_offd_j_new = hypre_CTAlloc(HYPRE_Int,     nnz_offd_alloc, HYPRE_MEMORY_HOST);
   A_offd_a_new = hypre_CTAlloc(HYPRE_Complex, nnz_offd_alloc, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrix *Anew;
   hypre_CSRMatrix    *Anew_diag;
   hypre_CSRMatrix    *Anew_offd;

   HYPRE_Real eps = 2.2e-16;

   /* Start with extracting the external rows */
   HYPRE_BigInt *ext_offd;
   ext_indices = hypre_CTAlloc(HYPRE_BigInt, num_ext_rows, HYPRE_MEMORY_HOST);
   j = 0;
   for (big_i = first_row_block; big_i < first_row; big_i++)
   {
      ext_indices[j++] = big_i;
   }
   for (big_i = end_row; big_i < end_row_block; big_i++)
   {
      ext_indices[j++] = big_i;
   }

   hypre_assert(j == num_ext_rows);

   /* create CommPkg for external rows */
   hypre_ParCSRFindExtendCommPkg(comm, nrow_global, first_row, nrow_local, row_starts,
                                 hypre_ParCSRMatrixAssumedPartition(A),
                                 num_ext_rows, ext_indices, &A->bdiaginv_comm_pkg);

   hypre_ParcsrGetExternalRowsInit(A, num_ext_rows, ext_indices, A->bdiaginv_comm_pkg, 1, &request);
   A_ext = hypre_ParcsrGetExternalRowsWait(request);

   hypre_TFree(ext_indices, HYPRE_MEMORY_HOST);

   A_ext_i = hypre_CSRMatrixI(A_ext);
   A_ext_j = hypre_CSRMatrixBigJ(A_ext);
   A_ext_a = hypre_CSRMatrixData(A_ext);
   A_ext_nnz = A_ext_i[num_ext_rows];
   ext_offd = hypre_CTAlloc(HYPRE_BigInt, A_ext_nnz, HYPRE_MEMORY_HOST);

   /* fint the offd incides in A_ext */
   for (i = 0, j = 0; i < A_ext_nnz; i++)
   {
      /* global index */
      HYPRE_BigInt cid = A_ext_j[i];
      /* keep the offd indices */
      if (cid < first_col || cid >= end_col)
      {
         ext_offd[j++] = cid;
      }
   }
   /* remove duplicates after sorting (TODO better ways?) */
   hypre_BigQsort0(ext_offd, 0, j - 1);
   for (i = 0, k = 0; i < j; i++)
   {
      if (i == 0 || ext_offd[i] != ext_offd[i - 1])
      {
         ext_offd[k++] = ext_offd[i];
      }
   }
   /* uniion these `k' new indices into col_map_offd_A */
   col_map_offd_A_new = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd + k, HYPRE_MEMORY_HOST);
   if (k)
   {
      /* map offd to offd_new */
      offd2new = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   hypre_union2(num_cols_A_offd, col_map_offd_A, k, ext_offd,
                &num_cols_A_offd_new, col_map_offd_A_new, offd2new, NULL);
   hypre_TFree(ext_offd, HYPRE_MEMORY_HOST);
   /*
    *   adjust column indices in A_ext
    */
   for (i = 0; i < A_ext_nnz; i++)
   {
      HYPRE_BigInt cid = A_ext_j[i];
      if (cid < first_col || cid >= end_col)
      {
         j = hypre_BigBinarySearch(col_map_offd_A_new, cid, num_cols_A_offd_new);
         /* searching must succeed */
         hypre_assert(j >= 0 && j < num_cols_A_offd_new);
         /* trick: save ncol_local + j back */
         A_ext_j[i] = ncol_local + j;
      }
      else
      {
         /* save local index: [0, ncol_local-1] */
         A_ext_j[i] = cid - first_col;
      }
   }

   /* marker for diag */
   marker_diag = hypre_TAlloc(HYPRE_Int, ncol_local, HYPRE_MEMORY_HOST);
   for (i = 0; i < ncol_local; i++)
   {
      marker_diag[i] = -1;
   }
   /* marker for newoffd */
   marker_newoffd = hypre_TAlloc(HYPRE_Int, num_cols_A_offd_new, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd_new; i++)
   {
      marker_newoffd[i] = -1;
   }

   /* outer most loop for blocks */
   for (block_start = first_row_block; block_start < end_row_block;
        block_start += (HYPRE_BigInt)blockSize)
   {
      HYPRE_BigInt big_i;
      block_end = hypre_min(block_start + (HYPRE_BigInt)blockSize, nrow_global);
      s = (HYPRE_Int)(block_end - block_start);

      /* 1. fill the dense block diag matrix */
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         /* row index in this block */
         HYPRE_Int block_i = (HYPRE_Int)(big_i - block_start);

         /* row index i: it can be local or external */
         if (big_i >= first_row && big_i < end_row)
         {
            /* is a local row */
            j = (HYPRE_Int)(big_i - first_row);
            for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
            {
               HYPRE_BigInt cid = (HYPRE_BigInt)A_diag_j[k] + first_col;
               if (cid >= block_start && cid < block_end)
               {
                  dense[block_i + (HYPRE_Int)(cid - block_start)*blockSize] = A_diag_a[k];
               }
            }
            if (num_cols_A_offd)
            {
               for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
               {
                  HYPRE_BigInt cid = col_map_offd_A[A_offd_j[k]];
                  if (cid >= block_start && cid < block_end)
                  {
                     dense[block_i + (HYPRE_Int)(cid - block_start)*blockSize] = A_offd_a[k];
                  }
               }
            }
         }
         else
         {
            /* is an external row */
            if (big_i < first_row)
            {
               j = (HYPRE_Int)(big_i - first_row_block);
            }
            else
            {
               j = (HYPRE_Int)(first_row - first_row_block + big_i - end_row);
            }
            for (k = A_ext_i[j]; k < A_ext_i[j + 1]; k++)
            {
               HYPRE_BigInt cid = A_ext_j[k];
               /* recover the global index */
               cid = cid < (HYPRE_BigInt)ncol_local ? cid + first_col : col_map_offd_A_new[cid - ncol_local];
               if (cid >= block_start && cid < block_end)
               {
                  dense[block_i + (HYPRE_Int)(cid - block_start)*blockSize] = A_ext_a[k];
               }
            }
         }
      }

      /* 2. invert the dense matrix */
      hypre_dgetrf(&s, &s, dense, &blockSize, IPIV, &lapack_info);

      hypre_assert(lapack_info == 0);

      if (lapack_info == 0)
      {
         HYPRE_Int query = -1;
         HYPRE_Real lwork_opt;
         /* query the optimal size of work */
         hypre_dgetri(&s, dense, &blockSize, IPIV, &lwork_opt, &query, &lapack_info);

         hypre_assert(lapack_info == 0);

         if (lwork_opt > dgetri_lwork)
         {
            dgetri_lwork = (HYPRE_Int)lwork_opt;
            dgetri_work = hypre_TReAlloc(dgetri_work, HYPRE_Complex, dgetri_lwork, HYPRE_MEMORY_HOST);
         }

         hypre_dgetri(&s, dense, &blockSize, IPIV, dgetri_work, &dgetri_lwork, &lapack_info);

         hypre_assert(lapack_info == 0);
      }

      /* filter out *zeros* */
      HYPRE_Real Fnorm = 0.0;
      for (i = 0; i < s; i++)
      {
         for (j = 0; j < s; j++)
         {
            HYPRE_Complex t = dense[j + i * blockSize];
            Fnorm += t * t;
         }
      }

      Fnorm = hypre_sqrt(Fnorm);

      for (i = 0; i < s; i++)
      {
         for (j = 0; j < s; j++)
         {
            if ( hypre_abs(dense[j + i * blockSize]) < eps * Fnorm )
            {
               dense[j + i * blockSize] = 0.0;
            }
         }
      }

      /* 3. premultiplication: one-pass dynamic allocation */
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         /* starting points of this row in j */
         HYPRE_Int diag_i_start = nnz_diag_new;
         HYPRE_Int offd_i_start = nnz_offd_new;

         /* compute a new row with global index 'i' and local index 'local_i' */
         HYPRE_Int local_i = (HYPRE_Int)(big_i - first_row);
         /* row index in this block */
         HYPRE_Int block_i = (HYPRE_Int)(big_i - block_start);

         if (big_i < first_row || big_i >= end_row)
         {
            continue;
         }

         /* if square^2: reserve the first space in diag part to the diag entry */
         if (square2)
         {
            marker_diag[local_i] = nnz_diag_new;
            if (nnz_diag_new == nnz_diag_alloc)
            {
               nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
               A_diag_j_new = hypre_TReAlloc(A_diag_j_new, HYPRE_Int,     nnz_diag_alloc, HYPRE_MEMORY_HOST);
               A_diag_a_new = hypre_TReAlloc(A_diag_a_new, HYPRE_Complex, nnz_diag_alloc, HYPRE_MEMORY_HOST);
            }
            A_diag_j_new[nnz_diag_new] = local_i;
            A_diag_a_new[nnz_diag_new] = 0.0;
            nnz_diag_new ++;
         }

         /* combine s rows */
         for (j = 0; j < s; j++)
         {
            /* row to combine: global row id */
            HYPRE_BigInt global_rid = block_start + (HYPRE_BigInt)j;
            /* the multipiler */
            HYPRE_Complex val = dense[block_i + j * blockSize];

            if (val == 0.0)
            {
               continue;
            }

            if (global_rid >= first_row && global_rid < end_row)
            {
               /* this row is local */
               HYPRE_Int rid = (HYPRE_Int)(global_rid - first_row);
               HYPRE_Int ii;

               for (ii = A_diag_i[rid]; ii < A_diag_i[rid + 1]; ii++)
               {
                  HYPRE_Int col = A_diag_j[ii];
                  HYPRE_Complex vv = A_diag_a[ii];

                  if (marker_diag[col] < diag_i_start)
                  {
                     /* this col has not been seen before, create new entry */
                     marker_diag[col] = nnz_diag_new;
                     if (nnz_diag_new == nnz_diag_alloc)
                     {
                        nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
                        A_diag_j_new = hypre_TReAlloc(A_diag_j_new, HYPRE_Int,     nnz_diag_alloc, HYPRE_MEMORY_HOST);
                        A_diag_a_new = hypre_TReAlloc(A_diag_a_new, HYPRE_Complex, nnz_diag_alloc, HYPRE_MEMORY_HOST);
                     }
                     A_diag_j_new[nnz_diag_new] = col;
                     A_diag_a_new[nnz_diag_new] = val * vv;
                     nnz_diag_new ++;
                  }
                  else
                  {
                     /* existing entry, update */
                     HYPRE_Int p = marker_diag[col];

                     hypre_assert(A_diag_j_new[p] == col);

                     A_diag_a_new[p] += val * vv;
                  }
               }

               for (ii = A_offd_i[rid]; ii < A_offd_i[rid + 1]; ii++)
               {
                  HYPRE_Int col = A_offd_j[ii];
                  /* use the mapper to map to new offd */
                  HYPRE_Int col_new = offd2new ? offd2new[col] : col;
                  HYPRE_Complex vv = A_offd_a[ii];

                  if (marker_newoffd[col_new] < offd_i_start)
                  {
                     /* this col has not been seen before, create new entry */
                     marker_newoffd[col_new] = nnz_offd_new;
                     if (nnz_offd_new == nnz_offd_alloc)
                     {
                        nnz_offd_alloc = nnz_offd_alloc * 2 + 1;
                        A_offd_j_new = hypre_TReAlloc(A_offd_j_new, HYPRE_Int,     nnz_offd_alloc, HYPRE_MEMORY_HOST);
                        A_offd_a_new = hypre_TReAlloc(A_offd_a_new, HYPRE_Complex, nnz_offd_alloc, HYPRE_MEMORY_HOST);
                     }
                     A_offd_j_new[nnz_offd_new] = col_new;
                     A_offd_a_new[nnz_offd_new] = val * vv;
                     nnz_offd_new ++;
                  }
                  else
                  {
                     /* existing entry, update */
                     HYPRE_Int p = marker_newoffd[col_new];

                     hypre_assert(A_offd_j_new[p] == col_new);

                     A_offd_a_new[p] += val * vv;
                  }
               }
            }
            else
            {
               /* this is an external row: go to A_ext */
               HYPRE_Int rid, ii;

               if (global_rid < first_row)
               {
                  rid = (HYPRE_Int)(global_rid - first_row_block);
               }
               else
               {
                  rid = (HYPRE_Int)(first_row - first_row_block + global_rid - end_row);
               }

               for (ii = A_ext_i[rid]; ii < A_ext_i[rid + 1]; ii++)
               {
                  HYPRE_Int col = (HYPRE_Int)A_ext_j[ii];
                  HYPRE_Complex vv = A_ext_a[ii];

                  if (col < ncol_local)
                  {
                     /* in diag part */
                     if (marker_diag[col] < diag_i_start)
                     {
                        /* this col has not been seen before, create new entry */
                        marker_diag[col] = nnz_diag_new;
                        if (nnz_diag_new == nnz_diag_alloc)
                        {
                           nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
                           A_diag_j_new = hypre_TReAlloc(A_diag_j_new, HYPRE_Int,     nnz_diag_alloc, HYPRE_MEMORY_HOST);
                           A_diag_a_new = hypre_TReAlloc(A_diag_a_new, HYPRE_Complex, nnz_diag_alloc, HYPRE_MEMORY_HOST);
                        }
                        A_diag_j_new[nnz_diag_new] = col;
                        A_diag_a_new[nnz_diag_new] = val * vv;
                        nnz_diag_new ++;
                     }
                     else
                     {
                        /* existing entry, update */
                        HYPRE_Int p = marker_diag[col];

                        hypre_assert(A_diag_j_new[p] == col);

                        A_diag_a_new[p] += val * vv;
                     }
                  }
                  else
                  {
                     /* in offd part */
                     col -= ncol_local;

                     if (marker_newoffd[col] < offd_i_start)
                     {
                        /* this col has not been seen before, create new entry */
                        marker_newoffd[col] = nnz_offd_new;
                        if (nnz_offd_new == nnz_offd_alloc)
                        {
                           nnz_offd_alloc = nnz_offd_alloc * 2 + 1;
                           A_offd_j_new = hypre_TReAlloc(A_offd_j_new, HYPRE_Int,     nnz_offd_alloc, HYPRE_MEMORY_HOST);
                           A_offd_a_new = hypre_TReAlloc(A_offd_a_new, HYPRE_Complex, nnz_offd_alloc, HYPRE_MEMORY_HOST);
                        }
                        A_offd_j_new[nnz_offd_new] = col;
                        A_offd_a_new[nnz_offd_new] = val * vv;
                        nnz_offd_new ++;
                     }
                     else
                     {
                        /* existing entry, update */
                        HYPRE_Int p = marker_newoffd[col];

                        hypre_assert(A_offd_j_new[p] == col);

                        A_offd_a_new[p] += val * vv;
                     }
                  }
               }
            }
         }

         /* done for row local_i */
         A_diag_i_new[local_i + 1] = nnz_diag_new;
         A_offd_i_new[local_i + 1] = nnz_offd_new;
      } /* for i, each row */

      dense += blockSize * blockSize;
   } /* for each block */

   /* done with all rows */
   /* resize properly */
   A_diag_j_new = hypre_TReAlloc(A_diag_j_new, HYPRE_Int,     nnz_diag_new, HYPRE_MEMORY_HOST);
   A_diag_a_new = hypre_TReAlloc(A_diag_a_new, HYPRE_Complex, nnz_diag_new, HYPRE_MEMORY_HOST);
   A_offd_j_new = hypre_TReAlloc(A_offd_j_new, HYPRE_Int,     nnz_offd_new, HYPRE_MEMORY_HOST);
   A_offd_a_new = hypre_TReAlloc(A_offd_a_new, HYPRE_Complex, nnz_offd_new, HYPRE_MEMORY_HOST);

   /* readjust col_map_offd_new */
   for (i = 0; i < num_cols_A_offd_new; i++)
   {
      marker_newoffd[i] = -1;
   }
   for (i = 0; i < nnz_offd_new; i++)
   {
      j = A_offd_j_new[i];
      if (marker_newoffd[j] == -1)
      {
         marker_newoffd[j] = 1;
      }
   }
   for (i = 0, j = 0; i < num_cols_A_offd_new; i++)
   {
      if (marker_newoffd[i] == 1)
      {
         col_map_offd_A_new[j] = col_map_offd_A_new[i];
         marker_newoffd[i] = j++;
      }
   }
   num_cols_A_offd_new = j;

   for (i = 0; i < nnz_offd_new; i++)
   {
      j = marker_newoffd[A_offd_j_new[i]];
      hypre_assert(j >= 0 && j < num_cols_A_offd_new);
      A_offd_j_new[i] = j;
   }

   /* Now, we should have everything of Parcsr matrix As */
   Anew = hypre_ParCSRMatrixCreate(comm,
                                   nrow_global,
                                   ncol_global,
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParCSRMatrixColStarts(A),
                                   num_cols_A_offd_new,
                                   nnz_diag_new,
                                   nnz_offd_new);

   Anew_diag = hypre_ParCSRMatrixDiag(Anew);
   hypre_CSRMatrixData(Anew_diag) = A_diag_a_new;
   hypre_CSRMatrixI(Anew_diag)    = A_diag_i_new;
   hypre_CSRMatrixJ(Anew_diag)    = A_diag_j_new;

   Anew_offd = hypre_ParCSRMatrixOffd(Anew);
   hypre_CSRMatrixData(Anew_offd) = A_offd_a_new;
   hypre_CSRMatrixI(Anew_offd)    = A_offd_i_new;
   hypre_CSRMatrixJ(Anew_offd)    = A_offd_j_new;

   hypre_ParCSRMatrixColMapOffd(Anew) = col_map_offd_A_new;

   hypre_ParCSRMatrixSetNumNonzeros(Anew);
   hypre_ParCSRMatrixDNumNonzeros(Anew) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(Anew);
   //printf("nnz_diag %d --> %d, nnz_offd %d --> %d\n", nnz_diag, nnz_diag_new, nnz_offd, nnz_offd_new);

   /* create CommPkg of Anew */
   hypre_MatvecCommPkgCreate(Anew);

   *As = Anew;

   /*
   if (bdiaginv)
   {
      *bdiaginv = dense_all;
   }
   else
   {
      hypre_TFree(dense_all, HYPRE_MEMORY_HOST);
   }
   */
   /* save diagonal blocks in A */
   A->bdiag_size = blockSize;
   A->bdiaginv = dense_all;

   /* free workspace */
   hypre_TFree(IPIV, HYPRE_MEMORY_HOST);
   hypre_TFree(dgetri_work, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(marker_newoffd, HYPRE_MEMORY_HOST);
   hypre_TFree(offd2new, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(A_ext);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParcsrGetExternalRowsInit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParcsrGetExternalRowsInit( hypre_ParCSRMatrix   *A,
                                 HYPRE_Int             indices_len,
                                 HYPRE_BigInt         *indices,
                                 hypre_ParCSRCommPkg  *comm_pkg,
                                 HYPRE_Int             want_data,
                                 void                **request_ptr)
{
   HYPRE_UNUSED_VAR(indices);

   MPI_Comm                 comm           = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt             first_col      = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt            *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);

   /* diag part of A */
   hypre_CSRMatrix         *A_diag    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real              *A_diag_a  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_diag_i  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_diag_j  = hypre_CSRMatrixJ(A_diag);

   /* off-diag part of A */
   hypre_CSRMatrix         *A_offd    = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real              *A_offd_a  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_offd_i  = hypre_CSRMatrixI(A_offd);
   HYPRE_Int               *A_offd_j  = hypre_CSRMatrixJ(A_offd);

   hypre_CSRMatrix         *A_ext;
   HYPRE_Int                num_procs, my_id;
   void                   **vrequest;

   HYPRE_Int                i, j, k;
   HYPRE_Int                num_sends, num_rows_send, num_nnz_send, *send_i;
   HYPRE_Int                num_recvs, num_rows_recv, num_nnz_recv, *recv_i;
   HYPRE_Int               *send_jstarts, *recv_jstarts, *send_i_offset;
   HYPRE_BigInt            *send_j, *recv_j;
   HYPRE_Complex           *send_a = NULL, *recv_a = NULL;
   hypre_ParCSRCommPkg     *comm_pkg_j = NULL;
   hypre_ParCSRCommHandle  *comm_handle, *comm_handle_j, *comm_handle_a;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* number of sends (#procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   num_rows_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   num_rows_recv = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);

   /* must be true if indices contains proper offd indices */
   hypre_assert(indices_len == num_rows_recv);

   /* send_i/recv_i:
    * the arrays to send and recv: we first send and recv the row lengths */
   send_i = hypre_TAlloc(HYPRE_Int, num_rows_send, HYPRE_MEMORY_HOST);
   recv_i = hypre_CTAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_HOST);
   /* fill the send array with row lengths */
   for (i = 0, num_nnz_send = 0; i < num_rows_send; i++)
   {
      /* j: row index to send */
      j = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      send_i[i] = A_diag_i[j + 1] - A_diag_i[j] + A_offd_i[j + 1] - A_offd_i[j];
      num_nnz_send += send_i[i];
   }

   /* send this array out: note the shift in recv_i by one (async) */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_i, recv_i + 1);

   /* prepare data to send out. overlap with the above commmunication */
   send_j = hypre_TAlloc(HYPRE_BigInt, num_nnz_send, HYPRE_MEMORY_HOST);
   if (want_data)
   {
      send_a = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_HOST);
   }

   send_i_offset = hypre_TAlloc(HYPRE_Int, num_rows_send + 1, HYPRE_MEMORY_HOST);
   send_i_offset[0] = 0;
   hypre_TMemcpy(send_i_offset + 1, send_i, HYPRE_Int, num_rows_send,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   /* prefix sum. TODO: OMP parallelization */
   for (i = 1; i <= num_rows_send; i++)
   {
      send_i_offset[i] += send_i_offset[i - 1];
   }
   hypre_assert(send_i_offset[num_rows_send] == num_nnz_send);

   /* pointers to each proc in send_j */
   send_jstarts = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i <= num_sends; i++)
   {
      send_jstarts[i] = send_i_offset[hypre_ParCSRCommPkgSendMapStart(comm_pkg, i)];
   }
   hypre_assert(send_jstarts[num_sends] == num_nnz_send);

   /* fill the CSR matrix: j and a */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE private(i,j,k)
#endif
   for (i = 0; i < num_rows_send; i++)
   {
      HYPRE_Int i1 = send_i_offset[i];
      j = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      /* open row j and fill ja and a to send */
      for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
      {
         send_j[i1] = first_col + A_diag_j[k];
         if (want_data)
         {
            send_a[i1] = A_diag_a[k];
         }
         i1++;
      }
      if (num_procs > 1)
      {
         for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
         {
            send_j[i1] = col_map_offd_A[A_offd_j[k]];
            if (want_data)
            {
               send_a[i1] = A_offd_a[k];
            }
            i1++;
         }
      }
      hypre_assert(send_i_offset[i + 1] == i1);
   }

   /* finish the above communication: send_i/recv_i */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust recv_i to ptrs */
   for (i = 1; i <= num_rows_recv; i++)
   {
      recv_i[i] += recv_i[i - 1];
   }
   num_nnz_recv = recv_i[num_rows_recv];
   recv_j = hypre_CTAlloc(HYPRE_BigInt, num_nnz_recv, HYPRE_MEMORY_HOST);
   if (want_data)
   {
      recv_a = hypre_CTAlloc(HYPRE_Complex, num_nnz_recv, HYPRE_MEMORY_HOST);
   }
   recv_jstarts = hypre_CTAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i <= num_recvs; i++)
   {
      j = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_jstarts[i] = recv_i[j];
   }

   /* Create communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_jstarts,
                                    num_sends,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_jstarts,
                                    NULL,
                                    &comm_pkg_j);

   /* init communication */
   /* ja */
   comm_handle_j = hypre_ParCSRCommHandleCreate(21, comm_pkg_j, send_j, recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = hypre_ParCSRCommHandleCreate(1, comm_pkg_j, send_a, recv_a);
   }
   else
   {
      comm_handle_a = NULL;
   }

   /* create A_ext */
   A_ext = hypre_CSRMatrixCreate(num_rows_recv, hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   hypre_CSRMatrixMemoryLocation(A_ext) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI   (A_ext) = recv_i;
   hypre_CSRMatrixBigJ(A_ext) = recv_j;
   hypre_CSRMatrixData(A_ext) = recv_a;

   /* output */
   vrequest = hypre_TAlloc(void *, 4, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;
   vrequest[3] = (void *) comm_pkg_j;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(send_i, HYPRE_MEMORY_HOST);
   hypre_TFree(send_i_offset, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParcsrGetExternalRowsWait
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_ParcsrGetExternalRowsWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *A_ext         = (hypre_CSRMatrix *)        request[2];
   hypre_ParCSRCommPkg    *comm_pkg_j    = (hypre_ParCSRCommPkg *)    request[3];
   HYPRE_BigInt           *send_j        = (HYPRE_BigInt *) hypre_ParCSRCommHandleSendData(
                                              comm_handle_j);

   if (comm_handle_a)
   {
      HYPRE_Complex *send_a = (HYPRE_Complex *) hypre_ParCSRCommHandleSendData(comm_handle_a);
      hypre_ParCSRCommHandleDestroy(comm_handle_a);
      hypre_TFree(send_a, HYPRE_MEMORY_HOST);
   }

   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_TFree(send_j, HYPRE_MEMORY_HOST);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return A_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixAddHost
 *
 * Host (CPU) version of hypre_ParCSRMatrixAdd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixAddHost( HYPRE_Complex        alpha,
                           hypre_ParCSRMatrix  *A,
                           HYPRE_Complex        beta,
                           hypre_ParCSRMatrix  *B,
                           hypre_ParCSRMatrix **C_ptr )
{
   /* ParCSRMatrix data */
   MPI_Comm          comm       = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt      num_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt      num_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   /* HYPRE_BigInt      num_rows_B = hypre_ParCSRMatrixGlobalNumRows(B); */
   /* HYPRE_BigInt      num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B); */

   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int     *rownnz_diag_A = hypre_CSRMatrixRownnz(A_diag);
   HYPRE_Int  num_rownnz_diag_A = hypre_CSRMatrixNumRownnz(A_diag);
   HYPRE_Int    num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int    num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);

   /* off-diag part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int     *rownnz_offd_A = hypre_CSRMatrixRownnz(A_offd);
   HYPRE_Int  num_rownnz_offd_A = hypre_CSRMatrixNumRownnz(A_offd);
   HYPRE_Int    num_rows_offd_A = hypre_CSRMatrixNumRows(A_offd);
   HYPRE_Int    num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int          *A2C_offd;

   /* diag part of B */
   hypre_CSRMatrix    *B_diag   = hypre_ParCSRMatrixDiag(B);
   HYPRE_Int     *rownnz_diag_B = hypre_CSRMatrixRownnz(B_diag);
   HYPRE_Int  num_rownnz_diag_B = hypre_CSRMatrixNumRownnz(B_diag);
   HYPRE_Int    num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   /* HYPRE_Int    num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag); */

   /* off-diag part of B */
   hypre_CSRMatrix    *B_offd   = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int     *rownnz_offd_B = hypre_CSRMatrixRownnz(B_offd);
   HYPRE_Int  num_rownnz_offd_B = hypre_CSRMatrixNumRownnz(B_offd);
   HYPRE_Int    num_rows_offd_B = hypre_CSRMatrixNumRows(B_offd);
   HYPRE_Int    num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_BigInt *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   HYPRE_Int          *B2C_offd;

   /* C data */
   hypre_ParCSRMatrix   *C;
   hypre_CSRMatrix      *C_diag;
   hypre_CSRMatrix      *C_offd;
   HYPRE_BigInt         *col_map_offd_C;
   HYPRE_Int            *C_diag_i, *C_offd_i;
   HYPRE_Int            *rownnz_diag_C = NULL;
   HYPRE_Int            *rownnz_offd_C = NULL;
   HYPRE_Int             num_rownnz_diag_C;
   HYPRE_Int             num_rownnz_offd_C;
   HYPRE_Int             num_rows_diag_C = num_rows_diag_A;
   HYPRE_Int             num_cols_diag_C = num_cols_diag_A;
   HYPRE_Int             num_rows_offd_C = num_rows_offd_A;
   HYPRE_Int             num_cols_offd_C = num_cols_offd_A + num_cols_offd_B;
   HYPRE_Int            *twspace;

   HYPRE_MemoryLocation  memory_location_A = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation  memory_location_B = hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   HYPRE_MemoryLocation  memory_location_C = hypre_max(memory_location_A, memory_location_B);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Allocate memory */
   twspace  = hypre_TAlloc(HYPRE_Int, hypre_NumThreads(), HYPRE_MEMORY_HOST);
   C_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A + 1, memory_location_C);
   C_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_offd_A + 1, memory_location_C);
   col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);

   /* Compute num_cols_offd_C, A2C_offd, and B2C_offd*/
   A2C_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
   B2C_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);
   hypre_union2(num_cols_offd_A, col_map_offd_A,
                num_cols_offd_B, col_map_offd_B,
                &num_cols_offd_C, col_map_offd_C,
                A2C_offd, B2C_offd);

   /* Set nonzero rows data of diag_C */
   num_rownnz_diag_C = num_rows_diag_A;
   if ((num_rownnz_diag_A < num_rows_diag_A) &&
       (num_rownnz_diag_B < num_rows_diag_B))
   {
      hypre_IntArray arr_diagA;
      hypre_IntArray arr_diagB;
      hypre_IntArray arr_diagC;

      hypre_IntArrayData(&arr_diagA) = rownnz_diag_A;
      hypre_IntArrayData(&arr_diagB) = rownnz_diag_B;
      hypre_IntArraySize(&arr_diagA) = num_rownnz_diag_A;
      hypre_IntArraySize(&arr_diagB) = num_rownnz_diag_B;
      hypre_IntArrayMemoryLocation(&arr_diagC) = memory_location_C;

      hypre_IntArrayMergeOrdered(&arr_diagA, &arr_diagB, &arr_diagC);

      num_rownnz_diag_C = hypre_IntArraySize(&arr_diagC);
      rownnz_diag_C     = hypre_IntArrayData(&arr_diagC);
   }

   /* Set nonzero rows data of offd_C */
   num_rownnz_offd_C = num_rows_offd_A;
   if ((num_rownnz_offd_A < num_rows_offd_A) &&
       (num_rownnz_offd_B < num_rows_offd_B))
   {
      hypre_IntArray arr_offdA;
      hypre_IntArray arr_offdB;
      hypre_IntArray arr_offdC;

      hypre_IntArrayData(&arr_offdA) = rownnz_offd_A;
      hypre_IntArrayData(&arr_offdB) = rownnz_offd_B;
      hypre_IntArraySize(&arr_offdA) = num_rownnz_offd_A;
      hypre_IntArraySize(&arr_offdB) = num_rownnz_offd_B;
      hypre_IntArrayMemoryLocation(&arr_offdC) = memory_location_C;

      hypre_IntArrayMergeOrdered(&arr_offdA, &arr_offdB, &arr_offdC);

      num_rownnz_offd_C = hypre_IntArraySize(&arr_offdC);
      rownnz_offd_C     = hypre_IntArrayData(&arr_offdC);
   }

   /* Set diag_C */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      HYPRE_Int   ii, num_threads;
      HYPRE_Int   size, rest, ns, ne;
      HYPRE_Int  *marker_diag;
      HYPRE_Int  *marker_offd;

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();

      /*-----------------------------------------------------------------------
       *  Compute C_diag = alpha*A_diag + beta*B_diag
       *-----------------------------------------------------------------------*/

      size = num_rownnz_diag_C / num_threads;
      rest = num_rownnz_diag_C - size * num_threads;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      marker_diag = hypre_TAlloc(HYPRE_Int, num_cols_diag_A, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker_diag,
                                  NULL, NULL, A_diag, B_diag,
                                  num_rows_diag_C, num_rownnz_diag_C,
                                  num_cols_diag_C, rownnz_diag_C,
                                  memory_location_C, C_diag_i, &C_diag);
      hypre_CSRMatrixAddSecondPass(ns, ne, marker_diag,
                                   NULL, NULL, rownnz_diag_C,
                                   alpha, beta, A_diag, B_diag, C_diag);
      hypre_TFree(marker_diag, HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       *  Compute C_offd = alpha*A_offd + beta*B_offd
       *-----------------------------------------------------------------------*/

      size = num_rownnz_offd_C / num_threads;
      rest = num_rownnz_offd_C - size * num_threads;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker_offd,
                                  A2C_offd, B2C_offd, A_offd, B_offd,
                                  num_rows_offd_C, num_rownnz_offd_C,
                                  num_cols_offd_C, rownnz_offd_C,
                                  memory_location_C, C_offd_i, &C_offd);
      hypre_CSRMatrixAddSecondPass(ns, ne, marker_offd,
                                   A2C_offd, B2C_offd, rownnz_offd_C,
                                   alpha, beta, A_offd, B_offd, C_offd);
      hypre_TFree(marker_offd, HYPRE_MEMORY_HOST);
   } /* end of omp parallel region */

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);
   hypre_TFree(A2C_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(B2C_offd, HYPRE_MEMORY_HOST);

   /* Create ParCSRMatrix C */
   C = hypre_ParCSRMatrixCreate(comm,
                                num_rows_A,
                                num_cols_A,
                                hypre_ParCSRMatrixRowStarts(A),
                                hypre_ParCSRMatrixColStarts(A),
                                num_cols_offd_C,
                                hypre_CSRMatrixNumNonzeros(C_diag),
                                hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;
   hypre_ParCSRMatrixOffd(C) = C_offd;
   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   hypre_ParCSRMatrixSetNumNonzeros(C);
   hypre_ParCSRMatrixDNumNonzeros(C) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(C);

   /* create CommPkg of C */
   hypre_MatvecCommPkgCreate(C);

   *C_ptr = C;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixAdd
 *
 * Interface for Host/Device functions for computing C = alpha*A + beta*B
 *
 * A and B are assumed to have the same row and column partitionings
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixAdd( HYPRE_Complex        alpha,
                       hypre_ParCSRMatrix  *A,
                       HYPRE_Complex        beta,
                       hypre_ParCSRMatrix  *B,
                       hypre_ParCSRMatrix **C_ptr )
{
   hypre_assert(hypre_ParCSRMatrixGlobalNumRows(A) == hypre_ParCSRMatrixGlobalNumRows(B));
   hypre_assert(hypre_ParCSRMatrixGlobalNumCols(A) == hypre_ParCSRMatrixGlobalNumCols(B));
   hypre_assert(hypre_ParCSRMatrixNumRows(A) == hypre_ParCSRMatrixNumRows(B));
   hypre_assert(hypre_ParCSRMatrixNumCols(A) == hypre_ParCSRMatrixNumCols(B));

#if defined(HYPRE_USING_GPU)
   if ( hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                              hypre_ParCSRMatrixMemoryLocation(B) ) == HYPRE_EXEC_DEVICE )
   {
      hypre_ParCSRMatrixAddDevice(alpha, A, beta, B, C_ptr);
   }
   else
#endif
   {
      hypre_ParCSRMatrixAddHost(alpha, A, beta, B, C_ptr);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixFnorm
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_ParCSRMatrixFnorm( hypre_ParCSRMatrix *A )
{
   MPI_Comm   comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Real f_diag, f_offd, local_result, result;

   f_diag = hypre_CSRMatrixFnorm(hypre_ParCSRMatrixDiag(A));
   f_offd = hypre_CSRMatrixFnorm(hypre_ParCSRMatrixOffd(A));
   local_result = f_diag * f_diag + f_offd * f_offd;

   hypre_MPI_Allreduce(&local_result, &result, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   return hypre_sqrt(result);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixInfNorm
 *
 * Computes the infinity norm of A:
 *
 *       norm = max_{i} sum_{j} |A_{ij}|
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixInfNorm( hypre_ParCSRMatrix  *A,
                           HYPRE_Real          *norm )
{
   MPI_Comm            comm     = hypre_ParCSRMatrixComm(A);

   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int    num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);

   /* off-diag part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);

   /* Local variables */
   HYPRE_Int           i, j;
   HYPRE_Real          maxsum = 0.0;
   HYPRE_Real          rowsum;

#ifdef _MSC_VER
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,rowsum)
#endif
   {
      HYPRE_Real maxsum_local;

      maxsum_local = 0.0;
#ifdef HYPRE_USING_OPENMP
      #pragma omp for HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows_diag_A; i++)
      {
         rowsum = 0.0;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            rowsum += hypre_cabs(A_diag_a[j]);
         }
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            rowsum += hypre_cabs(A_offd_a[j]);
         }

         maxsum_local = hypre_max(maxsum_local, rowsum);
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp critical
#endif
      {
         maxsum = hypre_max(maxsum, maxsum_local);
      }
   }
#else
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,rowsum) reduction(max:maxsum) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows_diag_A; i++)
   {
      rowsum = 0.0;
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rowsum += hypre_cabs(A_diag_a[j]);
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rowsum += hypre_cabs(A_offd_a[j]);
      }

      maxsum = hypre_max(maxsum, rowsum);
   }
#endif

   hypre_MPI_Allreduce(&maxsum, norm, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ExchangeExternalRowsInit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExchangeExternalRowsInit( hypre_CSRMatrix      *B_ext,
                                hypre_ParCSRCommPkg  *comm_pkg_A,
                                void                **request_ptr)
{
   MPI_Comm   comm             = hypre_ParCSRCommPkgComm(comm_pkg_A);
   HYPRE_Int  num_recvs        = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   HYPRE_Int *recv_procs       = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   HYPRE_Int *recv_vec_starts  = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   HYPRE_Int  num_sends        = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   HYPRE_Int *send_procs       = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   HYPRE_Int *send_map_starts  = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   HYPRE_Int  num_elmts_send   = send_map_starts[num_sends];
   HYPRE_Int  num_elmts_recv   = recv_vec_starts[num_recvs];

   HYPRE_Int     *B_ext_i      = B_ext ? hypre_CSRMatrixI(B_ext) : NULL;
   HYPRE_BigInt  *B_ext_j      = B_ext ? hypre_CSRMatrixBigJ(B_ext) : NULL;
   HYPRE_Complex *B_ext_data   = B_ext ? hypre_CSRMatrixData(B_ext) : NULL;
   HYPRE_Int      B_ext_ncols  = B_ext ? hypre_CSRMatrixNumCols(B_ext) : 0;
   HYPRE_Int      B_ext_nrows  = B_ext ? hypre_CSRMatrixNumRows(B_ext) : 0;
   HYPRE_Int     *B_ext_rownnz = hypre_CTAlloc(HYPRE_Int, B_ext_nrows, HYPRE_MEMORY_HOST);

   hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   hypre_CSRMatrix *B_int;
   HYPRE_Int        B_int_nrows = num_elmts_send;
   HYPRE_Int        B_int_ncols = B_ext_ncols;
   HYPRE_Int       *B_int_i     = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_BigInt    *B_int_j     = NULL;
   HYPRE_Complex   *B_int_data  = NULL;
   HYPRE_Int        B_int_nnz;

   hypre_ParCSRCommHandle *comm_handle, *comm_handle_j, *comm_handle_a;
   hypre_ParCSRCommPkg    *comm_pkg_j = NULL;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int i;
   HYPRE_Int num_procs;
   void **vrequest;

   hypre_MPI_Comm_size(comm, &num_procs);

   jdata_send_map_starts = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * B_ext_rownnz contains the number of elements of row j
    * (to be determined through send_map_elmnts on the receiving end)
    *--------------------------------------------------------------------------*/
   for (i = 0; i < B_ext_nrows; i++)
   {
      B_ext_rownnz[i] = B_ext_i[i + 1] - B_ext_i[i];
   }

   /*--------------------------------------------------------------------------
    * initialize communication: send/recv the row nnz
    * (note the use of comm_pkg_A, mode 12, as in transpose matvec
    *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, B_ext_rownnz, B_int_i + 1);

   jdata_recv_vec_starts = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts[0] = 0;
   for (i = 1; i <= num_recvs; i++)
   {
      jdata_recv_vec_starts[i] = B_ext_i[recv_vec_starts[i]];
   }

   /* Create communication package -  note the order of send/recv is reversed */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL,
                                    &comm_pkg_j);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*--------------------------------------------------------------------------
    * compute B_int: row nnz to row ptrs
    *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   for (i = 1; i <= B_int_nrows; i++)
   {
      B_int_i[i] += B_int_i[i - 1];
   }

   B_int_nnz = B_int_i[B_int_nrows];

   B_int_j    = hypre_TAlloc(HYPRE_BigInt,  B_int_nnz, HYPRE_MEMORY_HOST);
   B_int_data = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_HOST);

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i[send_map_starts[i]];
   }

   /* send/recv CSR rows */
   comm_handle_a = hypre_ParCSRCommHandleCreate( 1, comm_pkg_j, B_ext_data, B_int_data);
   comm_handle_j = hypre_ParCSRCommHandleCreate(21, comm_pkg_j, B_ext_j, B_int_j);

   /* create CSR */
   B_int = hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   hypre_CSRMatrixMemoryLocation(B_int) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(B_int)    = B_int_i;
   hypre_CSRMatrixBigJ(B_int) = B_int_j;
   hypre_CSRMatrixData(B_int) = B_int_data;

   /* output */
   vrequest = hypre_TAlloc(void *, 4, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int;
   vrequest[3] = (void *) comm_pkg_j;

   *request_ptr = (void *) vrequest;

   hypre_TFree(B_ext_rownnz, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ExchangeExternalRowsWait
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_ExchangeExternalRowsWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *B_int         = (hypre_CSRMatrix *)        request[2];
   hypre_ParCSRCommPkg    *comm_pkg_j    = (hypre_ParCSRCommPkg *)    request[3];

   /* communication done */
   hypre_ParCSRCommHandleDestroy(comm_handle_a);
   hypre_ParCSRCommHandleDestroy(comm_handle_j);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return B_int;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractSubmatrixFC
 *
 * extract submatrix A_{FF}, A_{FC}, A_{CF} or A_{CC}
 * char job[2] = "FF", "FC", "CF" or "CC"
 *
 * TODO (VPM): Can we do the same with hypre_ParCSRMatrixGenerateFFFC?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixExtractSubmatrixFC( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      HYPRE_BigInt        *cpts_starts,
                                      const char          *job,
                                      hypre_ParCSRMatrix **B_ptr,
                                      HYPRE_Real           strength_thresh)
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   //HYPRE_Int          *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);

   hypre_ParCSRMatrix *B;
   hypre_CSRMatrix    *B_diag, *B_offd;
   HYPRE_Real         *B_maxel_row;
   HYPRE_Int          *B_diag_i, *B_diag_j, *B_offd_i, *B_offd_j;
   HYPRE_Complex      *B_diag_a, *B_offd_a;
   HYPRE_Int           num_cols_B_offd;
   HYPRE_BigInt       *col_map_offd_B;

   HYPRE_Int           i, j, k, k1, k2;
   HYPRE_BigInt        B_nrow_global, B_ncol_global;
   HYPRE_Int           A_nlocal, B_nrow_local, B_ncol_local,
                       B_nnz_diag, B_nnz_offd;
   HYPRE_BigInt        total_global_fpts, total_global_cpts, fpts_starts[2];
   HYPRE_Int           nf_local, nc_local = 0;
   HYPRE_BigInt        big_nf_local;
   HYPRE_Int           row_set, col_set;
   HYPRE_BigInt       *B_row_starts, *B_col_starts, B_first_col;
   HYPRE_Int           my_id, num_procs;
   HYPRE_Int          *sub_idx_diag;
   HYPRE_BigInt       *sub_idx_offd;
   HYPRE_Int           num_sends;
   HYPRE_BigInt       *send_buf_data;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   row_set = job[0] == 'F' ? -1 : 1;
   col_set = job[1] == 'F' ? -1 : 1;

   A_nlocal = hypre_CSRMatrixNumRows(A_diag);

   /*-------------- global number of C points and local C points
    *               assuming cpts_starts is given */
   if (row_set == 1 || col_set == 1)
   {
      if (my_id == (num_procs - 1))
      {
         total_global_cpts = cpts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nc_local = (HYPRE_Int)(cpts_starts[1] - cpts_starts[0]);
   }

   /*-------------- global number of F points, local F points, and F starts */
   if (row_set == -1 || col_set == -1)
   {
      nf_local = 0;
      for (i = 0; i < A_nlocal; i++)
      {
         if (CF_marker[i] < 0)
         {
            nf_local++;
         }
      }
      big_nf_local = (HYPRE_BigInt) nf_local;
      hypre_MPI_Scan(&big_nf_local, fpts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      fpts_starts[0] = fpts_starts[1] - nf_local;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = fpts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   if (row_set == -1 && col_set == -1)
   {
      /* FF */
      B_nrow_local = nf_local;
      B_ncol_local = nf_local;
      B_nrow_global = total_global_fpts;
      B_ncol_global = total_global_fpts;

      B_row_starts = B_col_starts = fpts_starts;
   }
   else if (row_set == -1 && col_set == 1)
   {
      /* FC */
      B_nrow_local = nf_local;
      B_ncol_local = nc_local;
      B_nrow_global = total_global_fpts;
      B_ncol_global = total_global_cpts;

      B_row_starts = fpts_starts;
      B_col_starts = cpts_starts;
   }
   else if (row_set == 1 && col_set == -1)
   {
      /* CF */
      B_nrow_local = nc_local;
      B_ncol_local = nf_local;
      B_nrow_global = total_global_cpts;
      B_ncol_global = total_global_fpts;

      B_row_starts = cpts_starts;
      B_col_starts = fpts_starts;
   }
   else
   {
      /* CC */
      B_nrow_local = nc_local;
      B_ncol_local = nc_local;
      B_nrow_global = total_global_cpts;
      B_ncol_global = total_global_cpts;

      B_row_starts = B_col_starts = cpts_starts;
   }

   /* global index of my first col */
   B_first_col = B_col_starts[0];

   /* sub_idx_diag: [local] mapping from F+C to F/C, if not selected, be -1 */
   sub_idx_diag = hypre_TAlloc(HYPRE_Int, A_nlocal, HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < A_nlocal; i++)
   {
      HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i == col_set)
      {
         sub_idx_diag[i] = k++;
      }
      else
      {
         sub_idx_diag[i] = -1;
      }
   }

   hypre_assert(k == B_ncol_local);

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_data = hypre_TAlloc(HYPRE_BigInt,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);
   k = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      HYPRE_Int si = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      HYPRE_Int ei = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      /* loop through all elems to send_proc[i] */
      for (j = si; j < ei; j++)
      {
         /* j1: local idx */
         HYPRE_BigInt j1 = sub_idx_diag[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         if (j1 != -1)
         {
            /* adjust j1 to B global idx */
            j1 += B_first_col;
         }
         send_buf_data[k++] = j1;
      }
   }

   hypre_assert(k == hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   /* recv buffer */
   sub_idx_offd = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);
   /* create a handle to start communication. 11: for integer */
   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf_data, sub_idx_offd);
   /* destroy the handle to finish communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0, num_cols_B_offd = 0; i < num_cols_A_offd; i++)
   {
      if (sub_idx_offd[i] != -1)
      {
         num_cols_B_offd ++;
      }
   }
   col_map_offd_B = hypre_TAlloc(HYPRE_BigInt, num_cols_B_offd, HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (sub_idx_offd[i] != -1)
      {
         col_map_offd_B[k] = sub_idx_offd[i];
         sub_idx_offd[i] = k++;
      }
   }

   hypre_assert(k == num_cols_B_offd);

   /* count nnz and set ia */
   B_nnz_diag = B_nnz_offd = 0;
   B_maxel_row = hypre_TAlloc(HYPRE_Real, B_nrow_local, HYPRE_MEMORY_HOST);
   B_diag_i = hypre_TAlloc(HYPRE_Int, B_nrow_local + 1, HYPRE_MEMORY_HOST);
   B_offd_i = hypre_TAlloc(HYPRE_Int, B_nrow_local + 1, HYPRE_MEMORY_HOST);
   B_diag_i[0] = B_offd_i[0] = 0;

   for (i = 0, k = 0; i < A_nlocal; i++)
   {
      HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i != row_set)
      {
         continue;
      }
      k++;

      // Get max abs-value element of this row
      HYPRE_Real temp_max = 0;
      if (strength_thresh > 0)
      {
         for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
         {
            if (hypre_cabs(A_diag_a[j]) > temp_max)
            {
               temp_max = hypre_cabs(A_diag_a[j]);
            }
         }
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            if (hypre_cabs(A_offd_a[j]) > temp_max)
            {
               temp_max = hypre_cabs(A_offd_a[j]);
            }
         }
      }
      B_maxel_row[k - 1] = temp_max;

      // add one for diagonal element
      j = A_diag_i[i];
      if (sub_idx_diag[A_diag_j[j]] != -1)
      {
         B_nnz_diag++;
      }

      // Count nnzs larger than tolerance times max row element
      for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
      {
         if ( (sub_idx_diag[A_diag_j[j]] != -1) &&
              (hypre_cabs(A_diag_a[j]) > (strength_thresh * temp_max)) )
         {
            B_nnz_diag++;
         }
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         if ( (sub_idx_offd[A_offd_j[j]] != -1) &&
              (hypre_cabs(A_offd_a[j]) > (strength_thresh * temp_max)) )
         {
            B_nnz_offd++;
         }
      }
      B_diag_i[k] = B_nnz_diag;
      B_offd_i[k] = B_nnz_offd;
   }

   hypre_assert(k == B_nrow_local);

   B_diag_j = hypre_TAlloc(HYPRE_Int,     B_nnz_diag, HYPRE_MEMORY_HOST);
   B_diag_a = hypre_TAlloc(HYPRE_Complex, B_nnz_diag, HYPRE_MEMORY_HOST);
   B_offd_j = hypre_TAlloc(HYPRE_Int,     B_nnz_offd, HYPRE_MEMORY_HOST);
   B_offd_a = hypre_TAlloc(HYPRE_Complex, B_nnz_offd, HYPRE_MEMORY_HOST);

   for (i = 0, k = 0, k1 = 0, k2 = 0; i < A_nlocal; i++)
   {
      HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i != row_set)
      {
         continue;
      }
      HYPRE_Real maxel = B_maxel_row[k];
      k++;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         HYPRE_Int j1 = sub_idx_diag[A_diag_j[j]];
         if ( (j1 != -1) && ( (hypre_cabs(A_diag_a[j]) > (strength_thresh * maxel)) || j == A_diag_i[i] ) )
         {
            B_diag_j[k1] = j1;
            B_diag_a[k1] = A_diag_a[j];
            k1++;
         }
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         HYPRE_Int j1 = (HYPRE_Int) sub_idx_offd[A_offd_j[j]];
         if ((j1 != -1) && (hypre_cabs(A_offd_a[j]) > (strength_thresh * maxel)))
         {
            hypre_assert(j1 >= 0 && j1 < num_cols_B_offd);
            B_offd_j[k2] = j1;
            B_offd_a[k2] = A_offd_a[j];
            k2++;
         }
      }
   }

   hypre_assert(k1 == B_nnz_diag && k2 == B_nnz_offd);

   /* ready to create B = A(rowset, colset) */
   B = hypre_ParCSRMatrixCreate(comm,
                                B_nrow_global,
                                B_ncol_global,
                                B_row_starts,
                                B_col_starts,
                                num_cols_B_offd,
                                B_nnz_diag,
                                B_nnz_offd);

   B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrixMemoryLocation(B_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixData(B_diag) = B_diag_a;
   hypre_CSRMatrixI(B_diag)    = B_diag_i;
   hypre_CSRMatrixJ(B_diag)    = B_diag_j;

   B_offd = hypre_ParCSRMatrixOffd(B);
   hypre_CSRMatrixMemoryLocation(B_offd) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixData(B_offd) = B_offd_a;
   hypre_CSRMatrixI(B_offd)    = B_offd_i;
   hypre_CSRMatrixJ(B_offd)    = B_offd_j;

   hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;

   hypre_ParCSRMatrixSetNumNonzeros(B);
   hypre_ParCSRMatrixDNumNonzeros(B) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(B);

   hypre_MatvecCommPkgCreate(B);

   *B_ptr = B;

   hypre_TFree(B_maxel_row, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(sub_idx_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(sub_idx_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDropSmallEntriesHost
 *
 * drop the entries that are not on the diagonal and smaller than:
 *    type 0: tol (TODO)
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDropSmallEntriesHost( hypre_ParCSRMatrix *A,
                                        HYPRE_Real          tol,
                                        HYPRE_Int           type)
{
   HYPRE_Int i, j, k, nnz_diag, nnz_offd, A_diag_i_i, A_offd_i_i;

   MPI_Comm         comm     = hypre_ParCSRMatrixComm(A);
   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int  num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int *marker_offd = NULL;

   HYPRE_BigInt first_row  = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int nrow_local = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int my_id, num_procs;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);

   nnz_diag = nnz_offd = A_diag_i_i = A_offd_i_i = 0;
   for (i = 0; i < nrow_local; i++)
   {
      /* compute row norm */
      HYPRE_Real row_nrm = 0.0;
      for (j = A_diag_i_i; j < A_diag_i[i + 1]; j++)
      {
         HYPRE_Complex v = A_diag_a[j];
         if (type == 1)
         {
            row_nrm += hypre_cabs(v);
         }
         else if (type == 2)
         {
            row_nrm += v * v;
         }
         else
         {
            row_nrm = hypre_max(row_nrm, hypre_cabs(v));
         }
      }
      if (num_procs > 1)
      {
         for (j = A_offd_i_i; j < A_offd_i[i + 1]; j++)
         {
            HYPRE_Complex v = A_offd_a[j];
            if (type == 1)
            {
               row_nrm += hypre_cabs(v);
            }
            else if (type == 2)
            {
               row_nrm += v * v;
            }
            else
            {
               row_nrm = hypre_max(row_nrm, hypre_cabs(v));
            }
         }
      }

      if (type == 2)
      {
         row_nrm = hypre_sqrt(row_nrm);
      }

      /* drop small entries based on tol and row norm */
      for (j = A_diag_i_i; j < A_diag_i[i + 1]; j++)
      {
         HYPRE_Int     col = A_diag_j[j];
         HYPRE_Complex val = A_diag_a[j];
         if (i == col || hypre_cabs(val) >= tol * row_nrm)
         {
            A_diag_j[nnz_diag] = col;
            A_diag_a[nnz_diag] = val;
            nnz_diag ++;
         }
      }
      if (num_procs > 1)
      {
         for (j = A_offd_i_i; j < A_offd_i[i + 1]; j++)
         {
            HYPRE_Int     col = A_offd_j[j];
            HYPRE_Complex val = A_offd_a[j];
            /* in normal cases: diagonal entry should not
             * appear in A_offd (but this can still be possible) */
            if (i + first_row == col_map_offd_A[col] || hypre_cabs(val) >= tol * row_nrm)
            {
               if (0 == marker_offd[col])
               {
                  marker_offd[col] = 1;
               }
               A_offd_j[nnz_offd] = col;
               A_offd_a[nnz_offd] = val;
               nnz_offd ++;
            }
         }
      }
      A_diag_i_i = A_diag_i[i + 1];
      A_offd_i_i = A_offd_i[i + 1];
      A_diag_i[i + 1] = nnz_diag;
      A_offd_i[i + 1] = nnz_offd;
   }

   hypre_CSRMatrixNumNonzeros(A_diag) = nnz_diag;
   hypre_CSRMatrixNumNonzeros(A_offd) = nnz_offd;
   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(A) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A);

   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i])
      {
         col_map_offd_A[k] = col_map_offd_A[i];
         marker_offd[i] = k++;
      }
   }
   /* num_cols_A_offd = k; */
   hypre_CSRMatrixNumCols(A_offd) = k;
   for (i = 0; i < nnz_offd; i++)
   {
      A_offd_j[i] = marker_offd[A_offd_j[i]];
   }

   if ( hypre_ParCSRMatrixCommPkg(A) )
   {
      hypre_MatvecCommPkgDestroy( hypre_ParCSRMatrixCommPkg(A) );
   }
   hypre_MatvecCommPkgCreate(A);

   hypre_TFree(marker_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDropSmallEntries
 *
 * drop the entries that are not on the diagonal and smaller than
 *    type 0: tol
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row)
 *    NOTE: some type options above unavailable on either host or device
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDropSmallEntries( hypre_ParCSRMatrix *A,
                                    HYPRE_Real          tol,
                                    HYPRE_Int           type)
{
   if (tol <= 0.0)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixDropSmallEntriesDevice(A, tol, type);
   }
   else
#endif
   {
      hypre_ParCSRMatrixDropSmallEntriesHost(A, tol, type);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixScale
 *
 * Computes A = scalar * A
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixScale(hypre_ParCSRMatrix *A,
                        HYPRE_Complex       scalar)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrixScale(A_diag, scalar);
   hypre_CSRMatrixScale(A_offd, scalar);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDiagScaleHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDiagScaleHost( hypre_ParCSRMatrix *par_A,
                                 hypre_ParVector    *par_ld,
                                 hypre_ParVector    *par_rd )
{
   /* Input variables */
   hypre_ParCSRCommPkg   *comm_pkg  = hypre_ParCSRMatrixCommPkg(par_A);
   HYPRE_Int              num_sends;
   HYPRE_Int             *send_map_elmts;
   HYPRE_Int             *send_map_starts;

   hypre_CSRMatrix       *A_diag        = hypre_ParCSRMatrixDiag(par_A);
   hypre_CSRMatrix       *A_offd        = hypre_ParCSRMatrixOffd(par_A);
   HYPRE_Int              num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_Vector          *ld            = (par_ld) ? hypre_ParVectorLocalVector(par_ld) : NULL;
   hypre_Vector          *rd            = hypre_ParVectorLocalVector(par_rd);
   HYPRE_Complex         *rd_data       = hypre_VectorData(rd);

   /* Local variables */
   HYPRE_Int              i;
   hypre_Vector          *rdbuf;
   HYPRE_Complex         *recv_rdbuf_data;
   HYPRE_Complex         *send_rdbuf_data;

   /*---------------------------------------------------------------------
    * Communication phase
    *--------------------------------------------------------------------*/

   /* Create buffer vectors */
   rdbuf = hypre_SeqVectorCreate(num_cols_offd);

   /* If there exists no CommPkg for A, create it. */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(par_A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(par_A);
   }
   num_sends       = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_elmts  = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);

#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *comm_handle =
      hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);

   hypre_VectorData(rdbuf) = (HYPRE_Complex *)
                             hypre_ParCSRCommHandleRecvDataBuffer(comm_handle);
   hypre_SeqVectorSetDataOwner(rdbuf, 0);

#else
   hypre_ParCSRCommHandle *comm_handle;
#endif

   /* Initialize rdbuf */
   hypre_SeqVectorInitialize_v2(rdbuf, HYPRE_MEMORY_HOST);
   recv_rdbuf_data = hypre_VectorData(rdbuf);

   /* Allocate send buffer for rdbuf */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   send_rdbuf_data = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(comm_handle);
#else
   send_rdbuf_data = hypre_TAlloc(HYPRE_Complex, send_map_starts[num_sends], HYPRE_MEMORY_HOST);
#endif

   /* Pack send data */
#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = send_map_starts[0]; i < send_map_starts[num_sends]; i++)
   {
      send_rdbuf_data[i] = rd_data[send_map_elmts[i]];
   }

   /* Non-blocking communication starts */
#ifdef HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandleStart(comm_handle, HYPRE_MEMORY_HOST, send_rdbuf_data);

#else
   comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 HYPRE_MEMORY_HOST, send_rdbuf_data,
                                                 HYPRE_MEMORY_HOST, recv_rdbuf_data);
#endif

   /*---------------------------------------------------------------------
    * Computation phase
    *--------------------------------------------------------------------*/

   /* A_diag = diag(ld) * A_diag * diag(rd) */
   hypre_CSRMatrixDiagScale(A_diag, ld, rd);

   /* Non-blocking communication ends */
#ifdef HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandleWait(comm_handle, HYPRE_MEMORY_HOST, recv_rdbuf_data);
#else
   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

   /* A_offd = diag(ld) * A_offd * diag(rd) */
   hypre_CSRMatrixDiagScale(A_offd, ld, rdbuf);

   /* Free memory */
   hypre_SeqVectorDestroy(rdbuf);
#if !defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_TFree(send_rdbuf_data, HYPRE_MEMORY_HOST);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDiagScale
 *
 * Computes A = diag(ld) * A * diag(rd), where the diagonal matrices
 * "diag(ld)" and "diag(rd)" are stored as distributed vectors.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDiagScale( hypre_ParCSRMatrix *par_A,
                             hypre_ParVector    *par_ld,
                             hypre_ParVector    *par_rd )
{
   /* Input variables */
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(par_A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(par_A);
   hypre_Vector       *ld;

   /* Sanity check */
   if (!par_rd && !par_ld)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Scaling matrices are not set!\n");
      return hypre_error_flag;
   }

   /* Perform row scaling only (no communication) */
   if (!par_rd && par_ld)
   {
      ld = hypre_ParVectorLocalVector(par_ld);

      hypre_CSRMatrixDiagScale(A_diag, ld, NULL);
      hypre_CSRMatrixDiagScale(A_offd, ld, NULL);

      return hypre_error_flag;
   }

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(par_A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixDiagScaleDevice(par_A, par_ld, par_rd);
   }
   else
#endif
   {
      hypre_ParCSRMatrixDiagScaleHost(par_A, par_ld, par_rd);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixReorder:
 *
 * Reorders the column and data arrays of a the diagonal component of a square
 * ParCSR matrix, such that the first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixReorder(hypre_ParCSRMatrix *A)
{
   HYPRE_BigInt      nrows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt      ncols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   hypre_CSRMatrix  *A_diag  = hypre_ParCSRMatrixDiag(A);

   if (nrows_A != ncols_A)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Matrix should be square!\n");
      return hypre_error_flag;
   }

   hypre_CSRMatrixReorder(A_diag);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCompressOffdMap
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixCompressOffdMap(hypre_ParCSRMatrix *A)
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixCompressOffdMapDevice(A);
   }
#else
   // RL: I guess it's not needed for the host code [?]
   HYPRE_UNUSED_VAR(A);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleVectorHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRDiagScaleVectorHost( hypre_ParCSRMatrix *par_A,
                                 hypre_ParVector    *par_y,
                                 hypre_ParVector    *par_x )
{
   /* Local Matrix and Vectors */
   hypre_CSRMatrix    *A_diag        = hypre_ParCSRMatrixDiag(par_A);
   hypre_Vector       *x             = hypre_ParVectorLocalVector(par_x);
   hypre_Vector       *y             = hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   HYPRE_Complex      *x_data        = hypre_VectorData(x);
   HYPRE_Int           x_num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int           x_vecstride   = hypre_VectorVectorStride(x);

   /* Local vector y info */
   HYPRE_Complex      *y_data        = hypre_VectorData(y);
   HYPRE_Int           y_vecstride   = hypre_VectorVectorStride(y);

   /* Local matrix A info */
   HYPRE_Complex      *A_data        = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           num_rows      = hypre_CSRMatrixNumRows(A_diag);

   /* Local variables */
   HYPRE_Int           i, k;
   HYPRE_Complex       coef;

   switch (x_num_vectors)
   {
      case 1:
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            x_data[i] = y_data[i] / A_data[A_i[i]];
         }
         break;

      case 2:
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i + x_vecstride] = y_data[i + y_vecstride] * coef;
         }
         break;

      case 3:
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i +     x_vecstride] = y_data[i +     y_vecstride] * coef;
            x_data[i + 2 * x_vecstride] = y_data[i + 2 * y_vecstride] * coef;
         }
         break;

      case 4:
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i +     x_vecstride] = y_data[i +     y_vecstride] * coef;
            x_data[i + 2 * x_vecstride] = y_data[i + 2 * y_vecstride] * coef;
            x_data[i + 3 * x_vecstride] = y_data[i + 3 * y_vecstride] * coef;
         }
         break;

      default:
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, k, coef) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            for (k = 0; k < x_num_vectors; k++)
            {
               x_data[i + k * x_vecstride] = y_data[i + k * y_vecstride] * coef;
            }
         }
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRDiagScaleVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRDiagScaleVector( hypre_ParCSRMatrix *par_A,
                             hypre_ParVector    *par_y,
                             hypre_ParVector    *par_x )
{
   /* Local Matrix and Vectors */
   hypre_CSRMatrix    *A_diag        = hypre_ParCSRMatrixDiag(par_A);
   hypre_Vector       *x             = hypre_ParVectorLocalVector(par_x);
   hypre_Vector       *y             = hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   HYPRE_Int           x_size        = hypre_VectorSize(x);
   HYPRE_Int           x_num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int           x_vecstride   = hypre_VectorVectorStride(x);

   /* Local vector y info */
   HYPRE_Int           y_size        = hypre_VectorSize(y);
   HYPRE_Int           y_num_vectors = hypre_VectorNumVectors(y);
   HYPRE_Int           y_vecstride   = hypre_VectorVectorStride(y);

   /* Local matrix A info */
   HYPRE_Int           num_rows      = hypre_CSRMatrixNumRows(A_diag);

   /*---------------------------------------------
    * Sanity checks
    *---------------------------------------------*/

   if (x_num_vectors != y_num_vectors)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error! incompatible number of vectors!\n");
      return hypre_error_flag;
   }

   if (num_rows != x_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error! incompatible x size!\n");
      return hypre_error_flag;
   }

   if (x_size > 0 && x_vecstride <= 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error! non-positive x vector stride!\n");
      return hypre_error_flag;
   }

   if (y_size > 0 && y_vecstride <= 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error! non-positive y vector stride!\n");
      return hypre_error_flag;
   }

   if (num_rows != y_size)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error! incompatible y size!\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Computation
    *---------------------------------------------*/

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(par_A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRDiagScaleVectorDevice(par_A, par_y, par_x);
   }
   else
#endif
   {
      hypre_ParCSRDiagScaleVectorHost(par_A, par_y, par_x);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlockColSumHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlockColSumHost( hypre_ParCSRMatrix     *A,
                                   hypre_DenseBlockMatrix *B )
{
   /* ParCSRMatrix A */
   HYPRE_MemoryLocation    memory_location   = hypre_ParCSRMatrixMemoryLocation(A);

   /* A_diag */
   hypre_CSRMatrix        *A_diag            = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex          *A_diag_data       = hypre_CSRMatrixData(A_diag);
   HYPRE_Int              *A_diag_i          = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_diag_j          = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               num_rows_diag_A   = hypre_CSRMatrixNumRows(A_diag);

   /* A_offd */
   hypre_CSRMatrix        *A_offd            = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex          *A_offd_data       = hypre_CSRMatrixData(A_offd);
   HYPRE_Int              *A_offd_i          = hypre_CSRMatrixI(A_offd);
   HYPRE_Int              *A_offd_j          = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int               num_rows_offd_A   = hypre_CSRMatrixNumRows(A_offd);
   HYPRE_Int               num_cols_offd_A   = hypre_CSRMatrixNumCols(A_offd);

   /* Output vector variables */
   HYPRE_Int               num_cols_block_B  = hypre_DenseBlockMatrixNumColsBlock(B);

   /* Local variables */
   HYPRE_Int               i, j, col;
   HYPRE_Int               ib, ir, jr;
   HYPRE_Complex          *recv_data;
   HYPRE_Complex          *send_data;

   /* Communication variables */
   hypre_ParCSRCommPkg    *comm_pkg          = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               num_sends         = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int              *send_map_elmts;
   HYPRE_Int              *send_map_starts;
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *comm_handle;
#else
   hypre_ParCSRCommHandle           *comm_handle;
#endif

   /* Update commpkg offsets */
   hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, 1, 0, 1);
   send_map_elmts  = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);

   /* Allocate the recv and send buffers  */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(HYPRE_COMM_PKG_JOB_COMPLEX, comm_pkg);
   recv_data = (HYPRE_Complex *) hypre_ParCSRCommHandleRecvDataBuffer(comm_handle);
   send_data = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(comm_handle);
   send_data = hypre_Memset((void *) send_data, 0,
                            (size_t) (num_cols_offd_A) * sizeof(HYPRE_Complex),
                            memory_location);
#else
   send_data = hypre_CTAlloc(HYPRE_Complex, num_cols_offd_A, memory_location);
   recv_data = hypre_TAlloc(HYPRE_Complex, send_map_starts[num_sends], memory_location);
#endif

   /* Pack send data */
   for (i = 0; i < num_rows_offd_A; i++)
   {
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         col = A_offd_j[j];
         send_data[col] += A_offd_data[j];
      }
   }

   /* Non-blocking communication starts */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleStart(comm_handle, memory_location, send_data);

#else
   comm_handle = hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 memory_location, send_data,
                                                 memory_location, recv_data);
#endif

   /* Overlapped local computation. */
   for (i = 0; i < num_rows_diag_A; i++)
   {
      ir = i % num_cols_block_B;
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         col = A_diag_j[j];
         ib  = col / num_cols_block_B;
         jr  = col % num_cols_block_B;

         hypre_DenseBlockMatrixDataBIJ(B, ib, ir, jr) += A_diag_data[j];
      }
   }

   /* Non-blocking communication ends */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleWait(comm_handle, memory_location, recv_data);
#else
   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

   /* Unpack recv data */
   for (i = send_map_starts[0]; i < send_map_starts[num_sends]; i++)
   {
      col = send_map_elmts[i];
      ib  = col / num_cols_block_B;
      ir  = col % num_cols_block_B;
      jr  = i % num_cols_block_B;

      hypre_DenseBlockMatrixDataBIJ(B, ib, ir, jr) += recv_data[i];
   }

   /* Free memory */
#if !defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_TFree(send_data, memory_location);
   hypre_TFree(recv_data, memory_location);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlockColSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlockColSum( hypre_ParCSRMatrix      *A,
                               HYPRE_Int                row_major,
                               HYPRE_Int                num_rows_block,
                               HYPRE_Int                num_cols_block,
                               hypre_DenseBlockMatrix **B_ptr )
{
   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_BigInt             num_rows_A      = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols_A      = hypre_ParCSRMatrixGlobalNumCols(A);

   hypre_CSRMatrix         *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);

   hypre_DenseBlockMatrix  *B;

   /*---------------------------------------------
    * Sanity checks
    *---------------------------------------------*/

   if (num_rows_block < 1 || num_cols_block < 1)
   {
      *B_ptr = NULL;
      return hypre_error_flag;
   }

   if (num_rows_A % ((HYPRE_BigInt) num_rows_block))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Global number of rows is not divisable by the block dimension");
      return hypre_error_flag;
   }

   if (num_cols_A % ((HYPRE_BigInt) num_cols_block))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Global number of columns is not divisable by the block dimension");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   /*---------------------------------------------
    * Compute block column sum matrix
    *---------------------------------------------*/

   /* Create output matrix */
   B = hypre_DenseBlockMatrixCreate(row_major,
                                    num_rows_diag_A, num_cols_diag_A,
                                    num_rows_block, num_cols_block);

   /* Initialize the output matrix */
   hypre_DenseBlockMatrixInitializeOn(B, memory_location);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      /* TODO (VPM): hypre_ParCSRMatrixColSumReduceDevice */
      hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixBlockColSumHost(A, B);
      hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
      hypre_DenseBlockMatrixMigrate(B, HYPRE_MEMORY_DEVICE);
   }
   else
#endif
   {
      hypre_ParCSRMatrixBlockColSumHost(A, B);
   }

   /* Set output pointer */
   *B_ptr = B;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixColSumHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixColSumHost( hypre_ParCSRMatrix *A,
                              hypre_ParVector    *b )
{
   /* ParCSRMatrix A */
   HYPRE_MemoryLocation    memory_location   = hypre_ParCSRMatrixMemoryLocation(A);

   /* A_diag */
   hypre_CSRMatrix        *A_diag            = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex          *A_diag_data       = hypre_CSRMatrixData(A_diag);
   HYPRE_Int              *A_diag_i          = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_diag_j          = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int               num_rows_diag_A   = hypre_CSRMatrixNumRows(A_diag);

   /* A_offd */
   hypre_CSRMatrix        *A_offd            = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex          *A_offd_data       = hypre_CSRMatrixData(A_offd);
   HYPRE_Int              *A_offd_i          = hypre_CSRMatrixI(A_offd);
   HYPRE_Int              *A_offd_j          = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int               num_rows_offd_A   = hypre_CSRMatrixNumRows(A_offd);
   HYPRE_Int               num_cols_offd_A   = hypre_CSRMatrixNumCols(A_offd);

   /* Local variables */
   HYPRE_Int               i, j, col;
   HYPRE_Complex          *recv_data;
   HYPRE_Complex          *send_data;

   /* Communication variables */
   hypre_ParCSRCommPkg    *comm_pkg          = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               num_sends         = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int              *send_map_elmts;
   HYPRE_Int              *send_map_starts;
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandle *comm_handle;
#else
   hypre_ParCSRCommHandle           *comm_handle;
#endif

   /* Update commpkg offsets */
   hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, 1, 0, 1);
   send_map_elmts  = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);

   /* Allocate the recv and send buffers  */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(HYPRE_COMM_PKG_JOB_COMPLEX, comm_pkg);
   recv_data = (HYPRE_Complex *) hypre_ParCSRCommHandleRecvDataBuffer(comm_handle);
   send_data = (HYPRE_Complex *) hypre_ParCSRCommHandleSendDataBuffer(comm_handle);
   send_data = hypre_Memset((void *) send_data, 0,
                            (size_t) (num_cols_offd_A) * sizeof(HYPRE_Complex),
                            memory_location);
#else
   send_data = hypre_CTAlloc(HYPRE_Complex, num_cols_offd_A, memory_location);
   recv_data = hypre_TAlloc(HYPRE_Complex, send_map_starts[num_sends], memory_location);
#endif

   /* Pack send data */
   for (i = 0; i < num_rows_offd_A; i++)
   {
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         col = A_offd_j[j];
         send_data[col] += A_offd_data[j];
      }
   }

   /* Non-blocking communication starts */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleStart(comm_handle, memory_location, send_data);

#else
   comm_handle = hypre_ParCSRCommHandleCreate_v2(2, comm_pkg,
                                                 memory_location, send_data,
                                                 memory_location, recv_data);
#endif

   /* Overlapped local computation. */
   for (i = 0; i < num_rows_diag_A; i++)
   {
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         col = A_diag_j[j];
         hypre_ParVectorEntryI(b, col) += A_diag_data[j];
      }
   }

   /* Non-blocking communication ends */
#if defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_ParCSRPersistentCommHandleWait(comm_handle, memory_location, recv_data);
#else
   hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

   /* Unpack recv data */
   for (i = send_map_starts[0]; i < send_map_starts[num_sends]; i++)
   {
      col = send_map_elmts[i];
      hypre_ParVectorEntryI(b, col) += recv_data[i];
   }

   /* Free memory */
#if !defined(HYPRE_USING_PERSISTENT_COMM)
   hypre_TFree(send_data, memory_location);
   hypre_TFree(recv_data, memory_location);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixColSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixColSum( hypre_ParCSRMatrix   *A,
                          hypre_ParVector     **b_ptr )
{
   MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt             global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt            *col_starts      = hypre_ParCSRMatrixColStarts(A);
   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_ParVector         *b;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   /* Create output vector */
   b = hypre_ParVectorCreate(comm, global_num_cols, col_starts);

   /* Initialize the output vector */
   hypre_ParVectorInitialize_v2(b, memory_location);

   /*---------------------------------------------
    * Compute column sum vector
    *---------------------------------------------*/

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      /* TODO (VPM): hypre_ParCSRMatrixColSumDevice */
      hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);
      hypre_ParVectorMigrate(b, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColSumHost(A, b);
      hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
      hypre_ParVectorMigrate(b, HYPRE_MEMORY_DEVICE);
   }
   else
#endif
   {
      hypre_ParCSRMatrixColSumHost(A, b);
   }

   /* Set output pointer */
   *b_ptr = b;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
