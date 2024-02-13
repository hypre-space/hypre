/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Will compute A*A^T, A a Boolean matrix or matrix of doubles.
   based on par_csr_matop.c and mli_pcsr_bool_matop.c */

#include "_hypre_parcsr_mv.h"

extern hypre_CSRMatrix *
hypre_ParCSRMatrixExtractAExt( hypre_ParCSRMatrix *A,
                               HYPRE_Int data,
                               HYPRE_BigInt ** pA_ext_row_map );

void hypre_ParAat_RowSizes(
   HYPRE_Int ** C_diag_i,
   HYPRE_Int ** C_offd_i,
   HYPRE_Int * B_marker,
   HYPRE_Int * A_diag_i,
   HYPRE_Int * A_diag_j,
   HYPRE_Int * A_offd_i,
   HYPRE_Int * A_offd_j,
   HYPRE_BigInt * A_col_map_offd,
   HYPRE_Int * A_ext_i,
   HYPRE_BigInt * A_ext_j,
   HYPRE_BigInt * A_ext_row_map,
   HYPRE_Int *C_diag_size,
   HYPRE_Int *C_offd_size,
   HYPRE_Int num_rows_diag_A,
   HYPRE_Int num_cols_offd_A,
   HYPRE_Int num_rows_A_ext,
   HYPRE_BigInt first_col_diag_A,
   HYPRE_BigInt first_row_index_A
)
/* computes the sizes of the rows of C = A * A^T.
   Out: HYPRE_Int** C_diag_i, C_offd_i
   Workspace provided: HYPRE_Int * B_marker
   In: HYPRE_Int* A_diag_i, A_diag_j, A_offd_i, A_offd_j, A_ext_i, A_ext_j, A_ext_row_map
   Out: HYPRE_Int* C_diag_size, C_offd_size
   In: HYPRE_Int num_rows_diag_A, num_cols_offd_A, num_rows_offd_A_ext, first_row_index_A
*/
{
   /* There are 3 CSRMatrix or CSRBooleanMatrix objects behind the arrays here:
      Any ext*Y belongs to another processor.  And diag*offd, offd*diag never
      have any entries because by definition diag and offd have different
      columns.  So we have to do 4:
      offd*ext, diag*diag, diag*ext, and offd*offd.
   */
   HYPRE_Int i1, i3, jj2, jj3;
   HYPRE_BigInt big_i2;
   HYPRE_Int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_BigInt last_col_diag_C;
   HYPRE_Int start_indexing = 0; /* start indexing for C_data at 0 */

   *C_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A + 1, HYPRE_MEMORY_HOST);
   *C_offd_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A + 1, HYPRE_MEMORY_HOST);

   last_col_diag_C = first_row_index_A + (HYPRE_BigInt) num_rows_diag_A - 1;

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_rows_diag_A + num_rows_A_ext; i1++)
   {
      B_marker[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over rows i1 of A (or C).
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      /*--------------------------------------------------------------------
       *  Set count marker for diagonal entry, C_{i1,i1}.
       *--------------------------------------------------------------------*/

      B_marker[i1] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      jj_count_diag++;

      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_offd.
       *  For each such column we will find the contributions of
       *  the corresponding rows i2 of A^T to C=A*A^T - but in A^T we look
       *  only at the external part of A^T, i.e. with columns (rows of A)
       *  which live on other processors.
       *-----------------------------------------------------------------*/

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            big_i2 = A_col_map_offd[ A_offd_j[jj2] ];

            /* offd*ext */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
             *  That is, rows i3 having a column i2 of A_ext.
             *  For now, for each row i3 of A_ext we crudely check _all_
             *  columns to see whether one matches i2.
             *  For each entry (i2,i3) of (A_ext)^T, mark C(i1,i3)
             *  as a potential nonzero.
             *-----------------------------------------------------------*/

            for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
            {
               for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
               {
                  if ( A_ext_j[jj3] == big_i2 )
                  {
                     /* row i3, column i2 of A_ext; or,
                        row i2, column i3 of (A_ext)^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/

                     if ( A_ext_row_map[i3] < first_row_index_A ||
                          A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                     else                                                /* diag */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                  }
               }
            }

            /* offd*offd */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of A^T
             *  That is, rows i3 having a column i2 of A (local part).
             *  For now, for each row i3 of A we crudely check _all_
             *  columns to see whether one matches i2.
             *  This i3-loop is for the local off-diagonal part of A.
             *  For each entry (i2,i3) of A^T, mark C(i1,i3)
             *  as a potential nonzero.
             *-----------------------------------------------------------*/

            for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
            {
               /* ... note that num_rows_diag_A == num_rows_offd_A */
               for ( jj3 = A_offd_i[i3]; jj3 < A_offd_i[i3 + 1]; jj3++ )
               {
                  if ( A_col_map_offd[ A_offd_j[jj3] ] == big_i2 )
                  {
                     /* row i3, column i2 of A; or,
                        row i2, column i3 of A^T */
                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/

                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                        B_marker[i3] = jj_count_diag;
                        jj_count_diag++;
                     }
                  }
               }
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_diag.
       *  For each such column we will find the contributions of
       *  the corresponding rows i2 of A^T to C=A*A^T - but in A^T we look
       *  only at the external part of A^T, i.e. with columns (rows of A)
       *  which live on other processors.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         big_i2 = (HYPRE_BigInt)A_diag_j[jj2] + first_col_diag_A ;

         /* diag*ext */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
          *  That is, rows i3 having a column i2 of A_ext.
          *  For now, for each row i3 of A_ext we crudely check _all_
          *  columns to see whether one matches i2.
          *  For each entry (i2,i3) of (A_ext)^T, mark C(i1,i3)
          *  as a potential nonzero.
          *-----------------------------------------------------------*/

         for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
         {
            for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
            {
               if ( A_ext_j[jj3] == big_i2 )
               {
                  /* row i3, column i2 of A_ext; or,
                     row i2, column i3 of (A_ext)^T */

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
                  if ( A_ext_row_map[i3] < first_row_index_A ||
                       A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                        jj_count_offd++;
                     }
                  }
                  else                                                /* diag */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                        jj_count_diag++;
                     }
                  }
               }
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_diag.
       *  For each such column we will find the contributions of the
       *  corresponding rows i2 of A^T to C=A*A^T .  Now we only look
       *  at the local part of A^T - with columns (rows of A) living
       *  on this processor.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         big_i2 = (HYPRE_BigInt)A_diag_j[jj2] + first_col_diag_A ;

         /* diag*diag */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This first i3-loop is for the diagonal part of A.
          *  For each entry (i2,i3) of A^T, mark C(i1,i3)
          *  as a potential nonzero.
          *-----------------------------------------------------------*/
         for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
         {
            for ( jj3 = A_diag_i[i3]; jj3 < A_diag_i[i3 + 1]; jj3++ )
            {
               if ( (HYPRE_BigInt)A_diag_j[jj3] + first_col_diag_A == big_i2 )
               {
                  /* row i3, column i2 of A; or,
                     row i2, column i3 of A^T */
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
            }
         }
      }        /* end of second and last i2 loop */

      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/

      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;

   }              /* end of i1 loop */

   (*C_diag_i)[num_rows_diag_A] = jj_count_diag;
   (*C_offd_i)[num_rows_diag_A] = jj_count_offd;

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   *C_diag_size = jj_count_diag;
   *C_offd_size = jj_count_offd;

   /* End of First Pass */
}


/*--------------------------------------------------------------------------
 * hypre_ParCSRAAt : multiplies ParCSRMatrix A by its transpose, A*A^T
 * and returns the product in ParCSRMatrix C
 * Note that C does not own the partitionings
 *--------------------------------------------------------------------------*/
/* There are lots of possible optimizations.  There is excess communication
   going on, nothing is being done to take advantage of symmetry, and probably
   more things. */

hypre_ParCSRMatrix*
hypre_ParCSRAAt(hypre_ParCSRMatrix  *A)
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   HYPRE_Complex   *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Complex   *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt    *A_col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *A_ext_row_map;

   HYPRE_BigInt    *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_BigInt    *col_map_offd_C;

   hypre_CSRMatrix *C_diag;

   HYPRE_Complex   *C_diag_data;
   HYPRE_Int       *C_diag_i;
   HYPRE_Int       *C_diag_j;

   hypre_CSRMatrix *C_offd;

   HYPRE_Complex   *C_offd_data = NULL;
   HYPRE_Int       *C_offd_i = NULL;
   HYPRE_Int       *C_offd_j = NULL;
   HYPRE_Int       *new_C_offd_j;

   HYPRE_Int        C_diag_size;
   HYPRE_Int        C_offd_size;
   HYPRE_BigInt     last_col_diag_C;
   HYPRE_Int        num_cols_offd_C;

   hypre_CSRMatrix *A_ext = NULL;

   HYPRE_Complex   *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;
   HYPRE_Int        num_rows_A_ext = 0;

   HYPRE_BigInt     first_row_index_A = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt     first_col_diag_A = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int       *B_marker;

   HYPRE_Int        i;
   HYPRE_Int        i1, i2, i3;
   HYPRE_Int        jj2, jj3;

   HYPRE_Int        jj_count_diag, jj_count_offd;
   HYPRE_Int        jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int        start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_Int        count;
   HYPRE_BigInt     n_rows_A, n_cols_A;

   HYPRE_Complex    a_entry;
   HYPRE_Complex    a_b_product;

   HYPRE_Complex    zero = 0.0;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);

   if (n_cols_A != n_rows_A)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   /*-----------------------------------------------------------------------
    *  Extract A_ext, i.e. portion of A that is stored on neighbor procs
    *  and needed locally for A^T in the matrix matrix product A*A^T
    *-----------------------------------------------------------------------*/

   if ((HYPRE_BigInt)num_rows_diag_A != n_rows_A)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings
       *--------------------------------------------------------------------*/
      if (!hypre_ParCSRMatrixCommPkg(A))
      {
         hypre_MatTCommPkgCreate(A);
      }

      A_ext = hypre_ParCSRMatrixExtractAExt( A, 1, &A_ext_row_map );
      A_ext_data = hypre_CSRMatrixData(A_ext);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixBigJ(A_ext);
      num_rows_A_ext = hypre_CSRMatrixNumRows(A_ext);
   }
   /*-----------------------------------------------------------------------
    *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = hypre_CTAlloc(HYPRE_Int,  num_rows_diag_A + num_rows_A_ext, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for ( i1 = 0; i1 < num_rows_diag_A + num_rows_A_ext; ++i1 )
   {
      B_marker[i1] = -1;
   }


   hypre_ParAat_RowSizes(
      &C_diag_i, &C_offd_i, B_marker,
      A_diag_i, A_diag_j,
      A_offd_i, A_offd_j, A_col_map_offd,
      A_ext_i, A_ext_j, A_ext_row_map,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A,
      num_rows_A_ext,
      first_col_diag_A, first_row_index_A
   );

#if 0
   /* debugging output: */
   hypre_printf("A_ext_row_map (%i):", num_rows_A_ext);
   for ( i1 = 0; i1 < num_rows_A_ext; ++i1 ) { hypre_printf(" %i", A_ext_row_map[i1] ); }
   hypre_printf("\nC_diag_i (%i):", C_diag_size);
   for ( i1 = 0; i1 <= num_rows_diag_A; ++i1 ) { hypre_printf(" %i", C_diag_i[i1] ); }
   hypre_printf("\nC_offd_i (%i):", C_offd_size);
   for ( i1 = 0; i1 <= num_rows_diag_A; ++i1 ) { hypre_printf(" %i", C_offd_i[i1] ); }
   hypre_printf("\n");
#endif

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_C = first_row_index_A + (HYPRE_BigInt) num_rows_diag_A - 1;
   C_diag_data = hypre_CTAlloc(HYPRE_Complex, C_diag_size, HYPRE_MEMORY_HOST);
   C_diag_j    = hypre_CTAlloc(HYPRE_Int, C_diag_size, HYPRE_MEMORY_HOST);
   C_offd_data = hypre_CTAlloc(HYPRE_Complex, C_offd_size, HYPRE_MEMORY_HOST);
   C_offd_j    = hypre_CTAlloc(HYPRE_Int, C_offd_size, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for ( i1 = 0; i1 < num_rows_diag_A + num_rows_A_ext; ++i1 )
   {
      B_marker[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1}
       *--------------------------------------------------------------------*/

      B_marker[i1] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      C_diag_data[jj_count_diag] = zero;
      C_diag_j[jj_count_diag] = i1;
      jj_count_diag++;

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/

      /* There are 3 CSRMatrix or CSRBooleanMatrix objects here:
         ext*ext, ext*diag, and ext*offd belong to another processor.
         diag*offd and offd*diag don't count - never share a column by definition.
         So we have to do 4 cases:
         diag*ext, offd*ext, diag*diag, and offd*offd.
      */

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         i2 = A_diag_j[jj2];
         a_entry = A_diag_data[jj2];

         /* diag*ext */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
          *  That is, rows i3 having a column i2 of A_ext.
          *  For now, for each row i3 of A_ext we crudely check _all_
          *  columns to see whether one matches i2.
          *  For each entry (i2,i3) of (A_ext)^T, add A(i1,i2)*A(i3,i2)
          *  to C(i1,i3) .  This contributes to both the diag and offd
          *  blocks of C.
          *-----------------------------------------------------------*/

         for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
         {
            for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
            {
               if ( A_ext_j[jj3] == (HYPRE_BigInt)i2 + first_col_diag_A )
               {
                  /* row i3, column i2 of A_ext; or,
                     row i2, column i3 of (A_ext)^T */

                  a_b_product = a_entry * A_ext_data[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if ( A_ext_row_map[i3] < first_row_index_A ||
                       A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                        C_offd_data[jj_count_offd] = a_b_product;
                        C_offd_j[jj_count_offd] = i3;
                        jj_count_offd++;
                     }
                     else
                     {
                        C_offd_data[B_marker[i3 + num_rows_diag_A]] += a_b_product;
                     }
                  }
                  else                                                /* diag */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                        C_diag_data[jj_count_diag] = a_b_product;
                        C_diag_j[jj_count_diag] = (HYPRE_Int)(i3 - first_col_diag_A);
                        jj_count_diag++;
                     }
                     else
                     {
                        C_diag_data[B_marker[i3 + num_rows_diag_A]] += a_b_product;
                     }
                  }
               }
            }
         }
      }

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            a_entry = A_offd_data[jj2];

            /* offd * ext */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
             *  That is, rows i3 having a column i2 of A_ext.
             *  For now, for each row i3 of A_ext we crudely check _all_
             *  columns to see whether one matches i2.
             *  For each entry (i2,i3) of (A_ext)^T, add A(i1,i2)*A(i3,i2)
             *  to C(i1,i3) .  This contributes to both the diag and offd
             *  blocks of C.
             *-----------------------------------------------------------*/

            for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
            {
               for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
               {
                  if ( A_ext_j[jj3] == A_col_map_offd[i2] )
                  {
                     /* row i3, column i2 of A_ext; or,
                        row i2, column i3 of (A_ext)^T */

                     a_b_product = a_entry * A_ext_data[jj3];

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if ( A_ext_row_map[i3] < first_row_index_A ||
                          A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                           C_offd_data[jj_count_offd] = a_b_product;
                           C_offd_j[jj_count_offd] = i3;
                           jj_count_offd++;
                        }
                        else
                        {
                           C_offd_data[B_marker[i3 + num_rows_diag_A]] += a_b_product;
                        }
                     }
                     else                                                /* diag */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                           C_diag_data[jj_count_diag] = a_b_product;
                           C_diag_j[jj_count_diag] = (HYPRE_Int)(i3 - first_row_index_A);
                           jj_count_diag++;
                        }
                        else
                        {
                           C_diag_data[B_marker[i3 + num_rows_diag_A]] += a_b_product;
                        }
                     }
                  }
               }
            }
         }
      }

      /* diag * diag */
      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_diag.
       *  For each such column we will find the contributions of the
       *  corresponding rows i2 of A^T to C=A*A^T .  Now we only look
       *  at the local part of A^T - with columns (rows of A) living
       *  on this processor.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         i2 = A_diag_j[jj2];
         a_entry = A_diag_data[jj2];

         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This i3-loop is for the diagonal block of A.
          *  It contributes to the diagonal block of C.
          *  For each entry (i2,i3) of A^T,  add A(i1,i2)*A(i3,i2)
          *  to C(i1,i3)
          *-----------------------------------------------------------*/
         for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
         {
            for ( jj3 = A_diag_i[i3]; jj3 < A_diag_i[i3 + 1]; jj3++ )
            {
               if ( A_diag_j[jj3] == i2 )
               {
                  /* row i3, column i2 of A; or,
                     row i2, column i3 of A^T */
                  a_b_product = a_entry * A_diag_data[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_b_product;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[B_marker[i3]] += a_b_product;
                  }
               }
            }
         } /* end of i3 loop */
      } /* end of third i2 loop */

      /* offd * offd */
      /*-----------------------------------------------------------
       *  Loop over offd columns i2 of A in A*A^T.  Then
       *  loop over offd entries (columns) i3 in row i2 of A^T
       *  That is, rows i3 having a column i2 of A (local part).
       *  For now, for each row i3 of A we crudely check _all_
       *  columns to see whether one matches i2.
       *  This i3-loop is for the off-diagonal block of A.
       *  It contributes to the diag block of C.
       *  For each entry (i2,i3) of A^T, add A*A^T to C
       *-----------------------------------------------------------*/
      if (num_cols_offd_A)
      {

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            a_entry = A_offd_data[jj2];

            for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
            {
               /* ... note that num_rows_diag_A == num_rows_offd_A */
               for ( jj3 = A_offd_i[i3]; jj3 < A_offd_i[i3 + 1]; jj3++ )
               {
                  if ( A_offd_j[jj3] == i2 )
                  {
                     /* row i3, column i2 of A; or,
                        row i2, column i3 of A^T */
                     a_b_product = a_entry * A_offd_data[jj3];

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution
                      *--------------------------------------------------------*/

                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                        B_marker[i3] = jj_count_diag;
                        C_diag_data[jj_count_diag] = a_b_product;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                     }
                     else
                     {
                        C_diag_data[B_marker[i3]] += a_b_product;
                     }
                  }
               }
            }  /* end of last i3 loop */
         }     /* end of if (num_cols_offd_A) */

      }        /* end of fourth and last i2 loop */
#if 0          /* debugging printout */
      hypre_printf("end of i1 loop: i1=%i jj_count_diag=%i\n", i1, jj_count_diag );
      hypre_printf("  C_diag_j=");
      for ( jj3 = 0; jj3 < jj_count_diag; ++jj3) { hypre_printf("%i ", C_diag_j[jj3]); }
      hypre_printf("  C_diag_data=");
      for ( jj3 = 0; jj3 < jj_count_diag; ++jj3) { hypre_printf("%f ", C_diag_data[jj3]); }
      hypre_printf("\n");
      hypre_printf("  C_offd_j=");
      for ( jj3 = 0; jj3 < jj_count_offd; ++jj3) { hypre_printf("%i ", C_offd_j[jj3]); }
      hypre_printf("  C_offd_data=");
      for ( jj3 = 0; jj3 < jj_count_offd; ++jj3) { hypre_printf("%f ", C_offd_data[jj3]); }
      hypre_printf("\n");
      hypre_printf( "  B_marker =" );
      for ( it = 0; it < num_rows_diag_A + num_rows_A_ext; ++it )
      {
         hypre_printf(" %i", B_marker[it] );
      }
      hypre_printf( "\n" );
#endif
   }           /* end of i1 loop */

   /*-----------------------------------------------------------------------
    *  Delete 0-columns in C_offd, i.e. generate col_map_offd and reset
    *  C_offd_j.  Note that (with the indexing we have coming into this
    *  block) col_map_offd_C[i3]==A_ext_row_map[i3].
    *-----------------------------------------------------------------------*/

   for ( i = 0; i < num_rows_diag_A + num_rows_A_ext; ++i )
   {
      B_marker[i] = -1;
   }
   for ( i = 0; i < C_offd_size; i++ )
   {
      B_marker[ C_offd_j[i] ] = -2;
   }

   count = 0;
   for (i = 0; i < num_rows_diag_A + num_rows_A_ext; i++)
   {
      if (B_marker[i] == -2)
      {
         B_marker[i] = count;
         count++;
      }
   }
   num_cols_offd_C = count;

   if (num_cols_offd_C)
   {
      col_map_offd_C = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      new_C_offd_j = hypre_CTAlloc(HYPRE_Int, C_offd_size, HYPRE_MEMORY_HOST);
      /* ... a bit big, but num_cols_offd_C is too small.  It might be worth
         computing the correct size, which is sum( no. columns in row i, over all rows i )
      */

      for (i = 0; i < C_offd_size; i++)
      {
         new_C_offd_j[i] = B_marker[C_offd_j[i]];
         col_map_offd_C[ new_C_offd_j[i] ] = A_ext_row_map[ C_offd_j[i] ];
      }

      hypre_TFree(C_offd_j, HYPRE_MEMORY_HOST);
      C_offd_j = new_C_offd_j;

   }

   /*----------------------------------------------------------------
    * Create C
    *----------------------------------------------------------------*/

   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_rows_A, row_starts_A,
                                row_starts_A, num_cols_offd_C,
                                C_diag_size, C_offd_size);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data;
   hypre_CSRMatrixI(C_diag) = C_diag_i;
   hypre_CSRMatrixJ(C_diag) = C_diag_j;

   if (num_cols_offd_C)
   {
      C_offd = hypre_ParCSRMatrixOffd(C);
      hypre_CSRMatrixData(C_offd) = C_offd_data;
      hypre_CSRMatrixI(C_offd) = C_offd_i;
      hypre_CSRMatrixJ(C_offd) = C_offd_j;
      hypre_ParCSRMatrixOffd(C) = C_offd;
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }
   else
   {
      hypre_TFree(C_offd_i, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      hypre_CSRMatrixDestroy(A_ext);
      A_ext = NULL;
   }
   hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   if ( num_rows_diag_A != n_rows_A )
   {
      hypre_TFree(A_ext_row_map, HYPRE_MEMORY_HOST);
   }

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractAExt : extracts rows from A which are located on other
 * processors and needed for multiplying A^T with the local part of A. The rows
 * are returned as CSRMatrix.  A row map for A_ext (like the ParCSRColMap) is
 * returned through the third argument.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_ParCSRMatrixExtractAExt( hypre_ParCSRMatrix *A,
                               HYPRE_Int data,
                               HYPRE_BigInt ** pA_ext_row_map )
{
   /* Note that A's role as the first factor in A*A^T is used only
      through ...CommPkgT(A), which basically says which rows of A
      (columns of A^T) are needed.  In all the other places where A
      serves as an input, it is through its role as A^T, the matrix
      whose data needs to be passed between processors. */
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
   /*HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);*/
   HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkgT(A);
   /* ... CommPkgT(A) should identify all rows of A^T needed for A*A^T (that is
    * generally a bigger set than ...CommPkg(A), the rows of B needed for A*B) */
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);

   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Complex *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Complex *offd_data = hypre_CSRMatrixData(offd);

   HYPRE_BigInt num_cols_A;
   HYPRE_Int num_nonzeros;
   HYPRE_Int num_rows_A_ext;

   hypre_CSRMatrix *A_ext;

   HYPRE_Int *A_ext_i;
   HYPRE_BigInt *A_ext_j;
   HYPRE_Complex *A_ext_data;

   num_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   num_rows_A_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
   ( &A_ext_i, &A_ext_j, &A_ext_data, pA_ext_row_map,
     &num_nonzeros,
     data, 1, comm, comm_pkg,
     num_cols_A, num_recvs, num_sends,
     first_col_diag, A->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data
   );

   A_ext = hypre_CSRMatrixCreate(num_rows_A_ext, num_cols_A, num_nonzeros);
   hypre_CSRMatrixI(A_ext) = A_ext_i;
   hypre_CSRMatrixBigJ(A_ext) = A_ext_j;
   if (data) { hypre_CSRMatrixData(A_ext) = A_ext_data; }

   return A_ext;
}
