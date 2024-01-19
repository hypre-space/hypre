/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

hypre_ParCSRBooleanMatrix*
hypre_ParBooleanMatmul( hypre_ParCSRBooleanMatrix *A,
                        hypre_ParCSRBooleanMatrix *B )
{
   MPI_Comm       comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);

   hypre_CSRBooleanMatrix *A_diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);
   HYPRE_Int              *A_diag_i = hypre_CSRBooleanMatrix_Get_I(A_diag);
   HYPRE_Int              *A_diag_j = hypre_CSRBooleanMatrix_Get_J(A_diag);

   hypre_CSRBooleanMatrix *A_offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);
   HYPRE_Int              *A_offd_i = hypre_CSRBooleanMatrix_Get_I(A_offd);
   HYPRE_Int              *A_offd_j = hypre_CSRBooleanMatrix_Get_J(A_offd);

   HYPRE_BigInt *row_starts_A = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   HYPRE_Int   num_rows_diag_A = hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   HYPRE_Int   num_cols_diag_A = hypre_CSRBooleanMatrix_Get_NCols(A_diag);
   HYPRE_Int   num_cols_offd_A = hypre_CSRBooleanMatrix_Get_NCols(A_offd);

   hypre_CSRBooleanMatrix *B_diag = hypre_ParCSRBooleanMatrix_Get_Diag(B);
   HYPRE_Int              *B_diag_i = hypre_CSRBooleanMatrix_Get_I(B_diag);
   HYPRE_Int              *B_diag_j = hypre_CSRBooleanMatrix_Get_J(B_diag);

   hypre_CSRBooleanMatrix *B_offd = hypre_ParCSRBooleanMatrix_Get_Offd(B);
   HYPRE_BigInt        *col_map_offd_B = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);
   HYPRE_Int              *B_offd_i = hypre_CSRBooleanMatrix_Get_I(B_offd);
   HYPRE_Int              *B_offd_j = hypre_CSRBooleanMatrix_Get_J(B_offd);

   HYPRE_BigInt   first_col_diag_B = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   HYPRE_BigInt   last_col_diag_B;
   HYPRE_BigInt *col_starts_B = hypre_ParCSRBooleanMatrix_Get_ColStarts(B);
   HYPRE_Int   num_rows_diag_B = hypre_CSRBooleanMatrix_Get_NRows(B_diag);
   HYPRE_Int   num_cols_diag_B = hypre_CSRBooleanMatrix_Get_NCols(B_diag);
   HYPRE_Int   num_cols_offd_B = hypre_CSRBooleanMatrix_Get_NCols(B_offd);

   hypre_ParCSRBooleanMatrix *C;
   HYPRE_BigInt            *col_map_offd_C;
   HYPRE_Int            *map_B_to_C = NULL;

   hypre_CSRBooleanMatrix *C_diag;
   HYPRE_Int             *C_diag_i;
   HYPRE_Int             *C_diag_j;

   hypre_CSRBooleanMatrix *C_offd;
   HYPRE_Int             *C_offd_i = NULL;
   HYPRE_Int             *C_offd_j = NULL;

   HYPRE_Int              C_diag_size;
   HYPRE_Int              C_offd_size;
   HYPRE_Int          num_cols_offd_C = 0;

   hypre_CSRBooleanMatrix *Bs_ext = NULL;
   HYPRE_Int             *Bs_ext_i = NULL;
   HYPRE_BigInt          *Bs_ext_j = NULL;

   HYPRE_Int             *B_ext_diag_i = NULL;
   HYPRE_Int             *B_ext_diag_j = NULL;
   HYPRE_Int        B_ext_diag_size;

   HYPRE_Int             *B_ext_offd_i = NULL;
   HYPRE_Int             *B_ext_offd_j = NULL;
   HYPRE_BigInt          *B_tmp_offd_j = NULL;
   HYPRE_Int        B_ext_offd_size;

   HYPRE_Int       *B_marker;
   HYPRE_BigInt       *temp;

   HYPRE_Int              i, j;
   HYPRE_Int              i1, i2, i3;
   HYPRE_Int              jj2, jj3;

   HYPRE_Int              jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_BigInt        n_rows_A, n_cols_A;
   HYPRE_BigInt        n_rows_B, n_cols_B;
   HYPRE_Int              allsquare = 0;
   HYPRE_Int              cnt, cnt_offd, cnt_diag;
   HYPRE_Int              num_procs;
   HYPRE_BigInt           value;

   n_rows_A = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   n_rows_B = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(B);
   n_cols_B = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   if ( num_rows_diag_A == num_cols_diag_B ) { allsquare = 1; }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
      * If there exists no CommPkg for A, a CommPkg is generated using
      * equally load balanced partitionings
      *--------------------------------------------------------------------*/
      if (!hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
      {
         hypre_BooleanMatvecCommPkgCreate(A);
      }

      Bs_ext = hypre_ParCSRBooleanMatrixExtractBExt(B, A);
      Bs_ext_i    = hypre_CSRBooleanMatrix_Get_I(Bs_ext);
      Bs_ext_j    = hypre_CSRBooleanMatrix_Get_BigJ(Bs_ext);
   }

   B_ext_diag_i = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   B_ext_offd_i = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            B_ext_offd_size++;
         }
         else
         {
            B_ext_diag_size++;
         }
      B_ext_diag_i[i + 1] = B_ext_diag_size;
      B_ext_offd_i[i + 1] = B_ext_offd_size;
   }

   if (B_ext_diag_size)
   {
      B_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  B_ext_diag_size, HYPRE_MEMORY_HOST);
   }

   if (B_ext_offd_size)
   {
      B_ext_offd_j = hypre_CTAlloc(HYPRE_Int,  B_ext_offd_size, HYPRE_MEMORY_HOST);
      B_tmp_offd_j = hypre_CTAlloc(HYPRE_BigInt,  B_ext_offd_size, HYPRE_MEMORY_HOST);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            B_tmp_offd_j[cnt_offd++] = Bs_ext_j[j];
            //temp[cnt_offd++] = Bs_ext_j[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag++] = (HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
         }
   }

   if (num_procs > 1)
   {
      hypre_CSRBooleanMatrixDestroy(Bs_ext);
      Bs_ext = NULL;
   }

   cnt = 0;
   if (B_ext_offd_size || num_cols_offd_B)
   {
      temp = hypre_CTAlloc(HYPRE_BigInt,  B_ext_offd_size + num_cols_offd_B, HYPRE_MEMORY_HOST);
      for (i = 0; i < B_ext_offd_size; i++)
      {
         temp[i] = B_tmp_offd_j[i];
      }
      cnt = B_ext_offd_size;
      for (i = 0; i < num_cols_offd_B; i++)
      {
         temp[cnt++] = col_map_offd_B[i];
      }
   }
   if (cnt)
   {
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

   if (B_ext_offd_size || num_cols_offd_B)
   {
      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }

   for (i = 0 ; i < B_ext_offd_size; i++)
      B_ext_offd_j[i] = hypre_BigBinarySearch(col_map_offd_C,
                                              B_tmp_offd_j[i],
                                              num_cols_offd_C);
   if (B_ext_offd_size)
   {
      hypre_TFree(B_tmp_offd_j, HYPRE_MEMORY_HOST);
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
   }

   hypre_ParMatmul_RowSizes(
      /*&C_diag_i, &C_offd_i, &B_marker,*/
      /* BooleanMatrix only uses HOST memory for now */
      HYPRE_MEMORY_HOST,
      &C_diag_i, &C_offd_i, NULL,
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_diag_i, B_ext_diag_j,
      B_ext_offd_i, B_ext_offd_j, map_B_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_rows_diag_A,
      num_cols_offd_A, allsquare,
      num_cols_diag_B, num_cols_offd_B,
      num_cols_offd_C
   );


   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_B = first_col_diag_B + (HYPRE_BigInt)num_cols_diag_B - 1;
   C_diag_j    = hypre_CTAlloc(HYPRE_Int,  C_diag_size, HYPRE_MEMORY_HOST);
   if (C_offd_size)
   {
      C_offd_j    = hypre_CTAlloc(HYPRE_Int,  C_offd_size, HYPRE_MEMORY_HOST);
   }


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_j.
    *  Second Pass: Fill in C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_diag_B + num_cols_offd_C, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
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

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare )
      {
         B_marker[i1] = jj_count_diag;
         C_diag_j[jj_count_diag] = i1;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/

            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
            {
               i3 = num_cols_diag_B + B_ext_offd_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/
               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                  jj_count_offd++;
               }
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
               }
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_diag.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         i2 = A_diag_j[jj2];

         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/

         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
         {
            i3 = B_diag_j[jj3];

            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/

            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               C_diag_j[jj_count_diag] = i3;
               jj_count_diag++;
            }
         }
         if (num_cols_offd_B)
         {
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
            {
               i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                  jj_count_offd++;
               }
            }
         }
      }
   }

   C = hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                       col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(C, 0);

   C_diag = hypre_ParCSRBooleanMatrix_Get_Diag(C);
   hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i;
   hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j;
   C_offd = hypre_ParCSRBooleanMatrix_Get_Offd(C);
   hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i;
   hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j;
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(B_ext_diag_i, HYPRE_MEMORY_HOST);
   if (B_ext_diag_size)
   {
      hypre_TFree(B_ext_diag_j, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(B_ext_offd_i, HYPRE_MEMORY_HOST);
   if (B_ext_offd_size)
   {
      hypre_TFree(B_ext_offd_j, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_B) { hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST); }

   return C;

}



/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixExtractBExt :
 * extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRBooleanMatrix.
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix *
hypre_ParCSRBooleanMatrixExtractBExt
( hypre_ParCSRBooleanMatrix *B, hypre_ParCSRBooleanMatrix *A )
{
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(B);
   HYPRE_BigInt first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   /*HYPRE_Int first_row_index = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(B);*/
   HYPRE_BigInt *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBooleanMatrix_Get_CommPkg(A);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(B);
   HYPRE_Int *diag_i = hypre_CSRBooleanMatrix_Get_I(diag);
   HYPRE_Int *diag_j = hypre_CSRBooleanMatrix_Get_J(diag);

   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(B);
   HYPRE_Int *offd_i = hypre_CSRBooleanMatrix_Get_I(offd);
   HYPRE_Int *offd_j = hypre_CSRBooleanMatrix_Get_J(offd);

   HYPRE_Int num_cols_B, num_nonzeros;
   HYPRE_Int num_rows_B_ext;

   hypre_CSRBooleanMatrix *B_ext;
   HYPRE_Int *B_ext_i;
   HYPRE_BigInt *B_ext_j;

   HYPRE_Complex *B_ext_data = NULL, *diag_data = NULL, *offd_data = NULL;
   HYPRE_BigInt *B_ext_row_map = NULL;
   /* ... not referenced, but needed for function call */

   num_cols_B = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   hypre_ParCSRMatrixExtractBExt_Arrays
   ( &B_ext_i, &B_ext_j, &B_ext_data, &B_ext_row_map,
     &num_nonzeros,
     0, 0, comm, comm_pkg,
     num_cols_B, num_recvs, num_sends,
     first_col_diag, B->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data
   );

   B_ext = hypre_CSRBooleanMatrixCreate(num_rows_B_ext, num_cols_B, num_nonzeros);
   hypre_CSRBooleanMatrix_Get_I(B_ext) = B_ext_i;
   hypre_CSRBooleanMatrix_Get_BigJ(B_ext) = B_ext_j;

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBooleanMatrixExtractAExt : extracts rows from A which are located on other
 * processors and needed for multiplying A^T with the local part of A. The rows
 * are returned as CSRBooleanMatrix.  A row map for A_ext (like the ParCSRColMap) is
 * returned through the third argument.
 *--------------------------------------------------------------------------*/

hypre_CSRBooleanMatrix *
hypre_ParCSRBooleanMatrixExtractAExt( hypre_ParCSRBooleanMatrix *A,
                                      HYPRE_BigInt ** pA_ext_row_map )
{
   /* Note that A's role as the first factor in A*A^T is used only
      through ...CommPkgT(A), which basically says which rows of A
      (columns of A^T) are needed.  In all the other places where A
      serves as an input, it is through its role as A^T, the matrix
      whose data needs to be passed between processors. */
   MPI_Comm comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
   HYPRE_BigInt first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   /*HYPRE_Int first_row_index = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);*/
   HYPRE_BigInt *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBooleanMatrix_Get_CommPkgT(A);
   /* ... CommPkgT(A) should identify all rows of A^T needed for A*A^T (that is
    * generally a bigger set than ...CommPkg(A), the rows of B needed for A*B) */
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   hypre_CSRBooleanMatrix *diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);

   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);

   hypre_CSRBooleanMatrix *offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);

   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);

   HYPRE_BigInt num_cols_A;
   HYPRE_Int num_nonzeros;
   HYPRE_Int num_rows_A_ext;

   hypre_CSRBooleanMatrix *A_ext;

   HYPRE_Int *A_ext_i;
   HYPRE_BigInt *A_ext_j;

   HYPRE_Int data = 0;
   HYPRE_Complex *A_ext_data = NULL, *diag_data = NULL, *offd_data = NULL;
   /* ... not referenced, but needed for function call */

   num_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
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

   A_ext = hypre_CSRBooleanMatrixCreate(num_rows_A_ext, num_cols_A, num_nonzeros);
   hypre_CSRBooleanMatrix_Get_I(A_ext) = A_ext_i;
   hypre_CSRBooleanMatrix_Get_BigJ(A_ext) = A_ext_j;

   return A_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParBooleanAAT : multiplies hypre_ParCSRBooleanMatrix A by its transpose,
 * A*A^T, and returns the product in hypre_ParCSRBooleanMatrix C
 * Note that C does not own the partitionings
 * This is based on hypre_ParCSRAAt.
 *--------------------------------------------------------------------------*/

hypre_ParCSRBooleanMatrix*
hypre_ParBooleanAAt(hypre_ParCSRBooleanMatrix  *A)
{
   MPI_Comm       comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);

   hypre_CSRBooleanMatrix *A_diag = hypre_ParCSRBooleanMatrix_Get_Diag(A);

   HYPRE_Int             *A_diag_i = hypre_CSRBooleanMatrix_Get_I(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRBooleanMatrix_Get_J(A_diag);

   hypre_CSRBooleanMatrix *A_offd = hypre_ParCSRBooleanMatrix_Get_Offd(A);
   HYPRE_Int             *A_offd_i = hypre_CSRBooleanMatrix_Get_I(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBooleanMatrix_Get_J(A_offd);

   HYPRE_BigInt          *A_col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   HYPRE_BigInt          *A_ext_row_map;

   HYPRE_BigInt *row_starts_A = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   HYPRE_Int   num_rows_diag_A = hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   HYPRE_Int   num_cols_offd_A = hypre_CSRBooleanMatrix_Get_NCols(A_offd);

   hypre_ParCSRBooleanMatrix *C;
   HYPRE_BigInt            *col_map_offd_C;

   hypre_CSRBooleanMatrix *C_diag;

   HYPRE_Int             *C_diag_i;
   HYPRE_Int             *C_diag_j;

   hypre_CSRBooleanMatrix *C_offd;

   HYPRE_Int             *C_offd_i = NULL;
   HYPRE_Int             *C_offd_j = NULL;
   HYPRE_Int             *new_C_offd_j;

   HYPRE_Int              C_diag_size;
   HYPRE_Int              C_offd_size;
   HYPRE_BigInt           last_col_diag_C;
   HYPRE_Int              num_cols_offd_C;

   hypre_CSRBooleanMatrix *A_ext = NULL;

   HYPRE_Int             *A_ext_i = NULL;
   HYPRE_BigInt          *A_ext_j = NULL;
   HYPRE_Int             num_rows_A_ext = 0;

   HYPRE_BigInt   first_row_index_A = hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);
   HYPRE_BigInt   first_col_diag_A = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   HYPRE_Int         *B_marker;

   HYPRE_Int              i;
   HYPRE_Int              i1, i2, i3;
   HYPRE_Int              jj2, jj3;

   HYPRE_Int              jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_Int          count;
   HYPRE_BigInt          n_rows_A, n_cols_A;

   n_rows_A = hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);

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
      if (!hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
      {
         hypre_BooleanMatTCommPkgCreate(A);
      }

      A_ext = hypre_ParCSRBooleanMatrixExtractAExt( A, &A_ext_row_map );
      A_ext_i    = hypre_CSRBooleanMatrix_Get_I(A_ext);
      A_ext_j    = hypre_CSRBooleanMatrix_Get_BigJ(A_ext);
      num_rows_A_ext = hypre_CSRBooleanMatrix_Get_NRows(A_ext);
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
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_C = first_row_index_A + num_rows_diag_A - 1;
   C_diag_j    = hypre_CTAlloc(HYPRE_Int,  C_diag_size, HYPRE_MEMORY_HOST);
   if (C_offd_size)
   {
      C_offd_j    = hypre_CTAlloc(HYPRE_Int,  C_offd_size, HYPRE_MEMORY_HOST);
   }


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_j.
    *  Second Pass: Fill in C_offd_j.
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

         /* diag*ext */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
          *  That is, rows i3 having a column i2 of A_ext.
          *  For now, for each row i3 of A_ext we crudely check _all_
          *  columns to see whether one matches i2.
          *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
          *  C(i1,i3) .  This contributes to both the diag and offd
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

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *--------------------------------------------------------*/

                  if ( A_ext_row_map[i3] < first_row_index_A ||
                       A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                        C_offd_j[jj_count_offd] = i3;
                        jj_count_offd++;
                     }
                  }
                  else                                                /* diag */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3 - (HYPRE_Int)first_col_diag_A;
                        jj_count_diag++;
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

            /* offd * ext */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
             *  That is, rows i3 having a column i2 of A_ext.
             *  For now, for each row i3 of A_ext we crudely check _all_
             *  columns to see whether one matches i2.
             *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
             *  C(i1,i3) .  This contributes to both the diag and offd
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
                           C_offd_j[jj_count_offd] = i3;
                           jj_count_offd++;
                        }
                     }
                     else                                                /* diag */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                           C_diag_j[jj_count_diag] = i3 - (HYPRE_Int)first_row_index_A;
                           jj_count_diag++;
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

         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This i3-loop is for the diagonal block of A.
          *  It contributes to the diagonal block of C.
          *  For each entry (i2,i3) of A^T, A(i1,i2)*A(i3,i2) defines
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

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
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
       *  For each entry (i2,i3) of A^T, A*A^T defines C
       *-----------------------------------------------------------*/
      if (num_cols_offd_A)
      {

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
            {
               /* ... note that num_rows_diag_A == num_rows_offd_A */
               for ( jj3 = A_offd_i[i3]; jj3 < A_offd_i[i3 + 1]; jj3++ )
               {
                  if ( A_offd_j[jj3] == i2 )
                  {
                     /* row i3, column i2 of A; or,
                        row i2, column i3 of A^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution
                      *--------------------------------------------------------*/

                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                        B_marker[i3] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
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
      hypre_printf("\n");
      hypre_printf("  C_offd_j=");
      for ( jj3 = 0; jj3 < jj_count_offd; ++jj3) { hypre_printf("%i ", C_offd_j[jj3]); }
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

   C = hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_rows_A, row_starts_A,
                                       row_starts_A, num_cols_offd_C, C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   hypre_ParCSRBooleanMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRBooleanMatrixSetColStartsOwner(C, 0);

   C_diag = hypre_ParCSRBooleanMatrix_Get_Diag(C);
   hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i;
   hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j;

   if (num_cols_offd_C)
   {
      C_offd = hypre_ParCSRBooleanMatrix_Get_Offd(C);
      hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i;
      hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j;
      hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

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
      hypre_CSRBooleanMatrixDestroy(A_ext);
      A_ext = NULL;
   }
   hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   if ( num_rows_diag_A != n_rows_A )
   {
      hypre_TFree(A_ext_row_map, HYPRE_MEMORY_HOST);
   }

   return C;

}


/* ----------------------------------------------------------------------
 * hypre_BooleanMatTCommPkgCreate
 * generates a special comm_pkg for a Boolean matrix A - for use in multiplying
 * by its transpose, A * A^T
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d
 * ---------------------------------------------------------------------*/

HYPRE_Int
hypre_BooleanMatTCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
{
   MPI_Comm       comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
   HYPRE_BigInt  *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   HYPRE_BigInt   first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   HYPRE_BigInt  *col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   HYPRE_Int      num_rows_diag = hypre_CSRBooleanMatrix_Get_NRows(hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                      A));
   HYPRE_Int      num_cols_diag = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                      A));
   HYPRE_Int      num_cols_offd = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(
                                                                      A));
   HYPRE_BigInt  *row_starts = hypre_ParCSRBooleanMatrix_Get_RowStarts(A);

   HYPRE_Int      num_sends;
   HYPRE_Int     *send_procs;
   HYPRE_Int     *send_map_starts;
   HYPRE_Int     *send_map_elmts;
   HYPRE_Int      num_recvs;
   HYPRE_Int     *recv_procs;
   HYPRE_Int     *recv_vec_starts;

   hypre_ParCSRCommPkg  *comm_pkg = NULL;

   hypre_MatTCommPkgCreate_core (
      comm, col_map_offd, first_col_diag, col_starts,
      num_rows_diag, num_cols_diag, num_cols_offd, row_starts,
      hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A),
      hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A),
      hypre_CSRBooleanMatrix_Get_I( hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      hypre_CSRBooleanMatrix_Get_J( hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      hypre_CSRBooleanMatrix_Get_I( hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      hypre_CSRBooleanMatrix_Get_J( hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      0,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
   );

   /* Create communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   hypre_ParCSRBooleanMatrix_Get_CommPkgT(A) = comm_pkg;

   return hypre_error_flag;
}

/* ----------------------------------------------------------------------
 * hypre_BooleanMatvecCommPkgCreate
 * generates the comm_pkg for a Boolean matrix A , to be used for A*B.
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d
 * ---------------------------------------------------------------------*/

HYPRE_Int
hypre_BooleanMatvecCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
{
   MPI_Comm        comm = hypre_ParCSRBooleanMatrix_Get_Comm(A);
   HYPRE_BigInt   *col_map_offd = hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   HYPRE_BigInt    first_col_diag = hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   HYPRE_BigInt   *col_starts = hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   HYPRE_Int       num_cols_diag = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                       A));
   HYPRE_Int       num_cols_offd = hypre_CSRBooleanMatrix_Get_NCols(hypre_ParCSRBooleanMatrix_Get_Offd(
                                                                       A));

   HYPRE_Int       num_sends;
   HYPRE_Int      *send_procs;
   HYPRE_Int      *send_map_starts;
   HYPRE_Int      *send_map_elmts;
   HYPRE_Int       num_recvs;
   HYPRE_Int      *recv_procs;
   HYPRE_Int      *recv_vec_starts;

   hypre_ParCSRCommPkg  *comm_pkg = NULL;

   hypre_ParCSRCommPkgCreate_core
   (
      comm, col_map_offd, first_col_diag, col_starts,
      num_cols_diag, num_cols_offd,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
   );

   /* Create communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   hypre_ParCSRBooleanMatrix_Get_CommPkg(A) = comm_pkg;

   return hypre_error_flag;
}
