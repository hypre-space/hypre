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

#include "_hypre_parcsr_mv.h"

#include "_hypre_utilities.h"
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMat : multiplies two ParCSRMatrices A and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParCSRMatMat( hypre_ParCSRMatrix  *A,
                                        hypre_ParCSRMatrix  *B )
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int       *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);

   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int       *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   HYPRE_Int        first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_Int        last_col_diag_B;
   HYPRE_Int       *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int        num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int        num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int        num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_Int          *col_map_offd_C = NULL;
   HYPRE_Int          *map_B_to_C=NULL;

   hypre_CSRMatrix *C_diag = NULL;

   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_Int        num_cols_offd_C = 0;

   hypre_CSRMatrix *Bs_ext;

   hypre_CSRMatrix *Bext_diag;

   hypre_CSRMatrix *Bext_offd;

   hypre_CSRMatrix *AB_diag;
   hypre_CSRMatrix *AB_offd;
   HYPRE_Int        AB_offd_num_nonzeros;
   HYPRE_Int       *AB_offd_j;
   hypre_CSRMatrix *ABext_diag;
   hypre_CSRMatrix *ABext_offd;

   HYPRE_Int        n_rows_A, n_cols_A;
   HYPRE_Int        n_rows_B, n_cols_B;
   HYPRE_Int        cnt, i;
   HYPRE_Int        num_procs;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      hypre_error_in_arg(1);
      hypre_printf(" Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   last_col_diag_B = first_col_diag_B + num_cols_diag_B -1;

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext = hypre_ParCSRMatrixExtractBExt(B,A,1); /* contains communication
                                                        which should be explicitly included to allow for overlap */


      hypre_CSRMatrixSplit(Bs_ext, col_map_offd_B, first_col_diag_B, last_col_diag_B,
                           num_cols_offd_B, &num_cols_offd_C, &col_map_offd_C, &Bext_diag, &Bext_offd);
      hypre_CSRMatrixDestroy(Bs_ext);
      /* These are local and could be overlapped with communication */
      AB_diag = hypre_CSRMatrixMultiply(A_diag, B_diag);
      AB_offd = hypre_CSRMatrixMultiply(A_diag, B_offd);

      /* These require data from other processes */
      ABext_diag = hypre_CSRMatrixMultiply(A_offd, Bext_diag);
      ABext_offd = hypre_CSRMatrixMultiply(A_offd, Bext_offd);

      hypre_CSRMatrixDestroy(Bext_diag);
      hypre_CSRMatrixDestroy(Bext_offd);

      if (num_cols_offd_B)
      {
         HYPRE_Int i;
         map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i=0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
      }
      AB_offd_num_nonzeros = hypre_CSRMatrixNumNonzeros(AB_offd);
      AB_offd_j = hypre_CSRMatrixJ(AB_offd);
      for (i=0; i < AB_offd_num_nonzeros; i++)
      {
         AB_offd_j[i] = map_B_to_C[AB_offd_j[i]];
      }

      if (num_cols_offd_B)
      {
         hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
      }

      hypre_CSRMatrixNumCols(AB_diag) = num_cols_diag_B;
      hypre_CSRMatrixNumCols(ABext_diag) = num_cols_diag_B;
      hypre_CSRMatrixNumCols(AB_offd) = num_cols_offd_C;
      hypre_CSRMatrixNumCols(ABext_offd) = num_cols_offd_C;
      C_diag = hypre_CSRMatrixAdd(AB_diag, ABext_diag);
      C_offd = hypre_CSRMatrixAdd(AB_offd, ABext_offd);

      hypre_CSRMatrixDestroy(AB_diag);
      hypre_CSRMatrixDestroy(ABext_diag);
      hypre_CSRMatrixDestroy(AB_offd);
      hypre_CSRMatrixDestroy(ABext_offd);
   }
   else
   {
      C_diag = hypre_CSRMatrixMultiply(A_diag, B_diag);
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      hypre_CSRMatrixInitialize(C_offd);
   }

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/


   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag->num_nonzeros, C_offd->num_nonzeros);

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRTMatMatKT : multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParCSRTMatMatKT( hypre_ParCSRMatrix  *A,
                                           hypre_ParCSRMatrix  *B, HYPRE_Int keep_transpose)
{
   MPI_Comm   comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *AT_diag = NULL;

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);

   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int       *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   HYPRE_Int  first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_Int *col_starts_A = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int  num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int  num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int  num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_Int          *col_map_offd_C = NULL;
   HYPRE_Int          *map_B_to_C;

   hypre_CSRMatrix *C_diag = NULL;

   HYPRE_Int first_col_diag_C;
   HYPRE_Int last_col_diag_C;

   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_Int num_cols_offd_C = 0;

   HYPRE_Int j_indx;

   HYPRE_Int n_rows_A, n_cols_A;
   HYPRE_Int n_rows_B, n_cols_B;
   HYPRE_Int cnt;
   HYPRE_Int num_procs, my_id;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (n_rows_A != n_rows_B || num_rows_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);

   if (num_procs == 1)
   {
      C_diag = hypre_CSRMatrixMultiply(AT_diag, B_diag);
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      hypre_CSRMatrixInitialize(C_offd);
      if (keep_transpose)
      {
         A->diagT = AT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(AT_diag);
      }
   }
   else
   {
      hypre_CSRMatrix *AT_offd = NULL;
      hypre_CSRMatrix *C_tmp_diag = NULL;
      hypre_CSRMatrix *C_tmp_offd = NULL;
      hypre_CSRMatrix *C_int = NULL;
      hypre_CSRMatrix *C_ext = NULL;
      hypre_CSRMatrix *C_ext_diag = NULL;
      hypre_CSRMatrix *C_ext_offd = NULL;
      hypre_CSRMatrix *C_int_diag = NULL;
      hypre_CSRMatrix *C_int_offd = NULL;
      HYPRE_Int   *C_ext_i;
      HYPRE_Int   *C_ext_j;
      HYPRE_Complex   *C_ext_data;

      HYPRE_Int     *C_int_i;
      HYPRE_Int     *C_int_j = NULL;
      HYPRE_Real  *C_int_data = NULL;
      HYPRE_Int     num_cols = 0;

      MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg_A);
      HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
      HYPRE_Int *recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
      HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
      HYPRE_Int *send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
      HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

      hypre_ParCSRCommHandle *comm_handle = NULL;
      hypre_ParCSRCommHandle *comm_handle_2 = NULL;
      hypre_ParCSRCommPkg *tmp_comm_pkg;

      HYPRE_Int *jdata_recv_vec_starts;
      HYPRE_Int *jdata_send_map_starts;

      HYPRE_Int num_rows;
      HYPRE_Int num_nonzeros;
      HYPRE_Int i, j;

      HYPRE_Int   *C_tmp_offd_i;
      HYPRE_Int   *C_tmp_offd_j;

      HYPRE_Int   *send_map_elmts_A;

      hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
      C_int_diag = hypre_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = hypre_CSRMatrixMultiply(AT_offd, B_offd);

      hypre_ParCSRMatrixDiag(B) = C_int_diag;
      hypre_ParCSRMatrixOffd(B) = C_int_offd;
      C_int = hypre_MergeDiagAndOffd(B);
      hypre_ParCSRMatrixDiag(B) = B_diag;
      hypre_ParCSRMatrixOffd(B) = B_offd;
      /*C_ext = hypre_ExchangeRAPData(C_int, comm_pkg_A);*/ /* contains
      communication; should be explicitly included to allow for overlap */
      C_ext_i = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]+1, HYPRE_MEMORY_HOST);
      jdata_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, num_recvs+1, HYPRE_MEMORY_HOST);
      jdata_send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends+1, HYPRE_MEMORY_HOST);

      /*--------------------------------------------------------------------------
       * recompute C_int_i so that C_int_i[j+1] contains the number of
       * elements of row j (to be determined through send_map_elmnts on the
       * receiving end)
       *--------------------------------------------------------------------------*/

      if (num_recvs)
      {
         C_int_i = hypre_CSRMatrixI(C_int);
         C_int_j = hypre_CSRMatrixJ(C_int);
         C_int_data = hypre_CSRMatrixData(C_int);
         num_cols = hypre_CSRMatrixNumCols(C_int);
      }

      jdata_recv_vec_starts[0] = 0;
      for (i=0; i < num_recvs; i++)
      {
         jdata_recv_vec_starts[i+1] = C_int_i[recv_vec_starts[i+1]];
      }

      for (i=num_recvs; i > 0; i--)
      {
         for (j = recv_vec_starts[i]; j > recv_vec_starts[i-1]; j--)
         {
            C_int_i[j] -= C_int_i[j-1];
         }
      }

      /*--------------------------------------------------------------------------
       * initialize communication
       *--------------------------------------------------------------------------*/
      if (num_recvs && num_sends)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, &C_int_i[1], &C_ext_i[1]);
      }
      else if (num_recvs)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, &C_int_i[1], NULL);
      }
      else if (num_sends)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, NULL, &C_ext_i[1]);
      }

      tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
      hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
      hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_recvs;
      hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_sends;
      hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = recv_procs;
      hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = send_procs;

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

     /*--------------------------------------------------------------------------
      * compute num_nonzeros for C_ext
      *--------------------------------------------------------------------------*/
      for (i=0; i < num_sends; i++)
      {
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
        {
           C_ext_i[j+1] += C_ext_i[j];
        }
      }

      num_rows = send_map_starts[num_sends];
      num_nonzeros = C_ext_i[num_rows];
      if (num_nonzeros)
      {
         C_ext_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_HOST);
         C_ext_data = hypre_CTAlloc(HYPRE_Real, num_nonzeros, HYPRE_MEMORY_HOST);
      }

      for (i=0; i < num_sends+1; i++)
      {
         jdata_send_map_starts[i] = C_ext_i[send_map_starts[i]];
      }

      hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_send_map_starts;
      hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

      comm_handle = hypre_ParCSRCommHandleCreate(1, tmp_comm_pkg, C_int_data, C_ext_data);
      comm_handle_2 = hypre_ParCSRCommHandleCreate(11, tmp_comm_pkg, C_int_j, C_ext_j);
      C_ext = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

      hypre_CSRMatrixI(C_ext) = C_ext_i;
      if (num_nonzeros)
      {
         hypre_CSRMatrixJ(C_ext) = C_ext_j;
         hypre_CSRMatrixData(C_ext) = C_ext_data;
      }

      C_tmp_diag = hypre_CSRMatrixMultiply(AT_diag, B_diag);
      C_tmp_offd = hypre_CSRMatrixMultiply(AT_diag, B_offd);

      if (keep_transpose)
        A->diagT = AT_diag;
      else
        hypre_CSRMatrixDestroy(AT_diag);

      if (keep_transpose)
        A->offdT = AT_offd;
      else
        hypre_CSRMatrixDestroy(AT_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      hypre_ParCSRCommHandleDestroy(comm_handle_2);
      comm_handle = NULL;

      hypre_CSRMatrixDestroy(C_int);
      hypre_CSRMatrixDestroy(C_int_diag);
      hypre_CSRMatrixDestroy(C_int_offd);

      hypre_TFree(jdata_recv_vec_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(jdata_send_map_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */

      first_col_diag_C = first_col_diag_B;
      last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

      if (C_ext)
      {
         hypre_CSRMatrixSplit(C_ext, col_map_offd_B, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_B, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
      }

      C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);

      if (num_cols_offd_B)
      {
         map_B_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_B, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i=0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
         for (i=0; i < C_tmp_offd_i[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_B_to_C[j_indx];
         }
         hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
      }

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/
      send_map_elmts_A = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_diag = hypre_CSRMatrixAddPartial(C_tmp_diag, C_ext_diag, send_map_elmts_A);
      hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      C_offd = hypre_CSRMatrixAddPartial(C_tmp_offd, C_ext_offd, send_map_elmts_A);
      hypre_CSRMatrixDestroy(C_tmp_diag);
      hypre_CSRMatrixDestroy(C_tmp_offd);
      hypre_CSRMatrixDestroy(C_ext_diag);
      hypre_CSRMatrixDestroy(C_ext_offd);
   }

   C = hypre_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A,
                                col_starts_B, num_cols_offd_C, C_diag->num_nonzeros, C_offd->num_nonzeros);

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   return C;

}

hypre_ParCSRMatrix *hypre_ParCSRTMatMat( hypre_ParCSRMatrix  *A,
                    hypre_ParCSRMatrix  *B)
{
   return hypre_ParCSRTMatMatKT( A, B, 0);
}

hypre_ParCSRMatrix *hypre_ParCSRMatrixRAPKT( hypre_ParCSRMatrix *R,
                                             hypre_ParCSRMatrix *A,
                                             hypre_ParCSRMatrix *P,
                                             HYPRE_Int keep_transpose )
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int       *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int       *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);

   HYPRE_Int        first_col_diag_P = hypre_ParCSRMatrixFirstColDiag(P);
   HYPRE_Int       *col_starts_P = hypre_ParCSRMatrixColStarts(P);
   HYPRE_Int        num_rows_diag_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        num_cols_diag_P = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int        num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);

   hypre_ParCSRMatrix *Q;
   HYPRE_Int          *col_map_offd_Q = NULL;
   HYPRE_Int          *map_P_to_Q=NULL;

   hypre_CSRMatrix *Q_diag = NULL;

   hypre_CSRMatrix *Q_offd = NULL;

   HYPRE_Int        num_cols_offd_Q = 0;

   hypre_CSRMatrix *Ps_ext;

   hypre_CSRMatrix *Pext_diag;

   hypre_CSRMatrix *Pext_offd;

   hypre_CSRMatrix *AP_diag;
   hypre_CSRMatrix *AP_offd;
   HYPRE_Int        AP_offd_num_nonzeros;
   HYPRE_Int       *AP_offd_j;
   hypre_CSRMatrix *APext_diag;
   hypre_CSRMatrix *APext_offd;

   hypre_ParCSRCommPkg *comm_pkg_R = hypre_ParCSRMatrixCommPkg(R);

   hypre_CSRMatrix *R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrix *RT_diag = NULL;

   hypre_CSRMatrix *R_offd = hypre_ParCSRMatrixOffd(R);

   HYPRE_Int    num_rows_diag_R = hypre_CSRMatrixNumRows(R_diag);
   HYPRE_Int    num_cols_diag_R = hypre_CSRMatrixNumCols(R_diag);
   HYPRE_Int    num_cols_offd_R = hypre_CSRMatrixNumCols(R_offd);

   HYPRE_Int *col_starts_R = hypre_ParCSRMatrixColStarts(R);

   hypre_ParCSRMatrix *C;
   HYPRE_Int          *col_map_offd_C = NULL;
   HYPRE_Int          *map_Q_to_C;

   hypre_CSRMatrix *C_diag = NULL;

   HYPRE_Int    first_col_diag_C;
   HYPRE_Int    last_col_diag_C;

   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_Int        num_cols_offd_C = 0;

   HYPRE_Int        j_indx;

   HYPRE_Int        n_rows_R, n_cols_R;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Int        n_rows_A, n_cols_A;
   HYPRE_Int        n_rows_P, n_cols_P;
   HYPRE_Int        cnt, i;


   n_rows_R = hypre_ParCSRMatrixGlobalNumRows(R);
   n_cols_R = hypre_ParCSRMatrixGlobalNumCols(R);
   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_P = hypre_ParCSRMatrixGlobalNumRows(P);
   n_cols_P = hypre_ParCSRMatrixGlobalNumCols(P);

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (n_rows_R != n_rows_A || num_rows_diag_R != num_rows_diag_A
         || n_cols_A != n_rows_P || num_cols_diag_A != num_rows_diag_P)
   {
        hypre_error_w_msg(HYPRE_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
        return NULL;
   }


   /*hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);*/

   if (num_procs > 1)
   {
      HYPRE_Int        last_col_diag_P;
      hypre_CSRMatrix *RT_offd = NULL;
      hypre_CSRMatrix *C_tmp_diag = NULL;
      hypre_CSRMatrix *C_tmp_offd = NULL;
      hypre_CSRMatrix *C_int = NULL;
      hypre_CSRMatrix *C_ext = NULL;
      hypre_CSRMatrix *C_ext_diag = NULL;
      hypre_CSRMatrix *C_ext_offd = NULL;
      hypre_CSRMatrix *C_int_diag = NULL;
      hypre_CSRMatrix *C_int_offd = NULL;

      HYPRE_Int   *C_tmp_offd_i;
      HYPRE_Int   *C_tmp_offd_j;

      HYPRE_Int       *send_map_elmts_R;
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Ps_ext = hypre_ParCSRMatrixExtractBExt(P,A,1); /* contains communication
                                                        which should be explicitly included to allow for overlap */
      if (num_cols_offd_A)
      {
         last_col_diag_P = first_col_diag_P + num_cols_diag_P -1;
         hypre_CSRMatrixSplit(Ps_ext, col_map_offd_P, first_col_diag_P, last_col_diag_P,
                              num_cols_offd_P, &num_cols_offd_Q, &col_map_offd_Q, &Pext_diag, &Pext_offd);
         /* These require data from other processes */
         APext_diag = hypre_CSRMatrixMultiply(A_offd, Pext_diag);
         APext_offd = hypre_CSRMatrixMultiply(A_offd, Pext_offd);

         hypre_CSRMatrixDestroy(Pext_diag);
         hypre_CSRMatrixDestroy(Pext_offd);
      }
      else
      {
         num_cols_offd_Q = num_cols_offd_P;
         col_map_offd_Q = hypre_CTAlloc(HYPRE_Int, num_cols_offd_Q, HYPRE_MEMORY_HOST);
         for (i=0; i < num_cols_offd_P; i++)
         {
            col_map_offd_Q[i] = col_map_offd_P[i];
         }
      }
      hypre_CSRMatrixDestroy(Ps_ext);
      /* These are local and could be overlapped with communication */
      AP_diag = hypre_CSRMatrixMultiply(A_diag, P_diag);

      if (num_cols_offd_P)
      {
         HYPRE_Int i;
         AP_offd = hypre_CSRMatrixMultiply(A_diag, P_offd);
         if (num_cols_offd_Q > num_cols_offd_P)
         {
            map_P_to_Q = hypre_CTAlloc(HYPRE_Int,num_cols_offd_P, HYPRE_MEMORY_HOST);

            cnt = 0;
            for (i=0; i < num_cols_offd_Q; i++)
            {
               if (col_map_offd_Q[i] == col_map_offd_P[cnt])
               {
                  map_P_to_Q[cnt++] = i;
                  if (cnt == num_cols_offd_P)
                  {
                     break;
                  }
               }
            }
            AP_offd_num_nonzeros = hypre_CSRMatrixNumNonzeros(AP_offd);
            AP_offd_j = hypre_CSRMatrixJ(AP_offd);
            for (i=0; i < AP_offd_num_nonzeros; i++)
            {
               AP_offd_j[i] = map_P_to_Q[AP_offd_j[i]];
            }

            hypre_TFree(map_P_to_Q, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixNumCols(AP_offd) = num_cols_offd_Q;
         }
      }

      if (num_cols_offd_A) /* number of rows for Pext_diag */
      {
         Q_diag = hypre_CSRMatrixAdd(AP_diag, APext_diag);
         hypre_CSRMatrixDestroy(AP_diag);
         hypre_CSRMatrixDestroy(APext_diag);
      }
      else
      {
         Q_diag = AP_diag;
      }

      if (num_cols_offd_P && num_cols_offd_A)
      {
         Q_offd = hypre_CSRMatrixAdd(AP_offd, APext_offd);
         hypre_CSRMatrixDestroy(APext_offd);
         hypre_CSRMatrixDestroy(AP_offd);
      }
      else if (num_cols_offd_A)
      {
         Q_offd = APext_offd;
      }
      else if (num_cols_offd_P)
      {
         Q_offd = AP_offd;
      }
      else
      {
         Q_offd = hypre_CSRMatrixClone(A_offd);
      }

      Q = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_P, row_starts_A,
                                   col_starts_P, num_cols_offd_Q,
                                   Q_diag->num_nonzeros, Q_offd->num_nonzeros);

      /* Note that C does not own the partitionings */
      hypre_ParCSRMatrixSetRowStartsOwner(Q,0);
      hypre_ParCSRMatrixSetColStartsOwner(Q,0);
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(Q));
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(Q));
      hypre_ParCSRMatrixDiag(Q) = Q_diag;
      hypre_ParCSRMatrixOffd(Q) = Q_offd;
      hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;

      hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);
      C_tmp_diag = hypre_CSRMatrixMultiply(RT_diag, Q_diag);
      if (num_cols_offd_Q)
      {
         C_tmp_offd = hypre_CSRMatrixMultiply(RT_diag, Q_offd);
         if (C_tmp_offd->num_nonzeros == 0)
         {
            C_tmp_offd->num_cols = 0;
         }
      }
      else
      {
         C_tmp_offd = hypre_CSRMatrixClone(Q_offd);
         hypre_CSRMatrixNumRows(C_tmp_offd) = num_cols_diag_R;
      }

      if (keep_transpose)
      {
         R->diagT = RT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(RT_diag);
      }

      if (num_cols_offd_R)
      {
         hypre_CSRMatrixTranspose(R_offd, &RT_offd, 1);
         C_int_diag = hypre_CSRMatrixMultiply(RT_offd, Q_diag);
         C_int_offd = hypre_CSRMatrixMultiply(RT_offd, Q_offd);

         hypre_ParCSRMatrixDiag(Q) = C_int_diag;
         hypre_ParCSRMatrixOffd(Q) = C_int_offd;
         C_int = hypre_MergeDiagAndOffd(Q);
         hypre_ParCSRMatrixDiag(Q) = Q_diag;
         hypre_ParCSRMatrixOffd(Q) = Q_offd;
      }
      else
      {
         C_int = hypre_CSRMatrixCreate(0,0,0);
         hypre_CSRMatrixInitialize(C_int);
      }

      C_ext = hypre_ExchangeRAPData(C_int, comm_pkg_R); /* contains
      communication; should be explicitly included to allow for overlap */

      hypre_CSRMatrixDestroy(C_int);
      if (num_cols_offd_R)
      {
         hypre_CSRMatrixDestroy(C_int_diag);
         hypre_CSRMatrixDestroy(C_int_offd);
         if (keep_transpose)
         {
            R->offdT = RT_offd;
         }
         else
         {
            hypre_CSRMatrixDestroy(RT_offd);
         }
      }

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */

      if (C_ext)
      {
         first_col_diag_C = first_col_diag_P;
         last_col_diag_C = first_col_diag_P + num_cols_diag_P - 1;

         hypre_CSRMatrixSplit(C_ext, col_map_offd_Q, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_Q, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
         /*if (C_ext_offd->num_nonzeros == 0) C_ext_offd->num_cols = 0;*/
      }

      if (num_cols_offd_Q && C_tmp_offd->num_cols)
      {
         C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
         C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);

         map_Q_to_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_Q, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i=0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_Q[cnt])
            {
               map_Q_to_C[cnt++] = i;
               if (cnt == num_cols_offd_Q)
               {
                  break;
               }
            }
         }
         for (i=0; i < C_tmp_offd_i[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_Q_to_C[j_indx];
         }
         hypre_TFree(map_Q_to_C, HYPRE_MEMORY_HOST);
         hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      }
      hypre_ParCSRMatrixDestroy(Q);

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/
      send_map_elmts_R = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_R);
      if (C_ext_diag)
      {
         C_diag = hypre_CSRMatrixAddPartial(C_tmp_diag,C_ext_diag, send_map_elmts_R);
         hypre_CSRMatrixDestroy(C_tmp_diag);
         hypre_CSRMatrixDestroy(C_ext_diag);
      }
      else
         C_diag = C_tmp_diag;
      if (C_ext_offd)
      {
         C_offd = hypre_CSRMatrixAddPartial(C_tmp_offd,C_ext_offd, send_map_elmts_R);
         hypre_CSRMatrixDestroy(C_tmp_offd);
         hypre_CSRMatrixDestroy(C_ext_offd);
      }
      else
      {
         C_offd = C_tmp_offd;
      }
   }
   else
   {
      Q_diag = hypre_CSRMatrixMultiply(A_diag, P_diag);
      hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);
      C_diag = hypre_CSRMatrixMultiply(RT_diag, Q_diag);
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      hypre_CSRMatrixInitialize(C_offd);
      if (keep_transpose)
      {
         R->diagT = RT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(RT_diag);
      }
      hypre_CSRMatrixDestroy(Q_diag);
   }

   C = hypre_ParCSRMatrixCreate(comm, n_cols_R, n_cols_P, col_starts_R,
                                col_starts_P, num_cols_offd_C, 0, 0);

/* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetColStartsOwner(P,0);
   hypre_ParCSRMatrixSetColStartsOwner(R,0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   if (C_offd)
   {
      hypre_ParCSRMatrixOffd(C) = C_offd;
   }
   else
   {
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      hypre_CSRMatrixInitialize(C_offd);
      hypre_ParCSRMatrixOffd(C) = C_offd;
   }

   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   if (num_procs > 1)
   {
      /* hypre_GenerateRAPCommPkg(RAP, A); */
      hypre_MatvecCommPkgCreate(C);
   }

   return C;
}

hypre_ParCSRMatrix *hypre_ParCSRMatrixRAP( hypre_ParCSRMatrix *R,
           hypre_ParCSRMatrix  *A,
           hypre_ParCSRMatrix  *P )
{
   return hypre_ParCSRMatrixRAPKT( R, A, P, 0);
}

HYPRE_Int hypre_CSRMatrixSplit(hypre_CSRMatrix *Bs_ext, HYPRE_Int *col_map_offd_B,
                               HYPRE_Int first_col_diag_B, HYPRE_Int last_col_diag_B,
                               HYPRE_Int num_cols_offd_B, HYPRE_Int *num_cols_offd_C_ptr,
                               HYPRE_Int **col_map_offd_C_ptr,
                               hypre_CSRMatrix **Bext_diag_ptr, hypre_CSRMatrix **Bext_offd_ptr)
{
   HYPRE_Complex   *Bs_ext_data = hypre_CSRMatrixData(Bs_ext);
   HYPRE_Int    *Bs_ext_i    = hypre_CSRMatrixI(Bs_ext);
   HYPRE_Int    *Bs_ext_j    = hypre_CSRMatrixJ(Bs_ext);
   HYPRE_Int num_rows_Bext = hypre_CSRMatrixNumRows(Bs_ext);
   HYPRE_Int B_ext_diag_size = 0;
   HYPRE_Int B_ext_offd_size = 0;
   HYPRE_Int *B_ext_diag_i = NULL;
   HYPRE_Int *B_ext_diag_j = NULL;
   HYPRE_Complex *B_ext_diag_data = NULL;
   HYPRE_Int *B_ext_offd_i = NULL;
   HYPRE_Int *B_ext_offd_j = NULL;
   HYPRE_Complex *B_ext_offd_data = NULL;
   HYPRE_Int       *my_diag_array;
   HYPRE_Int       *my_offd_array;
   HYPRE_Int       *temp;
   HYPRE_Int        max_num_threads;
   HYPRE_Int        cnt = 0, value = 0;
   hypre_CSRMatrix *Bext_diag = NULL;
   hypre_CSRMatrix *Bext_offd = NULL;
   HYPRE_Int       *col_map_offd_C = NULL;
   HYPRE_Int        num_cols_offd_C=0;

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
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            my_offd_size++;
         else
            my_diag_size++;
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
           B_ext_diag_j = hypre_CTAlloc(HYPRE_Int, B_ext_diag_size, HYPRE_MEMORY_HOST);
           B_ext_diag_data = hypre_CTAlloc(HYPRE_Complex, B_ext_diag_size, HYPRE_MEMORY_HOST);
        }
        if (B_ext_offd_size)
        {
           B_ext_offd_j = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size, HYPRE_MEMORY_HOST);
           B_ext_offd_data = hypre_CTAlloc(HYPRE_Complex, B_ext_offd_size, HYPRE_MEMORY_HOST);
        }
        if (B_ext_offd_size || num_cols_offd_B)
        {
           temp = hypre_CTAlloc(HYPRE_Int, B_ext_offd_size+num_cols_offd_B, HYPRE_MEMORY_HOST);
        }
     }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     cnt_offd = B_ext_offd_i[ns];
     cnt_diag = B_ext_diag_i[ns];
     for (i=ns; i < ne; i++)
     {
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
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
              hypre_qsort0(temp, 0, cnt-1);
              num_cols_offd_C = 1;
              value = temp[0];
              for (i=1; i < cnt; i++)
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
              col_map_offd_C = hypre_CTAlloc(HYPRE_Int,num_cols_offd_C, HYPRE_MEMORY_HOST);
           }

           for (i=0; i < num_cols_offd_C; i++)
           {
              col_map_offd_C[i] = temp[i];
           }

           hypre_TFree(temp, HYPRE_MEMORY_HOST);
        }
     }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

     for (i=ns; i < ne; i++)
     {
        for (j=B_ext_offd_i[i]; j < B_ext_offd_i[i+1]; j++)
        {
            B_ext_offd_j[j] = hypre_BinarySearch(col_map_offd_C, B_ext_offd_j[j],
                                                 num_cols_offd_C);
        }
     }

    } /* end parallel region */
    hypre_TFree(my_diag_array, HYPRE_MEMORY_HOST);
    hypre_TFree(my_offd_array, HYPRE_MEMORY_HOST);

    Bext_diag = hypre_CSRMatrixCreate(num_rows_Bext, last_col_diag_B-first_col_diag_B+1, B_ext_diag_size);
    Bext_offd = hypre_CSRMatrixCreate(num_rows_Bext, num_cols_offd_C, B_ext_offd_size);
    hypre_CSRMatrixI(Bext_diag) = B_ext_diag_i;
    hypre_CSRMatrixJ(Bext_diag) = B_ext_diag_j;
    hypre_CSRMatrixData(Bext_diag) = B_ext_diag_data;
    hypre_CSRMatrixI(Bext_offd) = B_ext_offd_i;
    hypre_CSRMatrixJ(Bext_offd) = B_ext_offd_j;
    hypre_CSRMatrixData(Bext_offd) = B_ext_offd_data;
    *col_map_offd_C_ptr = col_map_offd_C;
    *Bext_diag_ptr = Bext_diag;
    *Bext_offd_ptr = Bext_offd;
    *num_cols_offd_C_ptr = num_cols_offd_C;

    return hypre_error_flag;
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
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1, HYPRE_MEMORY_SHARED);

   for (ia = 0; ia < ncols_A; ia++)
      marker[ia] = -1;

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
               break;
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
               break;
         }
      }
   }

   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(map, HYPRE_MEMORY_HOST);
   hypre_TFree(temp, HYPRE_MEMORY_HOST);
   return C;
}

/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings.
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily?
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_ExchangeRAPData( hypre_CSRMatrix *RAP_int,
                       hypre_ParCSRCommPkg *comm_pkg_RT)
{
   HYPRE_Int     *RAP_int_i;
   HYPRE_Int     *RAP_int_j = NULL;
   HYPRE_Real  *RAP_int_data = NULL;
   HYPRE_Int     num_cols = 0;

   MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg_RT);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
   HYPRE_Int *recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg_RT);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_RT);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
   HYPRE_Int *send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg_RT);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);

   hypre_CSRMatrix *RAP_ext;

   HYPRE_Int     *RAP_ext_i;
   HYPRE_Int     *RAP_ext_j = NULL;
   HYPRE_Real  *RAP_ext_data = NULL;

   hypre_ParCSRCommHandle *comm_handle = NULL;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int num_rows;
   HYPRE_Int num_nonzeros;
   HYPRE_Int i, j;
   HYPRE_Int num_procs;

   hypre_MPI_Comm_size(comm,&num_procs);

   RAP_ext_i = hypre_CTAlloc(HYPRE_Int,  send_map_starts[num_sends]+1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = hypre_TAlloc(HYPRE_Int,  num_recvs+1, HYPRE_MEMORY_HOST);
   jdata_send_map_starts = hypre_TAlloc(HYPRE_Int,  num_sends+1, HYPRE_MEMORY_HOST);

/*--------------------------------------------------------------------------
 * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
 * elements of row j (to be determined through send_map_elmnts on the
 * receiving end)
 *--------------------------------------------------------------------------*/

   if (num_recvs)
   {
        RAP_int_i = hypre_CSRMatrixI(RAP_int);
        RAP_int_j = hypre_CSRMatrixJ(RAP_int);
        RAP_int_data = hypre_CSRMatrixData(RAP_int);
        num_cols = hypre_CSRMatrixNumCols(RAP_int);
   }

   jdata_recv_vec_starts[0] = 0;
   for (i=0; i < num_recvs; i++)
   {
        jdata_recv_vec_starts[i+1] = RAP_int_i[recv_vec_starts[i+1]];
   }

   for (i=num_recvs; i > 0; i--)
        for (j = recv_vec_starts[i]; j > recv_vec_starts[i-1]; j--)
                RAP_int_i[j] -= RAP_int_i[j-1];

/*--------------------------------------------------------------------------
 * initialize communication
 *--------------------------------------------------------------------------*/

   if (num_recvs && num_sends)
      comm_handle = hypre_ParCSRCommHandleCreate(12,comm_pkg_RT,
                &RAP_int_i[1], &RAP_ext_i[1]);
   else if (num_recvs)
      comm_handle = hypre_ParCSRCommHandleCreate(12,comm_pkg_RT,
                &RAP_int_i[1], NULL);
   else if (num_sends)
      comm_handle = hypre_ParCSRCommHandleCreate(12,comm_pkg_RT,
                NULL, &RAP_ext_i[1]);

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,  1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = send_procs;

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

/*--------------------------------------------------------------------------
 * compute num_nonzeros for RAP_ext
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_sends; i++)
        for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
                RAP_ext_i[j+1] += RAP_ext_i[j];

   num_rows = send_map_starts[num_sends];
   num_nonzeros = RAP_ext_i[num_rows];
   if (num_nonzeros)
   {
      RAP_ext_j = hypre_TAlloc(HYPRE_Int,  num_nonzeros, HYPRE_MEMORY_HOST);
      RAP_ext_data = hypre_TAlloc(HYPRE_Real,  num_nonzeros, HYPRE_MEMORY_HOST);
   }

   for (i=0; i < num_sends+1; i++)
   {
        jdata_send_map_starts[i] = RAP_ext_i[send_map_starts[i]];
   }

   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_send_map_starts;
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,RAP_int_data,
                                        RAP_ext_data);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   comm_handle = hypre_ParCSRCommHandleCreate(11,tmp_comm_pkg,RAP_int_j,
                                        RAP_ext_j);
   RAP_ext = hypre_CSRMatrixCreate(num_rows,num_cols,num_nonzeros);

   hypre_CSRMatrixI(RAP_ext) = RAP_ext_i;
   if (num_nonzeros)
   {
      hypre_CSRMatrixJ(RAP_ext) = RAP_ext_j;
      hypre_CSRMatrixData(RAP_ext) = RAP_ext_data;
   }

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   hypre_TFree(jdata_recv_vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(jdata_send_map_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);

   return RAP_ext;
}

