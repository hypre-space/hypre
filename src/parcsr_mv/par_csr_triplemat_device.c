/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

#if defined(HYPRE_USING_CUDA)

hypre_ParCSRMatrix*
hypre_ParCSRMatMatDevice( hypre_ParCSRMatrix  *A,
                          hypre_ParCSRMatrix  *B )
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_BigInt    *row_starts_A    = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   /* HYPRE_BigInt       *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B); */

   HYPRE_BigInt     first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt     last_col_diag_B;
   HYPRE_BigInt    *col_starts_B     = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int        num_rows_diag_B  = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int        num_cols_diag_B  = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int        num_cols_offd_B  = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_BigInt       *col_map_offd_C = NULL;
   HYPRE_Int          *map_B_to_C = NULL;

   hypre_CSRMatrix *C_diag = NULL;
   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_Int        num_cols_offd_C = 0;

   hypre_CSRMatrix *Bext;
   hypre_CSRMatrix *Bext_diag;
   hypre_CSRMatrix *Bext_offd;

   hypre_CSRMatrix *AB_diag;
   hypre_CSRMatrix *AB_offd;
   HYPRE_Int        AB_offd_num_nonzeros;
   HYPRE_Int       *AB_offd_j;
   hypre_CSRMatrix *ABext_diag;
   hypre_CSRMatrix *ABext_offd;

   HYPRE_BigInt     n_rows_A, n_cols_A;
   HYPRE_BigInt     n_rows_B, n_cols_B;
   HYPRE_Int        num_procs;
   HYPRE_Int        my_id;

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

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/
   if (num_procs > 1)
   {
      void *request;
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      /* contains communication which should be explicitly included to allow for overlap */
      hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, 1, &request);

      //Bext = hypre_ParCSRMatrixExtractBExtDeviceWait(request);

      /* These are local and could be overlapped with communication */
      AB_diag = hypre_CSRMatrixMultiply(A_diag, B_diag);
      AB_offd = hypre_CSRMatrixMultiply(A_diag, B_offd);

      Bext = hypre_ParCSRMatrixExtractBExtDeviceWait(request);

      hypre_CSRMatrixSplitDevice(Bext, first_col_diag_B, last_col_diag_B,
                                 num_cols_offd_B, hypre_ParCSRMatrixDeviceColMapOffd(B), &map_B_to_C,
                                 &num_cols_offd_C, &col_map_offd_C,
                                 &Bext_diag, &Bext_offd);
      hypre_CSRMatrixDestroy(Bext);

      /* These require data from other processes */
      ABext_diag = hypre_CSRMatrixMultiply(A_offd, Bext_diag);
      ABext_offd = hypre_CSRMatrixMultiply(A_offd, Bext_offd);

      hypre_CSRMatrixDestroy(Bext_diag);
      hypre_CSRMatrixDestroy(Bext_offd);

      /* adjust AB_offd cols indices and number of cols of this matrix
       * NOTE: cannot adjust the cols of B_offd (which needs less work) beforehand, unless want to change B */
      AB_offd_num_nonzeros = hypre_CSRMatrixNumNonzeros(AB_offd);
      AB_offd_j = hypre_CSRMatrixJ(AB_offd);
      /* RL: TODO XXX thrust manual says map should not overlap result. but it seems work here. */
      HYPRE_THRUST_CALL(gather, AB_offd_j, AB_offd_j + AB_offd_num_nonzeros, map_B_to_C, AB_offd_j);

      hypre_TFree(map_B_to_C, HYPRE_MEMORY_DEVICE);

      /*
      hypre_CSRMatrixNumCols(AB_diag)    = num_cols_diag_B;
      hypre_CSRMatrixNumCols(ABext_diag) = num_cols_diag_B;
      */
      /* !!! adjust num of cols of AB_offd */
      hypre_CSRMatrixNumCols(AB_offd)    = num_cols_offd_C;
      /*
      hypre_CSRMatrixNumCols(ABext_offd) = num_cols_offd_C;
      */

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
      C_offd = hypre_CSRMatrixCreate(num_rows_diag_A, 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag->num_nonzeros, C_offd->num_nonzeros);

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRMatrixSetColStartsOwner(C, 0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   return C;
}

hypre_ParCSRMatrix*
hypre_ParCSRTMatMatKTDevice( hypre_ParCSRMatrix  *A,
                             hypre_ParCSRMatrix  *B,
                             HYPRE_Int            keep_transpose)
{
   MPI_Comm             comm       = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg_A = NULL;

   hypre_CSRMatrix *A_diag  = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd  = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *B_diag  = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd  = hypre_ParCSRMatrixOffd(B);
   hypre_CSRMatrix *AT_diag = NULL;

   HYPRE_Int    num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int    num_cols_diag_A  = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int    num_rows_diag_B  = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int    num_cols_diag_B  = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int    num_cols_offd_B  = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_BigInt first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt last_col_diag_B  = first_col_diag_B + num_cols_diag_B - 1;

   /* HYPRE_BigInt *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B); */

   HYPRE_BigInt *col_starts_A = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt *col_starts_B = hypre_ParCSRMatrixColStarts(B);

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix *C_diag = NULL;
   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_BigInt *col_map_offd_C = NULL;
   HYPRE_Int *map_B_to_C;
   /*
   HYPRE_Int  first_col_diag_C;
   HYPRE_Int  last_col_diag_C;
   */
   HYPRE_Int  num_cols_offd_C = 0;

   HYPRE_BigInt n_rows_A, n_cols_A;
   HYPRE_BigInt n_rows_B, n_cols_B;
   HYPRE_Int num_procs, my_id;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   hypre_MPI_Comm_size(comm, &num_procs);
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
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
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
      hypre_CSRMatrix *AT_offd;
      hypre_CSRMatrix *C_int;
      hypre_CSRMatrix *C_int_diag;
      hypre_CSRMatrix *C_int_offd;
      hypre_CSRMatrix *C_ext;
      hypre_CSRMatrix *C_ext_diag;
      hypre_CSRMatrix *C_ext_offd;
      hypre_CSRMatrix *C_tmp_diag;
      hypre_CSRMatrix *C_tmp_offd;
      HYPRE_Int       *C_tmp_offd_j;
      HYPRE_Int        num_sends_A;
      HYPRE_Int        num_elmts_send_A;
      HYPRE_Int       *h_send_map_elmts_A;
      HYPRE_Int       *d_send_map_elmts_A;
      void            *request;

      hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);

      /* Remark: do not do merge B first and then multiply.
       * A merged B with global column range has difficulty in GPU spmm rowest */
      C_int_diag = hypre_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = hypre_CSRMatrixMultiply(AT_offd, B_offd);

      hypre_ParCSRMatrixDiag(B) = C_int_diag;
      hypre_ParCSRMatrixOffd(B) = C_int_offd;

      C_int = hypre_MergeDiagAndOffdDevice(B);

      hypre_ParCSRMatrixDiag(B) = B_diag;
      hypre_ParCSRMatrixOffd(B) = B_offd;

      if (!hypre_ParCSRMatrixCommPkg(A))
      {
         hypre_MatvecCommPkgCreate(A);
      }
      comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

      hypre_ExchangeExternalRowsDeviceInit(C_int, comm_pkg_A, &request);

      //C_ext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_CSRMatrixDestroy(C_int_diag);
      hypre_CSRMatrixDestroy(C_int_offd);

      C_tmp_diag = hypre_CSRMatrixMultiply(AT_diag, B_diag);
      C_tmp_offd = hypre_CSRMatrixMultiply(AT_diag, B_offd);

      if (keep_transpose)
      {
        A->diagT = AT_diag;
      }
      else
      {
        hypre_CSRMatrixDestroy(AT_diag);
      }

      if (keep_transpose)
      {
        A->offdT = AT_offd;
      }
      else
      {
        hypre_CSRMatrixDestroy(AT_offd);
      }

      C_ext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_CSRMatrixDestroy(C_int);

      hypre_CSRMatrixSplitDevice(C_ext, first_col_diag_B, last_col_diag_B,
                                 num_cols_offd_B, hypre_ParCSRMatrixDeviceColMapOffd(B), &map_B_to_C,
                                 &num_cols_offd_C, &col_map_offd_C,
                                 &C_ext_diag, &C_ext_offd);
      hypre_CSRMatrixDestroy(C_ext);

      /* adjust C_tmp_offd cols indices and number of cols of this matrix
       * NOTE: cannot adjust the cols of B_offd (which needs less work) beforehand, unless want to change B */
      C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);
      HYPRE_THRUST_CALL(gather, C_tmp_offd_j, C_tmp_offd_j + hypre_CSRMatrixNumNonzeros(C_tmp_offd),
                        map_B_to_C, C_tmp_offd_j);
      hypre_TFree(map_B_to_C, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;

      /* add two parts together: a more general add, repeated rows */
      num_sends_A        = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
      num_elmts_send_A   = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, num_sends_A);
      h_send_map_elmts_A = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      d_send_map_elmts_A = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_A);
      if (d_send_map_elmts_A == NULL)
      {
         d_send_map_elmts_A = hypre_TAlloc(HYPRE_Int, num_elmts_send_A, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(d_send_map_elmts_A, h_send_map_elmts_A, HYPRE_Int, num_elmts_send_A,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg_A) = d_send_map_elmts_A;
      }

      hypre_assert(hypre_CSRMatrixNumRows(C_ext_diag) == num_elmts_send_A);
      hypre_assert(hypre_CSRMatrixNumRows(C_ext_offd) == num_elmts_send_A);

      C_diag = hypre_CSRMatrixAddPartialDevice(C_tmp_diag, C_ext_diag, d_send_map_elmts_A);
      C_offd = hypre_CSRMatrixAddPartialDevice(C_tmp_offd, C_ext_offd, d_send_map_elmts_A);

      hypre_CSRMatrixDestroy(C_tmp_diag);
      hypre_CSRMatrixDestroy(C_tmp_offd);
      hypre_CSRMatrixDestroy(C_ext_diag);
      hypre_CSRMatrixDestroy(C_ext_offd);
   }

   C = hypre_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A, col_starts_B,
                                num_cols_offd_C, C_diag->num_nonzeros, C_offd->num_nonzeros);

   /* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C, 0);
   hypre_ParCSRMatrixSetColStartsOwner(C, 0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   /* Move the diagonal entry to the first of each row */
   hypre_CSRMatrixMoveDiagFirstDevice(C_diag);

   hypre_SyncCudaComputeStream(hypre_handle());

   return C;
}

#endif // #if defined(HYPRE_USING_CUDA)

