/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_SYCL)

HYPRE_Int
hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix   *A,
                                       HYPRE_Int             indices_len,
                                       HYPRE_Int            *indices,
                                       hypre_ParCSRCommPkg  *comm_pkg,
                                       HYPRE_Int             want_data,
                                       void                **request_ptr)
{
   HYPRE_Int      i, j;
   HYPRE_Int      num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   HYPRE_Int     *d_send_i, *send_i, *d_send_map, *d_recv_i, *recv_i;
   HYPRE_BigInt  *d_send_j, *d_recv_j;
   HYPRE_Int     *send_jstarts, *recv_jstarts;
   HYPRE_Complex *d_send_a = nullptr, *d_recv_a = nullptr;
   hypre_ParCSRCommPkg     *comm_pkg_j;
   hypre_ParCSRCommHandle  *comm_handle, *comm_handle_j, *comm_handle_a;
   /* HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A); */
   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex   *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* HYPRE_Int local_num_rows  = hypre_CSRMatrixNumRows(A_diag); */
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex   *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /* HYPRE_Int       *row_starts      = hypre_ParCSRMatrixRowStarts(A); */
   /* HYPRE_Int        first_row       = hypre_ParCSRMatrixFirstRowIndex(A); */
   HYPRE_Int        first_col        = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt    *col_map_offd_A   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   MPI_Comm         comm  = hypre_ParCSRMatrixComm(A);

   HYPRE_Int        num_procs;
   HYPRE_Int        my_id;
   void           **vrequest;

   hypre_CSRMatrix *A_ext;

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
   d_send_i   = hypre_TAlloc(HYPRE_Int, num_rows_send + 1, HYPRE_MEMORY_DEVICE);
   d_send_map = hypre_TAlloc(HYPRE_Int, num_rows_send,     HYPRE_MEMORY_DEVICE);
   send_i     = hypre_TAlloc(HYPRE_Int, num_rows_send,     HYPRE_MEMORY_HOST);
   recv_i     = hypre_TAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_HOST);
   d_recv_i   = hypre_TAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_DEVICE);

   /* fill the send array with row lengths */
   hypre_TMemcpy(d_send_map, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), HYPRE_Int,
                 num_rows_send, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   hypre_Memset(d_send_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
   hypreDevice_GetRowNnz(num_rows_send, d_send_map, A_diag_i, A_offd_i, d_send_i+1);

   /* send array send_i out: deviceTohost first and MPI (async)
    * note the shift in recv_i by one */
   hypre_TMemcpy(send_i, d_send_i+1, HYPRE_Int, num_rows_send, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_i, recv_i+1);

   hypreDevice_IntegerInclusiveScan(num_rows_send + 1, d_send_i);

   /* total number of nnz to send */
   hypre_TMemcpy(&num_nnz_send, d_send_i+num_rows_send, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   /* prepare data to send out. overlap with the above commmunication */
   d_send_j = hypre_TAlloc(HYPRE_BigInt, num_nnz_send, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_send_a = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_DEVICE);
   }

   if (d_col_map_offd_A == nullptr)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   /* job == 2, d_send_i is input that contains row ptrs (length num_rows_send) */
   hypreDevice_CopyParCSRRows(num_rows_send, d_send_map, 2, num_procs > 1,
                              first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a,
                              A_offd_i, A_offd_j, A_offd_a,
                              d_send_i, d_send_j, d_send_a);

   /* pointers to each proc in send_j */
   send_jstarts = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
   send_jstarts[0] = 0;
   for (i = 1; i <= num_sends; i++)
   {
      send_jstarts[i] = send_jstarts[i-1];
      for ( j = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i-1);
            j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            j++ )
      {
         send_jstarts[i] += send_i[j];
      }
   }
   hypre_assert(send_jstarts[num_sends] == num_nnz_send);

   /* finish the above communication: send_i/recv_i */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust recv_i to ptrs */
   recv_i[0] = 0;
   for (i = 1; i <= num_rows_recv; i++)
   {
      recv_i[i] += recv_i[i-1];
   }
   num_nnz_recv = recv_i[num_rows_recv];

   /* allocate device memory for j and a */
   d_recv_j = hypre_TAlloc(HYPRE_BigInt, num_nnz_recv, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_recv_a = hypre_TAlloc(HYPRE_Complex, num_nnz_recv, HYPRE_MEMORY_DEVICE);
   }

   recv_jstarts = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   recv_jstarts[0] = 0;
   for (i = 1; i <= num_recvs; i++)
   {
      j = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_jstarts[i] = recv_i[j];
   }

   /* ready to send and recv: create a communication package for data */
   comm_pkg_j = hypre_CTAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm         (comm_pkg_j) = comm;
   hypre_ParCSRCommPkgNumSends     (comm_pkg_j) = num_sends;
   hypre_ParCSRCommPkgSendProcs    (comm_pkg_j) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j) = send_jstarts;
   hypre_ParCSRCommPkgNumRecvs     (comm_pkg_j) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs    (comm_pkg_j) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j) = recv_jstarts;

   /* init communication */
   /* ja */
   comm_handle_j = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   HYPRE_MEMORY_DEVICE, d_send_j,
                                                   HYPRE_MEMORY_DEVICE, d_recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg_j,
                                                      HYPRE_MEMORY_DEVICE, d_send_a,
                                                      HYPRE_MEMORY_DEVICE, d_recv_a);
   }
   else
   {
      comm_handle_a = nullptr;
   }

   hypre_TMemcpy(d_recv_i, recv_i, HYPRE_Int, num_rows_recv+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* create A_ext: on device */
   A_ext = hypre_CSRMatrixCreate(num_rows_recv, hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   hypre_CSRMatrixI   (A_ext) = d_recv_i;
   hypre_CSRMatrixBigJ(A_ext) = d_recv_j;
   hypre_CSRMatrixData(A_ext) = d_recv_a;
   hypre_CSRMatrixMemoryLocation(A_ext) = HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(send_i,     HYPRE_MEMORY_HOST);
   hypre_TFree(recv_i,     HYPRE_MEMORY_HOST);
   hypre_TFree(d_send_i,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_send_map, HYPRE_MEMORY_DEVICE);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ParcsrGetExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *A_ext         = (hypre_CSRMatrix *)        request[2];
   HYPRE_BigInt           *send_j        = comm_handle_j ? (HYPRE_BigInt *)  hypre_ParCSRCommHandleSendData(comm_handle_j) : nullptr;
   HYPRE_Complex          *send_a        = comm_handle_a ? (HYPRE_Complex *) hypre_ParCSRCommHandleSendData(comm_handle_a) : nullptr;

   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(send_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(send_a, HYPRE_MEMORY_DEVICE);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return A_ext;
}

hypre_CSRMatrix*
hypre_MergeDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   MPI_Comm         comm     = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex   *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex   *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        local_num_rows   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     glbal_num_cols   = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt     first_col        = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nrows = local_num_rows;
   HYPRE_BigInt     B_ncols = glbal_num_cols;
   HYPRE_Int       *B_i = hypre_TAlloc(HYPRE_Int, B_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt    *B_j;
   HYPRE_Complex   *B_a;
   HYPRE_Int        B_nnz;

   HYPRE_Int        num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_Memset(B_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(B_nrows, nullptr, A_diag_i, A_offd_i, B_i+1);

   hypreDevice_IntegerInclusiveScan(B_nrows+1, B_i);

   /* total number of nnz */
   hypre_TMemcpy(&B_nnz, B_i+B_nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   B_j = hypre_TAlloc(HYPRE_BigInt,  B_nnz, HYPRE_MEMORY_DEVICE);
   B_a = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   if (d_col_map_offd_A == nullptr)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   hypreDevice_CopyParCSRRows(B_nrows, nullptr, 2, num_procs > 1, first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a, A_offd_i, A_offd_j, A_offd_a,
                              B_i, B_j, B_a);

   /* output */
   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI   (B) = B_i;
   hypre_CSRMatrixBigJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   hypre_SyncSyclComputeStream(hypre_handle());

   return B;
}

HYPRE_Int
hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix      *B_ext,
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

   HYPRE_Int     *B_ext_i_d      = hypre_CSRMatrixI(B_ext);
   HYPRE_BigInt  *B_ext_j_d      = hypre_CSRMatrixBigJ(B_ext);
   HYPRE_Complex *B_ext_a_d      = hypre_CSRMatrixData(B_ext);
   HYPRE_Int      B_ext_ncols    = hypre_CSRMatrixNumCols(B_ext);
   HYPRE_Int      B_ext_nrows    = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int      B_ext_nnz      = hypre_CSRMatrixNumNonzeros(B_ext);
   HYPRE_Int     *B_ext_rownnz_d = hypre_TAlloc(HYPRE_Int, B_ext_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_rownnz_h = hypre_TAlloc(HYPRE_Int, B_ext_nrows,     HYPRE_MEMORY_HOST);
   HYPRE_Int     *B_ext_i_h      = hypre_TAlloc(HYPRE_Int, B_ext_nrows + 1, HYPRE_MEMORY_HOST);

   hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   hypre_CSRMatrix *B_int_d;
   HYPRE_Int        B_int_nrows = num_elmts_send;
   HYPRE_Int        B_int_ncols = B_ext_ncols;
   HYPRE_Int       *B_int_i_h   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int       *B_int_i_d   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt    *B_int_j_d   = nullptr;
   HYPRE_Complex   *B_int_a_d   = nullptr;
   HYPRE_Int        B_int_nnz;

   hypre_ParCSRCommHandle *comm_handle, *comm_handle_j, *comm_handle_a;
   hypre_ParCSRCommPkg    *comm_pkg_j;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int i;
   HYPRE_Int num_procs, my_id;
   void    **vrequest;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   jdata_send_map_starts = hypre_TAlloc(HYPRE_Int, num_sends+1, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * B_ext_rownnz contains the number of elements of row j
    * (to be determined through send_map_elmnts on the receiving end)
    *--------------------------------------------------------------------------*/
   HYPRE_ONEDPL_CALL(std::adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
   hypre_TMemcpy(B_ext_rownnz_h, B_ext_rownnz_d + 1, HYPRE_Int, B_ext_nrows,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   /*--------------------------------------------------------------------------
    * initialize communication: send/recv the row nnz
    * (note the use of comm_pkg_A, mode 12, as in transpose matvec
    *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_A, B_ext_rownnz_h, B_int_i_h + 1);

   jdata_recv_vec_starts = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts[0] = 0;

   B_ext_i_h[0] = 0;
   hypre_TMemcpy(B_ext_i_h + 1, B_ext_rownnz_h, HYPRE_Int, B_ext_nrows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   for (i = 1; i <= B_ext_nrows; i++)
   {
      B_ext_i_h[i] += B_ext_i_h[i-1];
   }

   hypre_assert(B_ext_i_h[B_ext_nrows] == B_ext_nnz);

   for (i = 1; i <= num_recvs; i++)
   {
      jdata_recv_vec_starts[i] = B_ext_i_h[recv_vec_starts[i]];
   }

   comm_pkg_j = hypre_CTAlloc(hypre_ParCSRCommPkg,  1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm(comm_pkg_j)      = comm;
   hypre_ParCSRCommPkgNumSends(comm_pkg_j)  = num_recvs;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg_j)  = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg_j) = recv_procs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg_j) = send_procs;

   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*--------------------------------------------------------------------------
    * compute B_int: row nnz to row ptrs
    *--------------------------------------------------------------------------*/
   B_int_i_h[0] = 0;
   for (i = 1; i <= B_int_nrows; i++)
   {
      B_int_i_h[i] += B_int_i_h[i-1];
   }

   B_int_nnz = B_int_i_h[B_int_nrows];

   B_int_j_d = hypre_TAlloc(HYPRE_BigInt,  B_int_nnz, HYPRE_MEMORY_DEVICE);
   B_int_a_d = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_DEVICE);

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* note the order of send/recv is reversed */
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j) = jdata_send_map_starts;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j) = jdata_recv_vec_starts;

   /* send/recv CSR rows */
   comm_handle_a = hypre_ParCSRCommHandleCreate_v2( 1, comm_pkg_j,
                                                    HYPRE_MEMORY_DEVICE, B_ext_a_d,
                                                    HYPRE_MEMORY_DEVICE, B_int_a_d );
   comm_handle_j = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg_j,
                                                   HYPRE_MEMORY_DEVICE, B_ext_j_d,
                                                   HYPRE_MEMORY_DEVICE, B_int_j_d );

   hypre_TMemcpy(B_int_i_d, B_int_i_h, HYPRE_Int, B_int_nrows+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* create CSR: on device */
   B_int_d = hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   hypre_CSRMatrixI(B_int_d)    = B_int_i_d;
   hypre_CSRMatrixBigJ(B_int_d) = B_int_j_d;
   hypre_CSRMatrixData(B_int_d) = B_int_a_d;
   hypre_CSRMatrixMemoryLocation(B_int_d) = HYPRE_MEMORY_DEVICE;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int_d;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(B_ext_rownnz_d, HYPRE_MEMORY_DEVICE);
   hypre_TFree(B_ext_rownnz_h, HYPRE_MEMORY_HOST);
   hypre_TFree(B_ext_i_h,      HYPRE_MEMORY_HOST);
   hypre_TFree(B_int_i_h,      HYPRE_MEMORY_HOST);

   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_j, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ExchangeExternalRowsDeviceWait(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *B_int_d       = (hypre_CSRMatrix *)        request[2];

   /* communication done */
   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return B_int_d;
}


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

HYPRE_Int
hypre_ParCSRMatrixExtractBExtDeviceInit( hypre_ParCSRMatrix  *B,
                                         hypre_ParCSRMatrix  *A,
                                         HYPRE_Int            want_data,
                                         void               **request_ptr)
{
   hypre_assert( hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B)) ==
                 hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(B)) );

   /*
   hypre_assert( hypre_GetActualMemLocation(
            hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B))) == HYPRE_MEMORY_DEVICE );
   */

   hypre_ParcsrGetExternalRowsDeviceInit(B,
                                         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)),
                                         hypre_ParCSRMatrixColMapOffd(A),
                                         hypre_ParCSRMatrixCommPkg(A),
                                         want_data,
                                         request_ptr);
   return hypre_error_flag;
}

hypre_CSRMatrix*
hypre_ParCSRMatrixExtractBExtDeviceWait(void *request)
{
   return hypre_ParcsrGetExternalRowsDeviceWait(request);
}

hypre_CSRMatrix*
hypre_ParCSRMatrixExtractBExtDevice( hypre_ParCSRMatrix *B,
                                     hypre_ParCSRMatrix *A,
                                     HYPRE_Int want_data )
{
   void *request;
   hypre_ParCSRMatrixExtractBExtDeviceInit(B, A, want_data, &request);
   return hypre_ParCSRMatrixExtractBExtDeviceWait(request);
}

/* return B = [Adiag, Aoffd] */
#if 1
void
hypreSYCLKernel_ConcatDiagAndOffd(sycl::nd_item<1>& item,
                                  HYPRE_Int  nrows,    HYPRE_Int  diag_ncol,
                                  HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                                  HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                                  HYPRE_Int *cols_offd_map,
                                  HYPRE_Int *d_ib,     HYPRE_Int *d_jb,     HYPRE_Complex *d_ab)
{
   sycl::group<1> grp = item.get_group();
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);
   const HYPRE_Int row = hypre_sycl_get_global_subgroup_id<1>(item);

   if (row >= nrows)
   {
      return;
   }

   /* lane id inside the subgroup */
   const HYPRE_Int lane_id = SG.get_local_linear_id();
   HYPRE_Int i, j, k, p, istart, iend, bstart;

   /* diag part */
   if (lane_id < 2)
   {
      j = read_only_load(d_diag_i + row + lane_id);
   }
   if (lane_id == 0)
   {
      k = read_only_load(d_ib + row);
   }
   istart = SG.shuffle(j, 0);
   iend   = SG.shuffle(j, 1);
   bstart = SG.shuffle(k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += sub_group_size)
   {
      d_jb[p+i] = read_only_load(d_diag_j + i);
      d_ab[p+i] = read_only_load(d_diag_a + i);
   }

   /* offd part */
   if (lane_id < 2)
   {
      j = read_only_load(d_offd_i + row + lane_id);
   }
   bstart += iend - istart;
   istart = SG.shuffle(j, 0);
   iend   = SG.shuffle(j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += sub_group_size)
   {
      const HYPRE_Int t = read_only_load(d_offd_j + i);
      d_jb[p+i] = (cols_offd_map ? read_only_load(&cols_offd_map[t]) : t) + diag_ncol;
      d_ab[p+i] = read_only_load(d_offd_a + i);
   }
}

hypre_CSRMatrix*
hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrix *B = hypre_CSRMatrixCreate( hypre_CSRMatrixNumRows(A_diag),
                                               hypre_CSRMatrixNumCols(A_diag) + hypre_CSRMatrixNumCols(A_offd),
                                               hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd) );

   hypre_CSRMatrixInitialize_v2(B, 0, HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(hypre_CSRMatrixNumRows(B), nullptr, hypre_CSRMatrixI(A_diag), hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));

   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      hypre_CSRMatrixI(B),
                      hypre_CSRMatrixI(B) + hypre_CSRMatrixNumRows(B) + 1,
                      hypre_CSRMatrixI(B) );

   const sycl::range<1> bDim = hypre_GetDefaultSYCLBlockDimension();
   const sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_ConcatDiagAndOffd,
                         gDim, bDim,
                         hypre_CSRMatrixNumRows(A_diag),
                         hypre_CSRMatrixNumCols(A_diag),
                         hypre_CSRMatrixI(A_diag),
                         hypre_CSRMatrixJ(A_diag),
                         hypre_CSRMatrixData(A_diag),
                         hypre_CSRMatrixI(A_offd),
                         hypre_CSRMatrixJ(A_offd),
                         hypre_CSRMatrixData(A_offd),
                         nullptr,
                         hypre_CSRMatrixI(B),
                         hypre_CSRMatrixJ(B),
                         hypre_CSRMatrixData(B) );

   return B;
}
#else
hypre_CSRMatrix*
hypre_ConcatDiagAndOffdDevice(hypre_ParCSRMatrix *A)
{
   hypre_CSRMatrix *A_diag     = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int       *A_diag_i   = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j   = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex   *A_diag_a   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrix *A_offd     = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i   = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j   = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex   *A_offd_a   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        B_ncols = hypre_CSRMatrixNumCols(A_diag) + hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz;
   HYPRE_Int       *B_ii = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_j  = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex   *B_a  = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   // Adiag
   HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_diag_nnz, A_diag_i);
   HYPRE_ONEDPL_CALL( std::copy_n,
                      oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                      A_diag_nnz,
                      oneapi::dpl::make_zip_iterator(B_ii, B_j, B_a) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_offd_nnz, A_offd_i);
   HYPRE_ONEDPL_CALL( std::copy_n,
                      oneapi::dpl::make_zip_iterator(A_offd_ii, A_offd_a),
                      A_offd_nnz,
                      oneapi::dpl::make_zip_iterator(B_ii, B_a) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_ONEDPL_CALL( std::transform,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      dpct::make_constant_iterator(hypre_CSRMatrixNumCols(A_diag)),
                      B_j + A_diag_nnz,
                      std::plus<HYPRE_Int>() );

   // B
   auto values_begin = oneapi::dpl::make_zip_iterator(B_j, B_a);
   auto zipped_begin = oneapi::dpl::make_zip_iterator(B_ii, values_begin);
   HYPRE_ONEDPL_CALL( std::stable_sort, //stable_sort_by_key
                      zipped_begin, zipped_begin + B_nnz );

   HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(B_nrows, B_nnz, B_ii);
   hypre_TFree(B_ii, HYPRE_MEMORY_DEVICE);

   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI(B) = B_i;
   hypre_CSRMatrixJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   return B;
}
#endif

/* return B = [Adiag, Aoffd; E] */
#if 1
HYPRE_Int
hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A,
                                 hypre_CSRMatrix    *E,
                                 hypre_CSRMatrix   **B_ptr,
                                 HYPRE_Int          *num_cols_offd_ptr,
                                 HYPRE_BigInt      **cols_map_offd_ptr)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *E_diag, *E_offd, *B;
   HYPRE_Int       *cols_offd_map, num_cols_offd;
   HYPRE_BigInt    *cols_map_offd;

   hypre_CSRMatrixSplitDevice(E, hypre_ParCSRMatrixFirstColDiag(A), hypre_ParCSRMatrixLastColDiag(A),
                              hypre_CSRMatrixNumCols(A_offd), hypre_ParCSRMatrixDeviceColMapOffd(A),
                              &cols_offd_map, &num_cols_offd, &cols_map_offd, &E_diag, &E_offd);

   B = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E),
                             hypre_ParCSRMatrixNumCols(A) + num_cols_offd,
                             hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd) + hypre_CSRMatrixNumNonzeros(E));
   hypre_CSRMatrixInitialize_v2(B, 0, HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(hypre_ParCSRMatrixNumRows(A), nullptr,
                         hypre_CSRMatrixI(A_diag), hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));
   HYPRE_ONEDPL_CALL( std::exclusive_scan,
                      hypre_CSRMatrixI(B),
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) );

   sycl::range<1> bDim = hypre_GetDefaultSYCLWorkgroupDimension();
   sycl::range<1> gDim = hypre_GetDefaultSYCLGridDimension(hypre_ParCSRMatrixNumRows(A), "warp", bDim);

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_ConcatDiagAndOffd,
                         gDim, bDim,
                         hypre_CSRMatrixNumRows(A_diag),
                         hypre_CSRMatrixNumCols(A_diag),
                         hypre_CSRMatrixI(A_diag),
                         hypre_CSRMatrixJ(A_diag),
                         hypre_CSRMatrixData(A_diag),
                         hypre_CSRMatrixI(A_offd),
                         hypre_CSRMatrixJ(A_offd),
                         hypre_CSRMatrixData(A_offd),
                         cols_offd_map,
                         hypre_CSRMatrixI(B),
                         hypre_CSRMatrixJ(B),
                         hypre_CSRMatrixData(B) );

   hypre_TFree(cols_offd_map, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1, hypre_CSRMatrixI(E) + 1, HYPRE_Int, hypre_CSRMatrixNumRows(E),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEDPL_CALL( std::transform,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E) + 1,
                      dpct::make_constant_iterator(hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd)),
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      std::plus<HYPRE_Int>() );

   gDim = hypre_GetDefaultSYCLGridDimension(hypre_CSRMatrixNumRows(E), "warp", bDim);

   hypre_assert(hypre_CSRMatrixNumCols(E_diag) == hypre_CSRMatrixNumCols(A_diag));

   HYPRE_SYCL_1D_LAUNCH( hypreSYCLKernel_ConcatDiagAndOffd,
                         gDim, bDim,
                         hypre_CSRMatrixNumRows(E_diag),
                         hypre_CSRMatrixNumCols(E_diag),
                         hypre_CSRMatrixI(E_diag),
                         hypre_CSRMatrixJ(E_diag),
                         hypre_CSRMatrixData(E_diag),
                         hypre_CSRMatrixI(E_offd),
                         hypre_CSRMatrixJ(E_offd),
                         hypre_CSRMatrixData(E_offd),
                         nullptr,
                         hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A),
                         hypre_CSRMatrixJ(B),
                         hypre_CSRMatrixData(B) );

   hypre_CSRMatrixDestroy(E_diag);
   hypre_CSRMatrixDestroy(E_offd);

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return hypre_error_flag;
}
#else
HYPRE_Int
hypre_ConcatDiagOffdAndExtDevice(hypre_ParCSRMatrix *A,
                                 hypre_CSRMatrix    *E,
                                 hypre_CSRMatrix   **B_ptr,
                                 HYPRE_Int          *num_cols_offd_ptr,
                                 HYPRE_BigInt      **cols_map_offd_ptr)
{
   hypre_CSRMatrix *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int        A_nrows         = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        A_ncols         = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int       *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex   *A_diag_a        = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        A_diag_nnz      = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrix *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Complex   *A_offd_a        = hypre_CSRMatrixData(A_offd);
   HYPRE_Int        A_offd_nnz      = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_BigInt     first_col_A     = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt     last_col_A      = hypre_ParCSRMatrixLastColDiag(A);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixDeviceColMapOffd(A);

   HYPRE_Int       *E_i     = hypre_CSRMatrixI(E);
   HYPRE_BigInt    *E_bigj  = hypre_CSRMatrixBigJ(E);
   HYPRE_Complex   *E_a     = hypre_CSRMatrixData(E);
   HYPRE_Int        E_nrows = hypre_CSRMatrixNumRows(E);
   HYPRE_Int        E_nnz   = hypre_CSRMatrixNumNonzeros(E);
   HYPRE_Int        E_diag_nnz, E_offd_nnz;

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nnz   = A_diag_nnz + A_offd_nnz + E_nnz;
   HYPRE_Int       *B_ii    = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_j     = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex   *B_a     = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   // E
   hypre_CSRMatrixSplitDevice_core(0, E_nrows, E_nnz, nullptr, E_bigj, nullptr, nullptr, first_col_A, last_col_A, num_cols_offd_A,
                                   nullptr, nullptr, nullptr, nullptr, &E_diag_nnz, nullptr, nullptr, nullptr, nullptr, &E_offd_nnz,
                                   nullptr, nullptr, nullptr, nullptr);

   HYPRE_Int    *cols_offd_map, num_cols_offd;
   HYPRE_BigInt *cols_map_offd;
   HYPRE_Int *E_ii = hypreDevice_CsrRowPtrsToIndices(E_nrows, E_nnz, E_i);

   hypre_CSRMatrixSplitDevice_core(1,
                                   E_nrows, E_nnz, E_ii, E_bigj, E_a, nullptr,
                                   first_col_A, last_col_A, num_cols_offd_A, col_map_offd_A,
                                   &cols_offd_map, &num_cols_offd, &cols_map_offd,
                                   &E_diag_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz,
                                   nullptr,
                                   &E_offd_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   nullptr);
   hypre_TFree(E_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_ONEDPL_CALL( std::transform,
                      B_ii + A_diag_nnz + A_offd_nnz,
                      B_ii + B_nnz,
                      dpct::make_constant_iterator(A_nrows),
                      B_ii + A_diag_nnz + A_offd_nnz,
                      std::plus<HYPRE_Int>() );

   // Adiag
   HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_diag_nnz, A_diag_i);
   HYPRE_ONEDPL_CALL( std::copy_n,
                      oneapi::dpl::make_zip_iterator(A_diag_ii, A_diag_j, A_diag_a),
                      A_diag_nnz,
                      oneapi::dpl::make_zip_iterator(B_ii, B_j, B_a) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_offd_nnz, A_offd_i);
   HYPRE_ONEDPL_CALL( std::copy_n,
                      oneapi::dpl::make_zip_iterator(A_offd_ii, A_offd_a),
                      A_offd_nnz,
                      oneapi::dpl::make_zip_iterator(B_ii, B_a) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_ONEDPL_CALL( gather,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      cols_offd_map,
                      B_j + A_diag_nnz);

   hypre_TFree(cols_offd_map, HYPRE_MEMORY_DEVICE);

   HYPRE_ONEDPL_CALL( std::transform,
                      B_j + A_diag_nnz,
                      B_j + A_diag_nnz + A_offd_nnz,
                      dpct::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz,
                      std::plus<HYPRE_Int>() );

   HYPRE_ONEDPL_CALL( std::transform,
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      B_j + B_nnz,
                      dpct::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      std::plus<HYPRE_Int>() );

   // B
   auto values_begin = oneapi::dpl::make_zip_iterator(B_j, B_a);
   auto zipped_begin = oneapi::dpl::make_zip_iterator(B_ii, values_begin);
   HYPRE_ONEDPL_CALL( std::stable_sort, //stable_sort_by_key,
                      zipped_begin, zipped_begin + B_nnz );

   HYPRE_Int *B_i = hypreDevice_CsrRowIndicesToPtrs(A_nrows + E_nrows, B_nnz, B_ii);
   hypre_TFree(B_ii, HYPRE_MEMORY_DEVICE);

   B = hypre_CSRMatrixCreate(A_nrows + E_nrows, A_ncols + num_cols_offd, B_nnz);
   hypre_CSRMatrixI(B) = B_i;
   hypre_CSRMatrixJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   *B_ptr = B;
   *num_cols_offd_ptr = num_cols_offd;
   *cols_map_offd_ptr = cols_map_offd;

   return hypre_error_flag;
}
#endif

HYPRE_Int
hypre_ParCSRMatrixGetRowDevice( hypre_ParCSRMatrix  *mat,
                                HYPRE_BigInt         row,
                                HYPRE_Int           *size,
                                HYPRE_BigInt       **col_ind,
                                HYPRE_Complex      **values )
{
   HYPRE_Int nrows, local_row;
   HYPRE_BigInt row_start, row_end;
   hypre_CSRMatrix *Aa;
   hypre_CSRMatrix *Ba;

   if (!mat)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   Aa = (hypre_CSRMatrix *) hypre_ParCSRMatrixDiag(mat);
   Ba = (hypre_CSRMatrix *) hypre_ParCSRMatrixOffd(mat);

   if (hypre_ParCSRMatrixGetrowactive(mat))
   {
      return(-1);
   }

   hypre_ParCSRMatrixGetrowactive(mat) = 1;

   row_start = hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = hypre_ParCSRMatrixLastRowIndex(mat) + 1;
   nrows = row_end - row_start;

   if (row < row_start || row >= row_end)
   {
      return(-1);
   }

   local_row = row - row_start;

   /* if buffer is not allocated and some information is requested, allocate buffer with the max row_nnz */
   if ( !hypre_ParCSRMatrixRowvalues(mat) && (col_ind || values) )
   {
      HYPRE_Int max_row_nnz;
      HYPRE_Int *row_nnz = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);

      hypreDevice_GetRowNnz(nrows, nullptr, hypre_CSRMatrixI(Aa), hypre_CSRMatrixI(Ba), row_nnz);

      hypre_TMemcpy(size, row_nnz + local_row, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      max_row_nnz = HYPRE_ONEDPL_CALL(std::reduce, row_nnz, row_nnz + nrows, 0, oneapi::dpl::maximum<HYPRE_Int>());

/*
      HYPRE_Int *max_row_nnz_d = HYPRE_ONEDPL_CALL(max_element, row_nnz, row_nnz + nrows);
      hypre_TMemcpy( &max_row_nnz, max_row_nnz_d,
                     HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE );
*/

      hypre_TFree(row_nnz, HYPRE_MEMORY_DEVICE);

      hypre_ParCSRMatrixRowvalues(mat)  =
         (HYPRE_Complex *) hypre_TAlloc(HYPRE_Complex, max_row_nnz, hypre_ParCSRMatrixMemoryLocation(mat));
      hypre_ParCSRMatrixRowindices(mat) =
         (HYPRE_BigInt *)  hypre_TAlloc(HYPRE_BigInt,  max_row_nnz, hypre_ParCSRMatrixMemoryLocation(mat));
   }
   else
   {
      HYPRE_Int *size_d = hypre_TAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);
      hypreDevice_GetRowNnz(1, nullptr, hypre_CSRMatrixI(Aa) + local_row, hypre_CSRMatrixI(Ba) + local_row, size_d);
      hypre_TMemcpy(size, size_d, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TFree(size_d, HYPRE_MEMORY_DEVICE);
   }

   if (col_ind || values)
   {
      if (hypre_ParCSRMatrixDeviceColMapOffd(mat) == nullptr)
      {
         hypre_ParCSRMatrixDeviceColMapOffd(mat) =
            hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumCols(Ba), HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy( hypre_ParCSRMatrixDeviceColMapOffd(mat),
                        hypre_ParCSRMatrixColMapOffd(mat),
                        HYPRE_BigInt,
                        hypre_CSRMatrixNumCols(Ba),
                        HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
      }

      hypreDevice_CopyParCSRRows( 1, nullptr, -1, Ba != nullptr,
                                  hypre_ParCSRMatrixFirstColDiag(mat),
                                  hypre_ParCSRMatrixDeviceColMapOffd(mat),
                                  hypre_CSRMatrixI(Aa) + local_row,
                                  hypre_CSRMatrixJ(Aa),
                                  hypre_CSRMatrixData(Aa),
                                  hypre_CSRMatrixI(Ba) + local_row,
                                  hypre_CSRMatrixJ(Ba),
                                  hypre_CSRMatrixData(Ba),
                                  nullptr,
                                  hypre_ParCSRMatrixRowindices(mat),
                                  hypre_ParCSRMatrixRowvalues(mat) );
   }

   if (col_ind)
   {
      *col_ind = hypre_ParCSRMatrixRowindices(mat);
   }

   if (values)
   {
      *values = hypre_ParCSRMatrixRowvalues(mat);
   }

   hypre_SyncSyclComputeStream(hypre_handle());

   return hypre_error_flag;
}

/* abs    == 1, use absolute values
 * option == 0, drop all the entries that are smaller than tol
 * TODO more options
 */
HYPRE_Int
hypre_ParCSRMatrixDropSmallEntriesDevice( hypre_ParCSRMatrix *A,
                                          HYPRE_Complex       tol,
                                          HYPRE_Int           abs,
                                          HYPRE_Int           option)
{
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *h_col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   if (col_map_offd_A == nullptr)
   {
      col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_A, h_col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   hypre_CSRMatrixDropSmallEntriesDevice(A_diag, tol, abs, option);
   hypre_CSRMatrixDropSmallEntriesDevice(A_offd, tol, abs, option);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(A) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A);

   /* squeeze out zero columns of A_offd */
   HYPRE_Int *tmp_j, *tmp_end, num_cols_A_offd_new;
   tmp_j = hypre_TAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(A_offd), HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(tmp_j, hypre_CSRMatrixJ(A_offd), HYPRE_Int, hypre_CSRMatrixNumNonzeros(A_offd),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEDPL_CALL( std::sort,
                      tmp_j,
                      tmp_j + hypre_CSRMatrixNumNonzeros(A_offd) );
   tmp_end = HYPRE_ONEDPL_CALL( std::unique,
                                tmp_j,
                                tmp_j + hypre_CSRMatrixNumNonzeros(A_offd) );
   num_cols_A_offd_new = tmp_end - tmp_j;

   hypre_assert(num_cols_A_offd_new <= num_cols_A_offd);

   if (num_cols_A_offd_new < num_cols_A_offd)
   {
      hypre_CSRMatrixNumCols(A_offd) = num_cols_A_offd_new;

      HYPRE_Int *offd_mark = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *col_map_offd_A_new = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd_new, HYPRE_MEMORY_DEVICE);

      HYPRE_ONEDPL_CALL( scatter,
                         oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                         oneapi::dpl::counting_iterator<HYPRE_Int>(num_cols_A_offd_new),
                         tmp_j,
                         offd_mark );
      HYPRE_ONEDPL_CALL( gather,
                         hypre_CSRMatrixJ(A_offd),
                         hypre_CSRMatrixJ(A_offd) + hypre_CSRMatrixNumNonzeros(A_offd),
                         offd_mark,
                         hypre_CSRMatrixJ(A_offd) );
      HYPRE_ONEDPL_CALL( gather,
                         tmp_j,
                         tmp_j + num_cols_A_offd_new,
                         col_map_offd_A,
                         col_map_offd_A_new );

      hypre_TFree(offd_mark, HYPRE_MEMORY_DEVICE);
      hypre_TFree(col_map_offd_A, HYPRE_MEMORY_DEVICE);
      hypre_TFree(h_col_map_offd_A, HYPRE_MEMORY_HOST);

      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A_new;
      hypre_ParCSRMatrixColMapOffd(A) = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd_new, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(A), col_map_offd_A_new, HYPRE_BigInt, num_cols_A_offd_new,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_SYCL)

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRDiagScale( HYPRE_ParCSRMatrix HA,
                       HYPRE_ParVector    Hy,
                       HYPRE_ParVector    Hx )
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;
   hypre_ParVector    *y = (hypre_ParVector *) Hy;
   hypre_ParVector    *x = (hypre_ParVector *) Hx;
   HYPRE_Real *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Real *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y));
   HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int local_size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   HYPRE_Int ierr = 0;
#if defined(HYPRE_USING_SYCL)
   hypreDevice_DiagScaleVector(local_size, A_i, A_data, y_data, 0.0, x_data);
   //hypre_SyncSyclComputeStream(hypre_handle());
#else /* #if defined(HYPRE_USING_SYCL) */
   HYPRE_Int i;
#if defined(HYPRE_USING_DEVICE_OPENMP)
#pragma omp target teams distribute parallel for private(i) is_device_ptr(x_data,y_data,A_data,A_i)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < local_size; i++)
   {
      x_data[i] = y_data[i] / A_data[A_i[i]];
   }
#endif /* #if defined(HYPRE_USING_SYCL) */

   return ierr;
}