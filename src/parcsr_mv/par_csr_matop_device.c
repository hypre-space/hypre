/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

HYPRE_Int
hypre_ParcsrGetExternalRowsDeviceInit( hypre_ParCSRMatrix   *A,
                                       HYPRE_Int             indices_len,
                                       HYPRE_BigInt         *indices,
                                       hypre_ParCSRCommPkg  *comm_pkg,
                                       HYPRE_Int             want_data,
                                       void                **request_ptr)
{
   HYPRE_Int      i, j;
   HYPRE_Int      num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   HYPRE_Int     *d_send_i, *send_i, *d_send_map, *d_recv_i, *recv_i;
   HYPRE_BigInt  *d_send_j, *d_recv_j;
   HYPRE_Int     *send_jstarts, *recv_jstarts;
   HYPRE_Complex *d_send_a = NULL, *d_recv_a = NULL;
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

   if (d_col_map_offd_A == NULL)
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
      comm_handle_a = NULL;
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
   HYPRE_BigInt           *send_j        = comm_handle_j ? (HYPRE_BigInt *)  hypre_ParCSRCommHandleSendData(comm_handle_j) : NULL;
   HYPRE_Complex          *send_a        = comm_handle_a ? (HYPRE_Complex *) hypre_ParCSRCommHandleSendData(comm_handle_a) : NULL;

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

   hypreDevice_GetRowNnz(B_nrows, NULL, A_diag_i, A_offd_i, B_i+1);

   hypreDevice_IntegerInclusiveScan(B_nrows+1, B_i);

   /* total number of nnz */
   hypre_TMemcpy(&B_nnz, B_i+B_nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   B_j = hypre_TAlloc(HYPRE_BigInt,  B_nnz, HYPRE_MEMORY_DEVICE);
   B_a = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   hypreDevice_CopyParCSRRows(B_nrows, NULL, 2, num_procs > 1, first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a, A_offd_i, A_offd_j, A_offd_a,
                              B_i, B_j, B_a);

   /* output */
   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI   (B) = B_i;
   hypre_CSRMatrixBigJ(B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   hypre_SyncCudaComputeStream(hypre_handle());

   return B;
}

HYPRE_Int
hypre_ExchangeExternalRowsDeviceInit( hypre_CSRMatrix      *B_ext,
                                      hypre_ParCSRCommPkg  *comm_pkg_A,
                                      HYPRE_Int             want_data,
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
   HYPRE_BigInt    *B_int_j_d   = NULL;
   HYPRE_Complex   *B_int_a_d   = NULL;
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
   HYPRE_THRUST_CALL(adjacent_difference, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
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
   if (want_data)
   {
      B_int_a_d = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_DEVICE);
   }

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* note the order of send/recv is reversed */
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j) = jdata_send_map_starts;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j) = jdata_recv_vec_starts;

   /* send/recv CSR rows */
   if (want_data)
   {
      comm_handle_a = hypre_ParCSRCommHandleCreate_v2( 1, comm_pkg_j,
                                                       HYPRE_MEMORY_DEVICE, B_ext_a_d,
                                                       HYPRE_MEMORY_DEVICE, B_int_a_d );
   }
   else
   {
      comm_handle_a = NULL;
   }

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

   if (!hypre_ParCSRMatrixCommPkg(A))
   {
      hypre_MatvecCommPkgCreate(A);
   }

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
__global__ void
hypreCUDAKernel_ConcatDiagAndOffd(HYPRE_Int  nrows,    HYPRE_Int  diag_ncol,
                                  HYPRE_Int *d_diag_i, HYPRE_Int *d_diag_j, HYPRE_Complex *d_diag_a,
                                  HYPRE_Int *d_offd_i, HYPRE_Int *d_offd_j, HYPRE_Complex *d_offd_a,
                                  HYPRE_Int *cols_offd_map,
                                  HYPRE_Int *d_ib,     HYPRE_Int *d_jb,     HYPRE_Complex *d_ab)
{
   const HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   /* lane id inside the warp */
   const HYPRE_Int lane_id = hypre_cuda_get_lane_id<1>();
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
   istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
   bstart = __shfl_sync(HYPRE_WARP_FULL_MASK, k, 0);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
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
   istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

   p = bstart - istart;
   for (i = istart + lane_id; i < iend; i += HYPRE_WARP_SIZE)
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

   hypreDevice_GetRowNnz(hypre_CSRMatrixNumRows(B), NULL, hypre_CSRMatrixI(A_diag), hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));

   HYPRE_THRUST_CALL( exclusive_scan,
                      hypre_CSRMatrixI(B),
                      hypre_CSRMatrixI(B) + hypre_CSRMatrixNumRows(B) + 1,
                      hypre_CSRMatrixI(B) );

   const dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   const dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ConcatDiagAndOffd,
                      gDim, bDim,
                      hypre_CSRMatrixNumRows(A_diag),
                      hypre_CSRMatrixNumCols(A_diag),
                      hypre_CSRMatrixI(A_diag),
                      hypre_CSRMatrixJ(A_diag),
                      hypre_CSRMatrixData(A_diag),
                      hypre_CSRMatrixI(A_offd),
                      hypre_CSRMatrixJ(A_offd),
                      hypre_CSRMatrixData(A_offd),
                      NULL,
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
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(B_nrows, A_offd_nnz, A_offd_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumCols(A_diag)),
                      B_j + A_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   // B
   HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

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

   hypreDevice_GetRowNnz(hypre_ParCSRMatrixNumRows(A), NULL, hypre_CSRMatrixI(A_diag), hypre_CSRMatrixI(A_offd), hypre_CSRMatrixI(B));
   HYPRE_THRUST_CALL( exclusive_scan,
                      hypre_CSRMatrixI(B),
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) );

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_ParCSRMatrixNumRows(A), "warp", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ConcatDiagAndOffd,
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
   HYPRE_THRUST_CALL( transform,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(E) + 1,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumNonzeros(A_diag) + hypre_CSRMatrixNumNonzeros(A_offd)),
                      hypre_CSRMatrixI(B) + hypre_ParCSRMatrixNumRows(A) + 1,
                      thrust::plus<HYPRE_Int>() );

   gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(E), "warp", bDim);

   hypre_assert(hypre_CSRMatrixNumCols(E_diag) == hypre_CSRMatrixNumCols(A_diag));

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_ConcatDiagAndOffd,
                      gDim, bDim,
                      hypre_CSRMatrixNumRows(E_diag),
                      hypre_CSRMatrixNumCols(E_diag),
                      hypre_CSRMatrixI(E_diag),
                      hypre_CSRMatrixJ(E_diag),
                      hypre_CSRMatrixData(E_diag),
                      hypre_CSRMatrixI(E_offd),
                      hypre_CSRMatrixJ(E_offd),
                      hypre_CSRMatrixData(E_offd),
                      NULL,
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
   hypre_CSRMatrixSplitDevice_core(0, E_nrows, E_nnz, NULL, E_bigj, NULL, NULL, first_col_A, last_col_A, num_cols_offd_A,
                                   NULL, NULL, NULL, NULL, &E_diag_nnz, NULL, NULL, NULL, NULL, &E_offd_nnz,
                                   NULL, NULL, NULL, NULL);

   HYPRE_Int    *cols_offd_map, num_cols_offd;
   HYPRE_BigInt *cols_map_offd;
   HYPRE_Int *E_ii = hypreDevice_CsrRowPtrsToIndices(E_nrows, E_nnz, E_i);

   hypre_CSRMatrixSplitDevice_core(1,
                                   E_nrows, E_nnz, E_ii, E_bigj, E_a, NULL,
                                   first_col_A, last_col_A, num_cols_offd_A, col_map_offd_A,
                                   &cols_offd_map, &num_cols_offd, &cols_map_offd,
                                   &E_diag_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz,
                                   NULL,
                                   &E_offd_nnz,
                                   B_ii + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_j  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   B_a  + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                                   NULL);
   hypre_TFree(E_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      B_ii + A_diag_nnz + A_offd_nnz,
                      B_ii + B_nnz,
                      thrust::make_constant_iterator(A_nrows),
                      B_ii + A_diag_nnz + A_offd_nnz,
                      thrust::plus<HYPRE_Int>() );

   // Adiag
   HYPRE_Int *A_diag_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_diag_nnz, A_diag_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                      A_diag_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_j, B_a)) );
   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   // Aoffd
   HYPRE_Int *A_offd_ii = hypreDevice_CsrRowPtrsToIndices(A_nrows, A_offd_nnz, A_offd_i);
   HYPRE_THRUST_CALL( copy_n,
                      thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, A_offd_a)),
                      A_offd_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_ii, B_a)) + A_diag_nnz );
   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( gather,
                      A_offd_j,
                      A_offd_j + A_offd_nnz,
                      cols_offd_map,
                      B_j + A_diag_nnz);

   hypre_TFree(cols_offd_map, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz,
                      B_j + A_diag_nnz + A_offd_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   HYPRE_THRUST_CALL( transform,
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      B_j + B_nnz,
                      thrust::make_constant_iterator(A_ncols),
                      B_j + A_diag_nnz + A_offd_nnz + E_diag_nnz,
                      thrust::plus<HYPRE_Int>() );

   // B
   HYPRE_THRUST_CALL( stable_sort_by_key,
                      B_ii,
                      B_ii + B_nnz,
                      thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)) );

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

      hypreDevice_GetRowNnz(nrows, NULL, hypre_CSRMatrixI(Aa), hypre_CSRMatrixI(Ba), row_nnz);

      hypre_TMemcpy(size, row_nnz + local_row, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      max_row_nnz = HYPRE_THRUST_CALL(reduce, row_nnz, row_nnz + nrows, 0, thrust::maximum<HYPRE_Int>());

/*
      HYPRE_Int *max_row_nnz_d = HYPRE_THRUST_CALL(max_element, row_nnz, row_nnz + nrows);
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
      hypreDevice_GetRowNnz(1, NULL, hypre_CSRMatrixI(Aa) + local_row, hypre_CSRMatrixI(Ba) + local_row, size_d);
      hypre_TMemcpy(size, size_d, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TFree(size_d, HYPRE_MEMORY_DEVICE);
   }

   if (col_ind || values)
   {
      if (hypre_ParCSRMatrixDeviceColMapOffd(mat) == NULL)
      {
         hypre_ParCSRMatrixDeviceColMapOffd(mat) =
            hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumCols(Ba), HYPRE_MEMORY_DEVICE);

         hypre_TMemcpy( hypre_ParCSRMatrixDeviceColMapOffd(mat),
                        hypre_ParCSRMatrixColMapOffd(mat),
                        HYPRE_BigInt,
                        hypre_CSRMatrixNumCols(Ba),
                        HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
      }

      hypreDevice_CopyParCSRRows( 1, NULL, -1, Ba != NULL,
                                  hypre_ParCSRMatrixFirstColDiag(mat),
                                  hypre_ParCSRMatrixDeviceColMapOffd(mat),
                                  hypre_CSRMatrixI(Aa) + local_row,
                                  hypre_CSRMatrixJ(Aa),
                                  hypre_CSRMatrixData(Aa),
                                  hypre_CSRMatrixI(Ba) + local_row,
                                  hypre_CSRMatrixJ(Ba),
                                  hypre_CSRMatrixData(Ba),
                                  NULL,
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

   hypre_SyncCudaComputeStream(hypre_handle());

   return hypre_error_flag;
}

/* Get element-wise tolerances based on row norms for ParCSRMatrix
 * NOTE: Keep the diagonal, i.e. elmt_tol = 0.0 for diagonals
 * Output vectors have size nnz:
 *    elmt_tols_diag[j] = tol * (norm of row i) for j in [ A_diag_i[i] , A_diag_i[i+1] )
 *    elmt_tols_offd[j] = tol * (norm of row i) for j in [ A_offd_i[i] , A_offd_i[i+1] )
 * type == -1, infinity norm,
 *         1, 1-norm
 *         2, 2-norm
 */
template<HYPRE_Int type>
__global__ void
hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols( HYPRE_Int      nrows,
                                                      HYPRE_Real     tol,
                                                      HYPRE_Int     *A_diag_i,
                                                      HYPRE_Int     *A_diag_j,
                                                      HYPRE_Complex *A_diag_a,
                                                      HYPRE_Int     *A_offd_i,
                                                      HYPRE_Complex *A_offd_a,
                                                      HYPRE_Real     *elmt_tols_diag,
                                                      HYPRE_Real     *elmt_tols_offd)
{
   HYPRE_Int row_i = hypre_cuda_get_grid_warp_id<1,1>();

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p_diag, p_offd, q_diag, q_offd;

   /* sum row norm over diag part */
   if (lane < 2)
   {
      p_diag = read_only_load(A_diag_i + row_i + lane);
   }
   q_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = __shfl_sync(HYPRE_WARP_FULL_MASK, p_diag, 0);

   HYPRE_Real row_norm_i = 0.0;

   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex val = A_diag_a[j];

      if (type == -1)
      {
         row_norm_i = hypre_max(row_norm_i, hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* sum row norm over offd part */
   if (lane < 2)
   {
      p_offd = read_only_load(A_offd_i + row_i + lane);
   }
   q_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = __shfl_sync(HYPRE_WARP_FULL_MASK, p_offd, 0);

   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex val = A_offd_a[j];

      if (type == -1)
      {
         row_norm_i = hypre_max(row_norm_i, hypre_cabs(val));
      }
      else if (type == 1)
      {
         row_norm_i += hypre_cabs(val);
      }
      else if (type == 2)
      {
         row_norm_i += val * val;
      }
   }

   /* allreduce to get the row norm on all threads */
   if (type == -1)
   {
      row_norm_i = warp_allreduce_max(row_norm_i);
   }
   else
   {
      row_norm_i = warp_allreduce_sum(row_norm_i);
   }
   if (type == 2)
   {
      row_norm_i = sqrt(row_norm_i);
   }

   /* set elmt_tols_diag */
   for (HYPRE_Int j = p_diag + lane; j < q_diag; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Int col = A_diag_j[j];

      /* elmt_tol = 0.0 ensures diagonal will be kept */
      if (col == row_i)
      {
         elmt_tols_diag[j] = 0.0;
      }
      else
      {
         elmt_tols_diag[j] = tol * row_norm_i;
      }
   }

   /* set elmt_tols_offd */
   for (HYPRE_Int j = p_offd + lane; j < q_offd; j += HYPRE_WARP_SIZE)
   {
      elmt_tols_offd[j] = tol * row_norm_i;
   }

}

/* drop the entries that are not on the diagonal and smaller than:
 *    type 0: tol
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row) */
HYPRE_Int
hypre_ParCSRMatrixDropSmallEntriesDevice( hypre_ParCSRMatrix *A,
                                          HYPRE_Complex       tol,
                                          HYPRE_Int           type)
{
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int        num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *h_col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt    *col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   HYPRE_Real      *elmt_tols_diag = NULL;
   HYPRE_Real      *elmt_tols_offd = NULL;

   if (col_map_offd_A == NULL)
   {
      col_map_offd_A = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_A, h_col_map_offd_A, HYPRE_BigInt, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = col_map_offd_A;
   }

   /* get elmement-wise tolerances if needed */
   if (type != 0)
   {
      elmt_tols_diag = hypre_TAlloc(HYPRE_Real, hypre_CSRMatrixNumNonzeros(A_diag), HYPRE_MEMORY_DEVICE);
      elmt_tols_offd = hypre_TAlloc(HYPRE_Real, hypre_CSRMatrixNumNonzeros(A_offd), HYPRE_MEMORY_DEVICE);
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A_diag), "warp", bDim);

   if (type == -1)
   {
      HYPRE_CUDA_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<-1>, gDim, bDim, 
                         hypre_CSRMatrixNumRows(A_diag), tol, hypre_CSRMatrixI(A_diag), 
                         hypre_CSRMatrixJ(A_diag), hypre_CSRMatrixData(A_diag), hypre_CSRMatrixI(A_offd), 
                         hypre_CSRMatrixData(A_offd), elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 1)
   {
      HYPRE_CUDA_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<1>, gDim, bDim, 
                         hypre_CSRMatrixNumRows(A_diag), tol, hypre_CSRMatrixI(A_diag), 
                         hypre_CSRMatrixJ(A_diag), hypre_CSRMatrixData(A_diag), hypre_CSRMatrixI(A_offd), 
                         hypre_CSRMatrixData(A_offd), elmt_tols_diag, elmt_tols_offd);
   }
   if (type == 2)
   {
      HYPRE_CUDA_LAUNCH( hypre_ParCSRMatrixDropSmallEntriesDevice_getElmtTols<2>, gDim, bDim, 
                         hypre_CSRMatrixNumRows(A_diag), tol, hypre_CSRMatrixI(A_diag), 
                         hypre_CSRMatrixJ(A_diag), hypre_CSRMatrixData(A_diag), hypre_CSRMatrixI(A_offd), 
                         hypre_CSRMatrixData(A_offd), elmt_tols_diag, elmt_tols_offd);
   }

   /* drop entries from diag and offd CSR matrices */
   hypre_CSRMatrixDropSmallEntriesDevice(A_diag, tol, elmt_tols_diag);
   hypre_CSRMatrixDropSmallEntriesDevice(A_offd, tol, elmt_tols_offd);

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(A) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A);

   /* squeeze out zero columns of A_offd */
   HYPRE_Int *tmp_j, *tmp_end, num_cols_A_offd_new;
   tmp_j = hypre_TAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(A_offd), HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(tmp_j, hypre_CSRMatrixJ(A_offd), HYPRE_Int, hypre_CSRMatrixNumNonzeros(A_offd),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( sort,
                      tmp_j,
                      tmp_j + hypre_CSRMatrixNumNonzeros(A_offd) );
   tmp_end = HYPRE_THRUST_CALL( unique,
                                tmp_j,
                                tmp_j + hypre_CSRMatrixNumNonzeros(A_offd) );
   num_cols_A_offd_new = tmp_end - tmp_j;

   hypre_assert(num_cols_A_offd_new <= num_cols_A_offd);

   if (num_cols_A_offd_new < num_cols_A_offd)
   {
      hypre_CSRMatrixNumCols(A_offd) = num_cols_A_offd_new;

      HYPRE_Int *offd_mark = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt *col_map_offd_A_new = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd_new, HYPRE_MEMORY_DEVICE);

      HYPRE_THRUST_CALL( scatter,
                         thrust::counting_iterator<HYPRE_Int>(0),
                         thrust::counting_iterator<HYPRE_Int>(num_cols_A_offd_new),
                         tmp_j,
                         offd_mark );
      HYPRE_THRUST_CALL( gather,
                         hypre_CSRMatrixJ(A_offd),
                         hypre_CSRMatrixJ(A_offd) + hypre_CSRMatrixNumNonzeros(A_offd),
                         offd_mark,
                         hypre_CSRMatrixJ(A_offd) );
      HYPRE_THRUST_CALL( gather,
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

   if (type != 0)
   {
      hypre_TFree(elmt_tols_diag, HYPRE_MEMORY_DEVICE);
      hypre_TFree(elmt_tols_offd, HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTransposeDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTransposeDevice( hypre_ParCSRMatrix  *A,
                                   hypre_ParCSRMatrix **AT_ptr,
                                   HYPRE_Int            data )
{
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix    *A_diagT;
   hypre_CSRMatrix    *AT_offd;
   HYPRE_Int           num_procs;
   HYPRE_Int           num_cols_offd_AT = 0;
   HYPRE_BigInt       *col_map_offd_AT = NULL;
   hypre_ParCSRMatrix *AT;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);

   if (num_procs > 1)
   {
      void *request;
      hypre_CSRMatrix *A_offdT, *Aext;
      HYPRE_Int *Aext_ii, *Aext_j, Aext_nnz;
      HYPRE_Complex *Aext_data;
      HYPRE_BigInt *tmp_bigj;

      hypre_CSRMatrixTranspose(A_offd, &A_offdT, data);
      hypre_CSRMatrixBigJ(A_offdT) = hypre_TAlloc(HYPRE_BigInt, hypre_CSRMatrixNumNonzeros(A_offdT), HYPRE_MEMORY_DEVICE);

      HYPRE_THRUST_CALL( transform,
                         hypre_CSRMatrixJ(A_offdT),
                         hypre_CSRMatrixJ(A_offdT) + hypre_CSRMatrixNumNonzeros(A_offdT),
                         thrust::make_constant_iterator(hypre_ParCSRMatrixFirstRowIndex(A)),
                         hypre_CSRMatrixBigJ(A_offdT),
                         thrust::plus<HYPRE_BigInt>() );

      if (!hypre_ParCSRMatrixCommPkg(A))
      {
         hypre_MatvecCommPkgCreate(A);
      }

      hypre_ExchangeExternalRowsDeviceInit(A_offdT, hypre_ParCSRMatrixCommPkg(A), data, &request);

      hypre_CSRMatrixTranspose(A_diag, &A_diagT, data);

      Aext = hypre_ExchangeExternalRowsDeviceWait(request);

      hypre_CSRMatrixDestroy(A_offdT);

      // Aext contains offd of AT
      Aext_nnz = hypre_CSRMatrixNumNonzeros(Aext);
      Aext_ii = hypreDevice_CsrRowPtrsToIndices(hypre_CSRMatrixNumRows(Aext), Aext_nnz, hypre_CSRMatrixI(Aext));

      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(hypre_ParCSRMatrixCommPkg(A));

      HYPRE_THRUST_CALL( gather,
                         Aext_ii,
                         Aext_ii + Aext_nnz,
                         hypre_ParCSRCommPkgDeviceSendMapElmts(hypre_ParCSRMatrixCommPkg(A)),
                         Aext_ii );

      tmp_bigj = hypre_TAlloc(HYPRE_BigInt, Aext_nnz, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp_bigj, hypre_CSRMatrixBigJ(Aext), HYPRE_BigInt, Aext_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      HYPRE_THRUST_CALL( sort,
                         tmp_bigj,
                         tmp_bigj + Aext_nnz );

      HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique,
                                                 tmp_bigj,
                                                 tmp_bigj + Aext_nnz );

      num_cols_offd_AT = new_end - tmp_bigj;
      col_map_offd_AT = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(col_map_offd_AT, tmp_bigj, HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      hypre_TFree(tmp_bigj, HYPRE_MEMORY_DEVICE);

      Aext_j = hypre_TAlloc(HYPRE_Int, Aext_nnz, HYPRE_MEMORY_DEVICE);

      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_AT,
                         col_map_offd_AT + num_cols_offd_AT,
                         hypre_CSRMatrixBigJ(Aext),
                         hypre_CSRMatrixBigJ(Aext) + Aext_nnz,
                         Aext_j );

      Aext_data = hypre_CSRMatrixData(Aext);
      hypre_CSRMatrixData(Aext) = NULL;
      hypre_CSRMatrixDestroy(Aext);

      if (data)
      {
         hypreDevice_StableSortByTupleKey(Aext_nnz, Aext_ii, Aext_j, Aext_data, 0);
      }
      else
      {
         HYPRE_THRUST_CALL( stable_sort,
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)),
                            thrust::make_zip_iterator(thrust::make_tuple(Aext_ii, Aext_j)) + Aext_nnz );
      }

      AT_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(A), num_cols_offd_AT, Aext_nnz);
      hypre_CSRMatrixJ(AT_offd) = Aext_j;
      hypre_CSRMatrixData(AT_offd) = Aext_data;
      hypre_CSRMatrixInitialize_v2(AT_offd, 0, HYPRE_MEMORY_DEVICE);
      hypreDevice_CsrRowIndicesToPtrs_v2(hypre_CSRMatrixNumRows(AT_offd), Aext_nnz, Aext_ii, hypre_CSRMatrixI(AT_offd));
      hypre_TFree(Aext_ii, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypre_CSRMatrixTransposeDevice(A_diag, &A_diagT, data);
      AT_offd = hypre_CSRMatrixCreate(hypre_ParCSRMatrixNumCols(A), 0, 0);
      hypre_CSRMatrixInitialize_v2(AT_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   AT = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 num_cols_offd_AT,
                                 hypre_CSRMatrixNumNonzeros(A_diagT),
                                 hypre_CSRMatrixNumNonzeros(AT_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(AT));
   hypre_ParCSRMatrixDiag(AT) = A_diagT;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(AT));
   hypre_ParCSRMatrixOffd(AT) = AT_offd;

   if (num_cols_offd_AT)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(AT) = col_map_offd_AT;

      hypre_ParCSRMatrixColMapOffd(AT) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_AT, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AT), col_map_offd_AT, HYPRE_BigInt, num_cols_offd_AT,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   *AT_ptr = AT;

   return hypre_error_flag;
}

HYPRE_Int
hypre_ParCSRMatrixAddDevice( HYPRE_Complex        alpha,
                             hypre_ParCSRMatrix  *A,
                             HYPRE_Complex        beta,
                             hypre_ParCSRMatrix  *B,
                             hypre_ParCSRMatrix **C_ptr )
{
   hypre_CSRMatrix *A_diag           = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd           = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *B_diag           = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix *B_offd           = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int        num_cols_offd_A  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int        num_cols_offd_B  = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_Int        num_cols_offd_C  = 0;
   HYPRE_BigInt    *d_col_map_offd_C = NULL;
   HYPRE_Int        num_procs;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &num_procs);

   hypre_CSRMatrix *C_diag = hypre_CSRMatrixAddDevice(alpha, A_diag, beta, B_diag);
   hypre_CSRMatrix *C_offd;

   //if (num_cols_offd_A || num_cols_offd_B)
   if (num_procs > 1)
   {
      hypre_ParCSRMatrixCopyColMapOffdToDevice(A);
      hypre_ParCSRMatrixCopyColMapOffdToDevice(B);

      HYPRE_BigInt *tmp = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_A + num_cols_offd_B, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp,                   hypre_ParCSRMatrixDeviceColMapOffd(A), HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(tmp + num_cols_offd_A, hypre_ParCSRMatrixDeviceColMapOffd(B), HYPRE_BigInt, num_cols_offd_B, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sort, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique, tmp, tmp + num_cols_offd_A + num_cols_offd_B );
      num_cols_offd_C = new_end - tmp;
      d_col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_C, tmp, HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      /* reuse memory of tmp */
      HYPRE_Int *offd_A2C = (HYPRE_Int *) tmp;
      HYPRE_Int *offd_B2C = offd_A2C + num_cols_offd_A;
      HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         hypre_ParCSRMatrixDeviceColMapOffd(A),
                         hypre_ParCSRMatrixDeviceColMapOffd(A) + num_cols_offd_A,
                         offd_A2C );
      HYPRE_THRUST_CALL( lower_bound,
                         d_col_map_offd_C,
                         d_col_map_offd_C + num_cols_offd_C,
                         hypre_ParCSRMatrixDeviceColMapOffd(B),
                         hypre_ParCSRMatrixDeviceColMapOffd(B) + num_cols_offd_B,
                         offd_B2C );

      HYPRE_Int *C_offd_i, *C_offd_j, nnzC_offd;
      HYPRE_Complex *C_offd_a;

      hypreDevice_CSRSpAdd( hypre_CSRMatrixNumRows(A_offd),
                            hypre_CSRMatrixNumRows(B_offd),
                            num_cols_offd_C,
                            hypre_CSRMatrixNumNonzeros(A_offd),
                            hypre_CSRMatrixNumNonzeros(B_offd),
                            hypre_CSRMatrixI(A_offd),
                            hypre_CSRMatrixJ(A_offd),
                            alpha,
                            hypre_CSRMatrixData(A_offd),
                            offd_A2C,
                            hypre_CSRMatrixI(B_offd),
                            hypre_CSRMatrixJ(B_offd),
                            beta,
                            hypre_CSRMatrixData(B_offd),
                            offd_B2C,
                            NULL,
                            &nnzC_offd,
                            &C_offd_i,
                            &C_offd_j,
                            &C_offd_a );

      hypre_TFree(tmp, HYPRE_MEMORY_DEVICE);

      C_offd = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(A_offd), num_cols_offd_C, nnzC_offd);
      hypre_CSRMatrixI(C_offd) = C_offd_i;
      hypre_CSRMatrixJ(C_offd) = C_offd_j;
      hypre_CSRMatrixData(C_offd) = C_offd_a;
      hypre_CSRMatrixMemoryLocation(C_offd) = HYPRE_MEMORY_DEVICE;
   }
   else
   {
      C_offd = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(A_offd), 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, HYPRE_MEMORY_DEVICE);
   }

   /* Create ParCSRMatrix C */
   HYPRE_BigInt *row_starts_C = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   HYPRE_BigInt *col_starts_C = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(row_starts_C, hypre_ParCSRMatrixRowStarts(A), HYPRE_BigInt, 2,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(col_starts_C, hypre_ParCSRMatrixColStarts(A), HYPRE_BigInt, 2,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrix *C = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                                    hypre_ParCSRMatrixGlobalNumCols(A),
                                                    row_starts_C,
                                                    col_starts_C,
                                                    num_cols_offd_C,
                                                    hypre_CSRMatrixNumNonzeros(C_diag),
                                                    hypre_CSRMatrixNumNonzeros(C_offd));

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixDeviceColMapOffd(C) = d_col_map_offd_C;

      hypre_ParCSRMatrixColMapOffd(C) = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(C), d_col_map_offd_C, HYPRE_BigInt, num_cols_offd_C,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }

   hypre_ParCSRMatrixSetNumNonzeros(C);
   hypre_ParCSRMatrixDNumNonzeros(C) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(C);

   /* create CommPkg of C */
   hypre_MatvecCommPkgCreate(C);

   *C_ptr = C;

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypreDevice_DiagScaleVector(local_size, A_i, A_data, y_data, 0.0, x_data);
   //hypre_SyncCudaComputeStream(hypre_handle());
#else /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */
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
#endif /* #if defined(HYPRE_USING_CUDA) */

   return ierr;
}
