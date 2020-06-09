/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "hypre_hopscotch_hash.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_lapack.h"
#include "_hypre_blas.h"

#if defined(HYPRE_USING_CUDA)

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

/*---------------------------
 *---------------------------*/
typedef thrust::tuple<HYPRE_Int, HYPRE_Int> Tuple;
//typedef thrust::tuple<HYPRE_Int, HYPRE_Int, HYPRe_Int> Tuple3;

struct FFFC_functor : public thrust::unary_function<Tuple, HYPRE_BigInt>
{
   HYPRE_BigInt CF_first[2];

   FFFC_functor(HYPRE_BigInt F_first_, HYPRE_BigInt C_first_)
   {
      CF_first[1] = F_first_;
      CF_first[0] = C_first_;
   }

   __host__ __device__
   HYPRE_BigInt operator()(const Tuple& t) const
   {
      const HYPRE_Int local_idx = thrust::get<0>(t);
      const HYPRE_Int cf_marker = thrust::get<1>(t);
      const HYPRE_Int s = cf_marker < 0;
      const HYPRE_Int m = 1 - 2*s;
      return m*(local_idx + CF_first[s] + s);
   }
};

template<bool FCOL, typename T>
struct FFFC_pred : public thrust::unary_function<Tuple, bool>
{
   HYPRE_Int *row_CF_marker;
   T         *col_CF_marker;

   FFFC_pred(HYPRE_Int *row_CF_marker_, T *col_CF_marker_)
   {
      row_CF_marker = row_CF_marker_;
      col_CF_marker = col_CF_marker_;
   }

   __host__ __device__
   bool operator()(const Tuple& t) const
   {
      const HYPRE_Int i = thrust::get<0>(t);
      const HYPRE_Int j = thrust::get<1>(t);
      if (FCOL)
      {
         /* AFF */
         return row_CF_marker[i] < 0 && (j == -2 || j >= 0 && col_CF_marker[j] < 0);
      }
      else
      {
         /* AFC */
         return row_CF_marker[i] < 0 && (j >= 0 && col_CF_marker[j] >= 0);
      }
   }
};

HYPRE_Int
hypre_ParCSRMatrixGenerateFFFCDevice( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker_host,
                                      HYPRE_BigInt        *cpts_starts,
                                      hypre_ParCSRMatrix  *S,
                                      hypre_ParCSRMatrix **AFC_ptr,
                                      hypre_ParCSRMatrix **AFF_ptr )
{
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                num_sends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int                num_elem_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   //HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int           A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);
   /* offd part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   //HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int           A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   /* SoC */
   HYPRE_Int          *Soc_diag_j = hypre_ParCSRMatrixSocDiagJ(S);
   HYPRE_Int          *Soc_offd_j = hypre_ParCSRMatrixSocOffdJ(S);
   /* MPI size and rank*/
   HYPRE_Int           my_id, num_procs;
   /* nF and nC */
   HYPRE_Int           n_local, nF_local, nC_local;
   HYPRE_BigInt       *fpts_starts, *row_starts;
   HYPRE_BigInt        n_global, nF_global, nC_global;
   HYPRE_BigInt        F_first, C_first;
   HYPRE_Int          *CF_marker;
   /* AFF */
   HYPRE_Int           AFF_diag_nnz, AFF_offd_nnz;
   HYPRE_Int          *AFF_diag_ii, *AFF_diag_i, *AFF_diag_j;
   HYPRE_Complex      *AFF_diag_a;
   HYPRE_Int          *AFF_offd_ii, *AFF_offd_i, *AFF_offd_j;
   HYPRE_Complex      *AFF_offd_a;
   hypre_ParCSRMatrix *AFF;
   hypre_CSRMatrix    *AFF_diag, *AFF_offd;
   HYPRE_BigInt       *col_map_offd_AFF;
   HYPRE_Int           num_cols_AFF_offd;
   /* AFC */
   HYPRE_Int           AFC_diag_nnz, AFC_offd_nnz;
   HYPRE_Int          *AFC_diag_ii, *AFC_diag_i, *AFC_diag_j;
   HYPRE_Complex      *AFC_diag_a;
   HYPRE_Int          *AFC_offd_ii, *AFC_offd_i, *AFC_offd_j;
   HYPRE_Complex      *AFC_offd_a;
   hypre_ParCSRMatrix *AFC;
   hypre_CSRMatrix    *AFC_diag, *AFC_offd;
   HYPRE_BigInt       *col_map_offd_AFC;
   HYPRE_Int           num_cols_AFC_offd;
   /* work arrays */
   HYPRE_Int          *map2FC, *itmp, *A_diag_ii, *A_offd_ii, *tmp_j, *offd_mark;
   HYPRE_BigInt       *send_buf, *recv_buf;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   n_global   = hypre_ParCSRMatrixGlobalNumRows(A);
   n_local    = hypre_ParCSRMatrixNumRows(A);
   row_starts = hypre_ParCSRMatrixRowStarts(A);

   map2FC     = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);
   itmp       = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);;
   recv_buf   = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   if (my_id == (num_procs -1))
   {
      nC_global = cpts_starts[1];
   }
   hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_INT, num_procs-1, comm);
   nC_local = (HYPRE_Int) (cpts_starts[1] - cpts_starts[0]);
   fpts_starts = hypre_TAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
   fpts_starts[0] = row_starts[0] - cpts_starts[0];
   fpts_starts[1] = row_starts[1] - cpts_starts[1];
   F_first = fpts_starts[0];
   C_first = cpts_starts[0];
#else
   nC_global = cpts_starts[num_procs];
   nC_local = (HYPRE_Int)(cpts_starts[my_id+1] - cpts_starts[my_id]);
   fpts_starts = hypre_TAlloc(HYPRE_BigInt, num_procs+1, HYPRE_MEMORY_HOST);
   for (i = 0; i <= num_procs; i++)
   {
      fpts_starts[i] = row_starts[i] - cpts_starts[i];
   }
   F_first = fpts_starts[myid];
   C_first = cpts_starts[myid];
#endif
   nF_local = n_local - nC_local;
   nF_global = n_global - nC_global;

   CF_marker = hypre_TAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy( CF_marker, CF_marker_host, HYPRE_Int, n_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );

   /* map from F+C to F/C indices */
   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_negative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_negative<HYPRE_Int>()),
                      map2FC ); /* F */

   HYPRE_THRUST_CALL( exclusive_scan,
                      thrust::make_transform_iterator(CF_marker,           is_nonnegative<HYPRE_Int>()),
                      thrust::make_transform_iterator(CF_marker + n_local, is_nonnegative<HYPRE_Int>()),
                      itmp ); /* C */

   HYPRE_THRUST_CALL( scatter_if,
                      itmp,
                      itmp + n_local,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::make_transform_iterator(CF_marker, is_nonnegative<HYPRE_Int>()),
                      map2FC ); /* FC combined */

   hypre_TFree(itmp, HYPRE_MEMORY_DEVICE);

   /* send_buf: global F/C indices. Note F-pts are saved as "-x-1" */
   send_buf = hypre_TAlloc(HYPRE_BigInt, num_elem_send, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);

   FFFC_functor functor(F_first, C_first);
   HYPRE_THRUST_CALL( gather,
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + num_elem_send,
                      thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(map2FC, CF_marker)), functor),
                      send_buf );

   comm_handle = hypre_ParCSRCommHandleCreate_v2(21, comm_pkg, HYPRE_MEMORY_DEVICE, send_buf, HYPRE_MEMORY_DEVICE, recv_buf);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   hypre_TFree(send_buf, HYPRE_MEMORY_DEVICE);

   /* Diag */
   thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*, HYPRE_Complex*> > new_end;

   A_diag_ii = hypre_TAlloc(HYPRE_Int, A_diag_nnz, HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_diag_nnz, A_diag_i, A_diag_ii);

   /* AFF Diag */
   FFFC_pred<true, HYPRE_Int> AFF_pred_diag(CF_marker, CF_marker);
   AFF_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                     thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                     AFF_pred_diag );

   AFF_diag_ii = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
   AFF_diag_j  = hypre_TAlloc(HYPRE_Int,     AFF_diag_nnz, HYPRE_MEMORY_DEVICE);
   AFF_diag_a  = hypre_TAlloc(HYPRE_Complex, AFF_diag_nnz, HYPRE_MEMORY_DEVICE);

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, A_diag_j, A_diag_a)) + A_diag_nnz,
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(AFF_diag_ii, AFF_diag_j, AFF_diag_a)),
                                AFF_pred_diag );

   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_diag_ii + AFF_diag_nnz );

   HYPRE_THRUST_CALL ( gather,
                       AFF_diag_j,
                       AFF_diag_j + AFF_diag_nnz,
                       map2FC,
                       AFF_diag_j );

   HYPRE_THRUST_CALL ( gather,
                       AFF_diag_ii,
                       AFF_diag_ii + AFF_diag_nnz,
                       map2FC,
                       AFF_diag_ii );

   AFF_diag_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFF_diag_nnz, AFF_diag_ii);
   hypre_TFree(AFF_diag_ii, HYPRE_MEMORY_DEVICE);

   /* AFC Diag */
   FFFC_pred<false, HYPRE_Int> AFC_pred_diag(CF_marker, CF_marker);
   AFC_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                     thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)) + A_diag_nnz,
                                     AFC_pred_diag );

   AFC_diag_ii = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
   AFC_diag_j  = hypre_TAlloc(HYPRE_Int,     AFC_diag_nnz, HYPRE_MEMORY_DEVICE);
   AFC_diag_a  = hypre_TAlloc(HYPRE_Complex, AFC_diag_nnz, HYPRE_MEMORY_DEVICE);

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j, A_diag_a)) + A_diag_nnz,
                                thrust::make_zip_iterator(thrust::make_tuple(A_diag_ii, Soc_diag_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(AFC_diag_ii, AFC_diag_j, AFC_diag_a)),
                                AFC_pred_diag );

   hypre_TFree(A_diag_ii, HYPRE_MEMORY_DEVICE);

   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_diag_ii + AFC_diag_nnz );

   HYPRE_THRUST_CALL ( gather,
                       AFC_diag_j,
                       AFC_diag_j + AFC_diag_nnz,
                       map2FC,
                       AFC_diag_j );

   HYPRE_THRUST_CALL ( gather,
                       AFC_diag_ii,
                       AFC_diag_ii + AFC_diag_nnz,
                       map2FC,
                       AFC_diag_ii );

   AFC_diag_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_diag_nnz, AFC_diag_ii);
   hypre_TFree(AFC_diag_ii, HYPRE_MEMORY_DEVICE);

   /* Offd */
   A_offd_ii = hypre_TAlloc(HYPRE_Int, A_offd_nnz, HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(n_local, A_offd_nnz, A_offd_i, A_offd_ii);

   /* AFF Offd */
   FFFC_pred<true, HYPRE_BigInt> AFF_pred_offd(CF_marker, recv_buf);
   AFF_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                     thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                     AFF_pred_offd );

   AFF_offd_ii = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
   AFF_offd_j  = hypre_TAlloc(HYPRE_Int,     AFF_offd_nnz, HYPRE_MEMORY_DEVICE);
   AFF_offd_a  = hypre_TAlloc(HYPRE_Complex, AFF_offd_nnz, HYPRE_MEMORY_DEVICE);

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(AFF_offd_ii, AFF_offd_j, AFF_offd_a)),
                                AFF_pred_offd );

   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFF_offd_ii + AFF_offd_nnz );

   HYPRE_THRUST_CALL ( gather,
                       AFF_offd_ii,
                       AFF_offd_ii + AFF_offd_nnz,
                       map2FC,
                       AFF_offd_ii );

   AFF_offd_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFF_offd_nnz, AFF_offd_ii);

   hypre_TFree(AFF_offd_ii, HYPRE_MEMORY_DEVICE);

   /* AFC Offd */
   FFFC_pred<false, HYPRE_BigInt> AFC_pred_offd(CF_marker, recv_buf);
   AFC_offd_nnz = HYPRE_THRUST_CALL( count_if,
                                     thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                     thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)) + A_offd_nnz,
                                     AFC_pred_offd );

   AFC_offd_ii = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
   AFC_offd_j  = hypre_TAlloc(HYPRE_Int,     AFC_offd_nnz, HYPRE_MEMORY_DEVICE);
   AFC_offd_a  = hypre_TAlloc(HYPRE_Complex, AFC_offd_nnz, HYPRE_MEMORY_DEVICE);

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j, A_offd_a)) + A_offd_nnz,
                                thrust::make_zip_iterator(thrust::make_tuple(A_offd_ii, Soc_offd_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(AFC_offd_ii, AFC_offd_j, AFC_offd_a)),
                                AFC_pred_offd );

   hypre_TFree(A_offd_ii, HYPRE_MEMORY_DEVICE);

   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == AFC_offd_ii + AFC_offd_nnz );

   HYPRE_THRUST_CALL ( gather,
                       AFC_offd_ii,
                       AFC_offd_ii + AFC_offd_nnz,
                       map2FC,
                       AFC_offd_ii );

   AFC_offd_i = hypreDevice_CsrRowIndicesToPtrs(nF_local, AFC_offd_nnz, AFC_offd_ii);

   hypre_TFree(AFC_offd_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_marker, HYPRE_MEMORY_DEVICE);
   hypre_TFree(map2FC, HYPRE_MEMORY_DEVICE);

   /* col_map_offd_AFF */
   HYPRE_Int tmp_j_size = hypre_max(hypre_max(AFF_offd_nnz, AFC_offd_nnz), num_cols_A_offd);
   tmp_j = hypre_TAlloc(HYPRE_Int, tmp_j_size, HYPRE_MEMORY_DEVICE);
   offd_mark = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *tmp_end;

   hypre_TMemcpy(tmp_j, AFF_offd_j, HYPRE_Int, AFF_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL(sort, tmp_j, tmp_j + AFF_offd_nnz);
   tmp_end = HYPRE_THRUST_CALL(unique, tmp_j, tmp_j + AFF_offd_nnz);
   num_cols_AFF_offd = tmp_end - tmp_j;
   HYPRE_THRUST_CALL(fill_n, offd_mark, num_cols_A_offd, 0);
   hypreDevice_ScatterConstant(offd_mark, num_cols_AFF_offd, tmp_j, 1);
   HYPRE_THRUST_CALL(exclusive_scan, offd_mark, offd_mark + num_cols_A_offd, tmp_j);
   HYPRE_THRUST_CALL(gather, AFF_offd_j, AFF_offd_j + AFF_offd_nnz, tmp_j, AFF_offd_j);
   col_map_offd_AFF = hypre_TAlloc(HYPRE_Int, num_cols_AFF_offd, HYPRE_MEMORY_DEVICE);
   tmp_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_transform_iterator(recv_buf, -_1-1),
                                thrust::make_transform_iterator(recv_buf, -_1-1) + num_cols_A_offd,
                                offd_mark,
                                col_map_offd_AFF,
                                thrust::identity<HYPRE_Int>() );
   hypre_assert(tmp_end - col_map_offd_AFF == num_cols_AFF_offd);

   /* col_map_offd_AFC */
   hypre_TMemcpy(tmp_j, AFC_offd_j, HYPRE_Int, AFC_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL(sort, tmp_j, tmp_j + AFC_offd_nnz);
   tmp_end = HYPRE_THRUST_CALL(unique, tmp_j, tmp_j + AFC_offd_nnz);
   num_cols_AFC_offd = tmp_end - tmp_j;
   HYPRE_THRUST_CALL(fill_n, offd_mark, num_cols_A_offd, 0);
   hypreDevice_ScatterConstant(offd_mark, num_cols_AFC_offd, tmp_j, 1);
   HYPRE_THRUST_CALL(exclusive_scan, offd_mark, offd_mark + num_cols_A_offd, tmp_j);
   HYPRE_THRUST_CALL(gather, AFC_offd_j, AFC_offd_j + AFC_offd_nnz, tmp_j, AFC_offd_j);
   col_map_offd_AFC = hypre_TAlloc(HYPRE_Int, num_cols_AFC_offd, HYPRE_MEMORY_DEVICE);
   tmp_end = HYPRE_THRUST_CALL( copy_if,
                                recv_buf,
                                recv_buf + num_cols_A_offd,
                                offd_mark,
                                col_map_offd_AFC,
                                thrust::identity<HYPRE_Int>());
   hypre_assert(tmp_end - col_map_offd_AFC == num_cols_AFC_offd);

   hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(offd_mark, HYPRE_MEMORY_DEVICE);
   hypre_TFree(recv_buf, HYPRE_MEMORY_DEVICE);

   //printf("AFF_diag_nnz %d, AFF_offd_nnz %d, AFC_diag_nnz %d, AFC_offd_nnz %d\n", AFF_diag_nnz, AFF_offd_nnz, AFC_diag_nnz, AFC_offd_nnz);

   /* AFF */
   AFF = hypre_ParCSRMatrixCreate(comm,
                                  nF_global,
                                  nF_global,
                                  fpts_starts,
                                  fpts_starts,
                                  num_cols_AFF_offd,
                                  AFF_diag_nnz,
                                  AFF_offd_nnz);

   hypre_ParCSRMatrixOwnsRowStarts(AFF) = 1;
   hypre_ParCSRMatrixOwnsColStarts(AFF) = 0;

   AFF_diag = hypre_ParCSRMatrixDiag(AFF);
   hypre_CSRMatrixData(AFF_diag) = AFF_diag_a;
   hypre_CSRMatrixI(AFF_diag)    = AFF_diag_i;
   hypre_CSRMatrixJ(AFF_diag)    = AFF_diag_j;

   AFF_offd = hypre_ParCSRMatrixOffd(AFF);
   hypre_CSRMatrixData(AFF_offd) = AFF_offd_a;
   hypre_CSRMatrixI(AFF_offd)    = AFF_offd_i;
   hypre_CSRMatrixJ(AFF_offd)    = AFF_offd_j;

   hypre_CSRMatrixMemoryLocation(AFF_diag) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(AFF_offd) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(AFF) = col_map_offd_AFF;
   hypre_ParCSRMatrixColMapOffd(AFF) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFF_offd, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AFF), col_map_offd_AFF, HYPRE_BigInt, num_cols_AFF_offd,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRMatrixSetNumNonzeros(AFF);
   hypre_ParCSRMatrixDNumNonzeros(AFF) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(AFF);
   hypre_MatvecCommPkgCreate(AFF);

   /* AFC */
   AFC = hypre_ParCSRMatrixCreate(comm,
                                  nF_global,
                                  nC_global,
                                  fpts_starts,
                                  cpts_starts,
                                  num_cols_AFC_offd,
                                  AFC_diag_nnz,
                                  AFC_offd_nnz);

   hypre_ParCSRMatrixOwnsRowStarts(AFC) = 0;
   hypre_ParCSRMatrixOwnsColStarts(AFC) = 0;

   AFC_diag = hypre_ParCSRMatrixDiag(AFC);
   hypre_CSRMatrixData(AFC_diag) = AFC_diag_a;
   hypre_CSRMatrixI(AFC_diag)    = AFC_diag_i;
   hypre_CSRMatrixJ(AFC_diag)    = AFC_diag_j;

   AFC_offd = hypre_ParCSRMatrixOffd(AFC);
   hypre_CSRMatrixData(AFC_offd) = AFC_offd_a;
   hypre_CSRMatrixI(AFC_offd)    = AFC_offd_i;
   hypre_CSRMatrixJ(AFC_offd)    = AFC_offd_j;

   hypre_CSRMatrixMemoryLocation(AFC_diag) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixMemoryLocation(AFC_offd) = HYPRE_MEMORY_DEVICE;

   hypre_ParCSRMatrixDeviceColMapOffd(AFC) = col_map_offd_AFC;
   hypre_ParCSRMatrixColMapOffd(AFC) = hypre_TAlloc(HYPRE_BigInt, num_cols_AFC_offd, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(AFC), col_map_offd_AFC, HYPRE_BigInt, num_cols_AFC_offd,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   hypre_ParCSRMatrixSetNumNonzeros(AFC);
   hypre_ParCSRMatrixDNumNonzeros(AFC) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(AFC);
   hypre_MatvecCommPkgCreate(AFC);

   *AFC_ptr = AFC;
   *AFF_ptr = AFF;

   return hypre_error_flag;
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

#endif // #if defined(HYPRE_USING_CUDA)

