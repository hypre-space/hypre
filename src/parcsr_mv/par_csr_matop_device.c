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
   HYPRE_Int  i, j;
   HYPRE_Int  num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   HYPRE_Int *d_send_i, *send_i, *d_send_j, *send_j, *d_send_map, *recv_i, *recv_j;
   HYPRE_Int *send_jstarts, *recv_jstarts;
   HYPRE_Complex *d_send_a = NULL, *send_a = NULL, *recv_a = NULL;
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
   HYPRE_Int        first_col       = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int       *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int       *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

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
   send_i     = hypre_TAlloc(HYPRE_Int, num_rows_send,     HYPRE_MEMORY_HOST);
   recv_i     = hypre_TAlloc(HYPRE_Int, num_rows_recv + 1, HYPRE_MEMORY_HOST);

   d_send_map = hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg);
   if (d_send_map == NULL)
   {
      d_send_map = hypre_TAlloc(HYPRE_Int, num_rows_send, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_send_map, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), HYPRE_Int,
                    num_rows_send, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) = d_send_map;
   }

   /* fill the send array with row lengths */
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
   d_send_j = hypre_TAlloc(HYPRE_Int, num_nnz_send, HYPRE_MEMORY_DEVICE);
   send_j   = hypre_TAlloc(HYPRE_Int, num_nnz_send, HYPRE_MEMORY_HOST);
   if (want_data)
   {
      d_send_a = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_DEVICE);
      send_a   = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_HOST);
   }

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_Int, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   /* job == 2, d_send_i is input that contains row ptrs (length num_rows_send) */
   hypreDevice_CopyParCSRRows(num_rows_send, d_send_map, 2, num_procs > 1,
                              first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a,
                              A_offd_i, A_offd_j, A_offd_a,
                              d_send_i, d_send_j, d_send_a);

   hypre_TMemcpy(send_j, d_send_j, HYPRE_Int,     num_nnz_send, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(send_a, d_send_a, HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

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
   recv_j = hypre_TAlloc(HYPRE_Int, num_nnz_recv, HYPRE_MEMORY_HOST);
   if (want_data)
   {
      recv_a = hypre_TAlloc(HYPRE_Complex, num_nnz_recv, HYPRE_MEMORY_HOST);
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
   comm_handle_j = hypre_ParCSRCommHandleCreate(11, comm_pkg_j, send_j, recv_j);
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
   hypre_CSRMatrixI   (A_ext) = recv_i;
   hypre_CSRMatrixJ   (A_ext) = recv_j;
   hypre_CSRMatrixData(A_ext) = recv_a;
   hypre_CSRMatrixMemoryLocation(A_ext) = HYPRE_MEMORY_HOST;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;

   *request_ptr = (void *) vrequest;

   /* free */
   hypre_TFree(send_i, HYPRE_MEMORY_HOST);
   hypre_TFree(d_send_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_send_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_send_a, HYPRE_MEMORY_DEVICE);

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
   HYPRE_Int              *send_j        = (HYPRE_Int *)     hypre_ParCSRCommHandleSendData(comm_handle_j);
   HYPRE_Complex          *send_a        = (HYPRE_Complex *) hypre_ParCSRCommHandleSendData(comm_handle_a);

   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(send_j, HYPRE_MEMORY_HOST);
   hypre_TFree(send_a, HYPRE_MEMORY_HOST);

   hypre_CSRMatrix *A_ext_device = hypre_CSRMatrixClone_v2(A_ext, 1, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(A_ext);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return A_ext_device;
}

HYPRE_Int
hypre_ParcsrGetExternalRowsDeviceInit_v2( hypre_ParCSRMatrix   *A,
                                          HYPRE_Int             indices_len,
                                          HYPRE_Int            *indices,
                                          hypre_ParCSRCommPkg  *comm_pkg,
                                          HYPRE_Int             want_data,
                                          void                **request_ptr)
{
   HYPRE_Int  i, j;
   HYPRE_Int  num_sends, num_rows_send, num_nnz_send, num_recvs, num_rows_recv, num_nnz_recv;
   HYPRE_Int *d_send_i, *send_i, *d_send_j, *d_send_map, *d_recv_i, *recv_i, *d_recv_j;
   HYPRE_Int *send_jstarts, *recv_jstarts;
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
   HYPRE_Int        first_col       = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int       *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int       *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

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
   d_send_j = hypre_TAlloc(HYPRE_Int, num_nnz_send, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_send_a = hypre_TAlloc(HYPRE_Complex, num_nnz_send, HYPRE_MEMORY_DEVICE);
   }

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_Int, num_cols_A_offd,
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
   d_recv_j = hypre_TAlloc(HYPRE_Int, num_nnz_recv, HYPRE_MEMORY_DEVICE);
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
   comm_handle_j = hypre_ParCSRCommHandleCreate(11, comm_pkg_j, d_send_j, d_recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = hypre_ParCSRCommHandleCreate(1, comm_pkg_j, d_send_a, d_recv_a);
   }
   else
   {
      comm_handle_a = NULL;
   }

   hypre_TMemcpy(d_recv_i, recv_i, HYPRE_Int, num_rows_recv+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* create A_ext: on device */
   A_ext = hypre_CSRMatrixCreate(num_rows_recv, hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   hypre_CSRMatrixI   (A_ext) = d_recv_i;
   hypre_CSRMatrixJ   (A_ext) = d_recv_j;
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
hypre_ParcsrGetExternalRowsDeviceWait_v2(void *vrequest)
{
   void **request = (void **) vrequest;

   hypre_ParCSRCommHandle *comm_handle_j = (hypre_ParCSRCommHandle *) request[0];
   hypre_ParCSRCommHandle *comm_handle_a = (hypre_ParCSRCommHandle *) request[1];
   hypre_CSRMatrix        *A_ext         = (hypre_CSRMatrix *)        request[2];

   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

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

   HYPRE_Int  local_num_rows   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int  glbal_num_cols   = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_Int  first_col        = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int  num_cols_A_offd  = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int *col_map_offd_A   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int *d_col_map_offd_A = hypre_ParCSRMatrixDeviceColMapOffd(A);

   hypre_CSRMatrix *B;
   HYPRE_Int        B_nrows = local_num_rows;
   HYPRE_Int        B_ncols = glbal_num_cols;
   HYPRE_Int       *B_i = hypre_TAlloc(HYPRE_Int, B_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_j;
   HYPRE_Complex   *B_a;
   HYPRE_Int        B_nnz;

   HYPRE_Int      num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_Memset(B_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   hypreDevice_GetRowNnz(B_nrows, NULL, A_diag_i, A_offd_i, B_i+1);

   hypreDevice_IntegerInclusiveScan(B_nrows+1, B_i);

   /* total number of nnz */
   hypre_TMemcpy(&B_nnz, B_i+B_nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   B_j = hypre_TAlloc(HYPRE_Int,     B_nnz, HYPRE_MEMORY_DEVICE);
   B_a = hypre_TAlloc(HYPRE_Complex, B_nnz, HYPRE_MEMORY_DEVICE);

   if (d_col_map_offd_A == NULL)
   {
      d_col_map_offd_A = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_col_map_offd_A, col_map_offd_A, HYPRE_Int, num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDeviceColMapOffd(A) = d_col_map_offd_A;
   }

   hypreDevice_CopyParCSRRows(B_nrows, NULL, 2, num_procs > 1, first_col, d_col_map_offd_A,
                              A_diag_i, A_diag_j, A_diag_a, A_offd_i, A_offd_j, A_offd_a,
                              B_i, B_j, B_a);


   /* output */
   B = hypre_CSRMatrixCreate(B_nrows, B_ncols, B_nnz);
   hypre_CSRMatrixI   (B) = B_i;
   hypre_CSRMatrixJ   (B) = B_j;
   hypre_CSRMatrixData(B) = B_a;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

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
   HYPRE_Int     *B_ext_j_d      = hypre_CSRMatrixJ(B_ext);
   HYPRE_Complex *B_ext_a_d      = hypre_CSRMatrixData(B_ext);
   HYPRE_Int      B_ext_ncols    = hypre_CSRMatrixNumCols(B_ext);
   HYPRE_Int      B_ext_nrows    = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int      B_ext_nnz      = hypre_CSRMatrixNumNonzeros(B_ext);
   HYPRE_Int     *B_ext_rownnz_d = hypre_TAlloc(HYPRE_Int,     B_ext_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_rownnz_h = hypre_TAlloc(HYPRE_Int,     B_ext_nrows,     HYPRE_MEMORY_HOST);
   HYPRE_Int     *B_ext_i_h      = hypre_TAlloc(HYPRE_Int,     B_ext_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int     *B_ext_j_h      = hypre_TAlloc(HYPRE_Int,     B_ext_nnz,       HYPRE_MEMORY_HOST);
   HYPRE_Complex *B_ext_a_h      = hypre_TAlloc(HYPRE_Complex, B_ext_nnz,       HYPRE_MEMORY_HOST);

   hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   hypre_CSRMatrix *B_int_h;
   HYPRE_Int        B_int_nrows = num_elmts_send;
   HYPRE_Int        B_int_ncols = B_ext_ncols;
   HYPRE_Int       *B_int_i_h   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int       *B_int_j_h   = NULL;
   HYPRE_Complex   *B_int_a_h   = NULL;
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
   thrust::adjacent_difference(thrust::device, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
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

   hypre_TMemcpy(B_ext_j_h, B_ext_j_d, HYPRE_Int,     B_ext_nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(B_ext_a_h, B_ext_a_d, HYPRE_Complex, B_ext_nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

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


   B_int_j_h = hypre_TAlloc(HYPRE_Int,     B_int_nnz, HYPRE_MEMORY_HOST);
   B_int_a_h = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_HOST);

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* note the order of send/recv is reversed */
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j) = jdata_send_map_starts;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j) = jdata_recv_vec_starts;

   /* send/recv CSR rows */
   comm_handle_a = hypre_ParCSRCommHandleCreate( 1, comm_pkg_j, B_ext_a_h, B_int_a_h);
   comm_handle_j = hypre_ParCSRCommHandleCreate(11, comm_pkg_j, B_ext_j_h, B_int_j_h);

   /* create CSR */
   B_int_h = hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   hypre_CSRMatrixI(B_int_h)    = B_int_i_h;
   hypre_CSRMatrixJ(B_int_h)    = B_int_j_h;
   hypre_CSRMatrixData(B_int_h) = B_int_a_h;
   hypre_CSRMatrixMemoryLocation(B_int_h) = HYPRE_MEMORY_HOST;

   /* output */
   vrequest = hypre_TAlloc(void *, 3, HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int_h;

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
   hypre_CSRMatrix        *B_int_h       = (hypre_CSRMatrix *)        request[2];
   HYPRE_Int              *B_ext_j_h     = (HYPRE_Int *)     hypre_ParCSRCommHandleSendData(comm_handle_j);
   HYPRE_Complex          *B_ext_a_h     = (HYPRE_Complex *) hypre_ParCSRCommHandleSendData(comm_handle_a);

   /* communication done */
   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_a);

   hypre_TFree(B_ext_j_h, HYPRE_MEMORY_HOST);
   hypre_TFree(B_ext_a_h, HYPRE_MEMORY_HOST);

   hypre_CSRMatrix *B_int_d = hypre_CSRMatrixClone_v2(B_int_h, 1, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(B_int_h);

   hypre_TFree(request, HYPRE_MEMORY_HOST);

   return B_int_d;
}

HYPRE_Int
hypre_ExchangeExternalRowsDeviceInit_v2( hypre_CSRMatrix      *B_ext,
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
   HYPRE_Int     *B_ext_j_d      = hypre_CSRMatrixJ(B_ext);
   HYPRE_Complex *B_ext_a_d      = hypre_CSRMatrixData(B_ext);
   HYPRE_Int      B_ext_ncols    = hypre_CSRMatrixNumCols(B_ext);
   HYPRE_Int      B_ext_nrows    = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int      B_ext_nnz      = hypre_CSRMatrixNumNonzeros(B_ext);
   HYPRE_Int     *B_ext_rownnz_d = hypre_TAlloc(HYPRE_Int,     B_ext_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_rownnz_h = hypre_TAlloc(HYPRE_Int,     B_ext_nrows,     HYPRE_MEMORY_HOST);
   HYPRE_Int     *B_ext_i_h      = hypre_TAlloc(HYPRE_Int,     B_ext_nrows + 1, HYPRE_MEMORY_HOST);

   hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   hypre_CSRMatrix *B_int_d;
   HYPRE_Int        B_int_nrows = num_elmts_send;
   HYPRE_Int        B_int_ncols = B_ext_ncols;
   HYPRE_Int       *B_int_i_h   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int       *B_int_i_d   = hypre_TAlloc(HYPRE_Int, B_int_nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *B_int_j_d   = NULL;
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
   thrust::adjacent_difference(thrust::device, B_ext_i_d, B_ext_i_d + B_ext_nrows + 1, B_ext_rownnz_d);
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

   B_int_j_d = hypre_TAlloc(HYPRE_Int,     B_int_nnz, HYPRE_MEMORY_DEVICE);
   B_int_a_d = hypre_TAlloc(HYPRE_Complex, B_int_nnz, HYPRE_MEMORY_DEVICE);

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i_h[send_map_starts[i]];
   }

   /* note the order of send/recv is reversed */
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j) = jdata_send_map_starts;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j) = jdata_recv_vec_starts;

   /* send/recv CSR rows */
   comm_handle_a = hypre_ParCSRCommHandleCreate( 1, comm_pkg_j, B_ext_a_d, B_int_a_d);
   comm_handle_j = hypre_ParCSRCommHandleCreate(11, comm_pkg_j, B_ext_j_d, B_int_j_d);

   hypre_TMemcpy(B_int_i_d, B_int_i_h, HYPRE_Int, B_int_nrows+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* create CSR: on device */
   B_int_d = hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   hypre_CSRMatrixI(B_int_d)    = B_int_i_d;
   hypre_CSRMatrixJ(B_int_d)    = B_int_j_d;
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
hypre_ExchangeExternalRowsDeviceWait_v2(void *vrequest)
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
   HYPRE_Int memory_location = hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B));

   hypre_assert( hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(B)) ==
                 hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(B)) );

   hypre_assert( hypre_GetActualMemLocation(memory_location) == HYPRE_MEMORY_DEVICE );

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

#endif // #if defined(HYPRE_USING_CUDA)

