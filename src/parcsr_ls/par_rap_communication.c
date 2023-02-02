/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

HYPRE_Int
hypre_GetCommPkgRTFromCommPkgA( hypre_ParCSRMatrix *RT,
                                hypre_ParCSRMatrix *A,
                                HYPRE_Int *fine_to_coarse,
                                HYPRE_Int *tmp_map_offd)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(RT);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   HYPRE_Int *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   HYPRE_Int *recv_vec_starts_A = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   HYPRE_Int num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   HYPRE_Int *send_procs_A = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   HYPRE_Int *send_map_starts_A = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   hypre_ParCSRCommPkg *comm_pkg = NULL;
   HYPRE_Int num_recvs_RT;
   HYPRE_Int *recv_procs_RT;
   HYPRE_Int *recv_vec_starts_RT;
   HYPRE_Int num_sends_RT;
   HYPRE_Int *send_procs_RT;
   HYPRE_Int *send_map_starts_RT;
   HYPRE_Int *send_map_elmts_RT;

   HYPRE_BigInt *col_map_offd_RT = hypre_ParCSRMatrixColMapOffd(RT);
   HYPRE_Int num_cols_offd_RT = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd(RT));
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(RT);
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_BigInt *fine_to_coarse_offd = NULL;
   HYPRE_BigInt *big_buf_data = NULL;
   HYPRE_BigInt *send_big_elmts = NULL;
   HYPRE_BigInt my_first_cpt;

   HYPRE_Int i, j;
   HYPRE_Int vec_len, vec_start;
   HYPRE_Int num_procs, my_id;
   HYPRE_Int ierr = 0;
   HYPRE_Int num_requests;
   HYPRE_Int offd_col, proc_num;
   HYPRE_Int num_threads = hypre_NumThreads();
   HYPRE_Int size, rest, ns, ne, start;
   HYPRE_Int index;

   HYPRE_Int *proc_mark;
   HYPRE_Int *change_array;
   HYPRE_Int *coarse_counter;
   HYPRE_Int coarse_shift;

   hypre_MPI_Request *requests;
   hypre_MPI_Status *status;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*--------------------------------------------------------------------------
    * determine num_recvs, recv_procs and recv_vec_starts for RT
    *--------------------------------------------------------------------------*/

   proc_mark = hypre_CTAlloc(HYPRE_Int,  num_recvs_A, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_recvs_A; i++)
   {
      proc_mark[i] = 0;
   }

   proc_num = 0;
   num_recvs_RT = 0;
   if (num_cols_offd_RT)
   {
      for (i = 0; i < num_recvs_A; i++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
         {
            offd_col = tmp_map_offd[proc_num];
            if (offd_col == j)
            {
               proc_mark[i]++;
               proc_num++;
               if (proc_num == num_cols_offd_RT) { break; }
            }
         }
         if (proc_mark[i]) { num_recvs_RT++; }
         if (proc_num == num_cols_offd_RT) { break; }
      }
   }

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt, send_map_starts_A[num_sends_A], HYPRE_MEMORY_HOST);
   coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);

   my_first_cpt = hypre_ParCSRMatrixColStarts(RT)[0];

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }

   index = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i + 1); j++)
         big_buf_data[index++] = my_first_cpt +
                                 (HYPRE_BigInt)fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A, j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg_A, big_buf_data,
                                               fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_cols_offd_RT; i++)
   {
      col_map_offd_RT[i] = fine_to_coarse_offd[tmp_map_offd[i]];
   }

   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   //hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);

   recv_procs_RT = hypre_CTAlloc(HYPRE_Int, num_recvs_RT, HYPRE_MEMORY_HOST);
   recv_vec_starts_RT = hypre_CTAlloc(HYPRE_Int,  num_recvs_RT + 1, HYPRE_MEMORY_HOST);

   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i = 0; i < num_recvs_A; i++)
   {
      if (proc_mark[i])
      {
         recv_procs_RT[j] = recv_procs_A[i];
         recv_vec_starts_RT[j + 1] = recv_vec_starts_RT[j] + proc_mark[i];
         j++;
      }
   }

   /*--------------------------------------------------------------------------
    * send num_changes to recv_procs_A and receive change_array from send_procs_A
    *--------------------------------------------------------------------------*/

   num_requests = num_recvs_A + num_sends_A;
   requests = hypre_CTAlloc(hypre_MPI_Request,  num_requests, HYPRE_MEMORY_HOST);
   status = hypre_CTAlloc(hypre_MPI_Status,  num_requests, HYPRE_MEMORY_HOST);

   change_array = hypre_CTAlloc(HYPRE_Int,  num_sends_A, HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends_A; i++)
      hypre_MPI_Irecv(&change_array[i], 1, HYPRE_MPI_INT, send_procs_A[i], 0, comm,
                      &requests[j++]);

   for (i = 0; i < num_recvs_A; i++)
      hypre_MPI_Isend(&proc_mark[i], 1, HYPRE_MPI_INT, recv_procs_A[i], 0, comm,
                      &requests[j++]);

   hypre_MPI_Waitall(num_requests, requests, status);

   hypre_TFree(proc_mark, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * if change_array[i] is 0 , omit send_procs_A[i] in send_procs_RT
    *--------------------------------------------------------------------------*/

   num_sends_RT = 0;
   for (i = 0; i < num_sends_A; i++)
      if (change_array[i])
      {
         num_sends_RT++;
      }

   send_procs_RT = hypre_CTAlloc(HYPRE_Int,  num_sends_RT, HYPRE_MEMORY_HOST);
   send_map_starts_RT = hypre_CTAlloc(HYPRE_Int,  num_sends_RT + 1, HYPRE_MEMORY_HOST);

   j = 0;
   send_map_starts_RT[0] = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      if (change_array[i])
      {
         send_procs_RT[j] = send_procs_A[i];
         send_map_starts_RT[j + 1] = send_map_starts_RT[j] + change_array[i];
         j++;
      }
   }

   /*--------------------------------------------------------------------------
    * generate send_map_elmts
    *--------------------------------------------------------------------------*/

   send_map_elmts_RT = hypre_CTAlloc(HYPRE_Int, send_map_starts_RT[num_sends_RT], HYPRE_MEMORY_HOST);
   send_big_elmts = hypre_CTAlloc(HYPRE_BigInt, send_map_starts_RT[num_sends_RT], HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends_RT; i++)
   {
      vec_start = send_map_starts_RT[i];
      vec_len = send_map_starts_RT[i + 1] - vec_start;
      hypre_MPI_Irecv(&send_big_elmts[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      send_procs_RT[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs_RT; i++)
   {
      vec_start = recv_vec_starts_RT[i];
      vec_len = recv_vec_starts_RT[i + 1] - vec_start;
      hypre_MPI_Isend(&col_map_offd_RT[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      recv_procs_RT[i], 0, comm, &requests[j++]);
   }

   hypre_MPI_Waitall(j, requests, status);

   for (i = 0; i < send_map_starts_RT[num_sends_RT]; i++)
   {
      send_map_elmts_RT[i] = (HYPRE_Int)(send_big_elmts[i] - first_col_diag);
   }

   /* Create and fill communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_RT, recv_procs_RT, recv_vec_starts_RT,
                                    num_sends_RT, send_procs_RT, send_map_starts_RT,
                                    send_map_elmts_RT,
                                    &comm_pkg);

   hypre_TFree(status, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(send_big_elmts, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixCommPkg(RT) = comm_pkg;
   hypre_TFree(change_array, HYPRE_MEMORY_HOST);

   return ierr;
}

HYPRE_Int
hypre_GenerateSendMapAndCommPkg(MPI_Comm comm, HYPRE_Int num_sends, HYPRE_Int num_recvs,
                                HYPRE_Int *recv_procs, HYPRE_Int *send_procs,
                                HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A)
{
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;
   HYPRE_Int i, j;
   HYPRE_Int num_requests = num_sends + num_recvs;
   hypre_MPI_Request *requests;
   hypre_MPI_Status *status;
   HYPRE_Int vec_len, vec_start;
   hypre_ParCSRCommPkg *comm_pkg = NULL;
   HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt *send_big_elmts = NULL;

   /*--------------------------------------------------------------------------
    * generate send_map_starts and send_map_elmts
    *--------------------------------------------------------------------------*/

   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests, HYPRE_MEMORY_HOST);
   status = hypre_CTAlloc(hypre_MPI_Status, num_requests, HYPRE_MEMORY_HOST);
   send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      hypre_MPI_Irecv(&send_map_starts[i + 1], 1, HYPRE_MPI_INT, send_procs[i], 0, comm,
                      &requests[j++]);
   }

   for (i = 0; i < num_recvs; i++)
   {
      vec_len = recv_vec_starts[i + 1] - recv_vec_starts[i];
      hypre_MPI_Isend(&vec_len, 1, HYPRE_MPI_INT, recv_procs[i], 0, comm, &requests[j++]);
   }

   hypre_MPI_Waitall(j, requests, status);

   send_map_starts[0] = 0;
   for (i = 0; i < num_sends; i++)
   {
      send_map_starts[i + 1] += send_map_starts[i];
   }

   send_map_elmts = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends], HYPRE_MEMORY_HOST);
   send_big_elmts = hypre_CTAlloc(HYPRE_BigInt, send_map_starts[num_sends], HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i + 1] - vec_start;
      hypre_MPI_Irecv(&send_big_elmts[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      send_procs[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i + 1] - vec_start;
      hypre_MPI_Isend(&col_map_offd[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      recv_procs[i], 0, comm, &requests[j++]);
   }

   hypre_MPI_Waitall(j, requests, status);

   for (i = 0; i < send_map_starts[num_sends]; i++)
   {
      send_map_elmts[i] = (HYPRE_Int)(send_big_elmts[i] - first_col_diag);
   }

   /* Create and fill communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   hypre_TFree(status, HYPRE_MEMORY_HOST);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(send_big_elmts, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;

   return hypre_error_flag;
}
