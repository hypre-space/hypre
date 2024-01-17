/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <HYPRE_config.h>
#include "_hypre_utilities.h"
#include "par_csr_block_matrix.h"
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * used in RAP function - block size must be an argument because RAP_int may
 * by NULL
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_ExchangeRAPBlockData(hypre_CSRBlockMatrix *RAP_int,
                           hypre_ParCSRCommPkg *comm_pkg_RT, HYPRE_Int block_size)
{
   HYPRE_Int     *RAP_int_i;
   HYPRE_BigInt  *RAP_int_j = NULL;
   HYPRE_Complex *RAP_int_data = NULL;
   HYPRE_Int     num_cols = 0;

   MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg_RT);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
   HYPRE_Int *recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg_RT);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_RT);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
   HYPRE_Int *send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg_RT);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);

   /*   HYPRE_Int block_size = hypre_CSRBlockMatrixBlockSize(RAP_int); */

   hypre_CSRBlockMatrix *RAP_ext;

   HYPRE_Int     *RAP_ext_i;
   HYPRE_BigInt  *RAP_ext_j = NULL;
   HYPRE_Complex *RAP_ext_data = NULL;

   hypre_ParCSRCommHandle *comm_handle = NULL;
   hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int num_rows;
   HYPRE_Int num_nonzeros;
   HYPRE_Int i, j, bnnz;
   HYPRE_Int num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   bnnz = block_size * block_size;

   RAP_ext_i = hypre_CTAlloc(HYPRE_Int,  send_map_starts[num_sends] + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_recvs + 1, HYPRE_MEMORY_HOST);
   jdata_send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
    * elements of row j (to be determined through send_map_elmnts on the
    * receiving end)
    *--------------------------------------------------------------------------*/

   if (num_recvs)
   {
      RAP_int_i = hypre_CSRBlockMatrixI(RAP_int);
      RAP_int_j = hypre_CSRBlockMatrixBigJ(RAP_int);
      RAP_int_data = hypre_CSRBlockMatrixData(RAP_int);
      num_cols = hypre_CSRBlockMatrixNumCols(RAP_int);
   }
   jdata_recv_vec_starts[0] = 0;
   for (i = 0; i < num_recvs; i++)
   {
      jdata_recv_vec_starts[i + 1] = RAP_int_i[recv_vec_starts[i + 1]];
   }

   for (i = num_recvs; i > 0; i--)
      for (j = recv_vec_starts[i]; j > recv_vec_starts[i - 1]; j--)
      {
         RAP_int_i[j] -= RAP_int_i[j - 1];
      }

   /*--------------------------------------------------------------------------
    * initialize communication
    *--------------------------------------------------------------------------*/

   if (num_recvs && num_sends)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 &RAP_int_i[1], &RAP_ext_i[1]);
   }
   else if (num_recvs)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 &RAP_int_i[1], NULL);
   }
   else if (num_sends)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 NULL, &RAP_ext_i[1]);
   }

   /* Create temporary communication package - note: send and recv are reversed */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL, &tmp_comm_pkg);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * compute num_nonzeros for RAP_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         RAP_ext_i[j + 1] += RAP_ext_i[j];
      }
   }

   num_rows = send_map_starts[num_sends];
   num_nonzeros = RAP_ext_i[num_rows];
   if (num_nonzeros)
   {
      RAP_ext_j = hypre_CTAlloc(HYPRE_BigInt,  num_nonzeros, HYPRE_MEMORY_HOST);
      RAP_ext_data = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros * bnnz, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_sends + 1; i++)
   {
      jdata_send_map_starts[i] = RAP_ext_i[send_map_starts[i]];
   }

   comm_handle = hypre_ParCSRBlockCommHandleCreate(1, bnnz, tmp_comm_pkg,
                                                   (void *) RAP_int_data, (void *) RAP_ext_data);
   hypre_ParCSRBlockCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   comm_handle = hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, RAP_int_j,
                                              RAP_ext_j);
   RAP_ext = hypre_CSRBlockMatrixCreate(block_size, num_rows, num_cols,
                                        num_nonzeros);

   hypre_CSRBlockMatrixI(RAP_ext) = RAP_ext_i;
   if (num_nonzeros)
   {
      hypre_CSRBlockMatrixBigJ(RAP_ext) = RAP_ext_j;
      hypre_CSRBlockMatrixData(RAP_ext) = RAP_ext_data;
   }

   /* Free memory */
   hypre_TFree(jdata_recv_vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(jdata_send_map_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   return RAP_ext;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGBuildCoarseOperator
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixRAP(hypre_ParCSRBlockMatrix  *RT,
                           hypre_ParCSRBlockMatrix  *A,
                           hypre_ParCSRBlockMatrix  *P,
                           hypre_ParCSRBlockMatrix **RAP_ptr )

{
   MPI_Comm        comm = hypre_ParCSRBlockMatrixComm(A);

   hypre_CSRBlockMatrix *RT_diag = hypre_ParCSRBlockMatrixDiag(RT);
   hypre_CSRBlockMatrix *RT_offd = hypre_ParCSRBlockMatrixOffd(RT);
   HYPRE_Int             num_cols_offd_RT = hypre_CSRBlockMatrixNumCols(RT_offd);
   HYPRE_Int             num_rows_offd_RT = hypre_CSRBlockMatrixNumRows(RT_offd);
   hypre_ParCSRCommPkg   *comm_pkg_RT = hypre_ParCSRBlockMatrixCommPkg(RT);
   HYPRE_Int             num_recvs_RT = 0;
   HYPRE_Int             num_sends_RT = 0;
   HYPRE_Int             *send_map_starts_RT = NULL;
   HYPRE_Int             *send_map_elmts_RT;

   hypre_CSRBlockMatrix *A_diag = hypre_ParCSRBlockMatrixDiag(A);

   HYPRE_Complex         *A_diag_data = hypre_CSRBlockMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRBlockMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRBlockMatrixJ(A_diag);
   HYPRE_Int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);

   hypre_CSRBlockMatrix *A_offd = hypre_ParCSRBlockMatrixOffd(A);

   HYPRE_Complex         *A_offd_data = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRBlockMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRBlockMatrixJ(A_offd);

   HYPRE_Int  num_cols_diag_A = hypre_CSRBlockMatrixNumCols(A_diag);
   HYPRE_Int  num_cols_offd_A = hypre_CSRBlockMatrixNumCols(A_offd);

   hypre_CSRBlockMatrix *P_diag = hypre_ParCSRBlockMatrixDiag(P);

   HYPRE_Complex         *P_diag_data = hypre_CSRBlockMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRBlockMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRBlockMatrixJ(P_diag);

   hypre_CSRBlockMatrix *P_offd = hypre_ParCSRBlockMatrixOffd(P);
   HYPRE_BigInt          *col_map_offd_P = hypre_ParCSRBlockMatrixColMapOffd(P);

   HYPRE_Complex         *P_offd_data = hypre_CSRBlockMatrixData(P_offd);
   HYPRE_Int             *P_offd_i = hypre_CSRBlockMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRBlockMatrixJ(P_offd);

   HYPRE_BigInt  first_col_diag_P = hypre_ParCSRBlockMatrixFirstColDiag(P);
   HYPRE_BigInt  last_col_diag_P;
   HYPRE_Int  num_cols_diag_P = hypre_CSRBlockMatrixNumCols(P_diag);
   HYPRE_Int  num_cols_offd_P = hypre_CSRBlockMatrixNumCols(P_offd);
   HYPRE_BigInt *coarse_partitioning = hypre_ParCSRBlockMatrixColStarts(P);
   HYPRE_BigInt row_starts[2], col_starts[2];

   hypre_ParCSRBlockMatrix *RAP;
   HYPRE_BigInt            *col_map_offd_RAP = NULL;

   hypre_CSRBlockMatrix  *RAP_int = NULL;

   HYPRE_Complex         *RAP_int_data;
   HYPRE_Int             *RAP_int_i;
   HYPRE_BigInt          *RAP_int_j;

   hypre_CSRBlockMatrix  *RAP_ext;

   HYPRE_Complex         *RAP_ext_data  = NULL;
   HYPRE_Int             *RAP_ext_i     = NULL;
   HYPRE_BigInt          *RAP_ext_j     = NULL;

   hypre_CSRBlockMatrix  *RAP_diag;
   HYPRE_Complex         *RAP_diag_data = NULL;
   HYPRE_Int             *RAP_diag_i    = NULL;
   HYPRE_Int             *RAP_diag_j    = NULL;

   hypre_CSRBlockMatrix  *RAP_offd;
   HYPRE_Complex         *RAP_offd_data = NULL;
   HYPRE_Int             *RAP_offd_i    = NULL;
   HYPRE_Int             *RAP_offd_j    = NULL;

   HYPRE_Int              RAP_size;
   HYPRE_Int              RAP_ext_size;
   HYPRE_Int              RAP_diag_size;
   HYPRE_Int              RAP_offd_size;
   HYPRE_Int              P_ext_diag_size;
   HYPRE_Int              P_ext_offd_size;
   HYPRE_BigInt           first_col_diag_RAP;
   HYPRE_BigInt           last_col_diag_RAP;
   HYPRE_Int              num_cols_offd_RAP = 0;

   hypre_CSRBlockMatrix  *R_diag;
   HYPRE_Complex         *R_diag_data = NULL;
   HYPRE_Int             *R_diag_i    = NULL;
   HYPRE_Int             *R_diag_j    = NULL;

   hypre_CSRBlockMatrix  *R_offd;
   HYPRE_Complex         *R_offd_data = NULL;
   HYPRE_Int             *R_offd_i    = NULL;
   HYPRE_Int             *R_offd_j    = NULL;

   hypre_CSRBlockMatrix  *Ps_ext          = NULL;
   HYPRE_Complex         *Ps_ext_data     = NULL;
   HYPRE_Int             *Ps_ext_i        = NULL;
   HYPRE_BigInt          *Ps_ext_j        = NULL;

   HYPRE_Complex         *P_ext_diag_data = NULL;
   HYPRE_Int             *P_ext_diag_i    = NULL;
   HYPRE_Int             *P_ext_diag_j    = NULL;

   HYPRE_Complex         *P_ext_offd_data = NULL;
   HYPRE_Int             *P_ext_offd_i    = NULL;
   HYPRE_Int             *P_ext_offd_j    = NULL;

   HYPRE_BigInt          *col_map_offd_Pext = NULL;
   HYPRE_Int             *map_P_to_Pext   = NULL;
   HYPRE_Int             *map_P_to_RAP    = NULL;
   HYPRE_Int             *map_Pext_to_RAP = NULL;

   HYPRE_Int             *P_marker = NULL;
   HYPRE_Int            **P_mark_array;
   HYPRE_Int            **A_mark_array;
   HYPRE_Int             *A_marker;
   HYPRE_BigInt          *temp = NULL;

   HYPRE_BigInt           n_coarse;
   HYPRE_Int              num_cols_offd_Pext = 0;

   HYPRE_Int              ic, i, j, k, bnnz, kk;
   HYPRE_Int              i1, i2, i3, ii, ns, ne, size, rest;
   HYPRE_Int              cnt, cnt_offd, cnt_diag;
   HYPRE_Int              jj1, jj2, jj3, jcol;
   HYPRE_BigInt           value;

   HYPRE_Int             *jj_count, *jj_cnt_diag, *jj_cnt_offd;
   HYPRE_Int              jj_counter, jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   HYPRE_Int              num_nz_cols_A;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads, ind;

   HYPRE_Complex          *r_entries;
   HYPRE_Complex          *r_a_products;
   HYPRE_Complex          *r_a_p_products;

   HYPRE_Complex          zero = 0.0;

   /*-----------------------------------------------------------------------
    *  Copy ParCSRBlockMatrix RT into CSRBlockMatrix R so that we have
    *  row-wise access to restriction .
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   /* num_threads = hypre_NumThreads(); */
   num_threads = 1;

   bnnz = block_size * block_size;
   r_a_products = hypre_TAlloc(HYPRE_Complex, bnnz, HYPRE_MEMORY_HOST);
   r_a_p_products = hypre_TAlloc(HYPRE_Complex, bnnz, HYPRE_MEMORY_HOST);

   if (comm_pkg_RT)
   {
      num_recvs_RT = hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      num_sends_RT = hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
      send_map_starts_RT = hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      send_map_elmts_RT = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   hypre_CSRBlockMatrixTranspose(RT_diag, &R_diag, 1);
   if (num_cols_offd_RT)
   {
      hypre_CSRBlockMatrixTranspose(RT_offd, &R_offd, 1);
      R_offd_data = hypre_CSRBlockMatrixData(R_offd);
      R_offd_i    = hypre_CSRBlockMatrixI(R_offd);
      R_offd_j    = hypre_CSRBlockMatrixJ(R_offd);
   }

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_diag_data = hypre_CSRBlockMatrixData(R_diag);
   R_diag_i    = hypre_CSRBlockMatrixI(R_diag);
   R_diag_j    = hypre_CSRBlockMatrixJ(R_diag);

   n_coarse = hypre_ParCSRBlockMatrixGlobalNumCols(P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   /*-----------------------------------------------------------------------
    *  Generate Ps_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for triple matrix product
    *-----------------------------------------------------------------------*/

   if (num_procs > 1)
   {
      Ps_ext = hypre_ParCSRBlockMatrixExtractBExt(P, A, 1);
      Ps_ext_data = hypre_CSRBlockMatrixData(Ps_ext);
      Ps_ext_i    = hypre_CSRBlockMatrixI(Ps_ext);
      Ps_ext_j    = hypre_CSRBlockMatrixBigJ(Ps_ext);
   }

   P_ext_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   P_ext_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A + 1, HYPRE_MEMORY_HOST);
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + (HYPRE_BigInt)num_cols_diag_P - 1;

   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
      {
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_size++;
         }
         else
         {
            P_ext_diag_size++;
         }
      }
      P_ext_diag_i[i + 1] = P_ext_diag_size;
      P_ext_offd_i[i + 1] = P_ext_offd_size;
   }

   if (P_ext_diag_size)
   {
      P_ext_diag_j = hypre_CTAlloc(HYPRE_Int,  P_ext_diag_size, HYPRE_MEMORY_HOST);
      P_ext_diag_data = hypre_CTAlloc(HYPRE_Complex,  P_ext_diag_size * bnnz, HYPRE_MEMORY_HOST);
   }
   if (P_ext_offd_size)
   {
      P_ext_offd_j = hypre_CTAlloc(HYPRE_Int,  P_ext_offd_size, HYPRE_MEMORY_HOST);
      P_ext_offd_data = hypre_CTAlloc(HYPRE_Complex,  P_ext_offd_size * bnnz, HYPRE_MEMORY_HOST);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   cnt = 0;
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
      {
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            Ps_ext_j[cnt_offd] = Ps_ext_j[j];
            for (kk = 0; kk < bnnz; kk++)
            {
               P_ext_offd_data[cnt_offd * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
            }
            cnt_offd++;
         }
         else
         {
            P_ext_diag_j[cnt_diag] = (HYPRE_Int)(Ps_ext_j[j] - first_col_diag_P);
            for (kk = 0; kk < bnnz; kk++)
            {
               P_ext_diag_data[cnt_diag * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
            }
            cnt_diag++;
         }
      }
   }
   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = hypre_CTAlloc(HYPRE_BigInt,  P_ext_offd_size + num_cols_offd_P, HYPRE_MEMORY_HOST);
      for (i = 0; i < P_ext_offd_size; i++)
      {
         temp[i] = Ps_ext_j[i];
      }
      cnt = P_ext_offd_size;
      for (i = 0; i < num_cols_offd_P; i++)
      {
         temp[cnt++] = col_map_offd_P[i];
      }
   }
   if (cnt)
   {
      hypre_BigQsort0(temp, 0, cnt - 1);

      num_cols_offd_Pext = 1;
      value = temp[0];
      for (i = 1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_Pext++] = value;
         }
      }
   }

   if (num_cols_offd_Pext)
   {
      col_map_offd_Pext = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_Pext, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_cols_offd_Pext; i++)
   {
      col_map_offd_Pext[i] = temp[i];
   }

   if (P_ext_offd_size || num_cols_offd_P)
   {
      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }

   for (i = 0 ; i < P_ext_offd_size; i++)
   {
      P_ext_offd_j[i] = hypre_BigBinarySearch(col_map_offd_Pext,
                                              Ps_ext_j[i],
                                              num_cols_offd_Pext);
   }

   if (num_cols_offd_P)
   {
      map_P_to_Pext = hypre_CTAlloc(HYPRE_Int, num_cols_offd_P, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_Pext; i++)
         if (col_map_offd_Pext[i] == col_map_offd_P[cnt])
         {
            map_P_to_Pext[cnt++] = i;
            if (cnt == num_cols_offd_P) { break; }
         }
   }

   if (num_procs > 1)
   {
      hypre_CSRBlockMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP_int and set up RAP_int_i if there
    *  are more than one processor and nonzero elements in R_offd
    *-----------------------------------------------------------------------*/

   P_mark_array = hypre_CTAlloc(HYPRE_Int *,  num_threads, HYPRE_MEMORY_HOST);
   A_mark_array = hypre_CTAlloc(HYPRE_Int *,  num_threads, HYPRE_MEMORY_HOST);

   if (num_cols_offd_RT)
   {
      jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

      for (ii = 0; ii < num_threads; ii++)
      {
         size = num_cols_offd_RT / num_threads;
         rest = num_cols_offd_RT - size * num_threads;
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

         /*--------------------------------------------------------------------
          *  Allocate marker arrays.
          *--------------------------------------------------------------------*/

         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_mark_array[ii] = hypre_CTAlloc(HYPRE_Int, num_cols_diag_P + num_cols_offd_Pext,
                                             HYPRE_MEMORY_HOST);
            P_marker = P_mark_array[ii];
         }
         A_mark_array[ii] = hypre_CTAlloc(HYPRE_Int,  num_nz_cols_A, HYPRE_MEMORY_HOST);
         A_marker = A_mark_array[ii];

         /*--------------------------------------------------------------------
          *  Initialize some stuff.
          *--------------------------------------------------------------------*/

         jj_counter = start_indexing;
         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++)
         {
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i++)
         {
            A_marker[i] = -1;
         }

         /*--------------------------------------------------------------------
          *  Loop over exterior c-points
          *--------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {

            jj_row_begining = jj_counter;

            /*-----------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *-----------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *--------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
               }
               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *--------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked
                   * points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *--------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_offd.
                      *--------------------------------------------------------*/

                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
               }
            }
         }
         jj_count[ii] = jj_counter;
      }

      /*-----------------------------------------------------------------------
       *  Allocate RAP_int_data and RAP_int_j arrays.
       *-----------------------------------------------------------------------*/

      for (i = 0; i < num_threads - 1; i++) { jj_count[i + 1] += jj_count[i]; }

      RAP_size = jj_count[num_threads - 1];
      RAP_int_i = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_RT + 1, HYPRE_MEMORY_HOST);
      RAP_int_data = hypre_CTAlloc(HYPRE_Complex,  RAP_size * bnnz, HYPRE_MEMORY_HOST);
      RAP_int_j    = hypre_CTAlloc(HYPRE_BigInt,  RAP_size, HYPRE_MEMORY_HOST);
      RAP_int_i[num_cols_offd_RT] = RAP_size;

      /*-----------------------------------------------------------------------
       *  Second Pass: Fill in RAP_int_data and RAP_int_j.
       *-----------------------------------------------------------------------*/

      for (ii = 0; ii < num_threads; ii++)
      {
         size = num_cols_offd_RT / num_threads;
         rest = num_cols_offd_RT - size * num_threads;
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

         /*--------------------------------------------------------------------
          *  Initialize some stuff.
          *--------------------------------------------------------------------*/

         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_marker = P_mark_array[ii];
         }
         A_marker = A_mark_array[ii];

         jj_counter = start_indexing;
         if (ii > 0) { jj_counter = jj_count[ii - 1]; }

         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++)
         {
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i++)
         {
            A_marker[i] = -1;
         }

         /*--------------------------------------------------------------------
          *  Loop over exterior c-points.
          *--------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {
            jj_row_begining = jj_counter;
            RAP_int_i[ic] = jj_counter;

            /*-----------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *-----------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];
               r_entries = &(R_offd_data[jj1 * bnnz]);

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *--------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_offd_data[jj2 * bnnz]), zero,
                                                   r_a_products, block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[jj_counter * bnnz + kk] =
                                 r_a_p_products[kk];
                           RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[P_marker[i3]*bnnz + kk] +=
                                 r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*--------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *--------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[jj_counter * bnnz + kk] =
                                 r_a_p_products[kk];
                           RAP_int_j[jj_counter]
                              = col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[P_marker[i3]*bnnz + kk] +=
                                 r_a_p_products[kk];
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        for (kk = 0; kk < bnnz; kk++)
                           RAP_int_data[P_marker[i3]*bnnz + kk] +=
                              r_a_p_products[kk];
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
               }

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *--------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];
                  hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_diag_data[jj2 * bnnz]), zero, r_a_products,
                                                   block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *--------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           ind = jj_counter * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_int_j[jj_counter] = (HYPRE_BigInt)i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           ind = jj_counter * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_int_j[jj_counter] =
                              col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
               }
            }
         }
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            hypre_TFree(P_mark_array[ii], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(A_mark_array[ii], HYPRE_MEMORY_HOST);
      }

      RAP_int = hypre_CSRBlockMatrixCreate(block_size, num_cols_offd_RT,
                                           num_rows_offd_RT, RAP_size);
      hypre_CSRBlockMatrixI(RAP_int) = RAP_int_i;
      hypre_CSRBlockMatrixBigJ(RAP_int) = RAP_int_j;
      hypre_CSRBlockMatrixData(RAP_int) = RAP_int_data;
      hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   }

   RAP_ext_size = 0;
   if (num_sends_RT || num_recvs_RT)
   {
      RAP_ext = hypre_ExchangeRAPBlockData(RAP_int, comm_pkg_RT, block_size);
      RAP_ext_i = hypre_CSRBlockMatrixI(RAP_ext);
      RAP_ext_j = hypre_CSRBlockMatrixBigJ(RAP_ext);
      RAP_ext_data = hypre_CSRBlockMatrixData(RAP_ext);
      RAP_ext_size = RAP_ext_i[hypre_CSRBlockMatrixNumRows(RAP_ext)];
   }
   if (num_cols_offd_RT)
   {
      hypre_CSRBlockMatrixDestroy(RAP_int);
      RAP_int = NULL;
   }

   RAP_diag_i = hypre_CTAlloc(HYPRE_Int,  num_cols_diag_P + 1, HYPRE_MEMORY_HOST);
   RAP_offd_i = hypre_CTAlloc(HYPRE_Int,  num_cols_diag_P + 1, HYPRE_MEMORY_HOST);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP  = first_col_diag_P + (HYPRE_BigInt) num_cols_diag_P - 1;

   /*-----------------------------------------------------------------------
    *  check for new nonzero columns in RAP_offd generated through RAP_ext
    *-----------------------------------------------------------------------*/

   if (RAP_ext_size || num_cols_offd_Pext)
   {
      temp = hypre_CTAlloc(HYPRE_BigInt, RAP_ext_size + num_cols_offd_Pext, HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < RAP_ext_size; i++)
      {
         if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP)
         {
            temp[cnt++] = RAP_ext_j[i];
         }
      }
      for (i = 0; i < num_cols_offd_Pext; i++)
      {
         temp[cnt++] = col_map_offd_Pext[i];
      }

      if (cnt)
      {
         hypre_BigQsort0(temp, 0, cnt - 1);
         value = temp[0];
         num_cols_offd_RAP = 1;
         for (i = 1; i < cnt; i++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_RAP++] = value;
            }
         }
      }

      /* now evaluate col_map_offd_RAP */
      if (num_cols_offd_RAP)
      {
         col_map_offd_RAP = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_RAP, HYPRE_MEMORY_HOST);
      }

      for (i = 0 ; i < num_cols_offd_RAP; i++)
      {
         col_map_offd_RAP[i] = temp[i];
      }

      hypre_TFree(temp, HYPRE_MEMORY_HOST);
   }

   if (num_cols_offd_P)
   {
      map_P_to_RAP = hypre_CTAlloc(HYPRE_Int, num_cols_offd_P, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i++)
      {
         if (col_map_offd_RAP[i] == col_map_offd_P[cnt])
         {
            map_P_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_P) { break; }
         }
      }
   }

   if (num_cols_offd_Pext)
   {
      map_Pext_to_RAP = hypre_CTAlloc(HYPRE_Int, num_cols_offd_Pext, HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i++)
      {
         if (col_map_offd_RAP[i] == col_map_offd_Pext[cnt])
         {
            map_Pext_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_Pext) { break; }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Convert RAP_ext column indices
    *-----------------------------------------------------------------------*/

   for (i = 0; i < RAP_ext_size; i++)
   {
      if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP)
      {
         RAP_ext_j[i] = (HYPRE_BigInt)(num_cols_diag_P)
                        + hypre_BigBinarySearch(col_map_offd_RAP,
                                                RAP_ext_j[i], num_cols_offd_RAP);
      }
      else
      {
         RAP_ext_j[i] -= first_col_diag_RAP;
      }
   }

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_cnt_diag = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   jj_cnt_offd = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);

   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_P / num_threads;
      rest = num_cols_diag_P - size * num_threads;
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

      P_mark_array[ii] = hypre_CTAlloc(HYPRE_Int,  num_cols_diag_P + num_cols_offd_RAP,
                                       HYPRE_MEMORY_HOST);
      A_mark_array[ii] = hypre_CTAlloc(HYPRE_Int,  num_nz_cols_A, HYPRE_MEMORY_HOST);
      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;

      for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++)
      {
         P_marker[ic] = -1;
      }
      for (i = 0; i < num_nz_cols_A; i++)
      {
         A_marker[i] = -1;
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/

      for (ic = ns; ic < ne; ic++)
      {

         /*--------------------------------------------------------------------
          *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
          *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
          *--------------------------------------------------------------------*/

         P_marker[ic] = jj_count_diag;
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         jj_count_diag++;

         for (i = 0; i < num_sends_RT; i++)
         {
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
            {
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++)
                  {
                     jcol = (HYPRE_Int)RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
                  break;
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ic of R_diag.
          *-----------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++)
         {
            i1  = R_diag_j[jj1];

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_offd.
             *-----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd++;
                        }
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

               /*--------------------------------------------------------------
                *  Check A_marker to see if point i2 has been previously
                *  visited. New entries in RAP only occur from unmarked points.
                *--------------------------------------------------------------*/

               if (A_marker[i2 + num_cols_offd_A] != ic)
               {

                  /*-----------------------------------------------------------
                   *  Mark i2 as visited.
                   *-----------------------------------------------------------*/

                  A_marker[i2 + num_cols_offd_A] = ic;

                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_diag.
                   *-----------------------------------------------------------*/

                  for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                  {
                     i3 = P_diag_j[jj3];

                     /*--------------------------------------------------------
                      *  Check P_marker to see that RAP_{ic,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        jj_count_diag++;
                     }
                  }

                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_offd.
                   *-----------------------------------------------------------*/

                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
               }
            }
         }

         /*--------------------------------------------------------------------
          * Set RAP_diag_i and RAP_offd_i for this row.
          *--------------------------------------------------------------------*/
      }
      jj_cnt_diag[ii] = jj_count_diag;
      jj_cnt_offd[ii] = jj_count_offd;
   }

   for (i = 0; i < num_threads - 1; i++)
   {
      jj_cnt_diag[i + 1] += jj_cnt_diag[i];
      jj_cnt_offd[i + 1] += jj_cnt_offd[i];
   }

   jj_count_diag = jj_cnt_diag[num_threads - 1];
   jj_count_offd = jj_cnt_offd[num_threads - 1];

   RAP_diag_i[num_cols_diag_P] = jj_count_diag;
   RAP_offd_i[num_cols_diag_P] = jj_count_offd;

   /*-----------------------------------------------------------------------
    *  Allocate RAP_diag_data and RAP_diag_j arrays.
    *  Allocate RAP_offd_data and RAP_offd_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_diag_size = jj_count_diag;
   if (RAP_diag_size)
   {
      RAP_diag_data = hypre_CTAlloc(HYPRE_Complex,  RAP_diag_size * bnnz, HYPRE_MEMORY_HOST);
      RAP_diag_j    = hypre_CTAlloc(HYPRE_Int,  RAP_diag_size, HYPRE_MEMORY_HOST);
   }

   RAP_offd_size = jj_count_offd;
   if (RAP_offd_size)
   {
      RAP_offd_data = hypre_CTAlloc(HYPRE_Complex,  RAP_offd_size * bnnz, HYPRE_MEMORY_HOST);
      RAP_offd_j    = hypre_CTAlloc(HYPRE_Int,  RAP_offd_size, HYPRE_MEMORY_HOST);
   }

   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      hypre_TFree(col_map_offd_RAP, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
    *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
    *-----------------------------------------------------------------------*/

   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_P / num_threads;
      rest = num_cols_diag_P - size * num_threads;
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

      /*-----------------------------------------------------------------------
       *  Initialize some stuff.
       *-----------------------------------------------------------------------*/

      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++)
      {
         P_marker[ic] = -1;
      }
      for (i = 0; i < num_nz_cols_A ; i++)
      {
         A_marker[i] = -1;
      }

      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;
      if (ii > 0)
      {
         jj_count_diag = jj_cnt_diag[ii - 1];
         jj_count_offd = jj_cnt_offd[ii - 1];
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/

      for (ic = ns; ic < ne; ic++)
      {
         /*--------------------------------------------------------------------
          *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext
          *--------------------------------------------------------------------*/

         P_marker[ic] = jj_count_diag;
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         RAP_diag_i[ic] = jj_row_begin_diag;
         RAP_offd_i[ic] = jj_row_begin_offd;
         ind = jj_count_diag * bnnz;
         for (kk = 0; kk < bnnz; kk++)
         {
            RAP_diag_data[ind++] = zero;
         }
         RAP_diag_j[jj_count_diag] = ic;
         jj_count_diag++;

         for (i = 0; i < num_sends_RT; i++)
         {
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
            {
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++)
                  {
                     jcol = (HYPRE_Int) RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           ind = jj_count_diag * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] = RAP_ext_data[k * bnnz + kk];
                           }
                           RAP_diag_j[jj_count_diag] = jcol;
                           jj_count_diag++;
                        }
                        else
                        {
                           ind = P_marker[jcol] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] += RAP_ext_data[k * bnnz + kk];
                           }
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = RAP_ext_data[k * bnnz + kk];
                           }
                           RAP_offd_j[jj_count_offd]
                              = jcol - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[jcol] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += RAP_ext_data[k * bnnz + kk];
                           }
                        }
                     }
                  }
                  break;
               }
            }
         }

         /*--------------------------------------------------------------------
          *  Loop over entries in row ic of R_diag.
          *--------------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++)
         {
            i1  = R_diag_j[jj1];
            r_entries = &(R_diag_data[jj1 * bnnz]);

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_offd.
             *-----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_offd_data[jj2 * bnnz]), zero, r_a_products,
                                                   block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           ind = jj_count_diag * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_diag_j[jj_count_diag] = i3;
                           jj_count_diag++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/
                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/
                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] += r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_offd_data[ind++] += r_a_p_products[kk];
                        }
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
               hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                &(A_diag_data[jj2 * bnnz]),
                                                zero, r_a_products, block_size);

               /*--------------------------------------------------------------
                *  Check A_marker to see if point i2 has been previously
                *  visited. New entries in RAP only occur from unmarked points.
                *--------------------------------------------------------------*/

               if (A_marker[i2 + num_cols_offd_A] != ic)
               {

                  /*-----------------------------------------------------------
                   *  Mark i2 as visited.
                   *-----------------------------------------------------------*/

                  A_marker[i2 + num_cols_offd_A] = ic;

                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_diag.
                   *-----------------------------------------------------------*/

                  for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                  {
                     i3 = P_diag_j[jj3];
                     hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                      &(P_diag_data[jj3 * bnnz]),
                                                      zero, r_a_p_products, block_size);

                     /*--------------------------------------------------------
                      *  Check P_marker to see that RAP_{ic,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        ind = jj_count_diag * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] = r_a_p_products[kk];
                        }
                        RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                        jj_count_diag++;
                     }
                     else
                     {
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                  }
               }

               /*--------------------------------------------------------------
                *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                *  no new entries in RAP and can just add new contributions.
                *--------------------------------------------------------------*/

               else
               {
                  for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                  {
                     i3 = P_diag_j[jj3];
                     hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                      &(P_diag_data[jj3 * bnnz]),
                                                      zero, r_a_p_products, block_size);
                     ind = P_marker[i3] * bnnz;
                     for (kk = 0; kk < bnnz; kk++)
                     {
                        RAP_diag_data[ind++] += r_a_p_products[kk];
                     }
                  }
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_offd_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
               }
            }
         }
      }
      hypre_TFree(P_mark_array[ii], HYPRE_MEMORY_HOST);
      hypre_TFree(A_mark_array[ii], HYPRE_MEMORY_HOST);
   }


   for (i = 0; i < 2; i++)
   {
      row_starts[i] = col_starts[i] = coarse_partitioning[i];
   }

   RAP = hypre_ParCSRBlockMatrixCreate(comm, block_size, n_coarse, n_coarse,
                                       row_starts, col_starts,
                                       num_cols_offd_RAP, RAP_diag_size, RAP_offd_size);

   RAP_diag = hypre_ParCSRBlockMatrixDiag(RAP);
   hypre_CSRBlockMatrixI(RAP_diag) = RAP_diag_i;
   if (RAP_diag_size)
   {
      hypre_CSRBlockMatrixData(RAP_diag) = RAP_diag_data;
      hypre_CSRBlockMatrixJ(RAP_diag) = RAP_diag_j;
   }

   RAP_offd = hypre_ParCSRBlockMatrixOffd(RAP);
   hypre_CSRBlockMatrixI(RAP_offd) = RAP_offd_i;
   if (num_cols_offd_RAP)
   {
      hypre_CSRBlockMatrixData(RAP_offd) = RAP_offd_data;
      hypre_CSRBlockMatrixJ(RAP_offd) = RAP_offd_j;
      hypre_ParCSRBlockMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
   if (num_procs > 1)
   {
      hypre_BlockMatvecCommPkgCreate(RAP);
   }

   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R, P_ext and marker arrays.
    *-----------------------------------------------------------------------*/

   hypre_CSRBlockMatrixDestroy(R_diag);
   R_diag = NULL;

   if (num_cols_offd_RT)
   {
      hypre_CSRBlockMatrixDestroy(R_offd);
      R_offd = NULL;
   }

   if (num_sends_RT || num_recvs_RT)
   {
      hypre_CSRBlockMatrixDestroy(RAP_ext);
      RAP_ext = NULL;
   }
   hypre_TFree(P_mark_array, HYPRE_MEMORY_HOST);
   hypre_TFree(A_mark_array, HYPRE_MEMORY_HOST);
   hypre_TFree(P_ext_diag_i, HYPRE_MEMORY_HOST);
   hypre_TFree(P_ext_offd_i, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_cnt_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_cnt_offd, HYPRE_MEMORY_HOST);
   if (num_cols_offd_P)
   {
      hypre_TFree(map_P_to_Pext, HYPRE_MEMORY_HOST);
      hypre_TFree(map_P_to_RAP, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_Pext)
   {
      hypre_TFree(col_map_offd_Pext, HYPRE_MEMORY_HOST);
      hypre_TFree(map_Pext_to_RAP, HYPRE_MEMORY_HOST);
   }
   if (P_ext_diag_size)
   {
      hypre_TFree(P_ext_diag_data, HYPRE_MEMORY_HOST);
      hypre_TFree(P_ext_diag_j, HYPRE_MEMORY_HOST);
   }
   if (P_ext_offd_size)
   {
      hypre_TFree(P_ext_offd_data, HYPRE_MEMORY_HOST);
      hypre_TFree(P_ext_offd_j, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(r_a_products, HYPRE_MEMORY_HOST);
   hypre_TFree(r_a_p_products, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
