/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_lapack.h"
#include "_hypre_blas.h"


// TODO : delete csrAi, csrAi_i, csrAi_j,
//             csrAi_a, csrAiT_i, csrAiT_j, csrAiT_a
//    Use
//       hypre_dense_topo_sort(HYPRE_Real *L, HYPRE_Int *ordering, HYPRE_Int n)
//    to get ordering for triangular solve. Can provide


HYPRE_Int AIR_TOT_SOL_SIZE = 0;
HYPRE_Int AIR_MAX_SOL_SIZE = 0;

#define AIR_DEBUG 0
#define EPSILON 1e-18
#define EPSIMAC 1e-16

void hypre_fgmresT(HYPRE_Int n, HYPRE_Complex *A, HYPRE_Complex *b, HYPRE_Real tol, HYPRE_Int kdim,
                   HYPRE_Complex *x, HYPRE_Real *relres, HYPRE_Int *iter, HYPRE_Int job);
void hypre_ordered_GS(const HYPRE_Complex L[], const HYPRE_Complex rhs[], HYPRE_Complex x[],
                      const HYPRE_Int n);

/*
HYPRE_Real air_time0 = 0.0;
HYPRE_Real air_time_comm = 0.0;
HYPRE_Real air_time1 = 0.0;
HYPRE_Real air_time2 = 0.0;
HYPRE_Real air_time3 = 0.0;
HYPRE_Real air_time4 = 0.0;
*/

HYPRE_Int
hypre_BoomerAMGBuildRestrDist2AIR( hypre_ParCSRMatrix   *A,
                                   HYPRE_Int            *CF_marker,
                                   hypre_ParCSRMatrix   *S,
                                   HYPRE_BigInt         *num_cpts_global,
                                   HYPRE_Int             num_functions,
                                   HYPRE_Int            *dof_func,
                                   HYPRE_Real            filter_thresholdR,
                                   HYPRE_Int             debug_flag,
                                   hypre_ParCSRMatrix  **R_ptr,
                                   HYPRE_Int             AIR1_5,
                                   HYPRE_Int             is_triangular,
                                   HYPRE_Int             gmres_switch)
{
   HYPRE_UNUSED_VAR(debug_flag);
   /* HYPRE_Real t0 = hypre_MPI_Wtime(); */

   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_ParCSRCommPkg     *comm_pkg_SF = NULL;

   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt    *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   hypre_CSRMatrix *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   hypre_CSRMatrix *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   hypre_ParCSRMatrix *R;
   /* csr's */
   hypre_CSRMatrix *R_diag;
   hypre_CSRMatrix *R_offd;
   /* arrays */
   HYPRE_Complex   *R_diag_data;
   HYPRE_Int       *R_diag_i;
   HYPRE_Int       *R_diag_j;
   HYPRE_Complex   *R_offd_data;
   HYPRE_Int       *R_offd_i;
   HYPRE_Int       *R_offd_j;
   HYPRE_BigInt    *col_map_offd_R;
   HYPRE_Int       *tmp_map_offd = NULL;
   /* CF marker off-diag part */
   HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   HYPRE_Int       *dof_func_offd  = NULL;

   HYPRE_BigInt     big_i1, big_j1, big_k1;
   HYPRE_Int        i, j, j1, j2, k, i1, i2, k1, k2, k3, rr, cc, ic, index, start, end,
                    local_max_size, local_size, num_cols_offd_R;
   /*HYPRE_Int        i6;*/
   HYPRE_BigInt     *FF2_offd;
   HYPRE_Int        FF2_offd_len;

   /* LAPACK */
   HYPRE_Complex *DAi, *Dbi, *Dxi;
#if AIR_DEBUG
   HYPRE_Complex *TMPA, *TMPb, *TMPd;
   hypre_Vector *tmpv;
#endif
   HYPRE_Int *Ipi, lapack_info, ione = 1, *RRi, *KKi;
   char charT = 'T';

   /* if the size of local system is larger than gmres_switch, use GMRES */
   char Aisol_method;
   HYPRE_Int gmresAi_maxit = 50;
   HYPRE_Real gmresAi_tol = 1e-3;

   HYPRE_Int my_id, num_procs;
   HYPRE_BigInt total_global_cpts/*, my_first_cpt*/;
   HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_Int *Marker_diag, *Marker_offd;
   HYPRE_Int *Marker_diag_j, Marker_diag_count;
   HYPRE_Int num_sends, num_recvs, num_elems_send;
   /* local size, local num of C points */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int n_cpts = 0;
   /* my column range */
   HYPRE_BigInt col_start = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt col_end   = col_start + (HYPRE_BigInt)n_fine;

   HYPRE_Int  *send_buf_i;

   /* recv_SF means the Strong F-neighbors of offd elements in col_map_offd */
   HYPRE_Int *send_SF_i, send_SF_jlen;
   HYPRE_BigInt *send_SF_j;
   HYPRE_BigInt *recv_SF_j;
   HYPRE_Int *recv_SF_i, *recv_SF_j2, recv_SF_jlen;
   HYPRE_Int *send_SF_jstarts, *recv_SF_jstarts;
   HYPRE_BigInt *recv_SF_offd_list;
   HYPRE_Int recv_SF_offd_list_len;
   HYPRE_Int *Mapper_recv_SF_offd_list, *Mapper_offd_A, *Marker_recv_SF_offd_list;
   HYPRE_Int *Marker_FF2_offd;
   HYPRE_Int *Marker_FF2_offd_j, Marker_FF2_offd_count;

   /* for communication of offd F and F^2 rows of A */
   hypre_ParCSRCommPkg *comm_pkg_FF2_i, *comm_pkg_FF2_j = NULL;
   HYPRE_BigInt *send_FF2_j, *recv_FF2_j;
   HYPRE_Int num_sends_FF2, *send_FF2_i, send_FF2_ilen, send_FF2_jlen,
             num_recvs_FF2, *recv_FF2_i, recv_FF2_ilen, recv_FF2_jlen,
             *send_FF2_jstarts, *recv_FF2_jstarts;
   HYPRE_Complex *send_FF2_a, *recv_FF2_a;

   /* ghost rows: offd F and F2-pts */
   hypre_CSRMatrix *A_offd_FF2   = NULL;

   /*
   HYPRE_Real tcomm = hypre_MPI_Wtime();
   */

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* init markers to zeros */
   Marker_diag = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   Marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);

   /* number of sends (number of procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   /* number of recvs (number of procs) */
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   /* number of elements to send */
   num_elems_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   send_buf_i = hypre_CTAlloc(HYPRE_Int, num_elems_send, HYPRE_MEMORY_HOST);

   /* copy CF markers of elements to send to buffer
    * RL: why copy them with two for loops? Why not just loop through all in one */
   for (i = 0, index = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         send_buf_i[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, CF_marker_offd);
   /* destroy the handle to finish communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      for (i = 0, index = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            send_buf_i[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *        Send/Recv Offd F-neighbors' strong F-neighbors
    *        F^2: OffdF - F
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   send_SF_i = hypre_CTAlloc(HYPRE_Int, num_elems_send, HYPRE_MEMORY_HOST);
   recv_SF_i = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd + 1, HYPRE_MEMORY_HOST);

   /* for each F-elem to send, find the number of strong F-neighbors */
   for (i = 0, send_SF_jlen = 0; i < num_elems_send; i++)
   {
      /* number of strong F-pts */
      send_SF_i[i] = 0;
      /* elem i1 */
      i1 = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      /* ignore C-pts */
      if (CF_marker[i1] >= 0)
      {
         continue;
      }
      /* diag part of row i1 */
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         if (CF_marker[S_diag_j[j]] < 0)
         {
            send_SF_i[i] ++;
         }
      }
      /* offd part of row i1 */
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (CF_marker_offd[j1] < 0)
         {
            send_SF_i[i] ++;
         }
      }

      /* add to the num of elems going to be sent */
      send_SF_jlen += send_SF_i[i];
   }

   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_SF_i, recv_SF_i + 1);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   send_SF_j = hypre_CTAlloc(HYPRE_BigInt, send_SF_jlen, HYPRE_MEMORY_HOST);
   send_SF_jstarts = hypre_CTAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);

   for (i = 0, i1 = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* 1-past-the-end pos */
      end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

      for (j = start; j < end; j++)
      {
         /* strong F-pt, j1 */
         j1 = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         /* ignore C-pts */
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* diag part of row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_SF_j[i1++] = col_start + (HYPRE_BigInt)k1;
            }
         }
         /* offd part of row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];
            if (CF_marker_offd[k1] < 0)
            {
               send_SF_j[i1++] = col_map_offd_A[k1];
            }
         }
      }
      send_SF_jstarts[i + 1] = i1;
   }

   hypre_assert(i1 == send_SF_jlen);

   /* adjust recv_SF_i to ptrs */
   for (i = 1; i <= num_cols_A_offd; i++)
   {
      recv_SF_i[i] += recv_SF_i[i - 1];
   }

   recv_SF_jlen = recv_SF_i[num_cols_A_offd];
   recv_SF_j = hypre_CTAlloc(HYPRE_BigInt, recv_SF_jlen, HYPRE_MEMORY_HOST);
   recv_SF_jstarts = hypre_CTAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs; i++)
   {
      start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_SF_jstarts[i] = recv_SF_i[start];
   }

   /* create a communication package for SF_j */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_SF_jstarts,
                                    num_sends,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_SF_jstarts,
                                    NULL,
                                    &comm_pkg_SF);

   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg_SF, send_SF_j, recv_SF_j);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * recv_SF_offd_list: a sorted list of offd elems in recv_SF_j
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   recv_SF_offd_list = hypre_CTAlloc(HYPRE_BigInt, recv_SF_jlen, HYPRE_MEMORY_HOST);
   for (i = 0, j = 0; i < recv_SF_jlen; i++)
   {
      HYPRE_Int flag = 1;
      big_i1 = recv_SF_j[i];
      /* offd */
      if (big_i1 < col_start || big_i1 >= col_end)
      {
         if (AIR1_5)
         {
            flag = hypre_BigBinarySearch(col_map_offd_A, big_i1, num_cols_A_offd) != -1;
         }
         if (flag)
         {
            recv_SF_offd_list[j++] = big_i1;
         }
      }
   }

   /* remove redundancy after sorting */
   hypre_BigQsort0(recv_SF_offd_list, 0, j - 1);

   for (i = 0, recv_SF_offd_list_len = 0; i < j; i++)
   {
      if (i == 0 || recv_SF_offd_list[i] != recv_SF_offd_list[i - 1])
      {
         recv_SF_offd_list[recv_SF_offd_list_len++] = recv_SF_offd_list[i];
      }
   }

   /* make a copy of recv_SF_j in which
    * adjust the offd indices corresponding to recv_SF_offd_list */
   recv_SF_j2 = hypre_CTAlloc(HYPRE_Int, recv_SF_jlen, HYPRE_MEMORY_HOST);
   for (i = 0; i < recv_SF_jlen; i++)
   {
      big_i1 = recv_SF_j[i];
      if (big_i1 < col_start || big_i1 >= col_end)
      {
         j = hypre_BigBinarySearch(recv_SF_offd_list, big_i1, recv_SF_offd_list_len);
         if (!AIR1_5)
         {
            hypre_assert(j >= 0 && j < recv_SF_offd_list_len);
         }
         recv_SF_j2[i] = j;
      }
      else
      {
         recv_SF_j2[i] = -1;
      }
   }

   /* mapping to col_map_offd_A */
   Mapper_recv_SF_offd_list = hypre_CTAlloc(HYPRE_Int, recv_SF_offd_list_len, HYPRE_MEMORY_HOST);
   Marker_recv_SF_offd_list = hypre_CTAlloc(HYPRE_Int, recv_SF_offd_list_len, HYPRE_MEMORY_HOST);

   /* create a mapping from recv_SF_offd_list to col_map_offd_A for their intersections */
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      big_i1 = recv_SF_offd_list[i];
      hypre_assert(big_i1 < col_start || big_i1 >= col_end);
      j = hypre_BigBinarySearch(col_map_offd_A, big_i1, num_cols_A_offd);
      /* mapping to col_map_offd_A, if not found equal to -1 */
      Mapper_recv_SF_offd_list[i] = j;
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *       Find offd F and F-F (F^2) neighboring points for C-pts
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   for (i = 0, FF2_offd_len = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* diag(F)-offd(F) */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         /* if it is F */
         if (CF_marker[j1] < 0)
         {
            /* go through its offd part */
            for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
            {
               k1 = S_offd_j[k];
               if (CF_marker_offd[k1] < 0)
               {
                  /* mark F pts */
                  if (!Marker_offd[k1])
                  {
                     FF2_offd_len ++;
                     Marker_offd[k1] = 1;
                  }
               }
            }
         }
      }

      /* offd(F) and offd(F)-offd(F)
       * NOTE: we are working with two marker arrays here: Marker_offd and Marker_recv_SF_offd_list
       * which may have overlap.
       * So, we always check the first marker array */
      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         j1 = S_offd_j[j];
         /* offd F pts */
         if (CF_marker_offd[j1] < 0)
         {
            if (!Marker_offd[j1])
            {
               FF2_offd_len ++;
               Marker_offd[j1] = 1;
            }
            /* offd(F)-offd(F), need to open recv_SF */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
            {
               /* k1: global index */
               big_k1 = recv_SF_j[k];
               /* if k1 is not in my range */
               if (big_k1 < col_start || big_k1 >= col_end)
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  if (AIR1_5 && k2 == -1)
                  {
                     continue;
                  }

                  hypre_assert(recv_SF_offd_list[k2] == big_k1);

                  /* map to offd_A */
                  k3 = Mapper_recv_SF_offd_list[k2];
                  if (k3 >= 0)
                  {
                     if (!Marker_offd[k3])
                     {
                        FF2_offd_len ++;
                        Marker_offd[k3] = 1;
                     }
                  }
                  else
                  {
                     if (!Marker_recv_SF_offd_list[k2])
                     {
                        FF2_offd_len ++;
                        Marker_recv_SF_offd_list[k2] = 1;
                     }
                  }
               }
            }
         }
      }
   }

   /* create a list of offd F, F2 points
    * and RESET the markers to ZEROs*/
   FF2_offd = hypre_CTAlloc(HYPRE_BigInt, FF2_offd_len, HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (Marker_offd[i])
      {
         FF2_offd[k++] = col_map_offd_A[i];
         Marker_offd[i] = 0;
      }
   }

   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      /* debug: if mapping exists, this marker should not be set */
      if (Mapper_recv_SF_offd_list[i] >= 0)
      {
         hypre_assert(Marker_recv_SF_offd_list[i] == 0);
      }

      if (Marker_recv_SF_offd_list[i])
      {
         big_i1 = recv_SF_offd_list[i];
         hypre_assert(big_i1 < col_start || big_i1 >= col_end);
         FF2_offd[k++] = big_i1;
         Marker_recv_SF_offd_list[i] = 0;
      }
   }
   hypre_assert(k == FF2_offd_len);

   /* sort the list */
   hypre_BigQsort0(FF2_offd, 0, FF2_offd_len - 1);

   /* there must be no repetition in FF2_offd */
   for (i = 1; i < FF2_offd_len; i++)
   {
      hypre_assert(FF2_offd[i] != FF2_offd[i - 1]);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *    Create CommPkgs for exchanging offd F and F2 rows of A
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* we will create TWO commPkg: one for row lengths and one for row data,
    * similar to what we have done above for SF_i, SF_j */
   hypre_ParCSRFindExtendCommPkg(comm,
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixFirstColDiag(A),
                                 hypre_CSRMatrixNumCols(A_diag),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_ParCSRMatrixAssumedPartition(A),
                                 FF2_offd_len,
                                 FF2_offd,
                                 &comm_pkg_FF2_i);
   /* number of sends (#procs) */
   num_sends_FF2 = hypre_ParCSRCommPkgNumSends(comm_pkg_FF2_i);
   /* number of rows to send */
   send_FF2_ilen = hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, num_sends_FF2);
   /* number of recvs (#procs) */
   num_recvs_FF2 = hypre_ParCSRCommPkgNumRecvs(comm_pkg_FF2_i);
   /* number of rows to recv */
   recv_FF2_ilen = hypre_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, num_recvs_FF2);

   hypre_assert(FF2_offd_len == recv_FF2_ilen);

   send_FF2_i = hypre_CTAlloc(HYPRE_Int, send_FF2_ilen, HYPRE_MEMORY_HOST);
   recv_FF2_i = hypre_CTAlloc(HYPRE_Int, recv_FF2_ilen + 1, HYPRE_MEMORY_HOST);
   for (i = 0, send_FF2_jlen = 0; i < send_FF2_ilen; i++)
   {
      j = hypre_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, i);
      for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
      {
         if (CF_marker[A_diag_j[k]] < 0)
         {
            send_FF2_i[i]++;
         }
      }
      if (num_procs > 1)
      {
         for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
         {
            if (CF_marker_offd[A_offd_j[k]] < 0)
            {
               send_FF2_i[i]++;
            }
         }
      }
      //send_FF2_i[i] = A_diag_i[j+1] - A_diag_i[j] + A_offd_i[j+1] - A_offd_i[j];
      send_FF2_jlen += send_FF2_i[i];
   }

   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg_FF2_i, send_FF2_i, recv_FF2_i + 1);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   send_FF2_j = hypre_CTAlloc(HYPRE_BigInt, send_FF2_jlen, HYPRE_MEMORY_HOST);
   send_FF2_a = hypre_CTAlloc(HYPRE_Complex, send_FF2_jlen, HYPRE_MEMORY_HOST);
   send_FF2_jstarts = hypre_CTAlloc(HYPRE_Int, num_sends_FF2 + 1, HYPRE_MEMORY_HOST);

   for (i = 0, i1 = 0; i < num_sends_FF2; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i);
      end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i + 1);
      for (j = start; j < end; j++)
      {
         /* will send row j1 to send_proc[i] */
         j1 = hypre_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, j);
         /* open row j1 and fill ja and a */
         for (k = A_diag_i[j1]; k < A_diag_i[j1 + 1]; k++)
         {
            HYPRE_Int k1 = A_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_FF2_j[i1] = col_start + k1;
               send_FF2_a[i1] = A_diag_a[k];
               i1++;
            }
         }
         if (num_procs > 1)
         {
            for (k = A_offd_i[j1]; k < A_offd_i[j1 + 1]; k++)
            {
               HYPRE_Int k1 = A_offd_j[k];
               if (CF_marker_offd[k1] < 0)
               {
                  send_FF2_j[i1] = col_map_offd_A[k1];
                  send_FF2_a[i1] = A_offd_a[k];
                  i1++;
               }
            }
         }
      }
      send_FF2_jstarts[i + 1] = i1;
   }
   hypre_assert(i1 == send_FF2_jlen);

   /* adjust recv_FF2_i to ptrs */
   for (i = 1; i <= recv_FF2_ilen; i++)
   {
      recv_FF2_i[i] += recv_FF2_i[i - 1];
   }

   recv_FF2_jlen = recv_FF2_i[recv_FF2_ilen];
   recv_FF2_j = hypre_CTAlloc(HYPRE_BigInt, recv_FF2_jlen, HYPRE_MEMORY_HOST);
   recv_FF2_a = hypre_CTAlloc(HYPRE_Complex, recv_FF2_jlen, HYPRE_MEMORY_HOST);
   recv_FF2_jstarts = hypre_CTAlloc(HYPRE_Int, num_recvs_FF2 + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs_FF2; i++)
   {
      start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, i);
      recv_FF2_jstarts[i] = recv_FF2_i[start];
   }

   /* create a communication package for FF2_j */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_FF2,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg_FF2_i),
                                    recv_FF2_jstarts,
                                    num_sends_FF2,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg_FF2_i),
                                    send_FF2_jstarts,
                                    NULL,
                                    &comm_pkg_FF2_j);

   /* do communication */
   /* ja */
   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg_FF2_j, send_FF2_j, recv_FF2_j);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* a */
   comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg_FF2_j, send_FF2_a, recv_FF2_a);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd_FF2 is ready ! */
   /* Careful! Wrong data type for number of columns ! */
   //A_offd_FF2 = hypre_CSRMatrixCreate(recv_FF2_ilen, hypre_ParCSRMatrixGlobalNumCols(A),
   /* Careful! Wrong column size! Hopefully won't matter! */
   A_offd_FF2 = hypre_CSRMatrixCreate(recv_FF2_ilen, recv_FF2_ilen,
                                      recv_FF2_jlen);

   hypre_CSRMatrixI   (A_offd_FF2) = recv_FF2_i;
   hypre_CSRMatrixBigJ (A_offd_FF2) = recv_FF2_j;
   hypre_CSRMatrixData(A_offd_FF2) = recv_FF2_a;

   /*
   for (i6 = 0; i6 < num_procs; i6 ++)
   {
      if (i6 == my_id)
      {
         hypre_assert(hypre_CSRMatrixNumNonzeros(A_offd_FF2) == \
                      hypre_CSRMatrixI(A_offd_FF2)[hypre_CSRMatrixNumRows(A_offd_FF2)]);

         for (i = 0; i < hypre_CSRMatrixNumRows(A_offd_FF2); i++)
         {
            for (j = hypre_CSRMatrixI(A_offd_FF2)[i]; j < hypre_CSRMatrixI(A_offd_FF2)[i+1]; j++)
            {
               HYPRE_Int r = FF2_offd[i];
               HYPRE_Int c = hypre_CSRMatrixJ(A_offd_FF2)[j];
               hypre_assert(c >= 0 && c < hypre_CSRMatrixNumCols(A_offd_FF2));
               HYPRE_Complex v = hypre_CSRMatrixData(A_offd_FF2)[j];
               hypre_printf("%8d %8d     % e\n", r, c, v);
            }
         }
         hypre_printf("\n\n");
      }
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   }
   */

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * FF2_offd contains all the offd indices and corresponds to matrix A_offd_FF2
    * So, we are able to use indices in terms of FF2_offd to bookkeeping all offd
    * information.
    * [ FF2_offd is a subset of col_map_offd_A UNION recv_SF_offd_list ]
    * Mappings from col_map_offd_A and recv_SF_offd_list will be created
    * markers for FF2_offd will also be created
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /* Mapping from col_map_offd_A */
   Mapper_offd_A = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd; i++)
   {
      Mapper_offd_A[i] = hypre_BigBinarySearch(FF2_offd, col_map_offd_A[i], FF2_offd_len);
   }

   /* Mapping from recv_SF_offd_list, overwrite the old one*/
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      Mapper_recv_SF_offd_list[i] = hypre_BigBinarySearch(FF2_offd, recv_SF_offd_list[i], FF2_offd_len);
   }

   /* marker */
   Marker_FF2_offd = hypre_CTAlloc(HYPRE_Int, FF2_offd_len, HYPRE_MEMORY_HOST);

   /*
   tcomm = hypre_MPI_Wtime() - tcomm;
   air_time_comm += tcomm;

   HYPRE_Real t1 = hypre_MPI_Wtime();
   */

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *  First Pass: Determine the nnz of R and the max local size
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;

   for (i = 0; i < n_fine; i++)
   {
      HYPRE_Int MARK = i + 1;

      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of the local dense problem */
      local_size = 0;

      /* i is a C-pt, increase the number of C-pts */
      n_cpts ++;

      /* diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] != MARK)
         {
            Marker_diag[j1] = MARK;
            local_size ++;
            cnt_diag ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] != MARK)
            {
               Marker_diag[k1] = MARK;
               local_size ++;
               cnt_diag ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];

            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];

               /* this mapping must be successful */
               hypre_assert(k2 >= 0 && k2 < FF2_offd_len);

               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] != MARK)
               {
                  Marker_FF2_offd[k2] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* offd part of row i */
      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         j1 = S_offd_j[j];

         if (CF_marker_offd[j1] >= 0)
         {
            continue;
         }

         /* map to FF2_offd */
         j2 = Mapper_offd_A[j1];

         /* this mapping must be successful */
         hypre_assert(j2 >= 0 && j2 < FF2_offd_len);

         /* j1, F: O1 */
         if (Marker_FF2_offd[j2] != MARK)
         {
            Marker_FF2_offd[j2] = MARK;
            local_size ++;
            cnt_offd ++;
         }

         /* F^2: O1-D2, O1-O2 */
         /* row j1 is an external row. check recv_SF for strong F-neighbors  */
         for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
         {
            /* k1: global index */
            big_k1 = recv_SF_j[k];
            /* if big_k1 is in the diag part */
            if (big_k1 >= col_start && big_k1 < col_end)
            {
               k3 = (HYPRE_Int)(big_k1 - col_start);
               hypre_assert(CF_marker[k3] < 0);
               if (Marker_diag[k3] != MARK)
               {
                  Marker_diag[k3] = MARK;
                  local_size ++;
                  cnt_diag ++;
               }
            }
            else /* k1 is in the offd part */
            {
               /* index in recv_SF_offd_list */
               k2 = recv_SF_j2[k];

               if (AIR1_5 && k2 == -1)
               {
                  continue;
               }

               hypre_assert(recv_SF_offd_list[k2] == big_k1);

               /* map to FF2_offd */
               k3 = Mapper_recv_SF_offd_list[k2];

               /* this mapping must be successful */
               hypre_assert(k3 >= 0 && k3 < FF2_offd_len);

               if (Marker_FF2_offd[k3] != MARK)
               {
                  Marker_FF2_offd[k3] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* keep ths max size */
      local_max_size = hypre_max(local_max_size, local_size);
   } /* for (i=0,...) */

   /*
   t1 = hypre_MPI_Wtime() - t1;
   air_time1 += t1;
   */

   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;

   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   R_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, HYPRE_MEMORY_HOST);
   R_diag_j    = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   R_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_diag, HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, HYPRE_MEMORY_HOST);
   R_offd_j    = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   R_offd_data = hypre_CTAlloc(HYPRE_Complex, nnz_offd, HYPRE_MEMORY_HOST);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   /* RESET marker arrays */
   for (i = 0; i < n_fine; i++)
   {
      Marker_diag[i] = -1;
   }
   Marker_diag_j = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < FF2_offd_len; i++)
   {
      Marker_FF2_offd[i] = -1;
   }
   Marker_FF2_offd_j = hypre_CTAlloc(HYPRE_Int, FF2_offd_len, HYPRE_MEMORY_HOST);

   // TODO bs : what is this for? Should we remove?
   //for (i = 0; i < num_cols_A_offd; i++)
   //{
   //   Marker_offd[i] = -1;
   //}
   //for (i = 0; i < recv_SF_offd_list_len; i++)
   //{
   //   Marker_recv_SF_list[i] = -1;
   //}
   //printf("AIR: max local dense solve size %d\n", local_max_size);

   // Allocate the rhs and dense local matrix in column-major form (for LAPACK)
   DAi = hypre_CTAlloc(HYPRE_Complex, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   Dbi = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   Dxi = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   Ipi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST); // pivot matrix

   // Allocate memory for GMRES if it will be used
   HYPRE_Int kdim_max = hypre_min(gmresAi_maxit, local_max_size);
   if (gmres_switch < local_max_size)
   {
      hypre_fgmresT(local_max_size, NULL, NULL, 0.0, kdim_max, NULL, NULL, NULL, -1);
   }

#if AIR_DEBUG
   /* FOR DEBUG */
   TMPA = hypre_CTAlloc(HYPRE_Complex, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   TMPb = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
   TMPd = hypre_CTAlloc(HYPRE_Complex, local_max_size, HYPRE_MEMORY_HOST);
#endif

   /*- - - - - - - - - - - - - - - - - - - - - - - - -
    * space to save row indices of the local problem,
    * if diag, save the local indices,
    * if offd, save the indices in FF2_offd,
    *          since we will use it to access A_offd_FF2
    *- - - - - - - - - - - - - - - - - - - - - - - - - */
   RRi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST);
   /* indicators for RRi of being local (0) or offd (1) */
   KKi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST);

   /*
   HYPRE_Real t2 = hypre_MPI_Wtime();
   */

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *                        Second Pass: Populate R
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      Marker_diag_count = 0;
      Marker_FF2_offd_count = 0;

      /* size of Ai, bi */
      local_size = 0;

      /* Access matrices for the First time, mark the points we want */
      /* diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] == -1)
         {
            RRi[local_size] = j1;
            KKi[local_size] = 0;
            Marker_diag_j[Marker_diag_count++] = j1;
            Marker_diag[j1] = local_size ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] == -1)
            {
               RRi[local_size] = k1;
               KKi[local_size] = 0;
               Marker_diag_j[Marker_diag_count++] = k1;
               Marker_diag[k1] = local_size ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];

            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];

               /* this mapping must be successful */
               hypre_assert(k2 >= 0 && k2 < FF2_offd_len);

               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] == -1)
               {
                  /* NOTE: we save this mapped index */
                  RRi[local_size] = k2;
                  KKi[local_size] = 1;
                  Marker_FF2_offd_j[Marker_FF2_offd_count++] = k2;
                  Marker_FF2_offd[k2] = local_size ++;
               }
            }
         }
      }

      /* offd part of row i */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            j1 = S_offd_j[j];

            if (CF_marker_offd[j1] >= 0)
            {
               continue;
            }

            /* map to FF2_offd */
            j2 = Mapper_offd_A[j1];

            /* this mapping must be successful */
            hypre_assert(j2 >= 0 && j2 < FF2_offd_len);

            /* j1, F: O1 */
            if (Marker_FF2_offd[j2] == -1)
            {
               /* NOTE: we save this mapped index */
               RRi[local_size] = j2;
               KKi[local_size] = 1;
               Marker_FF2_offd_j[Marker_FF2_offd_count++] = j2;
               Marker_FF2_offd[j2] = local_size ++;
            }

            /* F^2: O1-D2, O1-O2 */
            /* row j1 is an external row. check recv_SF for strong F-neighbors  */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
            {
               /* k1: global index */
               big_k1 = recv_SF_j[k];
               /* if big_k1 is in the diag part */
               if (big_k1 >= col_start && big_k1 < col_end)
               {
                  k3 = (HYPRE_Int)(big_k1 - col_start);

                  hypre_assert(CF_marker[k3] < 0);

                  if (Marker_diag[k3] == -1)
                  {
                     RRi[local_size] = k3;
                     KKi[local_size] = 0;
                     Marker_diag_j[Marker_diag_count++] = k3;
                     Marker_diag[k3] = local_size ++;
                  }
               }
               else /* k1 is in the offd part */
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  if (AIR1_5 && k2 == -1)
                  {
                     continue;
                  }

                  hypre_assert(recv_SF_offd_list[k2] == big_k1);

                  /* map to FF2_offd */
                  k3 = Mapper_recv_SF_offd_list[k2];

                  /* this mapping must be successful */
                  hypre_assert(k3 >= 0 && k3 < FF2_offd_len);

                  if (Marker_FF2_offd[k3] == -1)
                  {
                     /* NOTE: we save this mapped index */
                     RRi[local_size] = k3;
                     KKi[local_size] = 1;
                     Marker_FF2_offd_j[Marker_FF2_offd_count++] = k3;
                     Marker_FF2_offd[k3] = local_size ++;
                  }
               }
            }
         }
      }

      hypre_assert(local_size <= local_max_size);

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(HYPRE_Complex));
      memset(Dxi, 0, local_size * sizeof(HYPRE_Complex));
      memset(Dbi, 0, local_size * sizeof(HYPRE_Complex));


      /* we will populate Ai row-by-row */
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2)  /* i2 == 1, i1 is an offd row */
         {
            /* open row i1, a remote row */
            for (j = hypre_CSRMatrixI(A_offd_FF2)[i1]; j < hypre_CSRMatrixI(A_offd_FF2)[i1 + 1]; j++)
            {
               /* big_j1 is a global index */
               big_j1 = hypre_CSRMatrixBigJ(A_offd_FF2)[j];

               /* if big_j1 is in the diag part */
               if (big_j1 >= col_start && big_j1 < col_end)
               {
                  j2 = (HYPRE_Int)(big_j1 - col_start);
                  /* if this col is marked with its local dense id */
                  if ((cc = Marker_diag[j2]) >= 0)
                  {
                     hypre_assert(CF_marker[j2] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     HYPRE_Complex vv = hypre_CSRMatrixData(A_offd_FF2)[j];
                     DAi[rr + cc * local_size] = vv;

                  }
               }
               else
               {
                  /* big_j1 is in offd part, search it in FF2_offd */
                  j2 =  hypre_BigBinarySearch(FF2_offd, big_j1, FF2_offd_len);
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        HYPRE_Complex vv = hypre_CSRMatrixData(A_offd_FF2)[j];
                        DAi[rr + cc * local_size] = vv;
                     }
                  }
               }
            }
         }
         else /* i2 == 0, i1 is a local row */
         {
            /* open row i1, a local row */
            for (j = A_diag_i[i1]; j < A_diag_i[i1 + 1]; j++)
            {
               /* j1 is a local index */
               j1 = A_diag_j[j];
               /* if this col is marked with its local dense id */
               if ((cc = Marker_diag[j1]) >= 0)
               {
                  hypre_assert(CF_marker[j1] < 0);

                  /* copy the value */
                  /* rr and cc: local dense ids */
                  HYPRE_Complex vv = A_diag_a[j];
                  DAi[rr + cc * local_size] = vv;

               }
            }

            if (num_procs > 1)
            {
               for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
               {
                  j1 = A_offd_j[j];
                  /* map to FF2_offd */
                  j2 = Mapper_offd_A[j1];
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        hypre_assert(CF_marker_offd[j1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        HYPRE_Complex vv = A_offd_a[j];
                        DAi[rr + cc * local_size] = vv;

                     }
                  }
               }
            }
         }
         /* done with row rr */
      }

      /* TODO bs: remove?
      {
         char Buf[4096];
         char Buf2[4096];
         hypre_MPI_Status stat;
         hypre_sprintf(Buf, "size %d\n", local_size);
         HYPRE_Int ii, jj;
         for (ii = 0; ii < local_size; ii++)
         {
            for (jj = 0; jj < local_size; jj++)
            {
               hypre_sprintf(Buf+strlen(Buf), "% .1f ", DAi[ii + jj * local_size]);
            }
            hypre_sprintf(Buf+strlen(Buf), "\n");
         }
         hypre_sprintf(Buf+strlen(Buf), "\n");

         if (my_id)
         {
            hypre_MPI_Send(Buf, 4096, hypre_MPI_CHAR, 0, 0, hypre_MPI_COMM_WORLD);
         }

         if (my_id == 0)
         {
            hypre_fprintf(stdout, "%s\n", Buf);

            for (i6 = 1; i6 < num_procs; i6++)
            {
               hypre_MPI_Recv(Buf2, 4096, hypre_MPI_CHAR, i6, 0, hypre_MPI_COMM_WORLD, &stat);
               hypre_fprintf(stdout, "%s\n", Buf2);
            }
         }
      }
      */

      /* rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = Marker_diag[i1]) >= 0)
         {
            hypre_assert(i1 == RRi[cc] && KKi[cc] == 0);
            /* Note the sign change */
            Dbi[cc] = -A_diag_a[j];
            rr++;
         }
      }

      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            i1 = A_offd_j[j];
            i2 = Mapper_offd_A[i1];
            if (i2 > -1)
            {
               if ((cc = Marker_FF2_offd[i2]) >= 0)
               {
                  hypre_assert(i2 == RRi[cc] && KKi[cc] == 1);
                  /* Note the sign change */
                  Dbi[cc] = -A_offd_a[j];
                  rr++;
               }
            }
         }
      }

      hypre_assert(rr <= local_size);

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * We have Ai and bi built. Solve the linear system by:
       *    - forward solve for triangular matrix
       *    - LU factorization (LAPACK) for local_size <= gmres_switch
       *    - Dense GMRES for local_size > gmres_switch
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      Aisol_method = local_size <= gmres_switch ? 'L' : 'G';
      if (local_size > 0)
      {
         if (is_triangular)
         {
            hypre_ordered_GS(DAi, Dbi, Dxi, local_size);
#if AIR_DEBUG
            HYPRE_Real alp = -1.0, err;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               hypre_printf("triangular solve res: %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve using LAPACK and LU factorization
         else if (Aisol_method == 'L')
         {
#if AIR_DEBUG
            memcpy(TMPA, DAi, local_size * local_size * sizeof(HYPRE_Complex));
            memcpy(TMPb, Dbi, local_size * sizeof(HYPRE_Complex));
#endif
            hypre_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi,
                         &lapack_info);

            hypre_assert(lapack_info == 0);

            if (lapack_info == 0)
            {
               /* solve A_i^T x_i = b_i,
                * solution is saved in b_i on return */
               hypre_dgetrs(&charT, &local_size, &ione, DAi, &local_size,
                            Ipi, Dbi, &local_size, &lapack_info);
               hypre_assert(lapack_info == 0);
            }
#if AIR_DEBUG
            HYPRE_Real alp = 1.0, bet = 0.0, err;
            hypre_dgemv(&charT, &local_size, &local_size, &alp, TMPA, &local_size, Dbi,
                        &ione, &bet, TMPd, &ione);
            alp = -1.0;
            hypre_daxpy(&local_size, &alp, TMPb, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               hypre_printf("dense: local res norm %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve by GMRES
         else
         {
            HYPRE_Real gmresAi_res;
            HYPRE_Int  gmresAi_niter;
            HYPRE_Int kdim = hypre_min(gmresAi_maxit, local_size);

            hypre_fgmresT(local_size, DAi, Dbi, gmresAi_tol, kdim, Dxi,
                          &gmresAi_res, &gmresAi_niter, 0);

            if (gmresAi_res > gmresAi_tol)
            {
               hypre_printf("gmres/jacobi not converge to %e: final_res %e\n", gmresAi_tol, gmresAi_res);
            }

#if AIR_DEBUG
            HYPRE_Real err, nrmb;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            HYPRE_Real alp = -1.0;
            nrmb = hypre_dnrm2(&local_size, Dbi, &ione);
            hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = hypre_dnrm2(&local_size, TMPd, &ione);
            if (err / nrmb > gmresAi_tol)
            {
               hypre_printf("GMRES/Jacobi: res norm %e, nrmb %e, relative %e\n", err, nrmb, err / nrmb);
               hypre_printf("GMRES/Jacobi: relative %e\n", gmresAi_res);
               exit(0);
            }
#endif
         }
      }

      HYPRE_Complex *Soli = (is_triangular || (Aisol_method == 'G')) ? Dxi : Dbi;

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * Now we are ready to fill this row of R
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2) /* offd */
         {
            hypre_assert(Marker_FF2_offd[i1] == rr);

            /* col idx: use the index in FF2_offd,
             * and you will see why later (very soon!) */
            R_offd_j[cnt_offd] = i1;
            /* copy the value */
            R_offd_data[cnt_offd++] = Soli[rr];
         }
         else /* diag */
         {
            hypre_assert(Marker_diag[i1] == rr);

            /* col idx: use local index i1 */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Soli[rr];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i'' */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic + 1] = cnt_diag;

      R_offd_i[ic + 1] = cnt_offd;

      /* RESET marker arrays */
      for (j = 0; j < Marker_diag_count; j++)
      {
         Marker_diag[Marker_diag_j[j]] = -1;
      }

      for (j = 0; j < Marker_FF2_offd_count; j++)
      {
         Marker_FF2_offd[Marker_FF2_offd_j[j]] = -1;
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   /*
   hypre_MPI_Barrier(comm);
   t2 = hypre_MPI_Wtime() - t2;
   air_time2 += t2;
   */

   hypre_assert(ic == n_cpts);
   hypre_assert(cnt_diag == nnz_diag);
   hypre_assert(cnt_offd == nnz_offd);

   /*
   HYPRE_Real t3 = hypre_MPI_Wtime();
   */

   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, Marker_FF2_offd should be all -1 */
   /*
   for (i = 0; i < FF2_offd_len; i++)
   {
      hypre_assert(Marker_FF2_offd[i] == - 1);
   }
   */

   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (Marker_FF2_offd[i1] == -1)
      {
         num_cols_offd_R++;
         Marker_FF2_offd[i1] = 1;
      }
   }

   tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_R, HYPRE_MEMORY_HOST);
   col_map_offd_R = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_R, HYPRE_MEMORY_HOST);
   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the local indices in FF2_offd [will be changed] */
   for (i = 0, i1 = 0; i < FF2_offd_len; i++)
   {
      if (Marker_FF2_offd[i] == 1)
      {
         tmp_map_offd[i1++] = i;
      }
   }

   hypre_assert(i1 == num_cols_offd_R);
   //printf("FF2_offd_len %d, num_cols_offd_R %d\n", FF2_offd_len, num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t FF2_offd
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = hypre_BinarySearch(tmp_map_offd, i1, num_cols_offd_R);
      /* searching must succeed */
      hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      /* change index */
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids [guaranteed to be sorted] */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = FF2_offd[tmp_map_offd[i]];
   }

   /* Now, we should have everything of Parcsr matrix R */
   R = hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixData(R_diag) = R_diag_data;
   hypre_CSRMatrixI(R_diag)    = R_diag_i;
   hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixData(R_offd) = R_offd_data;
   hypre_CSRMatrixI(R_offd)    = R_offd_i;
   hypre_CSRMatrixJ(R_offd)    = R_offd_j;

   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /*
   t3 = hypre_MPI_Wtime() - t3;
   air_time3 += t3;

   HYPRE_Real t4 = hypre_MPI_Wtime();
   */

   /* create CommPkg of R */
   hypre_ParCSRMatrixAssumedPartition(R) = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   hypre_MatvecCommPkgCreate(R);

   /*
   t4 = hypre_MPI_Wtime() - t4;
   air_time4 += t4;
   */

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_i, HYPRE_MEMORY_HOST);
   hypre_TFree(send_SF_i, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_SF_i, HYPRE_MEMORY_HOST);
   hypre_TFree(send_SF_j, HYPRE_MEMORY_HOST);
   hypre_TFree(send_SF_jstarts, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_SF_j, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_SF_jstarts, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_SF, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_SF_offd_list, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_SF_j2, HYPRE_MEMORY_HOST);
   hypre_TFree(Mapper_recv_SF_offd_list, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_recv_SF_offd_list, HYPRE_MEMORY_HOST);
   hypre_TFree(FF2_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(send_FF2_i, HYPRE_MEMORY_HOST);
   /* hypre_TFree(recv_FF2_i); */
   hypre_TFree(send_FF2_j, HYPRE_MEMORY_HOST);
   hypre_TFree(send_FF2_a, HYPRE_MEMORY_HOST);
   hypre_TFree(send_FF2_jstarts, HYPRE_MEMORY_HOST);
   /* hypre_TFree(recv_FF2_j); */
   /* hypre_TFree(recv_FF2_a); */
   hypre_CSRMatrixDestroy(A_offd_FF2);
   hypre_TFree(recv_FF2_jstarts, HYPRE_MEMORY_HOST);
   hypre_MatvecCommPkgDestroy(comm_pkg_FF2_i);
   hypre_TFree(comm_pkg_FF2_j, HYPRE_MEMORY_HOST);
   hypre_TFree(Mapper_offd_A, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_FF2_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(Marker_FF2_offd_j, HYPRE_MEMORY_HOST);
   hypre_TFree(DAi, HYPRE_MEMORY_HOST);
   hypre_TFree(Dbi, HYPRE_MEMORY_HOST);
   hypre_TFree(Dxi, HYPRE_MEMORY_HOST);
   hypre_TFree(Ipi, HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   hypre_TFree(TMPA, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPb, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPd, HYPRE_MEMORY_HOST);
   hypre_SeqVectorDestroy(tmpv);
#endif
   hypre_TFree(RRi, HYPRE_MEMORY_HOST);
   hypre_TFree(KKi, HYPRE_MEMORY_HOST);

   if (gmres_switch < local_max_size)
   {
      hypre_fgmresT(0, NULL, NULL, 0.0, 0, NULL, NULL, NULL, -2);
   }

   /*
   t0 = hypre_MPI_Wtime() - t0;
   air_time0 += t0;
   */

   return 0;
}

HYPRE_Int
hypre_BoomerAMGBuildRestrNeumannAIRHost( hypre_ParCSRMatrix   *A,
                                         HYPRE_Int            *CF_marker,
                                         HYPRE_BigInt         *num_cpts_global,
                                         HYPRE_Int             num_functions,
                                         HYPRE_Int            *dof_func,
                                         HYPRE_Int             NeumannDeg,
                                         HYPRE_Real            strong_thresholdR,
                                         HYPRE_Real            filter_thresholdR,
                                         HYPRE_Int             debug_flag,
                                         hypre_ParCSRMatrix  **R_ptr)
{
   HYPRE_UNUSED_VAR(num_functions);
   HYPRE_UNUSED_VAR(dof_func);
   HYPRE_UNUSED_VAR(debug_flag);

   /* HYPRE_Real t0 = hypre_MPI_Wtime(); */
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);

   /* Restriction matrix R and CSR's */
   hypre_ParCSRMatrix *R;
   hypre_CSRMatrix *R_diag;
   hypre_CSRMatrix *R_offd;

   /* arrays */
   HYPRE_Complex   *R_diag_a;
   HYPRE_Int       *R_diag_i;
   HYPRE_Int       *R_diag_j;
   HYPRE_Complex   *R_offd_a;
   HYPRE_Int       *R_offd_i;
   HYPRE_Int       *R_offd_j;
   HYPRE_BigInt    *col_map_offd_R;

   HYPRE_Int        i, j, j1, ic,
                    num_cols_offd_R;
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt     total_global_cpts/*, my_first_cpt*/;
   HYPRE_Int        nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_BigInt    *send_buf_i;

   /* local size */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt col_start = hypre_ParCSRMatrixFirstRowIndex(A);

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_MemoryLocation memory_location_R = hypre_ParCSRMatrixMemoryLocation(A);

   /*-------------- global number of C points and my start position */
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
#if 0
   if (num_cols_A_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   }

   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* number of sends (number of procs) */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   /* number of recvs (number of procs) */
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   /* number of elements to send */
   num_elems_send = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   send_buf_i = hypre_CTAlloc(HYPRE_Int, num_elems_send, HYPRE_MEMORY_HOST);

   /* copy CF markers of elements to send to buffer */
   for (i = 0;  i < num_elems_send; i++)
   {
      send_buf_i[i] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, CF_marker_offd);
   /* destroy the handle to finish communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      for (i = 0; i < num_elems_send; i++)
      {
         send_buf_i[i] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, dof_func_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* init markers to zeros */
   Marker_diag = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   Marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
#endif

   hypre_ParCSRMatrix *AFF, *ACF, *X, *X2, *Z, *Z2;
   hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "FF", &AFF, strong_thresholdR);
   hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "CF", &ACF, strong_thresholdR);

   /* A_FF := I - D^{-1}*A_FF */
   hypre_CSRMatrix *AFF_diag = hypre_ParCSRMatrixDiag(AFF);
   hypre_CSRMatrix *AFF_offd = hypre_ParCSRMatrixOffd(AFF);
   HYPRE_Complex   *AFF_diag_a = hypre_CSRMatrixData(AFF_diag);
   HYPRE_Int       *AFF_diag_i = hypre_CSRMatrixI(AFF_diag);
   HYPRE_Int       *AFF_diag_j = hypre_CSRMatrixJ(AFF_diag);
   HYPRE_Complex   *AFF_offd_a = hypre_CSRMatrixData(AFF_offd);
   HYPRE_Int       *AFF_offd_i = hypre_CSRMatrixI(AFF_offd);
   HYPRE_Int       *AFF_offd_j = hypre_CSRMatrixJ(AFF_offd);
   HYPRE_Int        n_fpts = hypre_CSRMatrixNumRows(AFF_diag);
   HYPRE_Int        n_cpts = n_fine - n_fpts;
   hypre_assert(n_cpts == hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(ACF)));

   HYPRE_Int       *Fmap = hypre_TAlloc(HYPRE_Int, n_fpts, HYPRE_MEMORY_HOST);

   /* map from F-pts to all points */
   for (i = 0, j = 0; i < n_fine; i++)
   {
      if (CF_marker[i] < 0)
      {
         Fmap[j++] = i;
      }
   }

   hypre_assert(j == n_fpts);

   HYPRE_Complex *diag_entries = hypre_TAlloc(HYPRE_Complex, n_fpts, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fpts; i++)
   {
      /* find the diagonal element and store inverse */
      for (j = AFF_diag_i[i]; j < AFF_diag_i[i + 1]; j++)
      {
         if (AFF_diag_j[j] == i)
         {
            diag_entries[i] = 1.0 / AFF_diag_a[j];
            AFF_diag_a[j] = 0.0;
            break;
         }
      }

      for (j = AFF_diag_i[i]; j < AFF_diag_i[i + 1]; j++)
      {
         AFF_diag_a[j] *= -diag_entries[i];
      }
      if (num_procs > 1)
      {
         for (j = AFF_offd_i[i]; j < AFF_offd_i[i + 1]; j++)
         {
            hypre_assert( hypre_ParCSRMatrixColMapOffd(AFF)[AFF_offd_j[j]] != \
                          i + hypre_ParCSRMatrixFirstRowIndex(AFF) );

            AFF_offd_a[j] *= -diag_entries[i];
         }
      }
   }

   /* Z = Acf * (I + N + N^2 + ... + N^k] * D^{-1}
    * N = I - D^{-1} * A_FF (computed above)
    * the last D^{-1} will not be done here (but later)
    */
   if (NeumannDeg < 1)
   {
      Z = ACF;
   }
   else if (NeumannDeg == 1)
   {
      X = hypre_ParMatmul(ACF, AFF);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }
   else
   {
      X = hypre_ParMatmul(AFF, AFF);
      hypre_ParCSRMatrixAdd(1.0, AFF, 1.0, X, &Z);
      for (i = 2; i < NeumannDeg; i++)
      {
         X2 = hypre_ParMatmul(X, AFF);
         hypre_ParCSRMatrixAdd(1.0, Z, 1.0, X2, &Z2);
         hypre_ParCSRMatrixDestroy(X);
         hypre_ParCSRMatrixDestroy(Z);
         Z = Z2;
         X = X2;
      }
      hypre_ParCSRMatrixDestroy(X);
      X = hypre_ParMatmul(ACF, Z);
      hypre_ParCSRMatrixDestroy(Z);
      hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      hypre_ParCSRMatrixDestroy(X);
   }

   hypre_ParCSRMatrixDestroy(AFF);
   if (NeumannDeg >= 1)
   {
      hypre_ParCSRMatrixDestroy(ACF);
   }

   hypre_CSRMatrix *Z_diag = hypre_ParCSRMatrixDiag(Z);
   hypre_CSRMatrix *Z_offd = hypre_ParCSRMatrixOffd(Z);
   HYPRE_Complex   *Z_diag_a = hypre_CSRMatrixData(Z_diag);
   HYPRE_Int       *Z_diag_i = hypre_CSRMatrixI(Z_diag);
   HYPRE_Int       *Z_diag_j = hypre_CSRMatrixJ(Z_diag);
   HYPRE_Complex   *Z_offd_a = hypre_CSRMatrixData(Z_offd);
   HYPRE_Int       *Z_offd_i = hypre_CSRMatrixI(Z_offd);
   HYPRE_Int       *Z_offd_j = hypre_CSRMatrixJ(Z_offd);
   HYPRE_Int        num_cols_offd_Z = hypre_CSRMatrixNumCols(Z_offd);
   /*
   HYPRE_BigInt       *col_map_offd_Z  = hypre_ParCSRMatrixColMapOffd(Z);
   */
   /* send and recv diagonal entries (wrt Z) */
   HYPRE_Complex *diag_entries_offd = hypre_TAlloc(HYPRE_Complex, num_cols_offd_Z, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkg *comm_pkg_Z = hypre_ParCSRMatrixCommPkg(Z);
   HYPRE_Int num_sends_Z = hypre_ParCSRCommPkgNumSends(comm_pkg_Z);
   HYPRE_Int num_elems_send_Z = hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z);
   HYPRE_Complex *send_buf_Z = hypre_TAlloc(HYPRE_Complex, num_elems_send_Z, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_Z[i] = diag_entries[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)];
   }
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg_Z, send_buf_Z, diag_entries_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* send and recv Fmap (wrt Z): global */
   HYPRE_BigInt *Fmap_offd_global = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_Z, HYPRE_MEMORY_HOST);
   send_buf_i = hypre_TAlloc(HYPRE_BigInt, num_elems_send_Z, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_i[i] = Fmap[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)] + col_start;
   }
   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg_Z, send_buf_i, Fmap_offd_global);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   nnz_diag = hypre_CSRMatrixNumNonzeros(Z_diag) + n_cpts;
   nnz_offd = hypre_CSRMatrixNumNonzeros(Z_offd);

   /*------------- allocate arrays */
   R_diag_i = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, memory_location_R);
   R_diag_j = hypre_CTAlloc(HYPRE_Int,  nnz_diag, memory_location_R);
   R_diag_a = hypre_CTAlloc(HYPRE_Complex, nnz_diag, memory_location_R);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i = hypre_CTAlloc(HYPRE_Int,  n_cpts + 1, memory_location_R);
   R_offd_j = hypre_CTAlloc(HYPRE_Int,  nnz_offd, memory_location_R);
   R_offd_a = hypre_CTAlloc(HYPRE_Complex, nnz_offd, memory_location_R);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      for (j = Z_diag_i[ic]; j < Z_diag_i[ic + 1]; j++)
      {
         j1 = Z_diag_j[j];
         R_diag_j[cnt_diag] = Fmap[j1];
         R_diag_a[cnt_diag++] = -Z_diag_a[j] * diag_entries[j1];
      }

      /* identity */
      R_diag_j[cnt_diag] = i;
      R_diag_a[cnt_diag++] = 1.0;

      for (j = Z_offd_i[ic]; j < Z_offd_i[ic + 1]; j++)
      {
         j1 = Z_offd_j[j];
         R_offd_j[cnt_offd] = j1;
         R_offd_a[cnt_offd++] = -Z_offd_a[j] * diag_entries_offd[j1];
      }

      R_diag_i[ic + 1] = cnt_diag;
      R_offd_i[ic + 1] = cnt_offd;

      ic++;
   }

   hypre_assert(ic == n_cpts);
   hypre_assert(cnt_diag == nnz_diag);
   hypre_assert(cnt_offd == nnz_offd);

   num_cols_offd_R = num_cols_offd_Z;
   col_map_offd_R = Fmap_offd_global;

   /* Now, we should have everything of Parcsr matrix R */
   R = hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixData(R_diag) = R_diag_a;
   hypre_CSRMatrixI(R_diag)    = R_diag_i;
   hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixData(R_offd) = R_offd_a;
   hypre_CSRMatrixI(R_offd)    = R_offd_i;
   hypre_CSRMatrixJ(R_offd)    = R_offd_j;

   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_ParCSRMatrixAssumedPartition(R) = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   hypre_ParCSRMatrixDestroy(Z);
   hypre_TFree(Fmap, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_entries, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_entries_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_i, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_Z, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int
hypre_BoomerAMGBuildRestrNeumannAIR( hypre_ParCSRMatrix   *A,
                                     HYPRE_Int            *CF_marker,
                                     HYPRE_BigInt         *num_cpts_global,
                                     HYPRE_Int             num_functions,
                                     HYPRE_Int            *dof_func,
                                     HYPRE_Int             NeumannDeg,
                                     HYPRE_Real            strong_thresholdR,
                                     HYPRE_Real            filter_thresholdR,
                                     HYPRE_Int             debug_flag,
                                     hypre_ParCSRMatrix  **R_ptr)
{
   hypre_GpuProfilingPushRange("RestrNeumannAIR");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildRestrNeumannAIRDevice(A, CF_marker, num_cpts_global, num_functions,
                                                       dof_func,
                                                       NeumannDeg, strong_thresholdR, filter_thresholdR,
                                                       debug_flag, R_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildRestrNeumannAIRHost(A, CF_marker, num_cpts_global, num_functions,
                                                     dof_func,
                                                     NeumannDeg, strong_thresholdR, filter_thresholdR,
                                                     debug_flag, R_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}
