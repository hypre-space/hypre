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




#include "_hypre_parcsr_ls.h"
#include "_hypre_lapack.h"
#include "_hypre_blas.h"

#define AIR_DEBUG 0

//HYPRE_Real time1 = 0.0;
//HYPRE_Real time2 = 0.0;
//HYPRE_Real time3 = 0.0;

HYPRE_Int
hypre_BoomerAMGBuildRestrDist2AIR( hypre_ParCSRMatrix   *A,
                                   HYPRE_Int            *CF_marker,
                                   hypre_ParCSRMatrix   *S,
                                   HYPRE_Int            *num_cpts_global,
                                   HYPRE_Int             num_functions,
                                   HYPRE_Int            *dof_func,
                                   HYPRE_Int             debug_flag,
                                   HYPRE_Real            trunc_factor,
                                   HYPRE_Int             max_elmts,
                                   HYPRE_Int            *col_offd_S_to_A,
                                   hypre_ParCSRMatrix  **R_ptr) {
   
   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_ParCSRCommPkg     *comm_pkg_SF;
   
   /* diag part of A */
   hypre_CSRMatrix *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix *A_offd   = hypre_ParCSRMatrixOffd(A);   
   HYPRE_Real      *A_offd_a = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int       *col_map_offd_A  = hypre_ParCSRMatrixColMapOffd(A);
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
   HYPRE_Real      *R_diag_data;
   HYPRE_Int       *R_diag_i;
   HYPRE_Int       *R_diag_j;
   HYPRE_Real      *R_offd_data;
   HYPRE_Int       *R_offd_i;
   HYPRE_Int       *R_offd_j;
   HYPRE_Int       *col_map_offd_R;
   /* CF marker off-diag part */
   HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   HYPRE_Int       *dof_func_offd  = NULL;
   
   HYPRE_Int        i, j, j1, j2, k, i1, i2, k1, k2, k3, rr, cc, ic, index, start, end, 
                    local_max_size, local_size, num_cols_offd_R;
   /*HYPRE_Int        i6;*/
   HYPRE_Int        *FF2_offd, FF2_offd_len;

   /* LAPACK */
   HYPRE_Real *DAi, *Dbi;
#if AIR_DEBUG
   HYPRE_Real *TMPA, *TMPb, *TMPd;
#endif
   HYPRE_Int *Ipi, lapack_info, ione = 1, *RRi, *KKi;
   char charT = 'T';

   HYPRE_Int my_id, num_procs;
   HYPRE_Int total_global_cpts/*, my_first_cpt*/;
   HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   HYPRE_Int *Marker_diag, *Marker_offd;
   HYPRE_Int *Marker_diag_j, Marker_diag_count;
   HYPRE_Int num_sends, num_recvs, num_elems_send;
   /* local size, local num of C points */
   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int n_cpts = 0;
   /* my first column range */
   /* XXX is this also right?
   HYPRE_Int col_start = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_Int col_end   = hypre_ParCSRMatrixLastColDiag(A);
   */
   HYPRE_Int col_start = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int col_end   = col_start + n_fine;

   HYPRE_Int  *send_buf_i;
   
   /* recv_SF means the Strong F-neighbors of offd elements in col_map_offd */
   HYPRE_Int *send_SF_i, *send_SF_j, send_SF_jlen;
   HYPRE_Int *recv_SF_i, *recv_SF_j, *recv_SF_j2, recv_SF_jlen;
   HYPRE_Int *send_SF_jstarts, *recv_SF_jstarts;
   HYPRE_Int *recv_SF_offd_list, recv_SF_offd_list_len;
   HYPRE_Int *Mapper_recv_SF_offd_list, *Mapper_offd_A, *Marker_recv_SF_offd_list;
   HYPRE_Int *Marker_FF2_offd;
   HYPRE_Int *Marker_FF2_offd_j, Marker_FF2_offd_count;

   /* for communication of offd F and F^2 rows of A */
   hypre_ParCSRCommPkg *comm_pkg_FF2_i, *comm_pkg_FF2_j;
   HYPRE_Int num_sends_FF2, *send_FF2_i, send_FF2_ilen, *send_FF2_j, send_FF2_jlen,
             num_recvs_FF2, *recv_FF2_i, recv_FF2_ilen, *recv_FF2_j, recv_FF2_jlen,
             *send_FF2_jstarts, *recv_FF2_jstarts;
   HYPRE_Complex *send_FF2_a, *recv_FF2_a;

   /* ghost rows: offd F and F2-pts */
   hypre_CSRMatrix *A_offd_FF2   = NULL;
   
   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
#ifdef HYPRE_NO_GLOBAL_PARTITION
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
   /*my_first_cpt = num_cpts_global[my_id];*/
   total_global_cpts = num_cpts_global[num_procs];
#endif
 
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
   send_buf_i = hypre_CTAlloc(HYPRE_Int , num_elems_send, HYPRE_MEMORY_HOST);
   
   /* copy CF markers of elements to send to buffer 
    * RL: why copy them with two for loops? Why not just loop through all in one */
   for (i = 0, index = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         send_buf_i[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
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
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            send_buf_i[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
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
      for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
      {
         if (CF_marker[S_diag_j[j]] < 0)
         {
            send_SF_i[i] ++;
         }
      }
      /* offd part of row i1 */
      for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
         if (CF_marker_offd[j1] < 0)
         {
            send_SF_i[i] ++;
         }
      }

      /* add to the num of elems going to be sent */
      send_SF_jlen += send_SF_i[i];
   }

   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_SF_i, recv_SF_i+1);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);
   
   send_SF_j = hypre_CTAlloc(HYPRE_Int, send_SF_jlen, HYPRE_MEMORY_HOST);
   send_SF_jstarts = hypre_CTAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);

   for (i = 0, i1 = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* 1-past-the-end pos */
      end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);

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
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
         {
            k1 = S_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_SF_j[i1++] = col_start + k1;
            }
         }
         /* offd part of row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
            if (CF_marker_offd[k1] < 0)
            {
               send_SF_j[i1++] = col_map_offd_A[k1];
            }
         }
      }
      send_SF_jstarts[i+1] = i1;
   }

   hypre_assert(i1 == send_SF_jlen);

   /* adjust recv_SF_i to ptrs */
   for (i = 1; i <= num_cols_A_offd; i++)
   {
      recv_SF_i[i] += recv_SF_i[i-1];
   }

   recv_SF_jlen = recv_SF_i[num_cols_A_offd];
   recv_SF_j = hypre_CTAlloc(HYPRE_Int, recv_SF_jlen, HYPRE_MEMORY_HOST);
   recv_SF_jstarts = hypre_CTAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs; i++)
   {
      start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_SF_jstarts[i] = recv_SF_i[start];
   }

   /* create a communication package for SF_j */
   comm_pkg_SF = hypre_CTAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm         (comm_pkg_SF) = comm;
   hypre_ParCSRCommPkgNumSends     (comm_pkg_SF) = num_sends;
   hypre_ParCSRCommPkgSendProcs    (comm_pkg_SF) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_SF) = send_SF_jstarts;
   hypre_ParCSRCommPkgNumRecvs     (comm_pkg_SF) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs    (comm_pkg_SF) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_SF) = recv_SF_jstarts;

   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg_SF, send_SF_j, recv_SF_j);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * recv_SF_offd_list: a sorted list of offd elems in recv_SF_j 
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   recv_SF_offd_list = hypre_CTAlloc(HYPRE_Int, recv_SF_jlen, HYPRE_MEMORY_HOST);
   for (i = 0, j = 0; i < recv_SF_jlen; i++)
   {
      i1 = recv_SF_j[i];
      /* offd */
      if (i1 < col_start || i1 >= col_end)
      {
         recv_SF_offd_list[j++] = i1;
      }
   }

   /* remove redundancy after sorting */
   hypre_qsort0(recv_SF_offd_list, 0, j-1);

   for (i = 0, recv_SF_offd_list_len = 0; i < j; i++)
   {
      if (i == 0 || recv_SF_offd_list[i] != recv_SF_offd_list[i-1])
      {
         recv_SF_offd_list[recv_SF_offd_list_len++] = recv_SF_offd_list[i];
      }
   }

   /* make a copy of recv_SF_j in which
    * adjust the offd indices corresponding to recv_SF_offd_list */
   recv_SF_j2 = hypre_CTAlloc(HYPRE_Int, recv_SF_jlen, HYPRE_MEMORY_HOST);
   for (i = 0; i < recv_SF_jlen; i++)
   {
      i1 = recv_SF_j[i];
      if (i1 < col_start || i1 >= col_end)
      {
         j = hypre_BinarySearch(recv_SF_offd_list, i1, recv_SF_offd_list_len);
         hypre_assert(j >= 0 && j < recv_SF_offd_list_len);
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
      i1 = recv_SF_offd_list[i];
      hypre_assert(i1 < col_start || i1 >= col_end);
      j = hypre_BinarySearch(col_map_offd_A, i1, num_cols_A_offd);
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
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         j1 = S_diag_j[j];
         /* if it is F */
         if (CF_marker[j1] < 0)
         {
            /* go through its offd part */
            for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
            {
               k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
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
      for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
         /* offd F pts */
         if (CF_marker_offd[j1] < 0)
         {
            if (!Marker_offd[j1])
            {
               FF2_offd_len ++;
               Marker_offd[j1] = 1;
            }
            /* offd(F)-offd(F), need to open recv_SF */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
            {
               /* k1: global index */
               k1 = recv_SF_j[k];
               /* if k1 is not in my range */
               if (k1 < col_start || k1 >= col_end)
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  hypre_assert(recv_SF_offd_list[k2] == k1);
                  
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
   FF2_offd = hypre_CTAlloc(HYPRE_Int, FF2_offd_len, HYPRE_MEMORY_HOST);
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
         j = recv_SF_offd_list[i];
         hypre_assert(j < col_start || j >= col_end);
         FF2_offd[k++] = j;
         Marker_recv_SF_offd_list[i] = 0;
      }
   }
   hypre_assert(k == FF2_offd_len);

   /* sort the list */
   hypre_qsort0(FF2_offd, 0, FF2_offd_len-1);

   /* there must be no repetition in FF2_offd */
   for (i = 1; i < FF2_offd_len; i++)
   {
      hypre_assert(FF2_offd[i] != FF2_offd[i-1]);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *    Create CommPkgs for exchanging offd F and F2 rows of A
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* we will create TWO commPkg: one for row lengths and one for row data,
    * similar to what we have done above for SF_i, SF_j */
   hypre_ParCSRFindExtendCommPkg(A, FF2_offd_len, FF2_offd, &comm_pkg_FF2_i);
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
      send_FF2_i[i] = A_diag_i[j+1] - A_diag_i[j] + A_offd_i[j+1] - A_offd_i[j];
      send_FF2_jlen += send_FF2_i[i];
   }
 
   /* do communication */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg_FF2_i, send_FF2_i, recv_FF2_i+1);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   send_FF2_j = hypre_CTAlloc(HYPRE_Int, send_FF2_jlen, HYPRE_MEMORY_HOST);
   send_FF2_a = hypre_CTAlloc(HYPRE_Complex, send_FF2_jlen, HYPRE_MEMORY_HOST);
   send_FF2_jstarts = hypre_CTAlloc(HYPRE_Int, num_sends_FF2 + 1, HYPRE_MEMORY_HOST);
   
   for (i = 0, i1 = 0; i < num_sends_FF2; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i);
      end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i+1);
      for (j = start; j < end; j++)
      {
         /* will send row j1 to send_proc[i] */
         j1 = hypre_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, j);
         /* open row j1 and fill ja and a */
         for (k = A_diag_i[j1]; k < A_diag_i[j1+1]; k++)
         {
            send_FF2_j[i1] = col_start + A_diag_j[k];
            send_FF2_a[i1] = A_diag_a[k];
            i1++;
         }
         if (num_procs > 1)
         {
            for (k = A_offd_i[j1]; k < A_offd_i[j1+1]; k++)
            {
               send_FF2_j[i1] = col_map_offd_A[A_offd_j[k]];
               send_FF2_a[i1] = A_offd_a[k];
               i1++;
            }
         }
      }
      send_FF2_jstarts[i+1] = i1;
   }
   hypre_assert(i1 == send_FF2_jlen);
   
   /* adjust recv_FF2_i to ptrs */
   for (i = 1; i <= recv_FF2_ilen; i++)
   {
      recv_FF2_i[i] += recv_FF2_i[i-1];
   }

   recv_FF2_jlen = recv_FF2_i[recv_FF2_ilen];
   recv_FF2_j = hypre_CTAlloc(HYPRE_Int, recv_FF2_jlen, HYPRE_MEMORY_HOST);
   recv_FF2_a = hypre_CTAlloc(HYPRE_Complex, recv_FF2_jlen, HYPRE_MEMORY_HOST);
   recv_FF2_jstarts = hypre_CTAlloc(HYPRE_Int, num_recvs_FF2 + 1, HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs_FF2; i++)
   {
      start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, i);
      recv_FF2_jstarts[i] = recv_FF2_i[start];
   }

   /* create a communication package for FF2_j */
   comm_pkg_FF2_j = hypre_CTAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm         (comm_pkg_FF2_j) = comm;
   hypre_ParCSRCommPkgNumSends     (comm_pkg_FF2_j) = num_sends_FF2;
   hypre_ParCSRCommPkgSendProcs    (comm_pkg_FF2_j) = hypre_ParCSRCommPkgSendProcs(comm_pkg_FF2_i);
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_FF2_j) = send_FF2_jstarts;
   hypre_ParCSRCommPkgNumRecvs     (comm_pkg_FF2_j) = num_recvs_FF2;
   hypre_ParCSRCommPkgRecvProcs    (comm_pkg_FF2_j) = hypre_ParCSRCommPkgRecvProcs(comm_pkg_FF2_i);
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_FF2_j) = recv_FF2_jstarts;
   
   /* do communication */
   /* ja */
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg_FF2_j, send_FF2_j, recv_FF2_j);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);
  
   /* a */
   comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg_FF2_j, send_FF2_a, recv_FF2_a);
   /* ... */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd_FF2 is ready ! */
   A_offd_FF2 = hypre_CSRMatrixCreate(recv_FF2_ilen, hypre_ParCSRMatrixGlobalNumCols(A),
                                      recv_FF2_jlen);

   hypre_CSRMatrixI   (A_offd_FF2) = recv_FF2_i;
   hypre_CSRMatrixJ   (A_offd_FF2) = recv_FF2_j;
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
      Mapper_offd_A[i] = hypre_BinarySearch(FF2_offd, col_map_offd_A[i], FF2_offd_len);
   }
   
   /* Mapping from recv_SF_offd_list, overwrite the old one*/
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      Mapper_recv_SF_offd_list[i] = hypre_BinarySearch(FF2_offd, recv_SF_offd_list[i], FF2_offd_len);
   }

   /* marker */
   Marker_FF2_offd = hypre_CTAlloc(HYPRE_Int, FF2_offd_len, HYPRE_MEMORY_HOST);

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
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
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
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
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
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
            
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
      for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];

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
         for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
         {
            /* k1: global index */
            k1 = recv_SF_j[k];
            /* if k1 is in the diag part */
            if (k1 >= col_start && k1 < col_end)
            {
               k3 = k1 - col_start;
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

               hypre_assert(recv_SF_offd_list[k2] == k1);

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
 
   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;
 
   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;
 
   /*------------- allocate arrays */
   R_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts+1, HYPRE_MEMORY_HOST);
   R_diag_j    = hypre_CTAlloc(HYPRE_Int,  nnz_diag, HYPRE_MEMORY_HOST);
   R_diag_data = hypre_CTAlloc(HYPRE_Real, nnz_diag, HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'', 
    * allocation needed even for empty CSR */
   R_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_cpts+1, HYPRE_MEMORY_HOST);
   R_offd_j    = hypre_CTAlloc(HYPRE_Int,  nnz_offd, HYPRE_MEMORY_HOST);
   R_offd_data = hypre_CTAlloc(HYPRE_Real, nnz_offd, HYPRE_MEMORY_HOST);

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

   //for (i = 0; i < num_cols_A_offd; i++)
   //{
   //   Marker_offd[i] = -1;
   //}
   //for (i = 0; i < recv_SF_offd_list_len; i++)
   //{
   //   Marker_recv_SF_list[i] = -1;
   //}

   /* the local matrix and rhs (dense) 
    * column-major as always by BLAS/LAPACK */
   /* matrix */
   DAi = hypre_CTAlloc(HYPRE_Real, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   /* rhs */
   Dbi = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
   /* pivot */
   Ipi = hypre_CTAlloc(HYPRE_Int, local_max_size, HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   /* FOR DEBUG */
   TMPA = hypre_CTAlloc(HYPRE_Real, local_max_size * local_max_size, HYPRE_MEMORY_HOST);
   TMPb = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
   TMPd = hypre_CTAlloc(HYPRE_Real, local_max_size, HYPRE_MEMORY_HOST);
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
//HYPRE_Real t1 = hypre_MPI_Wtime();
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
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
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
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
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];

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
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];

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
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
            {
               /* k1: global index */
               k1 = recv_SF_j[k];
               /* if k1 is in the diag part */
               if (k1 >= col_start && k1 < col_end)
               {
                  k3 = k1 - col_start;

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

                  hypre_assert(recv_SF_offd_list[k2] == k1);

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

//t1 = hypre_MPI_Wtime() - t1;
//time1 += t1;

      hypre_assert(local_size <= local_max_size);

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries 
       * we need from these rows and put them in Ai and bi*/

//HYPRE_Real t2 = hypre_MPI_Wtime();
      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(HYPRE_Real));
      memset(Dbi, 0, local_size * sizeof(HYPRE_Real));

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
            for (j = hypre_CSRMatrixI(A_offd_FF2)[i1]; j < hypre_CSRMatrixI(A_offd_FF2)[i1+1]; j++)
            {
               /* j1 is a global index */
               j1 = hypre_CSRMatrixJ(A_offd_FF2)[j];

               /* if j1 is in the diag part */
               if (j1 >= col_start && j1 < col_end)
               {
                  j2 = j1 - col_start;
                  /* if this col is marked with its local dense id */
                  if ((cc = Marker_diag[j2]) >= 0)
                  {
                     hypre_assert(CF_marker[j2] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = hypre_CSRMatrixData(A_offd_FF2)[j];
                  }
               }
               else
               {
                  /* j1 is in offd part, search it in FF2_offd */
                  j2 =  hypre_BinarySearch(FF2_offd, j1, FF2_offd_len);
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = hypre_CSRMatrixData(A_offd_FF2)[j];
                     }
                  }
               }
            }
         }
         else /* i2 == 0, i1 is a local row */
         {
            /* open row i1, a local row */
            for (j = A_diag_i[i1]; j < A_diag_i[i1+1]; j++)
            {
               /* j1 is a local index */
               j1 = A_diag_j[j];
               /* if this col is marked with its local dense id */
               if ((cc = Marker_diag[j1]) >= 0)
               {
                  hypre_assert(CF_marker[j1] < 0);
                  
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_a[j];
               }
            }

            if (num_procs > 1)
            {
               for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
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
                        DAi[rr + cc * local_size] = A_offd_a[j];
                     }
                  }
               }
            }
         }
      }

      /*
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
      for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
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
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
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

//t2 = hypre_MPI_Wtime() - t2;
//time2 += t2;

      if (local_size > 0)
      {
         /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          * we have Ai and bi build 
          * Solve the linear system by LAPACK : LU factorization
          *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
#if AIR_DEBUG
         memcpy(TMPA, DAi, local_size*local_size*sizeof(HYPRE_Real));
         memcpy(TMPb, Dbi, local_size*sizeof(HYPRE_Real));
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
         HYPRE_Int one = 1;
         HYPRE_Real alp = 1.0, bet = 0.0;
         hypre_dgemv(&charT, &local_size, &local_size, &alp, TMPA, &local_size, Dbi, 
               &one, &bet, TMPd, &one);
         alp = -1.0;
         hypre_daxpy(&local_size, &alp, TMPb, &one, TMPd, &one);
         HYPRE_Real err = hypre_dnrm2(&local_size, TMPd, &one);
         if (err > 1e-8)
         {
            hypre_printf("local res norm %e\n", err);
         }
#endif
      }

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
            R_offd_data[cnt_offd++] = Dbi[rr];
         }
         else /* diag */
         {
            hypre_assert(Marker_diag[i1] == rr);

            /* col idx: use local index i1 */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Dbi[rr];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i'' */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic+1] = cnt_diag;

      R_offd_i[ic+1] = cnt_offd;

//HYPRE_Real t3 = hypre_MPI_Wtime();
      /* RESET marker arrays */
      for (j = 0; j < Marker_diag_count; j++)
      {
         Marker_diag[Marker_diag_j[j]] = -1;
      }

      for (j = 0; j < Marker_FF2_offd_count; j++)
      {
         Marker_FF2_offd[Marker_FF2_offd_j[j]] = -1;
      }

      /* never turn them on !!! */
      /*
      for (j = 0; j < n_fine; j++)
      {
         hypre_assert(Marker_diag[j] == -1);
      }

      for (j = 0; j < FF2_offd_len; j++)
      {
         hypre_assert(Marker_FF2_offd[j] == -1);
      }
      */

//t3 = hypre_MPI_Wtime() - t3;
//time3 += t3;

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */


   hypre_assert(ic == n_cpts)
   hypre_assert(cnt_diag == nnz_diag)
   hypre_assert(cnt_offd == nnz_offd)
   
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

   col_map_offd_R = hypre_CTAlloc(HYPRE_Int, num_cols_offd_R, HYPRE_MEMORY_HOST);
   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the local indices in FF2_offd [will be changed] */
   for (i = 0, i1 = 0; i < FF2_offd_len; i++)
   {
      if (Marker_FF2_offd[i] == 1)
      {
         col_map_offd_R[i1++] = i;
      }
   }

   hypre_assert(i1 == num_cols_offd_R);
   //printf("FF2_offd_len %d, num_cols_offd_R %d\n", FF2_offd_len, num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t FF2_offd
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = hypre_BinarySearch(col_map_offd_R, i1, num_cols_offd_R);
      /* searching must succeed */
      hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      /* change index */
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids [guaranteed to be sorted] */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = FF2_offd[col_map_offd_R[i]];
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
   /* R does not own ColStarts, since A does */
   hypre_ParCSRMatrixOwnsColStarts(R) = 0;
   
   hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   hypre_MatvecCommPkgCreate(R);

   *R_ptr = R;

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
   hypre_TFree(Ipi, HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   hypre_TFree(TMPA, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPb, HYPRE_MEMORY_HOST);
   hypre_TFree(TMPd, HYPRE_MEMORY_HOST);
#endif
   hypre_TFree(RRi, HYPRE_MEMORY_HOST);
   hypre_TFree(KKi, HYPRE_MEMORY_HOST);

   return 0;
}

