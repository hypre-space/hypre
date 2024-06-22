/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker

   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd
   nodes from extend_comm_pkg take up the second chunk of CF_marker_offd. */

HYPRE_Int
hypre_alt_insert_new_nodes(hypre_ParCSRCommPkg  *comm_pkg,
                           hypre_ParCSRCommPkg  *extend_comm_pkg,
                           HYPRE_Int            *IN_marker,
                           HYPRE_Int             full_off_procNodes,
                           HYPRE_Int            *OUT_marker)
{
   HYPRE_UNUSED_VAR(full_off_procNodes);

   hypre_ParCSRCommHandle  *comm_handle;

   HYPRE_Int  i, index, shift;
   HYPRE_Int  num_sends, num_recvs;
   HYPRE_Int *recv_vec_starts;
   HYPRE_Int  e_num_sends;
   HYPRE_Int *int_buf_data;
   HYPRE_Int *e_out_marker;

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


   index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

   int_buf_data = hypre_CTAlloc(HYPRE_Int,  index, HYPRE_MEMORY_HOST);

   /* orig commpkg data*/
   index = 0;

   HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] =
         IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               OUT_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /* now do the extend commpkg */

   /* first we need to shift our position in the OUT_marker */
   shift = recv_vec_starts[num_recvs];
   e_out_marker = OUT_marker + shift;

   index = 0;

   begin = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, 0);
   end = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends);
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] =
         IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, extend_comm_pkg, int_buf_data,
                                               e_out_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_big_insert_new_nodes
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_big_insert_new_nodes(hypre_ParCSRCommPkg  *comm_pkg,
                           hypre_ParCSRCommPkg  *extend_comm_pkg,
                           HYPRE_Int            *IN_marker,
                           HYPRE_Int             full_off_procNodes,
                           HYPRE_BigInt          offset,
                           HYPRE_BigInt         *OUT_marker)
{
   HYPRE_UNUSED_VAR(full_off_procNodes);

   hypre_ParCSRCommHandle  *comm_handle;

   HYPRE_Int                i, index, shift;
   HYPRE_Int                num_sends, num_recvs;
   HYPRE_Int               *recv_vec_starts;
   HYPRE_Int                e_num_sends;
   HYPRE_BigInt            *int_buf_data;
   HYPRE_BigInt            *e_out_marker;

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);

   index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));
   int_buf_data = hypre_CTAlloc(HYPRE_BigInt,  index, HYPRE_MEMORY_HOST);

   /* orig commpkg data*/
   index = 0;

   HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] = offset +
                                (HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, int_buf_data, OUT_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /* now do the extend commpkg */

   /* first we need to shift our position in the OUT_marker */
   shift = recv_vec_starts[num_recvs];
   e_out_marker = OUT_marker + shift;

   index = 0;
   begin = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, 0);
   end   = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends);
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] = offset +
                                (HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 21, extend_comm_pkg, int_buf_data,
                                               e_out_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ssort
 *
 * Sort for non-ordered arrays
 *
 * TODO (VPM): move this to utilities?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ssort(HYPRE_BigInt *data,
            HYPRE_Int     n)
{
   HYPRE_Int i, si;
   HYPRE_Int change = 0;

   if (n > 0)
   {
      for (i = n - 1; i > 0; i--)
      {
         si = hypre_index_of_minimum(data, i + 1);
         if (i != si)
         {
            hypre_swap_int(data, i, si);
            change = 1;
         }
      }
   }

   return change;
}

/*--------------------------------------------------------------------------
 * hypre_index_of_minimum
 *
 * TODO (VPM): move this to utilities?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_index_of_minimum(HYPRE_BigInt *data,
                       HYPRE_Int     n)
{
   HYPRE_Int answer;
   HYPRE_Int i;

   answer = 0;
   for (i = 1; i < n; i++)
   {
      if (data[answer] < data[i])
      {
         answer = i;
      }
   }

   return answer;
}

/*--------------------------------------------------------------------------
 * hypre_swap_int
 *
 * TODO (VPM): move this to utilities?
 *--------------------------------------------------------------------------*/

void
hypre_swap_int(HYPRE_BigInt *data,
               HYPRE_Int     a,
               HYPRE_Int     b)
{
   HYPRE_BigInt temp;

   temp    = data[a];
   data[a] = data[b];
   data[b] = temp;

   return;
}

/*--------------------------------------------------------------------------
 * hypre_initialize_vecs
 *
 * Initialize CF_marker_offd, CF_marker, P_marker, P_marker_offd, tmp
 *--------------------------------------------------------------------------*/

void
hypre_initialize_vecs(HYPRE_Int     diag_n,
                      HYPRE_Int     offd_n,
                      HYPRE_Int    *diag_ftc,
                      HYPRE_BigInt *offd_ftc,
                      HYPRE_Int    *diag_pm,
                      HYPRE_Int    *offd_pm,
                      HYPRE_Int    *tmp_CF)
{
   HYPRE_Int i;

   /* Quicker initialization */
   if (offd_n < diag_n)
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < offd_n; i++)
      {
         diag_ftc[i] = -1;
         offd_ftc[i] = -1;
         tmp_CF[i]   = -1;
         if (diag_pm != NULL)
         {
            diag_pm[i] = -1;
         }
         if (offd_pm != NULL)
         {
            offd_pm[i] = -1;
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = offd_n; i < diag_n; i++)
      {
         diag_ftc[i] = -1;
         if (diag_pm != NULL)
         {
            diag_pm[i] = -1;
         }
      }
   }
   else
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < diag_n; i++)
      {
         diag_ftc[i] = -1;
         offd_ftc[i] = -1;
         tmp_CF[i] = -1;
         if (diag_pm != NULL)
         {
            diag_pm[i] = -1;
         }
         if (offd_pm != NULL)
         {
            offd_pm[i] = -1;
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (i = diag_n; i < offd_n; i++)
      {
         offd_ftc[i] = -1;
         tmp_CF[i] = -1;
         if (offd_pm != NULL)
         {
            offd_pm[i] = -1;
         }
      }
   }
   return;
}

/*--------------------------------------------------------------------------
 * hypre_new_offd_nodes
 *
 * Find nodes that are offd and are not contained in original offd
 * (neighbors of neighbors)
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_new_offd_nodes(HYPRE_BigInt **found,
                     HYPRE_Int      num_cols_A_offd,
                     HYPRE_Int     *A_ext_i,
                     HYPRE_BigInt  *A_ext_j,
                     HYPRE_Int      num_cols_S_offd,
                     HYPRE_BigInt  *col_map_offd,
                     HYPRE_BigInt   col_1,
                     HYPRE_BigInt   col_n,
                     HYPRE_Int     *Sop_i,
                     HYPRE_BigInt  *Sop_j,
                     HYPRE_Int     *CF_marker_offd)
{
   HYPRE_UNUSED_VAR(num_cols_S_offd);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

   HYPRE_BigInt big_i1, big_k1;
   HYPRE_Int i, j, kk;
   HYPRE_Int got_loc, loc_col;

   /*HYPRE_Int min;*/
   HYPRE_Int newoff = 0;

#ifdef HYPRE_CONCURRENT_HOPSCOTCH
   hypre_UnorderedBigIntMap col_map_offd_inverse;
   hypre_UnorderedBigIntMapCreate(&col_map_offd_inverse,
                                  2 * num_cols_A_offd,
                                  16 * hypre_NumThreads());

   #pragma omp parallel for HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_A_offd; i++)
   {
      hypre_UnorderedBigIntMapPutIfAbsent(&col_map_offd_inverse, col_map_offd[i], i);
   }

   /* Find nodes that will be added to the off diag list */
   HYPRE_Int size_offP = A_ext_i[num_cols_A_offd];
   hypre_UnorderedBigIntSet set;
   hypre_UnorderedBigIntSetCreate(&set, size_offP, 16 * hypre_NumThreads());

   #pragma omp parallel private(i,j,big_i1)
   {
      #pragma omp for HYPRE_SMP_SCHEDULE
      for (i = 0; i < num_cols_A_offd; i++)
      {
         if (CF_marker_offd[i] < 0)
         {
            for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
            {
               big_i1 = A_ext_j[j];
               if (big_i1 < col_1 || big_i1 >= col_n)
               {
                  if (!hypre_UnorderedBigIntSetContains(&set, big_i1))
                  {
                     HYPRE_Int k = hypre_UnorderedBigIntMapGet(&col_map_offd_inverse, big_i1);
                     if (-1 == k)
                     {
                        hypre_UnorderedBigIntSetPut(&set, big_i1);
                     }
                     else
                     {
                        A_ext_j[j] = -k - 1;
                     }
                  }
               }
            }
            for (j = Sop_i[i]; j < Sop_i[i + 1]; j++)
            {
               big_i1 = Sop_j[j];
               if (big_i1 < col_1 || big_i1 >= col_n)
               {
                  if (!hypre_UnorderedBigIntSetContains(&set, big_i1))
                  {
                     HYPRE_Int k = hypre_UnorderedBigIntMapGet(&col_map_offd_inverse, big_i1);
                     if (-1 == k)
                     {
                        hypre_UnorderedBigIntSetPut(&set, big_i1);
                     }
                     else
                     {
                        Sop_j[j] = -k - 1;
                     }
                  }
               }
            }
         } /* CF_marker_offd[i] < 0 */
      } /* for each row */
   } /* omp parallel */

   hypre_UnorderedBigIntMapDestroy(&col_map_offd_inverse);
   HYPRE_BigInt *tmp_found = hypre_UnorderedBigIntSetCopyToArray(&set, &newoff);
   hypre_UnorderedBigIntSetDestroy(&set);

   /* Put found in monotone increasing order */
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

   hypre_UnorderedBigIntMap tmp_found_inverse;
   if (newoff > 0)
   {
      hypre_big_sort_and_create_inverse_map(tmp_found, newoff, &tmp_found, &tmp_found_inverse);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif

   /* Set column indices for Sop and A_ext such that offd nodes are
    * negatively indexed */
   #pragma omp parallel for private(kk,big_k1,got_loc,loc_col) HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (kk = Sop_i[i]; kk < Sop_i[i + 1]; kk++)
         {
            big_k1 = Sop_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_UnorderedBigIntMapGet(&tmp_found_inverse, big_k1);
               loc_col = got_loc + num_cols_A_offd;
               Sop_j[kk] = (HYPRE_BigInt)(-loc_col - 1);
            }
         }
         for (kk = A_ext_i[i]; kk < A_ext_i[i + 1]; kk++)
         {
            big_k1 = A_ext_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_UnorderedBigIntMapGet(&tmp_found_inverse, big_k1);
               loc_col = got_loc + num_cols_A_offd;
               A_ext_j[kk] = (HYPRE_BigInt)(-loc_col - 1);
            }
         }
      }
   }
   if (newoff)
   {
      hypre_UnorderedBigIntMapDestroy(&tmp_found_inverse);
   }
#else /* !HYPRE_CONCURRENT_HOPSCOTCH */
   HYPRE_Int size_offP;

   HYPRE_BigInt *tmp_found;
   HYPRE_Int min;
   HYPRE_Int ifound;
   HYPRE_BigInt ifound_big;

   size_offP = A_ext_i[num_cols_A_offd] + Sop_i[num_cols_A_offd];
   tmp_found = hypre_CTAlloc(HYPRE_BigInt, size_offP, HYPRE_MEMORY_HOST);

   /* Find nodes that will be added to the off diag list */
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
         {
            big_i1 = A_ext_j[j];
            if (big_i1 < col_1 || big_i1 >= col_n)
            {
               ifound = hypre_BigBinarySearch(col_map_offd, big_i1, num_cols_A_offd);
               if (ifound == -1)
               {
                  tmp_found[newoff] = big_i1;
                  newoff++;
               }
               else
               {
                  A_ext_j[j] = (HYPRE_BigInt)(-ifound - 1);
               }
            }
         }
         for (j = Sop_i[i]; j < Sop_i[i + 1]; j++)
         {
            big_i1 = Sop_j[j];
            if (big_i1 < col_1 || big_i1 >= col_n)
            {
               ifound = hypre_BigBinarySearch(col_map_offd, big_i1, num_cols_A_offd);
               if (ifound == -1)
               {
                  tmp_found[newoff] = big_i1;
                  newoff++;
               }
               else
               {
                  Sop_j[j] = (HYPRE_BigInt)(-ifound - 1);
               }
            }
         }
      }
   }
   /* Put found in monotone increasing order */
   if (newoff > 0)
   {
      hypre_BigQsort0(tmp_found, 0, newoff - 1);
      ifound_big = tmp_found[0];
      min = 1;
      for (i = 1; i < newoff; i++)
      {
         if (tmp_found[i] > ifound_big)
         {
            ifound_big = tmp_found[i];
            tmp_found[min++] = ifound_big;
         }
      }
      newoff = min;
   }

   /* Set column indices for Sop and A_ext such that offd nodes are
    * negatively indexed */
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (kk = Sop_i[i]; kk < Sop_i[i + 1]; kk++)
         {
            big_k1 = Sop_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_BigBinarySearch(tmp_found, big_k1, newoff);
               if (got_loc > -1)
               {
                  loc_col = got_loc + num_cols_A_offd;
                  Sop_j[kk] = (HYPRE_BigInt)(-loc_col - 1);
               }
            }
         }
         for (kk = A_ext_i[i]; kk < A_ext_i[i + 1]; kk++)
         {
            big_k1 = A_ext_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_BigBinarySearch(tmp_found, big_k1, newoff);
               if (got_loc > -1)
               {
                  loc_col = got_loc + num_cols_A_offd;
                  A_ext_j[kk] = (HYPRE_BigInt)(-loc_col - 1);
               }
            }
         }
      }
   }
#endif /* !HYPRE_CONCURRENT_HOPSCOTCH */

   *found = tmp_found;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif

   return newoff;
}

/*--------------------------------------------------------------------------
 * hypre_exchange_marker
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg,
                      HYPRE_Int           *IN_marker,
                      HYPRE_Int           *OUT_marker)
{
   HYPRE_Int               num_sends    = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int               begin        = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   HYPRE_Int               end          = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   HYPRE_Int              *int_buf_data = hypre_CTAlloc(HYPRE_Int, end, HYPRE_MEMORY_HOST);

   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Int               i;


#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] = IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, OUT_marker);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_exchange_interp_data
 *
 * skip_fine_or_same_sign: if we want to skip fine points in S and nnz with
 *                         the same sign as diagonal in A
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_exchange_interp_data(HYPRE_Int           **CF_marker_offd,
                           HYPRE_Int           **dof_func_offd,
                           hypre_CSRMatrix     **A_ext,
                           HYPRE_Int            *full_off_procNodes,
                           hypre_CSRMatrix     **Sop,
                           hypre_ParCSRCommPkg **extend_comm_pkg,
                           hypre_ParCSRMatrix   *A,
                           HYPRE_Int            *CF_marker,
                           hypre_ParCSRMatrix   *S,
                           HYPRE_Int             num_functions,
                           HYPRE_Int            *dof_func,
                           HYPRE_Int             skip_fine_or_same_sign)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif

   hypre_ParCSRCommPkg    *comm_pkg        = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix        *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix        *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int               num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_BigInt           *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt            col_1           = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int               local_numrows   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt            col_n           = col_1 + (HYPRE_BigInt) local_numrows;

   HYPRE_BigInt           *found           = NULL;
   hypre_ParCSRCommHandle *comm_handle_s_idx;

   /*----------------------------------------------------------------------
    * Get the off processors rows for A and S, associated with columns in
    * A_offd and S_offd.
    *---------------------------------------------------------------------*/
   *CF_marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
   hypre_exchange_marker(comm_pkg, CF_marker, *CF_marker_offd);

   hypre_ParCSRCommHandle *comm_handle_a_idx, *comm_handle_a_data;
   *A_ext = hypre_ParCSRMatrixExtractBExt_Overlap(A, A, 1, &comm_handle_a_idx,
                                                  &comm_handle_a_data,
                                                  CF_marker, *CF_marker_offd,
                                                  skip_fine_or_same_sign,
                                                  skip_fine_or_same_sign);
   HYPRE_Int    *A_ext_i    = hypre_CSRMatrixI(*A_ext);
   HYPRE_BigInt *A_ext_j    = hypre_CSRMatrixBigJ(*A_ext);
   HYPRE_Int     A_ext_rows = hypre_CSRMatrixNumRows(*A_ext);

   *Sop = hypre_ParCSRMatrixExtractBExt_Overlap(S, A, 0, &comm_handle_s_idx, NULL, CF_marker,
                                                *CF_marker_offd, skip_fine_or_same_sign, 0);

   HYPRE_Int    *Sop_i       = hypre_CSRMatrixI(*Sop);
   HYPRE_BigInt *Sop_j       = hypre_CSRMatrixBigJ(*Sop);
   HYPRE_Int     Soprows     = hypre_CSRMatrixNumRows(*Sop);
   HYPRE_Int    *send_idx    = (HYPRE_Int *) comm_handle_s_idx->send_data;

   hypre_ParCSRCommHandleDestroy(comm_handle_s_idx);
   hypre_TFree(send_idx, HYPRE_MEMORY_HOST);

   send_idx = (HYPRE_Int *)comm_handle_a_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_a_idx);
   hypre_TFree(send_idx, HYPRE_MEMORY_HOST);

   /* Find nodes that are neighbors of neighbors, not found in offd */
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif
   HYPRE_Int newoff = hypre_new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j,
                                           Soprows, col_map_offd, col_1, col_n,
                                           Sop_i, Sop_j, *CF_marker_offd);
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif
   if (newoff >= 0)
   {
      *full_off_procNodes = newoff + num_cols_A_offd;
   }
   else
   {
      return hypre_error_flag;
   }

   /* Possibly add new points and new processors to the comm_pkg, all
    * processors need new_comm_pkg */

   /* AHB - create a new comm package just for extended info -
      this will work better with the assumed partition*/
   hypre_ParCSRFindExtendCommPkg(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixFirstColDiag(A),
                                 hypre_CSRMatrixNumCols(A_diag),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_ParCSRMatrixAssumedPartition(A),
                                 newoff,
                                 found,
                                 extend_comm_pkg);

   *CF_marker_offd = hypre_TReAlloc(*CF_marker_offd, HYPRE_Int, *full_off_procNodes,
                                    HYPRE_MEMORY_HOST);
   hypre_exchange_marker(*extend_comm_pkg, CF_marker, *CF_marker_offd + A_ext_rows);

   if (num_functions > 1)
   {
      if (*full_off_procNodes > 0)
      {
         *dof_func_offd = hypre_CTAlloc(HYPRE_Int, *full_off_procNodes, HYPRE_MEMORY_HOST);
      }

      hypre_alt_insert_new_nodes(comm_pkg, *extend_comm_pkg, dof_func,
                                 *full_off_procNodes, *dof_func_offd);
   }

   hypre_TFree(found, HYPRE_MEMORY_HOST);

   HYPRE_Real *send_data = (HYPRE_Real *)comm_handle_a_data->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_a_data);
   hypre_TFree(send_data, HYPRE_MEMORY_HOST);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_build_interp_colmap
 *--------------------------------------------------------------------------*/

void
hypre_build_interp_colmap(hypre_ParCSRMatrix *P,
                          HYPRE_Int           full_off_procNodes,
                          HYPRE_Int          *tmp_CF_marker_offd,
                          HYPRE_BigInt       *fine_to_coarse_offd)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif
   HYPRE_Int     n_fine = hypre_CSRMatrixNumRows(P->diag);

   HYPRE_Int     P_offd_size = P->offd->i[n_fine];
   HYPRE_Int    *P_offd_j = P->offd->j;
   HYPRE_BigInt *col_map_offd_P = NULL;
   HYPRE_Int    *P_marker = NULL;
   HYPRE_Int    *prefix_sum_workspace;
   HYPRE_Int     num_cols_P_offd = 0;
   HYPRE_Int     i, index;

   if (full_off_procNodes)
   {
      P_marker = hypre_TAlloc(HYPRE_Int, full_off_procNodes, HYPRE_MEMORY_HOST);
   }
   prefix_sum_workspace = hypre_TAlloc(HYPRE_Int, hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker[i] = 0;
   }

   /* These two loops set P_marker[i] to 1 if it appears in P_offd_j and if
    * tmp_CF_marker_offd has i marked. num_cols_P_offd is then set to the
    * total number of times P_marker is set */
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,index) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < P_offd_size; i++)
   {
      index = P_offd_j[i];
      if (tmp_CF_marker_offd[index] >= 0)
      {
         P_marker[index] = 1;
      }
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i)
#endif
   {
      HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, full_off_procNodes);

      HYPRE_Int local_num_cols_P_offd = 0;
      for (i = i_begin; i < i_end; i++)
      {
         if (P_marker[i] == 1) { local_num_cols_P_offd++; }
      }

      hypre_prefix_sum(&local_num_cols_P_offd, &num_cols_P_offd, prefix_sum_workspace);

#ifdef HYPRE_USING_OPENMP
      #pragma omp master
#endif
      {
         if (num_cols_P_offd)
         {
            col_map_offd_P = hypre_TAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (P_marker[i] == 1)
         {
            col_map_offd_P[local_num_cols_P_offd++] = fine_to_coarse_offd[i];
         }
      }
   }

   hypre_UnorderedBigIntMap col_map_offd_P_inverse;
   hypre_big_sort_and_create_inverse_map(col_map_offd_P, num_cols_P_offd, &col_map_offd_P,
                                         &col_map_offd_P_inverse);

   // find old idx -> new idx map
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for
#endif
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker[i] = hypre_UnorderedBigIntMapGet(&col_map_offd_P_inverse, fine_to_coarse_offd[i]);
   }

   if (num_cols_P_offd)
   {
      hypre_UnorderedBigIntMapDestroy(&col_map_offd_P_inverse);
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for
#endif
   for (i = 0; i < P_offd_size; i++)
   {
      P_offd_j[i] = P_marker[P_offd_j[i]];
   }

   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(prefix_sum_workspace, HYPRE_MEMORY_HOST);

   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P->offd) = num_cols_P_offd;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif
}
