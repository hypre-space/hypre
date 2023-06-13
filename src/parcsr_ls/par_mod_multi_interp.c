/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGBuildModMultipass
 * This routine implements Stuben's direct interpolation with multiple passes.
 * expressed with matrix matrix multiplications
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildModMultipassHost( hypre_ParCSRMatrix  *A,
                                      HYPRE_Int           *CF_marker,
                                      hypre_ParCSRMatrix  *S,
                                      HYPRE_BigInt        *num_cpts_global,
                                      HYPRE_Real           trunc_factor,
                                      HYPRE_Int            P_max_elmts,
                                      HYPRE_Int            interp_type,
                                      HYPRE_Int            num_functions,
                                      HYPRE_Int           *dof_func,
                                      hypre_ParCSRMatrix **P_ptr )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MULTIPASS_INTERP] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);

   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   hypre_ParCSRMatrix **Pi;
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix *P_diag;
   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *P_diag_j;

   hypre_CSRMatrix *P_offd;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *P_offd_j = NULL;
   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_Int        num_cols_offd_P = 0;

   HYPRE_Int        num_sends = 0;
   HYPRE_Int       *int_buf_data = NULL;

   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *points_left;
   HYPRE_Int       *pass_marker;
   HYPRE_Int       *pass_marker_offd = NULL;
   HYPRE_Int       *pass_order;
   HYPRE_Int       *pass_starts;

   HYPRE_Int        i, j, i1, i2, j1;
   HYPRE_Int        num_passes, p;
   HYPRE_BigInt     global_remaining, remaining;
   HYPRE_Int        cnt, cnt_old, cnt_rem, current_pass;
   HYPRE_Int        startc, index;

   HYPRE_BigInt     total_global_cpts;
   HYPRE_Int        my_id, num_procs;
   HYPRE_Int        P_offd_size = 0;

   HYPRE_Int       *dof_func_offd = NULL;
   HYPRE_Real      *row_sums = NULL;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      if (my_id == num_procs - 1)
      {
         total_global_cpts = num_cpts_global[1];
      }
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      total_global_cpts = num_cpts_global[1];
   }

   if (total_global_cpts == 0)
   {
      *P_ptr = NULL;
      return hypre_error_flag;
   }
   /* Generate pass marker array */

   pass_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   /* contains pass numbers for each variable according to original order */
   pass_order = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   /* contains row numbers according to new order, pass 1 followed by pass 2 etc */
   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   /* reverse of pass_order, keeps track where original numbers go */
   points_left = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   /* contains row numbers of remaining points, auxiliary */
   pass_starts = hypre_CTAlloc(HYPRE_Int, 11, HYPRE_MEMORY_HOST);
   /* contains beginning for each pass in pass_order field, assume no more than 10 passes */

   P_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   cnt = 0;
   remaining = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == 1)
      {
         pass_marker[i] = 1;
         P_diag_i[i + 1] = 1;
         P_offd_i[i + 1] = 0;
         fine_to_coarse[i] = cnt;
         pass_order[cnt++] = i;
      }
      else
      {
         points_left[remaining++] = i;
      }
   }
   pass_starts[0] = 0;
   pass_starts[1] = cnt;

   if (num_functions > 1)
   {
      dof_func_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
      index = 0;
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);

      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

   if (num_procs > 1)
   {
      pass_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
      index = 0;
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = pass_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, pass_marker_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
   }
   current_pass = 1;
   num_passes = 1;
   /* color points according to pass number */
   hypre_MPI_Allreduce(&remaining, &global_remaining, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   while (global_remaining > 0)
   {
      HYPRE_Int remaining_pts = (HYPRE_Int) remaining;
      HYPRE_BigInt old_global_remaining = global_remaining;
      cnt_rem = 0;
      for (i = 0; i < remaining_pts; i++)
      {
         i1 = points_left[i];
         cnt_old = cnt;
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            if (pass_marker[j1] == current_pass)
            {
               pass_marker[i1] = current_pass + 1;
               pass_order[cnt++] = i1;
               remaining--;
               break;
            }
         }
         if (cnt == cnt_old)
         {
            for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
            {
               j1 = S_offd_j[j];
               if (pass_marker_offd[j1] == current_pass)
               {
                  pass_marker[i1] = current_pass + 1;
                  pass_order[cnt++] = i1;
                  remaining--;
                  break;
               }
            }
         }
         if (cnt == cnt_old)
         {
            points_left[cnt_rem++] = i1;
         }
      }
      remaining = (HYPRE_BigInt) cnt_rem;
      current_pass++;
      num_passes++;
      if (num_passes > 9)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Warning!!! too many passes! out of range!\n");
         break;
      }
      pass_starts[num_passes] = cnt;
      /* update pass_marker_offd */
      index = 0;
      if (num_procs > 1)
      {
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index++] = pass_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, pass_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
      old_global_remaining = global_remaining;
      hypre_MPI_Allreduce(&remaining, &global_remaining, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      /* if the number of remaining points does not change, we have a situation of isolated areas of
       * fine points that are not connected to any C-points, and the pass generation process breaks
       * down. Those points can be ignored, i.e. the corresponding rows in P will just be 0
       * and can be ignored for the algorithm. */
      if (old_global_remaining == global_remaining) { break; }
   }
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(points_left, HYPRE_MEMORY_HOST);

   /* generate row sum of weak points and C-points to be ignored */

   row_sums = hypre_CTAlloc(HYPRE_Real, n_fine, HYPRE_MEMORY_HOST);
   if (num_functions >  1)
   {
      for (i = 0; i < n_fine; i++)
      {
         if (CF_marker[i] < 0)
         {
            for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
            {
               if (dof_func[i] == dof_func[A_diag_j[j]])
               {
                  row_sums[i] += A_diag_data[j];
               }
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[j]])
               {
                  row_sums[i] += A_offd_data[j];
               }
            }
         }
      }
   }
   else
   {
      for (i = 0; i < n_fine; i++)
      {
         if (CF_marker[i] < 0)
         {
            for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
            {
               row_sums[i] += A_diag_data[j];
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               row_sums[i] += A_offd_data[j];
            }
         }
      }
   }

   Pi = hypre_CTAlloc(hypre_ParCSRMatrix*, num_passes, HYPRE_MEMORY_HOST);
   hypre_GenerateMultipassPi(A, S, num_cpts_global, &pass_order[pass_starts[1]], pass_marker,
                             pass_marker_offd, pass_starts[2] - pass_starts[1], 1, row_sums, &Pi[0]);
   if (interp_type == 8)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         hypre_ParCSRMatrix *Q;
         HYPRE_BigInt *c_pts_starts = hypre_ParCSRMatrixRowStarts(Pi[i - 1]);
         hypre_GenerateMultipassPi(A, S, c_pts_starts, &pass_order[pass_starts[i + 1]], pass_marker,
                                   pass_marker_offd, pass_starts[i + 2] - pass_starts[i + 1], i + 1, row_sums, &Q);
         Pi[i] = hypre_ParMatmul(Q, Pi[i - 1]);
         hypre_ParCSRMatrixDestroy(Q);
      }
   }
   else if (interp_type == 9)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         HYPRE_BigInt *c_pts_starts = hypre_ParCSRMatrixRowStarts(Pi[i - 1]);
         hypre_GenerateMultiPi(A, S, Pi[i - 1], c_pts_starts, &pass_order[pass_starts[i + 1]], pass_marker,
                               pass_marker_offd, pass_starts[i + 2] - pass_starts[i + 1], i + 1,
                               num_functions, dof_func, dof_func_offd, &Pi[i]);
      }
   }

   /* p pulate P_diag_i[i+1] with nnz of i-th row */
   for (i = 0; i < num_passes - 1; i++)
   {
      HYPRE_Int *Pi_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(Pi[i]));
      HYPRE_Int *Pi_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(Pi[i]));
      j1 = 0;
      for (j = pass_starts[i + 1]; j < pass_starts[i + 2]; j++)
      {
         i1 = pass_order[j];
         P_diag_i[i1 + 1] = Pi_diag_i[j1 + 1] - Pi_diag_i[j1];
         P_offd_i[i1 + 1] = Pi_offd_i[j1 + 1] - Pi_offd_i[j1];
         j1++;
      }
   }

   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i + 1] += P_diag_i[i];
      P_offd_i[i + 1] += P_offd_i[i];
   }

   P_diag_j = hypre_CTAlloc(HYPRE_Int, P_diag_i[n_fine], memory_location_P);
   P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_i[n_fine], memory_location_P);
   P_offd_j = hypre_CTAlloc(HYPRE_Int, P_offd_i[n_fine], memory_location_P);
   P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_i[n_fine], memory_location_P);

   /* insert weights for coarse points */
   for (i = 0; i < pass_starts[1]; i++)
   {
      i1 = pass_order[i];
      j = P_diag_i[i1];
      P_diag_j[j] = fine_to_coarse[i1];
      P_diag_data[j] = 1.0;
   }

   /* generate col_map_offd_P by combining all col_map_offd_Pi
    * and reompute indices if needed */

   /* insert remaining weights */
   for (p = 0; p < num_passes - 1; p++)
   {
      HYPRE_Int *Pi_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Int *Pi_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(Pi[p]));
      HYPRE_Int *Pi_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Int *Pi_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(Pi[p]));
      HYPRE_Real *Pi_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(Pi[p]));
      HYPRE_Real *Pi_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(Pi[p]));
      j1 = 0;
      for (i = pass_starts[p + 1]; i < pass_starts[p + 2]; i++)
      {
         i1 = pass_order[i];
         i2 = Pi_diag_i[j1];
         for (j = P_diag_i[i1]; j < P_diag_i[i1 + 1]; j++)
         {
            P_diag_j[j] = Pi_diag_j[i2];
            P_diag_data[j] = Pi_diag_data[i2++];
         }
         i2 = Pi_offd_i[j1];
         for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
         {
            P_offd_j[j] = Pi_offd_j[i2];
            P_offd_data[j] = Pi_offd_data[i2++];
         }
         j1++;
      }
   }
   /* Note that col indices in P_offd_j probably not consistent,
      this gets fixed after truncation */

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixRowStarts(A),
                                num_cpts_global,
                                num_cols_offd_P,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || P_max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, P_max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
   }

   num_cols_offd_P = 0;
   P_offd_size = P_offd_i[n_fine];
   if (P_offd_size)
   {
      HYPRE_BigInt *tmp_P_offd_j = hypre_CTAlloc(HYPRE_BigInt, P_offd_size, HYPRE_MEMORY_HOST);
      HYPRE_BigInt *big_P_offd_j = hypre_CTAlloc(HYPRE_BigInt, P_offd_size, HYPRE_MEMORY_HOST);
      for (p = 0; p < num_passes - 1; p++)
      {
         HYPRE_BigInt *col_map_offd_Pi = hypre_ParCSRMatrixColMapOffd(Pi[p]);
         for (i = pass_starts[p + 1]; i < pass_starts[p + 2]; i++)
         {
            i1 = pass_order[i];
            for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
            {
               big_P_offd_j[j] = col_map_offd_Pi[P_offd_j[j]];
            }
         }
      }

      for (i = 0; i < P_offd_size; i++)
      {
         tmp_P_offd_j[i] = big_P_offd_j[i];
      }

      hypre_BigQsort0(tmp_P_offd_j, 0, P_offd_size - 1);

      num_cols_offd_P = 1;
      for (i = 0; i < P_offd_size - 1; i++)
      {
         if (tmp_P_offd_j[i + 1] > tmp_P_offd_j[i])
         {
            tmp_P_offd_j[num_cols_offd_P++] = tmp_P_offd_j[i + 1];
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_offd_P; i++)
      {
         col_map_offd_P[i] = tmp_P_offd_j[i];
      }

      for (i = 0; i < P_offd_size; i++)
      {
         P_offd_j[i] = hypre_BigBinarySearch(col_map_offd_P,
                                             big_P_offd_j[i],
                                             num_cols_offd_P);
      }
      hypre_TFree(tmp_P_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(big_P_offd_j, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_passes - 1; i++)
   {
      hypre_ParCSRMatrixDestroy(Pi[i]);
   }
   hypre_TFree (Pi, HYPRE_MEMORY_HOST);
   hypre_TFree (pass_marker, HYPRE_MEMORY_HOST);
   hypre_TFree (pass_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree (pass_order, HYPRE_MEMORY_HOST);
   hypre_TFree (pass_starts, HYPRE_MEMORY_HOST);
   hypre_TFree (fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree (dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree (row_sums, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   }

   hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   hypre_CSRMatrixNumCols(P_offd) = num_cols_offd_P;

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   return hypre_error_flag;

}


HYPRE_Int
hypre_GenerateMultipassPi( hypre_ParCSRMatrix  *A,
                           hypre_ParCSRMatrix  *S,
                           HYPRE_BigInt        *c_pts_starts,
                           HYPRE_Int
                           *pass_order, /* array containing row numbers of rows in A and S to be considered */
                           HYPRE_Int           *pass_marker,
                           HYPRE_Int           *pass_marker_offd,
                           HYPRE_Int            num_points,
                           HYPRE_Int            color,
                           HYPRE_Real          *row_sums,
                           hypre_ParCSRMatrix **P_ptr )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_Int        num_cols_offd_P;
   HYPRE_Int        nnz_diag, nnz_offd;
   HYPRE_Int        n_cpts, i, j, i1, j1, j2;
   HYPRE_Int        startc, index;
   HYPRE_Int        cpt, cnt_diag, cnt_offd;

   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix *P_diag;
   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *P_diag_j;

   hypre_CSRMatrix *P_offd;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *P_offd_j = NULL;
   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *fine_to_coarse_offd = NULL;
   HYPRE_BigInt     f_pts_starts[2];
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt     total_global_fpts;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt    *big_convert;
   HYPRE_BigInt    *big_convert_offd = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;
   HYPRE_Int        num_sends;
   HYPRE_Real      *row_sum_C;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* define P matrices */

   P_diag_i = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_HOST);
   P_offd_i = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_HOST);
   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);

   /* fill P */

   n_cpts = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (pass_marker[i] == color)
      {
         fine_to_coarse[i] = n_cpts++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

   if (num_procs > 1)
   {
      HYPRE_BigInt big_Fpts;
      big_Fpts = num_points;

      hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   {
      big_convert = hypre_CTAlloc(HYPRE_BigInt, n_fine, HYPRE_MEMORY_HOST);
      for (i = 0; i < n_fine; i++)
      {
         if (pass_marker[i] == color)
         {
            big_convert[i] = (HYPRE_BigInt)fine_to_coarse[i] + c_pts_starts[0];
         }
      }

      num_cols_offd_P = 0;
      if (num_procs > 1)
      {
         big_convert_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_HOST);
         fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
         index = 0;
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         big_buf_data = hypre_CTAlloc(HYPRE_BigInt, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               big_buf_data[index++] = big_convert[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, big_convert_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_P = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               fine_to_coarse_offd[i] = num_cols_offd_P++;
            }
         }

         col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_P, HYPRE_MEMORY_HOST);

         cpt = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               col_map_offd_P[cpt++] = big_convert_offd[i];
            }
         }
      }
   }

   /* generate P_diag_i and P_offd_i */
   nnz_diag = 0;
   nnz_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
            P_diag_i[i + 1]++;
            nnz_diag++;
         }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
            P_offd_i[i + 1]++;
            nnz_offd++;
         }
      }
   }

   for (i = 1; i < num_points + 1; i++)
   {
      P_diag_i[i] += P_diag_i[i - 1];
      P_offd_i[i] += P_offd_i[i - 1];
   }

   P_diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag, HYPRE_MEMORY_HOST);
   P_diag_data = hypre_CTAlloc(HYPRE_Real, nnz_diag, HYPRE_MEMORY_HOST);
   P_offd_j = hypre_CTAlloc(HYPRE_Int, nnz_offd, HYPRE_MEMORY_HOST);
   P_offd_data = hypre_CTAlloc(HYPRE_Real, nnz_offd, HYPRE_MEMORY_HOST);

   cnt_diag = 0;
   cnt_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      j2 = A_diag_i[i1];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         while (A_diag_j[j2] != j1) { j2++; }
         if (pass_marker[j1] == color && A_diag_j[j2] == j1)
         {
            P_diag_j[cnt_diag] = fine_to_coarse[j1];
            P_diag_data[cnt_diag++] = A_diag_data[j2];
         }
      }
      j2 = A_offd_i[i1];
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         while (A_offd_j[j2] != j1) { j2++; }
         if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
         {
            P_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
            P_offd_data[cnt_offd++] = A_offd_data[j2];
         }
      }
   }

   //row_sums = hypre_CTAlloc(HYPRE_Real, num_points, HYPRE_MEMORY_HOST);
   row_sum_C = hypre_CTAlloc(HYPRE_Real, num_points, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_points; i++)
   {
      HYPRE_Real diagonal, value;
      i1 = pass_order[i];
      diagonal = A_diag_data[A_diag_i[i1]];
      /*for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
      {
         row_sums[i] += A_diag_data[j];
      }
      for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
      {
         row_sums[i] += A_offd_data[j];
      }*/
      for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
      {
         row_sum_C[i] += P_diag_data[j];
      }
      for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
      {
         row_sum_C[i] += P_offd_data[j];
      }
      value = row_sum_C[i] * diagonal;
      if (value != 0)
      {
         row_sums[i1] /= value;
      }
      for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
      {
         P_diag_data[j] = -P_diag_data[j] * row_sums[i1];
      }
      for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
      {
         P_offd_data[j] = -P_offd_data[j] * row_sums[i1];
      }
   }


   P = hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_P,
                                P_diag_i[num_points],
                                P_offd_i[num_points]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;

   hypre_CSRMatrixMemoryLocation(P_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(P_offd) = HYPRE_MEMORY_HOST;

   /* free stuff */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   //hypre_TFree(row_sums, HYPRE_MEMORY_HOST);
   hypre_TFree(row_sum_C, HYPRE_MEMORY_HOST);
   hypre_TFree(big_convert, HYPRE_MEMORY_HOST);
   hypre_TFree(big_convert_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);

   hypre_MatvecCommPkgCreate(P);
   *P_ptr = P;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GenerateMultiPi( hypre_ParCSRMatrix  *A,
                       hypre_ParCSRMatrix  *S,
                       hypre_ParCSRMatrix  *P,
                       HYPRE_BigInt        *c_pts_starts,
                       HYPRE_Int
                       *pass_order, /* array containing row numbers of rows in A and S to be considered */
                       HYPRE_Int           *pass_marker,
                       HYPRE_Int           *pass_marker_offd,
                       HYPRE_Int            num_points,
                       HYPRE_Int            color,
                       HYPRE_Int            num_functions,
                       HYPRE_Int           *dof_func,
                       HYPRE_Int           *dof_func_offd,
                       hypre_ParCSRMatrix **Pi_ptr )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int        num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_BigInt    *col_map_offd_Q = NULL;
   HYPRE_Int        num_cols_offd_Q;

   hypre_ParCSRMatrix *Pi;
   hypre_CSRMatrix *Pi_diag;
   HYPRE_Int       *Pi_diag_i;
   HYPRE_Real      *Pi_diag_data;

   hypre_CSRMatrix *Pi_offd;
   HYPRE_Int       *Pi_offd_i;
   HYPRE_Real      *Pi_offd_data;

   HYPRE_Int        nnz_diag, nnz_offd;
   HYPRE_Int        n_cpts, i, j, i1, j1, j2;
   HYPRE_Int        startc, index;
   HYPRE_Int        cpt, cnt_diag, cnt_offd;

   hypre_ParCSRMatrix *Q;
   hypre_CSRMatrix *Q_diag;
   HYPRE_Real      *Q_diag_data;
   HYPRE_Int       *Q_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *Q_diag_j;

   hypre_CSRMatrix *Q_offd;
   HYPRE_Real      *Q_offd_data = NULL;
   HYPRE_Int       *Q_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   HYPRE_Int       *Q_offd_j = NULL;
   HYPRE_Int       *fine_to_coarse;
   HYPRE_Int       *fine_to_coarse_offd = NULL;
   HYPRE_BigInt     f_pts_starts[2];
   HYPRE_Int        my_id, num_procs;
   HYPRE_BigInt     total_global_fpts;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt    *big_convert;
   HYPRE_BigInt    *big_convert_offd = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;
   HYPRE_Int        num_sends;
   //HYPRE_Real      *row_sums;
   HYPRE_Real      *row_sums_C;
   HYPRE_Real      *w_row_sum;

   /* MPI size and rank*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* define P matrices */

   Q_diag_i = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_HOST);
   Q_offd_i = hypre_CTAlloc(HYPRE_Int, num_points + 1, HYPRE_MEMORY_HOST);
   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);

   /* fill P */

   n_cpts = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (pass_marker[i] == color)
      {
         fine_to_coarse[i] = n_cpts++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

   if (num_procs > 1)
   {
      HYPRE_BigInt big_Fpts;
      big_Fpts = num_points;

      hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      hypre_MPI_Bcast(&total_global_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   {
      big_convert = hypre_CTAlloc(HYPRE_BigInt, n_fine, HYPRE_MEMORY_HOST);
      for (i = 0; i < n_fine; i++)
      {
         if (pass_marker[i] == color)
         {
            big_convert[i] = (HYPRE_BigInt)fine_to_coarse[i] + c_pts_starts[0];
         }
      }

      num_cols_offd_Q = 0;
      if (num_procs > 1)
      {
         big_convert_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_HOST);
         fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
         index = 0;
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         big_buf_data = hypre_CTAlloc(HYPRE_BigInt,  hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               big_buf_data[index++] = big_convert[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, big_convert_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_Q = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               fine_to_coarse_offd[i] = num_cols_offd_Q++;
            }
         }

         col_map_offd_Q = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_Q, HYPRE_MEMORY_HOST);

         cpt = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               col_map_offd_Q[cpt++] = big_convert_offd[i];
            }
         }
      }
   }

   /* generate Q_diag_i and Q_offd_i */
   nnz_diag = 0;
   nnz_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
            Q_diag_i[i + 1]++;
            nnz_diag++;
         }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
            Q_offd_i[i + 1]++;
            nnz_offd++;
         }
      }
   }

   for (i = 1; i < num_points + 1; i++)
   {
      Q_diag_i[i] += Q_diag_i[i - 1];
      Q_offd_i[i] += Q_offd_i[i - 1];
   }

   Q_diag_j = hypre_CTAlloc(HYPRE_Int, nnz_diag, HYPRE_MEMORY_HOST);
   Q_diag_data = hypre_CTAlloc(HYPRE_Real, nnz_diag, HYPRE_MEMORY_HOST);
   Q_offd_j = hypre_CTAlloc(HYPRE_Int, nnz_offd, HYPRE_MEMORY_HOST);
   Q_offd_data = hypre_CTAlloc(HYPRE_Real, nnz_offd, HYPRE_MEMORY_HOST);
   w_row_sum = hypre_CTAlloc(HYPRE_Real, num_points, HYPRE_MEMORY_HOST);

   cnt_diag = 0;
   cnt_offd = 0;
   if (num_functions > 1)
   {
      for (i = 0; i < num_points; i++)
      {
         i1 = pass_order[i];
         j2 = A_diag_i[i1] + 1;
         //if (w_row_minus) w_row_sum[i] = -w_row_minus[i1];
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            while (A_diag_j[j2] != j1)
            {
               if (dof_func[i1] == dof_func[A_diag_j[j2]])
               {
                  w_row_sum[i] += A_diag_data[j2];
               }
               j2++;
            }
            if (pass_marker[j1] == color && A_diag_j[j2] == j1)
            {
               Q_diag_j[cnt_diag] = fine_to_coarse[j1];
               Q_diag_data[cnt_diag++] = A_diag_data[j2++];
            }
            else
            {
               if (dof_func[i1] == dof_func[A_diag_j[j2]])
               {
                  w_row_sum[i] += A_diag_data[j2];
               }
               j2++;
            }
         }
         while (j2 < A_diag_i[i1 + 1])
         {
            if (dof_func[i1] == dof_func[A_diag_j[j2]])
            {
               w_row_sum[i] += A_diag_data[j2];
            }
            j2++;
         }
         j2 = A_offd_i[i1];
         for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
         {
            j1 = S_offd_j[j];
            while (A_offd_j[j2] != j1)
            {
               if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
               {
                  w_row_sum[i] += A_offd_data[j2];
               }
               j2++;
            }
            if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
            {
               Q_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
               Q_offd_data[cnt_offd++] = A_offd_data[j2++];
            }
            else
            {
               if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
               {
                  w_row_sum[i] += A_offd_data[j2];
               }
               j2++;
            }
         }
         while (j2 < A_offd_i[i1 + 1])
         {
            if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
            {
               w_row_sum[i] += A_offd_data[j2];
            }
            j2++;
         }
      }
   }
   else
   {
      for (i = 0; i < num_points; i++)
      {
         i1 = pass_order[i];
         j2 = A_diag_i[i1] + 1;
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            while (A_diag_j[j2] != j1)
            {
               w_row_sum[i] += A_diag_data[j2];
               j2++;
            }
            if (pass_marker[j1] == color && A_diag_j[j2] == j1)
            {
               Q_diag_j[cnt_diag] = fine_to_coarse[j1];
               Q_diag_data[cnt_diag++] = A_diag_data[j2++];
            }
            else
            {
               w_row_sum[i] += A_diag_data[j2];
               j2++;
            }
         }
         while (j2 < A_diag_i[i1 + 1])
         {
            w_row_sum[i] += A_diag_data[j2];
            j2++;
         }
         j2 = A_offd_i[i1];
         for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
         {
            j1 = S_offd_j[j];
            while (A_offd_j[j2] != j1)
            {
               w_row_sum[i] += A_offd_data[j2];
               j2++;
            }
            if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
            {
               Q_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
               Q_offd_data[cnt_offd++] = A_offd_data[j2++];
            }
            else
            {
               w_row_sum[i] += A_offd_data[j2];
               j2++;
            }
         }
         while (j2 < A_offd_i[i1 + 1])
         {
            w_row_sum[i] += A_offd_data[j2];
            j2++;
         }
      }
   }

   Q = hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_Q,
                                Q_diag_i[num_points],
                                Q_offd_i[num_points]);

   Q_diag = hypre_ParCSRMatrixDiag(Q);
   hypre_CSRMatrixData(Q_diag) = Q_diag_data;
   hypre_CSRMatrixI(Q_diag) = Q_diag_i;
   hypre_CSRMatrixJ(Q_diag) = Q_diag_j;
   Q_offd = hypre_ParCSRMatrixOffd(Q);
   hypre_CSRMatrixData(Q_offd) = Q_offd_data;
   hypre_CSRMatrixI(Q_offd) = Q_offd_i;
   hypre_CSRMatrixJ(Q_offd) = Q_offd_j;
   hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;

   hypre_CSRMatrixMemoryLocation(Q_diag) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixMemoryLocation(Q_offd) = HYPRE_MEMORY_HOST;

   /* free stuff */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(big_convert, HYPRE_MEMORY_HOST);
   hypre_TFree(big_convert_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);

   hypre_MatvecCommPkgCreate(Q);

   Pi = hypre_ParMatmul(Q, P);

   Pi_diag = hypre_ParCSRMatrixDiag(Pi);
   Pi_diag_data = hypre_CSRMatrixData(Pi_diag);
   Pi_diag_i = hypre_CSRMatrixI(Pi_diag);
   Pi_offd = hypre_ParCSRMatrixOffd(Pi);
   Pi_offd_data = hypre_CSRMatrixData(Pi_offd);
   Pi_offd_i = hypre_CSRMatrixI(Pi_offd);

   row_sums_C = hypre_CTAlloc(HYPRE_Real, num_points, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_points; i++)
   {
      HYPRE_Real diagonal, value;
      i1 = pass_order[i];
      diagonal = A_diag_data[A_diag_i[i1]];
      for (j = Pi_diag_i[i]; j < Pi_diag_i[i + 1]; j++)
      {
         row_sums_C[i] += Pi_diag_data[j];
      }
      for (j = Pi_offd_i[i]; j < Pi_offd_i[i + 1]; j++)
      {
         row_sums_C[i] += Pi_offd_data[j];
      }
      value = row_sums_C[i] * diagonal;
      row_sums_C[i] += w_row_sum[i];
      if (value != 0)
      {
         row_sums_C[i] /= value;
      }
      for (j = Pi_diag_i[i]; j < Pi_diag_i[i + 1]; j++)
      {
         Pi_diag_data[j] = -Pi_diag_data[j] * row_sums_C[i];
      }
      for (j = Pi_offd_i[i]; j < Pi_offd_i[i + 1]; j++)
      {
         Pi_offd_data[j] = -Pi_offd_data[j] * row_sums_C[i];
      }
   }

   hypre_ParCSRMatrixDestroy(Q);
   //hypre_TFree(row_sums, HYPRE_MEMORY_HOST);
   hypre_TFree(row_sums_C, HYPRE_MEMORY_HOST);
   hypre_TFree(w_row_sum, HYPRE_MEMORY_HOST);

   *Pi_ptr = Pi;

   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGBuildModMultipass( hypre_ParCSRMatrix  *A,
                                  HYPRE_Int           *CF_marker,
                                  hypre_ParCSRMatrix  *S,
                                  HYPRE_BigInt        *num_cpts_global,
                                  HYPRE_Real           trunc_factor,
                                  HYPRE_Int            P_max_elmts,
                                  HYPRE_Int            interp_type,
                                  HYPRE_Int            num_functions,
                                  HYPRE_Int           *dof_func,
                                  hypre_ParCSRMatrix **P_ptr )
{
   hypre_GpuProfilingPushRange("ModMultipass");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParCSRMatrixMemoryLocation(S) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildModMultipassDevice( A, CF_marker, S, num_cpts_global,
                                                     trunc_factor, P_max_elmts,
                                                     interp_type, num_functions,
                                                     dof_func, P_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildModMultipassHost( A, CF_marker, S, num_cpts_global,
                                                   trunc_factor, P_max_elmts,
                                                   interp_type, num_functions,
                                                   dof_func, P_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}

