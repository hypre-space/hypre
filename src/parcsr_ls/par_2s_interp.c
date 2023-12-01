/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildModExtInterp
 *  Comment:
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildModPartialExtInterpHost( hypre_ParCSRMatrix  *A,
                                             HYPRE_Int           *CF_marker,
                                             hypre_ParCSRMatrix  *S,
                                             HYPRE_BigInt        *num_cpts_global,
                                             HYPRE_BigInt        *num_old_cpts_global,
                                             HYPRE_Int            num_functions,
                                             HYPRE_Int           *dof_func,
                                             HYPRE_Int            debug_flag,
                                             HYPRE_Real           trunc_factor,
                                             HYPRE_Int            max_elmts,
                                             hypre_ParCSRMatrix **P_ptr )
{
   HYPRE_UNUSED_VAR(num_functions);
   HYPRE_UNUSED_VAR(dof_func);
   HYPRE_UNUSED_VAR(debug_flag);

   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRCommHandle  *comm_handle = NULL;
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   HYPRE_Int              my_id, num_procs;

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   //HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   /*HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);*/

   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt     total_old_global_cpts;

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /* Intermediate matrices */
   hypre_ParCSRMatrix *As_FF, *As_FC, *W;
   HYPRE_Real *D_q, *D_w;
   HYPRE_Real *D_q_offd = NULL;
   hypre_CSRMatrix *As_FF_diag;
   hypre_CSRMatrix *As_FF_offd;
   hypre_CSRMatrix *As_FC_diag;
   hypre_CSRMatrix *As_FC_offd;
   hypre_CSRMatrix *W_diag;
   hypre_CSRMatrix *W_offd;

   HYPRE_Int *As_FF_diag_i;
   HYPRE_Int *As_FF_diag_j;
   HYPRE_Int *As_FF_offd_i;
   HYPRE_Int *As_FF_offd_j;
   HYPRE_Int *As_FC_diag_i;
   HYPRE_Int *As_FC_offd_i;
   HYPRE_Int *W_diag_i;
   HYPRE_Int *W_offd_i;
   HYPRE_Int *W_diag_j;
   HYPRE_Int *W_offd_j;

   HYPRE_Real *As_FF_diag_data;
   HYPRE_Real *As_FF_offd_data;
   HYPRE_Real *As_FC_diag_data;
   HYPRE_Real *As_FC_offd_data;
   HYPRE_Real *W_diag_data;
   HYPRE_Real *W_offd_data;
   HYPRE_Real *buf_data = NULL;

   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_BigInt    *new_col_map_offd = NULL;
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int        num_cols_A_FF_offd;
   HYPRE_Int        new_ncols_P_offd;
   HYPRE_Int        num_cols_P_offd;
   HYPRE_Int       *P_marker = NULL;
   //HYPRE_Int       *dof_func_offd = NULL;

   /* Loop variables */
   HYPRE_Int        index;
   HYPRE_Int        i, j;
   HYPRE_Int       *cpt_array;
   HYPRE_Int       *new_fpt_array;
   HYPRE_Int       *start_array;
   HYPRE_Int       *new_fine_to_fine;
   HYPRE_Int start, stop, startf, stopf, startnewf, stopnewf;
   HYPRE_Int cnt_diag, cnt_offd, row, c_pt, fpt;
   HYPRE_Int startc, num_sends;

   /* Definitions */
   //HYPRE_Real       wall_time;
   HYPRE_Int n_Cpts, n_Fpts, n_old_Cpts, n_new_Fpts;
   HYPRE_Int num_threads = hypre_NumThreads();

   //if (debug_flag==4) wall_time = time_getWallclockSeconds();

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   if (my_id == (num_procs - 1)) { total_old_global_cpts = num_old_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   hypre_MPI_Bcast(&total_old_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   n_Cpts = num_cpts_global[1] - num_cpts_global[0];
   n_old_Cpts = num_old_cpts_global[1] - num_old_cpts_global[0];

   hypre_ParCSRMatrixGenerateFFFC3(A, CF_marker, num_cpts_global, S, &As_FC, &As_FF);

   As_FC_diag = hypre_ParCSRMatrixDiag(As_FC);
   As_FC_diag_i = hypre_CSRMatrixI(As_FC_diag);
   As_FC_diag_data = hypre_CSRMatrixData(As_FC_diag);
   As_FC_offd = hypre_ParCSRMatrixOffd(As_FC);
   As_FC_offd_i = hypre_CSRMatrixI(As_FC_offd);
   As_FC_offd_data = hypre_CSRMatrixData(As_FC_offd);
   As_FF_diag = hypre_ParCSRMatrixDiag(As_FF);
   As_FF_diag_i = hypre_CSRMatrixI(As_FF_diag);
   As_FF_diag_j = hypre_CSRMatrixJ(As_FF_diag);
   As_FF_diag_data = hypre_CSRMatrixData(As_FF_diag);
   As_FF_offd = hypre_ParCSRMatrixOffd(As_FF);
   As_FF_offd_i = hypre_CSRMatrixI(As_FF_offd);
   As_FF_offd_j = hypre_CSRMatrixJ(As_FF_offd);
   As_FF_offd_data = hypre_CSRMatrixData(As_FF_offd);
   n_new_Fpts = hypre_CSRMatrixNumRows(As_FF_diag);
   n_Fpts = hypre_CSRMatrixNumRows(As_FC_diag);
   n_new_Fpts = n_old_Cpts - n_Cpts;
   num_cols_A_FF_offd = hypre_CSRMatrixNumCols(As_FF_offd);

   D_q = hypre_CTAlloc(HYPRE_Real, n_Fpts, memory_location_P);
   new_fine_to_fine = hypre_CTAlloc(HYPRE_Int, n_new_Fpts, HYPRE_MEMORY_HOST);
   D_w = hypre_CTAlloc(HYPRE_Real, n_new_Fpts, memory_location_P);
   cpt_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   new_fpt_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   start_array = hypre_CTAlloc(HYPRE_Int, num_threads + 1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,start,stop,startf,stopf,startnewf,stopnewf,row,fpt)
#endif
   {
      HYPRE_Int my_thread_num = hypre_GetThreadNum();
      HYPRE_Real beta, gamma;

      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }
      start_array[my_thread_num + 1] = stop;
      row = 0;
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            cpt_array[my_thread_num]++;
         }
         else if (CF_marker[i] == -2)
         {
            new_fpt_array[my_thread_num]++;
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i = 1; i < num_threads; i++)
         {
            cpt_array[i] += cpt_array[i - 1];
            new_fpt_array[i] += new_fpt_array[i - 1];
         }
         /*if (num_functions > 1)
         {
            HYPRE_Int *int_buf_data = NULL;
            HYPRE_Int num_sends, startc;
            HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
            dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, memory_location_P);
            index = 0;
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends), memory_location_P);
            for (i = 0; i < num_sends; i++)
            {
               startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               {
                  int_buf_data[index++] = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }
            comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);
            hypre_ParCSRCommHandleDestroy(comm_handle);
            hypre_TFree(int_buf_data, memory_location_P);
         }*/
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num > 0)
      {
         startf = start - cpt_array[my_thread_num - 1];
      }
      else
      {
         startf = 0;
      }

      if (my_thread_num < num_threads - 1)
      {
         stopf = stop - cpt_array[my_thread_num];
      }
      else
      {
         stopf = n_Fpts;
      }

      /* Create D_q = D_beta */
      for (i = startf; i < stopf; i++)
      {
         for (j = As_FC_diag_i[i]; j < As_FC_diag_i[i + 1]; j++)
         {
            D_q[i] += As_FC_diag_data[j];
         }
         for (j = As_FC_offd_i[i]; j < As_FC_offd_i[i + 1]; j++)
         {
            D_q[i] += As_FC_offd_data[j];
         }
      }

      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      fpt = startf;
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            new_fine_to_fine[row++] = fpt++;
         }
         else if (CF_marker[i] < 0)
         {
            fpt++;
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         if (num_cols_A_FF_offd)
         {
            D_q_offd = hypre_CTAlloc(HYPRE_Real, num_cols_A_FF_offd, memory_location_P);
         }
         index = 0;
         comm_pkg = hypre_ParCSRMatrixCommPkg(As_FF);
         if (!comm_pkg)
         {
            hypre_MatvecCommPkgCreate(As_FF);
            comm_pkg = hypre_ParCSRMatrixCommPkg(As_FF);
         }
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  memory_location_P);
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               buf_data[index++] = D_q[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, buf_data, D_q_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);

      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      /* Create D_w = D_alpha + D_gamma */
      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            /*if (num_functions > 1)
            {
               HYPRE_Int jA, jC, jS;
               jC = A_diag_i[i];
               for (j=S_diag_i[i]; j < S_diag_i[i+1]; j++)
               {
                  jS = S_diag_j[j];
                  jA = A_diag_j[jC];
                  while (jA != jS)
                  {
                     if (dof_func[i] == dof_func[jA])
                     {
                        D_w[row] += A_diag_data[jC++];
                     }
                     else
                        jC++;
                     jA = A_diag_j[jC];
                  }
                  jC++;
               }
               for (j=jC; j < A_diag_i[i+1]; j++)
               {
                  if (dof_func[i] == dof_func[A_diag_j[j]])
                        D_w[row] += A_diag_data[j];
               }
               jC = A_offd_i[i];
               for (j=S_offd_i[i]; j < S_offd_i[i+1]; j++)
               {
                  jS = S_offd_j[j];
                  jA = A_offd_j[jC];
                  while (jA != jS)
                  {
                     if (dof_func[i] == dof_func_offd[jA])
                     {
                        D_w[row] += A_offd_data[jC++];
                     }
                     else
                        jC++;
                     jA = A_offd_j[jC];
                  }
                  jC++;
               }
               for (j=jC; j < A_offd_i[i+1]; j++)
               {
                  if (dof_func[i] == dof_func_offd[A_offd_j[j]])
                        D_w[row] += A_offd_data[j];
               }
               row++;
            }
            else*/
            {
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  D_w[row] += A_diag_data[j];
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  D_w[row] += A_offd_data[j];
               }
               for (j = As_FF_diag_i[row] + 1; j < As_FF_diag_i[row + 1]; j++)
               {
                  if (D_q[As_FF_diag_j[j]]) { D_w[row] -= As_FF_diag_data[j]; }
               }
               for (j = As_FF_offd_i[row]; j < As_FF_offd_i[row + 1]; j++)
               {
                  if (D_q_offd[As_FF_offd_j[j]]) { D_w[row] -= As_FF_offd_data[j]; }
               }
               D_w[row] -= D_q[new_fine_to_fine[row]];
               row++;
            }
         }
      }


#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      startnewf = 0;
      if (my_thread_num) { startnewf = new_fpt_array[my_thread_num - 1]; }
      stopnewf = new_fpt_array[my_thread_num];
      for (i = startnewf; i < stopnewf; i++)
      {
         j = As_FF_diag_i[i];
         if (D_w[i])
         {
            beta = 1.0 / D_w[i];
            As_FF_diag_data[j] = beta * D_q[new_fine_to_fine[i]];
            for (j = As_FF_diag_i[i] + 1; j < As_FF_diag_i[i + 1]; j++)
            {
               As_FF_diag_data[j] *= beta;
            }
            for (j = As_FF_offd_i[i]; j < As_FF_offd_i[i + 1]; j++)
            {
               As_FF_offd_data[j] *= beta;
            }
         }
      }
      for (i = startf; i < stopf; i++)
      {
         if (D_q[i]) { gamma = -1.0 / D_q[i]; }
         else { gamma = 0.0; }
         for (j = As_FC_diag_i[i]; j < As_FC_diag_i[i + 1]; j++)
         {
            As_FC_diag_data[j] *= gamma;
         }
         for (j = As_FC_offd_i[i]; j < As_FC_offd_i[i + 1]; j++)
         {
            As_FC_offd_data[j] *= gamma;
         }
      }

   }   /* end parallel region */

   W = hypre_ParMatmul(As_FF, As_FC);
   W_diag = hypre_ParCSRMatrixDiag(W);
   W_offd = hypre_ParCSRMatrixOffd(W);
   W_diag_i = hypre_CSRMatrixI(W_diag);
   W_diag_j = hypre_CSRMatrixJ(W_diag);
   W_diag_data = hypre_CSRMatrixData(W_diag);
   W_offd_i = hypre_CSRMatrixI(W_offd);
   W_offd_j = hypre_CSRMatrixJ(W_offd);
   W_offd_data = hypre_CSRMatrixData(W_offd);
   num_cols_P_offd = hypre_CSRMatrixNumCols(W_offd);
   /*-----------------------------------------------------------------------
    *  Intialize data for P
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_old_Cpts + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_old_Cpts + 1, memory_location_P);

   P_diag_size = n_Cpts + hypre_CSRMatrixI(W_diag)[n_new_Fpts];
   P_offd_size = hypre_CSRMatrixI(W_offd)[n_new_Fpts];

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, memory_location_P);
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,start,stop,startnewf,stopnewf,c_pt,row,cnt_diag,cnt_offd)
#endif
   {
      HYPRE_Int rowp;
      HYPRE_Int my_thread_num = hypre_GetThreadNum();
      start = start_array[my_thread_num];
      stop = start_array[my_thread_num + 1];

      if (my_thread_num > 0)
      {
         c_pt = cpt_array[my_thread_num - 1];
      }
      else
      {
         c_pt = 0;
      }
      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      rowp = row;
      if (my_thread_num > 0) { rowp = row + cpt_array[my_thread_num - 1]; }
      cnt_diag = W_diag_i[row] + c_pt;
      cnt_offd = W_offd_i[row];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            rowp++;
            P_diag_j[cnt_diag] = c_pt++;
            P_diag_data[cnt_diag++] = 1.0;
            P_diag_i[rowp] = cnt_diag;
            P_offd_i[rowp] = cnt_offd;
         }
         else if (CF_marker[i] == -2)
         {
            rowp++;
            for (j = W_diag_i[row]; j < W_diag_i[row + 1]; j++)
            {
               P_diag_j[cnt_diag] = W_diag_j[j];
               P_diag_data[cnt_diag++] = W_diag_data[j];
            }
            for (j = W_offd_i[row]; j < W_offd_i[row + 1]; j++)
            {
               P_offd_j[cnt_offd] = W_offd_j[j];
               P_offd_data[cnt_offd++] = W_offd_data[j];
            }
            row++;
            P_diag_i[rowp] = cnt_diag;
            P_offd_i[rowp] = cnt_offd;
         }
      }
   }   /* end parallel region */

   /*-----------------------------------------------------------------------
    *  Create matrix
    *-----------------------------------------------------------------------*/

   P = hypre_ParCSRMatrixCreate(comm,
                                total_old_global_cpts,
                                total_global_cpts,
                                num_old_cpts_global,
                                num_cpts_global,
                                num_cols_P_offd,
                                P_diag_i[n_old_Cpts],
                                P_offd_i[n_old_Cpts]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixColMapOffd(P) = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(W) = NULL;

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      HYPRE_Int *map;
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_old_Cpts];
      P_offd_size = P_offd_i[n_old_Cpts];

      col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);
      if (num_cols_P_offd)
      {
         P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
         for (i = 0; i < P_offd_size; i++)
         {
            P_marker[P_offd_j[i]] = 1;
         }

         new_ncols_P_offd = 0;
         for (i = 0; i < num_cols_P_offd; i++)
            if (P_marker[i]) { new_ncols_P_offd++; }

         new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, new_ncols_P_offd, HYPRE_MEMORY_HOST);
         map = hypre_CTAlloc(HYPRE_Int, new_ncols_P_offd, HYPRE_MEMORY_HOST);

         index = 0;
         for (i = 0; i < num_cols_P_offd; i++)
            if (P_marker[i])
            {
               new_col_map_offd[index] = col_map_offd_P[i];
               map[index++] = i;
            }
         hypre_TFree(P_marker, HYPRE_MEMORY_HOST);


#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < P_offd_size; i++)
         {
            P_offd_j[i] = hypre_BinarySearch(map, P_offd_j[i],
                                             new_ncols_P_offd);
         }
         hypre_TFree(col_map_offd_P, HYPRE_MEMORY_HOST);
         hypre_ParCSRMatrixColMapOffd(P) = new_col_map_offd;
         hypre_CSRMatrixNumCols(P_offd) = new_ncols_P_offd;
         hypre_TFree(map, HYPRE_MEMORY_HOST);
      }
   }

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(D_q, memory_location_P);
   hypre_TFree(D_q_offd, memory_location_P);
   hypre_TFree(D_w, memory_location_P);
   //hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(cpt_array, HYPRE_MEMORY_HOST);
   hypre_TFree(new_fpt_array, HYPRE_MEMORY_HOST);
   hypre_TFree(start_array, HYPRE_MEMORY_HOST);
   hypre_TFree(new_fine_to_fine, HYPRE_MEMORY_HOST);
   hypre_TFree(buf_data, memory_location_P);
   hypre_ParCSRMatrixDestroy(As_FF);
   hypre_ParCSRMatrixDestroy(As_FC);
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGBuildModPartialExtInterp( hypre_ParCSRMatrix  *A,
                                         HYPRE_Int           *CF_marker,
                                         hypre_ParCSRMatrix  *S,
                                         HYPRE_BigInt        *num_cpts_global,
                                         HYPRE_BigInt        *num_old_cpts_global,
                                         HYPRE_Int            num_functions,
                                         HYPRE_Int           *dof_func,
                                         HYPRE_Int            debug_flag,
                                         HYPRE_Real           trunc_factor,
                                         HYPRE_Int            max_elmts,
                                         hypre_ParCSRMatrix **P_ptr )
{
   hypre_GpuProfilingPushRange("PartialExtInterp");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildModPartialExtInterpDevice(A, CF_marker, S, num_cpts_global,
                                                           num_old_cpts_global,
                                                           debug_flag, trunc_factor, max_elmts, P_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildModPartialExtInterpHost(A, CF_marker, S, num_cpts_global,
                                                         num_old_cpts_global,
                                                         num_functions, dof_func,
                                                         debug_flag, trunc_factor, max_elmts, P_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}

HYPRE_Int
hypre_BoomerAMGBuildModPartialExtPEInterpHost( hypre_ParCSRMatrix  *A,
                                               HYPRE_Int           *CF_marker,
                                               hypre_ParCSRMatrix  *S,
                                               HYPRE_BigInt        *num_cpts_global,
                                               HYPRE_BigInt        *num_old_cpts_global,
                                               HYPRE_Int            num_functions,
                                               HYPRE_Int           *dof_func,
                                               HYPRE_Int            debug_flag,
                                               HYPRE_Real           trunc_factor,
                                               HYPRE_Int            max_elmts,
                                               hypre_ParCSRMatrix **P_ptr)
{
   HYPRE_UNUSED_VAR(debug_flag);

   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRCommHandle  *comm_handle = NULL;
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   HYPRE_Int              my_id, num_procs;

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);

   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt     total_old_global_cpts;

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /* Intermediate matrices */
   hypre_ParCSRMatrix *As_FF, *As_FC, *W;
   HYPRE_Real *D_q, *D_w, *D_lambda, *D_inv, *D_tau;
   HYPRE_Real *D_lambda_offd = NULL, *D_inv_offd = NULL;
   hypre_CSRMatrix *As_FF_diag;
   hypre_CSRMatrix *As_FF_offd;
   hypre_CSRMatrix *As_FC_diag;
   hypre_CSRMatrix *As_FC_offd;
   hypre_CSRMatrix *W_diag;
   hypre_CSRMatrix *W_offd;

   HYPRE_Int *As_FF_diag_i;
   HYPRE_Int *As_FF_diag_j;
   HYPRE_Int *As_FF_offd_i;
   HYPRE_Int *As_FF_offd_j;
   HYPRE_Int *As_FC_diag_i;
   HYPRE_Int *As_FC_offd_i;
   HYPRE_Int *W_diag_i;
   HYPRE_Int *W_offd_i;
   HYPRE_Int *W_diag_j;
   HYPRE_Int *W_offd_j;

   HYPRE_Real *As_FF_diag_data;
   HYPRE_Real *As_FF_offd_data;
   HYPRE_Real *As_FC_diag_data;
   HYPRE_Real *As_FC_offd_data;
   HYPRE_Real *W_diag_data;
   HYPRE_Real *W_offd_data;
   HYPRE_Real *buf_data = NULL;

   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_BigInt    *new_col_map_offd = NULL;
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int        num_cols_A_FF_offd;
   HYPRE_Int        new_ncols_P_offd;
   HYPRE_Int        num_cols_P_offd;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *dof_func_offd = NULL;

   /* Loop variables */
   HYPRE_Int        index;
   HYPRE_Int        i, j;
   HYPRE_Int       *cpt_array;
   HYPRE_Int       *new_fpt_array;
   HYPRE_Int       *start_array;
   HYPRE_Int       *new_fine_to_fine;
   HYPRE_Int start, stop, startf, stopf, startnewf, stopnewf;
   HYPRE_Int cnt_diag, cnt_offd, row, c_pt, fpt;
   HYPRE_Int startc, num_sends;

   /* Definitions */
   //HYPRE_Real       wall_time;
   HYPRE_Int n_Cpts, n_Fpts, n_old_Cpts, n_new_Fpts;
   HYPRE_Int num_threads = hypre_NumThreads();

   //if (debug_flag==4) wall_time = time_getWallclockSeconds();

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   if (my_id == (num_procs - 1)) { total_old_global_cpts = num_old_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   hypre_MPI_Bcast(&total_old_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   n_Cpts = num_cpts_global[1] - num_cpts_global[0];
   n_old_Cpts = num_old_cpts_global[1] - num_old_cpts_global[0];

   hypre_ParCSRMatrixGenerateFFFCD3(A, CF_marker, num_cpts_global, S, &As_FC, &As_FF, &D_lambda);

   As_FC_diag = hypre_ParCSRMatrixDiag(As_FC);
   As_FC_diag_i = hypre_CSRMatrixI(As_FC_diag);
   As_FC_diag_data = hypre_CSRMatrixData(As_FC_diag);
   As_FC_offd = hypre_ParCSRMatrixOffd(As_FC);
   As_FC_offd_i = hypre_CSRMatrixI(As_FC_offd);
   As_FC_offd_data = hypre_CSRMatrixData(As_FC_offd);
   As_FF_diag = hypre_ParCSRMatrixDiag(As_FF);
   As_FF_diag_i = hypre_CSRMatrixI(As_FF_diag);
   As_FF_diag_j = hypre_CSRMatrixJ(As_FF_diag);
   As_FF_diag_data = hypre_CSRMatrixData(As_FF_diag);
   As_FF_offd = hypre_ParCSRMatrixOffd(As_FF);
   As_FF_offd_i = hypre_CSRMatrixI(As_FF_offd);
   As_FF_offd_j = hypre_CSRMatrixJ(As_FF_offd);
   As_FF_offd_data = hypre_CSRMatrixData(As_FF_offd);
   n_new_Fpts = hypre_CSRMatrixNumRows(As_FF_diag);
   n_Fpts = hypre_CSRMatrixNumRows(As_FC_diag);
   n_new_Fpts = n_old_Cpts - n_Cpts;
   num_cols_A_FF_offd = hypre_CSRMatrixNumCols(As_FF_offd);

   D_q = hypre_CTAlloc(HYPRE_Real, n_Fpts, memory_location_P);
   D_inv = hypre_CTAlloc(HYPRE_Real, n_Fpts, memory_location_P);
   new_fine_to_fine = hypre_CTAlloc(HYPRE_Int, n_new_Fpts, HYPRE_MEMORY_HOST);
   D_w = hypre_CTAlloc(HYPRE_Real, n_new_Fpts, memory_location_P);
   D_tau = hypre_CTAlloc(HYPRE_Real, n_new_Fpts, memory_location_P);
   cpt_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   new_fpt_array = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   start_array = hypre_CTAlloc(HYPRE_Int, num_threads + 1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,start,stop,startf,stopf,startnewf,stopnewf,row,fpt,index)
#endif
   {
      HYPRE_Int my_thread_num = hypre_GetThreadNum();
      HYPRE_Real beta, gamma;

      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }
      start_array[my_thread_num + 1] = stop;
      row = 0;
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            cpt_array[my_thread_num]++;
         }
         else if (CF_marker[i] == -2)
         {
            new_fpt_array[my_thread_num]++;
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i = 1; i < num_threads; i++)
         {
            cpt_array[i] += cpt_array[i - 1];
            new_fpt_array[i] += new_fpt_array[i - 1];
         }
         if (num_functions > 1)
         {
            HYPRE_Int *int_buf_data = NULL;
            HYPRE_Int num_sends, startc;
            HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
            dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, memory_location_P);
            index = 0;
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                         memory_location_P);
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
            hypre_TFree(int_buf_data, memory_location_P);
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num > 0)
      {
         startf = start - cpt_array[my_thread_num - 1];
      }
      else
      {
         startf = 0;
      }

      if (my_thread_num < num_threads - 1)
      {
         stopf = stop - cpt_array[my_thread_num];
      }
      else
      {
         stopf = n_Fpts;
      }

      /* Create D_q = D_beta, D_inv = 1/(D_q+D_lambda) */
      for (i = startf; i < stopf; i++)
      {
         for (j = As_FC_diag_i[i]; j < As_FC_diag_i[i + 1]; j++)
         {
            D_q[i] += As_FC_diag_data[j];
         }
         for (j = As_FC_offd_i[i]; j < As_FC_offd_i[i + 1]; j++)
         {
            D_q[i] += As_FC_offd_data[j];
         }
         if (D_q[i] + D_lambda[i]) { D_inv[i] = 1.0 / (D_q[i] + D_lambda[i]); }
      }

      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      fpt = startf;
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            new_fine_to_fine[row++] = fpt++;
         }
         else if (CF_marker[i] < 0)
         {
            fpt++;
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         if (num_cols_A_FF_offd)
         {
            D_lambda_offd = hypre_CTAlloc(HYPRE_Real, num_cols_A_FF_offd, memory_location_P);
            D_inv_offd = hypre_CTAlloc(HYPRE_Real, num_cols_A_FF_offd, memory_location_P);
         }
         index = 0;
         comm_pkg = hypre_ParCSRMatrixCommPkg(As_FF);
         if (!comm_pkg)
         {
            hypre_MatvecCommPkgCreate(As_FF);
            comm_pkg = hypre_ParCSRMatrixCommPkg(As_FF);
         }
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         buf_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  memory_location_P);
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               buf_data[index++] = D_lambda[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, buf_data, D_lambda_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            startc = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               buf_data[index++] = D_inv[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, buf_data, D_inv_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);

      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      /* Create D_tau */
      startnewf = 0;
      if (my_thread_num) { startnewf = new_fpt_array[my_thread_num - 1]; }
      stopnewf = new_fpt_array[my_thread_num];
      for (i = startnewf; i < stopnewf; i++)
      {
         for (j = As_FF_diag_i[i] + 1; j < As_FF_diag_i[i + 1]; j++)
         {
            index = As_FF_diag_j[j];
            D_tau[i] += As_FF_diag_data[j] * D_lambda[index] * D_inv[index];
         }
         for (j = As_FF_offd_i[i]; j < As_FF_offd_i[i + 1]; j++)
         {
            index = As_FF_offd_j[j];
            D_tau[i] += As_FF_offd_data[j] * D_lambda_offd[index] * D_inv_offd[index];
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      /* Create D_w = D_alpha + D_gamma + D_tau */
      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            if (num_functions > 1)
            {
               HYPRE_Int jA, jC, jS;
               jC = A_diag_i[i];
               for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
               {
                  jS = S_diag_j[j];
                  jA = A_diag_j[jC];
                  while (jA != jS)
                  {
                     if (dof_func[i] == dof_func[jA])
                     {
                        D_w[row] += A_diag_data[jC++];
                     }
                     else
                     {
                        jC++;
                     }
                     jA = A_diag_j[jC];
                  }
                  jC++;
               }
               for (j = jC; j < A_diag_i[i + 1]; j++)
               {
                  if (dof_func[i] == dof_func[A_diag_j[j]])
                  {
                     D_w[row] += A_diag_data[j];
                  }
               }
               jC = A_offd_i[i];
               for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
               {
                  jS = S_offd_j[j];
                  jA = A_offd_j[jC];
                  while (jA != jS)
                  {
                     if (dof_func[i] == dof_func_offd[jA])
                     {
                        D_w[row] += A_offd_data[jC++];
                     }
                     else
                     {
                        jC++;
                     }
                     jA = A_offd_j[jC];
                  }
                  jC++;
               }
               for (j = jC; j < A_offd_i[i + 1]; j++)
               {
                  if (dof_func[i] == dof_func_offd[A_offd_j[j]])
                  {
                     D_w[row] += A_offd_data[j];
                  }
               }
               D_w[row] += D_tau[row];
               row++;
            }
            else
            {
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  D_w[row] += A_diag_data[j];
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  D_w[row] += A_offd_data[j];
               }
               for (j = As_FF_diag_i[row] + 1; j < As_FF_diag_i[row + 1]; j++)
               {
                  if (D_inv[As_FF_diag_j[j]]) { D_w[row] -= As_FF_diag_data[j]; }
               }
               for (j = As_FF_offd_i[row]; j < As_FF_offd_i[row + 1]; j++)
               {
                  if (D_inv_offd[As_FF_offd_j[j]]) { D_w[row] -= As_FF_offd_data[j]; }
               }
               D_w[row] += D_tau[row] - D_q[new_fine_to_fine[row]];
               row++;
            }
         }
      }

#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      startnewf = 0;
      if (my_thread_num) { startnewf = new_fpt_array[my_thread_num - 1]; }
      stopnewf = new_fpt_array[my_thread_num];
      for (i = startnewf; i < stopnewf; i++)
      {
         j = As_FF_diag_i[i];
         if (D_w[i])
         {
            beta = -1.0 / D_w[i];
            As_FF_diag_data[j] = beta * (D_q[new_fine_to_fine[i]] + D_lambda[new_fine_to_fine[i]]);
            for (j = As_FF_diag_i[i] + 1; j < As_FF_diag_i[i + 1]; j++)
            {
               As_FF_diag_data[j] *= beta;
            }
            for (j = As_FF_offd_i[i]; j < As_FF_offd_i[i + 1]; j++)
            {
               As_FF_offd_data[j] *= beta;
            }
         }
      }
      for (i = startf; i < stopf; i++)
      {
         gamma = D_inv[i];
         for (j = As_FC_diag_i[i]; j < As_FC_diag_i[i + 1]; j++)
         {
            As_FC_diag_data[j] *= gamma;
         }
         for (j = As_FC_offd_i[i]; j < As_FC_offd_i[i + 1]; j++)
         {
            As_FC_offd_data[j] *= gamma;
         }
      }

   }   /* end parallel region */

   W = hypre_ParMatmul(As_FF, As_FC);
   W_diag = hypre_ParCSRMatrixDiag(W);
   W_offd = hypre_ParCSRMatrixOffd(W);
   W_diag_i = hypre_CSRMatrixI(W_diag);
   W_diag_j = hypre_CSRMatrixJ(W_diag);
   W_diag_data = hypre_CSRMatrixData(W_diag);
   W_offd_i = hypre_CSRMatrixI(W_offd);
   W_offd_j = hypre_CSRMatrixJ(W_offd);
   W_offd_data = hypre_CSRMatrixData(W_offd);
   num_cols_P_offd = hypre_CSRMatrixNumCols(W_offd);
   /*-----------------------------------------------------------------------
    *  Intialize data for P
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_old_Cpts + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_old_Cpts + 1, memory_location_P);

   P_diag_size = n_Cpts + hypre_CSRMatrixI(W_diag)[n_new_Fpts];
   P_offd_size = hypre_CSRMatrixI(W_offd)[n_new_Fpts];

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, memory_location_P);
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,start,stop,c_pt,row,cnt_diag,cnt_offd)
#endif
   {
      HYPRE_Int rowp;
      HYPRE_Int my_thread_num = hypre_GetThreadNum();
      start = start_array[my_thread_num];
      stop = start_array[my_thread_num + 1];

      if (my_thread_num > 0)
      {
         c_pt = cpt_array[my_thread_num - 1];
      }
      else
      {
         c_pt = 0;
      }
      row = 0;
      if (my_thread_num) { row = new_fpt_array[my_thread_num - 1]; }
      rowp = row;
      if (my_thread_num > 0) { rowp = row + cpt_array[my_thread_num - 1]; }
      cnt_diag = W_diag_i[row] + c_pt;
      cnt_offd = W_offd_i[row];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            rowp++;
            P_diag_j[cnt_diag] = c_pt++;
            P_diag_data[cnt_diag++] = 1.0;
            P_diag_i[rowp] = cnt_diag;
            P_offd_i[rowp] = cnt_offd;
         }
         else if (CF_marker[i] == -2)
         {
            rowp++;
            for (j = W_diag_i[row]; j < W_diag_i[row + 1]; j++)
            {
               P_diag_j[cnt_diag] = W_diag_j[j];
               P_diag_data[cnt_diag++] = W_diag_data[j];
            }
            for (j = W_offd_i[row]; j < W_offd_i[row + 1]; j++)
            {
               P_offd_j[cnt_offd] = W_offd_j[j];
               P_offd_data[cnt_offd++] = W_offd_data[j];
            }
            row++;
            P_diag_i[rowp] = cnt_diag;
            P_offd_i[rowp] = cnt_offd;
         }
      }
   }   /* end parallel region */

   /*-----------------------------------------------------------------------
    *  Create matrix
    *-----------------------------------------------------------------------*/

   P = hypre_ParCSRMatrixCreate(comm,
                                total_old_global_cpts,
                                total_global_cpts,
                                num_old_cpts_global,
                                num_cpts_global,
                                num_cols_P_offd,
                                P_diag_i[n_old_Cpts],
                                P_offd_i[n_old_Cpts]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixColMapOffd(P) = hypre_ParCSRMatrixColMapOffd(W);
   hypre_ParCSRMatrixColMapOffd(W) = NULL;

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      HYPRE_Int *map;
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_old_Cpts];
      P_offd_size = P_offd_i[n_old_Cpts];

      col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);
      if (num_cols_P_offd)
      {
         P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
         for (i = 0; i < P_offd_size; i++)
         {
            P_marker[P_offd_j[i]] = 1;
         }

         new_ncols_P_offd = 0;
         for (i = 0; i < num_cols_P_offd; i++)
         {
            if (P_marker[i])
            {
               new_ncols_P_offd++;
            }
         }

         new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, new_ncols_P_offd, HYPRE_MEMORY_HOST);
         map = hypre_CTAlloc(HYPRE_Int, new_ncols_P_offd, HYPRE_MEMORY_HOST);

         index = 0;
         for (i = 0; i < num_cols_P_offd; i++)
         {
            if (P_marker[i])
            {
               new_col_map_offd[index] = col_map_offd_P[i];
               map[index++] = i;
            }
         }

         hypre_TFree(P_marker, HYPRE_MEMORY_HOST);


#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < P_offd_size; i++)
         {
            P_offd_j[i] = hypre_BinarySearch(map, P_offd_j[i], new_ncols_P_offd);
         }
         hypre_TFree(col_map_offd_P, HYPRE_MEMORY_HOST);
         hypre_ParCSRMatrixColMapOffd(P) = new_col_map_offd;
         hypre_CSRMatrixNumCols(P_offd) = new_ncols_P_offd;
         hypre_TFree(map, HYPRE_MEMORY_HOST);
      }
   }

   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(D_q, memory_location_P);
   hypre_TFree(D_inv, memory_location_P);
   hypre_TFree(D_inv_offd, memory_location_P);
   hypre_TFree(D_lambda, memory_location_P);
   hypre_TFree(D_lambda_offd, memory_location_P);
   hypre_TFree(D_tau, memory_location_P);
   hypre_TFree(D_w, memory_location_P);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(cpt_array, HYPRE_MEMORY_HOST);
   hypre_TFree(new_fpt_array, HYPRE_MEMORY_HOST);
   hypre_TFree(start_array, HYPRE_MEMORY_HOST);
   hypre_TFree(new_fine_to_fine, HYPRE_MEMORY_HOST);
   hypre_TFree(buf_data, memory_location_P);
   hypre_ParCSRMatrixDestroy(As_FF);
   hypre_ParCSRMatrixDestroy(As_FC);
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGBuildModPartialExtPEInterp( hypre_ParCSRMatrix  *A,
                                           HYPRE_Int           *CF_marker,
                                           hypre_ParCSRMatrix  *S,
                                           HYPRE_BigInt        *num_cpts_global,
                                           HYPRE_BigInt        *num_old_cpts_global,
                                           HYPRE_Int            num_functions,
                                           HYPRE_Int           *dof_func,
                                           HYPRE_Int            debug_flag,
                                           HYPRE_Real           trunc_factor,
                                           HYPRE_Int            max_elmts,
                                           hypre_ParCSRMatrix **P_ptr )
{
   hypre_GpuProfilingPushRange("PartialExtPEInterp");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildModPartialExtPEInterpDevice(A, CF_marker, S, num_cpts_global,
                                                             num_old_cpts_global,
                                                             debug_flag, trunc_factor, max_elmts, P_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildModPartialExtPEInterpHost(A, CF_marker, S, num_cpts_global,
                                                           num_old_cpts_global,
                                                           num_functions, dof_func,
                                                           debug_flag, trunc_factor, max_elmts, P_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}
