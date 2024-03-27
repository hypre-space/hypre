/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildStdInterp
 *  Comment: The interpolatory weighting can be changed with the sep_weight
 *           variable. This can enable not separating negative and positive
 *           off diagonals in the weight formula.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildStdInterp(hypre_ParCSRMatrix  *A,
                              HYPRE_Int           *CF_marker,
                              hypre_ParCSRMatrix  *S,
                              HYPRE_BigInt        *num_cpts_global,
                              HYPRE_Int            num_functions,
                              HYPRE_Int           *dof_func,
                              HYPRE_Int            debug_flag,
                              HYPRE_Real           trunc_factor,
                              HYPRE_Int            max_elmts,
                              HYPRE_Int            sep_weight,
                              hypre_ParCSRMatrix **P_ptr)
{
   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /* HYPRE_Int            *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;
   //HYPRE_BigInt    *found;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;
   HYPRE_Int       *ihat = NULL;
   HYPRE_Int       *ihat_offd = NULL;
   HYPRE_Int       *ipnt = NULL;
   HYPRE_Int       *ipnt_offd = NULL;
   HYPRE_Int        strong_f_marker = -2;

   /* Interpolation weight variables */
   HYPRE_Real      *ahat = NULL;
   HYPRE_Real      *ahat_offd = NULL;
   HYPRE_Real       sum_pos, sum_pos_C, sum_neg, sum_neg_C, sum, sum_C;
   HYPRE_Real       diagonal, distribute;
   HYPRE_Real       alfa = 1.;
   HYPRE_Real       beta = 1.;

   /* Loop variables */
   // HYPRE_Int              index;
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, j1, jj, kk, k1;
   HYPRE_Int        cnt_c, cnt_f, cnt_c_offd, cnt_f_offd, indx;
   HYPRE_BigInt     big_k1;

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   HYPRE_Real       wall_time;
   HYPRE_Real       wall_1 = 0;
   HYPRE_Real       wall_2 = 0;
   HYPRE_Real       wall_3 = 0;


   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 0);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
      P_marker       = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      P_marker_offd       = hypre_CTAlloc(HYPRE_Int,    full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd  = hypre_CTAlloc(HYPRE_Int,    full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }
      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else if (CF_marker[i] != -3)
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];
            if (CF_marker[i1] >= 0)
            {
               /* i1 is a C point */
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
            else if (CF_marker[i1] != -3)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < P_diag_i[i])
                     {
                        P_marker[k1] = jj_counter;
                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < P_offd_i[i])
                        {
                           tmp_CF_marker_offd[k1] = 1;
                           P_marker_offd[k1] = jj_counter_offd;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                  * numbers and entries that could be in S_diag or S_offd or
                  * neither. */
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] >= 0)
                        {
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                           }
                        }
                     }
                     else
                     {
                        loc_col = (HYPRE_Int)(-big_k1 - 1);
                        if (CF_marker_offd[loc_col] >= 0)
                        {
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     determine structure    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/


   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
   }

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                 full_off_procNodes, my_first_cpt,
                                 fine_to_coarse_offd);
   }

   /* Initialize ahat, which is a modification to a, used in the standard
    * interpolation routine. */
   if (n_fine)
   {
      ahat = hypre_CTAlloc(HYPRE_Real, n_fine, HYPRE_MEMORY_HOST);
      ihat = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      ipnt = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   }
   if (full_off_procNodes)
   {
      ahat_offd = hypre_CTAlloc(HYPRE_Real, full_off_procNodes, HYPRE_MEMORY_HOST);
      ihat_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      ipnt_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
      ahat[i] = 0;
      ihat[i] = -1;
   }
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
      ahat_offd[i] = 0;
      ihat_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }
         strong_f_marker--;
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = i1;
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }
            }
            else  if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < jj_begin_row)
                     {
                        P_marker[k1] = jj_counter;
                        P_diag_j[jj_counter] = k1;
                        P_diag_data[jj_counter] = zero;
                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < jj_begin_row_offd)
                        {
                           P_marker_offd[k1] = jj_counter_offd;
                           P_offd_j[jj_counter_offd] = k1;
                           P_offd_data[jj_counter_offd] = zero;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }

         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] >= 0)
                        {
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = loc_col;
                              P_diag_data[jj_counter] = zero;
                              jj_counter++;
                           }
                        }
                     }
                     else
                     {
                        loc_col = (HYPRE_Int)(-big_k1 - 1);
                        if (CF_marker_offd[loc_col] >= 0)
                        {
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }

         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_1 += wall_time;
            fflush(NULL);
         }
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds();
         }
         cnt_c = 0;
         cnt_f = jj_end_row - jj_begin_row;
         cnt_c_offd = 0;
         cnt_f_offd = jj_end_row_offd - jj_begin_row_offd;
         ihat[i] = cnt_f;
         ipnt[cnt_f] = i;
         ahat[cnt_f++] = A_diag_data[A_diag_i[i]];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is direct neighbor */
            i1 = A_diag_j[jj];
            if (P_marker[i1] != strong_f_marker)
            {
               indx = ihat[i1];
               if (indx > -1)
               {
                  ahat[indx] += A_diag_data[jj];
               }
               else if (P_marker[i1] >= jj_begin_row)
               {
                  ihat[i1] = cnt_c;
                  ipnt[cnt_c] = i1;
                  ahat[cnt_c++] += A_diag_data[jj];
               }
               else if (CF_marker[i1] != -3)
               {
                  ihat[i1] = cnt_f;
                  ipnt[cnt_f] = i1;
                  ahat[cnt_f++] += A_diag_data[jj];
               }
            }
            else
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  distribute = A_diag_data[jj] / A_diag_data[A_diag_i[i1]];
                  for (kk = A_diag_i[i1] + 1; kk < A_diag_i[i1 + 1]; kk++)
                  {
                     k1 = A_diag_j[kk];
                     indx = ihat[k1];
                     if (indx > -1)
                     {
                        ahat[indx] -= A_diag_data[kk] * distribute;
                     }
                     else if (P_marker[k1] >= jj_begin_row)
                     {
                        ihat[k1] = cnt_c;
                        ipnt[cnt_c] = k1;
                        ahat[cnt_c++] -= A_diag_data[kk] * distribute;
                     }
                     else
                     {
                        ihat[k1] = cnt_f;
                        ipnt[cnt_f] = k1;
                        ahat[cnt_f++] -= A_diag_data[kk] * distribute;
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = A_offd_i[i1]; kk < A_offd_i[i1 + 1]; kk++)
                     {
                        k1 = A_offd_j[kk];
                        indx = ihat_offd[k1];
                        if (num_functions == 1 || dof_func[i1] == dof_func_offd[k1])
                        {
                           if (indx > -1)
                           {
                              ahat_offd[indx] -= A_offd_data[kk] * distribute;
                           }
                           else if (P_marker_offd[k1] >= jj_begin_row_offd)
                           {
                              ihat_offd[k1] = cnt_c_offd;
                              ipnt_offd[cnt_c_offd] = k1;
                              ahat_offd[cnt_c_offd++] -= A_offd_data[kk] * distribute;
                           }
                           else
                           {
                              ihat_offd[k1] = cnt_f_offd;
                              ipnt_offd[cnt_f_offd] = k1;
                              ahat_offd[cnt_f_offd++] -= A_offd_data[kk] * distribute;
                           }
                        }
                     }
                  }
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] != strong_f_marker)
               {
                  indx = ihat_offd[i1];
                  if (indx > -1)
                  {
                     ahat_offd[indx] += A_offd_data[jj];
                  }
                  else if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     ihat_offd[i1] = cnt_c_offd;
                     ipnt_offd[cnt_c_offd] = i1;
                     ahat_offd[cnt_c_offd++] += A_offd_data[jj];
                  }
                  else if (CF_marker_offd[i1] != -3)
                  {
                     ihat_offd[i1] = cnt_f_offd;
                     ipnt_offd[cnt_f_offd] = i1;
                     ahat_offd[cnt_f_offd++] += A_offd_data[jj];
                  }
               }
               else
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     distribute = A_offd_data[jj] / A_ext_data[A_ext_i[i1]];
                     for (kk = A_ext_i[i1] + 1; kk < A_ext_i[i1 + 1]; kk++)
                     {
                        big_k1 = A_ext_j[kk];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /*diag*/
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           indx = ihat[loc_col];
                           if (indx > -1)
                           {
                              ahat[indx] -= A_ext_data[kk] * distribute;
                           }
                           else if (P_marker[loc_col] >= jj_begin_row)
                           {
                              ihat[loc_col] = cnt_c;
                              ipnt[cnt_c] = loc_col;
                              ahat[cnt_c++] -= A_ext_data[kk] * distribute;
                           }
                           else
                           {
                              ihat[loc_col] = cnt_f;
                              ipnt[cnt_f] = loc_col;
                              ahat[cnt_f++] -= A_ext_data[kk] * distribute;
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (num_functions == 1 || dof_func_offd[loc_col] == dof_func_offd[i1])
                           {
                              indx = ihat_offd[loc_col];
                              if (indx > -1)
                              {
                                 ahat_offd[indx] -= A_ext_data[kk] * distribute;
                              }
                              else if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                              {
                                 ihat_offd[loc_col] = cnt_c_offd;
                                 ipnt_offd[cnt_c_offd] = loc_col;
                                 ahat_offd[cnt_c_offd++] -= A_ext_data[kk] * distribute;
                              }
                              else
                              {
                                 ihat_offd[loc_col] = cnt_f_offd;
                                 ipnt_offd[cnt_f_offd] = loc_col;
                                 ahat_offd[cnt_f_offd++] -= A_ext_data[kk] * distribute;
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_2 += wall_time;
            fflush(NULL);
         }

         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }
         diagonal = ahat[cnt_c];
         ahat[cnt_c] = 0;
         sum_pos = 0;
         sum_pos_C = 0;
         sum_neg = 0;
         sum_neg_C = 0;
         sum = 0;
         sum_C = 0;
         if (sep_weight == 1)
         {
            for (jj = 0; jj < cnt_c; jj++)
            {
               if (ahat[jj] > 0)
               {
                  sum_pos_C += ahat[jj];
               }
               else
               {
                  sum_neg_C += ahat[jj];
               }
            }
            if (num_procs > 1)
            {
               for (jj = 0; jj < cnt_c_offd; jj++)
               {
                  if (ahat_offd[jj] > 0)
                  {
                     sum_pos_C += ahat_offd[jj];
                  }
                  else
                  {
                     sum_neg_C += ahat_offd[jj];
                  }
               }
            }
            sum_pos = sum_pos_C;
            sum_neg = sum_neg_C;
            for (jj = cnt_c + 1; jj < cnt_f; jj++)
            {
               if (ahat[jj] > 0)
               {
                  sum_pos += ahat[jj];
               }
               else
               {
                  sum_neg += ahat[jj];
               }
               ahat[jj] = 0;
            }
            if (num_procs > 1)
            {
               for (jj = cnt_c_offd; jj < cnt_f_offd; jj++)
               {
                  if (ahat_offd[jj] > 0)
                  {
                     sum_pos += ahat_offd[jj];
                  }
                  else
                  {
                     sum_neg += ahat_offd[jj];
                  }
                  ahat_offd[jj] = 0;
               }
            }
            if (sum_neg_C * diagonal != 0)
            {
               alfa = sum_neg / sum_neg_C / diagonal;
            }
            if (sum_pos_C * diagonal != 0)
            {
               beta = sum_pos / sum_pos_C / diagonal;
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               j1 = ihat[P_diag_j[jj]];
               if (ahat[j1] > 0)
               {
                  P_diag_data[jj] = -beta * ahat[j1];
               }
               else
               {
                  P_diag_data[jj] = -alfa * ahat[j1];
               }

               P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
               ahat[j1] = 0;
            }

            for (jj = 0; jj < cnt_f; jj++)
            {
               ihat[ipnt[jj]] = -1;
            }

            if (num_procs > 1)
            {
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  j1 = ihat_offd[P_offd_j[jj]];
                  if (ahat_offd[j1] > 0)
                  {
                     P_offd_data[jj] = -beta * ahat_offd[j1];
                  }
                  else
                  {
                     P_offd_data[jj] = -alfa * ahat_offd[j1];
                  }

                  ahat_offd[j1] = 0;
               }
               for (jj = 0; jj < cnt_f_offd; jj++)
               {
                  ihat_offd[ipnt_offd[jj]] = -1;
               }
            }
         }
         else
         {
            for (jj = 0; jj < cnt_c; jj++)
            {
               sum_C += ahat[jj];
            }
            if (num_procs > 1)
            {
               for (jj = 0; jj < cnt_c_offd; jj++)
               {
                  sum_C += ahat_offd[jj];
               }
            }
            sum = sum_C;
            for (jj = cnt_c + 1; jj < cnt_f; jj++)
            {
               sum += ahat[jj];
               ahat[jj] = 0;
            }
            if (num_procs > 1)
            {
               for (jj = cnt_c_offd; jj < cnt_f_offd; jj++)
               {
                  sum += ahat_offd[jj];
                  ahat_offd[jj] = 0;
               }
            }
            if (sum_C * diagonal != 0)
            {
               alfa = sum / sum_C / diagonal;
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               j1 = ihat[P_diag_j[jj]];
               P_diag_data[jj] = -alfa * ahat[j1];
               P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
               ahat[j1] = 0;
            }
            for (jj = 0; jj < cnt_f; jj++)
            {
               ihat[ipnt[jj]] = -1;
            }
            if (num_procs > 1)
            {
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  j1 = ihat_offd[P_offd_j[jj]];
                  P_offd_data[jj] = -alfa * ahat_offd[j1];
                  ahat_offd[j1] = 0;
               }
               for (jj = 0; jj < cnt_f_offd; jj++)
               {
                  ihat_offd[ipnt_offd[jj]] = -1;
               }
            }
         }
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            wall_3 += wall_time;
            fflush(NULL);
         }
      }
   }

   if (debug_flag == 4)
   {
      hypre_printf("Proc = %d fill part 1 %f part 2 %f  part 3 %f\n",
                   my_id, wall_1, wall_2, wall_3);
      fflush(NULL);
   }
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(ahat, HYPRE_MEMORY_HOST);
   hypre_TFree(ihat, HYPRE_MEMORY_HOST);
   hypre_TFree(ipnt, HYPRE_MEMORY_HOST);

   if (full_off_procNodes)
   {
      hypre_TFree(ahat_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(ihat_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(ipnt_offd, HYPRE_MEMORY_HOST);
   }
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }
      hypre_MatvecCommPkgDestroy(extend_comm_pkg);

   }

   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildExtPIInterp
 *  Comment:
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtPIInterpHost(hypre_ParCSRMatrix   *A,
                                    HYPRE_Int            *CF_marker,
                                    hypre_ParCSRMatrix   *S,
                                    HYPRE_BigInt         *num_cpts_global,
                                    HYPRE_Int             num_functions,
                                    HYPRE_Int            *dof_func,
                                    HYPRE_Int             debug_flag,
                                    HYPRE_Real            trunc_factor,
                                    HYPRE_Int             max_elmts,
                                    hypre_ParCSRMatrix  **P_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] -= hypre_MPI_Wtime();
#endif

   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /*HYPRE_Int             *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   HYPRE_Int        sgn = 1;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;

   /* Interpolation weight variables */
   HYPRE_Real       sum, diagonal, distribute;
   HYPRE_Int        strong_f_marker;

   /* Loop variables */
   /*HYPRE_Int              index;*/
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, i2, jj, kk, k1, jj1;
   HYPRE_BigInt     big_k1;

   /* Threading variables */
   HYPRE_Int my_thread_num, num_threads, start, stop;
   HYPRE_Int * max_num_threads = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);
   HYPRE_Int * diag_offset;
   HYPRE_Int * fine_to_coarse_offset;
   HYPRE_Int * offd_offset;

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   HYPRE_Real       wall_time;


   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 1);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd  = hypre_CTAlloc(HYPRE_Int,    full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   /* This function is smart enough to check P_marker and P_marker_offd only,
    * and set them if they are not NULL.  The other vectors are set regardless.*/
   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);


   /*-----------------------------------------------------------------------
    *  Initialize threading variables
    *-----------------------------------------------------------------------*/
   max_num_threads[0] = hypre_NumThreads();
   diag_offset           = hypre_CTAlloc(HYPRE_Int, max_num_threads[0], HYPRE_MEMORY_HOST);
   fine_to_coarse_offset = hypre_CTAlloc(HYPRE_Int, max_num_threads[0], HYPRE_MEMORY_HOST);
   offd_offset           = hypre_CTAlloc(HYPRE_Int, max_num_threads[0], HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_threads[0]; i++)
   {
      diag_offset[i] = 0;
      fine_to_coarse_offset[i] = 0;
      offd_offset[i] = 0;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,my_thread_num,num_threads,start,stop,coarse_counter,jj_counter,jj_counter_offd, P_marker, P_marker_offd,jj,kk,i1,k1,loc_col,jj_begin_row,jj_begin_row_offd,jj_end_row,jj_end_row_offd,diagonal,sum,sgn,jj1,i2,distribute,strong_f_marker, big_k1)
#endif
   {

      /* Parallelize by computing only over each thread's range of rows.
       *
       * The first large for loop computes ~locally~ for each thread P_diag_i,
       * P_offd_i and fine_to_coarse.  Then, the arrays are stitched together
       * For eaxample the first phase would compute
       * P_diag_i = [0, 2, 4, 7, 2, 5, 6]
       * for two threads.  P_diag_i[stop] points to the end of that
       * thread's data, but P_diag_i[start] points to the end of the
       * previous thread's row range.  This is then stitched together at the
       * end to yield,
       * P_diag_i = [0, 2, 4, 7, 9, 14, 15].
       *
       * The second large for loop computes interpolation weights and is
       * relatively straight-forward to thread.
       */

      /* initialize thread-wise variables */
      strong_f_marker = -2;
      coarse_counter = 0;
      jj_counter = start_indexing;
      jj_counter_offd = start_indexing;
      if (n_fine)
      {
         P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
         for (i = 0; i < n_fine; i++)
         {  P_marker[i] = -1; }
      }
      if (full_off_procNodes)
      {
         P_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
         for (i = 0; i < full_off_procNodes; i++)
         {  P_marker_offd[i] = -1;}
      }

      /* this thread's row range */
      my_thread_num = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {  stop = n_fine; }
      else
      {  stop = (n_fine / num_threads) * (my_thread_num + 1); }

      /* loop over rows */
      /* This loop counts the number of elements in P */
      /* is done by counting the elmements in the index set C-hat */

      for (i = start; i < stop; i++)
      {
         P_diag_i[i] = jj_counter;
         if (num_procs > 1)
         {
            P_offd_i[i] = jj_counter_offd;
         }

         if (CF_marker[i] >= 0)
         {
            /* row in P corresponding to a coarse pt., will only require one element (1 on the diagonal). */
            jj_counter++;
            fine_to_coarse[i] = coarse_counter;
            coarse_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i, or C-points that stronly influence F-points
          *  that strongly influence i.
          *--------------------------------------------------------------------*/
         else if (CF_marker[i] != -3)
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  /* i1 is a C point */
                  if (P_marker[i1] < P_diag_i[i])
                  {
                     P_marker[i1] = jj_counter;
                     jj_counter++;
                  }
               }
               else if (CF_marker[i1] != -3)
               {
                  /* i1 is a F point, loop through it's strong neighbors */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < P_diag_i[i])
                        {
                           P_marker[k1] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < P_offd_i[i])
                           {
                              tmp_CF_marker_offd[k1] = 1;
                              P_marker_offd[k1] = jj_counter_offd;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
            /* Look at off diag strong connections of i */
            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     if (P_marker_offd[i1] < P_offd_i[i])
                     {
                        tmp_CF_marker_offd[i1] = 1;
                        P_marker_offd[i1] = jj_counter_offd;
                        jj_counter_offd++;
                     }
                  }
                  else if (CF_marker_offd[i1] != -3)
                  {
                     /* F point; look at neighbors of i1. Sop contains global col
                      * numbers and entries that could be in S_diag or S_offd or
                      * neither. */
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        big_k1 = Sop_j[kk];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* In S_diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
      /*-----------------------------------------------------------------------
       *  End loop over fine grid.
       *-----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      P_diag_i[stop] = jj_counter;
      P_offd_i[stop] = jj_counter_offd;
      fine_to_coarse_offset[my_thread_num] = coarse_counter;
      diag_offset[my_thread_num] = jj_counter;
      offd_offset[my_thread_num] = jj_counter_offd;

      /* Stitch P_diag_i, P_offd_i and fine_to_coarse together */
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         /* Calculate the offset for P_diag_i and P_offd_i for each thread */
         for (i = 1; i < num_threads; i++)
         {
            diag_offset[i] = diag_offset[i - 1] + diag_offset[i];
            fine_to_coarse_offset[i] = fine_to_coarse_offset[i - 1] + fine_to_coarse_offset[i];
            offd_offset[i] = offd_offset[i - 1] + offd_offset[i];
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (my_thread_num > 0)
      {
         /* update row pointer array with offset,
          * making sure to update the row stop index */
         for (i = start + 1; i <= stop; i++)
         {
            P_diag_i[i] += diag_offset[my_thread_num - 1];
            P_offd_i[i] += offd_offset[my_thread_num - 1];
         }
         /* update fine_to_coarse by offsetting with the offset
          * from the preceding thread */
         for (i = start; i < stop; i++)
         {
            if (fine_to_coarse[i] >= 0)
            { fine_to_coarse[i] += fine_to_coarse_offset[my_thread_num - 1]; }
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (my_thread_num == 0)
      {
         if (debug_flag == 4)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            hypre_printf("Proc = %d     determine structure    %f\n",
                         my_id, wall_time);
            fflush(NULL);
         }
         /*-----------------------------------------------------------------------
          *  Allocate  arrays.
          *-----------------------------------------------------------------------*/

         if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

         P_diag_size =  P_diag_i[n_fine];
         P_offd_size = P_offd_i[n_fine];

         if (P_diag_size)
         {
            P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
            P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
         }

         if (P_offd_size)
         {
            P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
            P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
         }
      }

      /* Fine to coarse mapping */
      if (num_procs > 1   &&   my_thread_num == 0)
      {
         hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                    full_off_procNodes, my_first_cpt,
                                    fine_to_coarse_offd);
      }

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }

      for (i = 0; i < full_off_procNodes; i++)
      {
         P_marker_offd[i] = -1;
      }

      /*-----------------------------------------------------------------------
       *  Loop over fine grid points.
       *-----------------------------------------------------------------------*/
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      for (i = start; i < stop; i++)
      {
         jj_begin_row = P_diag_i[i];
         jj_begin_row_offd = P_offd_i[i];
         jj_counter = jj_begin_row;
         jj_counter_offd = jj_begin_row_offd;

         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }

         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/

         else if (CF_marker[i] != -3)
         {
            strong_f_marker--;
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  if (P_marker[i1] < jj_begin_row)
                  {
                     P_marker[i1] = jj_counter;
                     P_diag_j[jj_counter]    = fine_to_coarse[i1];
                     P_diag_data[jj_counter] = zero;
                     jj_counter++;
                  }
               }
               else  if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < jj_begin_row)
                        {
                           P_marker[k1] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[k1];
                           P_diag_data[jj_counter] = zero;
                           jj_counter++;
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < jj_begin_row_offd)
                           {
                              P_marker_offd[k1] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = k1;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }

            if ( num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if ( CF_marker_offd[i1] >= 0)
                  {
                     if (P_marker_offd[i1] < jj_begin_row_offd)
                     {
                        P_marker_offd[i1] = jj_counter_offd;
                        P_offd_j[jj_counter_offd] = i1;
                        P_offd_data[jj_counter_offd] = zero;
                        jj_counter_offd++;
                     }
                  }
                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        big_k1 = Sop_j[kk];
                        /* Find local col number */
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                              P_diag_data[jj_counter] = zero;
                              jj_counter++;
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }

            jj_end_row = jj_counter;
            jj_end_row_offd = jj_counter_offd;

            diagonal = A_diag_data[A_diag_i[i]];

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               /* i1 is a c-point and strongly influences i, accumulate
                * a_(i,i1) into interpolation weight */
               i1 = A_diag_j[jj];
               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }
               else if (P_marker[i1] == strong_f_marker)
               {
                  sum = zero;
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
                  /* Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i. */
                  for (jj1 = A_diag_i[i1] + 1; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if ((P_marker[i2] >= jj_begin_row || i2 == i) && (sgn * A_diag_data[jj1]) < 0)
                     {
                        sum += A_diag_data[jj1];
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd &&
                            (sgn * A_offd_data[jj1]) < 0)
                        {
                           sum += A_offd_data[jj1];
                        }
                     }
                  }
                  if (sum != 0)
                  {
                     distribute = A_diag_data[jj] / sum;
                     /* Loop over row of A for point i1 and do the distribution */
                     for (jj1 = A_diag_i[i1] + 1; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                           P_diag_data[P_marker[i2]] +=
                              distribute * A_diag_data[jj1];
                        if (i2 == i && (sgn * A_diag_data[jj1]) < 0)
                        {
                           diagonal += distribute * A_diag_data[jj1];
                        }
                     }
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd &&
                               (sgn * A_offd_data[jj1]) < 0)
                              P_offd_data[P_marker_offd[i2]] +=
                                 distribute * A_offd_data[jj1];
                        }
                     }
                  }
                  else
                  {
                     diagonal += A_diag_data[jj];
                  }
               }
               /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
                * diagonal */
               else if (CF_marker[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  {
                     diagonal += A_diag_data[jj];
                  }
               }
            }
            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }
                  else if (P_marker_offd[i1] == strong_f_marker)
                  {
                     sum = zero;
                     for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                     {
                        big_k1 = A_ext_j[jj1];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] >= jj_begin_row || loc_col == i)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                     }
                     if (sum != 0)
                     {
                        distribute = A_offd_data[jj] / sum;
                        for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                        {
                           big_k1 = A_ext_j[jj1];
                           if (big_k1 >= col_1 && big_k1 < col_n)
                           {
                              /* diag */
                              loc_col = (HYPRE_Int)(big_k1 - col_1);
                              if (P_marker[loc_col] >= jj_begin_row)
                                 P_diag_data[P_marker[loc_col]] += distribute *
                                                                   A_ext_data[jj1];
                              if (loc_col == i)
                              {
                                 diagonal += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              loc_col = (HYPRE_Int)(-big_k1 - 1);
                              if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                                 P_offd_data[P_marker_offd[loc_col]] += distribute *
                                                                        A_ext_data[jj1];
                           }
                        }
                     }
                     else
                     {
                        diagonal += A_offd_data[jj];
                     }
                  }
                  else if (CF_marker_offd[i1] != -3)
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     {
                        diagonal += A_offd_data[jj];
                     }
                  }
               }
            }
            if (diagonal)
            {
               for (jj = jj_begin_row; jj < jj_end_row; jj++)
               {
                  P_diag_data[jj] /= -diagonal;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  P_offd_data[jj] /= -diagonal;
               }
            }
         }
         strong_f_marker--;
      }
      /*-----------------------------------------------------------------------
       *  End large for loop over nfine
       *-----------------------------------------------------------------------*/

      if (n_fine)
      {
         hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      }

      if (full_off_procNodes)
      {
         hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      }
   }
   /*-----------------------------------------------------------------------
    *  End PAR_REGION
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     fill structure    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] -= hypre_MPI_Wtime();
#endif

      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(max_num_threads, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_offset, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_offset, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offset, HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildExtPICCInterp
 *  Comment: Only use FF when there is no common c point.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtPICCInterp(hypre_ParCSRMatrix  *A,
                                  HYPRE_Int           *CF_marker,
                                  hypre_ParCSRMatrix  *S,
                                  HYPRE_BigInt        *num_cpts_global,
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
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /*HYPRE_Int             *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;
   /*HYPRE_Int             **ext_p, **ext_p_offd;*/
   /*HYPRE_Int              ccounter_offd;
     HYPRE_Int             *clist_offd;*/
   HYPRE_Int        common_c;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   HYPRE_Int        sgn = 1;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;

   /* Interpolation weight variables */
   HYPRE_Real       sum, diagonal, distribute;
   HYPRE_Int        strong_f_marker = -2;

   /* Loop variables */
   /*HYPRE_Int              index;*/
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, i2, jj, kk, k1, jj1;
   HYPRE_BigInt     big_k1;
   /*HYPRE_Int              ccounter;
     HYPRE_Int             *clist, ccounter;*/

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;

   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 1);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   /*clist = hypre_CTAlloc(HYPRE_Int, MAX_C_CONNECTIONS);
     for (i = 0; i < MAX_C_CONNECTIONS; i++)
     clist[i] = 0;
     if (num_procs > 1)
     {
     clist_offd = hypre_CTAlloc(HYPRE_Int,  MAX_C_CONNECTIONS, HYPRE_MEMORY_HOST);
     for (i = 0; i < MAX_C_CONNECTIONS; i++)
     clist_offd[i] = 0;
     }*/

   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else if (CF_marker[i] != -3)
      {
         /* Initialize ccounter for each f point */
         /*ccounter = 0;
           ccounter_offd = 0;*/
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] > 0)
            {
               /* i1 is a C point */
               CF_marker[i1] = 2;
               if (P_marker[i1] < P_diag_i[i])
               {
                  /*clist[ccounter++] = i1;*/
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
         }
         /*qsort0(clist,0,ccounter-1);*/
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] > 0)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 2;
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     /*clist_offd[ccounter_offd++] = i1;*/
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
            }
            /*qsort0(clist_offd,0,ccounter_offd-1);*/
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search diag to find f neighbors and determine if common c point */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == -1)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     /*if (hypre_BinarySearch(clist,k1,ccounter) >= 0)
                       {*/
                     common_c = 1;
                     break;
                     /*kk = S_diag_i[i1+1];
                       }*/
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        /* k1 is a c point check if it is common */
                        /*if (hypre_BinarySearch(clist_offd,k1,ccounter_offd) >= 0)
                          {*/
                        common_c = 1;
                        break;
                        /*kk = S_offd_i[i1+1];
                          }*/
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] > 0)
                     {
                        if (P_marker[k1] < P_diag_i[i])
                        {
                           P_marker[k1] = jj_counter;
                           jj_counter++;
                           /*break;*/
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >  0)
                        {
                           if (P_marker_offd[k1] < P_offd_i[i])
                           {
                              tmp_CF_marker_offd[k1] = 1;
                              P_marker_offd[k1] = jj_counter_offd;
                              jj_counter_offd++;
                              /*break;*/
                           }
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == -1)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                   * numbers and entries that could be in S_diag or S_offd or
                   * neither. */
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           /*if (hypre_BinarySearch(clist,loc_col,ccounter) >= 0)
                             {*/
                           common_c = 1;
                           break;
                           /*kk = Sop_i[i1+1];
                             }*/
                        }
                     }
                     else
                     {
                        loc_col = (HYPRE_BigInt)(-big_k1 - 1);
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           /*if (hypre_BinarySearch(clist_offd,loc_col,ccounter_offd) >=
                             0)
                             {*/
                           common_c = 1;
                           break;
                           /*kk = Sop_i[i1+1];
                             }*/
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        /* Check if common c */
                        big_k1 = Sop_j[kk];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* In S_diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                              /*break;*/
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                              /*break;*/
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == 2)
            {
               CF_marker[i1] = 1;
            }
         }
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == 2)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 1;
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
   }

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   /*ccounter = start_indexing;
     ccounter_offd = start_indexing;*/

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                 full_off_procNodes, my_first_cpt,
                                 fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
   }

   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         /*ccounter = 0;
           ccounter_offd = 0;*/
         strong_f_marker--;

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >  0)
            {
               CF_marker[i1]  = 2;
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
                  /*clist[ccounter++] = i1;*/
               }
            }
         }
         /*qsort0(clist,0,ccounter-1);*/
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] > 0)
               {
                  CF_marker_offd[i1]  = 2;
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                     /*clist_offd[ccounter_offd++] = i1;*/
                  }
               }
            }
            /*qsort0(clist_offd,0,ccounter_offd-1);*/
         }

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search through F points */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == -1)
            {
               P_marker[i1] = strong_f_marker;
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     /*if (hypre_BinarySearch(clist,k1,ccounter) >= 0)
                       {*/
                     common_c = 1;
                     break;
                     /*kk = S_diag_i[i1+1];
                       }*/
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        /* k1 is a c point check if it is common */
                        /*if (hypre_BinarySearch(clist_offd,k1,ccounter_offd) >= 0)
                          {*/
                        common_c = 1;
                        break;
                        /*kk = S_offd_i[i1+1];
                          }*/
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < jj_begin_row)
                        {
                           P_marker[k1] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[k1];
                           P_diag_data[jj_counter] = zero;
                           jj_counter++;
                           /*break;*/
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < jj_begin_row_offd)
                           {
                              P_marker_offd[k1] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = k1;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                              /*break;*/
                           }
                        }
                     }
                  }
               }
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == -1)
               {
                  /* F points that are off proc */
                  P_marker_offd[i1] = strong_f_marker;
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           /*if (hypre_BinarySearch(clist,loc_col,ccounter) >= 0)
                             {*/
                           common_c = 1;
                           break;
                           /*kk = Sop_i[i1+1];
                             }*/
                        }
                     }
                     else
                     {
                        loc_col = (HYPRE_Int)(-big_k1 - 1);
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           /*if (hypre_BinarySearch(clist_offd,loc_col,ccounter_offd) >=
                             0)
                             {*/
                           common_c = 1;
                           break;
                           /*kk = Sop_i[i1+1];
                             }*/
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        big_k1 = Sop_j[kk];
                        /* Find local col number */
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                              P_diag_data[jj_counter] = zero;
                              jj_counter++;
                              /*break;*/
                           }
                        }
                        else
                        {
                           loc_col = (-big_k1 - 1);
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                              /*break;*/
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] == 2)
            {
               CF_marker[i1]  = 1;
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] == 2)
               {
                  CF_marker_offd[i1]  = 1;
               }
            }
         }


         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         diagonal = A_diag_data[A_diag_i[i]];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is a c-point and strongly influences i, accumulate
             * a_(i,i1) into interpolation weight */
            i1 = A_diag_j[jj];
            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               sgn = 1;
               if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
               for (jj1 = A_diag_i[i1] + 1; jj1 < A_diag_i[i1 + 1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if ((P_marker[i2] >= jj_begin_row || i2 == i)  && (sgn * A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd &&
                         (sgn * A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               }
               if (sum != 0)
               {
                  distribute = A_diag_data[jj] / sum;
                  /* Loop over row of A for point i1 and do the distribution */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                        P_diag_data[P_marker[i2]] +=
                           distribute * A_diag_data[jj1];
                     if (i2 == i && (sgn * A_diag_data[jj1]) < 0)
                     {
                        diagonal += distribute * A_diag_data[jj1];
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd &&
                            (sgn * A_offd_data[jj1]) < 0)
                           P_offd_data[P_marker_offd[i2]] +=
                              distribute * A_offd_data[jj1];
                     }
                  }
               }
               else
               {
                  diagonal += A_diag_data[jj];
               }
            }
            /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
             * diagonal */
            else if (CF_marker[i1] != -3)
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  diagonal += A_diag_data[jj];
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
                  sgn = 1;
                  for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                  {
                     big_k1 = A_ext_j[jj1];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] >= jj_begin_row || loc_col == i)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        loc_col = (HYPRE_Int)(-big_k1 - 1);
                        if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                  }
                  if (sum != 0)
                  {
                     distribute = A_offd_data[jj] / sum;
                     for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                     {
                        big_k1 = A_ext_j[jj1];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] >= jj_begin_row)
                              P_diag_data[P_marker[loc_col]] += distribute *
                                                                A_ext_data[jj1];
                           if (loc_col == i)
                           {
                              diagonal += distribute * A_ext_data[jj1];
                           }
                        }
                        else
                        {
                           loc_col = (HYPRE_Int)(-big_k1 - 1);
                           if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                              P_offd_data[P_marker_offd[loc_col]] += distribute *
                                                                     A_ext_data[jj1];
                        }
                     }
                  }
                  else
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
            }
         }
         if (diagonal)
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
      }
      strong_f_marker--;
   }

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   /*hypre_TFree(clist);*/

   if (num_procs > 1)
   {
      /*hypre_TFree(clist_offd);*/
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);
   }

   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildFFInterp
 *  Comment: Only use FF when there is no common c point.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildFFInterp(hypre_ParCSRMatrix  *A,
                             HYPRE_Int           *CF_marker,
                             hypre_ParCSRMatrix  *S,
                             HYPRE_BigInt        *num_cpts_global,
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
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix *P_diag;
   hypre_CSRMatrix *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /*HYPRE_Int             *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;
   /*HYPRE_Int              ccounter_offd;*/
   HYPRE_Int        common_c;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;

   /* Interpolation weight variables */
   HYPRE_Real       sum, diagonal, distribute;
   HYPRE_Int        strong_f_marker = -2;
   HYPRE_Int        sgn = 1;

   /* Loop variables */
   /*HYPRE_Int              index;*/
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, i2, jj, kk, k1, jj1;
   HYPRE_BigInt     big_k1;
   /*HYPRE_Int              ccounter;
     HYPRE_Int             *clist, ccounter;*/

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;

   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 1);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else
      {
         /* Initialize ccounter for each f point */
         /*ccounter = 0;
           ccounter_offd = 0;*/
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] > 0)
            {
               /* i1 is a C point */
               CF_marker[i1] = 2;
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] > 0)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 2;
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search diag to find f neighbors and determine if common c point */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] < 0)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     common_c = 1;
                     break;
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        common_c = 1;
                        break;
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] > 0)
                     {
                        if (P_marker[k1] < P_diag_i[i])
                        {
                           P_marker[k1] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >  0)
                        {
                           if (P_marker_offd[k1] < P_offd_i[i])
                           {
                              tmp_CF_marker_offd[k1] = 1;
                              P_marker_offd[k1] = jj_counter_offd;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] < 0)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                   * numbers and entries that could be in S_diag or S_offd or
                   * neither. */
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        /* Check if common c */
                        big_k1 = Sop_j[kk];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* In S_diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                           }
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == 2)
            {
               CF_marker[i1] = 1;
            }
         }
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == 2)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 1;
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
   }

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   /*ccounter = start_indexing;
     ccounter_offd = start_indexing;*/

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                 full_off_procNodes, my_first_cpt,
                                 fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
   }

   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   jj_begin_row_offd = 0;
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         /*ccounter = 0;
           ccounter_offd = 0;*/
         strong_f_marker--;

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >  0)
            {
               CF_marker[i1]  = 2;
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] > 0)
               {
                  CF_marker_offd[i1]  = 2;
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
            }
         }

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search through F points */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == -1)
            {
               P_marker[i1] = strong_f_marker;
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     common_c = 1;
                     break;
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        common_c = 1;
                        break;
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < jj_begin_row)
                        {
                           P_marker[k1] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[k1];
                           P_diag_data[jj_counter] = zero;
                           jj_counter++;
                        }
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < jj_begin_row_offd)
                           {
                              P_marker_offd[k1] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = k1;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == -1)
               {
                  /* F points that are off proc */
                  P_marker_offd[i1] = strong_f_marker;
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        big_k1 = Sop_j[kk];
                        /* Find local col number */
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                              P_diag_data[jj_counter] = zero;
                              jj_counter++;
                           }
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] == 2)
            {
               CF_marker[i1]  = 1;
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] == 2)
               {
                  CF_marker_offd[i1]  = 1;
               }
            }
         }


         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         diagonal = A_diag_data[A_diag_i[i]];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is a c-point and strongly influences i, accumulate
             * a_(i,i1) into interpolation weight */
            i1 = A_diag_j[jj];
            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
               /* Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly incluence i. */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd &&
                         (sgn * A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               }
               if (sum != 0)
               {
                  distribute = A_diag_data[jj] / sum;
                  /* Loop over row of A for point i1 and do the distribution */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                        P_diag_data[P_marker[i2]] +=
                           distribute * A_diag_data[jj1];
                  }
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd &&
                            (sgn * A_offd_data[jj1]) < 0)
                           P_offd_data[P_marker_offd[i2]] +=
                              distribute * A_offd_data[jj1];
                     }
                  }
               }
               else
               {
                  diagonal += A_diag_data[jj];
               }
            }
            /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
             * diagonal */
            else if (CF_marker[i1] != -3)
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  diagonal += A_diag_data[jj];
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
                  for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                  {
                     big_k1 = A_ext_j[jj1];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] >= jj_begin_row)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                  }
                  if (sum != 0)
                  {
                     distribute = A_offd_data[jj] / sum;
                     for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                     {
                        big_k1 = A_ext_j[jj1];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] >= jj_begin_row)
                              P_diag_data[P_marker[loc_col]] += distribute *
                                                                A_ext_data[jj1];
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                              P_offd_data[P_marker_offd[loc_col]] += distribute *
                                                                     A_ext_data[jj1];
                        }
                     }
                  }
                  else
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
            }
         }
         if (diagonal)
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
      }
      strong_f_marker--;
   }

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);

   }

   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildFF1Interp
 *  Comment: Only use FF when there is no common c point.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildFF1Interp(hypre_ParCSRMatrix  *A,
                              HYPRE_Int           *CF_marker,
                              hypre_ParCSRMatrix  *S,
                              HYPRE_BigInt        *num_cpts_global,
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
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /*HYPRE_Int             *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;
   /*HYPRE_Int             ccounter_offd;*/
   HYPRE_Int        common_c;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;

   /* Interpolation weight variables */
   HYPRE_Real       sum, diagonal, distribute;
   HYPRE_Int        strong_f_marker = -2;
   HYPRE_Int        sgn = 1;

   /* Loop variables */
   /*HYPRE_Int              index;*/
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, i2, jj, kk, k1, jj1;
   HYPRE_BigInt     big_k1;
   /*HYPRE_Int              ccounter;*/
   HYPRE_Int              found_c = 0;

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;

   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;
   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 1);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt,  full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else
      {
         /* Initialize ccounter for each f point */
         /*ccounter = 0;
           ccounter_offd = 0;*/
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] > 0)
            {
               /* i1 is a C point */
               CF_marker[i1] = 2;
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] > 0)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 2;
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search diag to find f neighbors and determine if common c point */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] < 0)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     common_c = 1;
                     break;
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        /* k1 is a c point check if it is common */
                        common_c = 1;
                        break;
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  found_c = 0;
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] > 0)
                     {
                        if (P_marker[k1] < P_diag_i[i])
                        {
                           P_marker[k1] = jj_counter;
                           jj_counter++;
                           found_c = 1;
                           break;
                        }
                     }
                  }
                  if (num_procs > 1 && !found_c)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >  0)
                        {
                           if (P_marker_offd[k1] < P_offd_i[i])
                           {
                              tmp_CF_marker_offd[k1] = 1;
                              P_marker_offd[k1] = jj_counter_offd;
                              jj_counter_offd++;
                              break;
                           }
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] < 0)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                   * numbers and entries that could be in S_diag or S_offd or
                   * neither. */
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        /* Check if common c */
                        big_k1 = Sop_j[kk];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* In S_diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < P_diag_i[i])
                           {
                              P_marker[loc_col] = jj_counter;
                              jj_counter++;
                              break;
                           }
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] < P_offd_i[i])
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              tmp_CF_marker_offd[loc_col] = 1;
                              jj_counter_offd++;
                              break;
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == 2)
            {
               CF_marker[i1] = 1;
            }
         }
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               /* search through offd to find all c neighbors */
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == 2)
               {
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 1;
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
   }

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   /*ccounter = start_indexing;
     ccounter_offd = start_indexing;*/

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                 full_off_procNodes, my_first_cpt,
                                 fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
   }

   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   jj_begin_row_offd = 0;
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         /*ccounter = 0;
           ccounter_offd = 0;*/
         strong_f_marker--;

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >  0)
            {
               CF_marker[i1]  = 2;
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] > 0)
               {
                  CF_marker_offd[i1]  = 2;
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
            }
         }

         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search through F points */
            i1 = S_diag_j[jj];
            if (CF_marker[i1] == -1)
            {
               P_marker[i1] = strong_f_marker;
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     common_c = 1;
                     break;
                  }
               }
               if (num_procs > 1 && common_c == 0)
               {
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];

                     if (CF_marker_offd[k1] == 2)
                     {
                        /* k1 is a c point check if it is common */
                        common_c = 1;
                        break;
                     }
                  }
               }
               if (!common_c)
               {
                  /* No common c point, extend the interp set */
                  found_c = 0;
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < jj_begin_row)
                        {
                           P_marker[k1] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[k1];
                           P_diag_data[jj_counter] = zero;
                           jj_counter++;
                           found_c = 1;
                           break;
                        }
                     }
                  }
                  if (num_procs > 1 && !found_c)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                     {
                        k1 = S_offd_j[kk];
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < jj_begin_row_offd)
                           {
                              P_marker_offd[k1] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = k1;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                              break;
                           }
                        }
                     }
                  }
               }
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] == -1)
               {
                  /* F points that are off proc */
                  P_marker_offd[i1] = strong_f_marker;
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     /* Check if common c */
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  }
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                     {
                        big_k1 = Sop_j[kk];
                        /* Find local col number */
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] < jj_begin_row)
                           {
                              P_marker[loc_col] = jj_counter;
                              P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                              P_diag_data[jj_counter] = zero;
                              jj_counter++;
                              break;
                           }
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] < jj_begin_row_offd)
                           {
                              P_marker_offd[loc_col] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = loc_col;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd++;
                              break;
                           }
                        }
                     }
                  }
               }
            }
         }
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            /* Search C points only */
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] == 2)
            {
               CF_marker[i1]  = 1;
            }
         }
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] == 2)
               {
                  CF_marker_offd[i1]  = 1;
               }
            }
         }


         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         diagonal = A_diag_data[A_diag_i[i]];
         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is a c-point and strongly influences i, accumulate
             * a_(i,i1) into interpolation weight */
            i1 = A_diag_j[jj];
            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
               /* Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly incluence i. */
               for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd &&
                         (sgn * A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               }
               if (sum != 0)
               {
                  distribute = A_diag_data[jj] / sum;
                  /* Loop over row of A for point i1 and do the distribution */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                        P_diag_data[P_marker[i2]] +=
                           distribute * A_diag_data[jj1];
                  }
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd &&
                            (sgn * A_offd_data[jj1]) < 0)
                           P_offd_data[P_marker_offd[i2]] +=
                              distribute * A_offd_data[jj1];
                     }
                  }
               }
               else
               {
                  diagonal += A_diag_data[jj];
               }
            }
            /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
             * diagonal */
            else if (CF_marker[i1] != -3)
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  diagonal += A_diag_data[jj];
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
                  for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                  {
                     big_k1 = A_ext_j[jj1];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] >= jj_begin_row)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                  }
                  if (sum != 0)
                  {
                     distribute = A_offd_data[jj] / sum;
                     for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                     {
                        big_k1 = A_ext_j[jj1];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] >= jj_begin_row)
                              P_diag_data[P_marker[loc_col]] += distribute *
                                                                A_ext_data[jj1];
                        }
                        else
                        {
                           loc_col = - (HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                              P_offd_data[P_marker_offd[loc_col]] += distribute *
                                                                     A_ext_data[jj1];
                        }
                     }
                  }
                  else
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
            }
         }
         if (diagonal)
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
      }
      strong_f_marker--;
   }

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   /*hynre_TFree(clist);*/

   if (num_procs > 1)
   {

      /*hypre_TFree(clist_offd);*/
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd, HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);

   }

   return hypre_error_flag;
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildExtInterp
 *  Comment:
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGBuildExtInterpHost(hypre_ParCSRMatrix  *A,
                                  HYPRE_Int           *CF_marker,
                                  hypre_ParCSRMatrix  *S,
                                  HYPRE_BigInt        *num_cpts_global,
                                  HYPRE_Int            num_functions,
                                  HYPRE_Int           *dof_func,
                                  HYPRE_Int            debug_flag,
                                  HYPRE_Real           trunc_factor,
                                  HYPRE_Int            max_elmts,
                                  hypre_ParCSRMatrix **P_ptr)
{
   /* Communication Variables */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int                my_id, num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* Variables to store input variables */
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /*HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
     HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);*/
   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int        local_numrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
   HYPRE_BigInt     total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int       *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int       *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int       *S_offd_j = hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data = NULL;
   HYPRE_Int       *P_diag_i, *P_diag_j = NULL;
   HYPRE_Real      *P_offd_data = NULL;
   HYPRE_Int       *P_offd_i, *P_offd_j = NULL;

   /*HYPRE_Int             *col_map_offd_P = NULL;*/
   HYPRE_Int        P_diag_size;
   HYPRE_Int        P_offd_size;
   HYPRE_Int       *P_marker = NULL;
   HYPRE_Int       *P_marker_offd = NULL;
   HYPRE_Int       *CF_marker_offd = NULL;
   HYPRE_Int       *tmp_CF_marker_offd = NULL;
   HYPRE_Int       *dof_func_offd = NULL;

   /* Full row information for columns of A that are off diag*/
   hypre_CSRMatrix *A_ext = NULL;
   HYPRE_Real      *A_ext_data = NULL;
   HYPRE_Int       *A_ext_i = NULL;
   HYPRE_BigInt    *A_ext_j = NULL;

   HYPRE_Int       *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   HYPRE_Int        loc_col;
   HYPRE_Int        full_off_procNodes;

   hypre_CSRMatrix *Sop = NULL;
   HYPRE_Int       *Sop_i = NULL;
   HYPRE_BigInt    *Sop_j = NULL;

   HYPRE_Int        sgn = 1;

   /* Variables to keep count of interpolatory points */
   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int        jj_begin_row, jj_end_row;
   HYPRE_Int        jj_begin_row_offd = 0;
   HYPRE_Int        jj_end_row_offd = 0;
   HYPRE_Int        coarse_counter;

   /* Interpolation weight variables */
   HYPRE_Real       sum, diagonal, distribute;
   HYPRE_Int        strong_f_marker = -2;

   /* Loop variables */
   /*HYPRE_Int              index;*/
   HYPRE_Int        start_indexing = 0;
   HYPRE_Int        i, i1, i2, jj, kk, k1, jj1;
   HYPRE_BigInt     big_k1;

   /* Definitions */
   HYPRE_Real       zero = 0.0;
   HYPRE_Real       one  = 1.0;
   HYPRE_Real       wall_time;


   hypre_ParCSRCommPkg   *extend_comm_pkg = NULL;

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /* BEGIN */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set up off processor information (specifically for neighbors of
    * neighbors */
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
      hypre_exchange_interp_data(
         &CF_marker_offd, &dof_func_offd, &A_ext, &full_off_procNodes, &Sop, &extend_comm_pkg,
         A, CF_marker, S, num_functions, dof_func, 1);
      {
#ifdef HYPRE_PROFILE
         hypre_profile_times[HYPRE_TIMER_ID_EXTENDED_I_INTERP] += hypre_MPI_Wtime();
#endif
      }

      A_ext_i       = hypre_CSRMatrixI(A_ext);
      A_ext_j       = hypre_CSRMatrixBigJ(A_ext);
      A_ext_data    = hypre_CSRMatrixData(A_ext);

      Sop_i         = hypre_CSRMatrixI(Sop);
      Sop_j         = hypre_CSRMatrixBigJ(Sop);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i = hypre_CTAlloc(HYPRE_Int, n_fine + 1, memory_location_P);

   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      P_marker       = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
   }

   if (full_off_procNodes)
   {
      P_marker_offd       = hypre_CTAlloc(HYPRE_Int,    full_off_procNodes, HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes, HYPRE_MEMORY_HOST);
      tmp_CF_marker_offd  = hypre_CTAlloc(HYPRE_Int,    full_off_procNodes, HYPRE_MEMORY_HOST);
   }

   hypre_initialize_vecs(n_fine, full_off_procNodes, fine_to_coarse,
                         fine_to_coarse_offd, P_marker, P_marker_offd,
                         tmp_CF_marker_offd);

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }

      if (CF_marker[i] >= 0)
      {
         jj_counter++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, interpolation is from the C-points that
       *  strongly influence i, or C-points that stronly influence F-points
       *  that strongly influence i.
       *--------------------------------------------------------------------*/
      else if (CF_marker[i] != -3)
      {
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];
            if (CF_marker[i1] >= 0)
            {
               /* i1 is a C point */
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter++;
               }
            }
            else if (CF_marker[i1] != -3)
            {
               /* i1 is a F point, loop through it's strong neighbors */
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < P_diag_i[i])
                     {
                        P_marker[k1] = jj_counter;
                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < P_offd_i[i])
                        {
                           tmp_CF_marker_offd[k1] = 1;
                           P_marker_offd[k1] = jj_counter_offd;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }
         /* Look at off diag strong connections of i */
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if (CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  /* F point; look at neighbors of i1. Sop contains global col
                   * numbers and entries that could be in S_diag or S_offd or
                   * neither. */
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     big_k1 = Sop_j[kk];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* In S_diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] < P_diag_i[i])
                        {
                           P_marker[loc_col] = jj_counter;
                           jj_counter++;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (P_marker_offd[loc_col] < P_offd_i[i])
                        {
                           P_marker_offd[loc_col] = jj_counter_offd;
                           tmp_CF_marker_offd[loc_col] = 1;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     determine structure    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds();
   }

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
      P_diag_data = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location_P);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
      P_offd_data = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location_P);
   }

   P_diag_i[n_fine] = jj_counter;
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      hypre_big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse,
                                 full_off_procNodes, my_first_cpt,
                                 fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_marker[i] = -1;
   }

   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      jj_begin_row = jj_counter;
      jj_begin_row_offd = jj_counter_offd;

      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }

      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/

      else if (CF_marker[i] != -3)
      {
         strong_f_marker--;
         for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
         {
            i1 = S_diag_j[jj];

            /*--------------------------------------------------------------
             * If neighbor i1 is a C-point, set column number in P_diag_j
             * and initialize interpolation weight to zero.
             *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }
            }
            else  if (CF_marker[i1] != -3)
            {
               P_marker[i1] = strong_f_marker;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1 + 1]; kk++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] >= 0)
                  {
                     if (P_marker[k1] < jj_begin_row)
                     {
                        P_marker[k1] = jj_counter;
                        P_diag_j[jj_counter] = fine_to_coarse[k1];
                        P_diag_data[jj_counter] = zero;
                        jj_counter++;
                     }
                  }
               }
               if (num_procs > 1)
               {
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1 + 1]; kk++)
                  {
                     k1 = S_offd_j[kk];
                     if (CF_marker_offd[k1] >= 0)
                     {
                        if (P_marker_offd[k1] < jj_begin_row_offd)
                        {
                           P_marker_offd[k1] = jj_counter_offd;
                           P_offd_j[jj_counter_offd] = k1;
                           P_offd_data[jj_counter_offd] = zero;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }

         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
            {
               i1 = S_offd_j[jj];
               if ( CF_marker_offd[i1] >= 0)
               {
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  P_marker_offd[i1] = strong_f_marker;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1 + 1]; kk++)
                  {
                     big_k1 = Sop_j[kk];
                     /* Find local col number */
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] < jj_begin_row)
                        {
                           P_marker[loc_col] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                           P_diag_data[jj_counter] = zero;
                           jj_counter++;
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (P_marker_offd[loc_col] < jj_begin_row_offd)
                        {
                           P_marker_offd[loc_col] = jj_counter_offd;
                           P_offd_j[jj_counter_offd] = loc_col;
                           P_offd_data[jj_counter_offd] = zero;
                           jj_counter_offd++;
                        }
                     }
                  }
               }
            }
         }

         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;

         diagonal = A_diag_data[A_diag_i[i]];

         for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
         {
            /* i1 is a c-point and strongly influences i, accumulate
             * a_(i,i1) into interpolation weight */
            i1 = A_diag_j[jj];
            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               sgn = 1;
               if (A_diag_data[A_diag_i[i1]] < 0)
               {
                  sgn = -1;
               }
               /* Loop over row of A for point i1 and calculate the sum
                * of the connections to c-points that strongly incluence i. */
               for (jj1 = A_diag_i[i1] + 1; jj1 < A_diag_i[i1 + 1]; jj1++)
               {
                  i2 = A_diag_j[jj1];
                  if ((P_marker[i2] >= jj_begin_row ) && (sgn * A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn * A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               }
               if (sum != 0)
               {
                  distribute = A_diag_data[jj] / sum;
                  /* Loop over row of A for point i1 and do the distribution */
                  for (jj1 = A_diag_i[i1] + 1; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn * A_diag_data[jj1]) < 0)
                     {
                        P_diag_data[P_marker[i2]] += distribute * A_diag_data[jj1];
                     }
                  }
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn * A_offd_data[jj1]) < 0)
                        {
                           P_offd_data[P_marker_offd[i2]] += distribute * A_offd_data[jj1];
                        }
                     }
                  }
               }
               else
               {
                  diagonal += A_diag_data[jj];
               }
            }
            /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
             * diagonal */
            else if (CF_marker[i1] != -3)
            {
               if (num_functions == 1 || dof_func[i] == dof_func[i1])
               {
                  diagonal += A_diag_data[jj];
               }
            }
         }
         if (num_procs > 1)
         {
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               i1 = A_offd_j[jj];
               if (P_marker_offd[i1] >= jj_begin_row_offd)
               {
                  P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
               }
               else if (P_marker_offd[i1] == strong_f_marker)
               {
                  sum = zero;
                  for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                  {
                     big_k1 = A_ext_j[jj1];
                     if (big_k1 >= col_1 && big_k1 < col_n)
                     {
                        /* diag */
                        loc_col = (HYPRE_Int)(big_k1 - col_1);
                        if (P_marker[loc_col] >= jj_begin_row )
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else
                     {
                        loc_col = -(HYPRE_Int)big_k1 - 1;
                        if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                  }
                  if (sum != 0)
                  {
                     distribute = A_offd_data[jj] / sum;
                     for (jj1 = A_ext_i[i1]; jj1 < A_ext_i[i1 + 1]; jj1++)
                     {
                        big_k1 = A_ext_j[jj1];
                        if (big_k1 >= col_1 && big_k1 < col_n)
                        {
                           /* diag */
                           loc_col = (HYPRE_Int)(big_k1 - col_1);
                           if (P_marker[loc_col] >= jj_begin_row)
                           {
                              P_diag_data[P_marker[loc_col]] += distribute * A_ext_data[jj1];
                           }
                        }
                        else
                        {
                           loc_col = -(HYPRE_Int)big_k1 - 1;
                           if (P_marker_offd[loc_col] >= jj_begin_row_offd)
                           {
                              P_offd_data[P_marker_offd[loc_col]] += distribute * A_ext_data[jj1];
                           }
                        }
                     }
                  }
                  else
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
               else if (CF_marker_offd[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                  {
                     diagonal += A_offd_data[jj];
                  }
               }
            }
         }
         if (diagonal)
         {
            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }
            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }
      }
      strong_f_marker--;
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     fill structure    %f\n", my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
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

   hypre_CSRMatrixMemoryLocation(P_diag) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(P_offd) = memory_location_P;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   if (P_offd_size)
   {
      hypre_build_interp_colmap(P, full_off_procNodes, tmp_CF_marker_offd, fine_to_coarse_offd);
   }

   hypre_MatvecCommPkgCreate(P);

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }

   *P_ptr = P;

   /* Deallocate memory */
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker,       HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Sop);
      hypre_CSRMatrixDestroy(A_ext);
      hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd,       HYPRE_MEMORY_HOST);
      hypre_TFree(CF_marker_offd,      HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_CF_marker_offd,  HYPRE_MEMORY_HOST);
      if (num_functions > 1)
      {
         hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
      }

      hypre_MatvecCommPkgDestroy(extend_comm_pkg);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGBuildExtInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                              hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
                              HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                              HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                              hypre_ParCSRMatrix  **P_ptr)
{
   hypre_GpuProfilingPushRange("ExtInterp");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildExtInterpDevice(A, CF_marker, S, num_cpts_global, num_functions,
                                                 dof_func,
                                                 debug_flag, trunc_factor, max_elmts, P_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildExtInterpHost(A, CF_marker, S, num_cpts_global, num_functions, dof_func,
                                               debug_flag, trunc_factor, max_elmts, P_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}

/*-----------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBuildExtPIInterp(hypre_ParCSRMatrix   *A,
                                HYPRE_Int            *CF_marker,
                                hypre_ParCSRMatrix   *S,
                                HYPRE_BigInt         *num_cpts_global,
                                HYPRE_Int             num_functions,
                                HYPRE_Int            *dof_func,
                                HYPRE_Int             debug_flag,
                                HYPRE_Real            trunc_factor,
                                HYPRE_Int             max_elmts,
                                hypre_ParCSRMatrix  **P_ptr)
{
   hypre_GpuProfilingPushRange("ExtPIInterp");

   HYPRE_Int ierr = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_BoomerAMGBuildExtPIInterpDevice(A, CF_marker, S, num_cpts_global, num_functions,
                                                   dof_func,
                                                   debug_flag, trunc_factor, max_elmts, P_ptr);
   }
   else
#endif
   {
      ierr = hypre_BoomerAMGBuildExtPIInterpHost(A, CF_marker, S, num_cpts_global, num_functions,
                                                 dof_func,
                                                 debug_flag, trunc_factor, max_elmts, P_ptr);
   }

   hypre_GpuProfilingPopRange();

   return ierr;
}
