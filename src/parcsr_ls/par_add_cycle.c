/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGAdditiveCycle( void              *amg_vdata)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   hypre_ParCSRMatrix    *Lambda;
   hypre_ParCSRMatrix    *Atilde;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Xtilde, *Rtilde;
   hypre_IntArray    **CF_marker_array;
   HYPRE_Int          *CF_marker;

   HYPRE_Int       num_levels;
   HYPRE_Int       addlvl, add_end;
   HYPRE_Int       additive;
   HYPRE_Int       mult_additive;
   HYPRE_Int       simple;
   HYPRE_Int       add_last_lvl;
   HYPRE_Int       i, j, num_rows;
   HYPRE_Int       n_global;
   HYPRE_Int       rlx_order;

   /* Local variables  */
   HYPRE_Int       Solve_err_flag = 0;
   HYPRE_Int       level;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       rlx_down;
   HYPRE_Int       rlx_up;
   HYPRE_Int       rlx_coarse;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int      *num_grid_sweeps;
   hypre_Vector  **l1_norms;
   HYPRE_Real      alpha, beta;
   HYPRE_Real     *u_data;
   HYPRE_Real     *v_data;
   hypre_Vector   *l1_norms_lvl;
   HYPRE_Real     *D_inv;
   HYPRE_Real     *x_global;
   HYPRE_Real     *r_global;
   HYPRE_Real     *relax_weight;
   HYPRE_Real     *omega;

#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   F_array           = hypre_ParAMGDataFArray(amg_data);
   U_array           = hypre_ParAMGDataUArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   additive          = hypre_ParAMGDataAdditive(amg_data);
   mult_additive     = hypre_ParAMGDataMultAdditive(amg_data);
   simple            = hypre_ParAMGDataSimple(amg_data);
   add_last_lvl      = hypre_ParAMGDataAddLastLvl(amg_data);
   grid_relax_type   = hypre_ParAMGDataGridRelaxType(amg_data);
   Lambda            = hypre_ParAMGDataLambda(amg_data);
   Atilde            = hypre_ParAMGDataAtilde(amg_data);
   Xtilde            = hypre_ParAMGDataXtilde(amg_data);
   Rtilde            = hypre_ParAMGDataRtilde(amg_data);
   l1_norms          = hypre_ParAMGDataL1Norms(amg_data);
   D_inv             = hypre_ParAMGDataDinv(amg_data);
   relax_weight      = hypre_ParAMGDataRelaxWeight(amg_data);
   omega             = hypre_ParAMGDataOmega(amg_data);
   rlx_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   num_grid_sweeps   = hypre_ParAMGDataNumGridSweeps(amg_data);

   /* Initialize */

   addlvl = hypre_max(additive, mult_additive);
   addlvl = hypre_max(addlvl, simple);
   if (add_last_lvl == -1 ) { add_end = num_levels - 1; }
   else { add_end = add_last_lvl; }
   Solve_err_flag = 0;

   /*---------------------------------------------------------------------
    * Main loop of cycling --- multiplicative version --- V-cycle
    *--------------------------------------------------------------------*/

   /* down cycle */
   rlx_down = grid_relax_type[1];
   rlx_up = grid_relax_type[2];
   rlx_coarse = grid_relax_type[3];
   for (level = 0; level < num_levels - 1; level++)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);

      fine_grid = level;
      coarse_grid = level + 1;

      u_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      v_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
      l1_norms_lvl = l1_norms[level];

      hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

      if (level < addlvl || level > add_end) /* multiplicative version */
      {
         /* smoothing step */

         if (rlx_down == 0)
         {
            HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  u_data[i] = relax_weight[level] * v_data[i] / A_data[A_i[i]];
               }
            }
         }

         else if (rlx_down != 18)
         {
            /*hypre_BoomerAMGRelax(A_array[fine_grid],F_array[fine_grid],NULL,rlx_down,0,*/
            CF_marker = hypre_IntArrayData(CF_marker_array[fine_grid]);
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                      CF_marker, rlx_down, rlx_order, 1,
                                      relax_weight[fine_grid], omega[fine_grid],
                                      l1_norms[level] ? hypre_VectorData(l1_norms[level]) : NULL,
                                      U_array[fine_grid], Vtemp, Ztemp);
               hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
            }
         }
         else
         {
            num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
#ifdef HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  u_data[i] += v_data[i] / hypre_VectorData(l1_norms_lvl)[i];
               }
            }
         }

         alpha = -1.0;
         beta = 1.0;
         hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                                  beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;
         hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);
      }
      else /* additive version */
      {
         hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
         if (level == 0) /* compute residual */
         {
            hypre_ParVectorCopy(Vtemp, Rtilde);
            hypre_ParVectorCopy(U_array[fine_grid], Xtilde);
         }
         alpha = 1.0;
         beta = 0.0;
         hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);
      }

      HYPRE_ANNOTATE_MGLEVEL_END(level);
   }

   /* additive smoothing and solve coarse grid */
   HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);
   if (addlvl < num_levels)
   {
      if (simple > -1)
      {
         x_global = hypre_VectorData(hypre_ParVectorLocalVector(Xtilde));
         r_global = hypre_VectorData(hypre_ParVectorLocalVector(Rtilde));
         n_global = hypre_VectorSize(hypre_ParVectorLocalVector(Xtilde));
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n_global; i++)
         {
            x_global[i] += D_inv[i] * r_global[i];
         }
      }
      else
      {
         if (num_grid_sweeps[1] > 1)
         {
            n_global = hypre_VectorSize(hypre_ParVectorLocalVector(Rtilde));
            hypre_ParVector *Tmptilde = hypre_CTAlloc(hypre_ParVector,  1, HYPRE_MEMORY_HOST);
            hypre_Vector *Tmptilde_local = hypre_SeqVectorCreate(n_global);
            hypre_SeqVectorInitialize(Tmptilde_local);
            hypre_ParVectorLocalVector(Tmptilde) = Tmptilde_local;
            hypre_ParVectorOwnsData(Tmptilde) = 1;
            hypre_ParCSRMatrixMatvec(1.0, Lambda, Rtilde, 0.0, Tmptilde);
            hypre_ParVectorScale(2.0, Rtilde);
            hypre_ParCSRMatrixMatvec(-1.0, Atilde, Tmptilde, 1.0, Rtilde);
            hypre_ParVectorDestroy(Tmptilde);
         }
         hypre_ParCSRMatrixMatvec(1.0, Lambda, Rtilde, 1.0, Xtilde);
      }
      if (addlvl == 0) { hypre_ParVectorCopy(Xtilde, U_array[0]); }
   }
   if (add_end < num_levels - 1)
   {
      fine_grid = num_levels - 1;
      for (j = 0; j < num_grid_sweeps[3]; j++)
         if (rlx_coarse == 18)
            hypre_ParCSRRelax(A_array[fine_grid], F_array[fine_grid],
                              1, 1,
                              l1_norms[fine_grid] ? hypre_VectorData(l1_norms[fine_grid]) : NULL,
                              1.0, 1.0, 0.0, 0.0, 0, 0.0,
                              U_array[fine_grid], Vtemp, Ztemp);
         else
            hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                   NULL, rlx_coarse, 0, 0,
                                   relax_weight[fine_grid], omega[fine_grid],
                                   l1_norms[fine_grid] ? hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                   U_array[fine_grid], Vtemp, Ztemp);
   }
   HYPRE_ANNOTATE_MGLEVEL_END(num_levels - 1);

   /* up cycle */
   for (level = num_levels - 1; level > 0; level--)
   {
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);

      fine_grid = level - 1;
      coarse_grid = level;

      if (level <= addlvl || level > add_end + 1) /* multiplicative version */
      {
         alpha = 1.0;
         beta = 1.0;
         hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  beta, U_array[fine_grid]);
         if (rlx_up != 18)
         {
            /*hypre_BoomerAMGRelax(A_array[fine_grid],F_array[fine_grid],NULL,rlx_up,0,*/
            CF_marker = hypre_IntArrayData(CF_marker_array[fine_grid]);
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                      CF_marker,
                                      rlx_up, rlx_order, 2,
                                      relax_weight[fine_grid], omega[fine_grid],
                                      l1_norms[fine_grid] ? hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                      U_array[fine_grid], Vtemp, Ztemp);
            }
         }
         else if (rlx_order)
         {
            CF_marker = hypre_IntArrayData(CF_marker_array[fine_grid]);
            HYPRE_Int loc_relax_points[2];
            loc_relax_points[0] = -1;
            loc_relax_points[1] = 1;
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               for (i = 0; i < 2; i++)
               {
                  hypre_ParCSRRelax_L1_Jacobi(A_array[fine_grid], F_array[fine_grid],
                                              CF_marker,
                                              loc_relax_points[i],
                                              1.0,
                                              l1_norms[fine_grid] ? hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                              U_array[fine_grid], Vtemp);
               }
            }
         }
         else
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               hypre_ParCSRRelax(A_array[fine_grid], F_array[fine_grid],
                                 1, 1,
                                 l1_norms[fine_grid] ? hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                 1.0, 1.0, 0.0, 0.0, 0, 0.0,
                                 U_array[fine_grid], Vtemp, Ztemp);
            }
      }
      else /* additive version */
      {
         alpha = 1.0;
         beta = 1.0;
         hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  beta, U_array[fine_grid]);
      }

      HYPRE_ANNOTATE_MGLEVEL_END(level);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}


HYPRE_Int hypre_CreateLambda(void *amg_vdata)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   MPI_Comm comm;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   hypre_ParCSRMatrix *A_tmp;
   hypre_ParCSRMatrix *Lambda;
   hypre_CSRMatrix *L_diag;
   hypre_CSRMatrix *L_offd;
   hypre_ParCSRMatrix *Atilde;
   hypre_CSRMatrix *Atilde_diag = NULL;
   hypre_CSRMatrix *Atilde_offd = NULL;
   HYPRE_Real    *Atilde_diag_data = NULL;
   HYPRE_Real    *Atilde_offd_data = NULL;
   hypre_CSRMatrix *A_tmp_diag;
   hypre_CSRMatrix *A_tmp_offd;
   hypre_ParVector *Xtilde;
   hypre_ParVector *Rtilde;
   hypre_Vector *Xtilde_local;
   hypre_Vector *Rtilde_local;
   hypre_ParCSRCommPkg *comm_pkg;
   hypre_ParCSRCommPkg *L_comm_pkg = NULL;
   hypre_ParCSRCommHandle *comm_handle;
   HYPRE_Real    *L_diag_data = NULL;
   HYPRE_Real    *L_offd_data = NULL;
   HYPRE_Real    *buf_data = NULL;
   HYPRE_Real    *tmp_data;
   HYPRE_Real    *x_data;
   HYPRE_Real    *r_data;
   hypre_Vector  *l1_norms;
   HYPRE_Real    *A_tmp_diag_data = NULL;
   HYPRE_Real    *A_tmp_offd_data = NULL;
   HYPRE_Real    *D_data = NULL;
   HYPRE_Real    *D_data_offd = NULL;
   HYPRE_Int *L_diag_i;
   HYPRE_Int *L_diag_j = NULL;
   HYPRE_Int *L_offd_i;
   HYPRE_Int *L_offd_j = NULL;
   HYPRE_Int *Atilde_diag_i = NULL;
   HYPRE_Int *Atilde_diag_j = NULL;
   HYPRE_Int *Atilde_offd_i = NULL;
   HYPRE_Int *Atilde_offd_j = NULL;
   HYPRE_Int *A_tmp_diag_i;
   HYPRE_Int *A_tmp_offd_i;
   HYPRE_Int *A_tmp_diag_j;
   HYPRE_Int *A_tmp_offd_j;
   HYPRE_Int *L_recv_ptr = NULL;
   HYPRE_Int *L_send_ptr = NULL;
   HYPRE_Int *L_recv_procs = NULL;
   HYPRE_Int *L_send_procs = NULL;
   HYPRE_Int *L_send_map_elmts = NULL;
   HYPRE_Int *recv_procs;
   HYPRE_Int *send_procs;
   HYPRE_Int *send_map_elmts = NULL;
   HYPRE_Int *send_map_starts = NULL;
   HYPRE_Int *recv_vec_starts;
   HYPRE_Int *all_send_procs = NULL;
   HYPRE_Int *all_recv_procs = NULL;
   HYPRE_Int *remap = NULL;
   HYPRE_Int *level_start;

   HYPRE_Int       addlvl;
   HYPRE_Int       additive;
   HYPRE_Int       mult_additive;
   HYPRE_Int       num_levels;
   HYPRE_Int       num_add_lvls;
   HYPRE_Int       num_procs;
   HYPRE_Int       num_sends, num_recvs;
   HYPRE_Int       num_sends_L = 0;
   HYPRE_Int       num_recvs_L = 0;
   HYPRE_Int       send_data_L = 0;
   HYPRE_Int       num_rows_L = 0;
   HYPRE_Int       num_rows_tmp = 0;
   HYPRE_Int       num_cols_offd_L = 0;
   HYPRE_Int       num_cols_offd = 0;
   HYPRE_Int       level, i, j, k;
   HYPRE_Int       this_proc, cnt, cnt_diag, cnt_offd;
   HYPRE_Int       A_cnt_diag, A_cnt_offd;
   HYPRE_Int       cnt_recv, cnt_send, cnt_row, row_start;
   HYPRE_Int       start_diag, start_offd, indx, cnt_map;
   HYPRE_Int       start, j_indx, index, cnt_level;
   HYPRE_Int       max_sends, max_recvs;
   HYPRE_Int       ns;

   /* Local variables  */
   HYPRE_Int       Solve_err_flag = 0;
   HYPRE_Int       num_nonzeros_diag;
   HYPRE_Int       num_nonzeros_offd;

   hypre_Vector  **l1_norms_ptr = NULL;
   /*HYPRE_Real   *relax_weight = NULL;
   HYPRE_Int      relax_type; */
   HYPRE_Int       add_rlx;
   HYPRE_Int       add_last_lvl, add_end;
   HYPRE_Real  add_rlx_wt;

   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   F_array           = hypre_ParAMGDataFArray(amg_data);
   U_array           = hypre_ParAMGDataUArray(amg_data);
   additive          = hypre_ParAMGDataAdditive(amg_data);
   mult_additive     = hypre_ParAMGDataMultAdditive(amg_data);
   add_last_lvl      = hypre_ParAMGDataAddLastLvl(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   /*relax_weight      = hypre_ParAMGDataRelaxWeight(amg_data);
   relax_type        = hypre_ParAMGDataGridRelaxType(amg_data)[1];*/
   comm              = hypre_ParCSRMatrixComm(A_array[0]);
   add_rlx           = hypre_ParAMGDataAddRelaxType(amg_data);
   add_rlx_wt        = hypre_ParAMGDataAddRelaxWt(amg_data);
   ns                = hypre_ParAMGDataNumGridSweeps(amg_data)[1];

   hypre_MPI_Comm_size(comm, &num_procs);

   l1_norms_ptr      = hypre_ParAMGDataL1Norms(amg_data);

   addlvl = hypre_max(additive, mult_additive);
   if (add_last_lvl != -1) { add_end = add_last_lvl + 1; }
   else { add_end = num_levels; }
   num_add_lvls = add_end + 1 - addlvl;

   level_start = hypre_CTAlloc(HYPRE_Int,  num_add_lvls + 1, HYPRE_MEMORY_HOST);
   send_data_L = 0;
   num_rows_L  = 0;
   num_cols_offd_L = 0;
   num_nonzeros_diag = 0;
   num_nonzeros_offd = 0;
   level_start[0] = 0;
   cnt = 1;
   max_sends = 0;
   max_recvs = 0;
   for (i = addlvl; i < add_end; i++)
   {
      A_tmp = A_array[i];
      A_tmp_diag = hypre_ParCSRMatrixDiag(A_tmp);
      A_tmp_offd = hypre_ParCSRMatrixOffd(A_tmp);
      A_tmp_diag_i = hypre_CSRMatrixI(A_tmp_diag);
      A_tmp_offd_i = hypre_CSRMatrixI(A_tmp_offd);
      num_rows_tmp = hypre_CSRMatrixNumRows(A_tmp_diag);
      num_cols_offd = hypre_CSRMatrixNumCols(A_tmp_offd);
      num_rows_L += num_rows_tmp;
      level_start[cnt] = level_start[cnt - 1] + num_rows_tmp;
      cnt++;
      num_cols_offd_L += num_cols_offd;
      num_nonzeros_diag += A_tmp_diag_i[num_rows_tmp];
      num_nonzeros_offd += A_tmp_offd_i[num_rows_tmp];
      comm_pkg = hypre_ParCSRMatrixCommPkg(A_tmp);
      if (comm_pkg)
      {
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         max_sends += num_sends;
         if (num_sends)
         {
            send_data_L += hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         }
         max_recvs += hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      }
   }
   if (max_sends >= num_procs || max_recvs >= num_procs)
   {
      max_sends = num_procs;
      max_recvs = num_procs;
   }
   if (max_sends) { all_send_procs = hypre_CTAlloc(HYPRE_Int,  max_sends, HYPRE_MEMORY_HOST); }
   if (max_recvs) { all_recv_procs = hypre_CTAlloc(HYPRE_Int,  max_recvs, HYPRE_MEMORY_HOST); }

   cnt_send = 0;
   cnt_recv = 0;
   if (max_sends || max_recvs)
   {
      if (max_sends < num_procs && max_recvs < num_procs)
      {
         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               for (j = 0; j < num_sends; j++)
               {
                  all_send_procs[cnt_send++] = send_procs[j];
               }
               for (j = 0; j < num_recvs; j++)
               {
                  all_recv_procs[cnt_recv++] = recv_procs[j];
               }
            }
         }
         if (max_sends)
         {
            hypre_qsort0(all_send_procs, 0, max_sends - 1);
            num_sends_L = 1;
            this_proc = all_send_procs[0];
            for (i = 1; i < max_sends; i++)
            {
               if (all_send_procs[i] > this_proc)
               {
                  this_proc = all_send_procs[i];
                  all_send_procs[num_sends_L++] = this_proc;
               }
            }
            L_send_procs = hypre_CTAlloc(HYPRE_Int,  num_sends_L, HYPRE_MEMORY_HOST);
            for (j = 0; j < num_sends_L; j++)
            {
               L_send_procs[j] = all_send_procs[j];
            }
            hypre_TFree(all_send_procs, HYPRE_MEMORY_HOST);
         }
         if (max_recvs)
         {
            hypre_qsort0(all_recv_procs, 0, max_recvs - 1);
            num_recvs_L = 1;
            this_proc = all_recv_procs[0];
            for (i = 1; i < max_recvs; i++)
            {
               if (all_recv_procs[i] > this_proc)
               {
                  this_proc = all_recv_procs[i];
                  all_recv_procs[num_recvs_L++] = this_proc;
               }
            }
            L_recv_procs = hypre_CTAlloc(HYPRE_Int,  num_recvs_L, HYPRE_MEMORY_HOST);
            for (j = 0; j < num_recvs_L; j++)
            {
               L_recv_procs[j] = all_recv_procs[j];
            }
            hypre_TFree(all_recv_procs, HYPRE_MEMORY_HOST);
         }

         L_recv_ptr = hypre_CTAlloc(HYPRE_Int,  num_recvs_L + 1, HYPRE_MEMORY_HOST);
         L_send_ptr = hypre_CTAlloc(HYPRE_Int,  num_sends_L + 1, HYPRE_MEMORY_HOST);

         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
               recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
            }
            else
            {
               num_sends = 0;
               num_recvs = 0;
            }
            for (k = 0; k < num_sends; k++)
            {
               this_proc = hypre_BinarySearch(L_send_procs, send_procs[k], num_sends_L);
               L_send_ptr[this_proc + 1] += send_map_starts[k + 1] - send_map_starts[k];
            }
            for (k = 0; k < num_recvs; k++)
            {
               this_proc = hypre_BinarySearch(L_recv_procs, recv_procs[k], num_recvs_L);
               L_recv_ptr[this_proc + 1] += recv_vec_starts[k + 1] - recv_vec_starts[k];
            }
         }

         L_recv_ptr[0] = 0;
         for (i = 1; i < num_recvs_L; i++)
         {
            L_recv_ptr[i + 1] += L_recv_ptr[i];
         }

         L_send_ptr[0] = 0;
         for (i = 1; i < num_sends_L; i++)
         {
            L_send_ptr[i + 1] += L_send_ptr[i];
         }
      }
      else
      {
         num_recvs_L = 0;
         num_sends_L = 0;
         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
               recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
               for (j = 0; j < num_sends; j++)
               {
                  this_proc = send_procs[j];
                  if (all_send_procs[this_proc] == 0)
                  {
                     num_sends_L++;
                  }
                  all_send_procs[this_proc] += send_map_starts[j + 1] - send_map_starts[j];
               }
               for (j = 0; j < num_recvs; j++)
               {
                  this_proc = recv_procs[j];
                  if (all_recv_procs[this_proc] == 0)
                  {
                     num_recvs_L++;
                  }
                  all_recv_procs[this_proc] += recv_vec_starts[j + 1] - recv_vec_starts[j];
               }
            }
         }
         if (max_sends)
         {
            L_send_procs = hypre_CTAlloc(HYPRE_Int,  num_sends_L, HYPRE_MEMORY_HOST);
            L_send_ptr = hypre_CTAlloc(HYPRE_Int,  num_sends_L + 1, HYPRE_MEMORY_HOST);
            num_sends_L = 0;
            for (j = 0; j < num_procs; j++)
            {
               this_proc = all_send_procs[j];
               if (this_proc)
               {
                  L_send_procs[num_sends_L++] = j;
                  L_send_ptr[num_sends_L] = this_proc + L_send_ptr[num_sends_L - 1];
               }
            }
         }
         if (max_recvs)
         {
            L_recv_procs = hypre_CTAlloc(HYPRE_Int,  num_recvs_L, HYPRE_MEMORY_HOST);
            L_recv_ptr = hypre_CTAlloc(HYPRE_Int,  num_recvs_L + 1, HYPRE_MEMORY_HOST);
            num_recvs_L = 0;
            for (j = 0; j < num_procs; j++)
            {
               this_proc = all_recv_procs[j];
               if (this_proc)
               {
                  L_recv_procs[num_recvs_L++] = j;
                  L_recv_ptr[num_recvs_L] = this_proc + L_recv_ptr[num_recvs_L - 1];
               }
            }
         }
      }
   }
   if (max_sends) { hypre_TFree(all_send_procs, HYPRE_MEMORY_HOST); }
   if (max_recvs) { hypre_TFree(all_recv_procs, HYPRE_MEMORY_HOST); }

   L_diag = hypre_CSRMatrixCreate(num_rows_L, num_rows_L, num_nonzeros_diag);
   L_offd = hypre_CSRMatrixCreate(num_rows_L, num_cols_offd_L, num_nonzeros_offd);
   hypre_CSRMatrixInitialize(L_diag);
   hypre_CSRMatrixInitialize(L_offd);

   if (num_nonzeros_diag)
   {
      L_diag_data = hypre_CSRMatrixData(L_diag);
      L_diag_j = hypre_CSRMatrixJ(L_diag);
   }
   L_diag_i = hypre_CSRMatrixI(L_diag);
   if (num_nonzeros_offd)
   {
      L_offd_data = hypre_CSRMatrixData(L_offd);
      L_offd_j = hypre_CSRMatrixJ(L_offd);
   }
   L_offd_i = hypre_CSRMatrixI(L_offd);

   if (ns > 1)
   {
      Atilde_diag = hypre_CSRMatrixCreate(num_rows_L, num_rows_L, num_nonzeros_diag);
      Atilde_offd = hypre_CSRMatrixCreate(num_rows_L, num_cols_offd_L, num_nonzeros_offd);
      hypre_CSRMatrixInitialize(Atilde_diag);
      hypre_CSRMatrixInitialize(Atilde_offd);
      if (num_nonzeros_diag)
      {
         Atilde_diag_data = hypre_CSRMatrixData(Atilde_diag);
         Atilde_diag_j = hypre_CSRMatrixJ(Atilde_diag);
      }
      Atilde_diag_i = hypre_CSRMatrixI(Atilde_diag);
      if (num_nonzeros_offd)
      {
         Atilde_offd_data = hypre_CSRMatrixData(Atilde_offd);
         Atilde_offd_j = hypre_CSRMatrixJ(Atilde_offd);
      }
      Atilde_offd_i = hypre_CSRMatrixI(Atilde_offd);
   }

   if (num_rows_L) { D_data = hypre_CTAlloc(HYPRE_Real, num_rows_L, HYPRE_MEMORY_HOST); }
   if (send_data_L)
   {
      L_send_map_elmts = hypre_CTAlloc(HYPRE_Int,  send_data_L, HYPRE_MEMORY_HOST);
      buf_data = hypre_CTAlloc(HYPRE_Real, send_data_L, HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_L)
   {
      D_data_offd = hypre_CTAlloc(HYPRE_Real, num_cols_offd_L, HYPRE_MEMORY_HOST);
      /*L_col_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd_L);*/
      remap = hypre_CTAlloc(HYPRE_Int,  num_cols_offd_L, HYPRE_MEMORY_HOST);
   }

   Rtilde = hypre_CTAlloc(hypre_ParVector,  1, HYPRE_MEMORY_HOST);
   Rtilde_local = hypre_SeqVectorCreate(num_rows_L);
   hypre_SeqVectorInitialize(Rtilde_local);
   hypre_ParVectorLocalVector(Rtilde) = Rtilde_local;
   hypre_ParVectorOwnsData(Rtilde) = 1;

   Xtilde = hypre_CTAlloc(hypre_ParVector,  1, HYPRE_MEMORY_HOST);
   Xtilde_local = hypre_SeqVectorCreate(num_rows_L);
   hypre_SeqVectorInitialize(Xtilde_local);
   hypre_ParVectorLocalVector(Xtilde) = Xtilde_local;
   hypre_ParVectorOwnsData(Xtilde) = 1;

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(Xtilde));
   r_data = hypre_VectorData(hypre_ParVectorLocalVector(Rtilde));

   cnt = 0;
   cnt_level = 0;
   cnt_diag = 0;
   cnt_offd = 0;
   cnt_row = 1;
   L_diag_i[0] = 0;
   L_offd_i[0] = 0;
   if (ns > 1)
   {
      A_cnt_diag = 0;
      A_cnt_offd = 0;
      Atilde_diag_i[0] = 0;
      Atilde_offd_i[0] = 0;
   }
   for (level = addlvl; level < add_end; level++)
   {
      row_start = level_start[cnt_level];
      if (level != 0)
      {
         tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
         if (tmp_data)
         {
            hypre_TFree(tmp_data, hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(F_array[level])));
         }
         hypre_VectorData(hypre_ParVectorLocalVector(F_array[level])) = &r_data[row_start];
         hypre_VectorOwnsData(hypre_ParVectorLocalVector(F_array[level])) = 0;

         tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
         if (tmp_data)
         {
            hypre_TFree(tmp_data, hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(U_array[level])));
         }
         hypre_VectorData(hypre_ParVectorLocalVector(U_array[level])) = &x_data[row_start];
         hypre_VectorOwnsData(hypre_ParVectorLocalVector(U_array[level])) = 0;
      }
      cnt_level++;

      start_diag = L_diag_i[cnt_row - 1];
      start_offd = L_offd_i[cnt_row - 1];
      A_tmp = A_array[level];
      A_tmp_diag = hypre_ParCSRMatrixDiag(A_tmp);
      A_tmp_offd = hypre_ParCSRMatrixOffd(A_tmp);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A_tmp);
      A_tmp_diag_i = hypre_CSRMatrixI(A_tmp_diag);
      A_tmp_offd_i = hypre_CSRMatrixI(A_tmp_offd);
      A_tmp_diag_j = hypre_CSRMatrixJ(A_tmp_diag);
      A_tmp_offd_j = hypre_CSRMatrixJ(A_tmp_offd);
      A_tmp_diag_data = hypre_CSRMatrixData(A_tmp_diag);
      A_tmp_offd_data = hypre_CSRMatrixData(A_tmp_offd);
      num_rows_tmp = hypre_CSRMatrixNumRows(A_tmp_diag);
      if (comm_pkg)
      {
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
         send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
         recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
         send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
         send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
         recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      }
      else
      {
         num_sends = 0;
         num_recvs = 0;
      }

      /* Compute new combined communication package */
      for (i = 0; i < num_sends; i++)
      {
         this_proc = hypre_BinarySearch(L_send_procs, send_procs[i], num_sends_L);
         indx = L_send_ptr[this_proc];
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            L_send_map_elmts[indx++] = row_start + send_map_elmts[j];
         }
         L_send_ptr[this_proc] = indx;
      }

      cnt_map = 0;
      for (i = 0; i < num_recvs; i++)
      {
         this_proc = hypre_BinarySearch(L_recv_procs, recv_procs[i], num_recvs_L);
         indx = L_recv_ptr[this_proc];
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
         {
            remap[cnt_map++] = indx++;
         }
         L_recv_ptr[this_proc] = indx;
      }

      /* Compute Lambda */
      if (add_rlx == 0)
      {
         /*HYPRE_Real rlx_wt = relax_weight[level];*/
#ifdef HYPRE_USING_OPENMP
         #pragma omp for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_data[i] = add_rlx_wt / A_tmp_diag_data[A_tmp_diag_i[i]];
            L_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
            L_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
         }
         if (ns > 1)
            for (i = 0; i < num_rows_tmp; i++)
            {
               Atilde_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
               Atilde_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
            }
      }
      else
      {
         l1_norms = l1_norms_ptr[level];
#ifdef HYPRE_USING_OPENMP
         #pragma omp for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_data[i] = 1.0 / hypre_VectorData(l1_norms)[i];
            L_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
            L_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
         }
         if (ns > 1)
         {
            for (i = 0; i < num_rows_tmp; i++)
            {
               Atilde_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
               Atilde_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
            }
         }
      }

      if (num_procs > 1)
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = send_map_starts[i];
            for (j = start; j < send_map_starts[i + 1]; j++)
            {
               buf_data[index++] = D_data[send_map_elmts[j]];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg,
                                                    buf_data, D_data_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      for (i = 0; i < num_rows_tmp; i++)
      {
         j_indx = A_tmp_diag_i[i];
         if (ns > 1)
         {
            Atilde_diag_data[A_cnt_diag] = A_tmp_diag_data[j_indx];
            Atilde_diag_j[A_cnt_diag++] = i + row_start;
         }
         L_diag_data[cnt_diag] = (2.0 - A_tmp_diag_data[j_indx] * D_data[i]) * D_data[i];
         L_diag_j[cnt_diag++] = i + row_start;
         for (j = A_tmp_diag_i[i] + 1; j < A_tmp_diag_i[i + 1]; j++)
         {
            j_indx = A_tmp_diag_j[j];
            L_diag_data[cnt_diag] = (- A_tmp_diag_data[j] * D_data[j_indx]) * D_data[i];
            L_diag_j[cnt_diag++] = j_indx + row_start;
         }
         for (j = A_tmp_offd_i[i]; j < A_tmp_offd_i[i + 1]; j++)
         {
            j_indx = A_tmp_offd_j[j];
            L_offd_data[cnt_offd] = (- A_tmp_offd_data[j] * D_data_offd[j_indx]) * D_data[i];
            L_offd_j[cnt_offd++] = remap[j_indx];
         }
         if (ns > 1)
         {
            for (j = A_tmp_diag_i[i] + 1; j < A_tmp_diag_i[i + 1]; j++)
            {
               j_indx = A_tmp_diag_j[j];
               Atilde_diag_data[A_cnt_diag] = A_tmp_diag_data[j];
               Atilde_diag_j[A_cnt_diag++] = j_indx + row_start;
            }
            for (j = A_tmp_offd_i[i]; j < A_tmp_offd_i[i + 1]; j++)
            {
               j_indx = A_tmp_offd_j[j];
               Atilde_offd_data[A_cnt_offd] = A_tmp_offd_data[j];
               Atilde_offd_j[A_cnt_offd++] = remap[j_indx];
            }
         }
      }
      cnt_row += num_rows_tmp;
   }

   if (L_send_ptr)
   {
      for (i = num_sends_L - 1; i > 0; i--)
      {
         L_send_ptr[i] = L_send_ptr[i - 1];
      }
      L_send_ptr[0] = 0;
   }
   else
   {
      L_send_ptr = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);
   }

   if (L_recv_ptr)
   {
      for (i = num_recvs_L - 1; i > 0; i--)
      {
         L_recv_ptr[i] = L_recv_ptr[i - 1];
      }
      L_recv_ptr[0] = 0;
   }
   else
   {
      L_recv_ptr = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);
   }

   /* Create and fill communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_L, L_recv_procs, L_recv_ptr,
                                    num_sends_L, L_send_procs, L_send_ptr,
                                    L_send_map_elmts,
                                    &L_comm_pkg);

   Lambda = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixDiag(Lambda) = L_diag;
   hypre_ParCSRMatrixOffd(Lambda) = L_offd;
   hypre_ParCSRMatrixCommPkg(Lambda) = L_comm_pkg;
   hypre_ParCSRMatrixComm(Lambda) = comm;
   hypre_ParCSRMatrixOwnsData(Lambda) = 1;

   if (ns > 1)
   {
      Atilde = hypre_CTAlloc(hypre_ParCSRMatrix,  1, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDiag(Atilde) = Atilde_diag;
      hypre_ParCSRMatrixOffd(Atilde) = Atilde_offd;
      hypre_ParCSRMatrixCommPkg(Atilde) = L_comm_pkg;
      hypre_ParCSRMatrixComm(Atilde) = comm;
      hypre_ParCSRMatrixOwnsData(Atilde) = 1;
      hypre_ParAMGDataAtilde(amg_data) = Atilde;
   }

   hypre_ParAMGDataLambda(amg_data) = Lambda;
   hypre_ParAMGDataRtilde(amg_data) = Rtilde;
   hypre_ParAMGDataXtilde(amg_data) = Xtilde;

   hypre_TFree(D_data_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(D_data, HYPRE_MEMORY_HOST);
   if (num_procs > 1) { hypre_TFree(buf_data, HYPRE_MEMORY_HOST); }
   hypre_TFree(remap, HYPRE_MEMORY_HOST);
   hypre_TFree(buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(level_start, HYPRE_MEMORY_HOST);

   return Solve_err_flag;
}

HYPRE_Int hypre_CreateDinv(void *amg_vdata)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   hypre_ParCSRMatrix *A_tmp;
   hypre_CSRMatrix *A_tmp_diag;
   hypre_ParVector *Xtilde;
   hypre_ParVector *Rtilde;
   hypre_Vector *Xtilde_local;
   hypre_Vector *Rtilde_local;
   HYPRE_Real    *x_data;
   HYPRE_Real    *r_data;
   HYPRE_Real    *tmp_data;
   HYPRE_Real    *D_inv = NULL;
   /*HYPRE_Real    *relax_weight = NULL;
   HYPRE_Real     relax_type;*/

   HYPRE_Int       addlvl;
   HYPRE_Int       num_levels;
   HYPRE_Int       num_rows_L;
   HYPRE_Int       num_rows_tmp;
   HYPRE_Int       level, i;
   HYPRE_Int       add_rlx;
   HYPRE_Real      add_rlx_wt;
   HYPRE_Int       add_last_lvl, add_end;

   /* Local variables  */
   HYPRE_Int       Solve_err_flag = 0;

   hypre_Vector  **l1_norms_ptr = NULL;
   hypre_Vector   *l1_norms;
   HYPRE_Int l1_start;

   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   F_array           = hypre_ParAMGDataFArray(amg_data);
   U_array           = hypre_ParAMGDataUArray(amg_data);
   addlvl            = hypre_ParAMGDataSimple(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   add_rlx_wt        = hypre_ParAMGDataAddRelaxWt(amg_data);
   add_rlx           = hypre_ParAMGDataAddRelaxType(amg_data);
   add_last_lvl      = hypre_ParAMGDataAddLastLvl(amg_data);
   /*relax_weight      = hypre_ParAMGDataRelaxWeight(amg_data);
   relax_type        = hypre_ParAMGDataGridRelaxType(amg_data)[1];*/

   l1_norms_ptr      = hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */
   if (add_last_lvl == -1 ) { add_end = num_levels; }
   else { add_end = add_last_lvl; }

   num_rows_L  = 0;
   for (i = addlvl; i < add_end; i++)
   {
      A_tmp = A_array[i];
      A_tmp_diag = hypre_ParCSRMatrixDiag(A_tmp);
      num_rows_tmp = hypre_CSRMatrixNumRows(A_tmp_diag);
      num_rows_L += num_rows_tmp;
   }

   Rtilde = hypre_CTAlloc(hypre_ParVector,  1, HYPRE_MEMORY_HOST);
   Rtilde_local = hypre_SeqVectorCreate(num_rows_L);
   hypre_SeqVectorInitialize(Rtilde_local);
   hypre_ParVectorLocalVector(Rtilde) = Rtilde_local;
   hypre_ParVectorOwnsData(Rtilde) = 1;

   Xtilde = hypre_CTAlloc(hypre_ParVector,  1, HYPRE_MEMORY_HOST);
   Xtilde_local = hypre_SeqVectorCreate(num_rows_L);
   hypre_SeqVectorInitialize(Xtilde_local);
   hypre_ParVectorLocalVector(Xtilde) = Xtilde_local;
   hypre_ParVectorOwnsData(Xtilde) = 1;

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(Xtilde));
   r_data = hypre_VectorData(hypre_ParVectorLocalVector(Rtilde));
   D_inv = hypre_CTAlloc(HYPRE_Real,  num_rows_L, HYPRE_MEMORY_HOST);

   l1_start = 0;
   for (level = addlvl; level < add_end; level++)
   {
      if (level != 0)
      {
         tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
         if (tmp_data)
         {
            hypre_TFree(tmp_data, hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(F_array[level])));
         }
         hypre_VectorData(hypre_ParVectorLocalVector(F_array[level])) = &r_data[l1_start];
         hypre_VectorOwnsData(hypre_ParVectorLocalVector(F_array[level])) = 0;

         tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
         if (tmp_data)
         {
            hypre_TFree(tmp_data, hypre_VectorMemoryLocation(hypre_ParVectorLocalVector(U_array[level])));
         }
         hypre_VectorData(hypre_ParVectorLocalVector(U_array[level])) = &x_data[l1_start];
         hypre_VectorOwnsData(hypre_ParVectorLocalVector(U_array[level])) = 0;
      }

      A_tmp = A_array[level];
      A_tmp_diag = hypre_ParCSRMatrixDiag(A_tmp);
      num_rows_tmp = hypre_CSRMatrixNumRows(A_tmp_diag);

      if (add_rlx == 0)
      {
         /*HYPRE_Real rlx_wt = relax_weight[level];*/
         HYPRE_Int *A_tmp_diag_i = hypre_CSRMatrixI(A_tmp_diag);
         HYPRE_Real *A_tmp_diag_data = hypre_CSRMatrixData(A_tmp_diag);
#ifdef HYPRE_USING_OPENMP
         #pragma omp for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_inv[l1_start + i] = add_rlx_wt / A_tmp_diag_data[A_tmp_diag_i[i]];
         }
      }
      else
      {
         l1_norms = l1_norms_ptr[level];
#ifdef HYPRE_USING_OPENMP
         #pragma omp for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_inv[l1_start + i] = 1.0 / hypre_VectorData(l1_norms)[i];
         }
      }
      l1_start += num_rows_tmp;
   }

   hypre_ParAMGDataDinv(amg_data) = D_inv;
   hypre_ParAMGDataRtilde(amg_data) = Rtilde;
   hypre_ParAMGDataXtilde(amg_data) = Xtilde;

   return Solve_err_flag;
}
