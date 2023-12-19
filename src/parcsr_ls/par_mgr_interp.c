/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"

/*--------------------------------------------------------------------------
 * hypre_MGRBuildInterp
 *
 * Build MGR's prolongation matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildInterp(hypre_ParCSRMatrix   *A,
                     hypre_ParCSRMatrix   *A_FF,
                     hypre_ParCSRMatrix   *A_FC,
                     HYPRE_Int            *CF_marker,
                     hypre_ParCSRMatrix   *aux_mat,
                     HYPRE_BigInt         *num_cpts_global,
                     HYPRE_Real            trunc_factor,
                     HYPRE_Int             max_elmts,
                     HYPRE_Int             blk_size,
                     hypre_ParCSRMatrix  **P_ptr,
                     HYPRE_Int             interp_type,
                     HYPRE_Int             num_sweeps_post)
{
   hypre_ParCSRMatrix    *P = NULL;
#if defined (HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* Interpolation for each level */
   if (interp_type < 3)
   {
#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_MGRBuildPDevice(A, CF_marker, num_cpts_global, interp_type, &P);
      }
      else
#endif
      {
         hypre_MGRBuildPHost(A, CF_marker, num_cpts_global, interp_type, &P);

         /* TODO (VPM): Revisit Prolongation post-smoothing */
#if 0
         if (interp_type == 2)
         {
            HYPRE_Real  jac_trunc_threshold = trunc_factor;
            HYPRE_Real  jac_trunc_threshold_minus = 0.5*jac_trunc_threshold;
            HYPRE_Int   i;

            for (i = 0; i < num_sweeps_post; i++)
            {
               hypre_BoomerAMGJacobiInterp(A, &P, S, 1, NULL, CF_marker, 0,
                                           jac_trunc_threshold, jac_trunc_threshold_minus);
            }
            hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
         }
#else
         HYPRE_UNUSED_VAR(num_sweeps_post);
#endif
      }
   }
   else if (interp_type == 4)
   {
#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_NoGPUSupport("interpolation");
      }
      else
#endif
      {
         hypre_MGRBuildInterpApproximateInverse(A, CF_marker, num_cpts_global, &P);
         hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      }
   }
   else if (interp_type == 5)
   {
      hypre_BoomerAMGBuildModExtInterp(A, CF_marker, aux_mat, num_cpts_global,
                                       1, NULL, 0, trunc_factor, max_elmts, &P);
   }
   else if (interp_type == 6)
   {
      hypre_BoomerAMGBuildModExtPIInterp(A, CF_marker, aux_mat, num_cpts_global,
                                         1, NULL, 0, trunc_factor, max_elmts, &P);
   }
   else if (interp_type == 7)
   {
      hypre_BoomerAMGBuildModExtPEInterp(A, CF_marker, aux_mat, num_cpts_global,
                                         1, NULL, 0, trunc_factor, max_elmts, &P);
   }
   else if (interp_type == 12)
   {
      hypre_MGRBuildPBlockJacobi(A, A_FF, A_FC, aux_mat, blk_size, CF_marker,
                                 num_cpts_global, &P);
   }
   else
   {
      /* Classical modified interpolation */
      hypre_BoomerAMGBuildInterp(A, CF_marker, aux_mat, num_cpts_global,
                                 1, NULL, 0, trunc_factor, max_elmts, &P);
   }

   /* set pointer to P */
   *P_ptr = P;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildInterp
 *
 * Setup restriction operator.
 *
 * TODOs (VPM):
 *   1) Change R -> RT (VPM)
 *   2) Add post-smoothing
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildRestrict( hypre_ParCSRMatrix    *A,
                        hypre_ParCSRMatrix    *A_FF,
                        hypre_ParCSRMatrix    *A_FC,
                        HYPRE_Int             *CF_marker,
                        HYPRE_BigInt          *num_cpts_global,
                        HYPRE_Real             trunc_factor,
                        HYPRE_Int              max_elmts,
                        HYPRE_Real             strong_threshold,
                        HYPRE_Real             max_row_sum,
                        HYPRE_Int              blk_size,
                        hypre_ParCSRMatrix   **R_ptr,
                        HYPRE_Int              restrict_type)
{
   hypre_ParCSRMatrix    *R     = NULL;
   hypre_ParCSRMatrix    *AT    = NULL;
   hypre_ParCSRMatrix    *A_FFT = NULL;
   hypre_ParCSRMatrix    *A_FCT = NULL;
   hypre_ParCSRMatrix    *ST    = NULL;
#if defined (HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* Build AT (transpose A) */
   if (restrict_type > 0)
   {
      hypre_ParCSRMatrixTranspose(A, &AT, 1);

      if (A_FF)
      {
         hypre_ParCSRMatrixTranspose(A_FF, &A_FFT, 1);
      }

      if (A_FC)
      {
         hypre_ParCSRMatrixTranspose(A_FC, &A_FCT, 1);
      }
   }

   /* Restriction for each level */
   if (restrict_type == 0)
   {
#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_MGRBuildPDevice(A, CF_marker, num_cpts_global, restrict_type, &R);
      }
      else
#endif
      {
         hypre_MGRBuildP(A, CF_marker, num_cpts_global, restrict_type, 0, &R);
      }
   }
   else if (restrict_type == 1 || restrict_type == 2)
   {
#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_MGRBuildPDevice(AT, CF_marker, num_cpts_global, restrict_type, &R);
      }
      else
#endif
      {
         hypre_MGRBuildP(AT, CF_marker, num_cpts_global, restrict_type, 0, &R);
      }
   }
   else if (restrict_type == 3)
   {
      /* move diagonal to first entry */
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(AT));
      hypre_MGRBuildInterpApproximateInverse(AT, CF_marker, num_cpts_global, &R);
      hypre_BoomerAMGInterpTruncation(R, trunc_factor, max_elmts);
   }
   else if (restrict_type == 12)
   {
      hypre_MGRBuildPBlockJacobi(AT, A_FFT, A_FCT, NULL, blk_size, CF_marker, num_cpts_global,
                                 &R);
   }
   else if (restrict_type == 13) // CPR-like restriction operator
   {
      /* TODO: create a function with this block (VPM) */
      hypre_ParCSRMatrix *blk_A_cf = NULL;
      hypre_ParCSRMatrix *blk_A_cf_transpose = NULL;
      hypre_ParCSRMatrix *Wr_transpose = NULL;
      hypre_ParCSRMatrix *blk_A_ff_inv_transpose = NULL;
      HYPRE_Int *c_marker = NULL;
      HYPRE_Int *f_marker = NULL;
      HYPRE_Int i;
      HYPRE_Int nrows = hypre_ParCSRMatrixNumRows(A);

      HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

      /* TODO: Port this to GPU (VPM) */
      /* create C and F markers to extract A_CF */
      c_marker = CF_marker;
      f_marker = hypre_CTAlloc(HYPRE_Int, nrows, memory_location);
      for (i = 0; i < nrows; i++)
      {
         f_marker[i] = - CF_marker[i];
      }

#if defined (HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_NoGPUSupport("restriction");
      }
      else
#endif
      {
         /* get block A_cf */
         hypre_MGRGetAcfCPR(A, blk_size, c_marker, f_marker, &blk_A_cf);

         /* transpose block A_cf */
         hypre_ParCSRMatrixTranspose(blk_A_cf, &blk_A_cf_transpose, 1);

         /* compute block diagonal A_ff */
         hypre_ParCSRMatrixBlockDiagMatrix(AT, blk_size, -1, CF_marker, 1,
                                           &blk_A_ff_inv_transpose);

         /* compute  Wr = A^{-T} * A_cf^{T}  */
         Wr_transpose = hypre_ParCSRMatMat(blk_A_ff_inv_transpose, blk_A_cf_transpose);

         /* compute restriction operator R = [-Wr  I] (transposed for use with RAP) */
         hypre_MGRBuildPFromWp(AT, Wr_transpose, CF_marker, &R);
      }
      hypre_ParCSRMatrixDestroy(blk_A_cf);
      hypre_ParCSRMatrixDestroy(blk_A_cf_transpose);
      hypre_ParCSRMatrixDestroy(Wr_transpose);
      hypre_ParCSRMatrixDestroy(blk_A_ff_inv_transpose);
      hypre_TFree(f_marker, memory_location);
   }
   else
   {
      /* Build new strength matrix */
      hypre_BoomerAMGCreateS(AT, strong_threshold, max_row_sum, 1, NULL, &ST);

      /* Classical modified interpolation */
      hypre_BoomerAMGBuildInterp(AT, CF_marker, ST, num_cpts_global, 1, NULL, 0,
                                 trunc_factor, max_elmts, &R);
   }

   /* Compute R^T so it can be used in the solve phase */
   hypre_ParCSRMatrixLocalTranspose(R);

   /* Set pointer to R */
   *R_ptr = R;

   /* Free memory */
   if (restrict_type > 0)
   {
      hypre_ParCSRMatrixDestroy(AT);
      hypre_ParCSRMatrixDestroy(A_FFT);
      hypre_ParCSRMatrixDestroy(A_FCT);
   }
   if (restrict_type > 5)
   {
      hypre_ParCSRMatrixDestroy(ST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPFromWp
 *
 * Build prolongation matrix from the Nf x Nc matrix
 *
 * TODO (VPM): Move this function to par_interp.c ?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPFromWp( hypre_ParCSRMatrix    *A,
                       hypre_ParCSRMatrix    *Wp,
                       HYPRE_Int             *CF_marker,
                       hypre_ParCSRMatrix   **P_ptr)
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_MGRBuildPFromWpDevice(A, Wp, CF_marker, P_ptr);
   }
   else
#endif
   {
      hypre_MGRBuildPFromWpHost(A, Wp, CF_marker, P_ptr);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPFromWpHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPFromWpHost( hypre_ParCSRMatrix    *A,
                           hypre_ParCSRMatrix    *Wp,
                           HYPRE_Int             *CF_marker,
                           hypre_ParCSRMatrix   **P_ptr)
{
   MPI_Comm               comm = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation   memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRMatrix    *P;

   hypre_CSRMatrix       *P_diag = NULL;
   hypre_CSRMatrix       *P_offd = NULL;
   hypre_CSRMatrix       *Wp_diag, *Wp_offd;

   HYPRE_Real            *P_diag_data, *Wp_diag_data;
   HYPRE_Int             *P_diag_i, *Wp_diag_i;
   HYPRE_Int             *P_diag_j, *Wp_diag_j;
   HYPRE_Real            *P_offd_data, *Wp_offd_data;
   HYPRE_Int             *P_offd_i, *Wp_offd_i;
   HYPRE_Int             *P_offd_j, *Wp_offd_j;

   HYPRE_Int              P_num_rows, P_diag_size, P_offd_size;
   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              i, jj;
   HYPRE_Int              row_Wp, coarse_counter;
   HYPRE_Real             one  = 1.0;
   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   P_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   Wp_diag = hypre_ParCSRMatrixDiag(Wp);
   Wp_offd = hypre_ParCSRMatrixOffd(Wp);
   Wp_diag_i = hypre_CSRMatrixI(Wp_diag);
   Wp_diag_j = hypre_CSRMatrixJ(Wp_diag);
   Wp_diag_data = hypre_CSRMatrixData(Wp_diag);
   Wp_offd_i = hypre_CSRMatrixI(Wp_offd);
   Wp_offd_j = hypre_CSRMatrixJ(Wp_offd);
   Wp_offd_data = hypre_CSRMatrixData(Wp_offd);

   /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/
   P_diag_size = hypre_CSRMatrixNumNonzeros(Wp_diag) + hypre_CSRMatrixNumCols(Wp_diag);

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  P_num_rows + 1, memory_location_P);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, memory_location_P);
   P_diag_i[P_num_rows] = P_diag_size;

   P_offd_size = hypre_CSRMatrixNumNonzeros(Wp_offd);

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  P_num_rows + 1, memory_location_P);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, memory_location_P);
   P_offd_i[P_num_rows] = P_offd_size;

   /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/
   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   row_Wp = 0;
   coarse_counter = 0;
   for (i = 0; i < P_num_rows; i++)
   {
      /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = coarse_counter;
         P_diag_data[jj_counter] = one;
         coarse_counter++;
         jj_counter++;
      }
      /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
      else
      {
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         for (jj = Wp_diag_i[row_Wp]; jj < Wp_diag_i[row_Wp + 1]; jj++)
         {
            P_diag_j[jj_counter]    = Wp_diag_j[jj];
            P_diag_data[jj_counter] = - Wp_diag_data[jj];
            jj_counter++;
         }

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         if (num_procs > 1)
         {
            for (jj = Wp_offd_i[row_Wp]; jj < Wp_offd_i[row_Wp + 1]; jj++)
            {
               P_offd_j[jj_counter_offd]    = Wp_offd_j[jj];
               P_offd_data[jj_counter_offd] = - Wp_offd_data[jj];
               jj_counter_offd++;
            }
         }
         row_Wp++;
      }
      P_offd_i[i + 1] = jj_counter_offd;
   }
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumCols(Wp),
                                hypre_ParCSRMatrixColStarts(A),
                                hypre_ParCSRMatrixColStarts(Wp),
                                hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(Wp)),
                                P_diag_size,
                                P_offd_size);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;

   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;

   hypre_ParCSRMatrixDeviceColMapOffd(P) = hypre_ParCSRMatrixDeviceColMapOffd(Wp);
   hypre_ParCSRMatrixColMapOffd(P)       = hypre_ParCSRMatrixColMapOffd(Wp);
   //hypre_ParCSRMatrixDeviceColMapOffd(Wp) = NULL;
   //hypre_ParCSRMatrixColMapOffd(Wp)       = NULL;

   hypre_ParCSRMatrixNumNonzeros(P)  = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(P)) +
                                       hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(P));
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);

   hypre_MatvecCommPkgCreate(P);
   *P_ptr = P;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildBlockJacobiWp
 *
 * TODO: Move this to hypre_MGRBuildPBlockJacobi? (VPM)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildBlockJacobiWp( hypre_ParCSRMatrix   *A_FF,
                             hypre_ParCSRMatrix   *A_FC,
                             HYPRE_Int             blk_size,
                             HYPRE_Int            *CF_marker,
                             HYPRE_BigInt         *cpts_starts,
                             hypre_ParCSRMatrix  **Wp_ptr )
{
   HYPRE_UNUSED_VAR(CF_marker);
   HYPRE_UNUSED_VAR(cpts_starts);

   hypre_ParCSRMatrix   *A_FF_inv;
   hypre_ParCSRMatrix   *Wp;

   /* Build A_FF_inv */
   hypre_ParCSRMatrixBlockDiagMatrix(A_FF, blk_size, -1, NULL, 1, &A_FF_inv);

   /* Compute Wp = A_FF_inv * A_FC */
   Wp = hypre_ParCSRMatMat(A_FF_inv, A_FC);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_FF_inv);

   /* Set output pointer */
   *Wp_ptr = Wp;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPBlockJacobi
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPBlockJacobi( hypre_ParCSRMatrix   *A,
                            hypre_ParCSRMatrix   *A_FF,
                            hypre_ParCSRMatrix   *A_FC,
                            hypre_ParCSRMatrix   *Wp,
                            HYPRE_Int             blk_size,
                            HYPRE_Int            *CF_marker,
                            HYPRE_BigInt         *cpts_starts,
                            hypre_ParCSRMatrix  **P_ptr)
{
   hypre_ParCSRMatrix   *Wp_tmp;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (Wp == NULL)
   {
      hypre_MGRBuildBlockJacobiWp(A_FF, A_FC, blk_size, CF_marker, cpts_starts, &Wp_tmp);
      hypre_MGRBuildPFromWp(A, Wp_tmp, CF_marker, P_ptr);

      hypre_ParCSRMatrixDeviceColMapOffd(Wp_tmp) = NULL;
      hypre_ParCSRMatrixColMapOffd(Wp_tmp)       = NULL;

      hypre_ParCSRMatrixDestroy(Wp_tmp);
   }
   else
   {
      hypre_MGRBuildPFromWp(A, Wp, CF_marker, P_ptr);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ExtendWtoPHost
 *
 * TODO: move this to par_interp.c?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ExtendWtoPHost(HYPRE_Int      P_nr_of_rows,
                     HYPRE_Int     *CF_marker,
                     HYPRE_Int     *W_diag_i,
                     HYPRE_Int     *W_diag_j,
                     HYPRE_Complex *W_diag_data,
                     HYPRE_Int     *P_diag_i,
                     HYPRE_Int     *P_diag_j,
                     HYPRE_Complex *P_diag_data,
                     HYPRE_Int     *W_offd_i,
                     HYPRE_Int     *P_offd_i )
{
   HYPRE_Int      jj_counter, jj_counter_offd;
   HYPRE_Int      start_indexing = 0; /* start indexing for P_data at 0 */
   HYPRE_Int     *fine_to_coarse = NULL;
   HYPRE_Int      coarse_counter;

   HYPRE_Int      i, jj;
   HYPRE_Real     one  = 1.0;

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  P_nr_of_rows, HYPRE_MEMORY_HOST);

   for (i = 0; i < P_nr_of_rows; i++) { fine_to_coarse[i] = -1; }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   HYPRE_Int row_counter = 0;
   coarse_counter = 0;
   for (i = 0; i < P_nr_of_rows; i++)
   {
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] > 0)
      {
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }
   }

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   row_counter = 0;
   for (i = 0; i < P_nr_of_rows; i++)
   {
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }
      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/
      else
      {
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         for (jj = W_diag_i[row_counter]; jj < W_diag_i[row_counter + 1]; jj++)
         {
            //P_marker[row_counter] = jj_counter;
            P_diag_j[jj_counter]    = W_diag_j[jj];
            P_diag_data[jj_counter] = W_diag_data[jj];
            jj_counter++;
         }

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_counter_offd += W_offd_i[row_counter + 1] - W_offd_i[row_counter];

         row_counter++;
      }
      /* update off-diagonal row pointer */
      P_offd_i[i + 1] = jj_counter_offd;
   }
   P_diag_i[P_nr_of_rows] = jj_counter;

   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPHost
 *
 * Interpolation for MGR - Adapted from BoomerAMGBuildInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPHost( hypre_ParCSRMatrix   *A,
                     HYPRE_Int            *CF_marker,
                     HYPRE_BigInt         *num_cpts_global,
                     HYPRE_Int             method,
                     hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            num_procs, my_id;
   HYPRE_Int            A_nr_of_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_ParCSRMatrix  *A_FF = NULL, *A_FC = NULL, *P = NULL;
   hypre_CSRMatrix     *W_diag = NULL, *W_offd = NULL;
   HYPRE_Int            P_diag_nnz, nfpoints;
   HYPRE_Int           *P_diag_i = NULL, *P_diag_j = NULL, *P_offd_i = NULL;
   HYPRE_Complex       *P_diag_data = NULL, *diag = NULL, *diag1 = NULL;
   HYPRE_BigInt         nC_global;
   HYPRE_Int            i;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   nfpoints = 0;
   for (i = 0; i < A_nr_of_rows; i++)
   {
      if (CF_marker[i] == -1)
      {
         nfpoints++;
      }
   }

   if (method > 0)
   {
      hypre_ParCSRMatrixGenerateFFFCHost(A, CF_marker, num_cpts_global, NULL, &A_FC, &A_FF);
      diag = hypre_CTAlloc(HYPRE_Complex, nfpoints, memory_location_P);
      if (method == 1)
      {
         // extract diag inverse sqrt
         //        hypre_CSRMatrixExtractDiagonalHost(hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         HYPRE_Complex     scal = 1.0;
         hypre_CSRMatrix  *A_FF_diag = hypre_ParCSRMatrixDiag(A_FF);
         hypre_CSRMatrix  *A_FC_diag = hypre_ParCSRMatrixDiag(A_FC);
         hypre_CSRMatrix  *A_FF_offd = hypre_ParCSRMatrixOffd(A_FF);
         hypre_CSRMatrix  *A_FC_offd = hypre_ParCSRMatrixOffd(A_FC);

         diag1 = hypre_CTAlloc(HYPRE_Complex, nfpoints, memory_location_P);
         hypre_CSRMatrixExtractDiagonalHost(hypre_ParCSRMatrixDiag(A_FF), diag, 0);
         hypre_CSRMatrixComputeRowSumHost(A_FF_diag, NULL, NULL, diag1, 1, 1.0, "set");
         hypre_CSRMatrixComputeRowSumHost(A_FC_diag, NULL, NULL, diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumHost(A_FF_offd, NULL, NULL, diag1, 1, 1.0, "add");
         hypre_CSRMatrixComputeRowSumHost(A_FC_offd, NULL, NULL, diag1, 1, 1.0, "add");

         for (i = 0; i < nfpoints; i++)
         {
            HYPRE_Complex dsum = diag[i] + scal * (diag1[i] - hypre_cabs(diag[i]));
            diag[i] = 1. / dsum;
         }
         hypre_TFree(diag1, memory_location_P);
      }
      else if (method == 2)
      {
         // extract diag inverse
         hypre_CSRMatrixExtractDiagonalHost(hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

      for (i = 0; i < nfpoints; i++)
      {
         diag[i] = -diag[i];
      }

      hypre_Vector *D_FF_inv = hypre_SeqVectorCreate(nfpoints);
      hypre_VectorData(D_FF_inv) = diag;
      hypre_SeqVectorInitialize_v2(D_FF_inv, memory_location_P);
      hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixDiag(A_FC), D_FF_inv, NULL);
      hypre_CSRMatrixDiagScale(hypre_ParCSRMatrixOffd(A_FC), D_FF_inv, NULL);
      hypre_SeqVectorDestroy(D_FF_inv);
      W_diag = hypre_ParCSRMatrixDiag(A_FC);
      W_offd = hypre_ParCSRMatrixOffd(A_FC);
      nC_global = hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      W_diag = hypre_CSRMatrixCreate(nfpoints, A_nr_of_rows - nfpoints, 0);
      W_offd = hypre_CSRMatrixCreate(nfpoints, 0, 0);
      hypre_CSRMatrixInitialize_v2(W_diag, 0, memory_location_P);
      hypre_CSRMatrixInitialize_v2(W_offd, 0, memory_location_P);

      if (my_id == (num_procs - 1))
      {
         nC_global = num_cpts_global[1];
      }
      hypre_MPI_Bcast(&nC_global, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   /* Construct P from matrix product W_diag */
   P_diag_nnz  = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,     A_nr_of_rows + 1, memory_location_P);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,     P_diag_nnz,     memory_location_P);
   P_diag_data = hypre_CTAlloc(HYPRE_Complex, P_diag_nnz,     memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,     A_nr_of_rows + 1, memory_location_P);

   /* Extend W data to P data */
   hypre_ExtendWtoPHost( A_nr_of_rows,
                         CF_marker,
                         hypre_CSRMatrixI(W_diag),
                         hypre_CSRMatrixJ(W_diag),
                         hypre_CSRMatrixData(W_diag),
                         P_diag_i,
                         P_diag_j,
                         P_diag_data,
                         hypre_CSRMatrixI(W_offd),
                         P_offd_i );

   // finalize P
   P = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                nC_global,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(W_offd) );

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = memory_location_P;

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
   hypre_CSRMatrixJ(W_offd)    = NULL;
   hypre_CSRMatrixData(W_offd) = NULL;

   if (method > 0)
   {
      hypre_ParCSRMatrixColMapOffd(P)    = hypre_ParCSRMatrixColMapOffd(A_FC);
      hypre_ParCSRMatrixColMapOffd(P)    = hypre_ParCSRMatrixColMapOffd(A_FC);
      hypre_ParCSRMatrixColMapOffd(A_FC) = NULL;
      hypre_ParCSRMatrixColMapOffd(A_FC) = NULL;
      hypre_ParCSRMatrixNumNonzeros(P)   = hypre_ParCSRMatrixNumNonzeros(A_FC) +
                                           hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      hypre_ParCSRMatrixNumNonzeros(P) = nC_global;
   }
   hypre_ParCSRMatrixDNumNonzeros(P) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(P);
   hypre_MatvecCommPkgCreate(P);

   /* Set output pointer */
   *P_ptr = P;

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_FF);
   hypre_ParCSRMatrixDestroy(A_FC);
   if (method <= 0)
   {
      hypre_CSRMatrixDestroy(W_diag);
      hypre_CSRMatrixDestroy(W_offd);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildP
 *
 * Interpolation for MGR - Adapted from BoomerAMGBuildInterp
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildP( hypre_ParCSRMatrix   *A,
                 HYPRE_Int            *CF_marker,
                 HYPRE_BigInt         *num_cpts_global,
                 HYPRE_Int             method,
                 HYPRE_Int             debug_flag,
                 hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Real      *a_diag;

   hypre_ParCSRMatrix    *P;
   HYPRE_BigInt    *col_map_offd_P;
   HYPRE_Int       *tmp_map_offd = NULL;

   HYPRE_Int       *CF_marker_offd = NULL;

   hypre_CSRMatrix *P_diag;
   hypre_CSRMatrix *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int       *P_diag_i;
   HYPRE_Int       *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int       *P_offd_i;
   HYPRE_Int       *P_offd_j;

   HYPRE_Int        P_diag_size, P_offd_size;

   HYPRE_Int       *P_marker, *P_marker_offd;

   HYPRE_Int        jj_counter, jj_counter_offd;
   HYPRE_Int       *jj_count, *jj_count_offd;
   //   HYPRE_Int              jj_begin_row,jj_begin_row_offd;
   //   HYPRE_Int              jj_end_row,jj_end_row_offd;

   HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int        n_fine = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int       *fine_to_coarse;
   //HYPRE_BigInt    *fine_to_coarse_offd;
   HYPRE_Int       *coarse_counter;
   HYPRE_Int        coarse_shift;
   HYPRE_BigInt     total_global_cpts;
   //HYPRE_BigInt     my_first_cpt;
   HYPRE_Int        num_cols_P_offd;

   HYPRE_Int        i, i1;
   HYPRE_Int        j, jl, jj;
   HYPRE_Int        start;

   HYPRE_Real       one  = 1.0;

   HYPRE_Int        my_id;
   HYPRE_Int        num_procs;
   HYPRE_Int        num_threads;
   HYPRE_Int        num_sends;
   HYPRE_Int        index;
   HYPRE_Int        ns, ne, size, rest;

   HYPRE_Int       *int_buf_data;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
   * Get the CF_marker data for the off-processor columns
   *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
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
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
         else
         {
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if ((CF_marker[i1] >= 0) && (method > 0))
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if ((CF_marker_offd[i1] >= 0) && (method > 0))
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, memory_location_P);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, memory_location_P);

   P_diag_i[n_fine] = jj_counter;

   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, memory_location_P);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, memory_location_P);

   /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
   *  Send and receive fine_to_coarse info.
   *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   //fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
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

   /*   index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            big_buf_data[index++]
               = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]+ my_first_cpt;
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data,
                                       fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   */
   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   //for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
   a_diag = hypre_CTAlloc(HYPRE_Real,  n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] < 0)
      {
         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
         {
            i1 = A_diag_j[jj];
            if ( i == i1 ) /* diagonal of A only */
            {
               a_diag[i] = 1.0 / A_diag_data[jj];
            }
         }
      }
   }

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
      {
         P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      }
      else
      {
         P_marker_offd = NULL;
      }

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
         *  If i is a c-point, interpolation is the identity.
         *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
         *  If i is an F-point, build interpolation.
         *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if ((CF_marker[i1] >= 0) && (method > 0))
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /*
                  if(method == 0)
                  {
                    P_diag_data[jj_counter] = 0.0;
                  }
                  */
                  if (method == 1)
                  {
                     P_diag_data[jj_counter] = - A_diag_data[jj];
                  }
                  else if (method == 2)
                  {
                     P_diag_data[jj_counter] = - A_diag_data[jj] * a_diag[i];
                  }
                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                  * If neighbor i1 is a C-point, set column number in P_offd_j
                  * and initialize interpolation weight to zero.
                  *-----------------------------------------------------------*/

                  if ((CF_marker_offd[i1] >= 0) && (method > 0))
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     /*
                     if(method == 0)
                     {
                       P_offd_data[jj_counter_offd] = 0.0;
                     }
                     */
                     if (method == 1)
                     {
                        P_offd_data[jj_counter_offd] = - A_offd_data[jj];
                     }
                     else if (method == 2)
                     {
                        P_offd_data[jj_counter_offd] = - A_offd_data[jj] * a_diag[i];
                     }
                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(a_diag, HYPRE_MEMORY_HOST);
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

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }
      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   // hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildPDRS
 *
 * Interpolation for MGR - Dynamic Row Sum method
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildPDRS( hypre_ParCSRMatrix   *A,
                    HYPRE_Int            *CF_marker,
                    HYPRE_BigInt         *num_cpts_global,
                    HYPRE_Int             debug_flag,
                    hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix         *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real              *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix         *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real              *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int               *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int                num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Real              *a_diag;

   hypre_ParCSRMatrix      *P;
   HYPRE_BigInt            *col_map_offd_P;
   HYPRE_Int               *tmp_map_offd = NULL;
   HYPRE_Int               *CF_marker_offd = NULL;

   hypre_CSRMatrix         *P_diag;
   hypre_CSRMatrix         *P_offd;

   HYPRE_Real              *P_diag_data;
   HYPRE_Int               *P_diag_i, *P_diag_j;
   HYPRE_Real              *P_offd_data;
   HYPRE_Int               *P_offd_i, *P_offd_j;

   HYPRE_Int                P_diag_size, P_offd_size;
   HYPRE_Int               *P_marker, *P_marker_offd;
   HYPRE_Int                jj_counter, jj_counter_offd;
   HYPRE_Int               *jj_count, *jj_count_offd;

   HYPRE_Int                start_indexing = 0; /* start indexing for P_data at 0 */
   HYPRE_Int                n_fine  = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int               *fine_to_coarse;
   HYPRE_Int               *coarse_counter;
   HYPRE_Int                coarse_shift;
   HYPRE_BigInt             total_global_cpts;
   HYPRE_Int                num_cols_P_offd;

   HYPRE_Int                i, i1;
   HYPRE_Int                j, jl, jj;
   HYPRE_Int                start;
   HYPRE_Real               one  = 1.0;
   HYPRE_Int                my_id, num_procs;
   HYPRE_Int                num_threads;
   HYPRE_Int                num_sends;
   HYPRE_Int                index;
   HYPRE_Int                ns, ne, size, rest;

   HYPRE_Int               *int_buf_data;
   HYPRE_Real               wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++] =
            CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
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
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
         else
         {
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
         /*--------------------------------------------------------------------
          *  Set up the indexes for the DRS method
          *--------------------------------------------------------------------*/

      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
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

   /*index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   */
   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif

   //for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   a_diag = hypre_CTAlloc(HYPRE_Real,  n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
      {
         i1 = A_diag_j[jj];
         if ( i == i1 ) /* diagonal of A only */
         {
            a_diag[i] = 1.0 / A_diag_data[jj];
         }
      }
   }

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
      {
         P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      }
      else
      {
         P_marker_offd = NULL;
      }

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = - A_diag_data[jj] * a_diag[i];

                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]    = i1;
                     P_offd_data[jj_counter_offd] = - A_offd_data[jj] * a_diag[i];

                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i + 1] = jj_counter_offd;
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
      hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(a_diag, HYPRE_MEMORY_HOST);
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

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }
      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < P_offd_size; i++)
      {
         P_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      }
      hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3)
      {
         CF_marker[i] = -1;
      }
   }
   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildInterpApproximateInverse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildInterpApproximateInverse(hypre_ParCSRMatrix   *A,
                                       HYPRE_Int            *CF_marker,
                                       HYPRE_BigInt         *num_cpts_global,
                                       hypre_ParCSRMatrix  **P_ptr)
{
   HYPRE_Int             *C_marker;
   HYPRE_Int             *F_marker;
   hypre_ParCSRMatrix    *A_ff;
   hypre_ParCSRMatrix    *A_fc;
   hypre_ParCSRMatrix    *A_ff_inv;
   hypre_ParCSRMatrix    *W;
   MPI_Comm               comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRMatrix    *P;
   HYPRE_BigInt          *col_map_offd_P = NULL;
   HYPRE_Real            *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int              P_diag_nnz;
   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_BigInt           total_global_cpts;
   HYPRE_Int              num_cols_P_offd;

   HYPRE_Int              i;

   HYPRE_Real             m_one = -1.0;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   C_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   F_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);

   // create C and F markers
   for (i = 0; i < n_fine; i++)
   {
      C_marker[i] = (CF_marker[i] == 1) ? 1 : -1;
      F_marker[i] = (CF_marker[i] == 1) ? -1 : 1;
   }

   // Get A_FF
   hypre_MGRGetSubBlock(A, F_marker, F_marker, 0, &A_ff);
   //  hypre_ParCSRMatrixPrintIJ(A_ff, 1, 1, "A_ff");
   // Get A_FC
   hypre_MGRGetSubBlock(A, F_marker, C_marker, 0, &A_fc);

   hypre_MGRApproximateInverse(A_ff, &A_ff_inv);
   //  hypre_ParCSRMatrixPrintIJ(A_ff_inv, 1, 1, "A_ff_inv");
   //  hypre_ParCSRMatrixPrintIJ(A_fc, 1, 1, "A_fc");
   W = hypre_ParMatmul(A_ff_inv, A_fc);
   hypre_ParCSRMatrixScale(W, m_one);
   //  hypre_ParCSRMatrixPrintIJ(W, 1, 1, "Wp");

   hypre_CSRMatrix *W_diag = hypre_ParCSRMatrixDiag(W);
   hypre_CSRMatrix *W_offd = hypre_ParCSRMatrixOffd(W);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_nnz  = hypre_CSRMatrixNumNonzeros(W_diag) + hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, memory_location_P);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_nnz, memory_location_P);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_nnz, memory_location_P);
   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine + 1, memory_location_P);

   /* Extend W data to P data */
   hypre_ExtendWtoPHost( n_fine,
                         CF_marker,
                         hypre_CSRMatrixI(W_diag),
                         hypre_CSRMatrixJ(W_diag),
                         hypre_CSRMatrixData(W_diag),
                         P_diag_i,
                         P_diag_j,
                         P_diag_data,
                         hypre_CSRMatrixI(W_offd),
                         P_offd_i );
   // final P
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                hypre_CSRMatrixNumNonzeros(W_offd) );

   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixDiag(P)) = memory_location_P;
   hypre_CSRMatrixMemoryLocation(hypre_ParCSRMatrixOffd(P)) = memory_location_P;

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(P))    = hypre_CSRMatrixJ(W_offd);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(P)) = hypre_CSRMatrixData(W_offd);
   hypre_CSRMatrixJ(W_offd)    = NULL;
   hypre_CSRMatrixData(W_offd) = NULL;

   num_cols_P_offd = hypre_CSRMatrixNumCols(W_offd);
   HYPRE_BigInt *col_map_offd_tmp = hypre_ParCSRMatrixColMapOffd(W);
   if (hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(P)))
   {
      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_P_offd; i++)
      {
         col_map_offd_P[i] = col_map_offd_tmp[i];
      }
   }

   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(P)) = num_cols_P_offd;
   }
   hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   hypre_TFree(C_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(F_marker, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixDestroy(A_ff);
   hypre_ParCSRMatrixDestroy(A_fc);
   hypre_ParCSRMatrixDestroy(A_ff_inv);
   hypre_ParCSRMatrixDestroy(W);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRGetAcfCPR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRGetAcfCPR(hypre_ParCSRMatrix     *A,
                   HYPRE_Int               blk_size,
                   HYPRE_Int              *c_marker,
                   HYPRE_Int              *f_marker,
                   hypre_ParCSRMatrix    **A_CF_ptr)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int i, j, jj, jj1;
   HYPRE_Int jj_counter, cpts_cnt;
   hypre_ParCSRMatrix *A_CF = NULL;
   hypre_CSRMatrix *A_CF_diag = NULL;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex *A_diag_data = hypre_CSRMatrixData(A_diag);

   HYPRE_Int total_fpts, n_fpoints;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int nnz_diag_new = 0;
   HYPRE_Int num_procs, my_id;
   hypre_IntArray *wrap_cf = NULL;
   hypre_IntArray *coarse_dof_func_ptr = NULL;
   HYPRE_BigInt num_row_cpts_global[2], num_col_fpts_global[2];
   HYPRE_BigInt total_global_row_cpts, total_global_col_fpts;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   // Count total F-points
   // Also setup F to C column map
   total_fpts = 0;
   HYPRE_Int *f_to_c_col_map = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_rows; i++)
   {
      //      if (c_marker[i] == 1)
      //      {
      //         total_cpts++;
      //      }
      if (f_marker[i] == 1)
      {
         f_to_c_col_map[i] = total_fpts;
         total_fpts++;
      }
   }
   n_fpoints = blk_size;
   /* get the number of coarse rows */
   wrap_cf = hypre_IntArrayCreate(num_rows);
   hypre_IntArrayMemoryLocation(wrap_cf) = HYPRE_MEMORY_HOST;
   hypre_IntArrayData(wrap_cf) = c_marker;
   hypre_BoomerAMGCoarseParms(comm, num_rows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_row_cpts_global);
   hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_row_cpts_global[0], num_row_cpts_global[1]);

   if (my_id == (num_procs - 1)) { total_global_row_cpts = num_row_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_row_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get the number of coarse rows */
   hypre_IntArrayData(wrap_cf) = f_marker;
   hypre_BoomerAMGCoarseParms(comm, num_rows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_col_fpts_global);
   hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;
   hypre_IntArrayData(wrap_cf) = NULL;
   hypre_IntArrayDestroy(wrap_cf);

   //hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_col_fpts_global[0], num_col_fpts_global[1]);

   if (my_id == (num_procs - 1)) { total_global_col_fpts = num_col_fpts_global[1]; }
   hypre_MPI_Bcast(&total_global_col_fpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   // First pass: count the nnz of A_CF
   jj_counter = 0;
   cpts_cnt = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (c_marker[i] == 1)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            jj = A_diag_j[j];
            if (f_marker[jj] == 1)
            {
               jj1 = f_to_c_col_map[jj];
               if (jj1 >= cpts_cnt * n_fpoints && jj1 < (cpts_cnt + 1)*n_fpoints)
               {
                  jj_counter++;
               }
            }
         }
         cpts_cnt++;
      }
   }
   nnz_diag_new = jj_counter;

   HYPRE_Int     *A_CF_diag_i    = hypre_CTAlloc(HYPRE_Int, cpts_cnt + 1, memory_location);
   HYPRE_Int     *A_CF_diag_j    = hypre_CTAlloc(HYPRE_Int, nnz_diag_new, memory_location);
   HYPRE_Complex *A_CF_diag_data = hypre_CTAlloc(HYPRE_Complex, nnz_diag_new, memory_location);
   A_CF_diag_i[cpts_cnt] = nnz_diag_new;

   jj_counter = 0;
   cpts_cnt = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (c_marker[i] == 1)
      {
         A_CF_diag_i[cpts_cnt] = jj_counter;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            jj = A_diag_j[j];
            if (f_marker[jj] == 1)
            {
               jj1 = f_to_c_col_map[jj];
               if (jj1 >= cpts_cnt * n_fpoints && jj1 < (cpts_cnt + 1)*n_fpoints)
               {
                  A_CF_diag_j[jj_counter] = jj1;
                  A_CF_diag_data[jj_counter] = A_diag_data[j];
                  jj_counter++;
               }
            }
         }
         cpts_cnt++;
      }
   }

   /* Create A_CF matrix */
   A_CF = hypre_ParCSRMatrixCreate(comm,
                                   total_global_row_cpts,
                                   total_global_col_fpts,
                                   num_row_cpts_global,
                                   num_col_fpts_global,
                                   0,
                                   nnz_diag_new,
                                   0);

   A_CF_diag = hypre_ParCSRMatrixDiag(A_CF);
   hypre_CSRMatrixData(A_CF_diag) = A_CF_diag_data;
   hypre_CSRMatrixI(A_CF_diag) = A_CF_diag_i;
   hypre_CSRMatrixJ(A_CF_diag) = A_CF_diag_j;

   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A_CF)) = NULL;
   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A_CF)) = NULL;
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A_CF)) = NULL;

   *A_CF_ptr = A_CF;

   hypre_TFree(f_to_c_col_map, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRTruncateAcfCPRDevice
 *
 * TODO (VPM): Port truncation to GPUs
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRTruncateAcfCPRDevice(hypre_ParCSRMatrix  *A_CF,
                              hypre_ParCSRMatrix **A_CF_new_ptr)
{
   hypre_ParCSRMatrix *hA_CF;
   hypre_ParCSRMatrix *A_CF_new;

   hypre_GpuProfilingPushRange("MGRTruncateAcfCPR");

   /* Clone matrix to host, truncate, and migrate result to device */
   hA_CF = hypre_ParCSRMatrixClone_v2(A_CF, 1, HYPRE_MEMORY_HOST);
   hypre_MGRTruncateAcfCPR(hA_CF, &A_CF_new);
   hypre_ParCSRMatrixMigrate(A_CF_new, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRMatrixDestroy(hA_CF);

   /* Set output pointer */
   *A_CF_new_ptr = A_CF_new;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRTruncateAcfCPR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRTruncateAcfCPR(hypre_ParCSRMatrix  *A_CF,
                        hypre_ParCSRMatrix **A_CF_new_ptr)
{
   /* Input matrix info */
   MPI_Comm             comm           = hypre_ParCSRMatrixComm(A_CF);
   HYPRE_BigInt         num_rows       = hypre_ParCSRMatrixGlobalNumRows(A_CF);
   HYPRE_BigInt         num_cols       = hypre_ParCSRMatrixGlobalNumCols(A_CF);

   hypre_CSRMatrix     *A_CF_diag      = hypre_ParCSRMatrixDiag(A_CF);
   HYPRE_Int           *A_CF_diag_i    = hypre_CSRMatrixI(A_CF_diag);
   HYPRE_Int           *A_CF_diag_j    = hypre_CSRMatrixJ(A_CF_diag);
   HYPRE_Complex       *A_CF_diag_data = hypre_CSRMatrixData(A_CF_diag);
   HYPRE_Int            num_rows_local = hypre_CSRMatrixNumRows(A_CF_diag);

   /* Output matrix info */
   hypre_ParCSRMatrix  *A_CF_new;
   hypre_CSRMatrix     *A_CF_diag_new;
   HYPRE_Int           *A_CF_diag_i_new;
   HYPRE_Int           *A_CF_diag_j_new;
   HYPRE_Complex       *A_CF_diag_data_new;
   HYPRE_Int            nnz_diag_new;

   /* Local variables */
   HYPRE_Int            i, j, jj;
   HYPRE_Int            jj_counter;
   HYPRE_Int            blk_size = num_cols / num_rows;

   /* Sanity check */
   hypre_assert(hypre_ParCSRMatrixMemoryLocation(A_CF) == HYPRE_MEMORY_HOST);

   /* First pass: count the nnz of truncated (new) A_CF */
   jj_counter = 0;
   for (i = 0; i < num_rows_local; i++)
   {
      for (j = A_CF_diag_i[i]; j < A_CF_diag_i[i + 1]; j++)
      {
         jj = A_CF_diag_j[j];
         if (jj >= i * blk_size && jj < (i + 1) * blk_size)
         {
            jj_counter++;
         }
      }
   }
   nnz_diag_new = jj_counter;

   /* Create truncated matrix */
   A_CF_new = hypre_ParCSRMatrixCreate(comm,
                                       num_rows,
                                       num_cols,
                                       hypre_ParCSRMatrixRowStarts(A_CF),
                                       hypre_ParCSRMatrixColStarts(A_CF),
                                       0,
                                       nnz_diag_new,
                                       0);

   hypre_ParCSRMatrixInitialize_v2(A_CF_new, HYPRE_MEMORY_HOST);
   A_CF_diag_new      = hypre_ParCSRMatrixDiag(A_CF_new);
   A_CF_diag_i_new    = hypre_CSRMatrixI(A_CF_diag_new);
   A_CF_diag_j_new    = hypre_CSRMatrixJ(A_CF_diag_new);
   A_CF_diag_data_new = hypre_CSRMatrixData(A_CF_diag_new);

   /* Second pass: fill entries of the truncated (new) A_CF */
   jj_counter = 0;
   for (i = 0; i < num_rows_local; i++)
   {
      A_CF_diag_i_new[i] = jj_counter;
      for (j = A_CF_diag_i[i]; j < A_CF_diag_i[i + 1]; j++)
      {
         jj = A_CF_diag_j[j];
         if (jj >= i * blk_size && jj < (i + 1) * blk_size)
         {
            A_CF_diag_j_new[jj_counter] = jj;
            A_CF_diag_data_new[jj_counter] = A_CF_diag_data[j];
            jj_counter++;
         }
      }
   }
   A_CF_diag_i_new[num_rows_local] = nnz_diag_new;

   /* Set output pointer */
   *A_CF_new_ptr = A_CF_new;

   return hypre_error_flag;
}
