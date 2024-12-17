/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_mgr.h"

/*--------------------------------------------------------------------------
 * hypre_MGRNonGalerkinTruncate
 *
 * Applies filtering in-place to the input matrix "A" based on the maximum
 * number of nonzero entries per row. This algorithm is tailored to the needs
 * of the Non-Galerkin approach in MGR.
 *
 *  - max_elmts == 0: no filtering
 *  - max_elmts == 1 and blk_dim == 1: keep diagonal entries
 *  - max_elmts == 1 and  blk_dim > 1: keep block diagonal entries
 *  - max_elmts > 1 and blk_dim == 1: keep diagonal entries and
 *                                    (max_elmts - 1) largest ones per row
 *  - max_elmts > blk_dim and blk_dim > 1: keep block diagonal entries and
 *                                         (max_elmts - blk_dim) largest ones
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRNonGalerkinTruncate(hypre_ParCSRMatrix *A,
                             HYPRE_Int           ordering,
                             HYPRE_Int           blk_dim,
                             HYPRE_Int           max_elmts)
{
   HYPRE_MemoryLocation   memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_Int              nrows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   hypre_CSRMatrix *A_diag    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex   *A_diag_a  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j  = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int        ncol_diag = hypre_CSRMatrixNumCols(A_diag);

   hypre_CSRMatrix *A_offd    = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex   *A_offd_a  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i  = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j  = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int        i, i1, jj;

   HYPRE_Int       *A_diag_i_new, *A_diag_j_new;
   HYPRE_Complex   *A_diag_a_new;
   HYPRE_Int        num_nonzeros_diag_new = 0;

   HYPRE_Int       *A_offd_i_new, *A_offd_j_new;
   HYPRE_Complex   *A_offd_a_new;
   HYPRE_Int        num_nonzeros_offd_new = 0;
   HYPRE_Int        num_nonzeros_max = (blk_dim + max_elmts) * nrows;
   HYPRE_Int        num_nonzeros_offd_max = max_elmts * nrows;

   HYPRE_Int        max_num_nonzeros;
   HYPRE_Int       *aux_j = NULL;
   HYPRE_Real      *aux_data = NULL;
   HYPRE_Int        row_start, row_stop, cnt;
   HYPRE_Int        col_idx;
   HYPRE_Real       col_value;

   /* Return if max_elmts is zero, i.e., no truncation */
   if (max_elmts == 0)
   {
      return hypre_error_flag;
   }

   /* Allocate new memory */
   if (ordering == 0)
   {
#if defined (HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
      {
         hypre_ParCSRMatrixMigrate(A, HYPRE_MEMORY_HOST);
      }
#endif

      A_diag_i_new = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
      A_diag_j_new = hypre_CTAlloc(HYPRE_Int, num_nonzeros_max, HYPRE_MEMORY_HOST);
      A_diag_a_new = hypre_CTAlloc(HYPRE_Complex, num_nonzeros_max, HYPRE_MEMORY_HOST);
      A_offd_i_new = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
      A_offd_j_new = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd_max, HYPRE_MEMORY_HOST);
      A_offd_a_new = hypre_CTAlloc(HYPRE_Complex, num_nonzeros_offd_max, HYPRE_MEMORY_HOST);

      if (max_elmts > 0)
      {
         max_num_nonzeros = 0;
         for (i = 0; i < nrows; i++)
         {
            max_num_nonzeros = hypre_max(max_num_nonzeros,
                                         (A_diag_i[i + 1] - A_diag_i[i]) +
                                         (A_offd_i[i + 1] - A_offd_i[i]));
         }
         aux_j = hypre_CTAlloc(HYPRE_Int, max_num_nonzeros, HYPRE_MEMORY_HOST);
         aux_data = hypre_CTAlloc(HYPRE_Real, max_num_nonzeros, HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < nrows; i++)
      {
         row_start = i - (i % blk_dim);
         row_stop  = row_start + blk_dim - 1;

         /* Copy (block) diagonal data to new arrays */
         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
         {
            i1 = A_diag_j[jj];
            if (i1 >= row_start && i1 <= row_stop)
            {
               A_diag_j_new[num_nonzeros_diag_new] = i1;
               A_diag_a_new[num_nonzeros_diag_new] = A_diag_a[jj];
               ++num_nonzeros_diag_new;
            }
         }

         /* Add other connections? */
         if (max_elmts > 0)
         {
            cnt = 0;
            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_offd_j[jj] + ncol_diag;
               aux_data[cnt] = A_offd_a[jj];
               cnt++;
            }

            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_diag_j[jj];
               aux_data[cnt] = A_diag_a[jj];
               cnt++;
            }
            hypre_qsort2_abs(aux_j, aux_data, 0, cnt - 1);

            for (jj = 0; jj < hypre_min(max_elmts, cnt); jj++)
            {
               col_idx   = aux_j[jj];
               col_value = aux_data[jj];
               if (col_idx < ncol_diag && (col_idx < row_start || col_idx > row_stop))
               {
                  A_diag_j_new[num_nonzeros_diag_new] = col_idx;
                  A_diag_a_new[num_nonzeros_diag_new] = col_value;
                  ++num_nonzeros_diag_new;
               }
               else if (col_idx >= ncol_diag)
               {
                  A_offd_j_new[num_nonzeros_offd_new] = col_idx - ncol_diag;
                  A_offd_a_new[num_nonzeros_offd_new] = col_value;
                  ++num_nonzeros_offd_new;
               }
            }
         }

         A_diag_i_new[i + 1] = num_nonzeros_diag_new;
         A_offd_i_new[i + 1] = num_nonzeros_offd_new;
      }

      hypre_TFree(aux_j, HYPRE_MEMORY_HOST);
      hypre_TFree(aux_data, HYPRE_MEMORY_HOST);

      /* Update input matrix */
      hypre_TFree(A_diag_i, HYPRE_MEMORY_HOST);
      hypre_TFree(A_diag_j, HYPRE_MEMORY_HOST);
      hypre_TFree(A_diag_a, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixI(A_diag) = A_diag_i_new;
      hypre_CSRMatrixJ(A_diag) = A_diag_j_new;
      hypre_CSRMatrixData(A_diag) = A_diag_a_new;
      hypre_CSRMatrixNumNonzeros(A_diag) = num_nonzeros_diag_new;

      hypre_TFree(A_offd_i, HYPRE_MEMORY_HOST);
      hypre_TFree(A_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(A_offd_a, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixI(A_offd) = A_offd_i_new;
      hypre_CSRMatrixJ(A_offd) = A_offd_j_new;
      hypre_CSRMatrixData(A_offd) = A_offd_a_new;
      hypre_CSRMatrixNumNonzeros(A_offd) = num_nonzeros_offd_new;

#if defined (HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
      {
         hypre_ParCSRMatrixMigrate(A, memory_location);
      }
#endif
   }
   else
   {
      /* Keep only the diagonal portion of A
         TODO (VPM): consider other combinations of max_elmts and blk_dim */
      hypre_CSRMatrixNumCols(A_offd) = 0;
      hypre_CSRMatrixNumNonzeros(A_offd) = 0;
      hypre_CSRMatrixNumRownnz(A_offd) = 0;
      hypre_TFree(hypre_CSRMatrixRownnz(A_offd), memory_location);
      hypre_TFree(hypre_CSRMatrixI(A_offd), memory_location);
      hypre_TFree(hypre_CSRMatrixJ(A_offd), memory_location);
      hypre_TFree(hypre_CSRMatrixData(A_offd), memory_location);
      hypre_TFree(hypre_ParCSRMatrixColMapOffd(A), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCSRMatrixDeviceColMapOffd(A), memory_location);
      hypre_CSRMatrixI(A_offd) = hypre_CTAlloc(HYPRE_Int, nrows + 1, memory_location);

      hypre_CSRMatrixTruncateDiag(A_diag);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildNonGalerkinCoarseOperatorHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildNonGalerkinCoarseOperatorHost(hypre_ParCSRMatrix    *A_FF,
                                            hypre_ParCSRMatrix    *A_FC,
                                            hypre_ParCSRMatrix    *A_CF,
                                            hypre_ParCSRMatrix    *A_CC,
                                            hypre_ParCSRMatrix    *Wp,
                                            hypre_ParCSRMatrix    *Wr,
                                            HYPRE_Int              fine_blk_dim,
                                            HYPRE_Int              coarse_blk_dim,
                                            HYPRE_Int              ordering,
                                            HYPRE_Int              method,
                                            HYPRE_Int              max_elmts,
                                            hypre_ParCSRMatrix   **A_H_ptr)
{
   hypre_ParCSRMatrix    *A_H = NULL;
   hypre_ParCSRMatrix    *A_Hc = NULL;
   hypre_ParCSRMatrix    *Wp_tmp = NULL;
   hypre_ParCSRMatrix    *Wr_tmp = NULL;
   hypre_ParCSRMatrix    *A_CF_truncated = NULL;
   hypre_ParCSRMatrix    *A_FF_inv = NULL;
   hypre_ParCSRMatrix    *minus_Wp = NULL;

   HYPRE_Int              blk_inv_size;
   HYPRE_Real             neg_one = -1.0;
   HYPRE_Real             one = 1.0;
   HYPRE_Real             beta = neg_one;

   if (Wp != NULL && max_elmts > 0)
   {
      /* A_Hc = diag(A_CF * Wp) */
      hypre_ParCSRMatMatDiag(A_CF, Wp, &A_Hc);

      /* Coarse grid / Schur complement
         Note that beta is one since A_Hc has positive sign */
      hypre_ParCSRMatrixAdd(one, A_CC, one, A_Hc, &A_H);

      /* Free memory */
      hypre_ParCSRMatrixDestroy(A_Hc);

      /* Set output pointer */
      *A_H_ptr = A_H;

      return hypre_error_flag;
   }

   if (method == 1)
   {
      if (Wp != NULL)
      {
         A_Hc = hypre_ParCSRMatMat(A_CF, Wp);
         beta = one;
      }
      else
      {
         // Build block diagonal inverse for A_FF
         hypre_ParCSRMatrixBlockDiagMatrix(A_FF, fine_blk_dim, -1, NULL, 1, &A_FF_inv);

         // compute Wp = A_FF_inv * A_FC
         // NOTE: Use hypre_ParMatmul here instead of hypre_ParCSRMatMat to avoid padding
         // zero entries at diagonals for the latter routine. Use MatMat once this padding
         // issue is resolved since it is more efficient.
         //         hypre_ParCSRMatrix *Wp_tmp = hypre_ParCSRMatMat(A_FF_inv, A_FC);
         Wp_tmp = hypre_ParMatmul(A_FF_inv, A_FC);

         /* Compute correction A_Hc = A_CF * (A_FF_inv * A_FC); */
         A_Hc = hypre_ParCSRMatMat(A_CF, Wp_tmp);
         hypre_ParCSRMatrixDestroy(Wp_tmp);
         hypre_ParCSRMatrixDestroy(A_FF_inv);
      }
   }
   else if (method == 2 || method == 3)
   {
      /* Extract the diagonal of A_CF */
      hypre_MGRTruncateAcfCPR(A_CF, &A_CF_truncated);
      if (Wp != NULL)
      {
         A_Hc = hypre_ParCSRMatMat(A_CF_truncated, Wp);
      }
      else
      {
         blk_inv_size = method == 2 ? 1 : fine_blk_dim;
         hypre_ParCSRMatrixBlockDiagMatrix(A_FF, blk_inv_size, -1, NULL, 1, &A_FF_inv);

         /* TODO (VPM): We shouldn't need to compute Wr_tmp since we are passing in Wr already */
         HYPRE_UNUSED_VAR(Wr);
         Wr_tmp = hypre_ParCSRMatMat(A_CF_truncated, A_FF_inv);
         A_Hc = hypre_ParCSRMatMat(Wr_tmp, A_FC);
         hypre_ParCSRMatrixDestroy(Wr_tmp);
         hypre_ParCSRMatrixDestroy(A_FF_inv);
      }
      hypre_ParCSRMatrixDestroy(A_CF_truncated);
   }
   else if (method == 4)
   {
      /* Approximate inverse for ideal interploation */
      hypre_MGRApproximateInverse(A_FF, &A_FF_inv);

      minus_Wp = hypre_ParCSRMatMat(A_FF_inv, A_FC);
      A_Hc = hypre_ParCSRMatMat(A_CF, minus_Wp);

      hypre_ParCSRMatrixDestroy(minus_Wp);
   }

   /* Drop small entries in the correction term A_Hc */
   hypre_MGRNonGalerkinTruncate(A_Hc, ordering, coarse_blk_dim, max_elmts);

   /* Coarse grid / Schur complement */
   hypre_ParCSRMatrixAdd(one, A_CC, beta, A_Hc, &A_H);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_Hc);

   /* Set output pointer */
   *A_H_ptr = A_H;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildNonGalerkinCoarseOperatorDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildNonGalerkinCoarseOperatorDevice(hypre_ParCSRMatrix    *A_FF,
                                              hypre_ParCSRMatrix    *A_FC,
                                              hypre_ParCSRMatrix    *A_CF,
                                              hypre_ParCSRMatrix    *A_CC,
                                              hypre_ParCSRMatrix    *Wp,
                                              hypre_ParCSRMatrix    *Wr,
                                              HYPRE_Int              fine_blk_dim,
                                              HYPRE_Int              coarse_blk_dim,
                                              HYPRE_Int              ordering,
                                              HYPRE_Int              method,
                                              HYPRE_Int              max_elmts,
                                              hypre_ParCSRMatrix   **A_H_ptr)
{
   /* Local variables */
   hypre_ParCSRMatrix   *A_H;
   hypre_ParCSRMatrix   *A_Hc;
   hypre_ParCSRMatrix   *A_CF_trunc;
   hypre_ParCSRMatrix   *Wp_tmp = Wp;
   HYPRE_Complex         alpha  = -1.0;

   hypre_GpuProfilingPushRange("MGRComputeNonGalerkinCG");

   /* Truncate A_CF according to the method */
   if (method == 2 || method == 3)
   {
      hypre_MGRTruncateAcfCPRDevice(A_CF, &A_CF_trunc);
   }
   else
   {
      A_CF_trunc = A_CF;
   }

   /* Compute Wp/Wr if not passed in */
   if (!Wp && (method == 1 || method == 2))
   {
      hypre_ParVector      *D_FF_inv;
      HYPRE_Complex        *data;

      /* Create vector to store A_FF's diagonal inverse  */
      D_FF_inv = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_FF),
                                       hypre_ParCSRMatrixGlobalNumRows(A_FF),
                                       hypre_ParCSRMatrixRowStarts(A_FF));
      hypre_ParVectorInitialize_v2(D_FF_inv, HYPRE_MEMORY_DEVICE);
      data = hypre_ParVectorLocalData(D_FF_inv);

      /* Compute the inverse of A_FF and compute its inverse */
      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A_FF), data, 2);
      hypre_ParVectorScale(-1.0, D_FF_inv);

      /* Compute D_FF_inv*A_FC */
      Wp_tmp = hypre_ParCSRMatrixClone(A_FC, 1);
      hypre_ParCSRMatrixDiagScale(Wp_tmp, D_FF_inv, NULL);

      /* Free memory */
      hypre_ParVectorDestroy(D_FF_inv);
   }
   else if (!Wp && (method == 3))
   {
      hypre_ParCSRMatrix  *B_FF_inv;

      /* Compute the block diagonal inverse of A_FF */
      hypre_ParCSRMatrixBlockDiagMatrix(A_FF, fine_blk_dim, -1, NULL, 1, &B_FF_inv);

      /* Compute Wp = A_FF_inv * A_FC */
      Wp_tmp = hypre_ParCSRMatMat(B_FF_inv, A_FC);
      hypre_ParCSRMatrixScale(Wp_tmp, -1.0);

      /* Free memory */
      hypre_ParCSRMatrixDestroy(B_FF_inv);
   }

   /* Compute A_Hc (the correction for A_H) */
   if (Wp_tmp)
   {
      if (max_elmts > 0)
      {
         /* A_Hc = diag(A_CF * Wp) */
         hypre_ParCSRMatMatDiag(A_CF_trunc, Wp_tmp, &A_Hc);

         /* Coarse grid / Schur complement */
         hypre_ParCSRMatrixAdd(1.0, A_CC, 1.0, A_Hc, &A_H);

         /* Free memory */
         hypre_ParCSRMatrixDestroy(A_Hc);
         if (method == 2 || method == 3)
         {
            hypre_ParCSRMatrixDestroy(A_CF_trunc);
         }
         if (Wp_tmp != Wp)
         {
            hypre_ParCSRMatrixDestroy(Wp_tmp);
         }

         /* Set output pointer */
         *A_H_ptr = A_H;

         hypre_GpuProfilingPopRange();

         return hypre_error_flag;
      }

      A_Hc = hypre_ParCSRMatMat(A_CF_trunc, Wp_tmp);
   }
   else if (Wr)
   {
      A_Hc = hypre_ParCSRMatMat(Wr, A_FC);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Wp/Wr matrices was not provided!");
      hypre_GpuProfilingPopRange();

      return hypre_error_flag;
   }

   /* Filter A_Hc */
   hypre_MGRNonGalerkinTruncate(A_Hc, ordering, coarse_blk_dim, max_elmts);

   /* Coarse grid (Schur complement) computation */
   hypre_ParCSRMatrixAdd(1.0, A_CC, alpha, A_Hc, &A_H);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_Hc);
   if (Wp_tmp != Wp)
   {
      hypre_ParCSRMatrixDestroy(Wp_tmp);
   }
   if (method == 2 || method == 3)
   {
      hypre_ParCSRMatrixDestroy(A_CF_trunc);
   }

   /* Set output pointer to coarse grid matrix */
   *A_H_ptr = A_H;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildNonGalerkinCoarseOperator
 *
 * Computes the coarse level operator A_H = RAP via a Non-Galerkin approach.
 *
 * Available methods:
 *   1: inv(A_FF) approximated by its (block) diagonal inverse
 *   2: CPR-like approx. with inv(A_FF) approx. by its diagonal inverse
 *   3: CPR-like approx. with inv(A_FF) approx. by its block diagonal inverse
 *   4: inv(A_FF) approximated by sparse approximate inverse
 *
 * Methods 1-4 assume that restriction is the injection operator.
 *
 * TODO (VPM): inv(A_FF)*A_FC might have been computed before. Reuse it!
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildNonGalerkinCoarseOperator(hypre_ParCSRMatrix    *A_FF,
                                        hypre_ParCSRMatrix    *A_FC,
                                        hypre_ParCSRMatrix    *A_CF,
                                        hypre_ParCSRMatrix    *A_CC,
                                        hypre_ParCSRMatrix    *Wp,
                                        hypre_ParCSRMatrix    *Wr,
                                        HYPRE_Int              fine_blk_dim,
                                        HYPRE_Int              coarse_blk_dim,
                                        HYPRE_Int              ordering,
                                        HYPRE_Int              method,
                                        HYPRE_Int              max_elmts,
                                        hypre_ParCSRMatrix   **A_H_ptr)
{
   hypre_ParCSRMatrix   *matrices[6] = {A_FF, A_FC, A_CF, A_CC, Wp, Wr};
   HYPRE_Int             i;

   /* Check that the memory locations of the input matrices match */
   for (i = 0; i < 5; i++)
   {
      if (matrices[i] && matrices[i + 1] &&
          hypre_ParCSRMatrixMemoryLocation(matrices[i]) !=
          hypre_ParCSRMatrixMemoryLocation(matrices[i + 1]))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Memory locations do not match!");
         return hypre_error_flag;
      }
   }

#if defined (HYPRE_USING_GPU)
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A_FF);

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      hypre_MGRBuildNonGalerkinCoarseOperatorDevice(A_FF, A_FC, A_CF, A_CC, Wp, Wr,
                                                    fine_blk_dim, coarse_blk_dim,
                                                    ordering, method, max_elmts, A_H_ptr);
   }
   else
#endif
   {
      hypre_MGRBuildNonGalerkinCoarseOperatorHost(A_FF, A_FC, A_CF, A_CC, Wp, Wr,
                                                  fine_blk_dim, coarse_blk_dim,
                                                  ordering, method, max_elmts, A_H_ptr);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBuildCoarseOperator
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBuildCoarseOperator(void                *mgr_vdata,
                             hypre_ParCSRMatrix  *A_FF,
                             hypre_ParCSRMatrix  *A_FC,
                             hypre_ParCSRMatrix  *A_CF,
                             hypre_ParCSRMatrix **A_CC_ptr,
                             hypre_ParCSRMatrix  *Wp,
                             hypre_ParCSRMatrix  *Wr,
                             HYPRE_Int            level)
{
   hypre_ParMGRData      *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   hypre_ParCSRMatrix    *A  = (mgr_data -> A_array)[level];
   hypre_ParCSRMatrix    *P  = (mgr_data -> P_array)[level];
   hypre_ParCSRMatrix    *R  = (mgr_data -> R_array)[level];
   hypre_ParCSRMatrix    *RT = (mgr_data -> RT_array)[level];
   hypre_ParCSRMatrix    *A_CC = *A_CC_ptr;

   HYPRE_Int             *blk_dims = (mgr_data -> block_num_coarse_indexes);
   HYPRE_Int              block_size = (mgr_data -> block_size);
   HYPRE_Int              method = (mgr_data -> coarse_grid_method)[level];
   HYPRE_Int              num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Int              ordering = (mgr_data -> set_c_points_method);
   HYPRE_Int              max_elmts = (mgr_data -> nonglk_max_elmts)[level];
   HYPRE_Real             threshold = (mgr_data -> truncate_coarse_grid_threshold);

   hypre_ParCSRMatrix    *AP, *RAP, *RAP_c;
   HYPRE_Int              fine_blk_dim = (level) ? blk_dims[level - 1] - blk_dims[level] :
                                         block_size - blk_dims[level];
   HYPRE_Int              coarse_blk_dim = blk_dims[level];
   HYPRE_Int              rebuild_commpkg = 0;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("RAP");

   if (!method)
   {
      /* Galerkin path */
      if (Wr && !Wp)
      {
         /* Prolongation is the injection operator (Wp == NULL) and
            Restriction is not the injection operator (Wr != NULL) */
         RAP_c = hypre_ParCSRMatMat(Wr, A_FC);
         hypre_ParCSRMatrixAdd(1.0, A_CC, -1.0, RAP_c, &RAP);
         hypre_ParCSRMatrixDestroy(RAP_c);
      }
      else if (RT)
      {
         RAP = hypre_ParCSRMatrixRAPKT(RT, A, P, 1);
      }
      else if (R)
      {
         AP  = hypre_ParCSRMatMat(A, P);
         RAP = hypre_ParCSRMatMat(R, AP);
         hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(RAP));
         hypre_ParCSRMatrixDestroy(AP);
      }
      else
      {
         hypre_GpuProfilingPopRange();
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Expected either R or RT!");
         return hypre_error_flag;
      }
   }
   else if (method == 5)
   {
      /* Approximate the coarse level matrix as A_CC */
      RAP = *A_CC_ptr;
      *A_CC_ptr = NULL;
   }
   else
   {
      /* Non-Galerkin path */
      hypre_MGRBuildNonGalerkinCoarseOperator(A_FF, A_FC, A_CF, A_CC, Wp, Wr,
                                              fine_blk_dim, coarse_blk_dim,
                                              ordering, method, max_elmts, &RAP);
   }

   /* Truncate coarse level matrix based on input threshold */
   if (threshold > 0.0)
   {
#if defined (HYPRE_USING_GPU)
      HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(RAP);

      if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
      {
         hypre_ParCSRMatrixDropSmallEntriesDevice(RAP, threshold, -1);
         rebuild_commpkg = 1;
      }
      else
#endif
      {
         hypre_ParCSRMatrixTruncate(RAP, threshold, 0, 0, 0);
      }
   }

   /* Compute/rebuild communication package */
   if (rebuild_commpkg)
   {
      if (hypre_ParCSRMatrixCommPkg(RAP))
      {
         hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(RAP));
      }
      hypre_MatvecCommPkgCreate(RAP);
   }
   if (!hypre_ParCSRMatrixCommPkg(RAP))
   {
      hypre_MatvecCommPkgCreate(RAP);
   }

   /* Set coarse grid matrix */
   (mgr_data -> A_array)[level + 1] = RAP;
   if ((level + 1) == num_coarse_levels)
   {
      (mgr_data -> RAP) = RAP;
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
