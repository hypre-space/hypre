/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

__global__ void
hypreCUDAKernel_InterpTruncation( HYPRE_Int   nrows,
                                  HYPRE_Real  trunc_factor,
                                  HYPRE_Int   max_elmts,
                                  HYPRE_Int  *P_i,
                                  HYPRE_Int  *P_j,
                                  HYPRE_Real *P_a)
{
   HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>(), p, q;

   /* 1. compute row max, rowsum */
   if (lane < 2)
   {
      p = read_only_load(P_i + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int i = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q); i += HYPRE_WARP_SIZE)
   {
      if (i < q)
      {
         HYPRE_Real v = read_only_load(&P_a[i]);
         row_max = hypre_max(row_max, fabs(v));
         row_sum += v;
      }
   }

   row_max = warp_allreduce_max(row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(row_sum);

   /* 2. mark dropped entries by -1 in P_j, and compute row_scal */
   HYPRE_Int last_pos = -1;
   for (HYPRE_Int i = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, i < q); i += HYPRE_WARP_SIZE)
   {
      HYPRE_Int cond = 0, cond_prev;

      cond_prev = i == p + lane || warp_allreduce_min(cond);

      if (i < q)
      {
         HYPRE_Real v;
         cond = cond_prev && (max_elmts == 0 || i < p + max_elmts);
         if (cond)
         {
            v = read_only_load(&P_a[i]);
         }
         cond = cond && fabs(v) >= row_max;

         if (cond)
         {
            last_pos = i;
            row_scal += v;
         }
         else
         {
            P_j[i] = -1;
         }
      }
   }

   row_scal = warp_allreduce_sum(row_scal);

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 3. scale the row */
   for (HYPRE_Int i = p + lane; i <= last_pos; i += HYPRE_WARP_SIZE)
   {
      P_a[i] *= row_scal;
   }
}

/*-----------------------------------------------------------------------*
 * RL: To be consistent with the CPU version, max_elmts == 0 means no limit on rownnz
 */
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor, HYPRE_Int max_elmts )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif

   hypre_CSRMatrix *P_diag      = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int       *P_diag_i    = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j    = hypre_CSRMatrixJ(P_diag);
   HYPRE_Real      *P_diag_a    = hypre_CSRMatrixData(P_diag);

   hypre_CSRMatrix *P_offd      = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int       *P_offd_i    = hypre_CSRMatrixI(P_offd);
   HYPRE_Int       *P_offd_j    = hypre_CSRMatrixJ(P_offd);
   HYPRE_Real      *P_offd_a    = hypre_CSRMatrixData(P_offd);

   //HYPRE_Int        ncols       = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int        nrows       = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        nnz_diag    = hypre_CSRMatrixNumNonzeros(P_diag);
   HYPRE_Int        nnz_offd    = hypre_CSRMatrixNumNonzeros(P_offd);
   HYPRE_Int        nnz_P       = nnz_diag + nnz_offd;
   HYPRE_Int       *P_i         = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_j         = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Real      *P_a         = hypre_TAlloc(HYPRE_Real, nnz_P,   HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_rowptr    = hypre_TAlloc(HYPRE_Int,  nrows+1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *tmp_rowid   = hypre_TAlloc(HYPRE_Int,  nnz_P,   HYPRE_MEMORY_DEVICE);

   HYPRE_Int        new_nnz_diag = 0, new_nnz_offd = 0;
   HYPRE_Int        ierr         = 0;

   HYPRE_MemoryLocation        memory_location = hypre_ParCSRMatrixMemoryLocation(P);

   /*
   HYPRE_Int        num_procs, my_id;
   MPI_Comm         comm = hypre_ParCSRMatrixComm(P);
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   */

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_diag, P_diag_i, P_i);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_offd, P_offd_i, P_i + nnz_diag);

   hypre_TMemcpy(P_j, P_diag_j, HYPRE_Int, nnz_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   /* offd col id := -2 - offd col id */
   HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag, -_1 - 2);

   hypre_TMemcpy(P_a,            P_diag_a, HYPRE_Real, nnz_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(P_a + nnz_diag, P_offd_a, HYPRE_Real, nnz_offd, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort rows based on (rowind, abs(P_a)) */
   hypreDevice_StableSortByTupleKey(nnz_P, P_i, P_a, P_j, 1);

   hypreDevice_CsrRowIndicesToPtrs_v2(nrows, nnz_P, P_i, P_rowptr);

   /* truncate P, unwanted entries are marked -1 in P_j */
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_InterpTruncation, gDim, bDim,
                      nrows, trunc_factor, max_elmts, P_rowptr, P_j, P_a );

   /* build new P_diag and P_offd */
   if (nnz_diag)
   {
      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
            thrust::make_zip_iterator(thrust::make_tuple(P_i+nnz_P, P_j+nnz_P, P_a+nnz_P)),
            P_j,
            thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_diag_j,  P_diag_a)),
            is_nonnegative<HYPRE_Int>() );

      new_nnz_diag = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;

      hypre_assert(new_nnz_diag <= nnz_diag);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_diag, tmp_rowid, P_diag_i);
   }

   if (nnz_offd)
   {
      less_than<HYPRE_Int> pred(-1);
      auto new_end = HYPRE_THRUST_CALL(
            copy_if,
            thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
            thrust::make_zip_iterator(thrust::make_tuple(P_i+nnz_P, P_j+nnz_P, P_a+nnz_P)),
            P_j,
            thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_offd_j,  P_offd_a)),
            pred );

      new_nnz_offd = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;

      hypre_assert(new_nnz_offd <= nnz_offd);

      HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j, -_1 - 2);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_offd, tmp_rowid, P_offd_i);
   }

   /*
   printf("nnz_diag %d, new nnz_diag %d\n", nnz_diag, new_nnz_diag);
   printf("nnz_offd %d, new nnz_offd %d\n", nnz_offd, new_nnz_offd);
   */

   hypre_CSRMatrixJ   (P_diag) = hypre_TReAlloc_v2(P_diag_j, HYPRE_Int,  nnz_diag, HYPRE_Int,  new_nnz_diag, memory_location);
   hypre_CSRMatrixData(P_diag) = hypre_TReAlloc_v2(P_diag_a, HYPRE_Real, nnz_diag, HYPRE_Real, new_nnz_diag, memory_location);
   hypre_CSRMatrixJ   (P_offd) = hypre_TReAlloc_v2(P_offd_j, HYPRE_Int,  nnz_offd, HYPRE_Int,  new_nnz_offd, memory_location);
   hypre_CSRMatrixData(P_offd) = hypre_TReAlloc_v2(P_offd_a, HYPRE_Real, nnz_offd, HYPRE_Real, new_nnz_offd, memory_location);
   hypre_CSRMatrixNumNonzeros(P_diag) = new_nnz_diag;
   hypre_CSRMatrixNumNonzeros(P_offd) = new_nnz_offd;

   hypre_TFree(P_i,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_j,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_a,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_rowptr,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp_rowid, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] += hypre_MPI_Wtime();
#endif

   return ierr;
}

#endif /* #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
