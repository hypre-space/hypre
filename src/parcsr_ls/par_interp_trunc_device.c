/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

#define HYPRE_INTERPTRUNC_ALGORITHM_SWITCH 8

/* special case for max_elmts = 0, i.e. no max_elmts limit */
__global__ void
hypreGPUKernel_InterpTruncationPass0_v1( hypre_DeviceItem &item,
                                         HYPRE_Int   nrows,
                                         HYPRE_Real  trunc_factor,
                                         HYPRE_Int  *P_diag_i,
                                         HYPRE_Int  *P_diag_j,
                                         HYPRE_Real *P_diag_a,
                                         HYPRE_Int  *P_offd_i,
                                         HYPRE_Int  *P_offd_j,
                                         HYPRE_Real *P_offd_a,
                                         HYPRE_Int  *P_diag_i_new,
                                         HYPRE_Int  *P_offd_i_new )
{
   HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;

   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p_diag = 0, q_diag = 0, p_offd = 0, q_offd = 0;

   if (lane < 2)
   {
      p_diag = read_only_load(P_diag_i + row + lane);
      p_offd = read_only_load(P_offd_i + row + lane);
   }
   q_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 1);
   p_diag = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_diag, 0);
   q_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 1);
   p_offd = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p_offd, 0);

   /* 1. compute row rowsum, rowmax */
   for (HYPRE_Int i = p_diag + lane; i < q_diag; i += HYPRE_WARP_SIZE)
   {
      HYPRE_Real v = P_diag_a[i];
      row_sum += v;
      row_max = hypre_max(row_max, hypre_abs(v));
   }

   for (HYPRE_Int i = p_offd + lane; i < q_offd; i += HYPRE_WARP_SIZE)
   {
      HYPRE_Real v = P_offd_a[i];
      row_sum += v;
      row_max = hypre_max(row_max, hypre_abs(v));
   }

   row_max = warp_allreduce_max(item, row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(item, row_sum);

   HYPRE_Int cnt_diag = 0, cnt_offd = 0;

   /* 2. move wanted entries to the front and row scal */
   for (HYPRE_Int i = p_diag + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < q_diag);
        i += HYPRE_WARP_SIZE)
   {
      HYPRE_Real v = 0.0;
      HYPRE_Int j = -1;

      if (i < q_diag)
      {
         v = P_diag_a[i];

         if (hypre_abs(v) >= row_max)
         {
            j = P_diag_j[i];
            row_scal += v;
         }
      }

      HYPRE_Int sum, pos;
      pos = warp_prefix_sum(item, lane, (HYPRE_Int) (j != -1), sum);

      if (j != -1)
      {
         P_diag_a[p_diag + cnt_diag + pos] = v;
         P_diag_j[p_diag + cnt_diag + pos] = j;
      }

      cnt_diag += sum;
   }

   for (HYPRE_Int i = p_offd + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < q_offd);
        i += HYPRE_WARP_SIZE)
   {
      HYPRE_Real v = 0.0;
      HYPRE_Int j = -1;

      if (i < q_offd)
      {
         v = P_offd_a[i];

         if (hypre_abs(v) >= row_max)
         {
            j = P_offd_j[i];
            row_scal += v;
         }
      }

      HYPRE_Int sum, pos;
      pos = warp_prefix_sum(item, lane, (HYPRE_Int) (j != -1), sum);

      if (j != -1)
      {
         P_offd_a[p_offd + cnt_offd + pos] = v;
         P_offd_j[p_offd + cnt_offd + pos] = j;
      }

      cnt_offd += sum;
   }

   row_scal = warp_allreduce_sum(item, row_scal);

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 3. scale the row */
   for (HYPRE_Int i = p_diag + lane; i < p_diag + cnt_diag; i += HYPRE_WARP_SIZE)
   {
      P_diag_a[i] *= row_scal;
   }

   for (HYPRE_Int i = p_offd + lane; i < p_offd + cnt_offd; i += HYPRE_WARP_SIZE)
   {
      P_offd_a[i] *= row_scal;
   }

   if (!lane)
   {
      P_diag_i_new[row] = cnt_diag;
      P_offd_i_new[row] = cnt_offd;
   }
}

static __device__ __forceinline__
void hypre_smallest_abs_val( HYPRE_Int   n,
                             HYPRE_Real *v,
                             HYPRE_Real &min_v,
                             HYPRE_Int  &min_j )
{
   min_v = hypre_abs(v[0]);
   min_j = 0;

   for (HYPRE_Int j = 1; j < n; j++)
   {
      const HYPRE_Real vj = hypre_abs(v[j]);
      if (vj < min_v)
      {
         min_v = vj;
         min_j = j;
      }
   }
}

/* TODO: using 1 thread per row, which can be suboptimal */
__global__ void
hypreGPUKernel_InterpTruncationPass1_v1( hypre_DeviceItem &item,
#if defined(HYPRE_USING_SYCL)
                                         char *shmem_ptr,
#endif
                                         HYPRE_Int   nrows,
                                         HYPRE_Real  trunc_factor,
                                         HYPRE_Int   max_elmts,
                                         HYPRE_Int  *P_diag_i,
                                         HYPRE_Int  *P_diag_j,
                                         HYPRE_Real *P_diag_a,
                                         HYPRE_Int  *P_offd_i,
                                         HYPRE_Int  *P_offd_j,
                                         HYPRE_Real *P_offd_a,
                                         HYPRE_Int  *P_diag_i_new,
                                         HYPRE_Int  *P_offd_i_new )
{
   const HYPRE_Int row = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   const HYPRE_Int p_diag = read_only_load(P_diag_i + row);
   const HYPRE_Int q_diag = read_only_load(P_diag_i + row + 1);
   const HYPRE_Int p_offd = read_only_load(P_offd_i + row);
   const HYPRE_Int q_offd = read_only_load(P_offd_i + row + 1);

   /* 1. get row max and compute truncation threshold, and compute row_sum */
   HYPRE_Real row_max = 0.0, row_sum = 0.0;

   for (HYPRE_Int i = p_diag; i < q_diag; i++)
   {
      HYPRE_Real v = P_diag_a[i];
      row_sum += v;
      row_max = hypre_max(row_max, hypre_abs(v));
   }

   for (HYPRE_Int i = p_offd; i < q_offd; i++)
   {
      HYPRE_Real v = P_offd_a[i];
      row_sum += v;
      row_max = hypre_max(row_max, hypre_abs(v));
   }

   row_max *= trunc_factor;

   /* 2. save the largest max_elmts entries in sh_val/pos */
   const HYPRE_Int nt = hypre_gpu_get_num_threads<1>(item);
   const HYPRE_Int tid = hypre_gpu_get_thread_id<1>(item);
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int *shared_mem = (HYPRE_Int*) shmem_ptr;
#else
   extern __shared__ HYPRE_Int shared_mem[];
#endif
   HYPRE_Int *sh_pos = &shared_mem[tid * max_elmts];
   HYPRE_Real *sh_val = &((HYPRE_Real *) &shared_mem[nt * max_elmts])[tid * max_elmts];
   HYPRE_Int cnt = 0;

   for (HYPRE_Int i = p_diag; i < q_diag; i++)
   {
      const HYPRE_Real v = P_diag_a[i];

      if (hypre_abs(v) < row_max) { continue; }

      if (cnt < max_elmts)
      {
         sh_val[cnt] = v;
         sh_pos[cnt ++] = i;
      }
      else
      {
         HYPRE_Real min_v;
         HYPRE_Int min_j;

         hypre_smallest_abs_val(max_elmts, sh_val, min_v, min_j);

         if (hypre_abs(v) > min_v)
         {
            sh_val[min_j] = v;
            sh_pos[min_j] = i;
         }
      }
   }

   for (HYPRE_Int i = p_offd; i < q_offd; i++)
   {
      const HYPRE_Real v = P_offd_a[i];

      if (hypre_abs(v) < row_max) { continue; }

      if (cnt < max_elmts)
      {
         sh_val[cnt] = v;
         sh_pos[cnt ++] = i + q_diag;
      }
      else
      {
         HYPRE_Real min_v;
         HYPRE_Int min_j;

         hypre_smallest_abs_val(max_elmts, sh_val, min_v, min_j);

         if (hypre_abs(v) > min_v)
         {
            sh_val[min_j] = v;
            sh_pos[min_j] = i + q_diag;
         }
      }
   }

   /* 3. load actual j and compute row_scal */
   HYPRE_Real row_scal = 0.0;

   for (HYPRE_Int i = 0; i < cnt; i++)
   {
      const HYPRE_Int j = sh_pos[i];

      if (j < q_diag)
      {
         sh_pos[i] = P_diag_j[j];
      }
      else
      {
         sh_pos[i] = -1 - P_offd_j[j - q_diag];
      }

      row_scal += sh_val[i];
   }

   if (row_scal)
   {
      row_scal = row_sum / row_scal;
   }
   else
   {
      row_scal = 1.0;
   }

   /* 4. write to P_diag_j and P_offd_j */
   HYPRE_Int cnt_diag = 0;
   for (HYPRE_Int i = 0; i < cnt; i++)
   {
      const HYPRE_Int j = sh_pos[i];

      if (j >= 0)
      {
         P_diag_j[p_diag + cnt_diag] = j;
         P_diag_a[p_diag + cnt_diag] = sh_val[i] * row_scal;
         cnt_diag ++;
      }
      else
      {
         P_offd_j[p_offd + i - cnt_diag] = -1 - j;
         P_offd_a[p_offd + i - cnt_diag] = sh_val[i] * row_scal;
      }
   }

   P_diag_i_new[row] = cnt_diag;
   P_offd_i_new[row] = cnt - cnt_diag;
}

/* using 1 warp per row */
__global__ void
hypreGPUKernel_InterpTruncationPass2_v1( hypre_DeviceItem &item,
                                         HYPRE_Int   nrows,
                                         HYPRE_Int  *P_diag_i,
                                         HYPRE_Int  *P_diag_j,
                                         HYPRE_Real *P_diag_a,
                                         HYPRE_Int  *P_offd_i,
                                         HYPRE_Int  *P_offd_j,
                                         HYPRE_Real *P_offd_a,
                                         HYPRE_Int  *P_diag_i_new,
                                         HYPRE_Int  *P_diag_j_new,
                                         HYPRE_Real *P_diag_a_new,
                                         HYPRE_Int  *P_offd_i_new,
                                         HYPRE_Int  *P_offd_j_new,
                                         HYPRE_Real *P_offd_a_new )
{
   HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, pnew = 0, qnew = 0, shift;

   if (lane < 2)
   {
      p = read_only_load(P_diag_i + i + lane);
      pnew = read_only_load(P_diag_i_new + i + lane);
   }
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);
   qnew = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pnew, 1);
   pnew = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pnew, 0);

   shift = p - pnew;
   for (HYPRE_Int k = pnew + lane; k < qnew; k += HYPRE_WARP_SIZE)
   {
      P_diag_j_new[k] = P_diag_j[k + shift];
      P_diag_a_new[k] = P_diag_a[k + shift];
   }

   if (lane < 2)
   {
      p = read_only_load(P_offd_i + i + lane);
      pnew = read_only_load(P_offd_i_new + i + lane);
   }
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);
   qnew = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pnew, 1);
   pnew = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, pnew, 0);

   shift = p - pnew;
   for (HYPRE_Int k = pnew + lane; k < qnew; k += HYPRE_WARP_SIZE)
   {
      P_offd_j_new[k] = P_offd_j[k + shift];
      P_offd_a_new[k] = P_offd_a[k + shift];
   }
}

/* This is a "fast" version that works for small max_elmts values */
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice_v1( hypre_ParCSRMatrix *P,
                                          HYPRE_Real          trunc_factor,
                                          HYPRE_Int           max_elmts )
{
   HYPRE_Int        nrows       = hypre_ParCSRMatrixNumRows(P);
   hypre_CSRMatrix *P_diag      = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int       *P_diag_i    = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j    = hypre_CSRMatrixJ(P_diag);
   HYPRE_Real      *P_diag_a    = hypre_CSRMatrixData(P_diag);
   hypre_CSRMatrix *P_offd      = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int       *P_offd_i    = hypre_CSRMatrixI(P_offd);
   HYPRE_Int       *P_offd_j    = hypre_CSRMatrixJ(P_offd);
   HYPRE_Real      *P_offd_a    = hypre_CSRMatrixData(P_offd);

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(P);

   HYPRE_Int *P_diag_i_new = hypre_TAlloc(HYPRE_Int, nrows + 1, memory_location);
   HYPRE_Int *P_offd_i_new = hypre_TAlloc(HYPRE_Int, nrows + 1, memory_location);

   /* truncate P, wanted entries are marked negative in P_diag/offd_j */
   if (max_elmts == 0)
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

      HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncationPass0_v1,
                        gDim, bDim,
                        nrows, trunc_factor,
                        P_diag_i, P_diag_j, P_diag_a,
                        P_offd_i, P_offd_j, P_offd_a,
                        P_diag_i_new, P_offd_i_new);
   }
   else
   {
      dim3 bDim = hypre_dim3(256);
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "thread", bDim);
#if defined(HYPRE_USING_SYCL)
      size_t shmem_bytes = bDim.get(2) * max_elmts * (sizeof(HYPRE_Int) + sizeof(HYPRE_Real));
#else
      size_t shmem_bytes = bDim.x * max_elmts * (sizeof(HYPRE_Int) + sizeof(HYPRE_Real));
#endif
      HYPRE_GPU_LAUNCH2( hypreGPUKernel_InterpTruncationPass1_v1,
                         gDim, bDim, shmem_bytes,
                         nrows, trunc_factor, max_elmts,
                         P_diag_i, P_diag_j, P_diag_a,
                         P_offd_i, P_offd_j, P_offd_a,
                         P_diag_i_new, P_offd_i_new);
   }

   hypre_Memset(&P_diag_i_new[nrows], 0, sizeof(HYPRE_Int), memory_location);
   hypre_Memset(&P_offd_i_new[nrows], 0, sizeof(HYPRE_Int), memory_location);

   hypreDevice_IntegerExclusiveScan(nrows + 1, P_diag_i_new);
   hypreDevice_IntegerExclusiveScan(nrows + 1, P_offd_i_new);

   HYPRE_Int nnz_diag, nnz_offd;

   hypre_TMemcpy(&nnz_diag, &P_diag_i_new[nrows], HYPRE_Int, 1,
                 HYPRE_MEMORY_HOST, memory_location);
   hypre_TMemcpy(&nnz_offd, &P_offd_i_new[nrows], HYPRE_Int, 1,
                 HYPRE_MEMORY_HOST, memory_location);

   hypre_CSRMatrixNumNonzeros(P_diag) = nnz_diag;
   hypre_CSRMatrixNumNonzeros(P_offd) = nnz_offd;

   HYPRE_Int  *P_diag_j_new = hypre_TAlloc(HYPRE_Int,  nnz_diag, memory_location);
   HYPRE_Real *P_diag_a_new = hypre_TAlloc(HYPRE_Real, nnz_diag, memory_location);
   HYPRE_Int  *P_offd_j_new = hypre_TAlloc(HYPRE_Int,  nnz_offd, memory_location);
   HYPRE_Real *P_offd_a_new = hypre_TAlloc(HYPRE_Real, nnz_offd, memory_location);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncationPass2_v1,
                     gDim, bDim,
                     nrows,
                     P_diag_i, P_diag_j, P_diag_a,
                     P_offd_i, P_offd_j, P_offd_a,
                     P_diag_i_new, P_diag_j_new, P_diag_a_new,
                     P_offd_i_new, P_offd_j_new, P_offd_a_new );

   hypre_CSRMatrixI   (P_diag) = P_diag_i_new;
   hypre_CSRMatrixJ   (P_diag) = P_diag_j_new;
   hypre_CSRMatrixData(P_diag) = P_diag_a_new;
   hypre_CSRMatrixI   (P_offd) = P_offd_i_new;
   hypre_CSRMatrixJ   (P_offd) = P_offd_j_new;
   hypre_CSRMatrixData(P_offd) = P_offd_a_new;

   hypre_TFree(P_diag_i, memory_location);
   hypre_TFree(P_diag_j, memory_location);
   hypre_TFree(P_diag_a, memory_location);
   hypre_TFree(P_offd_i, memory_location);
   hypre_TFree(P_offd_j, memory_location);
   hypre_TFree(P_offd_a, memory_location);

   return hypre_error_flag;
}

__global__ void
hypreGPUKernel_InterpTruncation_v2( hypre_DeviceItem &item,
                                    HYPRE_Int   nrows,
                                    HYPRE_Real  trunc_factor,
                                    HYPRE_Int   max_elmts,
                                    HYPRE_Int  *P_i,
                                    HYPRE_Int  *P_j,
                                    HYPRE_Real *P_a)
{
   HYPRE_Real row_max = 0.0, row_sum = 0.0, row_scal = 0.0;
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item), p = 0, q;

   /* 1. compute row max, rowsum */
   if (lane < 2)
   {
      p = read_only_load(P_i + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int i = p + lane; i < q; i += HYPRE_WARP_SIZE)
   {
      HYPRE_Real v = read_only_load(&P_a[i]);
      row_max = hypre_max(row_max, hypre_abs(v));
      row_sum += v;
   }

   row_max = warp_allreduce_max(item, row_max) * trunc_factor;
   row_sum = warp_allreduce_sum(item, row_sum);

   /* 2. mark dropped entries by -1 in P_j, and compute row_scal */
   HYPRE_Int last_pos = -1;
   for (HYPRE_Int i = p + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < q); i += HYPRE_WARP_SIZE)
   {
      HYPRE_Int cond = 0, cond_prev;

      cond_prev = i == p + lane || warp_allreduce_min(item, cond);

      if (i < q)
      {
         HYPRE_Real v;
         cond = cond_prev && (max_elmts == 0 || i < p + max_elmts);
         if (cond)
         {
            v = read_only_load(&P_a[i]);
         }
         cond = cond && hypre_abs(v) >= row_max;

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

   row_scal = warp_allreduce_sum(item, row_scal);

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

/*------------------------------------------------------------------------------------
 * RL: To be consistent with the CPU version, max_elmts == 0 means no limit on rownnz
 * This is a generic version that works for all max_elmts values
 */
HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice_v2( hypre_ParCSRMatrix *P,
                                          HYPRE_Real          trunc_factor,
                                          HYPRE_Int           max_elmts )
{
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
   HYPRE_Int       *P_i         = hypre_TAlloc(HYPRE_Int,  nnz_P,     HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_j         = hypre_TAlloc(HYPRE_Int,  nnz_P,     HYPRE_MEMORY_DEVICE);
   HYPRE_Real      *P_a         = hypre_TAlloc(HYPRE_Real, nnz_P,     HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *P_rowptr    = hypre_TAlloc(HYPRE_Int,  nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int       *tmp_rowid   = hypre_TAlloc(HYPRE_Int,  nnz_P,     HYPRE_MEMORY_DEVICE);

   HYPRE_Int        new_nnz_diag = 0, new_nnz_offd = 0;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(P);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_diag, P_diag_i, P_i);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnz_offd, P_offd_i, P_i + nnz_diag);

   hypre_TMemcpy(P_j, P_diag_j, HYPRE_Int, nnz_diag, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   /* offd col id := -2 - offd col id */
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag,
   [] (const auto & x) {return -x - 2;} );
#else
   HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + nnz_offd, P_j + nnz_diag, -_1 - 2);
#endif

   hypre_TMemcpy(P_a,            P_diag_a, HYPRE_Real, nnz_diag, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(P_a + nnz_diag, P_offd_a, HYPRE_Real, nnz_offd, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);

   /* sort rows based on (rowind, abs(P_a)) */
   hypreDevice_StableSortByTupleKey(nnz_P, P_i, P_a, P_j, 1);

   hypreDevice_CsrRowIndicesToPtrs_v2(nrows, nnz_P, P_i, P_rowptr);

   /* truncate P, unwanted entries are marked -1 in P_j */
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_InterpTruncation_v2, gDim, bDim,
                     nrows, trunc_factor, max_elmts, P_rowptr, P_j, P_a );

   /* build new P_diag and P_offd */
   if (nnz_diag)
   {
#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(P_i,       P_j,       P_a),
                                        oneapi::dpl::make_zip_iterator(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P),
                                        P_j,
                                        oneapi::dpl::make_zip_iterator(tmp_rowid, P_diag_j,  P_diag_a),
                                        is_nonnegative<HYPRE_Int>() );
      new_nnz_diag = std::get<0>(new_end.base()) - tmp_rowid;
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
                        thrust::make_zip_iterator(thrust::make_tuple(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P)),
                        P_j,
                        thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_diag_j,  P_diag_a)),
                        is_nonnegative<HYPRE_Int>() );
      new_nnz_diag = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;
#endif

      hypre_assert(new_nnz_diag <= nnz_diag);

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_diag, tmp_rowid, P_diag_i);
   }

   if (nnz_offd)
   {
      less_than<HYPRE_Int> pred(-1);
#if defined(HYPRE_USING_SYCL)
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(P_i,       P_j,       P_a),
                                        oneapi::dpl::make_zip_iterator(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P),
                                        P_j,
                                        oneapi::dpl::make_zip_iterator(tmp_rowid, P_offd_j,  P_offd_a),
                                        pred );
      new_nnz_offd = std::get<0>(new_end.base()) - tmp_rowid;
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(P_i,       P_j,       P_a)),
                        thrust::make_zip_iterator(thrust::make_tuple(P_i + nnz_P, P_j + nnz_P, P_a + nnz_P)),
                        P_j,
                        thrust::make_zip_iterator(thrust::make_tuple(tmp_rowid, P_offd_j,  P_offd_a)),
                        pred );
      new_nnz_offd = thrust::get<0>(new_end.get_iterator_tuple()) - tmp_rowid;
#endif

      hypre_assert(new_nnz_offd <= nnz_offd);

#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL(std::transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j,
      [] (const auto & x) {return -x - 2;} );
#else
      HYPRE_THRUST_CALL(transform, P_offd_j, P_offd_j + new_nnz_offd, P_offd_j, -_1 - 2);
#endif

      hypreDevice_CsrRowIndicesToPtrs_v2(nrows, new_nnz_offd, tmp_rowid, P_offd_i);
   }

   hypre_CSRMatrixJ   (P_diag) = hypre_TReAlloc_v2(P_diag_j, HYPRE_Int,  nnz_diag, HYPRE_Int,
                                                   new_nnz_diag, memory_location);
   hypre_CSRMatrixData(P_diag) = hypre_TReAlloc_v2(P_diag_a, HYPRE_Real, nnz_diag, HYPRE_Real,
                                                   new_nnz_diag, memory_location);
   hypre_CSRMatrixJ   (P_offd) = hypre_TReAlloc_v2(P_offd_j, HYPRE_Int,  nnz_offd, HYPRE_Int,
                                                   new_nnz_offd, memory_location);
   hypre_CSRMatrixData(P_offd) = hypre_TReAlloc_v2(P_offd_a, HYPRE_Real, nnz_offd, HYPRE_Real,
                                                   new_nnz_offd, memory_location);
   hypre_CSRMatrixNumNonzeros(P_diag) = new_nnz_diag;
   hypre_CSRMatrixNumNonzeros(P_offd) = new_nnz_offd;

   hypre_TFree(P_i,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_j,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_a,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(P_rowptr,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(tmp_rowid, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P,
                                       HYPRE_Real          trunc_factor,
                                       HYPRE_Int           max_elmts )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif
   hypre_GpuProfilingPushRange("Interp-Truncation");

   if (max_elmts <= HYPRE_INTERPTRUNC_ALGORITHM_SWITCH)
   {
      hypre_BoomerAMGInterpTruncationDevice_v1(P, trunc_factor, max_elmts);
   }
   else
   {
      hypre_BoomerAMGInterpTruncationDevice_v2(P, trunc_factor, max_elmts);
   }

   hypre_GpuProfilingPopRange();

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_GPU) */
