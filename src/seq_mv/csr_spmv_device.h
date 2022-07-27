/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_CSR_SPMV_DEVICE_H
#define hypre_CSR_SPMV_DEVICE_H

#define HYPRE_SPMV_BLOCKDIM 512
#define HYPRE_SPMV_VERSION 1

#define HYPRE_SPMV_ADD_SUM(p)                                                             \
{                                                                                         \
   const HYPRE_Int col = read_only_load(&d_ja[p]);                                        \
   if (F == 0)                                                                            \
   {                                                                                      \
      const T val = d_a ? read_only_load(&d_a[p]) : T(1);                                 \
      for (HYPRE_Int i = 0; i < NV; i++)                                                  \
      {                                                                                   \
         sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);       \
      }                                                                                   \
   }                                                                                      \
   else if (F == -1)                                                                      \
   {                                                                                      \
      if (col <= grid_group_id)                                                           \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < NV; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == 1)                                                                       \
   {                                                                                      \
      if (col >= grid_group_id)                                                           \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < NV; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == -2)                                                                      \
   {                                                                                      \
      if (col < grid_group_id)                                                            \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < NV; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == 2)                                                                       \
   {                                                                                      \
      if (col > grid_group_id)                                                            \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < NV; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
}

#define HYPRE_SPMV_GPU_LAUNCH(nv)                                                         \
   if (avg_rownnz >= 64)                                                                  \
   {                                                                                      \
      const HYPRE_Int group_size = 32;                                                    \
      const HYPRE_Int num_groups_per_block = HYPRE_SPMV_BLOCKDIM / group_size;            \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_GPU_LAUNCH( (hypreGPUKernel_CSRMatvecShuffle<F, group_size, nv, T>),          \
                        gDim, bDim, nrows, rowid, idxstride_x, idxstride_y, vecstride_x,  \
                        vecstride_y, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );            \
   }                                                                                      \
   else if (avg_rownnz >= 32)                                                             \
   {                                                                                      \
      const HYPRE_Int group_size = 16;                                                    \
      const HYPRE_Int num_groups_per_block = HYPRE_SPMV_BLOCKDIM / group_size;            \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_GPU_LAUNCH( (hypreGPUKernel_CSRMatvecShuffle<F, group_size, nv, T>),          \
                        gDim, bDim, nrows, rowid, idxstride_x, idxstride_y, vecstride_x,  \
                        vecstride_y, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );            \
   }                                                                                      \
   else if (avg_rownnz >= 16)                                                             \
   {                                                                                      \
      const HYPRE_Int group_size = 8;                                                     \
      const HYPRE_Int num_groups_per_block = HYPRE_SPMV_BLOCKDIM / group_size;            \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_GPU_LAUNCH( (hypreGPUKernel_CSRMatvecShuffle<F, group_size, nv, T>),          \
                        gDim, bDim, nrows, rowid, idxstride_x, idxstride_y, vecstride_x,  \
                        vecstride_y, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );            \
   }                                                                                      \
   else if (avg_rownnz >= 8)                                                              \
   {                                                                                      \
      const HYPRE_Int group_size = 4;                                                     \
      const HYPRE_Int num_groups_per_block = HYPRE_SPMV_BLOCKDIM / group_size;            \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_GPU_LAUNCH( (hypreGPUKernel_CSRMatvecShuffle<F, group_size, nv, T>),          \
                        gDim, bDim, nrows, rowid, idxstride_x, idxstride_y, vecstride_x,  \
                        vecstride_y, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );            \
   }                                                                                      \
   else                                                                                   \
   {                                                                                      \
      const HYPRE_Int group_size = 4;                                                     \
      const HYPRE_Int num_groups_per_block = HYPRE_SPMV_BLOCKDIM / group_size;            \
      const dim3 gDim((nrows + num_groups_per_block - 1) / num_groups_per_block);         \
      HYPRE_GPU_LAUNCH( (hypreGPUKernel_CSRMatvecShuffle<F, group_size, nv, T>),          \
                        gDim, bDim, nrows, rowid, idxstride_x, idxstride_y, vecstride_x,  \
                        vecstride_y, alpha, d_ia, d_ja, d_a, d_x, beta, d_y );            \
   }

#endif
