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
#define HYPRE_SPMV_FILL_STRICT_LOWER -2
#define HYPRE_SPMV_FILL_LOWER -1
#define HYPRE_SPMV_FILL_WHOLE 0
#define HYPRE_SPMV_FILL_UPPER 1
#define HYPRE_SPMV_FILL_STRICT_UPPER 2

#define HYPRE_SPMV_ADD_SUM(p, nv)                                                         \
{                                                                                         \
   const HYPRE_Int col = read_only_load(&d_ja[p]);                                        \
   if (F == HYPRE_SPMV_FILL_WHOLE)                                                        \
   {                                                                                      \
      const T val = d_a ? read_only_load(&d_a[p]) : T(1);                                 \
      for (HYPRE_Int i = 0; i < nv; i++)                                                  \
      {                                                                                   \
         sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);       \
      }                                                                                   \
   }                                                                                      \
   else if (F == HYPRE_SPMV_FILL_LOWER)                                                   \
   {                                                                                      \
      if (col <= grid_group_id)                                                           \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < nv; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == HYPRE_SPMV_FILL_UPPER)                                                   \
   {                                                                                      \
      if (col >= grid_group_id)                                                           \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < nv; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == HYPRE_SPMV_FILL_STRICT_LOWER)                                            \
   {                                                                                      \
      if (col < grid_group_id)                                                            \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < nv; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
   else if (F == HYPRE_SPMV_FILL_STRICT_UPPER)                                            \
   {                                                                                      \
      if (col > grid_group_id)                                                            \
      {                                                                                   \
         const T val = d_a ? read_only_load(&d_a[p]) : T(1);                              \
         for (HYPRE_Int i = 0; i < nv; i++)                                               \
         {                                                                                \
            sum[i] += val * read_only_load(&d_x[col * idxstride_x + i * vecstride_x]);    \
         }                                                                                \
      }                                                                                   \
   }                                                                                      \
}

#endif
