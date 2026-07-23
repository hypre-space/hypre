/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/* Need struct wrapper to pass hypre_Index as argument to GPU kernel */
typedef struct hypre_IndexDevice_struct
{
   hypre_Index idx;
} hypre_IndexDevice;

__global__ void
hypreGPUKernel_BoxRanksToIndexes( hypre_DeviceItem  &item,
                                  HYPRE_Int          ndim,
                                  hypre_IndexDevice  box_imin,
                                  hypre_IndexDevice  box_size,
                                  HYPRE_Int          box_volume,
                                  HYPRE_Int          num_ranks,
                                  HYPRE_Int         *ranks,
                                  HYPRE_Int        **indexes )
{
   HYPRE_Int  d, r, s;
   HYPRE_Int  i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < num_ranks)
   {
      r = ranks[i];
      s = box_volume;
      for (d = ndim - 1; d >= 0; d--)
      {
         s = s / box_size.idx[d];
         indexes[d][i] = (r / s) + box_imin.idx[d];
         r = r % s;
      }
   }
}

HYPRE_Int
hypre_BoxRanksToIndexesDevice( hypre_Box   *box,
                               HYPRE_Int    num_ranks,
                               HYPRE_Int   *ranks,
                               HYPRE_Int ***indexes_ptr )
{
   HYPRE_Int           d;
   hypre_IndexDevice   box_size_d;
   hypre_IndexDevice   box_imin_d;

   HYPRE_Int         **indexes;
   HYPRE_Int         **indexes_d;

   HYPRE_Int           ndim          = hypre_BoxNDim(box);
   HYPRE_Int           box_volume    = hypre_BoxVolume(box);

   const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   const dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_ranks, "thread", bDim);

   if (*indexes_ptr == NULL)
   {
      indexes = hypre_TAlloc(HYPRE_Int*, ndim, HYPRE_MEMORY_HOST);
      for (d = 0; d < ndim; d++)
      {
         hypre_IndexD(box_size_d.idx, d) = hypre_BoxSizeD(box, d);
         indexes[d] = hypre_TAlloc(HYPRE_Int, num_ranks, HYPRE_MEMORY_DEVICE);
      }
   }
   else
   {
      indexes = *indexes_ptr;
   }
   indexes_d = hypre_TAlloc(HYPRE_Int*, ndim, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(indexes_d, indexes, HYPRE_Int*, ndim, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   hypre_CopyIndex(hypre_BoxIMin(box), box_imin_d.idx);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_BoxRanksToIndexes, gDim, bDim, ndim,
                     box_imin_d, box_size_d, box_volume, num_ranks, ranks, indexes_d);

   *indexes_ptr = indexes;

   hypre_TFree(indexes_d, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif
