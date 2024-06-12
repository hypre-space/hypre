/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_StructData class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * Copy struct data with possibly different data spaces.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructDataCopy( HYPRE_Complex   *fr_data,        /* from */
                      hypre_BoxArray  *fr_data_space,
                      HYPRE_Int       *fr_ids,
                      HYPRE_Complex   *to_data,        /* to */
                      hypre_BoxArray  *to_data_space,
                      HYPRE_Int       *to_ids,
                      HYPRE_Int        ndim,
                      HYPRE_Int        nval )
{
   HYPRE_Int       fr_nboxes = hypre_BoxArraySize(fr_data_space);
   HYPRE_Int       to_nboxes = hypre_BoxArraySize(to_data_space);
   hypre_Box      *fr_data_box, *to_data_box;
   HYPRE_Int       fr_data_off,  to_data_off;
   HYPRE_Int       fr_data_vol,  to_data_vol;
   HYPRE_Complex  *fr_dp, *to_dp;
   HYPRE_Int       fb, tb, val;
   hypre_Box      *int_box;
   hypre_IndexRef  start;
   hypre_Index     stride, loop_size;

   if (fr_data == NULL || to_data == NULL)
   {
      return hypre_error_flag;
   }

   hypre_SetIndex(stride, 1);
   int_box = hypre_BoxCreate(ndim);

   fb = 0;
   fr_data_off = 0;
   to_data_off = 0;
   for (tb = 0; tb < to_nboxes; tb++)
   {
      to_data_box = hypre_BoxArrayBox(to_data_space, tb);
      to_data_vol = hypre_BoxVolume(to_data_box);

      while ((fb < fr_nboxes) && (fr_ids[fb] < to_ids[tb]))
      {
         fr_data_box = hypre_BoxArrayBox(fr_data_space, fb);
         fr_data_vol = hypre_BoxVolume(fr_data_box);
         fr_data_off += nval * fr_data_vol;
         fb++;
      }
      if ((fb < fr_nboxes) && (fr_ids[fb] == to_ids[tb]))
      {
         fr_data_box = hypre_BoxArrayBox(fr_data_space, fb);
         fr_data_vol = hypre_BoxVolume(fr_data_box);

         hypre_IntersectBoxes(fr_data_box, to_data_box, int_box);

         start = hypre_BoxIMin(int_box);
         hypre_BoxGetSize(int_box, loop_size);

         for (val = 0; val < nval; val++)
         {
            fr_dp = fr_data + fr_data_off + val * fr_data_vol;
            to_dp = to_data + to_data_off + val * to_data_vol;

#define DEVICE_VAR is_device_ptr(fr_dp, to_dp)
            hypre_BoxLoop2Begin(ndim, loop_size,
                                fr_data_box, start, stride, fi,
                                to_data_box, start, stride, ti);
            {
               to_dp[ti] = fr_dp[fi];
            }
            hypre_BoxLoop2End(fi, ti);
#undef DEVICE_VAR
         }
      }

      to_data_off += nval * to_data_vol;
   }

   hypre_BoxDestroy(int_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute num_ghost array from stencil
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructNumGhostFromStencil( hypre_StructStencil  *stencil,
                                 HYPRE_Int           **num_ghost_ptr )
{
   HYPRE_Int      *num_ghost;
   HYPRE_Int       ndim          = hypre_StructStencilNDim(stencil);
   hypre_Index    *stencil_shape = hypre_StructStencilShape(stencil);
   hypre_IndexRef  stencil_offset;
   HYPRE_Int       s, d, m;

   num_ghost = hypre_CTAlloc(HYPRE_Int, 2 * ndim, HYPRE_MEMORY_HOST);

   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      stencil_offset = stencil_shape[s];

      for (d = 0; d < ndim; d++)
      {
         m = stencil_offset[d];

         if (m < 0)
         {
            num_ghost[2 * d]     = hypre_max(num_ghost[2 * d],    -m);
         }
         else if (m > 0)
         {
            num_ghost[2 * d + 1] = hypre_max(num_ghost[2 * d + 1], m);
         }
      }
   }

   *num_ghost_ptr = num_ghost;

   return hypre_error_flag;
}
