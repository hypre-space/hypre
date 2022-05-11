/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fac.h"

#define AbsStencilShape(stencil, abs_shape) \
{\
   HYPRE_Int ii,jj,kk;\
   ii = hypre_IndexX(stencil);\
   jj = hypre_IndexY(stencil);\
   kk = hypre_IndexZ(stencil);\
   abs_shape= hypre_abs(ii) + hypre_abs(jj) + hypre_abs(kk); \
}

/*--------------------------------------------------------------------------
 * hypre_CF_StenBox: Given a cgrid_box, a fgrid_box, and a stencil_shape,
 * the stencil_shape direction. Returns an empty box if these two boxes
 * are not connected in the stencil_shape direction.
 *--------------------------------------------------------------------------*/
hypre_Box *
hypre_CF_StenBox( hypre_Box              *fgrid_box,
                  hypre_Box              *cgrid_box,
                  hypre_Index             stencil_shape,
                  hypre_Index             rfactors,
                  HYPRE_Int               ndim )
{
   hypre_Box              coarsen_box;
   hypre_Box              contracted_box;
   hypre_Box              extended_box;
   hypre_Box              intersect_box;
   hypre_Box             *stenbox;

   hypre_Box              shift_cbox, shift_ibox;
   hypre_Index            size_cbox, size_ibox;

   hypre_Index            temp_index;
   hypre_Index            shift_index;

   HYPRE_Int              i, remainder, intersect_size;

   hypre_ClearIndex(temp_index);
   stenbox = hypre_BoxCreate(ndim);

   hypre_BoxInit(&coarsen_box, ndim);
   hypre_BoxInit(&contracted_box, ndim);
   hypre_BoxInit(&extended_box, ndim);
   hypre_BoxInit(&intersect_box, ndim);
   hypre_BoxInit(&shift_cbox, ndim);
   hypre_BoxInit(&shift_ibox, ndim);

   /*--------------------------------------------------------------------------
    * Coarsen the fine box, extend it, and shift it to determine if there
    * is a reach between fgrid_box and cgrid_box in the stencil_shape direction.
    * Note: the fine_box may not align as the index rule assumes:
    *  [a_0,a_1,a_2]x[b_0,b_1,b_2], a_i= c_i*rfactors[i]
    *                               b_i= f_i*rfactors[i]+g_i, g_i= rfactors[i]-1.
    * When fine_box does not, then there must be a sibling box. fine_box
    * should be adjusted so that the flooring of the MapFineToCoarse does not
    * introduce extra coarse nodes in the coarsened box. Only the lower bound
    * needs to be adjusted.
    *--------------------------------------------------------------------------*/
   hypre_CopyBox(fgrid_box, &contracted_box);
   for (i = 0; i < ndim; i++)
   {
      remainder = hypre_BoxIMin(&contracted_box)[i] % rfactors[i];
      if (remainder)
      {
         hypre_BoxIMin(&contracted_box)[i] += rfactors[i] - remainder;
      }
   }

   hypre_StructMapFineToCoarse(hypre_BoxIMin(&contracted_box), temp_index,
                               rfactors, hypre_BoxIMin(&coarsen_box));
   hypre_StructMapFineToCoarse(hypre_BoxIMax(&contracted_box), temp_index,
                               rfactors, hypre_BoxIMax(&coarsen_box));

   hypre_ClearIndex(size_cbox);
   for (i = 0; i < ndim; i++)
   {
      size_cbox[i] = hypre_BoxSizeD(&coarsen_box, i) - 1;
   }

   /*---------------------------------------------------------------------
    * Extend the coarsened fgrid_box by one layer in each direction so
    * that actual cf interface is reached. If only coarsen_box were
    * extended, the actual cf interface may not be reached.
    *---------------------------------------------------------------------*/
   hypre_CopyBox(&coarsen_box, &extended_box);
   /*hypre_StructMapFineToCoarse(hypre_BoxIMin(fgrid_box), temp_index,
                               rfactors, hypre_BoxIMin(&extended_box));
   hypre_StructMapFineToCoarse(hypre_BoxIMax(fgrid_box), temp_index,
                               rfactors, hypre_BoxIMax(&extended_box));*/
   for (i = 0; i < ndim; i++)
   {
      hypre_BoxIMin(&extended_box)[i] -= 1;
      hypre_BoxIMax(&extended_box)[i] += 1;
   }

   hypre_IntersectBoxes(&extended_box, cgrid_box, &intersect_box);
   intersect_size = hypre_BoxVolume(&intersect_box);
   if (intersect_size == 0)
   {
      hypre_CopyBox(&intersect_box, stenbox);
      return stenbox;
   }

   hypre_ClearIndex(size_ibox);
   for (i = 0; i < ndim; i++)
   {
      size_ibox[i] = hypre_BoxSizeD(&intersect_box, i) - 1;
   }

   /*---------------------------------------------------------------------
    * To find the box extents that must be loop over, we need to take the
    * "opposite" stencil_shape and shift the coarsen and extended boxes.
    *---------------------------------------------------------------------*/
   hypre_SetIndex3(shift_index,
                   -size_ibox[0]*stencil_shape[0],
                   -size_ibox[1]*stencil_shape[1],
                   -size_ibox[2]*stencil_shape[2]);
   hypre_AddIndexes(shift_index, hypre_BoxIMin(&intersect_box), 3, hypre_BoxIMin(&shift_ibox));
   hypre_AddIndexes(shift_index, hypre_BoxIMax(&intersect_box), 3, hypre_BoxIMax(&shift_ibox));
   hypre_IntersectBoxes(&shift_ibox, &intersect_box, &shift_ibox);

   hypre_SetIndex3(shift_index,
                   -size_cbox[0]*stencil_shape[0],
                   -size_cbox[1]*stencil_shape[1],
                   -size_cbox[2]*stencil_shape[2]);
   hypre_AddIndexes(shift_index, hypre_BoxIMin(&coarsen_box), 3, hypre_BoxIMin(&shift_cbox));
   hypre_AddIndexes(shift_index, hypre_BoxIMax(&coarsen_box), 3, hypre_BoxIMax(&shift_cbox));
   hypre_IntersectBoxes(&shift_cbox, &coarsen_box, &shift_cbox);

   /*---------------------------------------------------------------------
    * shift_ibox & shift_cbox will contain the loop extents. Shifting
    * shift_cbox by -stencil_shape and then intersecting with shift_ibox
    * gives the exact extents.
    *---------------------------------------------------------------------*/
   hypre_SetIndex3(shift_index, -stencil_shape[0], -stencil_shape[1], -stencil_shape[2]);
   hypre_AddIndexes(shift_index, hypre_BoxIMin(&shift_cbox), 3, hypre_BoxIMin(&shift_cbox));
   hypre_AddIndexes(shift_index, hypre_BoxIMax(&shift_cbox), 3, hypre_BoxIMax(&shift_cbox));
   hypre_IntersectBoxes(&shift_cbox, &shift_ibox, stenbox);

   return stenbox;
}
