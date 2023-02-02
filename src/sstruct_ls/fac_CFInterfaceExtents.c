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
 * hypre_CFInterfaceExtents: Given a cgrid_box, a fgrid_box, and stencils,
 * find the extents of the C/F interface (interface nodes in the C box).
 * Boxes corresponding to stencil shifts are stored in the first stencil_size
 * boxes, and the union of these are appended to the end of the returned
 * box_array.
 *--------------------------------------------------------------------------*/
hypre_BoxArray *
hypre_CFInterfaceExtents( hypre_Box              *fgrid_box,
                          hypre_Box              *cgrid_box,
                          hypre_StructStencil    *stencils,
                          hypre_Index             rfactors )
{

   hypre_BoxArray        *stencil_box_extents;
   hypre_BoxArray        *union_boxes;
   hypre_Box             *cfine_box;
   hypre_Box             *box;

   hypre_Index            stencil_shape, cstart, zero_index, neg_index;
   HYPRE_Int              stencil_size;
   HYPRE_Int              abs_stencil;

   HYPRE_Int              ndim = hypre_StructStencilNDim(stencils);
   HYPRE_Int              i, j;

   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(neg_index);
   for (i = 0; i < ndim; i++)
   {
      neg_index[i] = -1;
   }
   hypre_CopyIndex(hypre_BoxIMin(cgrid_box), cstart);

   stencil_size       = hypre_StructStencilSize(stencils);
   stencil_box_extents = hypre_BoxArrayCreate(stencil_size, ndim);
   union_boxes        = hypre_BoxArrayCreate(0, ndim);

   for (i = 0; i < stencil_size; i++)
   {
      hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape);
      AbsStencilShape(stencil_shape, abs_stencil);

      if (abs_stencil)  /* only do if not the centre stencil */
      {
         cfine_box = hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape, rfactors,
                                      ndim);

         if ( hypre_BoxVolume(cfine_box) )
         {
            hypre_AppendBox(cfine_box, union_boxes);
            hypre_CopyBox(cfine_box, hypre_BoxArrayBox(stencil_box_extents, i));
            for (j = 0; j < ndim; j++)
            {
               hypre_BoxIMin(cfine_box)[j] -=  cstart[j];
               hypre_BoxIMax(cfine_box)[j] -=  cstart[j];
            }
            hypre_CopyBox(cfine_box, hypre_BoxArrayBox(stencil_box_extents, i));
         }

         else
         {
            hypre_BoxSetExtents(hypre_BoxArrayBox(stencil_box_extents, i),
                                zero_index, neg_index);
         }

         hypre_BoxDestroy(cfine_box);
      }

      else /* centre */
      {
         hypre_BoxSetExtents(hypre_BoxArrayBox(stencil_box_extents, i),
                             zero_index, neg_index);
      }
   }

   /*--------------------------------------------------------------------------
    * Union the stencil_box_extents to get the full CF extents and append to
    * the end of the stencil_box_extents BoxArray. Then shift the unioned boxes
    * by cstart.
    *--------------------------------------------------------------------------*/
   if (hypre_BoxArraySize(union_boxes) > 1)
   {
      hypre_UnionBoxes(union_boxes);
   }

   hypre_ForBoxI(i, union_boxes)
   {
      hypre_AppendBox(hypre_BoxArrayBox(union_boxes, i), stencil_box_extents);
   }
   hypre_BoxArrayDestroy(union_boxes);

   for (i = stencil_size; i < hypre_BoxArraySize(stencil_box_extents); i++)
   {
      box = hypre_BoxArrayBox(stencil_box_extents, i);
      for (j = 0; j < ndim; j++)
      {
         hypre_BoxIMin(box)[j] -=  cstart[j];
         hypre_BoxIMax(box)[j] -=  cstart[j];
      }
   }

   return stencil_box_extents;
}

HYPRE_Int
hypre_CFInterfaceExtents2( hypre_Box              *fgrid_box,
                           hypre_Box              *cgrid_box,
                           hypre_StructStencil    *stencils,
                           hypre_Index             rfactors,
                           hypre_BoxArray         *cf_interface )
{

   hypre_BoxArray        *stencil_box_extents;
   hypre_BoxArray        *union_boxes;
   hypre_Box             *cfine_box;

   hypre_Index            stencil_shape, zero_index, neg_index;
   HYPRE_Int              stencil_size;
   HYPRE_Int              abs_stencil;

   HYPRE_Int              ndim = hypre_StructStencilNDim(stencils);

   HYPRE_Int              i;
   HYPRE_Int              ierr = 0;

   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(neg_index);
   for (i = 0; i < ndim; i++)
   {
      neg_index[i] = -1;
   }

   stencil_size       = hypre_StructStencilSize(stencils);
   stencil_box_extents = hypre_BoxArrayCreate(stencil_size, ndim);
   union_boxes        = hypre_BoxArrayCreate(0, ndim);

   for (i = 0; i < stencil_size; i++)
   {
      hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape);
      AbsStencilShape(stencil_shape, abs_stencil);

      if (abs_stencil)  /* only do if not the centre stencil */
      {
         cfine_box = hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape,
                                      rfactors, ndim);

         if ( hypre_BoxVolume(cfine_box) )
         {
            hypre_AppendBox(cfine_box, union_boxes);
            hypre_CopyBox(cfine_box, hypre_BoxArrayBox(stencil_box_extents, i));
         }

         else
         {
            hypre_BoxSetExtents(hypre_BoxArrayBox(stencil_box_extents, i),
                                zero_index, neg_index);
         }

         hypre_BoxDestroy(cfine_box);
      }

      else /* centre */
      {
         hypre_BoxSetExtents(hypre_BoxArrayBox(stencil_box_extents, i),
                             zero_index, neg_index);
      }
   }

   /*--------------------------------------------------------------------------
    * Union the stencil_box_extents to get the full CF extents and append to
    * the end of the stencil_box_extents BoxArray.
    *--------------------------------------------------------------------------*/
   if (hypre_BoxArraySize(union_boxes) > 1)
   {
      hypre_UnionBoxes(union_boxes);
   }

   hypre_ForBoxI(i, union_boxes)
   {
      hypre_AppendBox(hypre_BoxArrayBox(union_boxes, i), stencil_box_extents);
   }
   hypre_AppendBoxArray(stencil_box_extents, cf_interface);

   hypre_BoxArrayDestroy(union_boxes);
   hypre_BoxArrayDestroy(stencil_box_extents);

   return ierr;
}
