/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "headers.h"

#define AbsStencilShape(stencil, abs_shape) \
{\
   int ii,jj,kk;\
   ii = hypre_IndexX(stencil);\
   jj = hypre_IndexY(stencil);\
   kk = hypre_IndexZ(stencil);\
   abs_shape= abs(ii) + abs(jj) + abs(kk); \
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
   int                    stencil_size;
   int                    abs_stencil;

   int                    ndim= hypre_StructStencilDim(stencils);
   int                    i, j;
    
   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(neg_index);
   for (i= 0; i< ndim; i++)
   {
      neg_index[i]= -1;
   }
   hypre_CopyIndex(hypre_BoxIMin(cgrid_box), cstart);

   stencil_size       = hypre_StructStencilSize(stencils);
   stencil_box_extents= hypre_BoxArrayCreate(stencil_size);
   union_boxes        = hypre_BoxArrayCreate(0);

   for (i= 0; i< stencil_size; i++)
   {
       hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape);
       AbsStencilShape(stencil_shape, abs_stencil);

       if (abs_stencil)  /* only do if not the centre stencil */
       {
          cfine_box= hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape, rfactors,
                                      ndim);

          if ( hypre_BoxVolume(cfine_box) )
          {
             hypre_AppendBox(cfine_box, union_boxes);
             hypre_CopyBox(cfine_box, hypre_BoxArrayBox(stencil_box_extents, i));
             for (j= 0; j< ndim; j++)
             {
                hypre_BoxIMin(cfine_box)[j]-=  cstart[j];
                hypre_BoxIMax(cfine_box)[j]-=  cstart[j];
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
      
   for (i= stencil_size; i< hypre_BoxArraySize(stencil_box_extents); i++)
   {
      box= hypre_BoxArrayBox(stencil_box_extents, i);
      for (j= 0; j< ndim; j++)
      {
         hypre_BoxIMin(box)[j]-=  cstart[j];
         hypre_BoxIMax(box)[j]-=  cstart[j];
      }
   }

   return stencil_box_extents;
}

int
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
   int                    stencil_size;
   int                    abs_stencil;

   int                    ndim= hypre_StructStencilDim(stencils);

   int                    i;
   int                    ierr= 0;
    
   hypre_ClearIndex(zero_index);
   hypre_ClearIndex(neg_index);
   for (i= 0; i< ndim; i++)
   {
      neg_index[i]= -1;
   }

   stencil_size       = hypre_StructStencilSize(stencils);
   stencil_box_extents= hypre_BoxArrayCreate(stencil_size);
   union_boxes        = hypre_BoxArrayCreate(0);

   for (i= 0; i< stencil_size; i++)
   {
       hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape);
       AbsStencilShape(stencil_shape, abs_stencil);

       if (abs_stencil)  /* only do if not the centre stencil */
       {
          cfine_box= hypre_CF_StenBox(fgrid_box, cgrid_box, stencil_shape, 
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
