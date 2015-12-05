/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Routines for "growing" boxes.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_GrowBoxByStencil:
 *    The argument `transpose' is a boolean that indicates whether
 *    or not to use the transpose of the stencil.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_GrowBoxByStencil( hypre_Box           *box,
                        hypre_StructStencil *stencil,
                        int                  transpose )
{
   hypre_BoxArray   *grow_box_array;
                  
   hypre_BoxArray   *shift_box_array;
   hypre_Box        *shift_box;

   hypre_Index      *stencil_shape;

   int               s, d;

   stencil_shape = hypre_StructStencilShape(stencil);

   shift_box_array = hypre_BoxArrayCreate(hypre_StructStencilSize(stencil));
   shift_box = hypre_BoxCreate();
   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      if (transpose)
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(shift_box, d) =
               hypre_BoxIMinD(box, d) - hypre_IndexD(stencil_shape[s], d);
            hypre_BoxIMaxD(shift_box, d) =
               hypre_BoxIMaxD(box, d) - hypre_IndexD(stencil_shape[s], d);
         }
      else
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(shift_box, d) =
               hypre_BoxIMinD(box, d) + hypre_IndexD(stencil_shape[s], d);
            hypre_BoxIMaxD(shift_box, d) =
               hypre_BoxIMaxD(box, d) + hypre_IndexD(stencil_shape[s], d);
         }

      hypre_CopyBox(shift_box, hypre_BoxArrayBox(shift_box_array, s));
   }
   hypre_BoxDestroy(shift_box);

   hypre_UnionBoxes(shift_box_array);
   grow_box_array = shift_box_array;

   return grow_box_array;
}

/*--------------------------------------------------------------------------
 * hypre_GrowBoxArrayByStencil:
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_GrowBoxArrayByStencil( hypre_BoxArray      *box_array,
                             hypre_StructStencil *stencil,
                             int                  transpose )
{
   hypre_BoxArrayArray     *grow_box_array_array;

   int                      i;

   grow_box_array_array =
      hypre_BoxArrayArrayCreate(hypre_BoxArraySize(box_array));

   hypre_ForBoxI(i, box_array)
      {
         hypre_BoxArrayDestroy(
            hypre_BoxArrayArrayBoxArray(grow_box_array_array, i));
         hypre_BoxArrayArrayBoxArray(grow_box_array_array, i) =
            hypre_GrowBoxByStencil(hypre_BoxArrayBox(box_array, i),
                                   stencil, transpose);
      }

   return grow_box_array_array;
}
