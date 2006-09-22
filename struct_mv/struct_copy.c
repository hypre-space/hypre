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
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Structured copy routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_StructCopy
 *--------------------------------------------------------------------------*/

int
hypre_StructCopy( hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   int ierr = 0;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   int              xi;
   int              yi;
                   
   double          *xp;
   double          *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   int              i;
   int              loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,yi
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               yp[yi] = xp[xi];
            }
         hypre_BoxLoop2End(xi, yi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructPartialCopy: copy only the components on a subset of the grid.
 * A BoxArrayArray of boxes are needed- for each box of x, only an array
 * of subboxes (i.e., a boxarray for each box of x) are copied.
 *--------------------------------------------------------------------------*/

int
hypre_StructPartialCopy( hypre_StructVector  *x,
                         hypre_StructVector  *y,    
                         hypre_BoxArrayArray *array_boxes )
{
   int ierr = 0;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   int              xi;
   int              yi;

   double          *xp;
   double          *yp;

   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;

   int              i, j ;
   int              loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   hypre_ForBoxArrayI(i, array_boxes)
      {
         boxes = hypre_BoxArrayArrayBoxArray(array_boxes, i);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         /* array of sub_boxes of box_i of the vector */
         hypre_ForBoxI(j, boxes)
         {
            box = hypre_BoxArrayBox(boxes, j);

            start = hypre_BoxIMin(box);
            hypre_BoxGetSize(box, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                x_data_box, start, unit_stride, xi,
                                y_data_box, start, unit_stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,yi
#include "hypre_box_smp_forloop.h"
            hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
              {
                  yp[yi] = xp[xi];
              }
            hypre_BoxLoop2End(xi, yi);
         }
      }

   return ierr;
}
