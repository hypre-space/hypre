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
 * Projection routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ProjectBox:
 *   Projects a box onto a strided index space that contains the
 *   index `index' and has stride `stride'.
 *
 *   Note: An "empty" projection is represented by a box with volume 0.
 *--------------------------------------------------------------------------*/

int
hypre_ProjectBox( hypre_Box    *box,
                  hypre_Index   index,
                  hypre_Index   stride )
{
   int  i, s, d, hl, hu, kl, ku;
   int  ierr = 0;

   /*------------------------------------------------------
    * project in all 3 dimensions
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {

      i = hypre_IndexD(index, d);
      s = hypre_IndexD(stride, d);

      hl = hypre_BoxIMinD(box, d) - i;
      hu = hypre_BoxIMaxD(box, d) - i;

      if ( hl <= 0 )
         kl = (int) (hl / s);
      else
         kl = (int) ((hl + (s-1)) / s);

      if ( hu >= 0 )
         ku = (int) (hu / s);
      else
         ku = (int) ((hu - (s-1)) / s);

      hypre_BoxIMinD(box, d) = i + kl * s;
      hypre_BoxIMaxD(box, d) = i + ku * s;

   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArray:
 *
 *   Note: The dimensions of the modified box array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

int
hypre_ProjectBoxArray( hypre_BoxArray  *box_array,
                       hypre_Index      index,
                       hypre_Index      stride    )
{
   hypre_Box  *box;
   int         i;
   int         ierr = 0;

   hypre_ForBoxI(i, box_array)
      {
         box = hypre_BoxArrayBox(box_array, i);
         hypre_ProjectBox(box, index, stride);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArrayArray:
 *
 *   Note: The dimensions of the modified box array-array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

int
hypre_ProjectBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_Index           index,
                            hypre_Index           stride          )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
   int              i, j;
   int              ierr = 0;

   hypre_ForBoxArrayI(i, box_array_array)
      {
         box_array = hypre_BoxArrayArrayBoxArray(box_array_array, i);
         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               hypre_ProjectBox(box, index, stride);
            }
      }

   return ierr;
}

