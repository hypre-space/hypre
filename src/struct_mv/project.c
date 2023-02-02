/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Projection routines.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ProjectBox:
 *   Projects a box onto a strided index space that contains the
 *   index `index' and has stride `stride'.
 *
 *   Note: An "empty" projection is represented by a box with volume 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBox( hypre_Box    *box,
                  hypre_Index   index,
                  hypre_Index   stride )
{
   HYPRE_Int  i, s, d, hl, hu, kl, ku, ndim = hypre_BoxNDim(box);

   /*------------------------------------------------------
    * project in all ndim dimensions
    *------------------------------------------------------*/

   for (d = 0; d < ndim; d++)
   {

      i = hypre_IndexD(index, d);
      s = hypre_IndexD(stride, d);

      hl = hypre_BoxIMinD(box, d) - i;
      hu = hypre_BoxIMaxD(box, d) - i;

      if ( hl <= 0 )
      {
         kl = (HYPRE_Int) (hl / s);
      }
      else
      {
         kl = (HYPRE_Int) ((hl + (s - 1)) / s);
      }

      if ( hu >= 0 )
      {
         ku = (HYPRE_Int) (hu / s);
      }
      else
      {
         ku = (HYPRE_Int) ((hu - (s - 1)) / s);
      }

      hypre_BoxIMinD(box, d) = i + kl * s;
      hypre_BoxIMaxD(box, d) = i + ku * s;

   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArray:
 *
 *   Note: The dimensions of the modified box array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBoxArray( hypre_BoxArray  *box_array,
                       hypre_Index      index,
                       hypre_Index      stride    )
{
   hypre_Box  *box;
   HYPRE_Int   i;

   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_ProjectBox(box, index, stride);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArrayArray:
 *
 *   Note: The dimensions of the modified box array-array are not changed.
 *   So, it is possible to have boxes with volume 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ProjectBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_Index           index,
                            hypre_Index           stride          )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
   HYPRE_Int        i, j;

   hypre_ForBoxArrayI(i, box_array_array)
   {
      box_array = hypre_BoxArrayArrayBoxArray(box_array_array, i);
      hypre_ForBoxI(j, box_array)
      {
         box = hypre_BoxArrayBox(box_array, j);
         hypre_ProjectBox(box, index, stride);
      }
   }

   return hypre_error_flag;
}

