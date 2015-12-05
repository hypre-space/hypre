/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
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

HYPRE_Int
hypre_ProjectBox( hypre_Box    *box,
                  hypre_Index   index,
                  hypre_Index   stride )
{
   HYPRE_Int  i, s, d, hl, hu, kl, ku;
   HYPRE_Int  ierr = 0;

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
         kl = (HYPRE_Int) (hl / s);
      else
         kl = (HYPRE_Int) ((hl + (s-1)) / s);

      if ( hu >= 0 )
         ku = (HYPRE_Int) (hu / s);
      else
         ku = (HYPRE_Int) ((hu - (s-1)) / s);

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

HYPRE_Int
hypre_ProjectBoxArray( hypre_BoxArray  *box_array,
                       hypre_Index      index,
                       hypre_Index      stride    )
{
   hypre_Box  *box;
   HYPRE_Int   i;
   HYPRE_Int   ierr = 0;

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

HYPRE_Int
hypre_ProjectBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_Index           index,
                            hypre_Index           stride          )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;
   HYPRE_Int        i, j;
   HYPRE_Int        ierr = 0;

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

