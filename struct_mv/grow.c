/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routines for "growing" boxes.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_GrowBoxByStencil:
 *    The argument `transpose' is a boolean that indicates whether
 *    or not to use the transpose of the stencil.
 *--------------------------------------------------------------------------*/

zzz_BoxArray *
zzz_GrowBoxByStencil( zzz_Box           *box,
                      zzz_StructStencil *stencil,
                      int                transpose )
{
   zzz_BoxArray   *grow_box_array;
                  
   zzz_BoxArray   *shift_box_array;
   zzz_Box        *shift_box;

   zzz_Index     **stencil_shape;

   int             s, d;

   shift_box = zzz_DuplicateBox(box);
   stencil_shape = zzz_StructStencilShape(stencil);

   shift_box_array = zzz_NewBoxArray();
   for (s = 0; s < zzz_StructStencilSize(stencil); s++)
   {
      if (transpose)
         for (d = 0; d < 3; d++)
         {
            zzz_BoxIMinD(shift_box, d) =
               zzz_BoxIMinD(box, d) - zzz_IndexD(stencil_shape[s], d);
            zzz_BoxIMaxD(shift_box, d) =
               zzz_BoxIMaxD(box, d) - zzz_IndexD(stencil_shape[s], d);
         }
      else
         for (d = 0; d < 3; d++)
         {
            zzz_BoxIMinD(shift_box, d) =
               zzz_BoxIMinD(box, d) + zzz_IndexD(stencil_shape[s], d);
            zzz_BoxIMaxD(shift_box, d) =
               zzz_BoxIMaxD(box, d) + zzz_IndexD(stencil_shape[s], d);
         }

      zzz_AppendBox(shift_box, shift_box_array);
   }

   grow_box_array = zzz_UnionBoxArray(shift_box_array);
   zzz_FreeBoxArray(shift_box_array);

   return grow_box_array;
}

/*--------------------------------------------------------------------------
 * zzz_GrowBoxArrayByStencil:
 *--------------------------------------------------------------------------*/

zzz_BoxArrayArray *
zzz_GrowBoxArrayByStencil( zzz_BoxArray      *box_array,
                           zzz_StructStencil *stencil,
                           int                transpose )
{
   zzz_BoxArrayArray     *grow_box_array_array;

   int                    i;

   grow_box_array_array = zzz_NewBoxArrayArray(zzz_BoxArraySize(box_array));

   zzz_ForBoxI(i, box_array)
   {
      zzz_FreeBoxArray(zzz_BoxArrayArrayBoxArray(grow_box_array_array, i));
      zzz_BoxArrayArrayBoxArray(grow_box_array_array, i) =
         zzz_GrowBoxByStencil(zzz_BoxArrayBox(box_array, i),
                              stencil, transpose);
   }

   return grow_box_array_array;
}
