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
 * Projection routines.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_ProjectBox:
 *   Projects a box onto a strided index space that contains the
 *   index `index' and has stride `stride'.
 *--------------------------------------------------------------------------*/

zzz_SBox *
zzz_ProjectBox( zzz_Box    *box,
                zzz_Index  *index,
                zzz_Index  *stride )
{
   zzz_SBox  *new_sbox;
   zzz_Box   *new_box;
   zzz_Index *new_stride;

   int        il, iu, i, s, d;

   /*------------------------------------------------------
    * project in all 3 dimensions
    *------------------------------------------------------*/

   new_box = zzz_DuplicateBox(box);

   for (d = 0; d < 3; d++)
   {
      i = zzz_IndexD(index, d);
      s = zzz_IndexD(stride, d);

      il = zzz_BoxIMinD(new_box, d) + i;
      iu = zzz_BoxIMaxD(new_box, d) + i + 1;

      il = ((int) ((il + (s-1)) / s)) * s - i;
      iu = ((int) ((iu + (s-1)) / s)) * s - i;

      zzz_BoxIMinD(new_box, d) = il;
      zzz_BoxIMaxD(new_box, d) = iu - 1;
   }

   /*------------------------------------------------------
    * set the strides
    *------------------------------------------------------*/

   new_stride = zzz_NewIndex();
   for (d = 0; d < 3; d++)
      zzz_IndexD(new_stride, d) = zzz_IndexD(stride, d);

   new_sbox = zzz_NewSBox(new_box, new_stride);

   return new_sbox;
}

/*--------------------------------------------------------------------------
 * zzz_ProjectBoxArrayArray:
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_ProjectBoxArrayArray( zzz_BoxArrayArray  *box_array_array,
                          zzz_Index          *index,
                          zzz_Index          *stride          )
{
   SBoxArrayArray  *new_sbox_array_array;
   SBoxArray       *new_sbox_array;
   SBox            *new_sbox;

   BoxArray        *box_array;
   Box             *box;

   int              i, j;

   new_sbox_array_array =
      zzz_NewSBoxArrayArray(zzz_BoxArrayArraySize(box_array_array));

   zzz_ForBoxArrayI(i, box_array_array)
      {
         box_array      = zzz_BoxArrayArrayBoxArray(box_array_array, i);
         new_sbox_array = zzz_SBoxArrayArraySBoxArray(new_sbox_array_array, i);

         zzz_ForBoxI(j, box_array)
            {
               box      = zzz_BoxArrayBox(box_array, j);
               new_sbox = zzz_ProjectBox(box, index, stride);
               if (zzz_SBoxTotalSize(new_sbox))
                  zzz_AppendSBox(new_sbox, new_sbox_array);
               else
                  zzz_FreeSBox(new_sbox);
            }
      }

   return new_sbox_array_array;
}








/*--------------------------------------------------------------------------
 * ProjectRBPoint:
 *   Project a box_array_array onto a red or black set of indices.
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_ProjectRBPoint( zzz_BoxArrayArray *box_array_array,
                    zzz_Index          rb[4]           )
{
   zzz_SBoxArrayArray *new_sbox_array_array;
   zzz_SBoxArrayArray *tmp_sbox_array_array;

   zzz_Index          *stride;

   int                 i, j;

   stride = zzz_NewIndex();
   for (d = 0; d < 3; d++)
      zzz_IndexD(stride, d) = 2;

   new_sbox_array_array =
      zzz_NewSBoxArrayArray(zzz_BoxArrayArraySize(box_array_array));

   for (i = 0; i < 4; i++)
   {
      tmp_sbox_array_array =
         zzz_ProjectBoxArrayArray(box_array_array, rb[i], stride);
      zzz_AppendSBoxArrayArray(tmp_sbox_array_array, new_sbox_array_array);
      zzz_FreeSBoxArrayArrayShell(tmp_sbox_array_array);
   }
   
   return new;
}

