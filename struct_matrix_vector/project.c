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
 *
 *   Note: An SBox is returned regardless of the outcome of the
 *   projection.  So, it is possible to return an SBox with volume 0.
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
      zzz_BoxIMaxD(new_box, d) = iu - s;
   }

   /*------------------------------------------------------
    * set the strides
    *------------------------------------------------------*/

   new_stride = zzz_NewIndex();
   zzz_CopyIndex(stride, new_stride);

   new_sbox = zzz_NewSBox(new_box, new_stride);

   return new_sbox;
}

/*--------------------------------------------------------------------------
 * zzz_ProjectBoxArray:
 *
 *   Note: The dimensions of the returned SBoxArray are the same as
 *   the input argument `box_array'.  So, it is possible for the
 *   returned SBoxArray to contain SBoxes with volume 0.
 *--------------------------------------------------------------------------*/

zzz_SBoxArray *
zzz_ProjectBoxArray( zzz_BoxArray  *box_array,
                     zzz_Index     *index,
                     zzz_Index     *stride    )
{
   zzz_SBoxArray       *new_sbox_array;
   zzz_SBox            *new_sbox;

   zzz_Box             *box;

   int                  i;

   new_sbox_array = zzz_NewSBoxArray();

   zzz_ForBoxI(i, box_array)
   {
      box      = zzz_BoxArrayBox(box_array, i);
      new_sbox = zzz_ProjectBox(box, index, stride);
      zzz_AppendSBox(new_sbox, new_sbox_array);
   }

   return new_sbox_array;
}

/*--------------------------------------------------------------------------
 * zzz_ProjectBoxArrayArray:
 *
 *   Note: The dimensions of the returned SBoxArrayArray are the same as
 *   the input argument `box_array_array'.  So, it is possible for the
 *   returned SBoxArrayArray to contain SBoxes with volume 0.
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_ProjectBoxArrayArray( zzz_BoxArrayArray  *box_array_array,
                          zzz_Index          *index,
                          zzz_Index          *stride          )
{
   zzz_SBoxArrayArray  *new_sbox_array_array;
   zzz_SBoxArray       *new_sbox_array;

   zzz_BoxArray        *box_array;

   int                  i;

   new_sbox_array_array =
      zzz_NewSBoxArrayArray(zzz_BoxArrayArraySize(box_array_array));

   zzz_ForBoxArrayI(i, box_array_array)
   {
      box_array      = zzz_BoxArrayArrayBoxArray(box_array_array, i);
      new_sbox_array = zzz_ProjectBoxArray(box_array, index, stride);

      zzz_FreeSBoxArray(zzz_SBoxArrayArraySBoxArray(new_sbox_array_array, i));
      zzz_SBoxArrayArraySBoxArray(new_sbox_array_array, i) = new_sbox_array;
   }

   return new_sbox_array_array;
}


#if 0
/*--------------------------------------------------------------------------
 * ProjectRBPoint:
 *   Project a box_array_array onto a red or black set of indices.
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_ProjectRBPoint( zzz_BoxArrayArray *box_array_array,
                    zzz_Index         *rb[4]           )
{
   zzz_SBoxArrayArray *new_sbox_array_array;
   zzz_SBoxArrayArray *tmp_sbox_array_array;

   zzz_Index          *stride;

   int                 i;

   stride = zzz_NewIndex();
   zzz_SetIndex(stride, 2, 2, 2);

   new_sbox_array_array =
      zzz_NewSBoxArrayArray(zzz_BoxArrayArraySize(box_array_array));

   for (i = 0; i < 4; i++)
   {
      tmp_sbox_array_array =
         zzz_ProjectBoxArrayArray(box_array_array, rb[i], stride);
      zzz_AppendSBoxArrayArray(tmp_sbox_array_array, new_sbox_array_array);
      zzz_FreeSBoxArrayArrayShell(tmp_sbox_array_array);
   }
   
   return new_sbox_array_array;
}
#endif
