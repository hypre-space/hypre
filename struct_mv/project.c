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
 * hypre_ProjectBox:
 *   Projects a box onto a strided index space that contains the
 *   index `index' and has stride `stride'.
 *
 *   Note: An SBox is returned regardless of the outcome of the
 *   projection.  So, it is possible to return an SBox with volume 0.
 *--------------------------------------------------------------------------*/

hypre_SBox *
hypre_ProjectBox( hypre_Box    *box,
                  hypre_Index   index,
                  hypre_Index   stride )
{
   hypre_SBox  *new_sbox;
   hypre_Box   *new_box;
   hypre_Index  new_stride;

   int          i, s, d, hl, hu, kl, ku;

   /*------------------------------------------------------
    * project in all 3 dimensions
    *------------------------------------------------------*/

   new_box = hypre_DuplicateBox(box);

   for (d = 0; d < 3; d++)
   {

      i = hypre_IndexD(index, d);
      s = hypre_IndexD(stride, d);

      hl = hypre_BoxIMinD(new_box, d) - i;
      hu = hypre_BoxIMaxD(new_box, d) - i;

      if ( hl <= 0 )
         kl = (int) (hl / s);
      else
         kl = (int) ((hl + (s-1)) / s);

      if ( hu >= 0 )
         ku = (int) (hu / s);
      else
         ku = (int) ((hu - (s-1)) / s);

      hypre_BoxIMinD(new_box, d) = i + kl * s;
      hypre_BoxIMaxD(new_box, d) = i + ku * s;

   }

   /*------------------------------------------------------
    * set the strides
    *------------------------------------------------------*/

   hypre_CopyIndex(stride, new_stride);

   new_sbox = hypre_NewSBox(new_box, new_stride);

   return new_sbox;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArray:
 *
 *   Note: The dimensions of the returned SBoxArray are the same as
 *   the input argument `box_array'.  So, it is possible for the
 *   returned SBoxArray to contain SBoxes with volume 0.
 *--------------------------------------------------------------------------*/

hypre_SBoxArray *
hypre_ProjectBoxArray( hypre_BoxArray  *box_array,
                       hypre_Index      index,
                       hypre_Index      stride    )
{
   hypre_SBoxArray  *new_sbox_array;
   hypre_SBox       *new_sbox;
                  
   hypre_Box        *box;
                  
   int               i;

   new_sbox_array = hypre_NewSBoxArray(hypre_BoxArraySize(box_array));

   hypre_ForBoxI(i, box_array)
      {
         box      = hypre_BoxArrayBox(box_array, i);
         new_sbox = hypre_ProjectBox(box, index, stride);
         hypre_AppendSBox(new_sbox, new_sbox_array);
      }

   return new_sbox_array;
}

/*--------------------------------------------------------------------------
 * hypre_ProjectBoxArrayArray:
 *
 *   Note: The dimensions of the returned SBoxArrayArray are the same as
 *   the input argument `box_array_array'.  So, it is possible for the
 *   returned SBoxArrayArray to contain SBoxes with volume 0.
 *--------------------------------------------------------------------------*/

hypre_SBoxArrayArray *
hypre_ProjectBoxArrayArray( hypre_BoxArrayArray  *box_array_array,
                            hypre_Index           index,
                            hypre_Index           stride          )
{
   hypre_SBoxArrayArray  *new_sbox_array_array;
   hypre_SBoxArray       *new_sbox_array;

   hypre_BoxArray        *box_array;

   int                    i;

   new_sbox_array_array =
      hypre_NewSBoxArrayArray(hypre_BoxArrayArraySize(box_array_array));

   hypre_ForBoxArrayI(i, box_array_array)
      {
         box_array      = hypre_BoxArrayArrayBoxArray(box_array_array, i);

         new_sbox_array =
            hypre_SBoxArrayArraySBoxArray(new_sbox_array_array, i);
         hypre_FreeSBoxArray(new_sbox_array);

         new_sbox_array = hypre_ProjectBoxArray(box_array, index, stride);
         hypre_SBoxArrayArraySBoxArray(new_sbox_array_array, i) =
            new_sbox_array;
      }

   return new_sbox_array_array;
}


#if 0
/*--------------------------------------------------------------------------
 * ProjectRBPoint:
 *   Project a box_array_array onto a red or black set of indices.
 *--------------------------------------------------------------------------*/

hypre_SBoxArrayArray *
hypre_ProjectRBPoint( hypre_BoxArrayArray *box_array_array,
                      hypre_Index          rb[4]           )
{
   hypre_SBoxArrayArray *new_sbox_array_array;
   hypre_SBoxArrayArray *tmp_sbox_array_array;

   hypre_Index           stride;

   int                   i;

   hypre_SetIndex(stride, 2, 2, 2);

   new_sbox_array_array =
      hypre_NewSBoxArrayArray(hypre_BoxArrayArraySize(box_array_array));

   for (i = 0; i < 4; i++)
   {
      tmp_sbox_array_array =
         hypre_ProjectBoxArrayArray(box_array_array, rb[i], stride);
      hypre_AppendSBoxArrayArray(tmp_sbox_array_array, new_sbox_array_array);
      hypre_FreeSBoxArrayArrayShell(tmp_sbox_array_array);
   }
   
   return new_sbox_array_array;
}
#endif
