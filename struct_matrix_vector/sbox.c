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
 * Member functions for zzz_SBox class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewSBox
 *
 *    Note: The `box' argument is modified by this routine.
 *--------------------------------------------------------------------------*/

zzz_SBox *
zzz_NewSBox( zzz_Box   *box,
             zzz_Index *stride )
{
   zzz_SBox *sbox;
   int       d;

   sbox = zzz_TAlloc(zzz_SBox, 1);

   zzz_SBoxBox(sbox)    = box;
   zzz_SBoxStride(sbox) = stride;

   /* adjust imax */
   for (d = 0; d < 3; d++)
      zzz_SBoxIMaxD(sbox, d) = zzz_SBoxIMinD(sbox, d) +
         ((zzz_SBoxSizeD(sbox, d) - 1) * zzz_SBoxStrideD(sbox, d));

   return sbox;
}

/*--------------------------------------------------------------------------
 * zzz_NewSBoxArray
 *--------------------------------------------------------------------------*/

zzz_SBoxArray *
zzz_NewSBoxArray( )
{
   zzz_SBoxArray *sbox_array;

   sbox_array = zzz_TAlloc(zzz_SBoxArray, 1);

   zzz_SBoxArraySBoxes(sbox_array) = NULL;
   zzz_SBoxArraySize(sbox_array)   = 0;

   return sbox_array;
}

/*--------------------------------------------------------------------------
 * zzz_NewSBoxArrayArray
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_NewSBoxArrayArray( int size )
{
   zzz_SBoxArrayArray  *sbox_array_array;
   int                  i;

   sbox_array_array = zzz_CTAlloc(zzz_SBoxArrayArray, 1);

   zzz_SBoxArrayArraySBoxArrays(sbox_array_array) =
      zzz_CTAlloc(zzz_SBoxArray *, size);

   for (i = 0; i < size; i++)
      zzz_SBoxArrayArraySBoxArray(sbox_array_array, i) = zzz_NewSBoxArray();
   zzz_SBoxArrayArraySize(sbox_array_array) = size;

   return sbox_array_array;
}

/*--------------------------------------------------------------------------
 * zzz_FreeSBox
 *--------------------------------------------------------------------------*/

void
zzz_FreeSBox( zzz_SBox *sbox )
{
   if (sbox)
   {
      zzz_FreeIndex(zzz_SBoxStride(sbox));
      zzz_FreeBox(zzz_SBoxBox(sbox));
      zzz_TFree(sbox);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeSBoxArrayShell:
 *   Frees everything but the sboxes.
 *--------------------------------------------------------------------------*/
 
void
zzz_FreeSBoxArrayShell( zzz_SBoxArray *sbox_array )
{
   if (sbox_array)
   {
      zzz_TFree(zzz_SBoxArraySBoxes(sbox_array));
      zzz_TFree(sbox_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeSBoxArray
 *--------------------------------------------------------------------------*/

void
zzz_FreeSBoxArray( zzz_SBoxArray *sbox_array )
{
   int  i;

   if (sbox_array)
   {
      if ( zzz_SBoxArraySBoxes(sbox_array)!= NULL )
      {
         zzz_ForSBoxI(i, sbox_array)
            zzz_FreeSBox(zzz_SBoxArraySBox(sbox_array, i));
      }

      zzz_FreeSBoxArrayShell(sbox_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeSBoxArrayArrayShell:
 *   Frees everything but the sbox_arrays.
 *--------------------------------------------------------------------------*/

void
zzz_FreeSBoxArrayArrayShell( zzz_SBoxArrayArray *sbox_array_array )
{
   if (sbox_array_array)
   {
      zzz_TFree(zzz_SBoxArrayArraySBoxArrays(sbox_array_array));
      zzz_TFree(sbox_array_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeSBoxArrayArray
 *--------------------------------------------------------------------------*/

void
zzz_FreeSBoxArrayArray( zzz_SBoxArrayArray *sbox_array_array )
{
   int  i;
 
   if (sbox_array_array)
   {
      zzz_ForSBoxArrayI(i, sbox_array_array)
         zzz_FreeSBoxArray(zzz_SBoxArrayArraySBoxArray(sbox_array_array, i));

      zzz_FreeSBoxArrayArrayShell(sbox_array_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateSBox:
 *   Return a duplicate sbox.
 *--------------------------------------------------------------------------*/

zzz_SBox *
zzz_DuplicateSBox( zzz_SBox *sbox )
{
   zzz_SBox  *new_sbox;
   zzz_Box   *new_box;


   new_box = zzz_DuplicateBox(zzz_SBoxBox(sbox));

   new_sbox = zzz_NewSBox(new_box, zzz_SBoxStride(sbox));

   return new_sbox;
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateSBoxArray:
 *   Return a duplicate sbox_array.
 *--------------------------------------------------------------------------*/

zzz_SBoxArray *
zzz_DuplicateSBoxArray( zzz_SBoxArray *sbox_array )
{
   zzz_SBoxArray  *new_sbox_array;
   zzz_SBox      **new_sboxes;
   int             new_size;

   zzz_SBox      **sboxes;
   int             i, data_sz;

   new_sbox_array = zzz_NewSBoxArray();
   new_sboxes = NULL;
   new_size = zzz_SBoxArraySize(sbox_array);

   if (new_size)
   {
      data_sz = ((((new_size - 1) / zzz_SBoxArrayBlocksize) + 1) *
                 zzz_SBoxArrayBlocksize);
      new_sboxes = zzz_CTAlloc(zzz_SBox *, data_sz);

      sboxes = zzz_SBoxArraySBoxes(sbox_array);

      for (i = 0; i < new_size; i++)
	 new_sboxes[i] = zzz_DuplicateSBox(sboxes[i]);
   }

   zzz_SBoxArraySBoxes(new_sbox_array) = new_sboxes;
   zzz_SBoxArraySize(new_sbox_array)   = new_size;

   return new_sbox_array;
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateSBoxArrayArray:
 *   Return a duplicate sbox_array_array.
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_DuplicateSBoxArrayArray( zzz_SBoxArrayArray *sbox_array_array )
{
   zzz_SBoxArrayArray  *new_sbox_array_array;
   zzz_SBoxArray      **new_sbox_arrays;
   int                  new_size;

   zzz_SBoxArray      **sbox_arrays;
   int                  i;

   new_size = zzz_SBoxArrayArraySize(sbox_array_array);
   new_sbox_array_array = zzz_NewSBoxArrayArray(new_size);

   if (new_size)
   {
      new_sbox_arrays = zzz_SBoxArrayArraySBoxArrays(new_sbox_array_array);
      sbox_arrays     = zzz_SBoxArrayArraySBoxArrays(sbox_array_array);

      for (i = 0; i < new_size; i++)
      {
	 zzz_FreeSBoxArray(new_sbox_arrays[i]);
	 new_sbox_arrays[i] =
	    zzz_DuplicateSBoxArray(sbox_arrays[i]);
      }
   }

   return new_sbox_array_array;
}

/*--------------------------------------------------------------------------
 * zzz_ConvertToSBox:
 *    Convert a Box to an SBox
 *--------------------------------------------------------------------------*/

zzz_SBox *
zzz_ConvertToSBox( zzz_Box *box )
{
   zzz_SBox  *sbox;
   zzz_Index *stride;

   stride = zzz_NewIndex();
   zzz_SetIndex(stride, 1, 1, 1);

   sbox = zzz_NewSBox(box, stride);

   return sbox;
}

/*--------------------------------------------------------------------------
 * zzz_ConvertToSBoxArray
 *    Convert a BoxArray to an SBoxArray
 *--------------------------------------------------------------------------*/

zzz_SBoxArray *
zzz_ConvertToSBoxArray( zzz_BoxArray *box_array )
{
   zzz_SBoxArray *sbox_array;
   zzz_SBox      *sbox;
   int            i;

   sbox_array = zzz_NewSBoxArray();
   zzz_ForBoxI(i, box_array)
   {
      sbox = zzz_ConvertToSBox(zzz_BoxArrayBox(box_array, i));
      zzz_AppendSBox(sbox, sbox_array);
   }

   zzz_FreeBoxArrayShell(box_array);

   return sbox_array;
}

/*--------------------------------------------------------------------------
 * zzz_ConvertToSBoxArrayArray
 *    Convert a BoxArrayArray to an SBoxArrayArray
 *--------------------------------------------------------------------------*/

zzz_SBoxArrayArray *
zzz_ConvertToSBoxArrayArray( zzz_BoxArrayArray *box_array_array )
{
   zzz_SBoxArrayArray  *sbox_array_array;
   int                  i;

   sbox_array_array =
      zzz_NewSBoxArrayArray(zzz_BoxArrayArraySize(box_array_array));
   zzz_ForBoxArrayI(i, box_array_array)
   {
      zzz_FreeSBoxArray(zzz_SBoxArrayArraySBoxArray(sbox_array_array, i));
      zzz_SBoxArrayArraySBoxArray(sbox_array_array, i) =
         zzz_ConvertToSBoxArray(zzz_BoxArrayArrayBoxArray(box_array_array, i));
   }

   zzz_FreeBoxArrayArrayShell(box_array_array);

   return sbox_array_array;

}

/*--------------------------------------------------------------------------
 * zzz_AppendSBox:
 *   Append sbox to the end of sbox_array.
 *   The sbox_array may be empty.
 *--------------------------------------------------------------------------*/
 
void
zzz_AppendSBox( zzz_SBox      *sbox,
                zzz_SBoxArray *sbox_array )
{
   zzz_SBox  **sboxes;
   int         size;
 
   zzz_SBox  **old_sboxes;
   int         i;
 
   size = zzz_SBoxArraySize(sbox_array);
   if (!(size % zzz_SBoxArrayBlocksize))
   {
      sboxes = zzz_CTAlloc(zzz_SBox *, size + zzz_SBoxArrayBlocksize);
      old_sboxes = zzz_SBoxArraySBoxes(sbox_array);
 
      for (i = 0; i < size; i++)
         sboxes[i] = old_sboxes[i];
 
      zzz_SBoxArraySBoxes(sbox_array) = sboxes;
 
      zzz_TFree(old_sboxes);
   }
 
   zzz_SBoxArraySBox(sbox_array, size) = sbox;
   zzz_SBoxArraySize(sbox_array) ++;
}

/*--------------------------------------------------------------------------
 * zzz_DeleteSBox:
 *   Delete sbox from sbox_array.
 *--------------------------------------------------------------------------*/
 
void
zzz_DeleteSBox( zzz_SBoxArray *sbox_array,
                int            index      )
{
   zzz_SBox  **sboxes;
 
   int         i;
 
   sboxes = zzz_SBoxArraySBoxes(sbox_array);
 
   zzz_FreeSBox(sboxes[index]);
   for (i = index; i < zzz_SBoxArraySize(sbox_array) - 1; i++)
      sboxes[i] = sboxes[i+1];
 
   zzz_SBoxArraySize(sbox_array) --;
}
 
/*--------------------------------------------------------------------------
 * zzz_AppendSBoxArray:
 *   Append sbox_array_0 to the end of sbox_array_1.
 *   The sbox_array_1 may be empty.
 *--------------------------------------------------------------------------*/
 
void
zzz_AppendSBoxArray( zzz_SBoxArray *sbox_array_0,
                     zzz_SBoxArray *sbox_array_1 )
{
   int  i;
 
   zzz_ForSBoxI(i, sbox_array_0)
      zzz_AppendSBox(zzz_SBoxArraySBox(sbox_array_0, i), sbox_array_1);
}

/*--------------------------------------------------------------------------
 * zzz_AppendSBoxArrayArray:
 *   Append sbox_array_array_0 to sbox_array_array_1.
 *   The two SBoxArrayArrays must be the same length.
 *--------------------------------------------------------------------------*/
 
void
zzz_AppendSBoxArrayArray( zzz_SBoxArrayArray *sbox_array_array_0,
                          zzz_SBoxArrayArray *sbox_array_array_1 )
{
   int  i;
 
   zzz_ForSBoxArrayI(i, sbox_array_array_0)
      zzz_AppendSBoxArray(zzz_SBoxArrayArraySBoxArray(sbox_array_array_0, i),
                          zzz_SBoxArrayArraySBoxArray(sbox_array_array_1, i));
}

