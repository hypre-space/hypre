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
 * Member functions for hypre_SBox class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewSBox
 *
 *    Note: The `box' argument is modified by this routine.
 *--------------------------------------------------------------------------*/

hypre_SBox *
hypre_NewSBox( hypre_Box   *box,
               hypre_Index  stride )
{
   hypre_SBox *sbox;
   int         d;

   sbox = hypre_TAlloc(hypre_SBox, 1);

   hypre_SBoxBox(sbox)    = box;

   for (d = 0; d < 3; d++)
   {
      hypre_SBoxStrideD(sbox, d) = hypre_IndexD(stride, d);

      /* adjust imax */
      hypre_SBoxIMaxD(sbox, d) = hypre_SBoxIMinD(sbox, d) +
         ((hypre_SBoxSizeD(sbox, d) - 1) * hypre_SBoxStrideD(sbox, d));
   }

   return sbox;
}

/*--------------------------------------------------------------------------
 * hypre_NewSBoxArray
 *--------------------------------------------------------------------------*/

hypre_SBoxArray *
hypre_NewSBoxArray( )
{
   hypre_SBoxArray *sbox_array;

   sbox_array = hypre_TAlloc(hypre_SBoxArray, 1);

   hypre_SBoxArraySBoxes(sbox_array) = NULL;
   hypre_SBoxArraySize(sbox_array)   = 0;

   return sbox_array;
}

/*--------------------------------------------------------------------------
 * hypre_NewSBoxArrayArray
 *--------------------------------------------------------------------------*/

hypre_SBoxArrayArray *
hypre_NewSBoxArrayArray( int size )
{
   hypre_SBoxArrayArray  *sbox_array_array;
   int                    i;

   sbox_array_array = hypre_CTAlloc(hypre_SBoxArrayArray, 1);

   hypre_SBoxArrayArraySBoxArrays(sbox_array_array) =
      hypre_CTAlloc(hypre_SBoxArray *, size);

   for (i = 0; i < size; i++)
      hypre_SBoxArrayArraySBoxArray(sbox_array_array, i) =
         hypre_NewSBoxArray();
   hypre_SBoxArrayArraySize(sbox_array_array) = size;

   return sbox_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_FreeSBox
 *--------------------------------------------------------------------------*/

void
hypre_FreeSBox( hypre_SBox *sbox )
{
   if (sbox)
   {
      hypre_FreeBox(hypre_SBoxBox(sbox));
      hypre_TFree(sbox);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeSBoxArrayShell:
 *   Frees everything but the sboxes.
 *--------------------------------------------------------------------------*/
 
void
hypre_FreeSBoxArrayShell( hypre_SBoxArray *sbox_array )
{
   if (sbox_array)
   {
      hypre_TFree(hypre_SBoxArraySBoxes(sbox_array));
      hypre_TFree(sbox_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeSBoxArray
 *--------------------------------------------------------------------------*/

void
hypre_FreeSBoxArray( hypre_SBoxArray *sbox_array )
{
   int  i;

   if (sbox_array)
   {
      if ( hypre_SBoxArraySBoxes(sbox_array)!= NULL )
      {
         hypre_ForSBoxI(i, sbox_array)
            hypre_FreeSBox(hypre_SBoxArraySBox(sbox_array, i));
      }

      hypre_FreeSBoxArrayShell(sbox_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeSBoxArrayArrayShell:
 *   Frees everything but the sbox_arrays.
 *--------------------------------------------------------------------------*/

void
hypre_FreeSBoxArrayArrayShell( hypre_SBoxArrayArray *sbox_array_array )
{
   if (sbox_array_array)
   {
      hypre_TFree(hypre_SBoxArrayArraySBoxArrays(sbox_array_array));
      hypre_TFree(sbox_array_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeSBoxArrayArray
 *--------------------------------------------------------------------------*/

void
hypre_FreeSBoxArrayArray( hypre_SBoxArrayArray *sbox_array_array )
{
   int  i;
 
   if (sbox_array_array)
   {
      hypre_ForSBoxArrayI(i, sbox_array_array)
         hypre_FreeSBoxArray(hypre_SBoxArrayArraySBoxArray(sbox_array_array, i));

      hypre_FreeSBoxArrayArrayShell(sbox_array_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateSBox:
 *   Return a duplicate sbox.
 *--------------------------------------------------------------------------*/

hypre_SBox *
hypre_DuplicateSBox( hypre_SBox *sbox )
{
   hypre_SBox  *new_sbox;
   hypre_Box   *new_box;


   new_box = hypre_DuplicateBox(hypre_SBoxBox(sbox));

   new_sbox = hypre_NewSBox(new_box, hypre_SBoxStride(sbox));

   return new_sbox;
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateSBoxArray:
 *   Return a duplicate sbox_array.
 *--------------------------------------------------------------------------*/

hypre_SBoxArray *
hypre_DuplicateSBoxArray( hypre_SBoxArray *sbox_array )
{
   hypre_SBoxArray  *new_sbox_array;
   hypre_SBox      **new_sboxes;
   int               new_size;

   hypre_SBox      **sboxes;
   int               i, data_sz;

   new_sbox_array = hypre_NewSBoxArray();
   new_sboxes = NULL;
   new_size = hypre_SBoxArraySize(sbox_array);

   if (new_size)
   {
      data_sz = ((((new_size - 1) / hypre_SBoxArrayBlocksize) + 1) *
                 hypre_SBoxArrayBlocksize);
      new_sboxes = hypre_CTAlloc(hypre_SBox *, data_sz);

      sboxes = hypre_SBoxArraySBoxes(sbox_array);

      for (i = 0; i < new_size; i++)
	 new_sboxes[i] = hypre_DuplicateSBox(sboxes[i]);
   }

   hypre_SBoxArraySBoxes(new_sbox_array) = new_sboxes;
   hypre_SBoxArraySize(new_sbox_array)   = new_size;

   return new_sbox_array;
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateSBoxArrayArray:
 *   Return a duplicate sbox_array_array.
 *--------------------------------------------------------------------------*/

hypre_SBoxArrayArray *
hypre_DuplicateSBoxArrayArray( hypre_SBoxArrayArray *sbox_array_array )
{
   hypre_SBoxArrayArray  *new_sbox_array_array;
   hypre_SBoxArray      **new_sbox_arrays;
   int                    new_size;

   hypre_SBoxArray      **sbox_arrays;
   int                    i;

   new_size = hypre_SBoxArrayArraySize(sbox_array_array);
   new_sbox_array_array = hypre_NewSBoxArrayArray(new_size);

   if (new_size)
   {
      new_sbox_arrays = hypre_SBoxArrayArraySBoxArrays(new_sbox_array_array);
      sbox_arrays     = hypre_SBoxArrayArraySBoxArrays(sbox_array_array);

      for (i = 0; i < new_size; i++)
      {
	 hypre_FreeSBoxArray(new_sbox_arrays[i]);
	 new_sbox_arrays[i] =
	    hypre_DuplicateSBoxArray(sbox_arrays[i]);
      }
   }

   return new_sbox_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_ConvertToSBox:
 *    Convert a Box to an SBox
 *--------------------------------------------------------------------------*/

hypre_SBox *
hypre_ConvertToSBox( hypre_Box *box )
{
   hypre_SBox  *sbox;
   hypre_Index  stride;

   hypre_SetIndex(stride, 1, 1, 1);

   sbox = hypre_NewSBox(box, stride);

   return sbox;
}

/*--------------------------------------------------------------------------
 * hypre_ConvertToSBoxArray
 *    Convert a BoxArray to an SBoxArray
 *--------------------------------------------------------------------------*/

hypre_SBoxArray *
hypre_ConvertToSBoxArray( hypre_BoxArray *box_array )
{
   hypre_SBoxArray *sbox_array;
   hypre_SBox      *sbox;
   int              i;

   sbox_array = hypre_NewSBoxArray();
   hypre_ForBoxI(i, box_array)
      {
         sbox = hypre_ConvertToSBox(hypre_BoxArrayBox(box_array, i));
         hypre_AppendSBox(sbox, sbox_array);
      }

   hypre_FreeBoxArrayShell(box_array);

   return sbox_array;
}

/*--------------------------------------------------------------------------
 * hypre_ConvertToSBoxArrayArray
 *    Convert a BoxArrayArray to an SBoxArrayArray
 *--------------------------------------------------------------------------*/

hypre_SBoxArrayArray *
hypre_ConvertToSBoxArrayArray( hypre_BoxArrayArray *box_array_array )
{
   hypre_SBoxArrayArray  *sbox_array_array;
   int                    i;

   sbox_array_array =
      hypre_NewSBoxArrayArray(hypre_BoxArrayArraySize(box_array_array));
   hypre_ForBoxArrayI(i, box_array_array)
      {
         hypre_FreeSBoxArray(hypre_SBoxArrayArraySBoxArray(sbox_array_array,
                                                           i));
         hypre_SBoxArrayArraySBoxArray(sbox_array_array, i) =
            hypre_ConvertToSBoxArray(hypre_BoxArrayArrayBoxArray(box_array_array, i));
      }

   hypre_FreeBoxArrayArrayShell(box_array_array);

   return sbox_array_array;

}

/*--------------------------------------------------------------------------
 * hypre_AppendSBox:
 *   Append sbox to the end of sbox_array.
 *   The sbox_array may be empty.
 *--------------------------------------------------------------------------*/
 
void
hypre_AppendSBox( hypre_SBox      *sbox,
                  hypre_SBoxArray *sbox_array )
{
   hypre_SBox  **sboxes;
   int           size;
 
   hypre_SBox  **old_sboxes;
   int           i;
 
   size = hypre_SBoxArraySize(sbox_array);
   if (!(size % hypre_SBoxArrayBlocksize))
   {
      sboxes = hypre_CTAlloc(hypre_SBox *, size + hypre_SBoxArrayBlocksize);
      old_sboxes = hypre_SBoxArraySBoxes(sbox_array);
 
      for (i = 0; i < size; i++)
         sboxes[i] = old_sboxes[i];
 
      hypre_SBoxArraySBoxes(sbox_array) = sboxes;
 
      hypre_TFree(old_sboxes);
   }
 
   hypre_SBoxArraySBox(sbox_array, size) = sbox;
   hypre_SBoxArraySize(sbox_array) ++;
}

/*--------------------------------------------------------------------------
 * hypre_DeleteSBox:
 *   Delete sbox from sbox_array.
 *--------------------------------------------------------------------------*/
 
void
hypre_DeleteSBox( hypre_SBoxArray *sbox_array,
                  int              index      )
{
   hypre_SBox  **sboxes;
 
   int           i;
 
   sboxes = hypre_SBoxArraySBoxes(sbox_array);
 
   hypre_FreeSBox(sboxes[index]);
   for (i = index; i < hypre_SBoxArraySize(sbox_array) - 1; i++)
      sboxes[i] = sboxes[i+1];
 
   hypre_SBoxArraySize(sbox_array) --;
}
 
/*--------------------------------------------------------------------------
 * hypre_AppendSBoxArray:
 *   Append sbox_array_0 to the end of sbox_array_1.
 *   The sbox_array_1 may be empty.
 *--------------------------------------------------------------------------*/
 
void
hypre_AppendSBoxArray( hypre_SBoxArray *sbox_array_0,
                       hypre_SBoxArray *sbox_array_1 )
{
   int  i;
 
   hypre_ForSBoxI(i, sbox_array_0)
      hypre_AppendSBox(hypre_SBoxArraySBox(sbox_array_0, i), sbox_array_1);
}

/*--------------------------------------------------------------------------
 * hypre_AppendSBoxArrayArray:
 *   Append sbox_array_array_0 to sbox_array_array_1.
 *   The two SBoxArrayArrays must be the same length.
 *--------------------------------------------------------------------------*/
 
void
hypre_AppendSBoxArrayArray( hypre_SBoxArrayArray *sbox_array_array_0,
                            hypre_SBoxArrayArray *sbox_array_array_1 )
{
   int  i;
 
   hypre_ForSBoxArrayI(i, sbox_array_array_0)
      hypre_AppendSBoxArray(hypre_SBoxArrayArraySBoxArray(sbox_array_array_0,
                                                          i),
                            hypre_SBoxArrayArraySBoxArray(sbox_array_array_1,
                                                          i));
}

/*--------------------------------------------------------------------------
 * hypre_GetSBoxSize:
 *--------------------------------------------------------------------------*/

int 
hypre_GetSBoxSize( hypre_SBox  *sbox,
                   hypre_Index  size )
{
   hypre_IndexX(size) = hypre_SBoxSizeX(sbox);
   hypre_IndexY(size) = hypre_SBoxSizeY(sbox);
   hypre_IndexZ(size) = hypre_SBoxSizeZ(sbox);

   return 0;
}

