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
 * Member functions for zzz_Box class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewBox
 *--------------------------------------------------------------------------*/

zzz_Box *
zzz_NewBox( zzz_Index *imin,
	    zzz_Index *imax )
{
   zzz_Box *box;

   box = talloc(zzz_Box, 1);

   zzz_BoxIMin(box) = imin;
   zzz_BoxIMax(box) = imax;

   return box;
}

/*--------------------------------------------------------------------------
 * zzz_NewBoxArray
 *--------------------------------------------------------------------------*/

zzz_BoxArray *
zzz_NewBoxArray( )
{
   zzz_BoxArray *box_array;

   box_array = talloc(zzz_BoxArray, 1);

   zzz_BoxArrayBoxes(box_array) = NULL;
   zzz_BoxArraySize(box_array)  = 0;

   return box_array;
}

/*--------------------------------------------------------------------------
 * zzz_NewBoxArrayArray
 *--------------------------------------------------------------------------*/

zzz_BoxArrayArray *
zzz_NewBoxArrayArray( int size )
{
   zzz_BoxArrayArray  *box_array_array;
   int                 i;
 
   box_array_array = ctalloc(zzz_BoxArrayArray, 1);
 
   zzz_BoxArrayArrayBoxArrays(box_array_array) =
      ctalloc(zzz_BoxArray *, size);
 
   for (i = 0; i < size; i++)
      zzz_BoxArrayArrayBoxArray(box_array_array, i) = zzz_NewBoxArray();
   zzz_BoxArrayArraySize(box_array_array) = size;
 
   return box_array_array;
}

/*--------------------------------------------------------------------------
 * zzz_FreeBox
 *--------------------------------------------------------------------------*/

void 
zzz_FreeBox( zzz_Box *box )
{
   if (box)
   {
      zzz_FreeIndex(zzz_BoxIMin(box));
      zzz_FreeIndex(zzz_BoxIMax(box));
      tfree(box);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeBoxArrayShell:
 *   Frees everything but the boxes.
 *--------------------------------------------------------------------------*/

void 
zzz_FreeBoxArrayShell( zzz_BoxArray *box_array )
{
   if (box_array)
   {
      tfree(zzz_BoxArrayBoxes(box_array));
      tfree(box_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeBoxArray
 *--------------------------------------------------------------------------*/

void 
zzz_FreeBoxArray( zzz_BoxArray *box_array )
{
   int  i;

   if (box_array)
   {
      if ( zzz_BoxArrayBoxes(box_array)!= NULL )
      {
         zzz_ForBoxI(i, box_array)
            zzz_FreeBox(zzz_BoxArrayBox(box_array, i));
      }

      zzz_FreeBoxArrayShell(box_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeBoxArrayArrayShell:
 *   Frees everything but the box_arrays.
 *--------------------------------------------------------------------------*/

void 
zzz_FreeBoxArrayArrayShell( zzz_BoxArrayArray *box_array_array )
{
   if (box_array_array)
   {
      tfree(zzz_BoxArrayArrayBoxArrays(box_array_array));
      tfree(box_array_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_FreeBoxArrayArray
 *--------------------------------------------------------------------------*/

void
zzz_FreeBoxArrayArray( zzz_BoxArrayArray *box_array_array )
{
   int  i;
 
   if (box_array_array)
   {
      zzz_ForBoxArrayI(i, box_array_array)
         zzz_FreeBoxArray(zzz_BoxArrayArrayBoxArray(box_array_array, i));

      zzz_FreeBoxArrayArrayShell(box_array_array);
   }
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateBox:
 *   Return a duplicate box.
 *--------------------------------------------------------------------------*/

zzz_Box *
zzz_DuplicateBox( zzz_Box *box )
{
   zzz_Box    *new_box;

   zzz_Index  *imin;
   zzz_Index  *imax;

   int         d;

   imin = zzz_NewIndex();
   imax = zzz_NewIndex();
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD(imin, d) = zzz_BoxIMinD(box, d);
      zzz_IndexD(imax, d) = zzz_BoxIMaxD(box, d);
   }

   new_box = zzz_NewBox(imin, imax);

   return new_box;
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateBoxArray:
 *   Return a duplicate box_array.
 *--------------------------------------------------------------------------*/

zzz_BoxArray *
zzz_DuplicateBoxArray( zzz_BoxArray *box_array )
{
   zzz_BoxArray  *new_box_array;
   zzz_Box      **new_boxes;
   int            new_size;

   zzz_Box      **boxes;
   int            i, data_sz;

   new_box_array = zzz_NewBoxArray();
   new_boxes = NULL;
   new_size = zzz_BoxArraySize(box_array);

   if (new_size)
   {
      data_sz = ((((new_size - 1) / zzz_BoxArrayBlocksize) + 1) *
		 zzz_BoxArrayBlocksize);
      new_boxes = ctalloc(zzz_Box *, data_sz);

      boxes = zzz_BoxArrayBoxes(box_array);

      for (i = 0; i < new_size; i++)
	 new_boxes[i] = zzz_DuplicateBox(boxes[i]);
   }

   zzz_BoxArrayBoxes(new_box_array) = new_boxes;
   zzz_BoxArraySize(new_box_array)  = new_size;

   return new_box_array;
}

/*--------------------------------------------------------------------------
 * zzz_DuplicateBoxArrayArray:
 *   Return a duplicate box_array_array.
 *--------------------------------------------------------------------------*/

zzz_BoxArrayArray *
zzz_DuplicateBoxArrayArray( zzz_BoxArrayArray *box_array_array )
{
   zzz_BoxArrayArray  *new_box_array_array;
   zzz_BoxArray      **new_box_arrays;
   int                 new_size;
 
   zzz_BoxArray      **box_arrays;
   int                 i;
 
   new_size = zzz_BoxArrayArraySize(box_array_array);
   new_box_array_array = zzz_NewBoxArrayArray(new_size);
 
   if (new_size)
   {
      new_box_arrays = zzz_BoxArrayArrayBoxArrays(new_box_array_array);
      box_arrays     = zzz_BoxArrayArrayBoxArrays(box_array_array);
 
      for (i = 0; i < new_size; i++)
      {
         zzz_FreeBoxArray(new_box_arrays[i]);
         new_box_arrays[i] =
            zzz_DuplicateBoxArray(box_arrays[i]);
      }
   }
 
   return new_box_array_array;
}

/*--------------------------------------------------------------------------
 * zzz_AppendBox:
 *   Append box to the end of box_array.
 *   The box_array may be empty.
 *--------------------------------------------------------------------------*/

void 
zzz_AppendBox( zzz_Box      *box,
	       zzz_BoxArray *box_array )
{
   zzz_Box  **boxes;
   int        size;

   zzz_Box  **old_boxes;
   int        i;

   size = zzz_BoxArraySize(box_array);
   if (!(size % zzz_BoxArrayBlocksize))
   {
      boxes = ctalloc(zzz_Box *, size + zzz_BoxArrayBlocksize);
      old_boxes = zzz_BoxArrayBoxes(box_array);

      for (i = 0; i < size; i++)
	 boxes[i] = old_boxes[i];

      zzz_BoxArrayBoxes(box_array) = boxes;

      tfree(old_boxes);
   }

   zzz_BoxArrayBox(box_array, size) = box;
   zzz_BoxArraySize(box_array) ++;
}

/*--------------------------------------------------------------------------
 * zzz_DeleteBox:
 *   Delete box from box_array.
 *--------------------------------------------------------------------------*/

void 
zzz_DeleteBox( zzz_BoxArray *box_array,
	       int           index     )
{
   zzz_Box  **boxes;

   int        i;

   boxes = zzz_BoxArrayBoxes(box_array);

   zzz_FreeBox(boxes[index]);
   for (i = index; i < zzz_BoxArraySize(box_array) - 1; i++)
      boxes[i] = boxes[i+1];

   zzz_BoxArraySize(box_array) --;
}

/*--------------------------------------------------------------------------
 * zzz_AppendBoxArray:
 *   Append box_array_0 to the end of box_array_1.
 *   The box_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void 
zzz_AppendBoxArray( zzz_BoxArray *box_array_0,
		    zzz_BoxArray *box_array_1 )
{
   int  i;

   zzz_ForBoxI(i, box_array_0)
      zzz_AppendBox(zzz_BoxArrayBox(box_array_0, i), box_array_1);
}

/*--------------------------------------------------------------------------
 * zzz_AppendBoxArrayArray:
 *   Append box_array_array_0 to box_array_array_1.
 *   The two BoxArrayArrays must be the same length.
 *--------------------------------------------------------------------------*/

void 
zzz_AppendBoxArrayArray( zzz_BoxArrayArray *box_array_array_0,
                         zzz_BoxArrayArray *box_array_array_1 )
{
   int  i;

   zzz_ForBoxArrayI(i, box_array_array_0)
      zzz_AppendBoxArray(zzz_BoxArrayArrayBoxArray(box_array_array_0, i),
                         zzz_BoxArrayArrayBoxArray(box_array_array_1, i));
}

