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
 * Member functions for hypre_Box class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "headers.h"
#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_NewBox
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_NewBox( hypre_Index imin,
              hypre_Index imax )
{
   hypre_Box *box;
   int        d;

#ifdef HYPRE_USE_PTHREADS
   box = hypre_TAlloc(hypre_Box, 1);
#else
   box = hypre_BoxAlloc();
#endif

   for (d = 0; d < 3; d++)
   {
      hypre_BoxIMinD(box, d) = hypre_IndexD(imin, d);
      hypre_BoxIMaxD(box, d) = hypre_IndexD(imax, d);
   }

   return box;
}

/*--------------------------------------------------------------------------
 * hypre_NewBoxArray
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_NewBoxArray( int alloc_size )
{
   hypre_BoxArray *box_array;

   box_array = hypre_TAlloc(hypre_BoxArray, 1);

   hypre_BoxArrayBoxes(box_array) = hypre_CTAlloc(hypre_Box *, alloc_size);
   hypre_BoxArraySize(box_array)  = 0;
   hypre_BoxArrayAllocSize(box_array)  = alloc_size;

   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_NewBoxArrayArray
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_NewBoxArrayArray( int size )
{
   hypre_BoxArrayArray  *box_array_array;
   int                   i;
 
   box_array_array = hypre_CTAlloc(hypre_BoxArrayArray, 1);
 
   hypre_BoxArrayArrayBoxArrays(box_array_array) =
      hypre_CTAlloc(hypre_BoxArray *, size);
 
   for (i = 0; i < size; i++)
      hypre_BoxArrayArrayBoxArray(box_array_array, i) = hypre_NewBoxArray(0);
   hypre_BoxArrayArraySize(box_array_array) = size;
 
   return box_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBox
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBox( hypre_Box *box )
{
   if (box)
   {
#ifdef HYPRE_USE_PTHREADS
      hypre_TFree(box);
#else
      hypre_BoxFree(box);
#endif
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArrayShell:
 *   Frees everything but the boxes.
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBoxArrayShell( hypre_BoxArray *box_array )
{
   if (box_array)
   {
      hypre_TFree(hypre_BoxArrayBoxes(box_array));
      hypre_TFree(box_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArray
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBoxArray( hypre_BoxArray *box_array )
{
   int  i;

   if (box_array)
   {
      if ( hypre_BoxArrayBoxes(box_array)!= NULL )
      {
         hypre_ForBoxI(i, box_array)
            hypre_FreeBox(hypre_BoxArrayBox(box_array, i));
      }

      hypre_FreeBoxArrayShell(box_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArrayArrayShell:
 *   Frees everything but the box_arrays.
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBoxArrayArrayShell( hypre_BoxArrayArray *box_array_array )
{
   if (box_array_array)
   {
      hypre_TFree(hypre_BoxArrayArrayBoxArrays(box_array_array));
      hypre_TFree(box_array_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArrayArray
 *--------------------------------------------------------------------------*/

void
hypre_FreeBoxArrayArray( hypre_BoxArrayArray *box_array_array )
{
   int  i;
 
   if (box_array_array)
   {
      hypre_ForBoxArrayI(i, box_array_array)
         hypre_FreeBoxArray(hypre_BoxArrayArrayBoxArray(box_array_array, i));

      hypre_FreeBoxArrayArrayShell(box_array_array);
   }
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateBox:
 *   Return a duplicate box.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_DuplicateBox( hypre_Box *box )
{
   hypre_Box  *new_box;

   new_box = hypre_NewBox(hypre_BoxIMin(box), hypre_BoxIMax(box));

   return new_box;
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateBoxArray:
 *   Return a duplicate box_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_DuplicateBoxArray( hypre_BoxArray *box_array )
{
   hypre_BoxArray  *new_box_array;

   hypre_Box      **boxes = hypre_BoxArrayBoxes(box_array);
   int              size  = hypre_BoxArraySize(box_array);

   int              i;

   new_box_array = hypre_NewBoxArray(size);
   hypre_ForBoxI(i, box_array)
      {
         hypre_AppendBox(hypre_DuplicateBox(boxes[i]), new_box_array);
      }

   return new_box_array;
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateBoxArrayArray:
 *   Return a duplicate box_array_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_DuplicateBoxArrayArray( hypre_BoxArrayArray *box_array_array )
{
   hypre_BoxArrayArray  *new_box_array_array;
   hypre_BoxArray      **new_box_arrays;
   int                   new_size;
 
   hypre_BoxArray      **box_arrays;
   int                   i;
 
   new_size = hypre_BoxArrayArraySize(box_array_array);
   new_box_array_array = hypre_NewBoxArrayArray(new_size);
 
   if (new_size)
   {
      new_box_arrays = hypre_BoxArrayArrayBoxArrays(new_box_array_array);
      box_arrays     = hypre_BoxArrayArrayBoxArrays(box_array_array);
 
      for (i = 0; i < new_size; i++)
      {
         hypre_FreeBoxArray(new_box_arrays[i]);
         new_box_arrays[i] = hypre_DuplicateBoxArray(box_arrays[i]);
      }
   }
 
   return new_box_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_AppendBox:
 *   Append box to the end of box_array.
 *   The box_array may be empty.
 *--------------------------------------------------------------------------*/

void 
hypre_AppendBox( hypre_Box      *box,
                 hypre_BoxArray *box_array )
{
   hypre_Box  **boxes;
   int          size;
   int          alloc_size;

   hypre_Box  **old_boxes;
   int          i;

   size = hypre_BoxArraySize(box_array);
   alloc_size = hypre_BoxArrayAllocSize(box_array);
   if (size == alloc_size)
   {
      hypre_BoxArrayAllocSize(box_array) += hypre_BoxArrayBlocksize;
      boxes = hypre_CTAlloc(hypre_Box *, hypre_BoxArrayAllocSize(box_array));
      old_boxes = hypre_BoxArrayBoxes(box_array);
      for (i = 0; i < size; i++)
	 boxes[i] = old_boxes[i];
      hypre_BoxArrayBoxes(box_array) = boxes;
      hypre_TFree(old_boxes);
   }

   hypre_BoxArrayBox(box_array, size) = box;
   hypre_BoxArraySize(box_array) ++;
}

/*--------------------------------------------------------------------------
 * hypre_DeleteBox:
 *   Delete box from box_array.
 *--------------------------------------------------------------------------*/

void 
hypre_DeleteBox( hypre_BoxArray *box_array,
                 int             index     )
{
   hypre_Box  **boxes;

   int          i;

   boxes = hypre_BoxArrayBoxes(box_array);

   hypre_FreeBox(boxes[index]);
   for (i = index; i < hypre_BoxArraySize(box_array) - 1; i++)
      boxes[i] = boxes[i+1];

   hypre_BoxArraySize(box_array) --;
}

/*--------------------------------------------------------------------------
 * hypre_AppendBoxArray:
 *   Append box_array_0 to the end of box_array_1.
 *   The box_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void 
hypre_AppendBoxArray( hypre_BoxArray *box_array_0,
                      hypre_BoxArray *box_array_1 )
{
   int  i;

   hypre_ForBoxI(i, box_array_0)
      hypre_AppendBox(hypre_BoxArrayBox(box_array_0, i), box_array_1);
}

/*--------------------------------------------------------------------------
 * hypre_AppendBoxArrayArray:
 *   Append box_array_array_0 to box_array_array_1.
 *   The two BoxArrayArrays must be the same length.
 *--------------------------------------------------------------------------*/

void 
hypre_AppendBoxArrayArray( hypre_BoxArrayArray *box_array_array_0,
                           hypre_BoxArrayArray *box_array_array_1 )
{
   int  i;

   hypre_ForBoxArrayI(i, box_array_array_0)
      hypre_AppendBoxArray(hypre_BoxArrayArrayBoxArray(box_array_array_0, i),
                           hypre_BoxArrayArrayBoxArray(box_array_array_1, i));
}

/*--------------------------------------------------------------------------
 * hypre_GetBoxSize:
 *--------------------------------------------------------------------------*/

int
hypre_GetBoxSize( hypre_Box   *box,
                  hypre_Index  size )
{
   hypre_IndexX(size) = hypre_BoxSizeX(box);
   hypre_IndexY(size) = hypre_BoxSizeY(box);
   hypre_IndexZ(size) = hypre_BoxSizeZ(box);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CopyBoxArrayData
 *  This function assumes only one box in box_array_in and
 *  that box_array_out consists of a sub_grid to that in box_array_in.
 *  This routine then copies data values from box_array_in to box_array_out.
 *  Author: pnb, 12-16-97
 *--------------------------------------------------------------------------*/

void
hypre_CopyBoxArrayData( hypre_BoxArray *box_array_in,
                        hypre_BoxArray *data_space_in,
                        int             num_values_in,
                        double         *data_in,
                        hypre_BoxArray *box_array_out,
                        hypre_BoxArray *data_space_out,
                        int             num_values_out,
                        double         *data_out       )
{
   hypre_Box    *box_in, *box_out;
   hypre_Box    *data_box_in, *data_box_out;
                
   int           data_box_volume_in, data_box_volume_out;
   int           datai_in, datai_out;
                
   hypre_Index   loop_size;
   hypre_Index   stride;
                
   int           j;
   int           loopi, loopj, loopk;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_SetIndex(stride, 1, 1, 1);

   box_in      = hypre_BoxArrayBox(box_array_in, 0);
   data_box_in = hypre_BoxArrayBox(data_space_in, 0);
   
   data_box_volume_in = hypre_BoxVolume(data_box_in);
   
   box_out      = hypre_BoxArrayBox(box_array_out, 0);
   data_box_out = hypre_BoxArrayBox(data_space_out, 0);
   
   data_box_volume_out = hypre_BoxVolume(data_box_out);

   hypre_GetBoxSize(box_out, loop_size);
   hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                  data_box_in, hypre_BoxIMin(box_out), stride, datai_in,
                  data_box_out, hypre_BoxIMin(box_out), stride, datai_out,
                  for (j = 0; j < num_values_out; j++)
                  {
                     data_out[datai_out + j*data_box_volume_out] =
                        data_in[datai_in + j*data_box_volume_in];
                  });
}

