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
hypre_NewBox( )
{
   hypre_Box *box;

#if 1
   box = hypre_TAlloc(hypre_Box, 1);
#else
   box = hypre_BoxAlloc();
#endif

   return box;
}

/*--------------------------------------------------------------------------
 * hypre_SetBoxExtents
 *--------------------------------------------------------------------------*/

int
hypre_SetBoxExtents( hypre_Box  *box,
                     hypre_Index imin,
                     hypre_Index imax )
{
   int        ierr = 0;

   hypre_CopyIndex(imin, hypre_BoxIMin(box));
   hypre_CopyIndex(imax, hypre_BoxIMax(box));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NewBoxArray
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_NewBoxArray( int size )
{
   hypre_BoxArray *box_array;

   box_array = hypre_TAlloc(hypre_BoxArray, 1);

   hypre_BoxArrayBoxes(box_array)     = hypre_CTAlloc(hypre_Box, size);
   hypre_BoxArraySize(box_array)      = size;
   hypre_BoxArrayAllocSize(box_array) = size;

   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_SetBoxArraySize
 *--------------------------------------------------------------------------*/

int
hypre_SetBoxArraySize( hypre_BoxArray  *box_array,
                       int              size      )
{
   int  ierr  = 0;
   int  alloc_size;

   alloc_size = hypre_BoxArrayAllocSize(box_array);

   if (size > alloc_size)
   {
      alloc_size = size + hypre_BoxArrayExcess;

      hypre_BoxArrayBoxes(box_array) =
         hypre_TReAlloc(hypre_BoxArrayBoxes(box_array),
                        hypre_Box, alloc_size);

      hypre_BoxArrayAllocSize(box_array) = alloc_size;
   }

   hypre_BoxArraySize(box_array) = size;

   return ierr;
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
   {
      hypre_BoxArrayArrayBoxArray(box_array_array, i) = hypre_NewBoxArray(0);
   }
   hypre_BoxArrayArraySize(box_array_array) = size;
 
   return box_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBox
 *--------------------------------------------------------------------------*/

int 
hypre_FreeBox( hypre_Box *box )
{
   int ierr = 0;

   if (box)
   {
#if 1
      hypre_TFree(box);
#else
      hypre_BoxFree(box);
#endif
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArray
 *--------------------------------------------------------------------------*/

int 
hypre_FreeBoxArray( hypre_BoxArray *box_array )
{
   int  ierr = 0;
   int  i;

   if (box_array)
   {
      hypre_TFree(hypre_BoxArrayBoxes(box_array));
      hypre_TFree(box_array);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArrayArray
 *--------------------------------------------------------------------------*/

int
hypre_FreeBoxArrayArray( hypre_BoxArrayArray *box_array_array )
{
   int  ierr = 0;
   int  i;
 
   if (box_array_array)
   {
      hypre_ForBoxArrayI(i, box_array_array)
         hypre_FreeBoxArray(hypre_BoxArrayArrayBoxArray(box_array_array, i));

      hypre_TFree(hypre_BoxArrayArrayBoxArrays(box_array_array));
      hypre_TFree(box_array_array);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateBox:
 *   Return a duplicate box.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_DuplicateBox( hypre_Box *box )
{
   hypre_Box  *new_box;

   new_box = hypre_NewBox();
   hypre_CopyBox(box, new_box);

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

   int              i;

   new_box_array = hypre_NewBoxArray(hypre_BoxArraySize(box_array));
   hypre_ForBoxI(i, box_array)
      {
         hypre_CopyBox(hypre_BoxArrayBox(box_array, i),
                       hypre_BoxArrayBox(new_box_array, i));
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
   hypre_Box            *new_box;
   int                   new_size;
 
   hypre_BoxArray      **box_arrays;
   int                   i, j;
 
   new_size = hypre_BoxArrayArraySize(box_array_array);
   new_box_array_array = hypre_NewBoxArrayArray(new_size);
 
   if (new_size)
   {
      new_box_arrays = hypre_BoxArrayArrayBoxArrays(new_box_array_array);
      box_arrays     = hypre_BoxArrayArrayBoxArrays(box_array_array);
 
      for (i = 0; i < new_size; i++)
      {
         hypre_AppendBoxArray(box_arrays[i], new_box_arrays[i]);
      }
   }
 
   return new_box_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_AppendBox:
 *   Append box to the end of box_array.
 *   The box_array may be empty.
 *--------------------------------------------------------------------------*/

int 
hypre_AppendBox( hypre_Box      *box,
                 hypre_BoxArray *box_array )
{
   int  ierr  = 0;
   int  size;

   size = hypre_BoxArraySize(box_array);
   hypre_SetBoxArraySize(box_array, (size + 1));
   hypre_CopyBox(box, hypre_BoxArrayBox(box_array, size));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_DeleteBox:
 *   Delete box from box_array.
 *--------------------------------------------------------------------------*/

int 
hypre_DeleteBox( hypre_BoxArray *box_array,
                 int             index     )
{
   int  ierr  = 0;
   int  i;

   for (i = index; i < hypre_BoxArraySize(box_array) - 1; i++)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array, i+1),
                    hypre_BoxArrayBox(box_array, i));
   }

   hypre_BoxArraySize(box_array) --;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AppendBoxArray:
 *   Append box_array_0 to the end of box_array_1.
 *   The box_array_1 may be empty.
 *--------------------------------------------------------------------------*/

int 
hypre_AppendBoxArray( hypre_BoxArray *box_array_0,
                      hypre_BoxArray *box_array_1 )
{
   int  ierr  = 0;
   int  size, size_0;
   int  i;

   size   = hypre_BoxArraySize(box_array_1);
   size_0 = hypre_BoxArraySize(box_array_0);
   hypre_SetBoxArraySize(box_array_1, (size + size_0));

   /* copy box_array_0 boxes into box_array_1 */
   for (i = 0; i < size_0; i++)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array_0, i),
                    hypre_BoxArrayBox(box_array_1, size + i));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GetBoxSize:
 *--------------------------------------------------------------------------*/

int
hypre_GetBoxSize( hypre_Box   *box,
                  hypre_Index  size )
{
   hypre_IndexD(size, 0) = hypre_BoxSizeD(box, 0);
   hypre_IndexD(size, 1) = hypre_BoxSizeD(box, 1);
   hypre_IndexD(size, 2) = hypre_BoxSizeD(box, 2);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_GetStrideBoxSize:
 *--------------------------------------------------------------------------*/

int 
hypre_GetStrideBoxSize( hypre_Box   *box,
                        hypre_Index  stride,
                        hypre_Index  size   )
{
   int  d, s;

   for (d = 0; d < 3; d++)
   {
      s = hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / hypre_IndexD(stride, d) + 1;
      }
      hypre_IndexD(size, d) = s;
   }

   return 0;
}

