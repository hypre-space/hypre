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

/*--------------------------------------------------------------------------
 * hypre_BoxCreate
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxCreate( )
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
 * hypre_BoxSetExtents
 *--------------------------------------------------------------------------*/

int
hypre_BoxSetExtents( hypre_Box  *box,
                     hypre_Index imin,
                     hypre_Index imax )
{
   int        ierr = 0;

   hypre_CopyIndex(imin, hypre_BoxIMin(box));
   hypre_CopyIndex(imax, hypre_BoxIMax(box));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayCreate
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_BoxArrayCreate( int size )
{
   hypre_BoxArray *box_array;

   box_array = hypre_TAlloc(hypre_BoxArray, 1);

   hypre_BoxArrayBoxes(box_array)     = hypre_CTAlloc(hypre_Box, size);
   hypre_BoxArraySize(box_array)      = size;
   hypre_BoxArrayAllocSize(box_array) = size;

   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArraySetSize
 *--------------------------------------------------------------------------*/

int
hypre_BoxArraySetSize( hypre_BoxArray  *box_array,
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
 * hypre_BoxArrayArrayCreate
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_BoxArrayArrayCreate( int size )
{
   hypre_BoxArrayArray  *box_array_array;
   int                   i;
 
   box_array_array = hypre_CTAlloc(hypre_BoxArrayArray, 1);
 
   hypre_BoxArrayArrayBoxArrays(box_array_array) =
      hypre_CTAlloc(hypre_BoxArray *, size);
 
   for (i = 0; i < size; i++)
   {
      hypre_BoxArrayArrayBoxArray(box_array_array, i) = hypre_BoxArrayCreate(0);
   }
   hypre_BoxArrayArraySize(box_array_array) = size;
 
   return box_array_array;
}

/*--------------------------------------------------------------------------
 * hypre_BoxDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_BoxDestroy( hypre_Box *box )
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
 * hypre_BoxArrayDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_BoxArrayDestroy( hypre_BoxArray *box_array )
{
   int  ierr = 0;

   if (box_array)
   {
      hypre_TFree(hypre_BoxArrayBoxes(box_array));
      hypre_TFree(box_array);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArrayDestroy
 *--------------------------------------------------------------------------*/

int
hypre_BoxArrayArrayDestroy( hypre_BoxArrayArray *box_array_array )
{
   int  ierr = 0;
   int  i;
 
   if (box_array_array)
   {
      hypre_ForBoxArrayI(i, box_array_array)
         hypre_BoxArrayDestroy(
            hypre_BoxArrayArrayBoxArray(box_array_array, i));

      hypre_TFree(hypre_BoxArrayArrayBoxArrays(box_array_array));
      hypre_TFree(box_array_array);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxDuplicate:
 *   Return a duplicate box.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxDuplicate( hypre_Box *box )
{
   hypre_Box  *new_box;

   new_box = hypre_BoxCreate();
   hypre_CopyBox(box, new_box);

   return new_box;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayDuplicate:
 *   Return a duplicate box_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_BoxArrayDuplicate( hypre_BoxArray *box_array )
{
   hypre_BoxArray  *new_box_array;

   int              i;

   new_box_array = hypre_BoxArrayCreate(hypre_BoxArraySize(box_array));
   hypre_ForBoxI(i, box_array)
      {
         hypre_CopyBox(hypre_BoxArrayBox(box_array, i),
                       hypre_BoxArrayBox(new_box_array, i));
      }

   return new_box_array;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayArrayDuplicate:
 *   Return a duplicate box_array_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_BoxArrayArrayDuplicate( hypre_BoxArrayArray *box_array_array )
{
   hypre_BoxArrayArray  *new_box_array_array;
   hypre_BoxArray      **new_box_arrays;
   int                   new_size;
 
   hypre_BoxArray      **box_arrays;
   int                   i;
 
   new_size = hypre_BoxArrayArraySize(box_array_array);
   new_box_array_array = hypre_BoxArrayArrayCreate(new_size);
 
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
   hypre_BoxArraySetSize(box_array, (size + 1));
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
   hypre_BoxArraySetSize(box_array_1, (size + size_0));

   /* copy box_array_0 boxes into box_array_1 */
   for (i = 0; i < size_0; i++)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array_0, i),
                    hypre_BoxArrayBox(box_array_1, size + i));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxGetSize:
 *--------------------------------------------------------------------------*/

int
hypre_BoxGetSize( hypre_Box   *box,
                  hypre_Index  size )
{
   hypre_IndexD(size, 0) = hypre_BoxSizeD(box, 0);
   hypre_IndexD(size, 1) = hypre_BoxSizeD(box, 1);
   hypre_IndexD(size, 2) = hypre_BoxSizeD(box, 2);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_BoxGetStrideSize:
 *--------------------------------------------------------------------------*/

int 
hypre_BoxGetStrideSize( hypre_Box   *box,
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

/*--------------------------------------------------------------------------
 * hypre_IModPeriod:
 *--------------------------------------------------------------------------*/

int 
hypre_IModPeriod( int   i,
                  int   period )
                        
{
   int  i_mod_p;
   int  shift;

   if (period == 0)
   {
      i_mod_p = i;
   }
   else if (i >= period)
   {
      i_mod_p = i % period;
   }
   else if (i < 0)
   {
      shift = ( -i / period + 1 ) * period;
      i_mod_p = ( i + shift ) % period;
   }
   else
   {
      i_mod_p = i;
   }

   return i_mod_p;
}

/*--------------------------------------------------------------------------
 * hypre_IModPeriodX:
 *  Perhaps should be a macro?
 *--------------------------------------------------------------------------*/

int
hypre_IModPeriodX( hypre_Index  index,
                   hypre_Index  periodic )
{
   return hypre_IModPeriod(hypre_IndexX(index), hypre_IndexX(periodic));
}


/*--------------------------------------------------------------------------
 * hypre_IModPeriodY:
 *  Perhaps should be a macro?
 *--------------------------------------------------------------------------*/

int
hypre_IModPeriodY( hypre_Index  index,
                   hypre_Index  periodic )
{
   return hypre_IModPeriod(hypre_IndexY(index), hypre_IndexY(periodic));
}


/*--------------------------------------------------------------------------
 * hypre_IModPeriodZ:
 *  Perhaps should be a macro?
 *--------------------------------------------------------------------------*/

int
hypre_IModPeriodZ( hypre_Index  index,
                   hypre_Index  periodic )
{
   return hypre_IModPeriod(hypre_IndexZ(index), hypre_IndexZ(periodic));
}


