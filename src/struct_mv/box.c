/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/



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

HYPRE_Int
hypre_BoxSetExtents( hypre_Box  *box,
                     hypre_Index imin,
                     hypre_Index imax )
{
   HYPRE_Int  ierr = 0;

   hypre_CopyIndex(imin, hypre_BoxIMin(box));
   hypre_CopyIndex(imax, hypre_BoxIMax(box));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayCreate
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_BoxArrayCreate( HYPRE_Int size )
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

HYPRE_Int
hypre_BoxArraySetSize( hypre_BoxArray  *box_array,
                       HYPRE_Int        size      )
{
   HYPRE_Int  ierr  = 0;
   HYPRE_Int  alloc_size;

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
hypre_BoxArrayArrayCreate( HYPRE_Int size )
{
   hypre_BoxArrayArray  *box_array_array;
   HYPRE_Int             i;
 
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

HYPRE_Int 
hypre_BoxDestroy( hypre_Box *box )
{
   HYPRE_Int ierr = 0;

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

HYPRE_Int 
hypre_BoxArrayDestroy( hypre_BoxArray *box_array )
{
   HYPRE_Int  ierr = 0;

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

HYPRE_Int
hypre_BoxArrayArrayDestroy( hypre_BoxArrayArray *box_array_array )
{
   HYPRE_Int  ierr = 0;
   HYPRE_Int  i;
 
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

   HYPRE_Int        i;

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
   HYPRE_Int             new_size;
 
   hypre_BoxArray      **box_arrays;
   HYPRE_Int             i;
 
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

HYPRE_Int 
hypre_AppendBox( hypre_Box      *box,
                 hypre_BoxArray *box_array )
{
   HYPRE_Int  ierr  = 0;
   HYPRE_Int  size;

   size = hypre_BoxArraySize(box_array);
   hypre_BoxArraySetSize(box_array, (size + 1));
   hypre_CopyBox(box, hypre_BoxArrayBox(box_array, size));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_DeleteBox:
 *   Delete box from box_array.
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DeleteBox( hypre_BoxArray *box_array,
                 HYPRE_Int       index     )
{
   HYPRE_Int  ierr  = 0;
   HYPRE_Int  i;

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

HYPRE_Int 
hypre_AppendBoxArray( hypre_BoxArray *box_array_0,
                      hypre_BoxArray *box_array_1 )
{
   HYPRE_Int  ierr  = 0;
   HYPRE_Int  size, size_0;
   HYPRE_Int  i;

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

HYPRE_Int
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

HYPRE_Int 
hypre_BoxGetStrideSize( hypre_Box   *box,
                        hypre_Index  stride,
                        hypre_Index  size   )
{
   HYPRE_Int  d, s;

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
 * hypre_BoxGetStrideVolume:
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_BoxGetStrideVolume( hypre_Box   *box,
                          hypre_Index  stride,
                          HYPRE_Int   *volume_ptr   )
{
   HYPRE_Int  volume = 1;
   HYPRE_Int  d, s;

   for (d = 0; d < 3; d++)
   {
      s = hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / hypre_IndexD(stride, d) + 1;
      }
      volume *= s;
   }

   *volume_ptr = volume;

   return 0;
}

/*--------------------------------------------------------------------------
 * GEC0209 function to expand a box given a ghostvector numexp
 * the idea is to dilatate the box using two vectors.
 *
 * even components of numexp shift negatively the imin of the box
 * odd  components of numexp shift positively the imax of the box
 * 
 * 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxExpand( hypre_Box   *box,
                 HYPRE_Int   *numexp)
{ 
  HYPRE_Int   ierr = 0;
  HYPRE_Int  *imin = hypre_BoxIMin(box);
  HYPRE_Int  *imax = hypre_BoxIMax(box);
  HYPRE_Int  d; 

  for (d = 0; d < 3; d++)
  {
    imin[d] -= numexp[2*d];
    imax[d] += numexp[2*d+1];
  }
  
  return ierr;
}
        

/*--------------------------------------------------------------------------
 * hypre_DeleteMultipleBoxes:
 *   Deletes boxes corrsponding to indices from box_array.
 *   Assumes indices are in ascending order. (AB 11/04)
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_DeleteMultipleBoxes( hypre_BoxArray *box_array,
                           HYPRE_Int*  indices , HYPRE_Int num )
{
   HYPRE_Int  ierr  = 0;
   HYPRE_Int  i, j, start, array_size;


   if (num < 1) return ierr;


   array_size =  hypre_BoxArraySize(box_array);   
   start = indices[0];
   j = 0;
   
   for (i = start; (i + j) < array_size; i++)
   {
      if (j < num)
      {
         while ((i+j) == indices[j]) /* see if deleting consecutive items */
         {
            j++; /*increase the shift*/
            if (j == num) break;
         }
      }
            
      if ( (i+j) < array_size)  /* if deleting the last item then no moving */
      {
         
         hypre_CopyBox(hypre_BoxArrayBox(box_array, i+j),
                       hypre_BoxArrayBox(box_array, i));
      }
      
      
   }


   hypre_BoxArraySize(box_array) = array_size - num;
   
   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MaxIndexPosition  - which index coordinate is the min.?
 *  (AB 11/04)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MaxIndexPosition(hypre_Index index, HYPRE_Int *position)
{
   HYPRE_Int  ierr = 0;
   HYPRE_Int  i, value;

   value = hypre_IndexD(index, 0);
   *position = 0;
   
   for (i = 1; i< 3; i++)
   {
      if (value <  hypre_IndexD(index, i))
      {
         value =   hypre_IndexD(index, i);
         *position = i;
      }
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_MinIndexPosition - which index coordinate is the max. ?
 *  (AB 11/04)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MinIndexPosition(hypre_Index index, HYPRE_Int *position)
{
   HYPRE_Int  ierr = 0;
   HYPRE_Int  i, value;

   value = hypre_IndexD(index, 0);
   *position = 0;
   
   for (i = 1; i< 3; i++)
   {
      if (value >  hypre_IndexD(index, i))
      {
         value =   hypre_IndexD(index, i);
         *position = i;
      }
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxExpandConstant - grow a box the same distance in each direction
 *   (AB 11/04)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxExpandConstant( hypre_Box   *box,
                         HYPRE_Int   expand)
{ 
  HYPRE_Int   ierr = 0;
  HYPRE_Int  *imin = hypre_BoxIMin(box);
  HYPRE_Int  *imax = hypre_BoxIMax(box);
  HYPRE_Int  d; 

  for (d = 0; d < 3; d++)
  {
    imin[d] -= expand;
    imax[d] += expand;
  }
  
  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxExpandDim - grow a box in each direction (the same pos as negative)
 * so expand is size 3
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxExpandConstantDim( hypre_Box   *box,
                         HYPRE_Int   *expand)
{ 
  HYPRE_Int   ierr = 0;
  HYPRE_Int  *imin = hypre_BoxIMin(box);
  HYPRE_Int  *imax = hypre_BoxIMax(box);
  HYPRE_Int  d; 

  for (d = 0; d < 3; d++)
  {
    imin[d] -= expand[d];
    imax[d] += expand[d];
  }
  
  return ierr;
}

  



