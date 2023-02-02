/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Box class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*==========================================================================
 * Member functions: hypre_Index
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetIndex( hypre_Index  index,
                HYPRE_Int    val )
{
   HYPRE_Int d;

   for (d = 0; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(index, d) = val;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CopyIndex( hypre_Index  in_index,
                 hypre_Index  out_index )
{
   HYPRE_Int d;

   for (d = 0; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(out_index, d) = hypre_IndexD(in_index, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CopyToCleanIndex( hypre_Index  in_index,
                        HYPRE_Int    ndim,
                        hypre_Index  out_index )
{
   HYPRE_Int d;
   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(out_index, d) = hypre_IndexD(in_index, d);
   }
   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(out_index, d) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexEqual( hypre_Index  index,
                  HYPRE_Int    val,
                  HYPRE_Int    ndim )
{
   HYPRE_Int d, equal;

   equal = 1;
   for (d = 0; d < ndim; d++)
   {
      if (hypre_IndexD(index, d) != val)
      {
         equal = 0;
         break;
      }
   }

   return equal;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexMin( hypre_Index  index,
                HYPRE_Int    ndim )
{
   HYPRE_Int d, min;

   min = hypre_IndexD(index, 0);
   for (d = 1; d < ndim; d++)
   {
      if (hypre_IndexD(index, d) < min)
      {
         min = hypre_IndexD(index, d);
      }
   }

   return min;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexMax( hypre_Index  index,
                HYPRE_Int    ndim )
{
   HYPRE_Int d, max;

   max = hypre_IndexD(index, 0);
   for (d = 1; d < ndim; d++)
   {
      if (hypre_IndexD(index, d) < max)
      {
         max = hypre_IndexD(index, d);
      }
   }

   return max;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AddIndexes( hypre_Index  index1,
                  hypre_Index  index2,
                  HYPRE_Int    ndim,
                  hypre_Index  result )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(result, d) = hypre_IndexD(index1, d) + hypre_IndexD(index2, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SubtractIndexes( hypre_Index  index1,
                       hypre_Index  index2,
                       HYPRE_Int    ndim,
                       hypre_Index  result )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(result, d) = hypre_IndexD(index1, d) - hypre_IndexD(index2, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexesEqual( hypre_Index  index1,
                    hypre_Index  index2,
                    HYPRE_Int    ndim )
{
   HYPRE_Int d, equal;

   equal = 1;
   for (d = 0; d < ndim; d++)
   {
      if (hypre_IndexD(index1, d) != hypre_IndexD(index2, d))
      {
         equal = 0;
         break;
      }
   }

   return equal;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexPrint( FILE        *file,
                  HYPRE_Int    ndim,
                  hypre_Index  index )
{
   HYPRE_Int d;

   hypre_fprintf(file, "[%d", hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fprintf(file, " %d", hypre_IndexD(index, d));
   }
   hypre_fprintf(file, "]");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexRead( FILE        *file,
                 HYPRE_Int    ndim,
                 hypre_Index  index )
{
   HYPRE_Int d;

   hypre_fscanf(file, "[%d", &hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fscanf(file, " %d", &hypre_IndexD(index, d));
   }
   hypre_fscanf(file, "]");

   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(index, d) = 0;
   }

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_Box
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxCreate( HYPRE_Int  ndim )
{
   hypre_Box *box;

   box = hypre_CTAlloc(hypre_Box,  1, HYPRE_MEMORY_HOST);
   hypre_BoxNDim(box) = ndim;

   return box;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxDestroy( hypre_Box *box )
{
   if (box)
   {
      hypre_TFree(box, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This is used to initialize ndim when the box has static storage
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxInit( hypre_Box *box,
               HYPRE_Int  ndim )
{
   hypre_BoxNDim(box) = ndim;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxSetExtents( hypre_Box  *box,
                     hypre_Index imin,
                     hypre_Index imax )
{
   hypre_CopyIndex(imin, hypre_BoxIMin(box));
   hypre_CopyIndex(imax, hypre_BoxIMax(box));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CopyBox( hypre_Box  *box1,
               hypre_Box  *box2 )
{
   hypre_CopyIndex(hypre_BoxIMin(box1), hypre_BoxIMin(box2));
   hypre_CopyIndex(hypre_BoxIMax(box1), hypre_BoxIMax(box2));
   hypre_BoxNDim(box2) = hypre_BoxNDim(box1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxDuplicate( hypre_Box *box )
{
   hypre_Box  *new_box;

   new_box = hypre_BoxCreate(hypre_BoxNDim(box));
   hypre_CopyBox(box, new_box);

   return new_box;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxVolume( hypre_Box *box )
{
   HYPRE_Int volume, d, ndim = hypre_BoxNDim(box);

   volume = 1;
   for (d = 0; d < ndim; d++)
   {
      volume *= hypre_BoxSizeD(box, d);
   }

   return volume;
}

/*--------------------------------------------------------------------------
 * To prevent overflow when needed
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_doubleBoxVolume( hypre_Box *box )
{
   HYPRE_Real    volume;
   HYPRE_Int d, ndim = hypre_BoxNDim(box);

   volume = 1.0;
   for (d = 0; d < ndim; d++)
   {
      volume *= hypre_BoxSizeD(box, d);
   }

   return volume;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IndexInBox( hypre_Index   index,
                  hypre_Box    *box )
{
   HYPRE_Int d, inbox, ndim = hypre_BoxNDim(box);

   inbox = 1;
   for (d = 0; d < ndim; d++)
   {
      if (!hypre_IndexDInBox(index, d, box))
      {
         inbox = 0;
         break;
      }
   }

   return inbox;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGetSize( hypre_Box   *box,
                  hypre_Index  size )
{
   HYPRE_Int d, ndim = hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(size, d) = hypre_BoxSizeD(box, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGetStrideSize( hypre_Box   *box,
                        hypre_Index  stride,
                        hypre_Index  size   )
{
   HYPRE_Int  d, s, ndim = hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      s = hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / hypre_IndexD(stride, d) + 1;
      }
      hypre_IndexD(size, d) = s;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGetStrideVolume( hypre_Box   *box,
                          hypre_Index  stride,
                          HYPRE_Int   *volume_ptr )
{
   HYPRE_Int  volume, d, s, ndim = hypre_BoxNDim(box);

   volume = 1;
   for (d = 0; d < ndim; d++)
   {
      s = hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / hypre_IndexD(stride, d) + 1;
      }
      volume *= s;
   }

   *volume_ptr = volume;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the rank of an index into a multi-D box where the assumed ordering is
 * dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxIndexRank( hypre_Box   *box,
                    hypre_Index  index )
{
   HYPRE_Int  rank, size, d, ndim = hypre_BoxNDim(box);

   rank = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      rank += (hypre_IndexD(index, d) - hypre_BoxIMinD(box, d)) * size;
      size *= hypre_BoxSizeD(box, d);
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * Computes an index into a multi-D box from a rank where the assumed ordering
 * is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxRankIndex( hypre_Box   *box,
                    HYPRE_Int    rank,
                    hypre_Index  index )
{
   HYPRE_Int  d, r, s, ndim = hypre_BoxNDim(box);

   r = rank;
   s = hypre_BoxVolume(box);
   for (d = ndim - 1; d >= 0; d--)
   {
      s = s / hypre_BoxSizeD(box, d);
      hypre_IndexD(index, d) = r / s;
      hypre_IndexD(index, d) += hypre_BoxIMinD(box, d);
      r = r % s;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the distance of an index offset in a multi-D box where the assumed
 * ordering is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxOffsetDistance( hypre_Box   *box,
                         hypre_Index  index )
{
   HYPRE_Int  dist, size, d, ndim = hypre_BoxNDim(box);

   dist = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      dist += hypre_IndexD(index, d) * size;
      size *= hypre_BoxSizeD(box, d);
   }

   return dist;
}

/*--------------------------------------------------------------------------
 * Shift a box by a positive shift
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxShiftPos( hypre_Box   *box,
                   hypre_Index  shift )
{
   HYPRE_Int  d, ndim = hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(box, d) += hypre_IndexD(shift, d);
      hypre_BoxIMaxD(box, d) += hypre_IndexD(shift, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Shift a box by a negative shift
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxShiftNeg( hypre_Box   *box,
                   hypre_Index  shift )
{
   HYPRE_Int  d, ndim = hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(box, d) -= hypre_IndexD(shift, d);
      hypre_BoxIMaxD(box, d) -= hypre_IndexD(shift, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box outward in each dimension as specified by index
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGrowByIndex( hypre_Box   *box,
                      hypre_Index  index )
{
   hypre_IndexRef  imin = hypre_BoxIMin(box);
   hypre_IndexRef  imax = hypre_BoxIMax(box);
   HYPRE_Int       ndim = hypre_BoxNDim(box);
   HYPRE_Int       d, i;

   for (d = 0; d < ndim; d++)
   {
      i = hypre_IndexD(index, d);
      hypre_IndexD(imin, d) -= i;
      hypre_IndexD(imax, d) += i;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box outward by val in each dimension
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGrowByValue( hypre_Box  *box,
                      HYPRE_Int   val )
{
   HYPRE_Int  *imin = hypre_BoxIMin(box);
   HYPRE_Int  *imax = hypre_BoxIMax(box);
   HYPRE_Int   ndim = hypre_BoxNDim(box);
   HYPRE_Int  d;

   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(imin, d) -= val;
      hypre_IndexD(imax, d) += val;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box as specified by array
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGrowByArray( hypre_Box  *box,
                      HYPRE_Int  *array )
{
   HYPRE_Int  *imin = hypre_BoxIMin(box);
   HYPRE_Int  *imax = hypre_BoxIMax(box);
   HYPRE_Int   ndim = hypre_BoxNDim(box);
   HYPRE_Int   d;

   for (d = 0; d < ndim; d++)
   {
      imin[d] -= array[2 * d];
      imax[d] += array[2 * d + 1];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print a box to file
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxPrint( FILE      *file,
                hypre_Box *box )
{
   HYPRE_Int   ndim = hypre_BoxNDim(box);
   HYPRE_Int   d;

   hypre_fprintf(file, "(%d", hypre_BoxIMinD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fprintf(file, ", %d", hypre_BoxIMinD(box, d));
   }
   hypre_fprintf(file, ") x (%d", hypre_BoxIMaxD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fprintf(file, ", %d", hypre_BoxIMaxD(box, d));
   }
   hypre_fprintf(file, ")");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Read a box from file
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxRead( FILE       *file,
               HYPRE_Int   ndim,
               hypre_Box **box_ptr )
{
   hypre_Box  *box;
   HYPRE_Int   d;

   /* Don't create a new box if the output box already exists */
   if (*box_ptr)
   {
      box = *box_ptr;
      hypre_BoxInit(box, ndim);
   }
   else
   {
      box = hypre_BoxCreate(ndim);
   }

   hypre_fscanf(file, "(%d", &hypre_BoxIMinD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fscanf(file, ", %d", &hypre_BoxIMinD(box, d));
   }
   hypre_fscanf(file, ") x (%d", &hypre_BoxIMaxD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fscanf(file, ", %d", &hypre_BoxIMaxD(box, d));
   }
   hypre_fscanf(file, ")");

   *box_ptr = box;

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_BoxArray
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_BoxArrayCreate( HYPRE_Int size,
                      HYPRE_Int ndim )
{
   HYPRE_Int       i;
   hypre_Box      *box;
   hypre_BoxArray *box_array;

   box_array = hypre_TAlloc(hypre_BoxArray,  1, HYPRE_MEMORY_HOST);

   hypre_BoxArrayBoxes(box_array)     = hypre_CTAlloc(hypre_Box,  size, HYPRE_MEMORY_HOST);
   hypre_BoxArraySize(box_array)      = size;
   hypre_BoxArrayAllocSize(box_array) = size;
   hypre_BoxArrayNDim(box_array)      = ndim;
   for (i = 0; i < size; i++)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_BoxNDim(box) = ndim;
   }

   return box_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayDestroy( hypre_BoxArray *box_array )
{
   if (box_array)
   {
      hypre_TFree(hypre_BoxArrayBoxes(box_array), HYPRE_MEMORY_HOST);
      hypre_TFree(box_array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArraySetSize( hypre_BoxArray  *box_array,
                       HYPRE_Int        size      )
{
   HYPRE_Int  alloc_size;

   alloc_size = hypre_BoxArrayAllocSize(box_array);

   if (size > alloc_size)
   {
      HYPRE_Int  i, old_alloc_size, ndim = hypre_BoxArrayNDim(box_array);
      hypre_Box *box;

      old_alloc_size = alloc_size;
      alloc_size = size + hypre_BoxArrayExcess;
      hypre_BoxArrayBoxes(box_array) =
         hypre_TReAlloc(hypre_BoxArrayBoxes(box_array),  hypre_Box,  alloc_size, HYPRE_MEMORY_HOST);
      hypre_BoxArrayAllocSize(box_array) = alloc_size;

      for (i = old_alloc_size; i < alloc_size; i++)
      {
         box = hypre_BoxArrayBox(box_array, i);
         hypre_BoxNDim(box) = ndim;
      }
   }

   hypre_BoxArraySize(box_array) = size;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_BoxArrayDuplicate( hypre_BoxArray *box_array )
{
   hypre_BoxArray  *new_box_array;

   HYPRE_Int        i;

   new_box_array = hypre_BoxArrayCreate(
                      hypre_BoxArraySize(box_array), hypre_BoxArrayNDim(box_array));
   hypre_ForBoxI(i, box_array)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array, i),
                    hypre_BoxArrayBox(new_box_array, i));
   }

   return new_box_array;
}

/*--------------------------------------------------------------------------
 * Append box to the end of box_array.
 * The box_array may be empty.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AppendBox( hypre_Box      *box,
                 hypre_BoxArray *box_array )
{
   HYPRE_Int  size;

   size = hypre_BoxArraySize(box_array);
   hypre_BoxArraySetSize(box_array, (size + 1));
   hypre_CopyBox(box, hypre_BoxArrayBox(box_array, size));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Delete box from box_array.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DeleteBox( hypre_BoxArray *box_array,
                 HYPRE_Int       index     )
{
   HYPRE_Int  i;

   for (i = index; i < hypre_BoxArraySize(box_array) - 1; i++)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array, i + 1),
                    hypre_BoxArrayBox(box_array, i));
   }

   hypre_BoxArraySize(box_array) --;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Deletes boxes corrsponding to indices from box_array.
 * Assumes indices are in ascending order. (AB 11/04)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DeleteMultipleBoxes( hypre_BoxArray *box_array,
                           HYPRE_Int*  indices,
                           HYPRE_Int num )
{
   HYPRE_Int  i, j, start, array_size;

   if (num < 1)
   {
      return hypre_error_flag;
   }

   array_size =  hypre_BoxArraySize(box_array);
   start = indices[0];
   j = 0;

   for (i = start; (i + j) < array_size; i++)
   {
      if (j < num)
      {
         while ((i + j) == indices[j]) /* see if deleting consecutive items */
         {
            j++; /*increase the shift*/
            if (j == num) { break; }
         }
      }

      if ( (i + j) < array_size) /* if deleting the last item then no moving */
      {
         hypre_CopyBox(hypre_BoxArrayBox(box_array, i + j),
                       hypre_BoxArrayBox(box_array, i));
      }
   }

   hypre_BoxArraySize(box_array) = array_size - num;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Append box_array_0 to the end of box_array_1.
 * The box_array_1 may be empty.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AppendBoxArray( hypre_BoxArray *box_array_0,
                      hypre_BoxArray *box_array_1 )
{
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

   return hypre_error_flag;
}

/*==========================================================================
 * Member functions: hypre_BoxArrayArray
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_BoxArrayArrayCreate( HYPRE_Int size,
                           HYPRE_Int ndim )
{
   hypre_BoxArrayArray  *box_array_array;
   HYPRE_Int             i;

   box_array_array = hypre_CTAlloc(hypre_BoxArrayArray,  1, HYPRE_MEMORY_HOST);

   hypre_BoxArrayArrayBoxArrays(box_array_array) =
      hypre_CTAlloc(hypre_BoxArray *,  size, HYPRE_MEMORY_HOST);

   for (i = 0; i < size; i++)
   {
      hypre_BoxArrayArrayBoxArray(box_array_array, i) =
         hypre_BoxArrayCreate(0, ndim);
   }
   hypre_BoxArrayArraySize(box_array_array) = size;
   hypre_BoxArrayArrayNDim(box_array_array) = ndim;

   return box_array_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayArrayDestroy( hypre_BoxArrayArray *box_array_array )
{
   HYPRE_Int  i;

   if (box_array_array)
   {
      hypre_ForBoxArrayI(i, box_array_array)
      hypre_BoxArrayDestroy(
         hypre_BoxArrayArrayBoxArray(box_array_array, i));

      hypre_TFree(hypre_BoxArrayArrayBoxArrays(box_array_array), HYPRE_MEMORY_HOST);
      hypre_TFree(box_array_array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box_array_array.
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
   new_box_array_array = hypre_BoxArrayArrayCreate(
                            new_size, hypre_BoxArrayArrayNDim(box_array_array));

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
