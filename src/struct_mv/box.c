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
hypre_CopyToIndex( hypre_Index  in_index,
                   HYPRE_Int    ndim,
                   hypre_Index  out_index )
{
   HYPRE_Int d;
   for (d = 0; d < ndim; d++)
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

   hypre_fprintf(file, "(%d", hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fprintf(file, ", %d", hypre_IndexD(index, d));
   }
   hypre_fprintf(file, ")");

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

   hypre_fscanf(file, "(%d", &hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      hypre_fscanf(file, ", %d", &hypre_IndexD(index, d));
   }
   hypre_fscanf(file, ")");

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

   box = hypre_CTAlloc(hypre_Box, 1, HYPRE_MEMORY_HOST);
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
hypre_BoxClone( hypre_Box *box )
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
   HYPRE_Real   volume;
   HYPRE_Int    d, ndim = hypre_BoxNDim(box);

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
hypre_BoxStrideVolume( hypre_Box   *box,
                       hypre_Index  stride)
{
   HYPRE_Int  ndim = hypre_BoxNDim(box);
   HYPRE_Int  volume, d, s;

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

   return volume;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxPartialVolume( hypre_Box   *box,
                        hypre_Index  partial_volume)
{
   HYPRE_Int d, ndim = hypre_BoxNDim(box);

   partial_volume[0] = hypre_BoxSizeD(box, 0);
   for (d = 1; d < ndim; d++)
   {
      partial_volume[d] = partial_volume[d - 1] * hypre_BoxSizeD(box, d);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxNnodes( hypre_Box *box )
{
   HYPRE_Int nnodes, d, ndim = hypre_BoxNDim(box);

   nnodes = 1;
   for (d = 0; d < ndim; d++)
   {
      nnodes *= (hypre_BoxSizeD(box, d) + 1);
   }

   return nnodes;
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
hypre_BoxMaxSize( hypre_Box   *box)
{
   HYPRE_Int  d, ndim = hypre_BoxNDim(box);
   HYPRE_Int  max;

   max = 0;
   for (d = 0; d < ndim; d++)
   {
      max = hypre_max(max, hypre_BoxSizeD(box, d));
   }

   return max;
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

   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(size, d) = 0;
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

   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      hypre_IndexD(size, d) = 0;
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
 * Grow a box outward to encompass a second box
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxGrowByBox( hypre_Box  *box,
                    hypre_Box  *gbox )
{
   HYPRE_Int  d, ndim = hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(box, d) =
         hypre_min(hypre_BoxIMinD(box, d), hypre_BoxIMinD(gbox, d));
      hypre_BoxIMaxD(box, d) =
         hypre_max(hypre_BoxIMaxD(box, d), hypre_BoxIMaxD(gbox, d));
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

   box_array = hypre_TAlloc(hypre_BoxArray, 1, HYPRE_MEMORY_HOST);

   hypre_BoxArrayBoxes(box_array)     = hypre_CTAlloc(hypre_Box, size, HYPRE_MEMORY_HOST);
   hypre_BoxArraySize(box_array)      = size;
   hypre_BoxArrayAllocSize(box_array) = size;
   hypre_BoxArrayNDim(box_array)      = ndim;
   hypre_BoxArrayIDs(box_array)       = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      hypre_BoxArrayID(box_array, i) = i;
      box = hypre_BoxArrayBox(box_array, i);
      hypre_BoxNDim(box) = ndim;
   }

   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_BoxArrayCreateFromIndices
 *
 * Build an array of boxes [box_array_ptr] spanning input indices [indices_in]
 *
 * Notes:
 *    1) indices_in is a (ndim x num_indices) two-dimensional array
 *
 * This is based on the Berger-Rigoutsos algorithm.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayCreateFromIndices( HYPRE_Int         ndim,
                                 HYPRE_Int         num_indices_in,
                                 HYPRE_Int       **indices_in,
                                 HYPRE_Real        threshold,
                                 hypre_BoxArray  **box_array_ptr )
{
   /* Data structures */
   hypre_BoxBinTree  *boxbt;
   hypre_BoxBTNode   *btnode;
   hypre_BoxBTNode   *lnode;
   hypre_BoxBTNode   *rnode;
   hypre_BoxBTQueue  *btqueue;
   hypre_BoxArray    *box_array;

   /* Local variables */
   HYPRE_Int         *signature[HYPRE_MAXDIM];
   HYPRE_Int         *laplacian[HYPRE_MAXDIM];
   HYPRE_Int          cut[HYPRE_MAXDIM];
   HYPRE_Int          direction[HYPRE_MAXDIM];
   HYPRE_Int          sign[HYPRE_MAXDIM];
   HYPRE_Int          signcoord[HYPRE_MAXDIM];

   hypre_Box         *box;
   hypre_Box         *bbox;

   HYPRE_Int         *indices[HYPRE_MAXDIM];
   HYPRE_Int         *lbox_indices[HYPRE_MAXDIM];
   HYPRE_Int         *rbox_indices[HYPRE_MAXDIM];
   HYPRE_Int          splitdir, dir, sign_change, d, i;
   HYPRE_Int          index, size, capacity, change;
   HYPRE_Int          num_indices;
   HYPRE_Int          num_lbox_indices, num_rbox_indices;
   HYPRE_Int          cut_by_hole;
   HYPRE_Real         box_efficiency;
   HYPRE_Real         box_dvolume;
   HYPRE_Real         box_minvol;

   /* Exit in trivial case */
   if (num_indices_in <= 0)
   {
      return hypre_error_flag;
   }

   /* Set defaults */
   box_minvol = 1;

   /* Compute bounding box */
   bbox = hypre_BoxCreate(ndim);
   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(bbox, d) = HYPRE_INT_MAX;
      hypre_BoxIMaxD(bbox, d) = - HYPRE_INT_MAX;
      for (i = 0; i < num_indices_in; i++)
      {
         hypre_BoxIMinD(bbox, d) = hypre_min(hypre_BoxIMinD(bbox, d),
                                             indices_in[d][i]);
         hypre_BoxIMaxD(bbox, d) = hypre_max(hypre_BoxIMaxD(bbox, d),
                                             indices_in[d][i]);
      }
   }

   /* Exit in the trivial case */
   box_dvolume = hypre_doubleBoxVolume(bbox);
   box_efficiency = (HYPRE_Real) num_indices_in / box_dvolume;
   if ((box_efficiency >= threshold) || (box_dvolume <= box_minvol))
   {
      if (*box_array_ptr == NULL)
      {
         *box_array_ptr = hypre_BoxArrayCreate(0, ndim);
      }
      hypre_AppendBox(bbox, *box_array_ptr);
      hypre_BoxDestroy(bbox);

      return hypre_error_flag;
   }

   /* Allocate memory */
   capacity = 0;
   for (d = 0; d < ndim; d++)
   {
      size = hypre_BoxSizeD(bbox, d);
      signature[d] = hypre_CTAlloc(HYPRE_Int, size + 2, HYPRE_MEMORY_HOST);
      laplacian[d] = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
      lbox_indices[d] = hypre_CTAlloc(HYPRE_Int, num_indices_in, HYPRE_MEMORY_HOST);
      rbox_indices[d] = hypre_CTAlloc(HYPRE_Int, num_indices_in, HYPRE_MEMORY_HOST);
      capacity += hypre_Log2(size);
   }
   hypre_BoxBinTreeCreate(ndim, &boxbt);
   hypre_BoxBTQueueCreate(&btqueue);

   /* Initialize data */
   hypre_BoxBinTreeInitialize(boxbt, num_indices_in, indices_in, bbox);
   hypre_BoxBTQueueInitialize(capacity, btqueue);
   hypre_BoxBTQueueInsert(hypre_BoxBinTreeRoot(boxbt), btqueue);

   /* Create output BoxArray */
   if (*box_array_ptr)
   {
      box_array = *box_array_ptr;
   }
   else
   {
      box_array = hypre_BoxArrayCreate(capacity, ndim);
      hypre_BoxArraySetSize(box_array, 0);
   }

   /* level order traversal */
   while (hypre_BoxBTQueueSize(btqueue) > 0)
   {
      /* Retrieve node data */
      hypre_BoxBTQueueDelete(btqueue, &btnode);
      box = hypre_BoxBTNodeBox(btnode);
      num_indices = hypre_BoxBTNodeNumIndices(btnode);
      for (d = 0; d < ndim; d++)
      {
         indices[d] = hypre_BoxBTNodeIndices(btnode, d);
      }
      hypre_assert(num_indices > 0);

      /* Update bounding box */
      for (d = 0; d < ndim; d++)
      {
         hypre_BoxIMinD(box, d) = HYPRE_INT_MAX;
         hypre_BoxIMaxD(box, d) = - HYPRE_INT_MAX;
         for (i = 0; i < num_indices; i++)
         {
            hypre_BoxIMinD(box, d) = hypre_min(hypre_BoxIMinD(box, d),
                                               indices[d][i]);
            hypre_BoxIMaxD(box, d) = hypre_max(hypre_BoxIMaxD(box, d),
                                               indices[d][i]);
         }
      }

      /* Compute box efficiency */
      box_dvolume = hypre_doubleBoxVolume(box);
      if (box_dvolume > 0.0)
      {
         box_efficiency = (HYPRE_Real) num_indices / box_dvolume;
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Found non-positive box volume!");
         return hypre_error_flag;
      }

      /* Decide wheter to split the box or not */
      if (box_efficiency < threshold && box_dvolume > box_minvol)
      {
         /* Build direction array */
         direction[0] = 0;
         for (d = 1; d < ndim; d++)
         {
            size = hypre_BoxSizeD(box, d);
            direction[d] = d;
            for (i = 0; i < d; i++)
            {
               if (size > hypre_BoxSizeD(box, i))
               {
                  hypre_swap(direction, i, d);
               }
            }
         }

         /* Compute signatures */
         for (d = 0; d < ndim; d++)
         {
            signature[d][0] = 0;
            for (i = 0; i < num_indices; i++)
            {
               index = indices[d][i] - hypre_BoxIMinD(box, d);
               signature[d][index + 1]++;
            }
            index = hypre_BoxSizeD(box, d);
            signature[d][index + 1] = 0;
         }

         /* Look for holes */
         hypre_SetIndex(cut, HYPRE_INT_MAX);
         cut_by_hole = 0;
         for (d = 0; d < ndim; d++)
         {
            dir = direction[d];
            for (i = 0; i < hypre_BoxSizeD(box, dir); i++)
            {
               if (signature[dir][i + 1] == 0)
               {
                  hypre_IndexD(cut, dir) = i + hypre_BoxIMinD(box, dir);
                  splitdir = dir;
                  break;
               }
            }
            if (hypre_IndexD(cut, dir) != HYPRE_INT_MAX)
            {
               cut_by_hole = 1;
               break;
            }
         }

         /* Look for inflection points in the laplacian */
         if (!cut_by_hole)
         {
            hypre_SetIndex(sign, 0);
            hypre_CopyToIndex(hypre_BoxIMin(box), ndim, signcoord);

            /* Compute laplacian */
            for (d = 0; d < ndim; d++)
            {
               for (i = 1; i < hypre_BoxSizeD(box, d) + 1; i++)
               {
                  laplacian[d][i - 1] = signature[d][i - 1] +
                                        signature[d][i + 1] -
                                        2 * signature[d][i];
               }

               /* Look for largest change in the current direction */
               for (i = 0; i < hypre_BoxSizeD(box, d) - 1; i++)
               {
                  if ((laplacian[d][i + 1] >= 0) != (laplacian[d][i] >= 0))
                  {
                     change = hypre_abs(laplacian[d][i + 1] - laplacian[d][i]);
                     if (change > hypre_IndexD(sign, d))
                     {
                        hypre_IndexD(sign, d) = change;
                        hypre_IndexD(signcoord, d) = i + hypre_BoxIMinD(box, d);
                     }
                  }
               }
            }

            /* Look for largest sign change among all directions */
            dir = 0;
            sign_change = 0;
            for (d = 1; d < ndim; d++)
            {
               if (hypre_IndexD(sign, dir) < hypre_IndexD(sign, d))
               {
                  dir = d;
                  sign_change = 1;
               }
            }

            /* Set cut direction and coordinate */
            hypre_IndexD(cut, dir) = hypre_IndexD(signcoord, dir);
            splitdir = dir;

            /* If no change of sign in the Laplacian was found, just cut the longest dim in half */
            if (!sign_change)
            {
               change = 0;
               for (d = 0; d < ndim; d++)
               {
                  if (change < hypre_BoxSizeD(box, d))
                  {
                     change = hypre_BoxSizeD(box, d);
                     dir = d;
                  }
               }
               hypre_IndexD(cut, dir) = (hypre_BoxIMinD(box, dir) + hypre_BoxIMaxD(box, dir)) / 2;
               splitdir = dir;
            }
         }
         hypre_assert((splitdir >= 0) && (splitdir < ndim));

         /* Create left/right nodes */
         hypre_BoxBTNodeCreate(ndim, &lnode);
         hypre_BoxBTNodeCreate(ndim, &rnode);
         hypre_BoxBTNodeLeft(btnode)  = lnode;
         hypre_BoxBTNodeRight(btnode) = rnode;

         /* Split indices */
         num_lbox_indices = 0;
         num_rbox_indices = 0;
         for (i = 0; i < num_indices; i++)
         {
            index = indices[splitdir][i];
            if (index <= hypre_IndexD(cut, splitdir))
            {
               for (d = 0; d < ndim; d++)
               {
                  lbox_indices[d][num_lbox_indices] = indices[d][i];
               }
               num_lbox_indices++;
            }
            else
            {
               for (d = 0; d < ndim; d++)
               {
                  rbox_indices[d][num_rbox_indices] = indices[d][i];
               }
               num_rbox_indices++;
            }
         }
         hypre_assert(num_lbox_indices > 0);
         hypre_assert(num_rbox_indices > 0);

         /* Copy splitted indices to leaf nodes */
         hypre_BoxBTNodeSetIndices(lnode, num_lbox_indices, lbox_indices);
         hypre_BoxBTNodeSetIndices(rnode, num_rbox_indices, rbox_indices);

         /* Insert newly created nodes to queue */
         hypre_BoxBTQueueInsert(lnode, btqueue);
         hypre_BoxBTQueueInsert(rnode, btqueue);

         /* Reset signatures */
         for (d = 0; d < ndim; d++)
         {
            for (i = 0; i < num_indices; i++)
            {
               index = indices[d][i] - hypre_BoxIMinD(box, d);
               signature[d][index + 1] = 0;
            }
         }
      }
      else
      {
         hypre_AppendBox(hypre_BoxBTNodeBox(btnode), box_array);
      } /* if (box_efficiency < threshold) */
   } /* while (hypre_BoxBTQueueSize(btqueue) > 0) */

   /* Set pointer to output */
   *box_array_ptr = box_array;

   /* Free memory */
   for (d = 0; d < ndim; d++)
   {
      hypre_TFree(signature[d], HYPRE_MEMORY_HOST);
      hypre_TFree(laplacian[d], HYPRE_MEMORY_HOST);
      hypre_TFree(lbox_indices[d], HYPRE_MEMORY_HOST);
      hypre_TFree(rbox_indices[d], HYPRE_MEMORY_HOST);
   }
   hypre_BoxDestroy(bbox);
   hypre_BoxBinTreeDestroy(boxbt);
   hypre_BoxBTQueueDestroy(btqueue);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayDestroy( hypre_BoxArray *box_array )
{
   if (box_array)
   {
      hypre_TFree(hypre_BoxArrayBoxes(box_array), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_BoxArrayIDs(box_array), HYPRE_MEMORY_HOST);
      hypre_TFree(box_array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayPrintToFile( FILE            *file,
                           hypre_BoxArray  *box_array )
{
   hypre_Box *box;

   HYPRE_Int  ndim;
   HYPRE_Int  size;
   HYPRE_Int  i;

   ndim = hypre_BoxArrayNDim(box_array);
   hypre_fprintf(file, "%d\n", ndim);

   size = hypre_BoxArraySize(box_array);
   hypre_fprintf(file, "%d\n", size);

   /* Print lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_fprintf(file, "%d:  ", i);
      hypre_IndexPrint(file, ndim, hypre_BoxIMin(box));
      hypre_fprintf(file, "  x  ");
      hypre_IndexPrint(file, ndim, hypre_BoxIMax(box));
      hypre_fprintf(file, "\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayReadFromFile( FILE             *file,
                            hypre_BoxArray  **box_array_ptr )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;

   HYPRE_Int  ndim;
   HYPRE_Int  size;
   HYPRE_Int  i, idummy;

   hypre_fscanf(file, "%d\n", &ndim);

   hypre_fscanf(file, "%d\n", &size);

   box_array = hypre_BoxArrayCreate(size, ndim);

   /* Print lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   hypre_ForBoxI(i, box_array)
   {
      box = hypre_BoxArrayBox(box_array, i);
      hypre_fscanf(file, "%d:  ", &idummy);
      hypre_IndexRead(file, ndim, hypre_BoxIMin(box));
      hypre_fscanf(file, "  x  ");
      hypre_IndexRead(file, ndim, hypre_BoxIMax(box));
      hypre_fprintf(file, "\n");
   }

   *box_array_ptr = box_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayPrint( MPI_Comm         comm,
                     const char      *filename,
                     hypre_BoxArray  *box_array )
{
   FILE      *file;
   char       new_filename[255];

   HYPRE_Int  myid;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   hypre_BoxArrayPrintToFile(file, box_array);

   fflush(file);
   fclose(file);

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
      hypre_BoxArrayBoxes(box_array) = hypre_TReAlloc(hypre_BoxArrayBoxes(box_array),
                                                      hypre_Box, alloc_size, HYPRE_MEMORY_HOST);
      hypre_BoxArrayIDs(box_array) = hypre_TReAlloc(hypre_BoxArrayIDs(box_array),
                                                    HYPRE_Int, alloc_size, HYPRE_MEMORY_HOST);
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
hypre_BoxArrayClone( hypre_BoxArray *box_array )
{
   hypre_BoxArray  *new_box_array;
   HYPRE_Int        size;
   HYPRE_Int        ndim;
   HYPRE_Int        i;

   hypre_assert(box_array != NULL);

   ndim = hypre_BoxArrayNDim(box_array);
   size = hypre_BoxArraySize(box_array);
   new_box_array = hypre_BoxArrayCreate(size, ndim);
   hypre_ForBoxI(i, box_array)
   {
      hypre_CopyBox(hypre_BoxArrayBox(box_array, i),
                    hypre_BoxArrayBox(new_box_array, i));
      hypre_BoxArrayID(new_box_array, i) = hypre_BoxArrayID(box_array, i);
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
      hypre_BoxArrayID(box_array, i) = hypre_BoxArrayID(box_array, i + 1);
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
                           HYPRE_Int      *indices,
                           HYPRE_Int       num )
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

      if ((i + j) < array_size) /* if deleting the last item then no moving */
      {
         hypre_CopyBox(hypre_BoxArrayBox(box_array, i + j),
                       hypre_BoxArrayBox(box_array, i));
         hypre_BoxArrayID(box_array, i) = hypre_BoxArrayID(box_array, i + j);
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayVolume( hypre_BoxArray *box_array )
{
   HYPRE_Int  volume, i;

   volume = 0;
   hypre_ForBoxI(i, box_array)
   {
      volume += hypre_BoxVolume(hypre_BoxArrayBox(box_array, i));
   }

   return volume;
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

   box_array_array = hypre_CTAlloc(hypre_BoxArrayArray, 1, HYPRE_MEMORY_HOST);

   hypre_BoxArrayArrayBoxArrays(box_array_array) =
      hypre_CTAlloc(hypre_BoxArray *, size, HYPRE_MEMORY_HOST);

   for (i = 0; i < size; i++)
   {
      hypre_BoxArrayArrayBoxArray(box_array_array, i) =
         hypre_BoxArrayCreate(0, ndim);
   }
   hypre_BoxArrayArraySize(box_array_array) = size;
   hypre_BoxArrayArrayAllocSize(box_array_array) = size;
   hypre_BoxArrayArrayNDim(box_array_array) = ndim;
   hypre_BoxArrayArrayIDs(box_array_array) =
      hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);

   return box_array_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayArrayDestroy( hypre_BoxArrayArray *box_array_array )
{
   HYPRE_Int  alloc_size;
   HYPRE_Int  i;

   if (box_array_array)
   {
      alloc_size = hypre_BoxArrayArrayAllocSize(box_array_array);

      for (i = 0; i < alloc_size; i++)
      {
         hypre_BoxArrayDestroy(
            hypre_BoxArrayArrayBoxArray(box_array_array, i));
      }

      hypre_TFree(hypre_BoxArrayArrayIDs(box_array_array), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_BoxArrayArrayBoxArrays(box_array_array), HYPRE_MEMORY_HOST);
      hypre_TFree(box_array_array, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box_array_array.
 *--------------------------------------------------------------------------*/

hypre_BoxArrayArray *
hypre_BoxArrayArrayClone( hypre_BoxArrayArray *box_array_array )
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
         hypre_BoxArrayArrayID(new_box_array_array, i) =
            hypre_BoxArrayArrayID(box_array_array, i);
      }
   }

   return new_box_array_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayArrayPrintToFile( FILE                 *file,
                                hypre_BoxArrayArray  *box_array_array )
{
   hypre_BoxArray  *box_array;

   HYPRE_Int        ndim;
   HYPRE_Int        size;
   HYPRE_Int        id;
   HYPRE_Int        i;

   ndim = hypre_BoxArrayArrayNDim(box_array_array);
   hypre_fprintf(file, "%d\n", ndim);

   size = hypre_BoxArrayArraySize(box_array_array);
   hypre_fprintf(file, "%d\n", size);

   hypre_ForBoxArrayI(i, box_array_array)
   {
      box_array = hypre_BoxArrayArrayBoxArray(box_array_array, i);
      id = hypre_BoxArrayArrayID(box_array_array, i);
      hypre_fprintf(file, "BoxArray %d, ID %d\n", i, id);

      hypre_BoxArrayPrintToFile(file, box_array);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoxArrayArrayPrint( MPI_Comm              comm,
                          const char           *filename,
                          hypre_BoxArrayArray  *box_array_array )
{
   FILE      *file;
   char       new_filename[255];

   HYPRE_Int  myid;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   hypre_BoxArrayArrayPrintToFile(file, box_array_array);

   fflush(file);
   fclose(file);

   return hypre_error_flag;
}
