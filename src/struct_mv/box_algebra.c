/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Box class:
 *   Box algebra functions.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Intersect box1 and box2.
 * If the boxes do not intersect, the result is a box with zero volume.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntersectBoxes( hypre_Box *box1,
                      hypre_Box *box2,
                      hypre_Box *ibox )
{
   HYPRE_Int d, ndim = hypre_BoxNDim(box1);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(ibox, d) =
         hypre_max(hypre_BoxIMinD(box1, d), hypre_BoxIMinD(box2, d));
      hypre_BoxIMaxD(ibox, d) =
         hypre_min(hypre_BoxIMaxD(box1, d), hypre_BoxIMaxD(box2, d));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute (box1 - box2) and append result to box_array.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SubtractBoxes( hypre_Box      *box1,
                     hypre_Box      *box2,
                     hypre_BoxArray *box_array )
{
   HYPRE_Int   d, size, maxboxes, ndim = hypre_BoxNDim(box1);
   hypre_Box  *box;
   hypre_Box  *rembox;

   /*------------------------------------------------------
    * Set the box array size to the maximum possible,
    * plus one, to have space for the remainder box.
    *------------------------------------------------------*/

   maxboxes = 2 * ndim;

   size = hypre_BoxArraySize(box_array);
   hypre_BoxArraySetSize(box_array, (size + maxboxes + 1));

   /*------------------------------------------------------
    * Subtract the boxes by cutting box1 in x, y, then z
    *------------------------------------------------------*/

   rembox = hypre_BoxArrayBox(box_array, (size + maxboxes));
   hypre_CopyBox(box1, rembox);

   for (d = 0; d < ndim; d++)
   {
      /* if the boxes do not intersect, the subtraction is trivial */
      if ( (hypre_BoxIMinD(box2, d) > hypre_BoxIMaxD(rembox, d)) ||
           (hypre_BoxIMaxD(box2, d) < hypre_BoxIMinD(rembox, d)) )
      {
         size = hypre_BoxArraySize(box_array) - maxboxes - 1;
         hypre_CopyBox(box1, hypre_BoxArrayBox(box_array, size));
         size++;
         break;
      }

      /* update the box array */
      else
      {
         if ( hypre_BoxIMinD(box2, d) > hypre_BoxIMinD(rembox, d) )
         {
            box = hypre_BoxArrayBox(box_array, size);
            hypre_CopyBox(rembox, box);
            hypre_BoxIMaxD(box, d) = hypre_BoxIMinD(box2, d) - 1;
            hypre_BoxIMinD(rembox, d) = hypre_BoxIMinD(box2, d);
            if ( hypre_BoxVolume(box) > 0 ) { size++; }
         }
         if ( hypre_BoxIMaxD(box2, d) < hypre_BoxIMaxD(rembox, d) )
         {
            box = hypre_BoxArrayBox(box_array, size);
            hypre_CopyBox(rembox, box);
            hypre_BoxIMinD(box, d) = hypre_BoxIMaxD(box2, d) + 1;
            hypre_BoxIMaxD(rembox, d) = hypre_BoxIMaxD(box2, d);
            if ( hypre_BoxVolume(box) > 0 ) { size++; }
         }
      }
   }
   hypre_BoxArraySetSize(box_array, size);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute (box_array1 - box_array2) and replace box_array1 with result.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SubtractBoxArrays( hypre_BoxArray *box_array1,
                         hypre_BoxArray *box_array2,
                         hypre_BoxArray *tmp_box_array )
{
   hypre_BoxArray *diff_boxes     = box_array1;
   hypre_BoxArray *new_diff_boxes = tmp_box_array;
   hypre_BoxArray  box_array;
   hypre_Box      *box1;
   hypre_Box      *box2;
   HYPRE_Int       i, k;

   hypre_ForBoxI(i, box_array2)
   {
      box2 = hypre_BoxArrayBox(box_array2, i);

      /* compute new_diff_boxes = (diff_boxes - box2) */
      hypre_BoxArraySetSize(new_diff_boxes, 0);
      hypre_ForBoxI(k, diff_boxes)
      {
         box1 = hypre_BoxArrayBox(diff_boxes, k);
         hypre_SubtractBoxes(box1, box2, new_diff_boxes);
      }

      /* swap internals of diff_boxes and new_diff_boxes */
      box_array       = *new_diff_boxes;
      *new_diff_boxes = *diff_boxes;
      *diff_boxes     = box_array;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Avoid using - this only works for ndim < 4
 *
 * Compute the union of all boxes.
 *
 * To compute the union, we first construct a logically rectangular,
 * variably spaced, 3D grid called block.  Each cell (i,j,k) of block
 * corresponds to a box with extents given by
 *
 *   iminx = block_index[0][i]
 *   iminy = block_index[1][j]
 *   iminz = block_index[2][k]
 *   imaxx = block_index[0][i+1] - 1
 *   imaxy = block_index[1][j+1] - 1
 *   imaxz = block_index[2][k+1] - 1
 *
 * The size of block is given by
 *
 *   sizex = block_sz[0]
 *   sizey = block_sz[1]
 *   sizez = block_sz[2]
 *
 * We initially set all cells of block that are part of the union to
 *
 *   factor[2] + factor[1] + factor[0]
 *
 * where
 *
 *   factor[0] = 1;
 *   factor[1] = (block_sz[0] + 1);
 *   factor[2] = (block_sz[1] + 1) * factor[1];
 *
 * The cells of block are then "joined" in x first, then y, then z.
 * The result is that each nonzero entry of block corresponds to a
 * box in the union with extents defined by factoring the entry, then
 * indexing into the block_index array.
 *
 * Note: Special care has to be taken for boxes of size 0.
 *
 *--------------------------------------------------------------------------*/

/* ONLY3D */

HYPRE_Int
hypre_UnionBoxes( hypre_BoxArray *boxes )
{
   hypre_Box       *box;

   HYPRE_Int       *block_index[3];
   HYPRE_Int        block_sz[3], block_volume;
   HYPRE_Int       *block;
   HYPRE_Int        index;
   HYPRE_Int        size;
   HYPRE_Int        factor[3];

   HYPRE_Int        iminmax[2], imin[3], imax[3];
   HYPRE_Int        ii[3], dd[3];
   HYPRE_Int        join;
   HYPRE_Int        i_tmp0, i_tmp1;
   HYPRE_Int        ioff, joff, koff;
   HYPRE_Int        bi, d, i, j, k;

   HYPRE_Int        index_not_there;

   /*------------------------------------------------------
    * If the size of boxes is less than 2, return
    *------------------------------------------------------*/

   if (hypre_BoxArraySize(boxes) < 2)
   {
      return hypre_error_flag;
   }

   /*------------------------------------------------------
    * Set up the block_index array
    *------------------------------------------------------*/

   i_tmp0 = 2 * hypre_BoxArraySize(boxes);
   block_index[0] = hypre_TAlloc(HYPRE_Int,  3 * i_tmp0, HYPRE_MEMORY_HOST);
   block_sz[0] = 0;
   for (d = 1; d < 3; d++)
   {
      block_index[d] = block_index[d - 1] + i_tmp0;
      block_sz[d] = 0;
   }

   hypre_ForBoxI(bi, boxes)
   {
      box = hypre_BoxArrayBox(boxes, bi);

      for (d = 0; d < 3; d++)
      {
         iminmax[0] = hypre_BoxIMinD(box, d);
         iminmax[1] = hypre_BoxIMaxD(box, d) + 1;

         for (i = 0; i < 2; i++)
         {
            /* find the new index position in the block_index array */
            index_not_there = 1;
            for (j = 0; j < block_sz[d]; j++)
            {
               if (iminmax[i] <= block_index[d][j])
               {
                  if (iminmax[i] == block_index[d][j])
                  {
                     index_not_there = 0;
                  }
                  break;
               }
            }

            /* if the index is already there, don't add it again */
            if (index_not_there)
            {
               for (k = block_sz[d]; k > j; k--)
               {
                  block_index[d][k] = block_index[d][k - 1];
               }
               block_index[d][j] = iminmax[i];
               block_sz[d]++;
            }
         }
      }
   }

   for (d = 0; d < 3; d++)
   {
      block_sz[d]--;
   }
   block_volume = block_sz[0] * block_sz[1] * block_sz[2];

   /*------------------------------------------------------
    * Set factor values
    *------------------------------------------------------*/

   factor[0] = 1;
   factor[1] = (block_sz[0] + 1);
   factor[2] = (block_sz[1] + 1) * factor[1];

   /*------------------------------------------------------
    * Set up the block array
    *------------------------------------------------------*/

   block = hypre_CTAlloc(HYPRE_Int,  block_volume, HYPRE_MEMORY_HOST);

   hypre_ForBoxI(bi, boxes)
   {
      box = hypre_BoxArrayBox(boxes, bi);

      /* find the block_index indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
         j = 0;

         while (hypre_BoxIMinD(box, d) != block_index[d][j])
         {
            j++;
         }
         imin[d] = j;

         while (hypre_BoxIMaxD(box, d) + 1 != block_index[d][j])
         {
            j++;
         }
         imax[d] = j;
      }

      /* note: boxes of size zero will not be added to block */
      for (k = imin[2]; k < imax[2]; k++)
      {
         for (j = imin[1]; j < imax[1]; j++)
         {
            for (i = imin[0]; i < imax[0]; i++)
            {
               index = ((k) * block_sz[1] + j) * block_sz[0] + i;

               block[index] = factor[2] + factor[1] + factor[0];
            }
         }
      }
   }

   /*------------------------------------------------------
    * Join block array in x, then y, then z
    *
    * Notes:
    *   - ii[0], ii[1], and ii[2] correspond to indices
    *     in x, y, and z respectively.
    *   - dd specifies the order in which to loop over
    *     the three dimensions.
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      switch (d)
      {
         case 0: /* join in x */
            dd[0] = 0;
            dd[1] = 1;
            dd[2] = 2;
            break;

         case 1: /* join in y */
            dd[0] = 1;
            dd[1] = 0;
            dd[2] = 2;
            break;

         case 2: /* join in z */
            dd[0] = 2;
            dd[1] = 1;
            dd[2] = 0;
            break;
      }

      for (ii[dd[2]] = 0; ii[dd[2]] < block_sz[dd[2]]; ii[dd[2]]++)
      {
         for (ii[dd[1]] = 0; ii[dd[1]] < block_sz[dd[1]]; ii[dd[1]]++)
         {
            join = 0;
            for (ii[dd[0]] = 0; ii[dd[0]] < block_sz[dd[0]]; ii[dd[0]]++)
            {
               index = ((ii[2]) * block_sz[1] + ii[1]) * block_sz[0] + ii[0];

               if ((join) && (block[index] == i_tmp1))
               {
                  block[index]  = 0;
                  block[i_tmp0] += factor[dd[0]];
               }
               else
               {
                  if (block[index])
                  {
                     i_tmp0 = index;
                     i_tmp1 = block[index];
                     join  = 1;
                  }
                  else
                  {
                     join = 0;
                  }
               }
            }
         }
      }
   }

   /*------------------------------------------------------
    * Set up the boxes BoxArray
    *------------------------------------------------------*/

   size = 0;
   for (index = 0; index < block_volume; index++)
   {
      if (block[index])
      {
         size++;
      }
   }
   hypre_BoxArraySetSize(boxes, size);

   index = 0;
   size = 0;
   for (k = 0; k < block_sz[2]; k++)
   {
      for (j = 0; j < block_sz[1]; j++)
      {
         for (i = 0; i < block_sz[0]; i++)
         {
            if (block[index])
            {
               ioff = (block[index] % factor[1])            ;
               joff = (block[index] % factor[2]) / factor[1];
               koff = (block[index]            ) / factor[2];

               box = hypre_BoxArrayBox(boxes, size);
               hypre_BoxIMinD(box, 0) = block_index[0][i];
               hypre_BoxIMinD(box, 1) = block_index[1][j];
               hypre_BoxIMinD(box, 2) = block_index[2][k];
               hypre_BoxIMaxD(box, 0) = block_index[0][i + ioff] - 1;
               hypre_BoxIMaxD(box, 1) = block_index[1][j + joff] - 1;
               hypre_BoxIMaxD(box, 2) = block_index[2][k + koff] - 1;

               size++;
            }

            index++;
         }
      }
   }

   /*---------------------------------------------------------
    * Clean up and return
    *---------------------------------------------------------*/

   hypre_TFree(block_index[0], HYPRE_MEMORY_HOST);
   hypre_TFree(block, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Avoid using - this only works for ndim < 4
 *
 * Compute the union of all boxes such that the minimum number of boxes is
 * generated. Accomplished by making six calls to hypre_UnionBoxes and then
 * taking the union that has the least no. of boxes. The six calls union in the
 * order xzy, yzx, yxz, zxy, zyx, xyz
 *--------------------------------------------------------------------------*/

/* ONLY3D */

HYPRE_Int
hypre_MinUnionBoxes( hypre_BoxArray *boxes )
{
   hypre_BoxArrayArray     *rotated_array;
   hypre_BoxArray          *rotated_boxes;
   hypre_Box               *box, *rotated_box;
   hypre_Index              lower, upper;

   HYPRE_Int                i, j, size, min_size, array;

   size = hypre_BoxArraySize(boxes);
   rotated_box = hypre_CTAlloc(hypre_Box,  1, HYPRE_MEMORY_HOST);
   rotated_array = hypre_BoxArrayArrayCreate(5, hypre_BoxArrayNDim(boxes));

   for (i = 0; i < 5; i++)
   {
      rotated_boxes = hypre_BoxArrayArrayBoxArray(rotated_array, i);
      switch (i)
      {
         case 0:
            for (j = 0; j < size; j++)
            {
               box = hypre_BoxArrayBox(boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(box)[0],  hypre_BoxIMin(box)[2],
                               hypre_BoxIMin(box)[1]);
               hypre_SetIndex3(upper, hypre_BoxIMax(box)[0],  hypre_BoxIMax(box)[2],
                               hypre_BoxIMax(box)[1]);
               hypre_BoxSetExtents(rotated_box, lower, upper);
               hypre_AppendBox(rotated_box, rotated_boxes);
            }
            hypre_UnionBoxes(rotated_boxes);
            break;

         case 1:
            for (j = 0; j < size; j++)
            {
               box = hypre_BoxArrayBox(boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(box)[1],  hypre_BoxIMin(box)[2],
                               hypre_BoxIMin(box)[0]);
               hypre_SetIndex3(upper, hypre_BoxIMax(box)[1],  hypre_BoxIMax(box)[2],
                               hypre_BoxIMax(box)[0]);
               hypre_BoxSetExtents(rotated_box, lower, upper);
               hypre_AppendBox(rotated_box, rotated_boxes);
            }
            hypre_UnionBoxes(rotated_boxes);
            break;

         case 2:
            for (j = 0; j < size; j++)
            {
               box = hypre_BoxArrayBox(boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(box)[1],  hypre_BoxIMin(box)[0],
                               hypre_BoxIMin(box)[2]);
               hypre_SetIndex3(upper, hypre_BoxIMax(box)[1],  hypre_BoxIMax(box)[0],
                               hypre_BoxIMax(box)[2]);
               hypre_BoxSetExtents(rotated_box, lower, upper);
               hypre_AppendBox(rotated_box, rotated_boxes);
            }
            hypre_UnionBoxes(rotated_boxes);
            break;

         case 3:
            for (j = 0; j < size; j++)
            {
               box = hypre_BoxArrayBox(boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(box)[2],  hypre_BoxIMin(box)[0],
                               hypre_BoxIMin(box)[1]);
               hypre_SetIndex3(upper, hypre_BoxIMax(box)[2],  hypre_BoxIMax(box)[0],
                               hypre_BoxIMax(box)[1]);
               hypre_BoxSetExtents(rotated_box, lower, upper);
               hypre_AppendBox(rotated_box, rotated_boxes);
            }
            hypre_UnionBoxes(rotated_boxes);
            break;

         case 4:
            for (j = 0; j < size; j++)
            {
               box = hypre_BoxArrayBox(boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(box)[2],  hypre_BoxIMin(box)[1],
                               hypre_BoxIMin(box)[0]);
               hypre_SetIndex3(upper, hypre_BoxIMax(box)[2],  hypre_BoxIMax(box)[1],
                               hypre_BoxIMax(box)[0]);
               hypre_BoxSetExtents(rotated_box, lower, upper);
               hypre_AppendBox(rotated_box, rotated_boxes);
            }
            hypre_UnionBoxes(rotated_boxes);
            break;

      } /*switch(i) */
   }    /* for (i= 0; i< 5; i++) */
   hypre_TFree(rotated_box, HYPRE_MEMORY_HOST);

   hypre_UnionBoxes(boxes);

   array = 5;
   min_size = hypre_BoxArraySize(boxes);

   for (i = 0; i < 5; i++)
   {
      rotated_boxes = hypre_BoxArrayArrayBoxArray(rotated_array, i);
      if (hypre_BoxArraySize(rotated_boxes) < min_size)
      {
         min_size = hypre_BoxArraySize(rotated_boxes);
         array = i;
      }
   }

   /* copy the box_array with the minimum number of boxes to boxes */
   if (array != 5)
   {
      rotated_boxes = hypre_BoxArrayArrayBoxArray(rotated_array, array);
      hypre_BoxArraySize(boxes) = min_size;

      switch (array)
      {
         case 0:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(rotated_box)[0],
                               hypre_BoxIMin(rotated_box)[2],
                               hypre_BoxIMin(rotated_box)[1]);
               hypre_SetIndex3(upper, hypre_BoxIMax(rotated_box)[0],
                               hypre_BoxIMax(rotated_box)[2],
                               hypre_BoxIMax(rotated_box)[1]);

               hypre_BoxSetExtents( hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 1:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(rotated_box)[2],
                               hypre_BoxIMin(rotated_box)[0],
                               hypre_BoxIMin(rotated_box)[1]);
               hypre_SetIndex3(upper, hypre_BoxIMax(rotated_box)[2],
                               hypre_BoxIMax(rotated_box)[0],
                               hypre_BoxIMax(rotated_box)[1]);

               hypre_BoxSetExtents( hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 2:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(rotated_box)[1],
                               hypre_BoxIMin(rotated_box)[0],
                               hypre_BoxIMin(rotated_box)[2]);
               hypre_SetIndex3(upper, hypre_BoxIMax(rotated_box)[1],
                               hypre_BoxIMax(rotated_box)[0],
                               hypre_BoxIMax(rotated_box)[2]);

               hypre_BoxSetExtents( hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 3:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(rotated_box)[1],
                               hypre_BoxIMin(rotated_box)[2],
                               hypre_BoxIMin(rotated_box)[0]);
               hypre_SetIndex3(upper, hypre_BoxIMax(rotated_box)[1],
                               hypre_BoxIMax(rotated_box)[2],
                               hypre_BoxIMax(rotated_box)[0]);

               hypre_BoxSetExtents( hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 4:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
               hypre_SetIndex3(lower, hypre_BoxIMin(rotated_box)[2],
                               hypre_BoxIMin(rotated_box)[1],
                               hypre_BoxIMin(rotated_box)[0]);
               hypre_SetIndex3(upper, hypre_BoxIMax(rotated_box)[2],
                               hypre_BoxIMax(rotated_box)[1],
                               hypre_BoxIMax(rotated_box)[0]);

               hypre_BoxSetExtents( hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

      }   /* switch(array) */
   }      /* if (array != 5) */

   hypre_BoxArrayArrayDestroy(rotated_array);

   return hypre_error_flag;
}
