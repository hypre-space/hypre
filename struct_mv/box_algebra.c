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
 *   Box algebra functions.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_IntersectBoxes:
 *   Intersect box1 and box2.
 *   If the boxes do not intersect, the result is a box with zero volume.
 *--------------------------------------------------------------------------*/

int
hypre_IntersectBoxes( hypre_Box *box1,
                      hypre_Box *box2,
                      hypre_Box *ibox )
{
   int          ierr = 0;
   int          d;

   /* find x, y, and z bounds */
   for (d = 0; d < 3; d++)
   {
      hypre_BoxIMinD(ibox, d) =
         hypre_max(hypre_BoxIMinD(box1, d), hypre_BoxIMinD(box2, d));
      hypre_BoxIMaxD(ibox, d) =
         hypre_min(hypre_BoxIMaxD(box1, d), hypre_BoxIMaxD(box2, d));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SubtractBoxes:
 *   Compute box1 - box2.
 *--------------------------------------------------------------------------*/

int
hypre_SubtractBoxes( hypre_Box      *box1,
                     hypre_Box      *box2,
                     hypre_BoxArray *box_array )
{
   int         ierr = 0;
              
   hypre_Box  *box;
   hypre_Box  *rembox;
   int         d, size;

   /*------------------------------------------------------
    * Set the box array size to the maximum possible,
    * plus one, to have space for the remainder box.
    *------------------------------------------------------*/

   hypre_BoxArraySetSize(box_array, 7);

   /*------------------------------------------------------
    * Subtract the boxes by cutting box1 in x, y, then z
    *------------------------------------------------------*/

   rembox = hypre_BoxArrayBox(box_array, 6);
   hypre_CopyBox(box1, rembox);

   size = 0;
   for (d = 0; d < 3; d++)
   {
      /* if the boxes do not intersect, the subtraction is trivial */
      if ( (hypre_BoxIMinD(box2, d) > hypre_BoxIMaxD(rembox, d)) ||
           (hypre_BoxIMaxD(box2, d) < hypre_BoxIMinD(rembox, d)) )
      {
         hypre_CopyBox(box1, hypre_BoxArrayBox(box_array, 0));
         size = 1;
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
            size++;
         }
         if ( hypre_BoxIMaxD(box2, d) < hypre_BoxIMaxD(rembox, d) )
         {
            box = hypre_BoxArrayBox(box_array, size);
            hypre_CopyBox(rembox, box);
            hypre_BoxIMinD(box, d) = hypre_BoxIMaxD(box2, d) + 1;
            hypre_BoxIMaxD(rembox, d) = hypre_BoxIMaxD(box2, d);
            size++;
         }
      }
   }
   hypre_BoxArraySetSize(box_array, size);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_UnionBoxes:
 *   Compute the union of all boxes.
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

int
hypre_UnionBoxes( hypre_BoxArray *boxes )
{
   int              ierr = 0;

   hypre_Box       *box;

   int             *block_index[3];
   int              block_sz[3], block_volume;
   int             *block;
   int              index;
   int              size;
   int              factor[3];
                  
   int              iminmax[2], imin[3], imax[3];
   int              ii[3], dd[3];
   int              join;
   int              i_tmp0, i_tmp1;
   int              ioff, joff, koff;
   int              bi, d, i, j, k;
                  
   int              index_not_there;
            
   /*------------------------------------------------------
    * If the size of boxes is less than 2, return
    *------------------------------------------------------*/

   if (hypre_BoxArraySize(boxes) < 2)
   {
      return ierr;
   }
      
   /*------------------------------------------------------
    * Set up the block_index array
    *------------------------------------------------------*/
      
   i_tmp0 = 2 * hypre_BoxArraySize(boxes);
   block_index[0] = hypre_TAlloc(int, 3 * i_tmp0);
   block_sz[0] = 0;
   for (d = 1; d < 3; d++)
   {
      block_index[d] = block_index[d-1] + i_tmp0;
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
                        index_not_there = 0;
                     break;
                  }
               }

               /* if the index is already there, don't add it again */
               if (index_not_there)
               {
                  for (k = block_sz[d]; k > j; k--)
                     block_index[d][k] = block_index[d][k-1];
                  block_index[d][j] = iminmax[i];
                  block_sz[d]++;
               }
            }
         }
      }

   for (d = 0; d < 3; d++)
      block_sz[d]--;
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
      
   block = hypre_CTAlloc(int, block_volume);
      
   hypre_ForBoxI(bi, boxes)
      {
         box = hypre_BoxArrayBox(boxes, bi);

         /* find the block_index indices corresponding to the current box */
         for (d = 0; d < 3; d++)
         {
            j = 0;

            while (hypre_BoxIMinD(box, d) != block_index[d][j])
               j++;
            imin[d] = j;

            while (hypre_BoxIMaxD(box, d) + 1 != block_index[d][j])
               j++;
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
      switch(d)
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
                     join = 0;
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
         size++;
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

   hypre_TFree(block_index[0]);
   hypre_TFree(block);
   
   return ierr;
}

