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
 *   Box algebra functions.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_IntersectBoxes:
 *   Intersect box1 and box2.
 *   If no intersection, return NULL.
 *--------------------------------------------------------------------------*/

zzz_Box *
zzz_IntersectBoxes( zzz_Box *box1,
		    zzz_Box *box2 )
{
   zzz_Box   *box;

   zzz_Index *imin;
   zzz_Index *imax;

   int        d;

   imin = zzz_NewIndex();
   imax = zzz_NewIndex();

   /* find x, y, and z bounds */
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD(imin, d) = max(zzz_BoxIMinD(box1, d), zzz_BoxIMinD(box2, d));
      zzz_IndexD(imax, d) = min(zzz_BoxIMaxD(box1, d), zzz_BoxIMaxD(box2, d));
      if (zzz_IndexD(imax, d) < zzz_IndexD(imin, d))
      {
	 zzz_FreeIndex(imin);
	 zzz_FreeIndex(imax);
	 return NULL;
      }
   }

   /* return intersection */
   box = zzz_NewBox(imin, imax);
   return box;
}

/*--------------------------------------------------------------------------
 * zzz_IntersectBoxArrays:
 *   Intersect box_array1 and box_array2.
 *   If no intersection, return NULL.
 *--------------------------------------------------------------------------*/

zzz_BoxArray *
zzz_IntersectBoxArrays( zzz_BoxArray *box_array1,
			zzz_BoxArray *box_array2 )
{
   zzz_BoxArray  *box_array;
   zzz_Box       *box;

   zzz_Box       *box1;
   zzz_Box       *box2;

   int            i1, i2;

   box_array = zzz_NewBoxArray();

   zzz_ForBoxI(i1, box_array1)
   {
      box1 = zzz_BoxArrayBox(box_array1, i1);

      zzz_ForBoxI(i2, box_array2)
      {
	 box2 = zzz_BoxArrayBox(box_array2, i2);

	 box = zzz_IntersectBoxes(box1, box2);
	 if (box)
	    zzz_AppendBox(box, box_array);
      }
   }

   if (zzz_BoxArraySize(box_array) == 0)
   {
      zzz_FreeBoxArray(box_array);
      return NULL;
   }
   else
      return box_array;
}

/*--------------------------------------------------------------------------
 * zzz_SubtractBoxes:
 *   Compute box1 - box2.
 *--------------------------------------------------------------------------*/

zzz_BoxArray *
zzz_SubtractBoxes( zzz_Box *box1,
		   zzz_Box *box2 )
{
   zzz_BoxArray  *box_array;
   zzz_Box       *box;

   zzz_Box       *cutbox;

   int            d;

   /*------------------------------------------------------
    * Do a quick check to see if the boxes intersect.
    * If they don't, the subtraction is trivial.
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      if ( (zzz_BoxIMinD(box2, d) > zzz_BoxIMaxD(box1, d)) ||
	   (zzz_BoxIMaxD(box2, d) < zzz_BoxIMinD(box1, d)) )
      {
	 box_array = zzz_NewBoxArray();
	 zzz_AppendBox(zzz_DuplicateBox(box1), box_array);
	 return box_array;
      }
   }

   /*------------------------------------------------------
    * create BoxArray
    *------------------------------------------------------*/

   box_array = zzz_NewBoxArray();
   cutbox = zzz_DuplicateBox(box1);

   /* cut cutbox in x, then y, then z */
   for (d = 0; d < 3; d++)
   {
      if ( (zzz_BoxIMinD(box2, d) >  zzz_BoxIMinD(cutbox, d)) &&
	   (zzz_BoxIMinD(box2, d) <= zzz_BoxIMaxD(cutbox, d)) )
      {
	 box = zzz_DuplicateBox(cutbox);
	 zzz_BoxIMaxD(box, d) = zzz_BoxIMinD(box2, d) - 1;
	 zzz_AppendBox(box, box_array);

	 zzz_BoxIMinD(cutbox, d) = zzz_BoxIMinD(box2, d);
      }
      if ( (zzz_BoxIMaxD(box2, d) >= zzz_BoxIMinD(cutbox, d)) &&
	   (zzz_BoxIMaxD(box2, d) <  zzz_BoxIMaxD(cutbox, d)) )
      {
	 box = zzz_DuplicateBox(cutbox);
	 zzz_BoxIMinD(box, d) = zzz_BoxIMaxD(box2, d) + 1;
	 zzz_AppendBox(box, box_array);

	 zzz_BoxIMaxD(cutbox, d) = zzz_BoxIMaxD(box2, d);
      }
   }

   zzz_FreeBox(cutbox);
   return box_array;
}

/*--------------------------------------------------------------------------
 * zzz_UnionBoxArray:
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

zzz_BoxArray *
zzz_UnionBoxArray( zzz_BoxArray *boxes )
{
   zzz_BoxArray  *box_union;

   zzz_Box       *box;
   zzz_Index     *imin;
   zzz_Index     *imax;

   int     	 *block_index[3];
   int     	  block_sz[3];
   int     	 *block;
   int     	  index;
   int     	  factor[3];
           	
   int     	  iminmax[2];
   int     	  ii[3], dd[3];
   int     	  join;
   int     	  i_tmp0, i_tmp1;
   int            ioff, joff, koff;
   int     	  bi, d, i, j, k;

   int            index_not_there;
	    
   /*------------------------------------------------------
    * If the size of boxes is 0, return an empty union
    *------------------------------------------------------*/

   if (zzz_BoxArraySize(boxes) == 0)
   {
      box_union = zzz_NewBoxArray();
      return box_union;
   }
      
   /*------------------------------------------------------
    * Set up the block_index array
    *------------------------------------------------------*/
      
   for (d = 0; d < 3; d++)
   {
      block_index[d] = talloc(int, 2 * zzz_BoxArraySize(boxes));
      block_sz[d] = 0;
   }
      
   zzz_ForBoxI(bi, boxes)
   {
      box = zzz_BoxArrayBox(boxes, bi);

      for (d = 0; d < 3; d++)
      {
	 iminmax[0] = zzz_BoxIMinD(box, d);
	 iminmax[1] = zzz_BoxIMaxD(box, d) + 1;

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
      
   /*------------------------------------------------------
    * Set factor values
    *------------------------------------------------------*/
      
   factor[0] = 1;
   factor[1] = (block_sz[0] + 1);
   factor[2] = (block_sz[1] + 1) * factor[1];
      
   /*------------------------------------------------------
    * Set up the block array
    *------------------------------------------------------*/
      
   block = ctalloc(int, (block_sz[0] * block_sz[1] * block_sz[2]));
      
   imin = zzz_NewIndex();
   imax = zzz_NewIndex();

   zzz_ForBoxI(bi, boxes)
   {
      box = zzz_BoxArrayBox(boxes, bi);

      /* find the block_index indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
	 j = 0;

	 while (zzz_BoxIMinD(box, d) != block_index[d][j])
	    j++;
	 zzz_IndexD(imin, d) = j;

	 while (zzz_BoxIMaxD(box, d) + 1 != block_index[d][j])
	    j++;
	 zzz_IndexD(imax, d) = j;
      }

      /* note: boxes of size zero will not be added to block */
      for (k = zzz_IndexD(imin, 2); k < zzz_IndexD(imax, 2); k++)
      {
	 for (j = zzz_IndexD(imin, 1); j < zzz_IndexD(imax, 1); j++)
	 {
	    for (i = zzz_IndexD(imin, 0); i < zzz_IndexD(imax, 0); i++)
	    {
	       index = ((k) * block_sz[1] + j) * block_sz[0] + i;

	       block[index] = factor[2] + factor[1] + factor[0];
	    }
	 }
      }
   }
      
   zzz_FreeIndex(imin);
   zzz_FreeIndex(imax);

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
    * Set up the box_union BoxArray
    *------------------------------------------------------*/

   box_union = zzz_NewBoxArray();

   index = 0;
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

	       imin = zzz_NewIndex();
	       imax = zzz_NewIndex();

	       zzz_IndexD(imin, 0) = block_index[0][i];
	       zzz_IndexD(imin, 1) = block_index[1][j];
	       zzz_IndexD(imin, 2) = block_index[2][k];
	       zzz_IndexD(imax, 0) = block_index[0][i + ioff] - 1;
	       zzz_IndexD(imax, 1) = block_index[1][j + joff] - 1;
	       zzz_IndexD(imax, 2) = block_index[2][k + koff] - 1;

	       box = zzz_NewBox(imin, imax);

	       zzz_AppendBox(box, box_union);
	    }
	       
	    index++;
	 }
      }
   }

   /*------------------------------------------------------
    * Free up block_index and block
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
      tfree(block_index[d]);

   tfree(block);
   
   /*---------------------------------------------------------
    * Return box_union
    *---------------------------------------------------------*/

   return box_union;
}

