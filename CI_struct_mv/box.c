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
 * Member functions for hypre_Box class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewBox
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_NewBox( hypre_Index *imin,
	    hypre_Index *imax )
{
   hypre_Box *box;

   box = hypre_CTAlloc(hypre_Box, 1);

   hypre_BoxIMin(box) = imin;
   hypre_BoxIMax(box) = imax;

   return box;
}

/*--------------------------------------------------------------------------
 * hypre_NewBoxArray
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_NewBoxArray()
{
   hypre_BoxArray *box_array;

   box_array = hypre_CTAlloc(hypre_BoxArray, 1);

   hypre_BoxArrayBoxes(box_array) = NULL;
   hypre_BoxArraySize(box_array)  = 0;

   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBox
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBox( hypre_Box *box )
{
   if ( box == NULL ) return;

   hypre_FreeIndex(hypre_BoxIMin(box));
   hypre_FreeIndex(hypre_BoxIMax(box));
   hypre_TFree(box);
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxArray
 *--------------------------------------------------------------------------*/

void 
hypre_FreeBoxArray( hypre_BoxArray *box_array )
{
   int  i;

   if( box_array == NULL ) return;

   if( hypre_BoxArrayBoxes(box_array)!=NULL )
   {
      hypre_ForBoxI(i, box_array)
         hypre_FreeBox(hypre_BoxArrayBox(box_array, i));
   }

   hypre_TFree(hypre_BoxArrayBoxes(box_array));

   hypre_TFree(box_array);
}

/*--------------------------------------------------------------------------
 * hypre_DuplicateBox:
 *   Return a duplicate box.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_DuplicateBox( hypre_Box *box )
{
   hypre_Box *new_box;

   hypre_Index *imin;
   hypre_Index *imax;

   int        d;

   imin = hypre_NewIndex();
   imax = hypre_NewIndex();
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(imin, d) = hypre_BoxIMinD(box, d);
      hypre_IndexD(imax, d) = hypre_BoxIMaxD(box, d);
   }

   new_box = hypre_NewBox(imin, imax);

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
   hypre_Box      **new_boxes;
   int            new_size;

   hypre_Box      **boxes;
   int            i, data_sz;

   new_box_array = hypre_NewBoxArray();
   new_boxes = NULL;
   new_size = hypre_BoxArraySize(box_array);

   if (new_size)
   {
      data_sz = ((((new_size - 1) / hypre_BoxArrayBlocksize) + 1) *
		 hypre_BoxArrayBlocksize);
      new_boxes = hypre_CTAlloc(hypre_Box *, data_sz);

      boxes = hypre_BoxArrayBoxes(box_array);

      for (i = 0; i < new_size; i++)
	 new_boxes[i] = hypre_DuplicateBox(boxes[i]);
   }

   hypre_BoxArrayBoxes(new_box_array) = new_boxes;
   hypre_BoxArraySize(new_box_array)  = new_size;

   return new_box_array;
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
   int        size;

   hypre_Box  **old_boxes;
   int        i;

   size = hypre_BoxArraySize(box_array);
   if (!(size % hypre_BoxArrayBlocksize))
   {
      boxes = hypre_CTAlloc(hypre_Box *, size + hypre_BoxArrayBlocksize);
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
	       int           index     )
{
   hypre_Box  **boxes;

   int        i;

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
 * hypre_IntersectBoxes:
 *   Intersect box1 and box2.
 *   If no intersection, return NULL.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_IntersectBoxes( hypre_Box *box1,
		    hypre_Box *box2 )
{
   hypre_Box   *box;

   hypre_Index *imin;
   hypre_Index *imax;

   int        d;

   imin = hypre_NewIndex();
   imax = hypre_NewIndex();

   /* find x, y, and z bounds */
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(imin, d) = max(hypre_BoxIMinD(box1, d), hypre_BoxIMinD(box2, d));
      hypre_IndexD(imax, d) = min(hypre_BoxIMaxD(box1, d), hypre_BoxIMaxD(box2, d));
      if (hypre_IndexD(imax, d) < hypre_IndexD(imin, d))
      {
	 hypre_FreeIndex(imin);
	 hypre_FreeIndex(imax);
	 return NULL;
      }
   }

   /* return intersection */
   box = hypre_NewBox(imin, imax);
   return box;
}

/*--------------------------------------------------------------------------
 * hypre_IntersectBoxArrays:
 *   Intersect box_array1 and box_array2.
 *   If no intersection, return NULL.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_IntersectBoxArrays( hypre_BoxArray *box_array1,
			hypre_BoxArray *box_array2 )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;

   hypre_Box       *box1;
   hypre_Box       *box2;

   int            i1, i2;

   box_array = hypre_NewBoxArray();

   hypre_ForBoxI(i1, box_array1)
   {
      box1 = hypre_BoxArrayBox(box_array1, i1);

      hypre_ForBoxI(i2, box_array2)
      {
	 box2 = hypre_BoxArrayBox(box_array2, i2);

	 box = hypre_IntersectBoxes(box1, box2);
	 if (box)
	    hypre_AppendBox(box, box_array);
      }
   }

   if (hypre_BoxArraySize(box_array) == 0)
   {
      hypre_FreeBoxArray(box_array);
      return NULL;
   }
   else
      return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_SubtractBoxes:
 *   Compute box1 - box2.
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_SubtractBoxes( hypre_Box *box1,
		   hypre_Box *box2 )
{
   hypre_BoxArray  *box_array;
   hypre_Box       *box;

   hypre_Box       *cutbox;

   int            d;

   /*------------------------------------------------------
    * Do a quick check to see if the boxes intersect.
    * If they don't, the subtraction is trivial.
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      if ( (hypre_BoxIMinD(box2, d) > hypre_BoxIMaxD(box1, d)) ||
	   (hypre_BoxIMaxD(box2, d) < hypre_BoxIMinD(box1, d)) )
      {
	 box_array = hypre_NewBoxArray();
	 hypre_AppendBox(box1, box_array);
	 return box_array;
      }
   }

   /*------------------------------------------------------
    * create BoxArray
    *------------------------------------------------------*/

   box_array = hypre_NewBoxArray();
   cutbox = hypre_DuplicateBox(box1);

   /* cut cutbox in x, then y, then z */
   for (d = 0; d < 3; d++)
   {
      if ( (hypre_BoxIMinD(box2, d) >  hypre_BoxIMinD(cutbox, d)) &&
	   (hypre_BoxIMinD(box2, d) <= hypre_BoxIMaxD(cutbox, d)) )
      {
	 box = hypre_DuplicateBox(cutbox);
	 hypre_BoxIMaxD(box, d) = hypre_BoxIMinD(box2, d) - 1;
	 hypre_AppendBox(box, box_array);

	 hypre_BoxIMinD(cutbox, d) = hypre_BoxIMinD(box2, d);
      }
      if ( (hypre_BoxIMaxD(box2, d) >= hypre_BoxIMinD(cutbox, d)) &&
	   (hypre_BoxIMaxD(box2, d) <  hypre_BoxIMaxD(cutbox, d)) )
      {
	 box = hypre_DuplicateBox(cutbox);
	 hypre_BoxIMinD(box, d) = hypre_BoxIMaxD(box2, d) + 1;
	 hypre_AppendBox(box, box_array);

	 hypre_BoxIMaxD(cutbox, d) = hypre_BoxIMaxD(box2, d);
      }
   }

   hypre_FreeBox(cutbox);
   return box_array;
}

/*--------------------------------------------------------------------------
 * hypre_UnionBoxArray:
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
 *--------------------------------------------------------------------------*/

hypre_BoxArray *
hypre_UnionBoxArray( hypre_BoxArray *boxes )
{
   hypre_BoxArray  *box_union;

   hypre_Box       *box;
   hypre_Index     *imin;
   hypre_Index     *imax;

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
	    
   /*------------------------------------------------------
    * Set up the block_index array
    *------------------------------------------------------*/
      
   for (d = 0; d < 3; d++)
   {
      block_index[d] = hypre_CTAlloc(int, 2 * hypre_BoxArraySize(boxes));
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
	    for (j = 0; j < block_sz[d]; j++)
	    {
	       if (iminmax[i] <= block_index[d][j])
		  break;
	    }

	    /* if the index is already there, don't add it again */
	    if (iminmax[i] < block_index[d][j])
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
      
   block = hypre_CTAlloc(int, (block_sz[0] * block_sz[1] * block_sz[2]));
      
   imin = hypre_NewIndex();
   imax = hypre_NewIndex();

   hypre_ForBoxI(bi, boxes)
   {
      box = hypre_BoxArrayBox(boxes, bi);

      /* find the block_index indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
	 j = 0;

	 while (hypre_BoxIMinD(box, d) != block_index[d][j])
	    j++;
	 hypre_IndexD(imin, d) = j;

	 while (hypre_BoxIMaxD(box, d) + 1 != block_index[d][j])
	    j++;
	 hypre_IndexD(imax, d) = j;
      }

      for (k = hypre_IndexD(imin, 2); k < hypre_IndexD(imax, 2); k++)
      {
	 for (j = hypre_IndexD(imin, 1); j < hypre_IndexD(imax, 1); j++)
	 {
	    for (i = hypre_IndexD(imin, 0); i < hypre_IndexD(imax, 0); i++)
	    {
	       index = ((k) * block_sz[1] + j) * block_sz[0] + i;

	       block[index] = factor[2] + factor[1] + factor[0];
	    }
	 }
      }
   }
      
   hypre_FreeIndex(imin);
   hypre_FreeIndex(imax);

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

   box_union = hypre_NewBoxArray();

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

	       imin = hypre_NewIndex();
	       imax = hypre_NewIndex();

	       hypre_IndexD(imin, 0) = block_index[0][i];
	       hypre_IndexD(imin, 1) = block_index[1][j];
	       hypre_IndexD(imin, 2) = block_index[2][k];
	       hypre_IndexD(imax, 0) = block_index[0][i + ioff] - 1;
	       hypre_IndexD(imax, 1) = block_index[1][j + joff] - 1;
	       hypre_IndexD(imax, 2) = block_index[2][k + koff] - 1;

	       box = hypre_NewBox(imin, imax);

	       hypre_AppendBox(box, box_union);
	    }
	       
	    index++;
	 }
      }
   }

   /*------------------------------------------------------
    * Free up block_index and block
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
      hypre_TFree(block_index[d]);

   hypre_TFree(block);
   
   /*---------------------------------------------------------
    * Return box_union
    *---------------------------------------------------------*/

   return box_union;
}

