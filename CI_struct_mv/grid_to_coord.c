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
 * Member functions for hypre_StructGridToCoord translator class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructGridToCoordTable
 *--------------------------------------------------------------------------*/

hypre_StructGridToCoordTable *
hypre_NewStructGridToCoordTable( hypre_StructGrid    *grid,
			 hypre_StructStencil *stencil )
{
   hypre_StructGridToCoordTable       *table;
			  
   hypre_StructGridToCoordTableEntry **entries;
   int                        *indices[3];
   int                         size[3];

   hypre_StructGridToCoordTableEntry  *entry;
   int                         offset;
   int                         ni;
   int                         nj;
			  
   hypre_BoxArray               *all_boxes;
   hypre_BoxArray               *boxes;
			  
   hypre_Box                    *box;
   hypre_Index                  imin;
   hypre_Index                  imax;
	                  
   int                        *box_neighborhood;
   int                        *box_offsets, box_offset;

   int                         iminmax[2];
   int                         index_not_there;
   int                         b, d, i, j, k;
	    
   table = hypre_CTAlloc(hypre_StructGridToCoordTable, 1);

   /*------------------------------------------------------
    * Put neighborhood boxes into `boxes' hypre_BoxArray
    *------------------------------------------------------*/

   boxes = hypre_StructGridBoxes(grid);
   all_boxes = hypre_StructGridAllBoxes(grid);
   box_neighborhood = hypre_FindBoxApproxNeighborhood(boxes, all_boxes, stencil);

   boxes = hypre_BoxArrayCreate(0);
   for (i = 0; i < hypre_BoxArraySize(all_boxes); i++)
   {
      if (box_neighborhood[i])
	 hypre_AppendBox(hypre_BoxArrayBox(all_boxes, i), boxes);
   }
      
   /*------------------------------------------------------
    * Compute box_offsets for `boxes' hypre_BoxArray
    *------------------------------------------------------*/

   box_offsets = hypre_CTAlloc(int, hypre_BoxArraySize(boxes));
   box_offset = 0;
   j = 0;
   for (i = 0; i < hypre_BoxArraySize(all_boxes); i++)
   {
      if (box_neighborhood[i])
	 box_offsets[j++] = box_offset;

      box_offset += hypre_BoxVolume(hypre_BoxArrayBox(all_boxes, i));
   }
      
   /*------------------------------------------------------
    * Set up the indices array
    *------------------------------------------------------*/
      
   for (d = 0; d < 3; d++)
   {
      indices[d] = hypre_CTAlloc(int, 2 * hypre_BoxArraySize(boxes));
      size[d] = 0;
   }
      
   hypre_ForBoxI(b, boxes)
   {
      box = hypre_BoxArrayBox(boxes, b);

      for (d = 0; d < 3; d++)
      {
	 iminmax[0] = hypre_BoxIMinD(box, d);
	 iminmax[1] = hypre_BoxIMaxD(box, d) + 1;

	 for (i = 0; i < 2; i++)
	 {
	    /* find the new index position in the indices array */
	    index_not_there = 1;
	    for (j = 0; j < size[d]; j++)
	    {
	       if (iminmax[i] <= indices[d][j])
	       {
		  if (iminmax[i] == indices[d][j])
		     index_not_there = 0;
		  break;
	       }
	    }

	    /* if the index is already there, don't add it again */
	    if (index_not_there)
	    {
	       for (k = size[d]; k > j; k--)
		  indices[d][k] = indices[d][k-1];
	       indices[d][j] = iminmax[i];
	       size[d]++;
	    }
	 }
      }
   }

   for (d = 0; d < 3; d++)
      size[d]--;
      
   /*------------------------------------------------------
    * Set up the entries array
    *------------------------------------------------------*/
      
   entries = hypre_CTAlloc( hypre_StructGridToCoordTableEntry *,
		      (size[0] * size[1] * size[2]) );
      
   hypre_ForBoxI(b, boxes)
   {
      box = hypre_BoxArrayBox(boxes, b);

      /* find the indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
	 j = 0;

	 while (hypre_BoxIMinD(box, d) != indices[d][j])
	    j++;
	 hypre_IndexD(imin, d) = j;

	 while (hypre_BoxIMaxD(box, d) + 1 != indices[d][j])
	    j++;
	 hypre_IndexD(imax, d) = j;
      }

      /* set offset, ni, and nj */
      ni = hypre_BoxSizeD(box, 0);
      nj = hypre_BoxSizeD(box, 1);
      offset = box_offsets[b] -
	 ( ( hypre_BoxIMinD(box, 2)*nj + hypre_BoxIMinD(box, 1) )*ni +
	   hypre_BoxIMinD(box, 0) );

      for (k = hypre_IndexD(imin, 2); k < hypre_IndexD(imax, 2); k++)
      {
	 for (j = hypre_IndexD(imin, 1); j < hypre_IndexD(imax, 1); j++)
	 {
	    for (i = hypre_IndexD(imin, 0); i < hypre_IndexD(imax, 0); i++)
	    {
	       entry = hypre_CTAlloc(hypre_StructGridToCoordTableEntry, 1);
	       hypre_StructGridToCoordTableEntryOffset(entry) = offset;
	       hypre_StructGridToCoordTableEntryNI(entry)     = ni;
	       hypre_StructGridToCoordTableEntryNJ(entry)     = nj;

	       entries[((k) * size[1] + j) * size[0] + i] = entry;
	    }
	 }
      }
   }
      

   /*------------------------------------------------------
    * Set up the table
    *------------------------------------------------------*/

   hypre_StructGridToCoordTableEntries(table) = entries;
   for (d = 0; d < 3; d++)
   {
      hypre_StructGridToCoordTableIndexListD(table,d) = indices[d];
      hypre_StructGridToCoordTableSizeD(table, d) = size[d];
      hypre_StructGridToCoordTableLastIndexD(table, d) = 0;
   }

   /* this box array points to grid boxes in all_boxes */
   hypre_BoxArraySize(boxes) = 0;
   hypre_BoxArrayDestroy(boxes);

   hypre_TFree(box_neighborhood);
   hypre_TFree(box_offsets);
      
   return table;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructGridToCoordTable
 *--------------------------------------------------------------------------*/

void 
hypre_FreeStructGridToCoordTable( hypre_StructGridToCoordTable *table )
{
   int i, j, k;
   int d;

   for (k = 0; k < hypre_StructGridToCoordTableSizeD(table, 2); k++)
      for (j = 0; j < hypre_StructGridToCoordTableSizeD(table, 1); j++)
	 for (i = 0; i < hypre_StructGridToCoordTableSizeD(table, 0); i++)
	 {
	    hypre_TFree(hypre_StructGridToCoordTableEntry(table, i, j, k));
	 }
   hypre_TFree(hypre_StructGridToCoordTableEntries(table));

   for (d = 0; d < 3; d++)
      hypre_TFree(hypre_StructGridToCoordTableIndexListD(table, d));

   hypre_TFree(table);
}

/*--------------------------------------------------------------------------
 * hypre_FindStructGridToCoordTableEntry
 *--------------------------------------------------------------------------*/

hypre_StructGridToCoordTableEntry *
hypre_FindStructGridToCoordTableEntry( hypre_Index            index, 
			       hypre_StructGridToCoordTable *table )
{
   hypre_StructGridToCoordTableEntry *entry;

   int table_coords[3];
   int target_index;
   int table_index;
   int table_size;
   int d;
  
   for ( d = 0; d < 3; d++)
   {
      /* Find location of dimension d of index in table */
      target_index = hypre_IndexD( index, d );

      /* Start looking in place indicated by last_index stored in table */
      table_index = hypre_StructGridToCoordTableLastIndexD( table, d );

      /* Loop downward if target index is less than current location */
      while ( (table_index >= 0 ) &&
	      (target_index < hypre_StructGridToCoordTableIndexD( table, d,
							  table_index) ) )
	 table_index --;

      /* Loop upward if target index is greater than current location */
      table_size = hypre_StructGridToCoordTableSizeD( table, d );
      while ( (table_index <= (table_size-1) ) &&
	      (target_index >= hypre_StructGridToCoordTableIndexD( table, d,
							   table_index+1) ) )
	 table_index ++;

      if( ( table_index < 0 ) || ( table_index > (table_size-1) ) )
      {
	 return( NULL );
      }
      else
      {
	 table_coords[d] = table_index;
      }
   }

   /* If code reaches this point, then the table entry was succesfully found */
   entry = hypre_StructGridToCoordTableEntry( table,
				      table_coords[0],
				      table_coords[1],
				      table_coords[2] );

   /* Reset the "last_index" in the table */
   for ( d = 0; d < 3; d++ )
      hypre_StructGridToCoordTableLastIndexD( table , d ) = table_coords[d];

   return( entry );
}

