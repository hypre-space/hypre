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
 * Member functions for zzz_StructGridToCoord translator class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructGridToCoordTable
 *--------------------------------------------------------------------------*/

zzz_StructGridToCoordTable *
zzz_NewStructGridToCoordTable( zzz_StructGrid    *grid,
			 zzz_StructStencil *stencil )
{
   zzz_StructGridToCoordTable       *table;
			  
   zzz_StructGridToCoordTableEntry **entries;
   int                        *indices[3];
   int                         size[3];

   zzz_StructGridToCoordTableEntry  *entry;
   int                         offset;
   int                         ni;
   int                         nj;
			  
   zzz_BoxArray               *all_boxes;
   zzz_BoxArray               *boxes;
			  
   zzz_Box                    *box;
   zzz_Index                  *imin;
   zzz_Index                  *imax;
	                  
   int                        *box_neighborhood;
   int                        *box_offsets, box_offset;

   int                         iminmax[2];
   int                         index_not_there;
   int                         b, d, i, j, k;
	    
   table = talloc(zzz_StructGridToCoordTable, 1);

   /*------------------------------------------------------
    * Put neighborhood boxes into `boxes' zzz_BoxArray
    *------------------------------------------------------*/

   boxes = zzz_StructGridBoxes(grid);
   all_boxes = zzz_StructGridAllBoxes(grid);
   box_neighborhood = zzz_FindBoxApproxNeighborhood(boxes, all_boxes, stencil);

   boxes = zzz_NewBoxArray();
   for (i = 0; i < zzz_BoxArraySize(all_boxes); i++)
   {
      if (box_neighborhood[i])
	 zzz_AppendBox(zzz_BoxArrayBox(all_boxes, i), boxes);
   }
      
   /*------------------------------------------------------
    * Compute box_offsets for `boxes' zzz_BoxArray
    *------------------------------------------------------*/

   box_offsets = ctalloc(int, zzz_BoxArraySize(boxes));
   box_offset = 0;
   j = 0;
   for (i = 0; i < zzz_BoxArraySize(all_boxes); i++)
   {
      if (box_neighborhood[i])
	 box_offsets[j++] = box_offset;

      box_offset += zzz_BoxTotalSize(zzz_BoxArrayBox(all_boxes, i));
   }
      
   /*------------------------------------------------------
    * Set up the indices array
    *------------------------------------------------------*/
      
   for (d = 0; d < 3; d++)
   {
      indices[d] = talloc(int, 2 * zzz_BoxArraySize(boxes));
      size[d] = 0;
   }
      
   zzz_ForBoxI(b, boxes)
   {
      box = zzz_BoxArrayBox(boxes, b);

      for (d = 0; d < 3; d++)
      {
	 iminmax[0] = zzz_BoxIMinD(box, d);
	 iminmax[1] = zzz_BoxIMaxD(box, d) + 1;

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
      
   entries = ctalloc( zzz_StructGridToCoordTableEntry *,
		      (size[0] * size[1] * size[2]) );
      
   imin = zzz_NewIndex();
   imax = zzz_NewIndex();

   zzz_ForBoxI(b, boxes)
   {
      box = zzz_BoxArrayBox(boxes, b);

      /* find the indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
	 j = 0;

	 while (zzz_BoxIMinD(box, d) != indices[d][j])
	    j++;
	 zzz_IndexD(imin, d) = j;

	 while (zzz_BoxIMaxD(box, d) + 1 != indices[d][j])
	    j++;
	 zzz_IndexD(imax, d) = j;
      }

      /* set offset, ni, and nj */
      ni = zzz_BoxSizeD(box, 0);
      nj = zzz_BoxSizeD(box, 1);
      offset = box_offsets[b] -
	 ( ( zzz_BoxIMinD(box, 2)*nj + zzz_BoxIMinD(box, 1) )*ni +
	   zzz_BoxIMinD(box, 0) );

      for (k = zzz_IndexD(imin, 2); k < zzz_IndexD(imax, 2); k++)
      {
	 for (j = zzz_IndexD(imin, 1); j < zzz_IndexD(imax, 1); j++)
	 {
	    for (i = zzz_IndexD(imin, 0); i < zzz_IndexD(imax, 0); i++)
	    {
	       entry = ctalloc(zzz_StructGridToCoordTableEntry, 1);
	       zzz_StructGridToCoordTableEntryOffset(entry) = offset;
	       zzz_StructGridToCoordTableEntryNI(entry)     = ni;
	       zzz_StructGridToCoordTableEntryNJ(entry)     = nj;

	       entries[((k) * size[1] + j) * size[0] + i] = entry;
	    }
	 }
      }
   }
      
   zzz_FreeIndex(imin);
   zzz_FreeIndex(imax);

   /*------------------------------------------------------
    * Set up the table
    *------------------------------------------------------*/

   zzz_StructGridToCoordTableEntries(table) = entries;
   for (d = 0; d < 3; d++)
   {
      zzz_StructGridToCoordTableIndexListD(table,d) = indices[d];
      zzz_StructGridToCoordTableSizeD(table, d) = size[d];
      zzz_StructGridToCoordTableLastIndexD(table, d) = 0;
   }

   /* this box array points to grid boxes in all_boxes */
   zzz_BoxArraySize(boxes) = 0;
   zzz_FreeBoxArray(boxes);

   tfree(box_neighborhood);
   tfree(box_offsets);
      
   return table;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructGridToCoordTable
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructGridToCoordTable( zzz_StructGridToCoordTable *table )
{
   int i, j, k;
   int d;

   for (k = 0; k < zzz_StructGridToCoordTableSizeD(table, 2); k++)
      for (j = 0; j < zzz_StructGridToCoordTableSizeD(table, 1); j++)
	 for (i = 0; i < zzz_StructGridToCoordTableSizeD(table, 0); i++)
	 {
	    tfree(zzz_StructGridToCoordTableEntry(table, i, j, k));
	 }
   tfree(zzz_StructGridToCoordTableEntries(table));

   for (d = 0; d < 3; d++)
      tfree(zzz_StructGridToCoordTableIndexListD(table, d));

   tfree(table);
}

/*--------------------------------------------------------------------------
 * zzz_FindStructGridToCoordTableEntry
 *--------------------------------------------------------------------------*/

zzz_StructGridToCoordTableEntry *
zzz_FindStructGridToCoordTableEntry( zzz_Index            *index, 
			       zzz_StructGridToCoordTable *table )
{
   zzz_StructGridToCoordTableEntry *entry;

   int table_coords[3];
   int target_index;
   int table_index;
   int table_size;
   int d;
  
   for ( d = 0; d < 3; d++)
   {
      /* Find location of dimension d of index in table */
      target_index = zzz_IndexD( index, d );

      /* Start looking in place indicated by last_index stored in table */
      table_index = zzz_StructGridToCoordTableLastIndexD( table, d );

      /* Loop downward if target index is less than current location */
      while ( (table_index >= 0 ) &&
	      (target_index < zzz_StructGridToCoordTableIndexD( table, d,
							  table_index) ) )
	 table_index --;

      /* Loop upward if target index is greater than current location */
      table_size = zzz_StructGridToCoordTableSizeD( table, d );
      while ( (table_index <= (table_size-1) ) &&
	      (target_index >= zzz_StructGridToCoordTableIndexD( table, d,
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
   entry = zzz_StructGridToCoordTableEntry( table,
				      table_coords[0],
				      table_coords[1],
				      table_coords[2] );

   /* Reset the "last_index" in the table */
   for ( d = 0; d < 3; d++ )
      zzz_StructGridToCoordTableLastIndexD( table , d ) = table_coords[d];

   return( entry );
}

