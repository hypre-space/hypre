/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_StructMap class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructMapCreate
 *--------------------------------------------------------------------------*/

int
hypre_StructMapCreate( hypre_StructGrid   *sgrid,
                       hypre_StructMap   **map_ptr )
{
   int                         ierr = 0;

   MPI_Comm                    comm = hypre_StructGridComm(sgrid);
   int                         ndim = hypre_StructGridDim(sgrid);

   hypre_StructMap            *map;
                          
   hypre_StructMapEntry       *entries;
   int                        *table;
   int                        *indexes[3];
   int                         size[3] = {1, 1, 1};
   int                         start_rank;

   hypre_StructMapEntry       *entry;
   int                         offset;
   int                         stridej;
   int                         stridek;
                          
   hypre_BoxArray             *sgrid_boxes;
   hypre_BoxArray             *boxes;
   int                        *procs;
   int                         first_local;
                          
   hypre_Box                  *box;
   hypre_Index                 imin;
   hypre_Index                 imax;
                          
   int                        *box_offsets, box_offset;

   int                         iminmax[2];
   int                         index_not_there;
   int                         b, d, i, j, k;
            
   /*------------------------------------------------------
    * Compute box_offsets for neighborhood boxes
    *------------------------------------------------------*/

   /* NOTE: With neighborhood info from the user, don't need all gather */
   sgrid_boxes = hypre_StructGridBoxes(sgrid);
   hypre_GatherAllBoxes(comm, sgrid_boxes, &boxes, &procs, &first_local);

   box_offsets = hypre_CTAlloc(int, hypre_BoxArraySize(boxes));
   box_offset  = 0;
   for (b = 0; b < hypre_BoxArraySize(boxes); b++)
   {
      box = hypre_BoxArrayBox(boxes, b);

      box_offsets[b] = box_offset;
      if (b == first_local)
      {
         start_rank = box_offset;
      }
      box_offset += hypre_BoxVolume(box);
   }

   /*------------------------------------------------------
    * Set up the indexes array
    *------------------------------------------------------*/
      
   for (d = 0; d < 3; d++)
   {
      indexes[d] = hypre_CTAlloc(int, 2 * hypre_BoxArraySize(boxes));
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
               /* find the new index position in the indexes array */
               index_not_there = 1;
               for (j = 0; j < size[d]; j++)
               {
                  if (iminmax[i] <= indexes[d][j])
                  {
                     if (iminmax[i] == indexes[d][j])
                     {
                        index_not_there = 0;
                     }
                     break;
                  }
               }

               /* if the index is already there, don't add it again */
               if (index_not_there)
               {
                  for (k = size[d]; k > j; k--)
                  {
                     indexes[d][k] = indexes[d][k-1];
                  }
                  indexes[d][j] = iminmax[i];
                  size[d]++;
               }
            }
         }
      }

   for (d = 0; d < 3; d++)
   {
      size[d]--;
   }
      
   /*------------------------------------------------------
    * Set up the entries array
    *------------------------------------------------------*/
      
   entries = hypre_CTAlloc(hypre_StructMapEntry, hypre_BoxArraySize(boxes));
   table = hypre_CTAlloc(int, (size[0] * size[1] * size[2]));
      
   hypre_ForBoxI(b, boxes)
      {
         box = hypre_BoxArrayBox(boxes, b);

         /* set up map entry */
         stridej = hypre_BoxSizeD(box, 0);
         stridek = hypre_BoxSizeD(box, 1) * stridej;
         offset = box_offsets[b] -
            hypre_BoxIMinD(box, 2) * stridek -
            hypre_BoxIMinD(box, 1) * stridej -
            hypre_BoxIMinD(box, 0);

         entry = &entries[b];
         hypre_StructMapEntryOffset(entry)  = offset;
         hypre_StructMapEntryStrideJ(entry) = stridej;
         hypre_StructMapEntryStrideK(entry) = stridek;

         /* find the indexes corresponding to the current box */
         for (d = 0; d < 3; d++)
         {
            j = 0;

            while (hypre_BoxIMinD(box, d) != indexes[d][j])
            {
               j++;
            }
            hypre_IndexD(imin, d) = j;

            while (hypre_BoxIMaxD(box, d) + 1 != indexes[d][j])
            {
               j++;
            }
            hypre_IndexD(imax, d) = j;
         }

         /* set up map table */
         for (k = hypre_IndexD(imin, 2); k < hypre_IndexD(imax, 2); k++)
         {
            for (j = hypre_IndexD(imin, 1); j < hypre_IndexD(imax, 1); j++)
            {
               for (i = hypre_IndexD(imin, 0); i < hypre_IndexD(imax, 0); i++)
               {
                  table[((k) * size[1] + j) * size[0] + i] = b;
               }
            }
         }
      }
      
   /*------------------------------------------------------
    * Set up the map
    *------------------------------------------------------*/

   map = hypre_CTAlloc(hypre_StructMap, 1);

   hypre_StructMapNDim(map)      = ndim;
   hypre_StructMapEntries(map)   = entries;
   hypre_StructMapProcs(map)     = procs;
   hypre_StructMapTable(map)     = table;
   hypre_StructMapStartRank(map) = start_rank;
   for (d = 0; d < 3; d++)
   {
      hypre_StructMapIndexesD(map,d) = indexes[d];
      hypre_StructMapSizeD(map, d) = size[d];
      hypre_StructMapLastIndexD(map, d) = 0;
   }

   hypre_TFree(box_offsets);

   hypre_BoxArrayDestroy(boxes);
      
   *map_ptr = map;
      
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMapDestroy
 *--------------------------------------------------------------------------*/

int
hypre_StructMapDestroy( hypre_StructMap *map )
{
   int ierr = 0;
   int d;

   if (map)
   {
      hypre_TFree(hypre_StructMapEntries(map));
      hypre_TFree(hypre_StructMapProcs(map));
      hypre_TFree(hypre_StructMapTable(map));
      
      for (d = 0; d < 3; d++)
      {
         hypre_TFree(hypre_StructMapIndexesD(map, d));
      }

      hypre_TFree(map);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMapIndexToBox
 *
 * Maps an index to a unique integer identifier (not necessarily the
 * same as the box ID) for a box in the associated grid.  This
 * identifier can then be used to extract additional information such
 * as the processor where the box lives, the global rank of the index,
 * or the box itself.
 *--------------------------------------------------------------------------*/

int
hypre_StructMapIndexToBox( hypre_StructMap  *map,
                           hypre_Index       index,
                           int              *box_ptr )
{
   int  ierr = 0;

   int  index_d;
   int  map_index[3] = {0, 0, 0};
   int *map_indexes_d;
   int  map_index_d;
   int  map_size_d;
   int  d;
  
   for (d = 0; d < 3; d++)
   {
      map_indexes_d = hypre_StructMapIndexesD(map, d);
      map_size_d    = hypre_StructMapSizeD(map, d);

      /* Find location of dimension d of index in map */
      index_d = hypre_IndexD(index, d);

      /* Start looking in place indicated by last_index stored in map */
      map_index_d = hypre_StructMapLastIndexD(map, d);

      /* Loop downward if target index is less than current location */
      while ( (map_index_d >= 0 ) &&
              (index_d < map_indexes_d[map_index_d]) )
      {
         map_index_d --;
      }

      /* Loop upward if target index is greater than current location */
      while ( (map_index_d <= (map_size_d-1)) &&
              (index_d >= map_indexes_d[map_index_d+1]) )
      {
         map_index_d ++;
      }

      if( ( map_index_d < 0 ) || ( map_index_d > (map_size_d-1) ) )
      {
         *box_ptr = -1;
         return ierr;
      }
      else
      {
         map_index[d] = map_index_d;
      }
   }

   /* If code reaches this point, then the box was succesfully found */
   *box_ptr = hypre_StructMapBox(map,
                                 map_index[0], map_index[1], map_index[2]);

   /* Reset the last index in the map */
   for (d = 0; d < 3; d++)
   {
      hypre_StructMapLastIndexD(map, d) = map_index[d];
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructMapIndexToRank
 *--------------------------------------------------------------------------*/

int
hypre_StructMapIndexToRank( hypre_StructMap       *map,
                            int                    box,
                            hypre_Index            index,
                            int                   *rank_ptr )
{
   int ierr = 0;

   hypre_StructMapEntry  *entry = hypre_StructMapEntry(map, box);

   *rank_ptr = hypre_StructMapEntryOffset(entry) + 
      hypre_IndexD(index, 2) * hypre_StructMapEntryStrideK(entry) + 
      hypre_IndexD(index, 1) * hypre_StructMapEntryStrideJ(entry) +
      hypre_IndexD(index, 0);

   return ierr;
}

