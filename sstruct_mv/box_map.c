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
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapEntrySetInfo( hypre_BoxMapEntry  *entry,
                          void               *info )
{
   int ierr = 0;

   hypre_BoxMapEntryInfo(entry) = info;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapEntryGetInfo( hypre_BoxMapEntry  *entry,
                          void              **info_ptr )
{
   int ierr = 0;

   *info_ptr = hypre_BoxMapEntryInfo(entry);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapEntryGetExtents( hypre_BoxMapEntry  *entry,
                             hypre_Index         imin,
                             hypre_Index         imax )
{
   int ierr = 0;
   hypre_IndexRef  entry_imin = hypre_BoxMapEntryIMin(entry);
   hypre_IndexRef  entry_imax = hypre_BoxMapEntryIMax(entry);
   int             d;

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(imin, d) = hypre_IndexD(entry_imin, d);
      hypre_IndexD(imax, d) = hypre_IndexD(entry_imax, d);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapCreate( int            max_nentries,
                    hypre_Index    global_imin,
                    hypre_Index    global_imax,
                    hypre_BoxMap **map_ptr )
{
   int ierr = 0;

   hypre_BoxMap   *map;
   hypre_IndexRef  global_imin_ref;
   hypre_IndexRef  global_imax_ref;
   int             d;
                          
   map = hypre_CTAlloc(hypre_BoxMap, 1);
   hypre_BoxMapMaxNEntries(map) = max_nentries;
   global_imin_ref = hypre_BoxMapGlobalIMin(map);
   global_imax_ref = hypre_BoxMapGlobalIMax(map);
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(global_imin_ref, d) = hypre_IndexD(global_imin, d);
      hypre_IndexD(global_imax_ref, d) = hypre_IndexD(global_imax, d);
      hypre_BoxMapIndexesD(map, d)     = NULL;
   }
   hypre_BoxMapNEntries(map) = 0;
   hypre_BoxMapEntries(map)  = hypre_CTAlloc(hypre_BoxMapEntry, max_nentries);
   hypre_BoxMapTable(map)    = NULL;
      
   *map_ptr = map;
      
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapAddEntry( hypre_BoxMap *map,
                      hypre_Index   imin,
                      hypre_Index   imax,
                      void         *info )
{
   int ierr = 0;

   int                 nentries = hypre_BoxMapNEntries(map);
   hypre_BoxMapEntry  *entries  = hypre_BoxMapEntries(map);
   hypre_BoxMapEntry  *entry;
   hypre_IndexRef      entry_imin;
   hypre_IndexRef      entry_imax;
   int                 d;

   entry = &entries[nentries];
   entry_imin = hypre_BoxMapEntryIMin(entry);
   entry_imax = hypre_BoxMapEntryIMax(entry);
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD(entry_imin, d) = hypre_IndexD(imin, d);
      hypre_IndexD(entry_imax, d) = hypre_IndexD(imax, d);
   }
   hypre_BoxMapEntryInfo(entry) = info;
   hypre_BoxMapNEntries(map) = nentries + 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapAssemble( hypre_BoxMap *map )
{
   int ierr = 0;

   int                         nentries = hypre_BoxMapNEntries(map);
   hypre_BoxMapEntry          *entries  = hypre_BoxMapEntries(map);
   hypre_BoxMapEntry         **table;
   int                        *indexes[3];
   int                         size[3];

   hypre_BoxMapEntry          *entry;
   hypre_IndexRef              entry_imin;
   hypre_IndexRef              entry_imax;

   int                         imin[3];
   int                         imax[3];
   int                         iminmax[2];
   int                         index_not_there;
   int                         b, d, i, j, k;
            
   /*------------------------------------------------------
    * Set up the indexes array
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      indexes[d] = hypre_CTAlloc(int, 2*nentries);
      size[d] = 0;
   }

   for (b = 0; b < nentries; b++)
   {
      entry  = &entries[b];
      entry_imin = hypre_BoxMapEntryIMin(entry);
      entry_imax = hypre_BoxMapEntryIMax(entry);

      for (d = 0; d < 3; d++)
      {
         iminmax[0] = hypre_IndexD(entry_imin, d);
         iminmax[1] = hypre_IndexD(entry_imax, d) + 1;

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
    * Set up the table
    *------------------------------------------------------*/
      
   table = hypre_CTAlloc(hypre_BoxMapEntry *, (size[0] * size[1] * size[2]));
      
   for (b = 0; b < nentries; b++)
   {
      entry = &entries[b];
      entry_imin = hypre_BoxMapEntryIMin(entry);
      entry_imax = hypre_BoxMapEntryIMax(entry);

      /* find the indexes corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
         j = 0;

         while (hypre_IndexD(entry_imin, d) != indexes[d][j])
         {
            j++;
         }
         hypre_IndexD(imin, d) = j;

         while (hypre_IndexD(entry_imax, d) + 1 != indexes[d][j])
         {
            j++;
         }
         hypre_IndexD(imax, d) = j;
      }

      /* set up map table */
      for (k = imin[2]; k < imax[2]; k++)
      {
         for (j = imin[1]; j < imax[1]; j++)
         {
            for (i = imin[0]; i < imax[0]; i++)
            {
               table[((k) * size[1] + j) * size[0] + i] = entry;
            }
         }
      }
   }
      
   /*------------------------------------------------------
    * Set up the map
    *------------------------------------------------------*/

   hypre_TFree(hypre_BoxMapTable(map));
   hypre_BoxMapTable(map) = table;
   for (d = 0; d < 3; d++)
   {
      hypre_TFree(hypre_BoxMapIndexesD(map, d));
      hypre_BoxMapIndexesD(map, d) = indexes[d];
      hypre_BoxMapSizeD(map, d) = size[d];
      hypre_BoxMapLastIndexD(map, d) = 0;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapDestroy( hypre_BoxMap *map )
{
   int ierr = 0;
   int d;

   if (map)
   {
      hypre_TFree(hypre_BoxMapEntries(map));
      hypre_TFree(hypre_BoxMapTable(map));
      
      for (d = 0; d < 3; d++)
      {
         hypre_TFree(hypre_BoxMapIndexesD(map, d));
      }

      hypre_TFree(map);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * This routine return a NULL 'entry_ptr' if an entry is not found
 *--------------------------------------------------------------------------*/

int
hypre_BoxMapFindEntry( hypre_BoxMap       *map,
                       hypre_Index         index,
                       hypre_BoxMapEntry **entry_ptr )
{
   int ierr = 0;

   int  index_d;
   int  map_index[3] = {0, 0, 0};
   int *map_indexes_d;
   int  map_index_d;
   int  map_size_d;
   int  d;
  
   for (d = 0; d < 3; d++)
   {
      map_indexes_d = hypre_BoxMapIndexesD(map, d);
      map_size_d    = hypre_BoxMapSizeD(map, d);

      /* Find location of dimension d of index in map */
      index_d = hypre_IndexD(index, d);

      /* Start looking in place indicated by last_index stored in map */
      map_index_d = hypre_BoxMapLastIndexD(map, d);

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
         *entry_ptr = NULL;
         return ierr;
      }
      else
      {
         map_index[d] = map_index_d;
      }
   }

   /* If code reaches this point, then the entry was succesfully found */
   *entry_ptr = hypre_BoxMapTableEntry(map,
                                       map_index[0],
                                       map_index[1],
                                       map_index[2]);

   /* Reset the last index in the map */
   for (d = 0; d < 3; d++)
   {
      hypre_BoxMapLastIndexD(map, d) = map_index[d];
   }

   return ierr;
}

