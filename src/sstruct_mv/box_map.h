/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.2 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#ifndef hypre_BOX_MAP_HEADER
#define hypre_BOX_MAP_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructMap:
 *--------------------------------------------------------------------------*/

typedef struct hypre_BoxMapEntry_struct
{
   hypre_Index  imin;
   hypre_Index  imax;

  /* GEC0902 additional information for ghost calculation in the offset */
   int   num_ghost[6];

   void        *info;

  /* link list of hypre_BoxMapEntries, ones on the process listed first. */
   struct hypre_BoxMapEntry_struct  *next;

} hypre_BoxMapEntry;

typedef struct
{
   int                 max_nentries;
   hypre_Index         global_imin;
   hypre_Index         global_imax;

  /* GEC0902 additional information for ghost calculation in the offset */
   int                num_ghost[6];

   int                 nentries;
   hypre_BoxMapEntry  *entries;
   hypre_BoxMapEntry **table; /* this points into 'entries' array */
   hypre_BoxMapEntry  *boxproc_table; /* (proc, local_box) table pointer */
   int                *indexes[3];
   int                 size[3];
   int                *boxproc_offset;                        

   int                 last_index[3];

} hypre_BoxMap;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxMap
 *--------------------------------------------------------------------------*/

#define hypre_BoxMapMaxNEntries(map)    ((map) -> max_nentries)
#define hypre_BoxMapGlobalIMin(map)     ((map) -> global_imin)
#define hypre_BoxMapGlobalIMax(map)     ((map) -> global_imax)
#define hypre_BoxMapNEntries(map)       ((map) -> nentries)
#define hypre_BoxMapEntries(map)        ((map) -> entries)
#define hypre_BoxMapTable(map)          ((map) -> table)
#define hypre_BoxMapBoxProcTable(map)   ((map) -> boxproc_table)
#define hypre_BoxMapBoxProcOffset(map)  ((map) -> boxproc_offset)
#define hypre_BoxMapIndexes(map)        ((map) -> indexes)
#define hypre_BoxMapSize(map)           ((map) -> size)
#define hypre_BoxMapLastIndex(map)      ((map) -> last_index)
#define hypre_BoxMapNumGhost(map)       ((map) -> num_ghost)

#define hypre_BoxMapIndexesD(map, d)    hypre_BoxMapIndexes(map)[d]
#define hypre_BoxMapSizeD(map, d)       hypre_BoxMapSize(map)[d]
#define hypre_BoxMapLastIndexD(map, d)  hypre_BoxMapLastIndex(map)[d]

#define hypre_BoxMapTableEntry(map, i, j, k) \
hypre_BoxMapTable(map)[((k*hypre_BoxMapSizeD(map, 1) + j)*\
                           hypre_BoxMapSizeD(map, 0) + i)]

#define hypre_BoxMapBoxProcTableEntry(map, box, proc) \
hypre_BoxMapBoxProcTable(map)[ box + \
                               hypre_BoxMapBoxProcOffset(map)[proc] ]

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxMapEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxMapEntryIMin(entry)  ((entry) -> imin)
#define hypre_BoxMapEntryIMax(entry)  ((entry) -> imax)
#define hypre_BoxMapEntryInfo(entry)  ((entry) -> info)
#define hypre_BoxMapEntryNumGhost(entry) ((entry) -> num_ghost)
#define hypre_BoxMapEntryNext(entry)  ((entry) -> next)

#endif
