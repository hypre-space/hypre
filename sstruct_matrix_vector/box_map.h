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

#ifndef hypre_BOX_MAP_HEADER
#define hypre_BOX_MAP_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructMap:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index  imin;
   hypre_Index  imax;
   void        *info;

} hypre_BoxMapEntry;

typedef struct
{
   int                 max_nentries;
   hypre_Index         global_imin;
   hypre_Index         global_imax;
   int                 nentries;
   hypre_BoxMapEntry  *entries;
   hypre_BoxMapEntry **table; /* this points into 'entries' array */
   int                *indexes[3];
   int                 size[3];
                         
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
#define hypre_BoxMapIndexes(map)        ((map) -> indexes)
#define hypre_BoxMapSize(map)           ((map) -> size)
#define hypre_BoxMapLastIndex(map)      ((map) -> last_index)

#define hypre_BoxMapIndexesD(map, d)    hypre_BoxMapIndexes(map)[d]
#define hypre_BoxMapSizeD(map, d)       hypre_BoxMapSize(map)[d]
#define hypre_BoxMapLastIndexD(map, d)  hypre_BoxMapLastIndex(map)[d]

#define hypre_BoxMapTableEntry(map, i, j, k) \
hypre_BoxMapTable(map)[((k*hypre_BoxMapSizeD(map, 1) + j)*\
                           hypre_BoxMapSizeD(map, 0) + i)]

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_BoxMapEntry
 *--------------------------------------------------------------------------*/

#define hypre_BoxMapEntryIMin(entry)  ((entry) -> imin)
#define hypre_BoxMapEntryIMax(entry)  ((entry) -> imax)
#define hypre_BoxMapEntryInfo(entry)  ((entry) -> info)

#endif
