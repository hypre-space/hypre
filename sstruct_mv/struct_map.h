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
 * Header info for the hypre_StructGridToCoord structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_MAP_HEADER
#define hypre_STRUCT_MAP_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructMap:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int   offset;
   int   stridej;
   int   stridek;
   int   proc;

} hypre_StructMapEntry;

typedef struct
{
   int                     ndim;
   hypre_StructMapEntry   *entries;
   int                    *procs;
   int                    *table;
   int                    *indexes[3];
   int                     size[3];
   int                     start_rank;
                          
   int                     last_index[3];

} hypre_StructMap;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructMap
 *--------------------------------------------------------------------------*/

#define hypre_StructMapNDim(map)           ((map) -> ndim)
#define hypre_StructMapEntries(map)        ((map) -> entries)
#define hypre_StructMapEntry(map, b)      &((map) -> entries[b])
#define hypre_StructMapProcs(map)          ((map) -> procs)
#define hypre_StructMapProc(map)           ((map) -> procs[b])
#define hypre_StructMapTable(map)          ((map) -> table)
#define hypre_StructMapIndexes(map)        ((map) -> indexes)
#define hypre_StructMapSize(map)           ((map) -> size)
#define hypre_StructMapStartRank(map)      ((map) -> start_rank)
#define hypre_StructMapLastIndex(map)      ((map) -> last_index)

#define hypre_StructMapIndexesD(map, d)    hypre_StructMapIndexes(map)[d]
#define hypre_StructMapIndexD(map, d, i)   hypre_StructMapIndexes(map)[d][i]
#define hypre_StructMapSizeD(map, d)       hypre_StructMapSize(map)[d]
#define hypre_StructMapLastIndexD(map, d)  hypre_StructMapLastIndex(map)[d]

#define hypre_StructMapBox(map, i, j, k) \
hypre_StructMapTable(map)[((k*hypre_StructMapSizeD(map, 1) + j)*\
                           hypre_StructMapSizeD(map, 0) + i)]

#define hypre_StructMapEntryOffset(entry)   ((entry) -> offset)
#define hypre_StructMapEntryStrideJ(entry)  ((entry) -> stridej)
#define hypre_StructMapEntryStrideK(entry)  ((entry) -> stridek)

#endif
