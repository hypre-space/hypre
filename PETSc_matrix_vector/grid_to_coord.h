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
 * Header info for the zzz_StructGridToCoord structures
 *
 *****************************************************************************/

#ifndef zzz_GRID_TO_COORD_HEADER
#define zzz_GRID_TO_COORD_HEADER


/*--------------------------------------------------------------------------
 * zzz_StructGridToCoordTable:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int   offset;
   int   ni;
   int   nj;

} zzz_StructGridToCoordTableEntry;

typedef struct
{
   zzz_StructGridToCoordTableEntry  **entries;
   int           	       *indices[3];
   int           	        size[3];

   int                          last_index[3];

} zzz_StructGridToCoordTable;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_StructGridToCoordTable
 *--------------------------------------------------------------------------*/

#define zzz_StructGridToCoordTableEntries(table)        ((table) -> entries)
#define zzz_StructGridToCoordTableIndices(table)        ((table) -> indices)
#define zzz_StructGridToCoordTableSize(table)           ((table) -> size)
#define zzz_StructGridToCoordTableLastIndex(table)      ((table) -> last_index)

#define zzz_StructGridToCoordTableIndexListD(table, d)  \
zzz_StructGridToCoordTableIndices(table)[d]
#define zzz_StructGridToCoordTableIndexD(table, d, i) \
zzz_StructGridToCoordTableIndices(table)[d][i]
#define zzz_StructGridToCoordTableSizeD(table, d) \
zzz_StructGridToCoordTableSize(table)[d]
#define zzz_StructGridToCoordTableLastIndexD(table, d) \
zzz_StructGridToCoordTableLastIndex(table)[d]

#define zzz_StructGridToCoordTableEntry(table, i, j, k) \
zzz_StructGridToCoordTableEntries(table)\
[((k*zzz_StructGridToCoordTableSizeD(table, 1) + j)*\
  zzz_StructGridToCoordTableSizeD(table, 0) + i)]

#define zzz_StructGridToCoordTableEntryOffset(entry)   ((entry) -> offset)
#define zzz_StructGridToCoordTableEntryNI(entry)       ((entry) -> ni)
#define zzz_StructGridToCoordTableEntryNJ(entry)       ((entry) -> nj)

/*--------------------------------------------------------------------------
 * Member macros for zzz_StructGridToCoord translator class
 *--------------------------------------------------------------------------*/

#define zzz_MapStructGridToCoord(index, entry) \
(zzz_StructGridToCoordTableEntryOffset(entry) + \
 ((index[2]*zzz_StructGridToCoordTableEntryNJ(entry) + \
   index[1])*zzz_StructGridToCoordTableEntryNI(entry) + index[0]))


#endif
