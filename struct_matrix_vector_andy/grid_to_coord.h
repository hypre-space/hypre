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
 * Header info for the hypre_StructGridToCoord structures
 *
 *****************************************************************************/

#ifndef hypre_GRID_TO_COORD_HEADER
#define hypre_GRID_TO_COORD_HEADER


/*--------------------------------------------------------------------------
 * hypre_StructGridToCoordTable:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int   offset;
   int   ni;
   int   nj;

} hypre_StructGridToCoordTableEntry;

typedef struct
{
   hypre_StructGridToCoordTableEntry  **entries;
   int           	       *indices[3];
   int           	        size[3];

   int                          last_index[3];

} hypre_StructGridToCoordTable;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructGridToCoordTable
 *--------------------------------------------------------------------------*/

#define hypre_StructGridToCoordTableEntries(table)        ((table) -> entries)
#define hypre_StructGridToCoordTableIndices(table)        ((table) -> indices)
#define hypre_StructGridToCoordTableSize(table)           ((table) -> size)
#define hypre_StructGridToCoordTableLastIndex(table)      ((table) -> last_index)

#define hypre_StructGridToCoordTableIndexListD(table, d)  \
hypre_StructGridToCoordTableIndices(table)[d]
#define hypre_StructGridToCoordTableIndexD(table, d, i) \
hypre_StructGridToCoordTableIndices(table)[d][i]
#define hypre_StructGridToCoordTableSizeD(table, d) \
hypre_StructGridToCoordTableSize(table)[d]
#define hypre_StructGridToCoordTableLastIndexD(table, d) \
hypre_StructGridToCoordTableLastIndex(table)[d]

#define hypre_StructGridToCoordTableEntry(table, i, j, k) \
hypre_StructGridToCoordTableEntries(table)\
[((k*hypre_StructGridToCoordTableSizeD(table, 1) + j)*\
  hypre_StructGridToCoordTableSizeD(table, 0) + i)]

#define hypre_StructGridToCoordTableEntryOffset(entry)   ((entry) -> offset)
#define hypre_StructGridToCoordTableEntryNI(entry)       ((entry) -> ni)
#define hypre_StructGridToCoordTableEntryNJ(entry)       ((entry) -> nj)

/*--------------------------------------------------------------------------
 * Member macros for hypre_StructGridToCoord translator class
 *--------------------------------------------------------------------------*/

#define hypre_MapStructGridToCoord(index, entry) \
(hypre_StructGridToCoordTableEntryOffset(entry) + \
 ((index[2]*hypre_StructGridToCoordTableEntryNJ(entry) + \
   index[1])*hypre_StructGridToCoordTableEntryNI(entry) + index[0]))


#endif
