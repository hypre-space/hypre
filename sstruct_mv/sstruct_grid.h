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
 * Header info for the hypre_SStructGrid structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_GRID_HEADER
#define hypre_SSTRUCT_GRID_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructGrid:
 *
 * NOTE: Since variables may be replicated across different processes,
 * a separate set of "interface grids" is retained so that data can be
 * migrated onto and off of the internal (non-replicated) grids.
 *--------------------------------------------------------------------------*/

typedef enum hypre_SStructVariable_enum hypre_SStructVariable;

typedef struct
{
   HYPRE_SStructVariable  type;
   int                    rank;     /* local rank */
   int                    proc;

} hypre_SStructUVar;

typedef struct
{
   int                    part;
   hypre_Index            cell;
   int                    nuvars;
   hypre_SStructUVar     *uvars;

} hypre_SStructUCVar;

typedef struct
{
   MPI_Comm                comm;          /* TODO: use different comms */
   int                     ndim;
   int                     nvars;         /* number of variables */
   HYPRE_SStructVariable  *vartypes;      /* types of variables */
   hypre_StructGrid       *sgrids[8];     /* struct grids for each vartype */
   hypre_BoxArray         *iboxarrays[8]; /* interface boxes */
                                       
   /* info for mapping (index, var) --> rank */
   hypre_StructMap        *maps[8];       /* map for each vartype */
   int                    *offsets;       /* offset for each var */
   int                     start_rank;

   int                     local_size;    /* Number of variables locally */
   int                     global_size;   /* Total number of variables */
                           
} hypre_SStructPGrid;

typedef struct
{
   hypre_BoxArray *boxes;
   hypre_Index    *ilowers;
   hypre_Index     coord;
   hypre_Index     dir;

} hypre_SStructNeighbor;

typedef struct hypre_SStructGrid_struct
{
   MPI_Comm                 comm;
   int                      ndim;
   int                      nparts;

   /* s-variable info */
   hypre_SStructPGrid     **pgrids;

   /* neighbor info */
   hypre_SStructNeighbor ***neighbors; /* nparts x nparts array */

   /* u-variables info: During construction, array entries are consecutive.
    * After 'Assemble', entries are referenced via local cell rank. */
   int                      nucvars;
   hypre_SStructUCVar     **ucvars;

   /* info for mapping (part, index, var) --> rank */
   int                     *offsets;     /* offset for each part */
   int                      uoffset;     /* offset for u-variables */
   int                      start_rank;

   int                      local_size;  /* Number of variables locally */
   int                      global_size; /* Total number of variables */
                           
   int                      ref_count;

} hypre_SStructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGrid
 *--------------------------------------------------------------------------*/

#define hypre_SStructGridComm(grid)          ((grid) -> comm)
#define hypre_SStructGridNDim(grid)          ((grid) -> ndim)
#define hypre_SStructGridNParts(grid)        ((grid) -> nparts)
#define hypre_SStructGridPGrids(grid)        ((grid) -> pgrids)
#define hypre_SStructGridPGrid(grid, part)   ((grid) -> pgrids[part])
#define hypre_SStructGridNeighbors(grid)     ((grid) -> neighbors)
#define hypre_SStructGridNUCVars(grid)       ((grid) -> nucvars)
#define hypre_SStructGridUCVars(grid)        ((grid) -> ucvars)
#define hypre_SStructGridUCVar(grid, i)      ((grid) -> ucvars[i])
#define hypre_SStructGridOffsets(grid)       ((grid) -> offsets)
#define hypre_SStructGridOffset(grid, part)  ((grid) -> offsets[part])
#define hypre_SStructGridUOffset(grid)       ((grid) -> uoffset)
#define hypre_SStructGridStartRank(grid)     ((grid) -> start_rank)
#define hypre_SStructGridLocalSize(grid)     ((grid) -> local_size)
#define hypre_SStructGridGlobalSize(grid)    ((grid) -> global_size)
#define hypre_SStructGridRefCount(grid)      ((grid) -> ref_count)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPGrid
 *--------------------------------------------------------------------------*/

#define hypre_SStructPGridComm(pgrid)             ((pgrid) -> comm)
#define hypre_SStructPGridNDim(pgrid)             ((pgrid) -> ndim)
#define hypre_SStructPGridNVars(pgrid)            ((pgrid) -> nvars)
#define hypre_SStructPGridVarTypes(pgrid)         ((pgrid) -> vartypes)
#define hypre_SStructPGridVarType(pgrid, var)     ((pgrid) -> vartypes[var])

#define hypre_SStructPGridSGrids(pgrid)           ((pgrid) -> sgrids)
#define hypre_SStructPGridSGrid(pgrid, var) \
((pgrid) -> sgrids[hypre_SStructPGridVarType(pgrid, var)])
#define hypre_SStructPGridCellSGrid(pgrid) \
((pgrid) -> sgrids[HYPRE_SSTRUCT_VARIABLE_CELL])
#define hypre_SStructPGridVTSGrid(pgrid, vartype) ((pgrid) -> sgrids[vartype])

#define hypre_SStructPGridIBoxArrays(pgrid)       ((pgrid) -> iboxarrays)
#define hypre_SStructPGridIBoxArray(pgrid, var) \
((pgrid) -> iboxarrays[hypre_SStructPGridVarType(pgrid, var)])
#define hypre_SStructPGridCellIBoxArray(pgrid) \
((pgrid) -> iboxarrays[HYPRE_SSTRUCT_VARIABLE_CELL])
#define hypre_SStructPGridVTIBoxArray(pgrid, vartype) \
((pgrid) -> iboxarrays[vartype])

#define hypre_SStructPGridMaps(pgrid)             ((pgrid) -> maps)
#define hypre_SStructPGridMap(pgrid, var) \
((pgrid) -> maps[hypre_SStructPGridVarType(pgrid, var)])
#define hypre_SStructPGridCellMap(pgrid) \
((pgrid) -> maps[HYPRE_SSTRUCT_VARIABLE_CELL])
#define hypre_SStructPGridVTMap(pgrid, vartype)   ((pgrid) -> maps[vartype])

#define hypre_SStructPGridOffsets(pgrid)          ((pgrid) -> offsets)
#define hypre_SStructPGridOffset(pgrid, var)      ((pgrid) -> offsets[var])
#define hypre_SStructPGridStartRank(pgrid)        ((pgrid) -> start_rank)
#define hypre_SStructPGridLocalSize(pgrid)        ((pgrid) -> local_size)
#define hypre_SStructPGridGlobalSize(pgrid)       ((pgrid) -> global_size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructNeighbor
 *--------------------------------------------------------------------------*/

#define hypre_SStructNeighborBoxes(neighbor)     ((neighbor) -> boxes)
#define hypre_SStructNeighborBox(neighbor, i)    ((neighbor) -> boxes[i])
#define hypre_SStructNeighborILowers(neighbor)   ((neighbor) -> ilowers)
#define hypre_SStructNeighborILower(neighbor, i) ((neighbor) -> ilowers[i])
#define hypre_SStructNeighborCoord(neighbor)     ((neighbor) -> coord)
#define hypre_SStructNeighborDir(neighbor)       ((neighbor) -> dir)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUCVar
 *--------------------------------------------------------------------------*/

#define hypre_SStructUCVarPart(uc)     ((uc) -> part)
#define hypre_SStructUCVarCell(uc)     ((uc) -> cell)
#define hypre_SStructUCVarNUVars(uc)   ((uc) -> nuvars)
#define hypre_SStructUCVarUVars(uc)    ((uc) -> uvars)
#define hypre_SStructUCVarType(uc, i)  ((uc) -> uvars[i].type)
#define hypre_SStructUCVarRank(uc, i)  ((uc) -> uvars[i].rank)
#define hypre_SStructUCVarProc(uc, i)  ((uc) -> uvars[i].proc)

#endif

