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
                                       
   hypre_BoxArray         *pneighbors;

   int                     local_size;    /* Number of variables locally */
   int                     global_size;   /* Total number of variables */

   hypre_Index             periodic;      /* Indicates if pgrid is periodic */
                           
} hypre_SStructPGrid;

typedef struct
{
   hypre_Box    box;
   int          part;
   hypre_Index  ilower;
   hypre_Index  coord;
   hypre_Index  dir;

} hypre_SStructNeighbor;

enum hypre_SStructMapInfoType
{
   hypre_SSTRUCT_MAP_INFO_DEFAULT  = 0,
   hypre_SSTRUCT_MAP_INFO_NEIGHBOR = 1
};

typedef struct
{
   int  type;
   int  proc;
   int  offset;

} hypre_SStructMapInfo;

typedef struct
{
   int          type;
   int          proc;
   int          offset;
   int          part;
   hypre_Index  ilower;
   hypre_Index  coord;
   hypre_Index  dir;

} hypre_SStructNMapInfo;

typedef struct hypre_SStructGrid_struct
{
   MPI_Comm                   comm;
   int                        ndim;
   int                        nparts;
                          
   /* s-variable info */  
   hypre_SStructPGrid       **pgrids;
                          
   /* neighbor info */    
   int                       *nneighbors;
   hypre_SStructNeighbor    **neighbors;
   int                      **nvneighbors;
   hypre_SStructNeighbor   ***vneighbors;

   /* u-variables info: During construction, array entries are consecutive.
    * After 'Assemble', entries are referenced via local cell rank. */
   int                        nucvars;
   hypre_SStructUCVar       **ucvars;

   /* info for mapping (part, index, var) --> rank */
   hypre_BoxMap            ***maps;        /* map for each part, var */
   hypre_SStructMapInfo    ***info;
   hypre_SStructNMapInfo   ***ninfo;
   int                        start_rank;

   int                        local_size;  /* Number of variables locally */
   int                        global_size; /* Total number of variables */
                              
   int                        ref_count;

} hypre_SStructGrid;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGrid
 *--------------------------------------------------------------------------*/

#define hypre_SStructGridComm(grid)           ((grid) -> comm)
#define hypre_SStructGridNDim(grid)           ((grid) -> ndim)
#define hypre_SStructGridNParts(grid)         ((grid) -> nparts)
#define hypre_SStructGridPGrids(grid)         ((grid) -> pgrids)
#define hypre_SStructGridPGrid(grid, part)    ((grid) -> pgrids[part])
#define hypre_SStructGridNNeighbors(grid)     ((grid) -> nneighbors)
#define hypre_SStructGridNeighbors(grid)      ((grid) -> neighbors)
#define hypre_SStructGridNVNeighbors(grid)    ((grid) -> nvneighbors)
#define hypre_SStructGridVNeighbors(grid)     ((grid) -> vneighbors)
#define hypre_SStructGridNUCVars(grid)        ((grid) -> nucvars)
#define hypre_SStructGridUCVars(grid)         ((grid) -> ucvars)
#define hypre_SStructGridUCVar(grid, i)       ((grid) -> ucvars[i])
#define hypre_SStructGridMaps(grid)           ((grid) -> maps)
#define hypre_SStructGridMap(grid, part, var) ((grid) -> maps[part][var])
#define hypre_SStructGridInfo(grid)           ((grid) -> info)
#define hypre_SStructGridNInfo(grid)          ((grid) -> ninfo)
#define hypre_SStructGridStartRank(grid)      ((grid) -> start_rank)
#define hypre_SStructGridLocalSize(grid)      ((grid) -> local_size)
#define hypre_SStructGridGlobalSize(grid)     ((grid) -> global_size)
#define hypre_SStructGridRefCount(grid)       ((grid) -> ref_count)

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

#define hypre_SStructPGridPNeighbors(pgrid)       ((pgrid) -> pneighbors)
#define hypre_SStructPGridLocalSize(pgrid)        ((pgrid) -> local_size)
#define hypre_SStructPGridGlobalSize(pgrid)       ((pgrid) -> global_size)
#define hypre_SStructPGridPeriodic(pgrid)         ((pgrid) -> periodic)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMapInfo
 *--------------------------------------------------------------------------*/

#define hypre_SStructMapInfoType(info)   ((info) -> type)
#define hypre_SStructMapInfoProc(info)   ((info) -> proc)
#define hypre_SStructMapInfoOffset(info) ((info) -> offset)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMapInfo
 *--------------------------------------------------------------------------*/

#define hypre_SStructNMapInfoType(info)   ((info) -> type)
#define hypre_SStructNMapInfoProc(info)   ((info) -> proc)
#define hypre_SStructNMapInfoOffset(info) ((info) -> offset)
#define hypre_SStructNMapInfoPart(info)   ((info) -> part)
#define hypre_SStructNMapInfoILower(info) ((info) -> ilower)
#define hypre_SStructNMapInfoCoord(info)  ((info) -> coord)
#define hypre_SStructNMapInfoDir(info)    ((info) -> dir)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructNeighbor
 *--------------------------------------------------------------------------*/

#define hypre_SStructNeighborBox(neighbor)    &((neighbor) -> box)
#define hypre_SStructNeighborPart(neighbor)    ((neighbor) -> part)
#define hypre_SStructNeighborILower(neighbor)  ((neighbor) -> ilower)
#define hypre_SStructNeighborCoord(neighbor)   ((neighbor) -> coord)
#define hypre_SStructNeighborDir(neighbor)     ((neighbor) -> dir)

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

