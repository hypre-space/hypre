
#include <HYPRE_config.h>

#include "HYPRE_sstruct_mv.h"

#ifndef hypre_SSTRUCT_MV_HEADER
#define hypre_SSTRUCT_MV_HEADER

#include "utilities.h"
#include "struct_mv.h"
#include "IJ_mv.h"
#include "HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * Header info for hypre_SStructStencil data structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_STENCIL_HEADER
#define hypre_SSTRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructStencil
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructStencil_struct
{
   hypre_StructStencil  *sstencil;
   int                  *vars;

   int                   ref_count;

} hypre_SStructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_SStructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_SStructStencilSStencil(stencil)     ((stencil) -> sstencil)
#define hypre_SStructStencilVars(stencil)         ((stencil) -> vars)
#define hypre_SStructStencilVar(stencil, i)       ((stencil) -> vars[i])
#define hypre_SStructStencilRefCount(stencil)     ((stencil) -> ref_count)

#define hypre_SStructStencilShape(stencil) \
hypre_StructStencilShape( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilSize(stencil) \
hypre_StructStencilSize( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilMaxOffset(stencil) \
hypre_StructStencilMaxOffset( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilNDim(stencil) \
hypre_StructStencilDim( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilEntry(stencil, i) \
hypre_StructStencilElement( hypre_SStructStencilSStencil(stencil), i )

#endif
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
 * Header info for the hypre_SStructGraph structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_GRAPH_HEADER
#define hypre_SSTRUCT_GRAPH_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructGraph:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int           to_part;
   hypre_Index   to_index;
   int           to_var;
   int           rank;

} hypre_SStructUEntry;

typedef struct
{
   int                  part;
   hypre_Index          index;
   int                  var;
   int                  nUentries;
   hypre_SStructUEntry *Uentries;

} hypre_SStructUVEntry;

typedef struct hypre_SStructGraph_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGrid      *grid;
   int                     nparts;
   hypre_SStructPGrid    **pgrids;
   hypre_SStructStencil ***stencils; /* each (part, var) has a stencil */

   /* U-graph info: Entries are referenced via local grid-variable rank. */
   int                     nUventries;  /* number of iUventries */
   int                     aUventries;  /* alloc size of iUventries */
   int                    *iUventries;
   hypre_SStructUVEntry  **Uventries;
   int                     totUentries;

   int                     ref_count;

} hypre_SStructGraph;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGraph
 *--------------------------------------------------------------------------*/

#define hypre_SStructGraphComm(graph)           ((graph) -> comm)
#define hypre_SStructGraphNDim(graph)           ((graph) -> ndim)
#define hypre_SStructGraphGrid(graph)           ((graph) -> grid)
#define hypre_SStructGraphNParts(graph)         ((graph) -> nparts)
#define hypre_SStructGraphPGrids(graph)         ((graph) -> pgrids)
#define hypre_SStructGraphPGrid(graph, p)       ((graph) -> pgrids[p])
#define hypre_SStructGraphStencils(graph)       ((graph) -> stencils)
#define hypre_SStructGraphStencil(graph, p, v)  ((graph) -> stencils[p][v])
#define hypre_SStructGraphNUVEntries(graph)     ((graph) -> nUventries)
#define hypre_SStructGraphAUVEntries(graph)     ((graph) -> aUventries)
#define hypre_SStructGraphIUVEntries(graph)     ((graph) -> iUventries)
#define hypre_SStructGraphIUVEntry(graph, i)    ((graph) -> iUventries[i])
#define hypre_SStructGraphUVEntries(graph)      ((graph) -> Uventries)
#define hypre_SStructGraphUVEntry(graph, i)     ((graph) -> Uventries[i])
#define hypre_SStructGraphTotUEntries(graph)    ((graph) -> totUentries)
#define hypre_SStructGraphRefCount(graph)       ((graph) -> ref_count)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUVEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUVEntryPart(Uv)        ((Uv) -> part)
#define hypre_SStructUVEntryIndex(Uv)       ((Uv) -> index)
#define hypre_SStructUVEntryVar(Uv)         ((Uv) -> var)
#define hypre_SStructUVEntryNUEntries(Uv)   ((Uv) -> nUentries)
#define hypre_SStructUVEntryUEntries(Uv)    ((Uv) -> Uentries)
#define hypre_SStructUVEntryUEntry(Uv, i)  &((Uv) -> Uentries[i])
#define hypre_SStructUVEntryToPart(Uv, i)   ((Uv) -> Uentries[i].to_part)
#define hypre_SStructUVEntryToIndex(Uv, i)  ((Uv) -> Uentries[i].to_index)
#define hypre_SStructUVEntryToVar(Uv, i)    ((Uv) -> Uentries[i].to_var)
#define hypre_SStructUVEntryRank(Uv, i)     ((Uv) -> Uentries[i].rank)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUEntryToPart(U)   ((U) -> to_part)
#define hypre_SStructUEntryToIndex(U)  ((U) -> to_index)
#define hypre_SStructUEntryToVar(U)    ((U) -> to_var)
#define hypre_SStructUEntryRank(U)     ((U) -> rank)

#endif

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
 * Header info for the hypre_SStructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_MATRIX_HEADER
#define hypre_SSTRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **stencils;     /* nvar array of stencils */

   int                     nvars;
   int                   **smaps;
   hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */

   /* temporary storage for SetValues routines */
   int                     sentries_size;
   int                    *sentries;

   int                     complex;      /* Is the matrix complex */

   int                     ref_count;

} hypre_SStructPMatrix;

typedef struct hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGraph     *graph;
   int                  ***splits;   /* S/U-matrix split for each stencil */

   /* S-matrix info */
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;

   /* U-matrix info */
   HYPRE_IJMatrix          ijmatrix;
   hypre_ParCSRMatrix     *parcsrmatrix;
                         
   /* temporary storage for SetValues routines */
   int                     entries_size;
   int                    *Sentries;
   int                    *Uentries;
   int                    *tmp_col_coords;
   double                 *tmp_coeffs;

   int                     symmetric;    /* Is the matrix symmetric */
   int                     complex;      /* Is the matrix complex */
   int                     global_size;  /* Total number of nonzero coeffs */

   int                     ref_count;

} hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructMatrixComm(mat)           ((mat) -> comm)
#define hypre_SStructMatrixNDim(mat)           ((mat) -> ndim)
#define hypre_SStructMatrixGraph(mat)          ((mat) -> graph)
#define hypre_SStructMatrixSplits(mat)         ((mat)-> splits)
#define hypre_SStructMatrixSplit(mat, p, v)    ((mat) -> splits[p][v])
#define hypre_SStructMatrixNParts(mat)         ((mat) -> nparts)
#define hypre_SStructMatrixPMatrices(mat)      ((mat) -> pmatrices)
#define hypre_SStructMatrixPMatrix(mat, part)  ((mat) -> pmatrices[part])
#define hypre_SStructMatrixIJMatrix(mat)       ((mat) -> ijmatrix)
#define hypre_SStructMatrixParCSRMatrix(mat)   ((mat) -> parcsrmatrix)
#define hypre_SStructMatrixEntriesSize(mat)    ((mat) -> entries_size)
#define hypre_SStructMatrixSEntries(mat)       ((mat) -> Sentries)
#define hypre_SStructMatrixUEntries(mat)       ((mat) -> Uentries)
#define hypre_SStructMatrixTmpColCoords(mat)   ((mat) -> tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffs(mat)      ((mat) -> tmp_coeffs)
#define hypre_SStructMatrixSymmetric(mat)      ((mat) -> symmetric)
#define hypre_SStructMatrixComplex(mat)        ((mat) -> complex)
#define hypre_SStructMatrixGlobalSize(mat)     ((mat) -> global_size)
#define hypre_SStructMatrixRefCount(mat)       ((mat) -> ref_count)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructPMatrixComm(pmat)              ((pmat) -> comm)
#define hypre_SStructPMatrixPGrid(pmat)             ((pmat) -> pgrid)
#define hypre_SStructPMatrixStencils(pmat)          ((pmat) -> stencils)
#define hypre_SStructPMatrixNVars(pmat)             ((pmat) -> nvars)
#define hypre_SStructPMatrixStencil(pmat, var)      ((pmat) -> stencils[var])
#define hypre_SStructPMatrixSMaps(pmat)             ((pmat) -> smaps)
#define hypre_SStructPMatrixSMap(pmat, var)         ((pmat) -> smaps[var])
#define hypre_SStructPMatrixSStencils(pmat)         ((pmat) -> sstencils)
#define hypre_SStructPMatrixSStencil(pmat, vi, vj) \
((pmat) -> sstencils[vi][vj])
#define hypre_SStructPMatrixSMatrices(pmat)         ((pmat) -> smatrices)
#define hypre_SStructPMatrixSMatrix(pmat, vi, vj)  \
((pmat) -> smatrices[vi][vj])
#define hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define hypre_SStructPMatrixComplex(pmat)           ((pmat) -> complex)
#define hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
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
 * Header info for the hypre_SStructVector structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_VECTOR_HEADER
#define hypre_SSTRUCT_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;

   int                     nvars;
   hypre_StructVector    **svectors;     /* nvar array of svectors */
   hypre_CommPkg         **comm_pkgs;    /* nvar array of comm pkgs */

   int                     complex;      /* Is the vector complex */

   int                     ref_count;

} hypre_SStructPVector;

typedef struct hypre_SStructVector_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGrid      *grid;
   int                     object_type;

   /* s-vector info */
   int                     nparts;
   hypre_SStructPVector  **pvectors;
   hypre_CommPkg        ***comm_pkgs;    /* nvar array of comm pkgs */

   /* u-vector info */
   HYPRE_IJVector          ijvector;
   hypre_ParVector        *parvector;

   int                     complex;      /* Is the vector complex */
   int                     global_size;  /* Total number coefficients */

   int                     ref_count;

} hypre_SStructVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructVector
 *--------------------------------------------------------------------------*/

#define hypre_SStructVectorComm(vec)           ((vec) -> comm)
#define hypre_SStructVectorNDim(vec)           ((vec) -> ndim)
#define hypre_SStructVectorGrid(vec)           ((vec) -> grid)
#define hypre_SStructVectorObjectType(vec)     ((vec) -> object_type)
#define hypre_SStructVectorNParts(vec)         ((vec) -> nparts)
#define hypre_SStructVectorPVectors(vec)       ((vec) -> pvectors)
#define hypre_SStructVectorPVector(vec, part)  ((vec) -> pvectors[part])
#define hypre_SStructVectorIJVector(vec)       ((vec) -> ijvector)
#define hypre_SStructVectorParVector(vec)      ((vec) -> parvector)
#define hypre_SStructVectorComplex(vec)        ((vec) -> complex)
#define hypre_SStructVectorGlobalSize(vec)     ((vec) -> global_size)
#define hypre_SStructVectorRefCount(vec)       ((vec) -> ref_count)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPVector
 *--------------------------------------------------------------------------*/

#define hypre_SStructPVectorComm(pvec)        ((pvec) -> comm)
#define hypre_SStructPVectorPGrid(pvec)       ((pvec) -> pgrid)
#define hypre_SStructPVectorNVars(pvec)       ((pvec) -> nvars)
#define hypre_SStructPVectorSVectors(pvec)    ((pvec) -> svectors)
#define hypre_SStructPVectorSVector(pvec, v)  ((pvec) -> svectors[v])
#define hypre_SStructPVectorCommPkgs(pvec)    ((pvec) -> comm_pkgs)
#define hypre_SStructPVectorCommPkg(pvec, v)  ((pvec) -> comm_pkgs[v])
#define hypre_SStructPVectorComplex(pvec)     ((pvec) -> complex)
#define hypre_SStructPVectorRefCount(vec)     ((vec) -> ref_count)

#endif

/* HYPRE_sstruct_graph.c */
int HYPRE_SStructGraphCreate( MPI_Comm comm , HYPRE_SStructGrid grid , HYPRE_SStructGraph *graph_ptr );
int HYPRE_SStructGraphDestroy( HYPRE_SStructGraph graph );
int HYPRE_SStructGraphSetStencil( HYPRE_SStructGraph graph , int part , int var , HYPRE_SStructStencil stencil );
int HYPRE_SStructGraphAddEntries( HYPRE_SStructGraph graph , int part , int *index , int var , int to_part , int *to_index , int to_var );
int HYPRE_SStructGraphAssemble( HYPRE_SStructGraph graph );

/* HYPRE_sstruct_grid.c */
int HYPRE_SStructGridCreate( MPI_Comm comm , int ndim , int nparts , HYPRE_SStructGrid *grid_ptr );
int HYPRE_SStructGridDestroy( HYPRE_SStructGrid grid );
int HYPRE_SStructGridSetExtents( HYPRE_SStructGrid grid , int part , int *ilower , int *iupper );
int HYPRE_SStructGridSetVariables( HYPRE_SStructGrid grid , int part , int nvars , HYPRE_SStructVariable *vartypes );
int HYPRE_SStructGridAddVariables( HYPRE_SStructGrid grid , int part , int *index , int nvars , HYPRE_SStructVariable *vartypes );
int HYPRE_SStructGridSetNeighborBox( HYPRE_SStructGrid grid , int part , int *ilower , int *iupper , int nbor_part , int *nbor_ilower , int *nbor_iupper , int *index_map );
int HYPRE_SStructGridAddUnstructuredPart( HYPRE_SStructGrid grid , int ilower , int iupper );
int HYPRE_SStructGridAssemble( HYPRE_SStructGrid grid );
int HYPRE_SStructGridSetPeriodic( HYPRE_SStructGrid grid , int part , int *periodic );

/* HYPRE_sstruct_matrix.c */
int HYPRE_SStructMatrixCreate( MPI_Comm comm , HYPRE_SStructGraph graph , HYPRE_SStructMatrix *matrix_ptr );
int HYPRE_SStructMatrixDestroy( HYPRE_SStructMatrix matrix );
int HYPRE_SStructMatrixInitialize( HYPRE_SStructMatrix matrix );
int HYPRE_SStructMatrixSetValues( HYPRE_SStructMatrix matrix , int part , int *index , int var , int nentries , int *entries , double *values );
int HYPRE_SStructMatrixSetBoxValues( HYPRE_SStructMatrix matrix , int part , int *ilower , int *iupper , int var , int nentries , int *entries , double *values );
int HYPRE_SStructMatrixAddToValues( HYPRE_SStructMatrix matrix , int part , int *index , int var , int nentries , int *entries , double *values );
int HYPRE_SStructMatrixAddToBoxValues( HYPRE_SStructMatrix matrix , int part , int *ilower , int *iupper , int var , int nentries , int *entries , double *values );
int HYPRE_SStructMatrixAssemble( HYPRE_SStructMatrix matrix );
int HYPRE_SStructMatrixSetSymmetric( HYPRE_SStructMatrix matrix , int symmetric );
int HYPRE_SStructMatrixSetObjectType( HYPRE_SStructMatrix matrix , int type );
int HYPRE_SStructMatrixGetObject( HYPRE_SStructMatrix matrix , void **object );
int HYPRE_SStructMatrixPrint( const char *filename , HYPRE_SStructMatrix matrix , int all );

/* HYPRE_sstruct_stencil.c */
int HYPRE_SStructStencilCreate( int ndim , int size , HYPRE_SStructStencil *stencil_ptr );
int HYPRE_SStructStencilDestroy( HYPRE_SStructStencil stencil );
int HYPRE_SStructStencilSetEntry( HYPRE_SStructStencil stencil , int entry , int *offset , int var );

/* HYPRE_sstruct_vector.c */
int HYPRE_SStructVectorCreate( MPI_Comm comm , HYPRE_SStructGrid grid , HYPRE_SStructVector *vector_ptr );
int HYPRE_SStructVectorDestroy( HYPRE_SStructVector vector );
int HYPRE_SStructVectorInitialize( HYPRE_SStructVector vector );
int HYPRE_SStructVectorSetValues( HYPRE_SStructVector vector , int part , int *index , int var , double *value );
int HYPRE_SStructVectorSetBoxValues( HYPRE_SStructVector vector , int part , int *ilower , int *iupper , int var , double *values );
int HYPRE_SStructVectorAddToValues( HYPRE_SStructVector vector , int part , int *index , int var , double *value );
int HYPRE_SStructVectorAddToBoxValues( HYPRE_SStructVector vector , int part , int *ilower , int *iupper , int var , double *values );
int HYPRE_SStructVectorAssemble( HYPRE_SStructVector vector );
int HYPRE_SStructVectorGather( HYPRE_SStructVector vector );
int HYPRE_SStructVectorGetValues( HYPRE_SStructVector vector , int part , int *index , int var , double *value );
int HYPRE_SStructVectorGetBoxValues( HYPRE_SStructVector vector , int part , int *ilower , int *iupper , int var , double *values );
int HYPRE_SStructVectorSetObjectType( HYPRE_SStructVector vector , int type );
int HYPRE_SStructVectorGetObject( HYPRE_SStructVector vector , void **object );
int HYPRE_SStructVectorPrint( const char *filename , HYPRE_SStructVector vector , int all );

/* box_map.c */
int hypre_BoxMapEntrySetInfo( hypre_BoxMapEntry *entry , void *info );
int hypre_BoxMapEntryGetInfo( hypre_BoxMapEntry *entry , void **info_ptr );
int hypre_BoxMapEntryGetExtents( hypre_BoxMapEntry *entry , hypre_Index imin , hypre_Index imax );
int hypre_BoxMapCreate( int max_nentries , hypre_Index global_imin , hypre_Index global_imax , hypre_BoxMap **map_ptr );
int hypre_BoxMapAddEntry( hypre_BoxMap *map , hypre_Index imin , hypre_Index imax , void *info );
int hypre_BoxMapAssemble( hypre_BoxMap *map );
int hypre_BoxMapDestroy( hypre_BoxMap *map );
int hypre_BoxMapFindEntry( hypre_BoxMap *map , hypre_Index index , hypre_BoxMapEntry **entry_ptr );

/* sstruct_axpy.c */
int hypre_SStructPAxpy( double alpha , hypre_SStructPVector *px , hypre_SStructPVector *py );
int hypre_SStructAxpy( double alpha , hypre_SStructVector *x , hypre_SStructVector *y );

/* sstruct_copy.c */
int hypre_SStructPCopy( hypre_SStructPVector *px , hypre_SStructPVector *py );
int hypre_SStructCopy( hypre_SStructVector *x , hypre_SStructVector *y );

/* sstruct_graph.c */
int hypre_SStructGraphRef( hypre_SStructGraph *graph , hypre_SStructGraph **graph_ref );
int hypre_SStructGraphFindUVEntry( hypre_SStructGraph *graph , int part , hypre_Index index , int var , hypre_SStructUVEntry **Uventry_ptr );

/* sstruct_grid.c */
int hypre_SStructVariableGetOffset( HYPRE_SStructVariable vartype , int ndim , hypre_Index varoffset );
int hypre_SStructPGridCreate( MPI_Comm comm , int ndim , hypre_SStructPGrid **pgrid_ptr );
int hypre_SStructPGridDestroy( hypre_SStructPGrid *pgrid );
int hypre_SStructPGridSetExtents( hypre_SStructPGrid *pgrid , hypre_Index ilower , hypre_Index iupper );
int hypre_SStructPGridSetVariables( hypre_SStructPGrid *pgrid , int nvars , HYPRE_SStructVariable *vartypes );
int hypre_SStructPGridSetPNeighbor( hypre_SStructPGrid *pgrid , hypre_Box *pneighbor_box );
int hypre_SStructPGridAssemble( hypre_SStructPGrid *pgrid );
int hypre_SStructGridRef( hypre_SStructGrid *grid , hypre_SStructGrid **grid_ref );
int hypre_SStructGridAssembleMaps( hypre_SStructGrid *grid );
int hypre_SStructGridFindMapEntry( hypre_SStructGrid *grid , int part , hypre_Index index , int var , hypre_BoxMapEntry **entry_ptr );
int hypre_SStructMapEntryGetGlobalRank( hypre_BoxMapEntry *entry , hypre_Index index , int *rank_ptr );
int hypre_SStructMapEntryGetProcess( hypre_BoxMapEntry *entry , int *proc_ptr );

/* sstruct_innerprod.c */
int hypre_SStructPInnerProd( hypre_SStructPVector *px , hypre_SStructPVector *py , double *presult_ptr );
int hypre_SStructInnerProd( hypre_SStructVector *x , hypre_SStructVector *y , double *result_ptr );

/* sstruct_matrix.c */
int hypre_SStructPMatrixRef( hypre_SStructPMatrix *matrix , hypre_SStructPMatrix **matrix_ref );
int hypre_SStructPMatrixCreate( MPI_Comm comm , hypre_SStructPGrid *pgrid , hypre_SStructStencil **stencils , hypre_SStructPMatrix **pmatrix_ptr );
int hypre_SStructPMatrixDestroy( hypre_SStructPMatrix *pmatrix );
int hypre_SStructPMatrixInitialize( hypre_SStructPMatrix *pmatrix );
int hypre_SStructPMatrixSetValues( hypre_SStructPMatrix *pmatrix , hypre_Index index , int var , int nentries , int *entries , double *values , int add_to );
int hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix , hypre_Index ilower , hypre_Index iupper , int var , int nentries , int *entries , double *values , int add_to );
int hypre_SStructPMatrixAssemble( hypre_SStructPMatrix *pmatrix );
int hypre_SStructPMatrixPrint( const char *filename , hypre_SStructPMatrix *pmatrix , int all );
int hypre_SStructUMatrixInitialize( hypre_SStructMatrix *matrix );
int hypre_SStructUMatrixSetValues( hypre_SStructMatrix *matrix , int part , hypre_Index index , int var , int nentries , int *entries , double *values , int add_to );
int hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix , int part , hypre_Index ilower , hypre_Index iupper , int var , int nentries , int *entries , double *values , int add_to );
int hypre_SStructUMatrixAssemble( hypre_SStructMatrix *matrix );
int hypre_SStructMatrixRef( hypre_SStructMatrix *matrix , hypre_SStructMatrix **matrix_ref );
int hypre_SStructMatrixSplitEntries( hypre_SStructMatrix *matrix , int part , int var , int nentries , int *entries , int *nSentries_ptr , int **Sentries_ptr , int *nUentries_ptr , int **Uentries_ptr );

/* sstruct_matvec.c */
int hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr );
int hypre_SStructPMatvecSetup( void *pmatvec_vdata , hypre_SStructPMatrix *pA , hypre_SStructPVector *px );
int hypre_SStructPMatvecCompute( void *pmatvec_vdata , double alpha , hypre_SStructPMatrix *pA , hypre_SStructPVector *px , double beta , hypre_SStructPVector *py );
int hypre_SStructPMatvecDestroy( void *pmatvec_vdata );
int hypre_SStructPMatvec( double alpha , hypre_SStructPMatrix *pA , hypre_SStructPVector *px , double beta , hypre_SStructPVector *py );
int hypre_SStructMatvecCreate( void **matvec_vdata_ptr );
int hypre_SStructMatvecSetup( void *matvec_vdata , hypre_SStructMatrix *A , hypre_SStructVector *x );
int hypre_SStructMatvecCompute( void *matvec_vdata , double alpha , hypre_SStructMatrix *A , hypre_SStructVector *x , double beta , hypre_SStructVector *y );
int hypre_SStructMatvecDestroy( void *matvec_vdata );
int hypre_SStructMatvec( double alpha , hypre_SStructMatrix *A , hypre_SStructVector *x , double beta , hypre_SStructVector *y );

/* sstruct_scale.c */
int hypre_SStructPScale( double alpha , hypre_SStructPVector *py );
int hypre_SStructScale( double alpha , hypre_SStructVector *y );

/* sstruct_stencil.c */
int hypre_SStructStencilRef( hypre_SStructStencil *stencil , hypre_SStructStencil **stencil_ref );

/* sstruct_vector.c */
int hypre_SStructPVectorRef( hypre_SStructPVector *vector , hypre_SStructPVector **vector_ref );
int hypre_SStructPVectorCreate( MPI_Comm comm , hypre_SStructPGrid *pgrid , hypre_SStructPVector **pvector_ptr );
int hypre_SStructPVectorDestroy( hypre_SStructPVector *pvector );
int hypre_SStructPVectorInitialize( hypre_SStructPVector *pvector );
int hypre_SStructPVectorSetValues( hypre_SStructPVector *pvector , hypre_Index index , int var , double *value , int add_to );
int hypre_SStructPVectorSetBoxValues( hypre_SStructPVector *pvector , hypre_Index ilower , hypre_Index iupper , int var , double *values , int add_to );
int hypre_SStructPVectorAssemble( hypre_SStructPVector *pvector );
int hypre_SStructPVectorGather( hypre_SStructPVector *pvector );
int hypre_SStructPVectorGetValues( hypre_SStructPVector *pvector , hypre_Index index , int var , double *value );
int hypre_SStructPVectorGetBoxValues( hypre_SStructPVector *pvector , hypre_Index ilower , hypre_Index iupper , int var , double *values );
int hypre_SStructPVectorSetConstantValues( hypre_SStructPVector *pvector , double value );
int hypre_SStructPVectorPrint( const char *filename , hypre_SStructPVector *pvector , int all );
int hypre_SStructVectorRef( hypre_SStructVector *vector , hypre_SStructVector **vector_ref );
int hypre_SStructVectorSetConstantValues( hypre_SStructVector *vector , double value );
int hypre_SStructVectorConvert( hypre_SStructVector *vector , hypre_ParVector **parvector_ptr );
int hypre_SStructVectorRestore( hypre_SStructVector *vector , hypre_ParVector *parvector );


#ifdef __cplusplus
}
#endif

#endif

