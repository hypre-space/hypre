/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/

#ifndef hypre_SSTRUCT_MV_HEADER
#define hypre_SSTRUCT_MV_HEADER

#include "HYPRE_sstruct_mv.h"

#include "_hypre_utilities.h"
#include "_hypre_struct_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE.h"

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/


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

typedef HYPRE_Int hypre_SStructVariable;

typedef struct
{
   HYPRE_SStructVariable  type;
   HYPRE_Int              rank;     /* local rank */
   HYPRE_Int              proc;

} hypre_SStructUVar;

typedef struct
{
   HYPRE_Int              part;
   hypre_Index            cell;
   HYPRE_Int              nuvars;
   hypre_SStructUVar     *uvars;

} hypre_SStructUCVar;

typedef struct
{
   MPI_Comm                comm;             /* TODO: use different comms */
   HYPRE_Int               ndim;
   HYPRE_Int               nvars;            /* number of variables */
   HYPRE_SStructVariable  *vartypes;         /* types of variables */
   hypre_StructGrid       *sgrids[8];        /* struct grids for each vartype */
   hypre_BoxArray         *iboxarrays[8];    /* interface boxes */
                                       
   hypre_BoxArray         *pneighbors;
   hypre_Index            *pnbor_offsets;

   HYPRE_Int               local_size;       /* Number of variables locally */
   HYPRE_Int               global_size;      /* Total number of variables */

   hypre_Index             periodic;         /* Indicates if pgrid is periodic */

  /* GEC0902 additions for ghost expansion of boxes */

   HYPRE_Int               ghlocal_size;     /* Number of vars including ghosts */
                           
   HYPRE_Int               cell_sgrid_done;  /* =1 implies cell grid already assembled */
} hypre_SStructPGrid;

typedef struct
{
   hypre_Box    box;
   HYPRE_Int    part;
   hypre_Index  ilower; /* box ilower, but on the neighbor index-space */
   hypre_Index  coord;  /* lives on local index-space */
   hypre_Index  dir;    /* lives on local index-space */

} hypre_SStructNeighbor;

enum hypre_SStructBoxManInfoType
{
   hypre_SSTRUCT_BOXMAN_INFO_DEFAULT  = 0,
   hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR = 1
};

typedef struct
{
   HYPRE_Int  type;
   HYPRE_Int  offset;
   HYPRE_Int  ghoffset; 

} hypre_SStructBoxManInfo;

typedef struct
{
   HYPRE_Int    type;
   HYPRE_Int    offset;   /* minimum offset for this box */
   HYPRE_Int    ghoffset; /* minimum offset ghost for this box */
   HYPRE_Int    proc;      /* redundant with the proc in the entry, but
                              makes some coding easier */
   HYPRE_Int    boxnum;   /* this is different from the entry id */ 
   HYPRE_Int    part;     /* part the box lives on */
   hypre_Index  ilower;   /* box ilower, but on the neighbor index-space */
   hypre_Index  coord;    /* lives on local index-space */
   hypre_Index  dir;      /* lives on local index-space */
   hypre_Index  stride;   /* lives on local index-space */
   hypre_Index  ghstride; /* the ghost equivalent of strides */ 

} hypre_SStructBoxManNborInfo;

typedef struct
{
   hypre_CommInfo  *comm_info;
   HYPRE_Int        send_part;
   HYPRE_Int        recv_part;
   HYPRE_Int        send_var;
   HYPRE_Int        recv_var;
   
} hypre_SStructCommInfo;

typedef struct hypre_SStructGrid_struct
{
   MPI_Comm                   comm;
   HYPRE_Int                  ndim;
   HYPRE_Int                  nparts;
                          
   /* s-variable info */  
   hypre_SStructPGrid       **pgrids;
                          
   /* neighbor info */    
   HYPRE_Int                 *nneighbors;
   hypre_SStructNeighbor    **neighbors;
   hypre_Index              **nbor_offsets;
   HYPRE_Int                **nvneighbors;
   hypre_SStructNeighbor   ***vneighbors;
   hypre_SStructCommInfo    **vnbor_comm_info; /* for updating shared data */
   HYPRE_Int                  vnbor_ncomms;

   /* u-variables info: During construction, array entries are consecutive.
    * After 'Assemble', entries are referenced via local cell rank. */
   HYPRE_Int                  nucvars;
   hypre_SStructUCVar       **ucvars;

   /* info for fem-based user input (for each part) */
   HYPRE_Int                 *fem_nvars;
   HYPRE_Int                **fem_vars;
   hypre_Index              **fem_offsets;

   /* info for mapping (part, index, var) --> rank */
   hypre_BoxManager        ***boxmans;      /* manager for each part, var */
   hypre_BoxManager        ***nbor_boxmans; /* manager for each part, var */

   HYPRE_Int                  start_rank;

   HYPRE_Int                  local_size;  /* Number of variables locally */
   HYPRE_Int                  global_size; /* Total number of variables */
                              
   HYPRE_Int                  ref_count;

 /* GEC0902 additions for ghost expansion of boxes */

   HYPRE_Int               ghlocal_size;  /* GEC0902 Number of vars including ghosts */
   HYPRE_Int               ghstart_rank;  /* GEC0902 start rank including ghosts  */

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
#define hypre_SStructGridNborOffsets(grid)    ((grid) -> nbor_offsets)
#define hypre_SStructGridNVNeighbors(grid)    ((grid) -> nvneighbors)
#define hypre_SStructGridVNeighbors(grid)     ((grid) -> vneighbors)
#define hypre_SStructGridVNborCommInfo(grid)  ((grid) -> vnbor_comm_info)
#define hypre_SStructGridVNborNComms(grid)    ((grid) -> vnbor_ncomms)
#define hypre_SStructGridNUCVars(grid)        ((grid) -> nucvars)
#define hypre_SStructGridUCVars(grid)         ((grid) -> ucvars)
#define hypre_SStructGridUCVar(grid, i)       ((grid) -> ucvars[i])

#define hypre_SStructGridFEMNVars(grid)       ((grid) -> fem_nvars)
#define hypre_SStructGridFEMVars(grid)        ((grid) -> fem_vars)
#define hypre_SStructGridFEMOffsets(grid)     ((grid) -> fem_offsets)
#define hypre_SStructGridFEMPNVars(grid, part)   ((grid) -> fem_nvars[part])
#define hypre_SStructGridFEMPVars(grid, part)    ((grid) -> fem_vars[part])
#define hypre_SStructGridFEMPOffsets(grid, part) ((grid) -> fem_offsets[part])

#define hypre_SStructGridBoxManagers(grid)           ((grid) -> boxmans)
#define hypre_SStructGridBoxManager(grid, part, var) ((grid) -> boxmans[part][var])

#define hypre_SStructGridNborBoxManagers(grid)           ((grid) -> nbor_boxmans)
#define hypre_SStructGridNborBoxManager(grid, part, var) ((grid) -> nbor_boxmans[part][var])

#define hypre_SStructGridStartRank(grid)      ((grid) -> start_rank)
#define hypre_SStructGridLocalSize(grid)      ((grid) -> local_size)
#define hypre_SStructGridGlobalSize(grid)     ((grid) -> global_size)
#define hypre_SStructGridRefCount(grid)       ((grid) -> ref_count)
#define hypre_SStructGridGhlocalSize(grid)    ((grid) -> ghlocal_size)
#define hypre_SStructGridGhstartRank(grid)    ((grid) -> ghstart_rank)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPGrid
 *--------------------------------------------------------------------------*/

#define hypre_SStructPGridComm(pgrid)             ((pgrid) -> comm)
#define hypre_SStructPGridNDim(pgrid)             ((pgrid) -> ndim)
#define hypre_SStructPGridNVars(pgrid)            ((pgrid) -> nvars)
#define hypre_SStructPGridVarTypes(pgrid)         ((pgrid) -> vartypes)
#define hypre_SStructPGridVarType(pgrid, var)     ((pgrid) -> vartypes[var])
#define hypre_SStructPGridCellSGridDone(pgrid)    ((pgrid) -> cell_sgrid_done)

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
#define hypre_SStructPGridPNborOffsets(pgrid)     ((pgrid) -> pnbor_offsets)
#define hypre_SStructPGridLocalSize(pgrid)        ((pgrid) -> local_size)
#define hypre_SStructPGridGlobalSize(pgrid)       ((pgrid) -> global_size)
#define hypre_SStructPGridPeriodic(pgrid)         ((pgrid) -> periodic)
#define hypre_SStructPGridGhlocalSize(pgrid)      ((pgrid) -> ghlocal_size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructBoxManInfo
 *--------------------------------------------------------------------------*/

#define hypre_SStructBoxManInfoType(info)            ((info) -> type)
#define hypre_SStructBoxManInfoOffset(info)          ((info) -> offset)
#define hypre_SStructBoxManInfoGhoffset(info)        ((info) -> ghoffset)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructBoxManInfo
 *--------------------------------------------------------------------------*/

/* Use the MapInfo macros to access the first three structure components */
#define hypre_SStructBoxManNborInfoProc(info)    ((info) -> proc)
#define hypre_SStructBoxManNborInfoBoxnum(info)  ((info) -> boxnum)
#define hypre_SStructBoxManNborInfoPart(info)    ((info) -> part)
#define hypre_SStructBoxManNborInfoILower(info)  ((info) -> ilower)
#define hypre_SStructBoxManNborInfoCoord(info)   ((info) -> coord)
#define hypre_SStructBoxManNborInfoDir(info)     ((info) -> dir)
#define hypre_SStructBoxManNborInfoStride(info)  ((info) -> stride)
#define hypre_SStructBoxManNborInfoGhstride(info)  ((info) -> ghstride)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructNeighbor
 *--------------------------------------------------------------------------*/

#define hypre_SStructNeighborBox(neighbor)    &((neighbor) -> box)
#define hypre_SStructNeighborPart(neighbor)    ((neighbor) -> part)
#define hypre_SStructNeighborILower(neighbor)  ((neighbor) -> ilower)
#define hypre_SStructNeighborCoord(neighbor)   ((neighbor) -> coord)
#define hypre_SStructNeighborDir(neighbor)     ((neighbor) -> dir)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructCommInfo
 *--------------------------------------------------------------------------*/

#define hypre_SStructCommInfoCommInfo(cinfo)  ((cinfo) -> comm_info)
#define hypre_SStructCommInfoSendPart(cinfo)  ((cinfo) -> send_part)
#define hypre_SStructCommInfoRecvPart(cinfo)  ((cinfo) -> recv_part)
#define hypre_SStructCommInfoSendVar(cinfo)   ((cinfo) -> send_var)
#define hypre_SStructCommInfoRecvVar(cinfo)   ((cinfo) -> recv_var)

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
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




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
   HYPRE_Int            *vars;

   HYPRE_Int             ref_count;

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
#define hypre_SStructStencilNDim(stencil) \
hypre_StructStencilDim( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilEntry(stencil, i) \
hypre_StructStencilElement( hypre_SStructStencilSStencil(stencil), i )

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/


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
   HYPRE_Int     part;
   hypre_Index   index;
   HYPRE_Int     var;
   HYPRE_Int     to_part;     
   hypre_Index   to_index;
   HYPRE_Int     to_var;

} hypre_SStructGraphEntry;



typedef struct
{
   HYPRE_Int     to_part;
   hypre_Index   to_index;
   HYPRE_Int     to_var;
   HYPRE_Int     to_boxnum;      /* local box number */
   HYPRE_Int     to_proc;
   HYPRE_Int     rank;

} hypre_SStructUEntry;

typedef struct
{
   HYPRE_Int            part;
   hypre_Index          index;
   HYPRE_Int            var;
   HYPRE_Int            boxnum;  /* local box number */
   HYPRE_Int            nUentries;
   hypre_SStructUEntry *Uentries;

} hypre_SStructUVEntry;

typedef struct hypre_SStructGraph_struct
{
   MPI_Comm                comm;
   HYPRE_Int               ndim;
   hypre_SStructGrid      *grid;
   hypre_SStructGrid      *domain_grid; /* same as grid by default */
   HYPRE_Int               nparts;
   hypre_SStructPGrid    **pgrids;
   hypre_SStructStencil ***stencils; /* each (part, var) has a stencil */

   /* info for fem-based user input */
   HYPRE_Int              *fem_nsparse;
   HYPRE_Int             **fem_sparse_i;
   HYPRE_Int             **fem_sparse_j;
   HYPRE_Int             **fem_entries;

   /* U-graph info: Entries are referenced via local grid-variable rank. */
   HYPRE_Int               nUventries;  /* number of iUventries */
   HYPRE_Int               aUventries;  /* alloc size of iUventries */
   HYPRE_Int              *iUventries;

   hypre_SStructUVEntry  **Uventries;
   HYPRE_Int               totUentries;

   HYPRE_Int               ref_count;

   HYPRE_Int               type;    /* GEC0203 */

   hypre_SStructGraphEntry **graph_entries; /* these are stored from
                                             * the AddGraphEntries calls
                                             * and then deleted in the
                                             * GraphAssemble */
   HYPRE_Int               n_graph_entries; /* number graph entries */
   HYPRE_Int               a_graph_entries; /* alloced graph entries */
   


} hypre_SStructGraph;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGraph
 *--------------------------------------------------------------------------*/

#define hypre_SStructGraphComm(graph)           ((graph) -> comm)
#define hypre_SStructGraphNDim(graph)           ((graph) -> ndim)
#define hypre_SStructGraphGrid(graph)           ((graph) -> grid)
#define hypre_SStructGraphDomainGrid(graph)     ((graph) -> domain_grid)
#define hypre_SStructGraphNParts(graph)         ((graph) -> nparts)
#define hypre_SStructGraphPGrids(graph) \
   hypre_SStructGridPGrids(hypre_SStructGraphGrid(graph))
#define hypre_SStructGraphPGrid(graph, p) \
   hypre_SStructGridPGrid(hypre_SStructGraphGrid(graph), p)
#define hypre_SStructGraphStencils(graph)       ((graph) -> stencils)
#define hypre_SStructGraphStencil(graph, p, v)  ((graph) -> stencils[p][v])

#define hypre_SStructGraphFEMNSparse(graph)     ((graph) -> fem_nsparse)
#define hypre_SStructGraphFEMSparseI(graph)     ((graph) -> fem_sparse_i)
#define hypre_SStructGraphFEMSparseJ(graph)     ((graph) -> fem_sparse_j)
#define hypre_SStructGraphFEMEntries(graph)     ((graph) -> fem_entries)
#define hypre_SStructGraphFEMPNSparse(graph, p) ((graph) -> fem_nsparse[p])
#define hypre_SStructGraphFEMPSparseI(graph, p) ((graph) -> fem_sparse_i[p])
#define hypre_SStructGraphFEMPSparseJ(graph, p) ((graph) -> fem_sparse_j[p])
#define hypre_SStructGraphFEMPEntries(graph, p) ((graph) -> fem_entries[p])

#define hypre_SStructGraphNUVEntries(graph)     ((graph) -> nUventries)
#define hypre_SStructGraphAUVEntries(graph)     ((graph) -> aUventries)
#define hypre_SStructGraphIUVEntries(graph)     ((graph) -> iUventries)
#define hypre_SStructGraphIUVEntry(graph, i)    ((graph) -> iUventries[i])
#define hypre_SStructGraphUVEntries(graph)      ((graph) -> Uventries)
#define hypre_SStructGraphUVEntry(graph, i)     ((graph) -> Uventries[i])
#define hypre_SStructGraphTotUEntries(graph)    ((graph) -> totUentries)
#define hypre_SStructGraphRefCount(graph)       ((graph) -> ref_count)
#define hypre_SStructGraphObjectType(graph)     ((graph) -> type)
#define hypre_SStructGraphEntries(graph)        ((graph) -> graph_entries)
#define hypre_SStructNGraphEntries(graph)       ((graph) -> n_graph_entries)
#define hypre_SStructAGraphEntries(graph)       ((graph) -> a_graph_entries)


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUVEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUVEntryPart(Uv)        ((Uv) -> part)
#define hypre_SStructUVEntryIndex(Uv)       ((Uv) -> index)
#define hypre_SStructUVEntryVar(Uv)         ((Uv) -> var)
#define hypre_SStructUVEntryBoxnum(Uv)      ((Uv) -> boxnum)
#define hypre_SStructUVEntryNUEntries(Uv)   ((Uv) -> nUentries)
#define hypre_SStructUVEntryUEntries(Uv)    ((Uv) -> Uentries)
#define hypre_SStructUVEntryUEntry(Uv, i)  &((Uv) -> Uentries[i])
#define hypre_SStructUVEntryToPart(Uv, i)   ((Uv) -> Uentries[i].to_part)
#define hypre_SStructUVEntryToIndex(Uv, i)  ((Uv) -> Uentries[i].to_index)
#define hypre_SStructUVEntryToVar(Uv, i)    ((Uv) -> Uentries[i].to_var)
#define hypre_SStructUVEntryToBoxnum(Uv, i) ((Uv) -> Uentries[i].to_boxnum)
#define hypre_SStructUVEntryToProc(Uv, i)   ((Uv) -> Uentries[i].to_proc)
#define hypre_SStructUVEntryRank(Uv, i)     ((Uv) -> Uentries[i].rank)
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUEntryToPart(U)   ((U) -> to_part)
#define hypre_SStructUEntryToIndex(U)  ((U) -> to_index)
#define hypre_SStructUEntryToVar(U)    ((U) -> to_var)
#define hypre_SStructUEntryToBoxnum(U) ((U) -> to_boxnum)
#define hypre_SStructUEntryToProc(U)   ((U) -> to_proc)
#define hypre_SStructUEntryRank(U)     ((U) -> rank)


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGraphEntry
 *--------------------------------------------------------------------------*/
#define hypre_SStructGraphEntryPart(g)     ((g) -> part)
#define hypre_SStructGraphEntryIndex(g)    ((g) -> index)
#define hypre_SStructGraphEntryVar(g)      ((g) -> var)
#define hypre_SStructGraphEntryToPart(g)   ((g) -> to_part)
#define hypre_SStructGraphEntryToIndex(g)  ((g) -> to_index)
#define hypre_SStructGraphEntryToVar(g)    ((g) -> to_var)




#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




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

   HYPRE_Int               nvars;
   HYPRE_Int             **smaps;
   hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */
   HYPRE_Int             **symmetric;    /* Stencil entries symmetric?
                                          * (nvar x nvar array) */

   /* temporary storage for SetValues routines */
   HYPRE_Int               sentries_size;
   HYPRE_Int              *sentries;

   HYPRE_Int               accumulated;  /* AddTo values accumulated? */
   HYPRE_Int               complex;      /* Matrix complex? */

   HYPRE_Int               ref_count;

} hypre_SStructPMatrix;

typedef struct hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   HYPRE_Int               ndim;
   hypre_SStructGraph     *graph;
   HYPRE_Int            ***splits;   /* S/U-matrix split for each stencil */

   /* S-matrix info */
   HYPRE_Int               nparts;
   hypre_SStructPMatrix  **pmatrices;
   HYPRE_Int            ***symmetric;    /* Stencil entries symmetric?
                                          * (nparts x nvar x nvar array) */

   /* U-matrix info */
   HYPRE_IJMatrix          ijmatrix;
   hypre_ParCSRMatrix     *parcsrmatrix;
                         
   /* temporary storage for SetValues routines */
   HYPRE_Int               entries_size;
   HYPRE_Int              *Sentries;
   HYPRE_Int              *Uentries;
   HYPRE_Int              *tmp_col_coords;
   double                 *tmp_coeffs;

   HYPRE_Int               ns_symmetric; /* Non-stencil entries symmetric? */
   HYPRE_Int               complex;      /* Matrix complex? */
   HYPRE_Int               global_size;  /* Total number of nonzero coeffs */

   HYPRE_Int               ref_count;

  /* GEC0902   adding an object type to the matrix  */
   HYPRE_Int               object_type;

} hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructMatrixComm(mat)           ((mat) -> comm)
#define hypre_SStructMatrixNDim(mat)           ((mat) -> ndim)
#define hypre_SStructMatrixGraph(mat)          ((mat) -> graph)
#define hypre_SStructMatrixSplits(mat)         ((mat) -> splits)
#define hypre_SStructMatrixSplit(mat, p, v)    ((mat) -> splits[p][v])
#define hypre_SStructMatrixNParts(mat)         ((mat) -> nparts)
#define hypre_SStructMatrixPMatrices(mat)      ((mat) -> pmatrices)
#define hypre_SStructMatrixPMatrix(mat, part)  ((mat) -> pmatrices[part])
#define hypre_SStructMatrixSymmetric(mat)      ((mat) -> symmetric)
#define hypre_SStructMatrixIJMatrix(mat)       ((mat) -> ijmatrix)
#define hypre_SStructMatrixParCSRMatrix(mat)   ((mat) -> parcsrmatrix)
#define hypre_SStructMatrixEntriesSize(mat)    ((mat) -> entries_size)
#define hypre_SStructMatrixSEntries(mat)       ((mat) -> Sentries)
#define hypre_SStructMatrixUEntries(mat)       ((mat) -> Uentries)
#define hypre_SStructMatrixTmpColCoords(mat)   ((mat) -> tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffs(mat)      ((mat) -> tmp_coeffs)
#define hypre_SStructMatrixNSSymmetric(mat)    ((mat) -> ns_symmetric)
#define hypre_SStructMatrixComplex(mat)        ((mat) -> complex)
#define hypre_SStructMatrixGlobalSize(mat)     ((mat) -> global_size)
#define hypre_SStructMatrixRefCount(mat)       ((mat) -> ref_count)
#define hypre_SStructMatrixObjectType(mat)       ((mat) -> object_type)

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
#define hypre_SStructPMatrixSymmetric(pmat)         ((pmat) -> symmetric)
#define hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define hypre_SStructPMatrixAccumulated(pmat)       ((pmat) -> accumulated)
#define hypre_SStructPMatrixComplex(pmat)           ((pmat) -> complex)
#define hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




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

   HYPRE_Int               nvars;
   hypre_StructVector    **svectors;     /* nvar array of svectors */
   hypre_CommPkg         **comm_pkgs;    /* nvar array of comm pkgs */

   HYPRE_Int               accumulated;  /* AddTo values accumulated? */
   HYPRE_Int               complex;      /* Is the vector complex */

   HYPRE_Int               ref_count;

   HYPRE_Int              *dataindices;  /* GEC1002 array for starting index of the 
                                            svector. pdataindices[varx] */
   HYPRE_Int               datasize;     /* Size of the pvector = sums size of svectors */

} hypre_SStructPVector;

typedef struct hypre_SStructVector_struct
{
   MPI_Comm                comm;
   HYPRE_Int               ndim;
   hypre_SStructGrid      *grid;
   HYPRE_Int               object_type;

   /* s-vector info */
   HYPRE_Int               nparts;
   hypre_SStructPVector  **pvectors;
   hypre_CommPkg        ***comm_pkgs;    /* nvar array of comm pkgs */

   /* u-vector info */
   HYPRE_IJVector          ijvector;
   hypre_ParVector        *parvector;

   /* inter-part communication info */
   HYPRE_Int               nbor_ncomms;  /* num comm_pkgs with neighbor parts */

  /* GEC10020902 pointer to big chunk of memory and auxiliary information   */

   double                  *data;        /* GEC1002 pointer to chunk data  */
   HYPRE_Int               *dataindices; /* GEC1002 dataindices[partx] is the starting index
                                          of vector data for the part=partx    */
   HYPRE_Int               datasize    ;  /* GEC1002 size of all data = ghlocalsize */

   HYPRE_Int               complex;      /* Is the vector complex */
   HYPRE_Int               global_size;  /* Total number coefficients */

   HYPRE_Int               ref_count;

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
#define hypre_SStructVectorNborNComms(vec)     ((vec) -> nbor_ncomms)
#define hypre_SStructVectorComplex(vec)        ((vec) -> complex)
#define hypre_SStructVectorGlobalSize(vec)     ((vec) -> global_size)
#define hypre_SStructVectorRefCount(vec)       ((vec) -> ref_count)
#define hypre_SStructVectorData(vec)           ((vec) -> data )
#define hypre_SStructVectorDataIndices(vec)    ((vec) -> dataindices)
#define hypre_SStructVectorDataSize(vec)       ((vec) -> datasize)


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
#define hypre_SStructPVectorAccumulated(pvec) ((pvec) -> accumulated)
#define hypre_SStructPVectorComplex(pvec)     ((pvec) -> complex)
#define hypre_SStructPVectorRefCount(pvec)    ((pvec) -> ref_count)
#define hypre_SStructPVectorDataIndices(pvec) ((pvec) -> dataindices  )
#define hypre_SStructPVectorDataSize(pvec)    ((pvec) -> datasize  )

#endif

/* HYPRE_sstruct_graph.c */
HYPRE_Int HYPRE_SStructGraphCreate ( MPI_Comm comm , HYPRE_SStructGrid grid , HYPRE_SStructGraph *graph_ptr );
HYPRE_Int HYPRE_SStructGraphDestroy ( HYPRE_SStructGraph graph );
HYPRE_Int HYPRE_SStructGraphSetDomainGrid ( HYPRE_SStructGraph graph , HYPRE_SStructGrid domain_grid );
HYPRE_Int HYPRE_SStructGraphSetStencil ( HYPRE_SStructGraph graph , HYPRE_Int part , HYPRE_Int var , HYPRE_SStructStencil stencil );
HYPRE_Int HYPRE_SStructGraphSetFEM ( HYPRE_SStructGraph graph , HYPRE_Int part );
HYPRE_Int HYPRE_SStructGraphSetFEMSparsity ( HYPRE_SStructGraph graph , HYPRE_Int part , HYPRE_Int nsparse , HYPRE_Int *sparsity );
HYPRE_Int HYPRE_SStructGraphAddEntries ( HYPRE_SStructGraph graph , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_Int to_part , HYPRE_Int *to_index , HYPRE_Int to_var );
HYPRE_Int HYPRE_SStructGraphAssemble ( HYPRE_SStructGraph graph );
HYPRE_Int HYPRE_SStructGraphSetObjectType ( HYPRE_SStructGraph graph , HYPRE_Int type );

/* HYPRE_sstruct_grid.c */
HYPRE_Int HYPRE_SStructGridCreate ( MPI_Comm comm , HYPRE_Int ndim , HYPRE_Int nparts , HYPRE_SStructGrid *grid_ptr );
HYPRE_Int HYPRE_SStructGridDestroy ( HYPRE_SStructGrid grid );
HYPRE_Int HYPRE_SStructGridSetExtents ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper );
HYPRE_Int HYPRE_SStructGridSetVariables ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int nvars , HYPRE_SStructVariable *vartypes );
HYPRE_Int HYPRE_SStructGridSetVariable ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int var , HYPRE_Int nvars , HYPRE_SStructVariable vartype );
HYPRE_Int HYPRE_SStructGridAddVariables ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int nvars , HYPRE_SStructVariable *vartypes );
HYPRE_Int HYPRE_SStructGridAddVariable ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_SStructVariable vartype );
HYPRE_Int HYPRE_SStructGridSetFEMOrdering ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *ordering );
HYPRE_Int HYPRE_SStructGridSetNeighborPart ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int nbor_part , HYPRE_Int *nbor_ilower , HYPRE_Int *nbor_iupper , HYPRE_Int *index_map , HYPRE_Int *index_dir );
HYPRE_Int HYPRE_SStructGridSetSharedPart ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int *offset , HYPRE_Int shared_part , HYPRE_Int *shared_ilower , HYPRE_Int *shared_iupper , HYPRE_Int *shared_offset , HYPRE_Int *index_map , HYPRE_Int *index_dir );
HYPRE_Int HYPRE_SStructGridAddUnstructuredPart ( HYPRE_SStructGrid grid , HYPRE_Int ilower , HYPRE_Int iupper );
HYPRE_Int HYPRE_SStructGridAssemble ( HYPRE_SStructGrid grid );
HYPRE_Int HYPRE_SStructGridSetPeriodic ( HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int *periodic );
HYPRE_Int HYPRE_SStructGridSetNumGhost ( HYPRE_SStructGrid grid , HYPRE_Int *num_ghost );

/* HYPRE_sstruct_matrix.c */
HYPRE_Int HYPRE_SStructMatrixCreate ( MPI_Comm comm , HYPRE_SStructGraph graph , HYPRE_SStructMatrix *matrix_ptr );
HYPRE_Int HYPRE_SStructMatrixDestroy ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixInitialize ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixSetValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixAddToValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixAddFEMValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , double *values );
HYPRE_Int HYPRE_SStructMatrixGetValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixGetFEMValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , double *values );
HYPRE_Int HYPRE_SStructMatrixSetBoxValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixAddToBoxValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixGetBoxValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values );
HYPRE_Int HYPRE_SStructMatrixAssemble ( HYPRE_SStructMatrix matrix );
HYPRE_Int HYPRE_SStructMatrixSetSymmetric ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int var , HYPRE_Int to_var , HYPRE_Int symmetric );
HYPRE_Int HYPRE_SStructMatrixSetNSSymmetric ( HYPRE_SStructMatrix matrix , HYPRE_Int symmetric );
HYPRE_Int HYPRE_SStructMatrixSetObjectType ( HYPRE_SStructMatrix matrix , HYPRE_Int type );
HYPRE_Int HYPRE_SStructMatrixGetObject ( HYPRE_SStructMatrix matrix , void **object );
HYPRE_Int HYPRE_SStructMatrixPrint ( const char *filename , HYPRE_SStructMatrix matrix , HYPRE_Int all );
HYPRE_Int HYPRE_SStructMatrixMatvec ( double alpha , HYPRE_SStructMatrix A , HYPRE_SStructVector x , double beta , HYPRE_SStructVector y );

/* HYPRE_sstruct_stencil.c */
HYPRE_Int HYPRE_SStructStencilCreate ( HYPRE_Int ndim , HYPRE_Int size , HYPRE_SStructStencil *stencil_ptr );
HYPRE_Int HYPRE_SStructStencilDestroy ( HYPRE_SStructStencil stencil );
HYPRE_Int HYPRE_SStructStencilSetEntry ( HYPRE_SStructStencil stencil , HYPRE_Int entry , HYPRE_Int *offset , HYPRE_Int var );

/* HYPRE_sstruct_vector.c */
HYPRE_Int HYPRE_SStructVectorCreate ( MPI_Comm comm , HYPRE_SStructGrid grid , HYPRE_SStructVector *vector_ptr );
HYPRE_Int HYPRE_SStructVectorDestroy ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorInitialize ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorSetValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , double *value );
HYPRE_Int HYPRE_SStructVectorAddToValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , double *value );
HYPRE_Int HYPRE_SStructVectorAddFEMValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *index , double *values );
HYPRE_Int HYPRE_SStructVectorGetValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , double *value );
HYPRE_Int HYPRE_SStructVectorGetFEMValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *index , double *values );
HYPRE_Int HYPRE_SStructVectorSetBoxValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , double *values );
HYPRE_Int HYPRE_SStructVectorAddToBoxValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , double *values );
HYPRE_Int HYPRE_SStructVectorGetBoxValues ( HYPRE_SStructVector vector , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , double *values );
HYPRE_Int HYPRE_SStructVectorAssemble ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorGather ( HYPRE_SStructVector vector );
HYPRE_Int HYPRE_SStructVectorSetConstantValues ( HYPRE_SStructVector vector , double value );
HYPRE_Int HYPRE_SStructVectorSetObjectType ( HYPRE_SStructVector vector , HYPRE_Int type );
HYPRE_Int HYPRE_SStructVectorGetObject ( HYPRE_SStructVector vector , void **object );
HYPRE_Int HYPRE_SStructVectorPrint ( const char *filename , HYPRE_SStructVector vector , HYPRE_Int all );
HYPRE_Int HYPRE_SStructVectorCopy ( HYPRE_SStructVector x , HYPRE_SStructVector y );
HYPRE_Int HYPRE_SStructVectorScale ( double alpha , HYPRE_SStructVector y );
HYPRE_Int HYPRE_SStructInnerProd ( HYPRE_SStructVector x , HYPRE_SStructVector y , double *result );
HYPRE_Int HYPRE_SStructAxpy ( double alpha , HYPRE_SStructVector x , HYPRE_SStructVector y );

/* sstruct_axpy.c */
HYPRE_Int hypre_SStructPAxpy ( double alpha , hypre_SStructPVector *px , hypre_SStructPVector *py );
HYPRE_Int hypre_SStructAxpy ( double alpha , hypre_SStructVector *x , hypre_SStructVector *y );

/* sstruct_copy.c */
HYPRE_Int hypre_SStructPCopy ( hypre_SStructPVector *px , hypre_SStructPVector *py );
HYPRE_Int hypre_SStructPartialPCopy ( hypre_SStructPVector *px , hypre_SStructPVector *py , hypre_BoxArrayArray **array_boxes );
HYPRE_Int hypre_SStructCopy ( hypre_SStructVector *x , hypre_SStructVector *y );

/* sstruct_graph.c */
HYPRE_Int hypre_SStructGraphRef ( hypre_SStructGraph *graph , hypre_SStructGraph **graph_ref );
HYPRE_Int hypre_SStructGraphFindUVEntry ( hypre_SStructGraph *graph , HYPRE_Int part , hypre_Index index , HYPRE_Int var , hypre_SStructUVEntry **Uventry_ptr );
HYPRE_Int hypre_SStructGraphFindBoxEndpt ( hypre_SStructGraph *graph , HYPRE_Int part , HYPRE_Int var , HYPRE_Int proc , HYPRE_Int endpt , HYPRE_Int boxi );
HYPRE_Int hypre_SStructGraphFindSGridEndpts ( hypre_SStructGraph *graph , HYPRE_Int part , HYPRE_Int var , HYPRE_Int proc , HYPRE_Int endpt , HYPRE_Int *endpts );

/* sstruct_grid.c */
HYPRE_Int hypre_SStructVariableGetOffset ( HYPRE_SStructVariable vartype , HYPRE_Int ndim , hypre_Index varoffset );
HYPRE_Int hypre_SStructPGridCreate ( MPI_Comm comm , HYPRE_Int ndim , hypre_SStructPGrid **pgrid_ptr );
HYPRE_Int hypre_SStructPGridDestroy ( hypre_SStructPGrid *pgrid );
HYPRE_Int hypre_SStructPGridSetExtents ( hypre_SStructPGrid *pgrid , hypre_Index ilower , hypre_Index iupper );
HYPRE_Int hypre_SStructPGridSetCellSGrid ( hypre_SStructPGrid *pgrid , hypre_StructGrid *cell_sgrid );
HYPRE_Int hypre_SStructPGridSetVariables ( hypre_SStructPGrid *pgrid , HYPRE_Int nvars , HYPRE_SStructVariable *vartypes );
HYPRE_Int hypre_SStructPGridSetVariable ( hypre_SStructPGrid *pgrid , HYPRE_Int var , HYPRE_Int nvars , HYPRE_SStructVariable vartype );
HYPRE_Int hypre_SStructPGridSetPNeighbor ( hypre_SStructPGrid *pgrid , hypre_Box *pneighbor_box , hypre_Index pnbor_offset );
HYPRE_Int hypre_SStructPGridAssemble ( hypre_SStructPGrid *pgrid );
HYPRE_Int hypre_SStructGridRef ( hypre_SStructGrid *grid , hypre_SStructGrid **grid_ref );
HYPRE_Int hypre_SStructGridAssembleBoxManagers ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridAssembleNborBoxManagers ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridCreateCommInfo ( hypre_SStructGrid *grid );
HYPRE_Int hypre_SStructGridFindBoxManEntry ( hypre_SStructGrid *grid , HYPRE_Int part , hypre_Index index , HYPRE_Int var , hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructGridFindNborBoxManEntry ( hypre_SStructGrid *grid , HYPRE_Int part , hypre_Index index , HYPRE_Int var , hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructGridBoxProcFindBoxManEntry ( hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int var , HYPRE_Int box , HYPRE_Int proc , hypre_BoxManEntry **entry_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetCSRstrides ( hypre_BoxManEntry *entry , hypre_Index strides );
HYPRE_Int hypre_SStructBoxManEntryGetGhstrides ( hypre_BoxManEntry *entry , hypre_Index strides );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalCSRank ( hypre_BoxManEntry *entry , hypre_Index index , HYPRE_Int *rank_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalGhrank ( hypre_BoxManEntry *entry , hypre_Index index , HYPRE_Int *rank_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetProcess ( hypre_BoxManEntry *entry , HYPRE_Int *proc_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetBoxnum ( hypre_BoxManEntry *entry , HYPRE_Int *id_ptr );
HYPRE_Int hypre_SStructBoxManEntryGetPart ( hypre_BoxManEntry *entry , HYPRE_Int part , HYPRE_Int *part_ptr );
HYPRE_Int hypre_SStructBoxToNborBox ( hypre_Box *box , hypre_Index index , hypre_Index nbor_index , hypre_Index coord , hypre_Index dir );
HYPRE_Int hypre_SStructNborBoxToBox ( hypre_Box *nbor_box , hypre_Index index , hypre_Index nbor_index , hypre_Index coord , hypre_Index dir );
HYPRE_Int hypre_SStructVarToNborVar ( hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int var , HYPRE_Int *coord , HYPRE_Int *nbor_var_ptr );
HYPRE_Int hypre_SStructGridSetNumGhost ( hypre_SStructGrid *grid , HYPRE_Int *num_ghost );
HYPRE_Int hypre_SStructBoxManEntryGetGlobalRank ( hypre_BoxManEntry *entry , hypre_Index index , HYPRE_Int *rank_ptr , HYPRE_Int type );
HYPRE_Int hypre_SStructBoxManEntryGetStrides ( hypre_BoxManEntry *entry , hypre_Index strides , HYPRE_Int type );
HYPRE_Int hypre_SStructBoxNumMap ( hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int boxnum , HYPRE_Int **num_varboxes_ptr , HYPRE_Int ***map_ptr );
HYPRE_Int hypre_SStructCellGridBoxNumMap ( hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int ***num_varboxes_ptr , HYPRE_Int ****map_ptr );
HYPRE_Int hypre_SStructCellBoxToVarBox ( hypre_Box *box , hypre_Index offset , hypre_Index varoffset , HYPRE_Int *valid );
HYPRE_Int hypre_SStructGridIntersect ( hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int var , hypre_Box *box , HYPRE_Int action , hypre_BoxManEntry ***entries_ptr , HYPRE_Int *nentries_ptr );

/* sstruct_innerprod.c */
HYPRE_Int hypre_SStructPInnerProd ( hypre_SStructPVector *px , hypre_SStructPVector *py , double *presult_ptr );
HYPRE_Int hypre_SStructInnerProd ( hypre_SStructVector *x , hypre_SStructVector *y , double *result_ptr );

/* sstruct_matrix.c */
HYPRE_Int hypre_SStructPMatrixRef ( hypre_SStructPMatrix *matrix , hypre_SStructPMatrix **matrix_ref );
HYPRE_Int hypre_SStructPMatrixCreate ( MPI_Comm comm , hypre_SStructPGrid *pgrid , hypre_SStructStencil **stencils , hypre_SStructPMatrix **pmatrix_ptr );
HYPRE_Int hypre_SStructPMatrixDestroy ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixInitialize ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixSetValues ( hypre_SStructPMatrix *pmatrix , hypre_Index index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructPMatrixSetBoxValues ( hypre_SStructPMatrix *pmatrix , hypre_Index ilower , hypre_Index iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructPMatrixAccumulate ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixAssemble ( hypre_SStructPMatrix *pmatrix );
HYPRE_Int hypre_SStructPMatrixSetSymmetric ( hypre_SStructPMatrix *pmatrix , HYPRE_Int var , HYPRE_Int to_var , HYPRE_Int symmetric );
HYPRE_Int hypre_SStructPMatrixPrint ( const char *filename , hypre_SStructPMatrix *pmatrix , HYPRE_Int all );
HYPRE_Int hypre_SStructUMatrixInitialize ( hypre_SStructMatrix *matrix );
HYPRE_Int hypre_SStructUMatrixSetValues ( hypre_SStructMatrix *matrix , HYPRE_Int part , hypre_Index index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructUMatrixSetBoxValues ( hypre_SStructMatrix *matrix , HYPRE_Int part , hypre_Index ilower , hypre_Index iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructUMatrixAssemble ( hypre_SStructMatrix *matrix );
HYPRE_Int hypre_SStructMatrixRef ( hypre_SStructMatrix *matrix , hypre_SStructMatrix **matrix_ref );
HYPRE_Int hypre_SStructMatrixSplitEntries ( hypre_SStructMatrix *matrix , HYPRE_Int part , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , HYPRE_Int *nSentries_ptr , HYPRE_Int **Sentries_ptr , HYPRE_Int *nUentries_ptr , HYPRE_Int **Uentries_ptr );
HYPRE_Int hypre_SStructMatrixSetValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *index , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructMatrixSetBoxValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , HYPRE_Int *ilower , HYPRE_Int *iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructMatrixSetInterPartValues ( HYPRE_SStructMatrix matrix , HYPRE_Int part , hypre_Index ilower , hypre_Index iupper , HYPRE_Int var , HYPRE_Int nentries , HYPRE_Int *entries , double *values , HYPRE_Int action );

/* sstruct_matvec.c */
HYPRE_Int hypre_SStructPMatvecCreate ( void **pmatvec_vdata_ptr );
HYPRE_Int hypre_SStructPMatvecSetup ( void *pmatvec_vdata , hypre_SStructPMatrix *pA , hypre_SStructPVector *px );
HYPRE_Int hypre_SStructPMatvecCompute ( void *pmatvec_vdata , double alpha , hypre_SStructPMatrix *pA , hypre_SStructPVector *px , double beta , hypre_SStructPVector *py );
HYPRE_Int hypre_SStructPMatvecDestroy ( void *pmatvec_vdata );
HYPRE_Int hypre_SStructPMatvec ( double alpha , hypre_SStructPMatrix *pA , hypre_SStructPVector *px , double beta , hypre_SStructPVector *py );
HYPRE_Int hypre_SStructMatvecCreate ( void **matvec_vdata_ptr );
HYPRE_Int hypre_SStructMatvecSetup ( void *matvec_vdata , hypre_SStructMatrix *A , hypre_SStructVector *x );
HYPRE_Int hypre_SStructMatvecCompute ( void *matvec_vdata , double alpha , hypre_SStructMatrix *A , hypre_SStructVector *x , double beta , hypre_SStructVector *y );
HYPRE_Int hypre_SStructMatvecDestroy ( void *matvec_vdata );
HYPRE_Int hypre_SStructMatvec ( double alpha , hypre_SStructMatrix *A , hypre_SStructVector *x , double beta , hypre_SStructVector *y );

/* sstruct_overlap_innerprod.c */
HYPRE_Int hypre_SStructPOverlapInnerProd ( hypre_SStructPVector *px , hypre_SStructPVector *py , double *presult_ptr );
HYPRE_Int hypre_SStructOverlapInnerProd ( hypre_SStructVector *x , hypre_SStructVector *y , double *result_ptr );

/* sstruct_scale.c */
HYPRE_Int hypre_SStructPScale ( double alpha , hypre_SStructPVector *py );
HYPRE_Int hypre_SStructScale ( double alpha , hypre_SStructVector *y );

/* sstruct_stencil.c */
HYPRE_Int hypre_SStructStencilRef ( hypre_SStructStencil *stencil , hypre_SStructStencil **stencil_ref );

/* sstruct_vector.c */
HYPRE_Int hypre_SStructPVectorRef ( hypre_SStructPVector *vector , hypre_SStructPVector **vector_ref );
HYPRE_Int hypre_SStructPVectorCreate ( MPI_Comm comm , hypre_SStructPGrid *pgrid , hypre_SStructPVector **pvector_ptr );
HYPRE_Int hypre_SStructPVectorDestroy ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorInitialize ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorSetValues ( hypre_SStructPVector *pvector , hypre_Index index , HYPRE_Int var , double *value , HYPRE_Int action );
HYPRE_Int hypre_SStructPVectorSetBoxValues ( hypre_SStructPVector *pvector , hypre_Index ilower , hypre_Index iupper , HYPRE_Int var , double *values , HYPRE_Int action );
HYPRE_Int hypre_SStructPVectorAccumulate ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorAssemble ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorGather ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructPVectorGetValues ( hypre_SStructPVector *pvector , hypre_Index index , HYPRE_Int var , double *value );
HYPRE_Int hypre_SStructPVectorGetBoxValues ( hypre_SStructPVector *pvector , hypre_Index ilower , hypre_Index iupper , HYPRE_Int var , double *values );
HYPRE_Int hypre_SStructPVectorSetConstantValues ( hypre_SStructPVector *pvector , double value );
HYPRE_Int hypre_SStructPVectorPrint ( const char *filename , hypre_SStructPVector *pvector , HYPRE_Int all );
HYPRE_Int hypre_SStructVectorRef ( hypre_SStructVector *vector , hypre_SStructVector **vector_ref );
HYPRE_Int hypre_SStructVectorSetConstantValues ( hypre_SStructVector *vector , double value );
HYPRE_Int hypre_SStructVectorConvert ( hypre_SStructVector *vector , hypre_ParVector **parvector_ptr );
HYPRE_Int hypre_SStructVectorParConvert ( hypre_SStructVector *vector , hypre_ParVector **parvector_ptr );
HYPRE_Int hypre_SStructVectorRestore ( hypre_SStructVector *vector , hypre_ParVector *parvector );
HYPRE_Int hypre_SStructVectorParRestore ( hypre_SStructVector *vector , hypre_ParVector *parvector );
HYPRE_Int hypre_SStructPVectorInitializeShell ( hypre_SStructPVector *pvector );
HYPRE_Int hypre_SStructVectorInitializeShell ( hypre_SStructVector *vector );
HYPRE_Int hypre_SStructVectorClearGhostValues ( hypre_SStructVector *vector );

#ifdef __cplusplus
}
#endif

#endif

