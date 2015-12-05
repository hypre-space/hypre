/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
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

