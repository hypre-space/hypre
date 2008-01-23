/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
   MPI_Comm                comm;             /* TODO: use different comms */
   int                     ndim;
   int                     nvars;            /* number of variables */
   HYPRE_SStructVariable  *vartypes;         /* types of variables */
   hypre_StructGrid       *sgrids[8];        /* struct grids for each vartype */
   hypre_BoxArray         *iboxarrays[8];    /* interface boxes */
                                       
   hypre_BoxArray         *pneighbors;

   int                     local_size;       /* Number of variables locally */
   int                     global_size;      /* Total number of variables */

   hypre_Index             periodic;         /* Indicates if pgrid is periodic */

  /* GEC0902 additions for ghost expansion of boxes */

   int                     ghlocal_size;     /* Number of vars including ghosts */
                           
   int                     cell_sgrid_done;  /* =1 implies cell grid already assembled */
} hypre_SStructPGrid;

typedef struct
{
   hypre_Box    box;
   int          part;
   hypre_Index  ilower; /* box ilower, but on the neighbor index-space */
   hypre_Index  coord;  /* lives on local index-space */
   hypre_Index  dir;    /* lives on local index-space */

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
   int  boxnum;
   int  ghoffset; /* GEC0902 ghost offset   */

} hypre_SStructMapInfo;

typedef struct
{
   int          type;
   int          proc;
   int          offset;   /* minimum offset for this box */
   int          boxnum;
   int          ghoffset; /* GEC0902 minimum offset ghost for this box */
   int          part;     /* part the box lives on */
   hypre_Index  ilower;   /* box ilower, but on the neighbor index-space */
   hypre_Index  coord;    /* lives on local index-space */
   hypre_Index  dir;      /* lives on local index-space */
   hypre_Index  stride;   /* lives on local index-space */
   hypre_Index  ghstride; /* GEC1002 the ghost equivalent of strides */ 

} hypre_SStructNMapInfo;

typedef struct
{
   hypre_CommInfo  *comm_info;
   int              send_part;
   int              recv_part;
   int              send_var;
   int              recv_var;
   
} hypre_SStructCommInfo;

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
   hypre_SStructCommInfo    **vnbor_comm_info; /* for updating shared data */
   int                        vnbor_ncomms;

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

 /* GEC0902 additions for ghost expansion of boxes */

   int                     ghlocal_size;  /* GEC0902 Number of vars including ghosts */
   int                     ghstart_rank;  /* GEC0902 start rank including ghosts  */

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
#define hypre_SStructGridVNborCommInfo(grid)  ((grid) -> vnbor_comm_info)
#define hypre_SStructGridVNborNComms(grid)    ((grid) -> vnbor_ncomms)
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
#define hypre_SStructPGridLocalSize(pgrid)        ((pgrid) -> local_size)
#define hypre_SStructPGridGlobalSize(pgrid)       ((pgrid) -> global_size)
#define hypre_SStructPGridPeriodic(pgrid)         ((pgrid) -> periodic)
#define hypre_SStructPGridGhlocalSize(pgrid)      ((pgrid) -> ghlocal_size)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMapInfo
 *--------------------------------------------------------------------------*/

#define hypre_SStructMapInfoType(info)            ((info) -> type)
#define hypre_SStructMapInfoProc(info)            ((info) -> proc)
#define hypre_SStructMapInfoOffset(info)          ((info) -> offset)
#define hypre_SStructMapInfoBoxnum(info)          ((info) -> boxnum)
#define hypre_SStructMapInfoGhoffset(info)        ((info) -> ghoffset)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructNMapInfo
 *--------------------------------------------------------------------------*/

/* Use the MapInfo macros to access the first five structure components */

#define hypre_SStructNMapInfoPart(info)    ((info) -> part)
#define hypre_SStructNMapInfoILower(info)  ((info) -> ilower)
#define hypre_SStructNMapInfoCoord(info)   ((info) -> coord)
#define hypre_SStructNMapInfoDir(info)     ((info) -> dir)
#define hypre_SStructNMapInfoStride(info)  ((info) -> stride)
#define hypre_SStructNMapInfoGhstride(info)  ((info) -> ghstride)

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

