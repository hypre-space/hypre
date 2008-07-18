/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Member functions for hypre_SStructGrid class.
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================
 * SStructVariable routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructVariableGetOffset
 *--------------------------------------------------------------------------*/

int
hypre_SStructVariableGetOffset( HYPRE_SStructVariable  vartype,
                                int                    ndim,
                                hypre_Index            varoffset )
{
   int d;

   switch(vartype)
   {
      case HYPRE_SSTRUCT_VARIABLE_CELL:
         hypre_SetIndex(varoffset, 0, 0, 0);
         break;
      case HYPRE_SSTRUCT_VARIABLE_NODE:
         hypre_SetIndex(varoffset, 1, 1, 1);
         break;
      case HYPRE_SSTRUCT_VARIABLE_XFACE:
         hypre_SetIndex(varoffset, 1, 0, 0);
         break;
      case HYPRE_SSTRUCT_VARIABLE_YFACE:
         hypre_SetIndex(varoffset, 0, 1, 0);
         break;
      case HYPRE_SSTRUCT_VARIABLE_ZFACE:
         hypre_SetIndex(varoffset, 0, 0, 1);
         break;
      case HYPRE_SSTRUCT_VARIABLE_XEDGE:
         hypre_SetIndex(varoffset, 0, 1, 1);
         break;
      case HYPRE_SSTRUCT_VARIABLE_YEDGE:
         hypre_SetIndex(varoffset, 1, 0, 1);
         break;
      case HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         hypre_SetIndex(varoffset, 1, 1, 0);
         break;
      case HYPRE_SSTRUCT_VARIABLE_UNDEFINED:
         break;
   }
   for (d = ndim; d < 3; d++)
   {
      hypre_IndexD(varoffset, d) = 0;
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructPGrid routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPGridCreate
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridCreate( MPI_Comm             comm,
                          int                  ndim,
                          hypre_SStructPGrid **pgrid_ptr )
{
   hypre_SStructPGrid  *pgrid;
   hypre_StructGrid    *sgrid;
   int                  t;

   pgrid = hypre_TAlloc(hypre_SStructPGrid, 1);

   hypre_SStructPGridComm(pgrid)             = comm;
   hypre_SStructPGridNDim(pgrid)             = ndim;
   hypre_SStructPGridNVars(pgrid)            = 0;
   hypre_SStructPGridCellSGridDone(pgrid)    = 0;
   hypre_SStructPGridVarTypes(pgrid)         = NULL;
   
   for (t = 0; t < 8; t++)
   {
      hypre_SStructPGridVTSGrid(pgrid, t)     = NULL;
      hypre_SStructPGridVTIBoxArray(pgrid, t) = NULL;
   }
   HYPRE_StructGridCreate(comm, ndim, &sgrid);
   hypre_SStructPGridCellSGrid(pgrid) = sgrid;
   
   hypre_SStructPGridPNeighbors(pgrid) = hypre_BoxArrayCreate(0);

   hypre_SStructPGridLocalSize(pgrid)  = 0;
   hypre_SStructPGridGlobalSize(pgrid) = 0;

   /* GEC0902 ghost addition to the grid    */
   hypre_SStructPGridGhlocalSize(pgrid)   = 0;
   
   hypre_ClearIndex(hypre_SStructPGridPeriodic(pgrid));

   *pgrid_ptr = pgrid;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridDestroy( hypre_SStructPGrid *pgrid )
{
   hypre_StructGrid **sgrids;
   hypre_BoxArray   **iboxarrays;
   int                t;

   if (pgrid)
   {
      sgrids     = hypre_SStructPGridSGrids(pgrid);
      iboxarrays = hypre_SStructPGridIBoxArrays(pgrid);
      hypre_TFree(hypre_SStructPGridVarTypes(pgrid));
      for (t = 0; t < 8; t++)
      {
         HYPRE_StructGridDestroy(sgrids[t]);
         hypre_BoxArrayDestroy(iboxarrays[t]);
      }
      hypre_BoxArrayDestroy(hypre_SStructPGridPNeighbors(pgrid));
      hypre_TFree(pgrid);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetExtents
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridSetExtents( hypre_SStructPGrid  *pgrid,
                              hypre_Index          ilower,
                              hypre_Index          iupper )
{
   hypre_StructGrid *sgrid = hypre_SStructPGridCellSGrid(pgrid);

   HYPRE_StructGridSetExtents(sgrid, ilower, iupper);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetCellSGrid
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridSetCellSGrid( hypre_SStructPGrid  *pgrid,
                                hypre_StructGrid    *cell_sgrid )
{
   hypre_SStructPGridCellSGrid(pgrid) = cell_sgrid;
   hypre_SStructPGridCellSGridDone(pgrid) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetVariables
 *--------------------------------------------------------------------------*/

int hypre_SStructPGridSetVariables( hypre_SStructPGrid    *pgrid,
                                    int                    nvars,
                                    HYPRE_SStructVariable *vartypes )
{
   hypre_SStructVariable  *new_vartypes;
   int                     i;

   hypre_TFree(hypre_SStructPGridVarTypes(pgrid));

   new_vartypes = hypre_TAlloc(hypre_SStructVariable, nvars);
   for (i = 0; i < nvars; i++)
   {
      new_vartypes[i] = vartypes[i];
   }

   hypre_SStructPGridNVars(pgrid)    = nvars;
   hypre_SStructPGridVarTypes(pgrid) = new_vartypes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetVariable
 * Like hypre_SStructPGridSetVariables, but do one variable at a time.
 * Nevertheless, you still must provide nvars, for memory allocation.
 *--------------------------------------------------------------------------*/

int hypre_SStructPGridSetVariable( hypre_SStructPGrid    *pgrid,
                                   int                    var,
                                   int                    nvars,
                                   HYPRE_SStructVariable  vartype )
{
   hypre_SStructVariable  *vartypes;

   if ( hypre_SStructPGridVarTypes(pgrid) == NULL )
   {
      vartypes = hypre_TAlloc(hypre_SStructVariable, nvars);
      hypre_SStructPGridNVars(pgrid)    = nvars;
      hypre_SStructPGridVarTypes(pgrid) = vartypes;
   }
   else
   {
      vartypes = hypre_SStructPGridVarTypes(pgrid);
   }

   vartypes[var] = vartype;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetPNeighbor
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridSetPNeighbor( hypre_SStructPGrid  *pgrid,
                                hypre_Box           *pneighbor_box )
{
   hypre_AppendBox(pneighbor_box, hypre_SStructPGridPNeighbors(pgrid));

   return hypre_error_flag;
}




/*--------------------------------------------------------------------------
   hypre_SStructPGridAssemble 

  11/06 AHB - modified to use the box manager
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridAssemble( hypre_SStructPGrid  *pgrid )
{
   MPI_Comm               comm       = hypre_SStructPGridComm(pgrid);
   int                    ndim       = hypre_SStructPGridNDim(pgrid);
   int                    nvars      = hypre_SStructPGridNVars(pgrid);
   HYPRE_SStructVariable *vartypes   = hypre_SStructPGridVarTypes(pgrid);
   hypre_StructGrid     **sgrids     = hypre_SStructPGridSGrids(pgrid);
   hypre_BoxArray       **iboxarrays = hypre_SStructPGridIBoxArrays(pgrid);
   hypre_BoxArray        *pneighbors = hypre_SStructPGridPNeighbors(pgrid);
   hypre_IndexRef         periodic   = hypre_SStructPGridPeriodic(pgrid);

   hypre_StructGrid      *cell_sgrid;
   hypre_IndexRef         cell_imax;
   hypre_StructGrid      *sgrid;
   hypre_BoxArray        *iboxarray;
   hypre_BoxManager      *boxman;
   hypre_BoxArray        *hood_boxes;
   int                    hood_first_local;
   int                    hood_num_local;
   hypre_BoxArray        *nbor_boxes;
   hypre_BoxArray        *diff_boxes;
   hypre_BoxArray        *tmp_boxes;
   hypre_BoxArray        *boxes;
   hypre_Box             *box;
   hypre_Index            varoffset;
   int                    pneighbors_size;
   int                    nbor_boxes_size;

   int                    t, var, i, j, d;

   /*-------------------------------------------------------------
    * set up the uniquely distributed sgrids for each vartype
    *-------------------------------------------------------------*/

   cell_sgrid = hypre_SStructPGridCellSGrid(pgrid);
   HYPRE_StructGridSetPeriodic(cell_sgrid, periodic);
   if (!hypre_SStructPGridCellSGridDone(pgrid))
      HYPRE_StructGridAssemble(cell_sgrid);

   /* this is used to truncate boxes when periodicity is on */
   cell_imax = hypre_BoxIMax(hypre_StructGridBoundingBox(cell_sgrid));

   /* get neighbor info from the struct grid box manager */
   boxman     = hypre_StructGridBoxMan(cell_sgrid);
   hood_boxes =  hypre_BoxArrayCreate(0);
   hypre_BoxManGetAllEntriesBoxes(boxman, hood_boxes);
   hood_first_local = hypre_BoxManFirstLocal(boxman);
   hood_num_local   = hypre_BoxManNumMyEntries(boxman);

   pneighbors_size = hypre_BoxArraySize(pneighbors);
   nbor_boxes_size = pneighbors_size + hood_first_local + hood_num_local;

   nbor_boxes = hypre_BoxArrayCreate(nbor_boxes_size);
   diff_boxes = hypre_BoxArrayCreate(0);
   tmp_boxes  = hypre_BoxArrayCreate(0);

   for (var = 0; var < nvars; var++)
   {
      t = vartypes[var];

      if ((t > 0) && (sgrids[t] == NULL))
      {
         HYPRE_StructGridCreate(comm, ndim, &sgrid);
         boxes = hypre_BoxArrayCreate(0);
         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffset);

         /* create nbor_boxes for this variable type */
         for (i = 0; i < pneighbors_size; i++)
         {
            hypre_CopyBox(hypre_BoxArrayBox(pneighbors, i),
                          hypre_BoxArrayBox(nbor_boxes, i));
         }
         for (i = 0; i < (hood_first_local + hood_num_local); i++)
         {
            hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, i),
                          hypre_BoxArrayBox(nbor_boxes, pneighbors_size + i));
         }
         for (i = 0; i < nbor_boxes_size; i++)
         {
            box = hypre_BoxArrayBox(nbor_boxes, i);
            hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
            hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
            hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
         }

         /* boxes = (local boxes - neighbors with smaller ID - pneighbors) */
         for (i = 0; i < hood_num_local; i++)
         {
            j = pneighbors_size + hood_first_local + i;
            hypre_BoxArraySetSize(diff_boxes, 1);
            hypre_CopyBox(hypre_BoxArrayBox(nbor_boxes, j),
                          hypre_BoxArrayBox(diff_boxes, 0));
            hypre_BoxArraySetSize(nbor_boxes, j);

            hypre_SubtractBoxArrays(diff_boxes, nbor_boxes, tmp_boxes);
            hypre_AppendBoxArray(diff_boxes, boxes);
         }

         /* truncate if necessary when periodic */
         for (d = 0; d < 3; d++)
         {
            if (hypre_IndexD(periodic, d) && hypre_IndexD(varoffset, d))
            {
               hypre_ForBoxI(i, boxes)
                  {
                     box = hypre_BoxArrayBox(boxes, i);
                     if (hypre_BoxIMaxD(box, d) == hypre_IndexD(cell_imax, d))
                     {
                        hypre_BoxIMaxD(box, d) --;
                     }
                  }
            }
         }
         HYPRE_StructGridSetPeriodic(sgrid, periodic);

         hypre_StructGridSetBoxes(sgrid, boxes);
         HYPRE_StructGridAssemble(sgrid);

         sgrids[t] = sgrid;
      }            
   }

   hypre_BoxArrayDestroy(hood_boxes);
   

   hypre_BoxArrayDestroy(nbor_boxes);
   hypre_BoxArrayDestroy(diff_boxes);
   hypre_BoxArrayDestroy(tmp_boxes);

   /*-------------------------------------------------------------
    * compute iboxarrays
    *-------------------------------------------------------------*/

   for (t = 0; t < 8; t++)
   {
      sgrid = sgrids[t];
      if (sgrid != NULL)
      {
         iboxarray = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(sgrid));

         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffset);
         hypre_ForBoxI(i, iboxarray)
            {
               /* grow the boxes */
               box = hypre_BoxArrayBox(iboxarray, i);
               hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
               hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
               hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
               hypre_BoxIMaxX(box) += hypre_IndexX(varoffset);
               hypre_BoxIMaxY(box) += hypre_IndexY(varoffset);
               hypre_BoxIMaxZ(box) += hypre_IndexZ(varoffset);
            }

         iboxarrays[t] = iboxarray;
      }
   }

   /*-------------------------------------------------------------
    * set up the size info
    * GEC0902 addition of the local ghost size for pgrid.At first pgridghlocalsize=0
    *-------------------------------------------------------------*/

   for (var = 0; var < nvars; var++)
   {
      sgrid = hypre_SStructPGridSGrid(pgrid, var);
      hypre_SStructPGridLocalSize(pgrid)  += hypre_StructGridLocalSize(sgrid);
      hypre_SStructPGridGlobalSize(pgrid) += hypre_StructGridGlobalSize(sgrid);
      hypre_SStructPGridGhlocalSize(pgrid) += hypre_StructGridGhlocalSize(sgrid);
      
   }

   return hypre_error_flag;
}


/*==========================================================================
 * SStructGrid routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructGridRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridRef( hypre_SStructGrid  *grid,
                      hypre_SStructGrid **grid_ref)
{
   hypre_SStructGridRefCount(grid) ++;
   *grid_ref = grid;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridAssembleMaps( hypre_SStructGrid *grid )
{
   MPI_Comm                   comm        = hypre_SStructGridComm(grid);
   int                        nparts      = hypre_SStructGridNParts(grid);
   int                        local_size  = hypre_SStructGridLocalSize(grid);
   hypre_BoxMap            ***maps;
   hypre_SStructMapInfo    ***info;
   hypre_SStructPGrid        *pgrid;
   int                        nvars;
   hypre_StructGrid          *sgrid;
   hypre_Box                 *bounding_box;

   int                       *offsets;
   hypre_SStructMapInfo      *entry_info;
   hypre_BoxArray            *boxes;
   hypre_Box                 *box;

   int                       *procs;
   int                       *local_boxnums;
   int                       *boxproc_offset;
   int                        first_local;

   int                        nprocs, myproc;
   int                        proc, part, var, b;

   /* GEC0902 variable for ghost calculation */
   hypre_Box                 *ghostbox;
   int                       * num_ghost;
   int                       *ghoffsets;
   int                        ghlocal_size  = hypre_SStructGridGhlocalSize(grid);


   /*------------------------------------------------------
    * Build map info for grid boxes
    *------------------------------------------------------*/

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myproc);

   offsets = hypre_TAlloc(int, nprocs + 1);
   offsets[0] = 0;
   MPI_Allgather(&local_size, 1, MPI_INT, &offsets[1], 1, MPI_INT, comm);

   /* GEC0902 calculate a ghost piece for each mapentry using the ghlocalsize of the grid.
    * Gather each ghlocalsize in each of the processors in comm */    

   ghoffsets = hypre_TAlloc(int, nprocs +1);
   ghoffsets[0] = 0;
   MPI_Allgather(&ghlocal_size, 1, MPI_INT, &ghoffsets[1], 1, MPI_INT, comm);


   for (proc = 1; proc < (nprocs + 1); proc++)
   {
          
      offsets[proc] += offsets[proc-1];
      ghoffsets[proc] += ghoffsets[proc-1];
            
   }

   hypre_SStructGridStartRank(grid) = offsets[myproc];

   hypre_SStructGridGhstartRank(grid) = ghoffsets[myproc];

   maps = hypre_TAlloc(hypre_BoxMap **, nparts);
   info = hypre_TAlloc(hypre_SStructMapInfo **, nparts);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      maps[part] = hypre_TAlloc(hypre_BoxMap *, nvars);
      info[part] = hypre_TAlloc(hypre_SStructMapInfo *, nvars);
      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);

         /* NOTE: With neighborhood info from user, don't need all gather */
         hypre_GatherAllBoxes(comm, hypre_StructGridBoxes(sgrid),
                              &boxes, &procs, &first_local);
         bounding_box = hypre_StructGridBoundingBox(sgrid);

         /* get the local box numbers for all the boxes*/
         hypre_ComputeBoxnums(boxes, procs, &local_boxnums);

         hypre_BoxMapCreate(hypre_BoxArraySize(boxes),
                            hypre_BoxIMin(bounding_box),
                            hypre_BoxIMax(bounding_box),
                            nprocs,
                           &maps[part][var]);

         info[part][var] = hypre_TAlloc(hypre_SStructMapInfo,
                                        hypre_BoxArraySize(boxes));

	 /* GEC0902 adding the ghost in the boxmap using a new function and sgrid info 
          * each sgrid has num_ghost, we just inject the ghost into the boxmap */

         num_ghost = hypre_StructGridNumGhost(sgrid);
         hypre_BoxMapSetNumGhost(maps[part][var], num_ghost);

	 ghostbox = hypre_BoxCreate();

         boxproc_offset= hypre_BoxMapBoxProcOffset(maps[part][var]);
         proc= -1;
         for (b = 0; b < hypre_BoxArraySize(boxes); b++)
         {
            box = hypre_BoxArrayBox(boxes, b);
            if (proc != procs[b])
            {
               proc= procs[b];
               boxproc_offset[proc]= b;  /* record the proc box offset */
            }
            
            entry_info = &info[part][var][b];
            hypre_SStructMapInfoType(entry_info) =
               hypre_SSTRUCT_MAP_INFO_DEFAULT;
            hypre_SStructMapInfoProc(entry_info)   = proc;
            hypre_SStructMapInfoOffset(entry_info) = offsets[proc];
            hypre_SStructMapInfoBoxnum(entry_info) = local_boxnums[b];

	    /* GEC0902 ghoffsets added to entry_info   */

	    hypre_SStructMapInfoGhoffset(entry_info) = ghoffsets[proc];
            
            hypre_BoxMapAddEntry(maps[part][var],
                                 hypre_BoxIMin(box),
                                 hypre_BoxIMax(box),
                                 entry_info);

            offsets[proc] += hypre_BoxVolume(box);

	    /* GEC0902 expanding box to compute expanded volume for ghost calculation */

	    /*            ghostbox = hypre_BoxCreate();  */
            hypre_CopyBox(box, ghostbox);
	    hypre_BoxExpand(ghostbox,num_ghost);         
       
	    ghoffsets[proc] += hypre_BoxVolume(ghostbox); 
            /* hypre_BoxDestroy(ghostbox);  */           
         }

         hypre_BoxDestroy(ghostbox);
         hypre_BoxArrayDestroy(boxes);
         hypre_TFree(procs);
         hypre_TFree(local_boxnums);

         hypre_BoxMapAssemble(maps[part][var], comm);
      }
   }
   hypre_TFree(offsets);
   hypre_TFree(ghoffsets);
   hypre_SStructGridMaps(grid) = maps;
   hypre_SStructGridInfo(grid) = info;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridAssembleNborMaps( hypre_SStructGrid *grid )
{
   MPI_Comm                   comm        = hypre_SStructGridComm(grid);
   int                        nparts      = hypre_SStructGridNParts(grid);
   int                      **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor   ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor     *vneighbor;
   hypre_BoxMap            ***maps        = hypre_SStructGridMaps(grid);
   hypre_SStructNMapInfo   ***ninfo;
   hypre_SStructPGrid        *pgrid;
   int                        nvars;

   hypre_SStructNMapInfo     *entry_ninfo;

   hypre_BoxMapEntry         *map_entry;
   hypre_Box                 *nbor_box;
   hypre_Box                 *box;
   int                        nbor_part, nbor_var;
   int                        nbor_boxnum;
   hypre_Index                nbor_ilower;
   int                        nbor_offset;
   int                        nbor_proc;
   hypre_Index                c;
   int                       *d, *stride;

   int                        part, var, b;

   /*  GEC1002 additional ghost box    */

   hypre_Box                  *ghostbox ;
   int                        nbor_ghoffset;
   int                        *ghstride;
   int                        *num_ghost;

   /*------------------------------------------------------
    * Add neighbor boxes to maps and re-assemble
    *------------------------------------------------------*/

   box = hypre_BoxCreate();

   /* GEC1002 creating a ghostbox for strides calculation */

   ghostbox = hypre_BoxCreate();

   ninfo = hypre_TAlloc(hypre_SStructNMapInfo **, nparts);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      ninfo[part] = hypre_TAlloc(hypre_SStructNMapInfo *, nvars);

      for (var = 0; var < nvars; var++)
      {
         ninfo[part][var] = hypre_TAlloc(hypre_SStructNMapInfo,
                                         nvneighbors[part][var]);

         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];
            nbor_box = hypre_SStructNeighborBox(vneighbor);
            nbor_part = hypre_SStructNeighborPart(vneighbor);
            hypre_CopyIndex(hypre_SStructNeighborILower(vneighbor), nbor_ilower);
            hypre_SStructVarToNborVar(
               grid, part, var, hypre_SStructNeighborCoord(vneighbor), &nbor_var);
            /*
             * Note that this assumes that the entire neighbor box
             * actually lives on the global grid.  This was insured by
             * intersecting the neighbor boxes with the global grid.
             * We also assume that each neighbour box intersects
             * with only one box on the neighbouring processor. This
             * should be the case since we only have one map_entry.
             */
            hypre_SStructGridFindMapEntry(grid, nbor_part, nbor_ilower, nbor_var,
                                          &map_entry);

            hypre_BoxMapEntryGetExtents(map_entry,
                                        hypre_BoxIMin(box),
                                        hypre_BoxIMax(box));
            hypre_SStructMapEntryGetProcess(map_entry, &nbor_proc);
            hypre_SStructMapEntryGetBoxnum(map_entry, &nbor_boxnum);

            /* GEC1002 using the globalcsrank for inclusion in the nmapinfo  */
            hypre_SStructMapEntryGetGlobalCSRank(map_entry, nbor_ilower,
                                                 &nbor_offset);
            /* GEC1002 using the ghglobalrank for inclusion in the nmapinfo  */
            hypre_SStructMapEntryGetGlobalGhrank(map_entry, nbor_ilower,
                                                 &nbor_ghoffset);

            num_ghost = hypre_BoxMapEntryNumGhost(map_entry);

            entry_ninfo = &ninfo[part][var][b];
            hypre_SStructMapInfoType(entry_ninfo) =
               hypre_SSTRUCT_MAP_INFO_NEIGHBOR;
            hypre_SStructMapInfoProc(entry_ninfo)   = nbor_proc;
            hypre_SStructMapInfoBoxnum(entry_ninfo) = nbor_boxnum;
            hypre_SStructMapInfoOffset(entry_ninfo) = nbor_offset;
            /* GEC1002 inclusion of ghoffset for the ninfo  */
            hypre_SStructMapInfoGhoffset(entry_ninfo) = nbor_ghoffset;
           
            hypre_SStructNMapInfoPart(entry_ninfo)   = nbor_part;
            hypre_CopyIndex(nbor_ilower,
                            hypre_SStructNMapInfoILower(entry_ninfo));
            hypre_CopyIndex(hypre_SStructNeighborCoord(vneighbor),
                            hypre_SStructNMapInfoCoord(entry_ninfo));
            hypre_CopyIndex(hypre_SStructNeighborDir(vneighbor),
                            hypre_SStructNMapInfoDir(entry_ninfo));

            /*------------------------------------------------------
             * This computes strides in the local index-space,
             * so they may be negative.
             *------------------------------------------------------*/

            /* want `c' to map from neighbor index-space back */
            d = hypre_SStructNMapInfoCoord(entry_ninfo);
            c[d[0]] = 0;
            c[d[1]] = 1;
            c[d[2]] = 2;
            d = hypre_SStructNMapInfoDir(entry_ninfo);

            stride = hypre_SStructNMapInfoStride(entry_ninfo);
            stride[c[0]] = 1;
            stride[c[1]] = hypre_BoxSizeD(box, 0);
            stride[c[2]] = hypre_BoxSizeD(box, 1) * stride[c[1]];
            stride[c[0]] *= d[c[0]];
            stride[c[1]] *= d[c[1]];
            stride[c[2]] *= d[c[2]];

            /* GEC1002 expanding the ghostbox to compute the strides based on ghosts vector  */

            hypre_CopyBox(box, ghostbox);
            hypre_BoxExpand(ghostbox,num_ghost);
            ghstride = hypre_SStructNMapInfoGhstride(entry_ninfo);
            ghstride[c[0]] = 1;
            ghstride[c[1]] = hypre_BoxSizeD(ghostbox, 0);
            ghstride[c[2]] = hypre_BoxSizeD(ghostbox, 1) * ghstride[c[1]];
            ghstride[c[0]] *= d[c[0]];
            ghstride[c[1]] *= d[c[1]];
            ghstride[c[2]] *= d[c[2]];
        }
      }

      /* NOTE: It is important not to change the map in the above
       * loop because it is needed in the 'FindMapEntry' call */
      for (var = 0; var < nvars; var++)
      {
         hypre_BoxMapIncSize(maps[part][var], nvneighbors[part][var]);

         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];
            nbor_box = hypre_SStructNeighborBox(vneighbor);
            hypre_BoxMapAddEntry(maps[part][var],
                                 hypre_BoxIMin(nbor_box),
                                 hypre_BoxIMax(nbor_box),
                                 &ninfo[part][var][b]);
         }

         hypre_BoxMapAssemble(maps[part][var], comm);
      }
   }

   hypre_SStructGridNInfo(grid) = ninfo;

   hypre_BoxDestroy(box);

   hypre_BoxDestroy(ghostbox);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine computes the inter-part communication information for updating
 * shared variable data.
 *
 * It grows each local box according to vartype and intersects with the BoxMap
 * to get map entries.  Then, for each of the neighbor-type entries, it grows
 * either the local box or the neighbor box based on which one is the "owner"
 * (the part number determines this).
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridCreateCommInfo( hypre_SStructGrid  *grid )
{
   int                      ndim = hypre_SStructGridNDim(grid);
   int                      nparts = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids = hypre_SStructGridPGrids(grid);
   hypre_SStructCommInfo  **vnbor_comm_info;
   int                      vnbor_ncomms;
   hypre_SStructCommInfo   *comm_info;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;

   typedef struct
   {
      hypre_BoxArrayArray    *boxes;
      int                   **procs;
      int                   **rboxnums;
      hypre_BoxArrayArray    *rboxes;
      int                    *num_transforms; /* reference to num transforms */
      hypre_Index            *coords;
      hypre_Index            *dirs;
      int                   **transforms;

   } CInfo;

   hypre_IndexRef           coord, dir;

   CInfo                  **cinfo_a;  /* array of size (nparts^2)(maxvars^2) */
   CInfo                   *cinfo, *send_cinfo, *recv_cinfo;
   int                      cinfoi, cinfoj, maxvars;
   hypre_BoxArray          *cbox_a;
   int                    **cprocs;
   int                    **crboxnums;
   hypre_BoxArray          *crbox_a;
   int                     *cnum_transforms;
   hypre_Index             *ccoords;
   hypre_Index             *cdirs;
   int                    **ctransforms;

   hypre_SStructPGrid      *pgrid;
   hypre_StructGrid        *sgrid;
   hypre_BoxMap            *map;

   hypre_BoxMapEntry      **entries;
   hypre_BoxMapEntry       *entry;
   hypre_SStructNMapInfo   *entry_info;
   int                      nentries, info_type;

   hypre_Box               *box, *nbor_box, *int_box, *int_rbox;
   hypre_Index              imin0;

   int                      nvars, num_boxes, size;
   int                      pi, pj, vi, vj, bi, ei, ti;

   box = hypre_BoxCreate();
   nbor_box = hypre_BoxCreate();
   int_box = hypre_BoxCreate();
   int_rbox = hypre_BoxCreate();

   /* initialize cinfo_a array */
   maxvars = 0;
   for (pi = 0; pi < nparts; pi++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[pi]);
      if ( maxvars < nvars )
      {
         maxvars = nvars;
      }
   }
   cinfo_a = hypre_CTAlloc(CInfo *, nparts*nparts*maxvars*maxvars);

   /* loop over local boxes and compute send/recv CommInfo */

   vnbor_ncomms = 0;
   for (pi = 0; pi < nparts; pi++)
   {
      pgrid  = pgrids[pi];
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);

      for (vi = 0; vi < nvars; vi++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, vi);
         num_boxes = hypre_StructGridNumBoxes(sgrid);
         map = hypre_SStructGridMap(grid, pi, vi);
         hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);

         for (bi = 0; bi < num_boxes; bi++)
         {
            /* grow the local box and intersect with the BoxMap */
            hypre_CopyBox(hypre_StructGridBox(sgrid, bi), box);
            hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
            hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
            hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
            hypre_BoxIMaxX(box) += hypre_IndexX(varoffset);
            hypre_BoxIMaxY(box) += hypre_IndexY(varoffset);
            hypre_BoxIMaxZ(box) += hypre_IndexZ(varoffset);
            hypre_BoxMapIntersect(map, hypre_BoxIMin(box), hypre_BoxIMax(box),
                                  &entries, &nentries );

            for (ei = 0; ei < nentries; ei++)
            {
               entry = entries[ei];
               hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
               info_type = hypre_SStructMapInfoType(entry_info);

               /* if this is a neighbor box, compute send/recv boxes */
               if (info_type == hypre_SSTRUCT_MAP_INFO_NEIGHBOR)
               {
                  hypre_CopyBox(hypre_StructGridBox(sgrid, bi), box);
                  hypre_BoxMapEntryGetExtents(
                     entry, hypre_BoxIMin(nbor_box), hypre_BoxIMax(nbor_box));
                  hypre_CopyIndex(hypre_BoxIMin(nbor_box), imin0);

                  pj = hypre_SStructNMapInfoPart(entry_info);
                  coord = hypre_SStructNMapInfoCoord(entry_info);
                  dir = hypre_SStructNMapInfoDir(entry_info);
                  hypre_SStructVarToNborVar(grid, pi, vi, coord, &vj);

                  cinfoi = (((pi)*maxvars + vi)*nparts + pj)*maxvars + vj;
                  cinfoj = (((pj)*maxvars + vj)*nparts + pi)*maxvars + vi;

                  /* allocate CommInfo arguments */
                  if (cinfo_a[cinfoi] == NULL)
                  {
                     int  j_num_boxes = hypre_StructGridNumBoxes(
                        hypre_SStructPGridSGrid(pgrids[pj], vj));

                     cnum_transforms = hypre_CTAlloc(int, 1);
                     ccoords = hypre_CTAlloc(hypre_Index,
                                             hypre_BoxMapNEntries(map));
                     cdirs   = hypre_CTAlloc(hypre_Index,
                                             hypre_BoxMapNEntries(map));

                     cinfo = hypre_TAlloc(CInfo, 1);
                     (cinfo -> boxes) = hypre_BoxArrayArrayCreate(num_boxes);
                     (cinfo -> procs) = hypre_CTAlloc(int *, num_boxes);
                     (cinfo -> rboxnums) = hypre_CTAlloc(int *, num_boxes);
                     (cinfo -> rboxes) = hypre_BoxArrayArrayCreate(num_boxes);
                     (cinfo -> num_transforms) = cnum_transforms;
                     (cinfo -> coords) = ccoords;
                     (cinfo -> dirs) = cdirs;
                     (cinfo -> transforms) = hypre_CTAlloc(int *, num_boxes);
                     cinfo_a[cinfoi] = cinfo;

                     cinfo = hypre_TAlloc(CInfo, 1);
                     (cinfo -> boxes) = hypre_BoxArrayArrayCreate(j_num_boxes);
                     (cinfo -> procs) = hypre_CTAlloc(int *, j_num_boxes);
                     (cinfo -> rboxnums) = hypre_CTAlloc(int *, j_num_boxes);
                     (cinfo -> rboxes) = hypre_BoxArrayArrayCreate(j_num_boxes);
                     (cinfo -> num_transforms) = cnum_transforms;
                     (cinfo -> coords) = ccoords;
                     (cinfo -> dirs) = cdirs;
                     (cinfo -> transforms) = hypre_CTAlloc(int *, j_num_boxes);
                     cinfo_a[cinfoj] = cinfo;

                     vnbor_ncomms++;
                  }

                  cinfo = cinfo_a[cinfoi];

                  /* part with the smaller ID owns the primary variables */
                  if (pi < pj)
                  {
                     /* grow nbor_box to compute send boxes */
                     hypre_BoxIMinX(nbor_box) -= hypre_IndexX(varoffset);
                     hypre_BoxIMinY(nbor_box) -= hypre_IndexY(varoffset);
                     hypre_BoxIMinZ(nbor_box) -= hypre_IndexZ(varoffset);
                     hypre_BoxIMaxX(nbor_box) += hypre_IndexX(varoffset);
                     hypre_BoxIMaxY(nbor_box) += hypre_IndexY(varoffset);
                     hypre_BoxIMaxZ(nbor_box) += hypre_IndexZ(varoffset);
                  }
                  else if (pj < pi)
                  {
                     /* grow box to compute recv boxes */
                     hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
                     hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
                     hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
                     hypre_BoxIMaxX(box) += hypre_IndexX(varoffset);
                     hypre_BoxIMaxY(box) += hypre_IndexY(varoffset);
                     hypre_BoxIMaxZ(box) += hypre_IndexZ(varoffset);
                  }

                  cbox_a = hypre_BoxArrayArrayBoxArray((cinfo -> boxes), bi);
                  cprocs = (cinfo -> procs);
                  crboxnums = (cinfo -> rboxnums);
                  crbox_a = hypre_BoxArrayArrayBoxArray((cinfo -> rboxes), bi);
                  cnum_transforms = (cinfo -> num_transforms);
                  ccoords = (cinfo -> coords);
                  cdirs = (cinfo -> dirs);
                  ctransforms = (cinfo -> transforms);

                  hypre_IntersectBoxes(box, nbor_box, int_box);

                  if (hypre_BoxVolume(int_box))
                  {
                     /* map to neighbor part index space */
                     hypre_CopyBox(int_box, int_rbox);
                     hypre_SStructBoxToNborBox(
                        int_rbox, imin0, hypre_SStructNMapInfoILower(entry_info),
                        coord, dir);
                           
                     size = hypre_BoxArraySize(cbox_a);
                     if (size == 0)
                     {
                        /* use nentries as an upper bound */
                        cprocs[bi]    = hypre_CTAlloc(int, nentries);
                        crboxnums[bi] = hypre_CTAlloc(int, nentries);
                        ctransforms[bi] = hypre_CTAlloc(int, nentries);
                     }
                     hypre_AppendBox(int_box, cbox_a);
                     cprocs[bi][size] = hypre_SStructMapInfoProc(entry_info);
                     crboxnums[bi][size] = hypre_SStructMapInfoBoxnum(entry_info);
                     hypre_AppendBox(int_rbox, crbox_a);
                     /* search for transform */
                     for (ti = 0; ti < *cnum_transforms; ti++)
                     {
                        if ( (hypre_IndexX(coord) == hypre_IndexX(ccoords[ti])) &&
                             (hypre_IndexY(coord) == hypre_IndexY(ccoords[ti])) &&
                             (hypre_IndexZ(coord) == hypre_IndexZ(ccoords[ti])) &&
                             (hypre_IndexX(dir) == hypre_IndexX(cdirs[ti])) &&
                             (hypre_IndexY(dir) == hypre_IndexY(cdirs[ti])) &&
                             (hypre_IndexZ(dir) == hypre_IndexZ(cdirs[ti])) )
                        {
                           break;
                        }
                     }
                     /* set transform */
                     if (ti >= *cnum_transforms)
                     {
                        hypre_CopyIndex(coord, ccoords[ti]);
                        hypre_CopyIndex(dir, cdirs[ti]);
                        (*cnum_transforms)++;
                     }
                     ctransforms[bi][size] = ti;
                  }
               }
            }
            hypre_TFree(entries);
         }
      }
   }

   /* loop through the upper triangle and create vnbor_comm_info */
   vnbor_comm_info = hypre_TAlloc(hypre_SStructCommInfo *, vnbor_ncomms);
   vnbor_ncomms = 0;
   for (pi = 0; pi < nparts; pi++)
   {
      for (vi = 0; vi < maxvars; vi++)
      {
         for (pj = (pi+1); pj < nparts; pj++)
         {
            for (vj = 0; vj < maxvars; vj++)
            {
               cinfoi = (((pi)*maxvars + vi)*nparts + pj)*maxvars + vj;

               if (cinfo_a[cinfoi] != NULL)
               {
                  comm_info = hypre_TAlloc(hypre_SStructCommInfo, 1);
                  
                  cinfoj = (((pj)*maxvars + vj)*nparts + pi)*maxvars + vi;
                  send_cinfo = cinfo_a[cinfoi];
                  recv_cinfo = cinfo_a[cinfoj];
                  
                  /* send/recv boxes may not match (2nd to last argument) */
                  hypre_CommInfoCreate(
                     (send_cinfo -> boxes), (recv_cinfo -> boxes),
                     (send_cinfo -> procs), (recv_cinfo -> procs),
                     (send_cinfo -> rboxnums), (recv_cinfo -> rboxnums),
                     (send_cinfo -> rboxes), (recv_cinfo -> rboxes),
                     0, &hypre_SStructCommInfoCommInfo(comm_info));
                  hypre_CommInfoSetTransforms(
                     hypre_SStructCommInfoCommInfo(comm_info),
                     *(send_cinfo -> num_transforms),
                     (send_cinfo -> coords), (send_cinfo -> dirs),
                     (send_cinfo -> transforms), (recv_cinfo -> transforms));
                  hypre_TFree(send_cinfo -> num_transforms);
                  
                  hypre_SStructCommInfoSendPart(comm_info) = pi;
                  hypre_SStructCommInfoRecvPart(comm_info) = pj;
                  hypre_SStructCommInfoSendVar(comm_info) = vi;
                  hypre_SStructCommInfoRecvVar(comm_info) = vj;
                  
                  vnbor_comm_info[vnbor_ncomms] = comm_info;
                  vnbor_ncomms++;
               }
            }
         }
      }
   }
   hypre_SStructGridVNborCommInfo(grid) = vnbor_comm_info;
   hypre_SStructGridVNborNComms(grid) = vnbor_ncomms;

   size = nparts*nparts*maxvars*maxvars;
   for (cinfoi = 0; cinfoi < size; cinfoi++)
   {
      hypre_TFree(cinfo_a[cinfoi]);
   }
   hypre_TFree(cinfo_a);
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(nbor_box);
   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(int_rbox);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine returns a NULL 'entry_ptr' if an entry is not found
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridFindMapEntry( hypre_SStructGrid  *grid,
                               int                 part,
                               hypre_Index         index,
                               int                 var,
                               hypre_BoxMapEntry **entry_ptr )
{
   hypre_BoxMapFindEntry(hypre_SStructGridMap(grid, part, var),
                         index, entry_ptr);

   return hypre_error_flag;
}

int
hypre_SStructGridBoxProcFindMapEntry( hypre_SStructGrid  *grid,
                                      int                 part,
                                      int                 var,
                                      int                 box,
                                      int                 proc,
                                      hypre_BoxMapEntry **entry_ptr )
{
   hypre_BoxMapFindBoxProcEntry(hypre_SStructGridMap(grid, part, var),
                                box, proc, entry_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetCSRstrides( hypre_BoxMapEntry *entry,
                                   hypre_Index        strides )
{
   hypre_SStructMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);

   if (hypre_SStructMapInfoType(entry_info) == hypre_SSTRUCT_MAP_INFO_DEFAULT)
   {
      hypre_Index  imin;
      hypre_Index  imax;

      hypre_BoxMapEntryGetExtents(entry, imin, imax);

      strides[0] = 1;
      strides[1] = hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1;
      strides[2] = hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1;
      strides[2] *= strides[1];
   }
   else
   {
      hypre_SStructNMapInfo *entry_ninfo;

      entry_ninfo = (hypre_SStructNMapInfo *) entry_info;
      hypre_CopyIndex(hypre_SStructNMapInfoStride(entry_ninfo), strides);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 addition for a ghost stride calculation
 * same function except that you modify imin, imax with the ghost and
 * when the info is type nmapinfo you pull the ghoststrides.
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetGhstrides( hypre_BoxMapEntry *entry,
                                   hypre_Index        strides )
{
   hypre_SStructMapInfo *entry_info;
   int         *numghost;
   int         d ;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);

   if (hypre_SStructMapInfoType(entry_info) == hypre_SSTRUCT_MAP_INFO_DEFAULT)
   {
      hypre_Index  imin;
      hypre_Index  imax;

      hypre_BoxMapEntryGetExtents(entry, imin, imax);

      /* GEC1002 getting the ghost from the mapentry to modify imin, imax */

      numghost = hypre_BoxMapEntryNumGhost(entry);

      for (d = 0; d < 3; d++)
      { 
	imax[d] += numghost[2*d+1];
        imin[d] -= numghost[2*d];
      }  

      /* GEC1002 imin, imax modified now and calculation identical.  */

      strides[0] = 1;
      strides[1] = hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1;
      strides[2] = hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1;
      strides[2] *= strides[1];
   }
   else
   {
      hypre_SStructNMapInfo *entry_ninfo;
      /* GEC1002 we now get the ghost strides using the macro   */
      entry_ninfo = (hypre_SStructNMapInfo *) entry_info;
      hypre_CopyIndex(hypre_SStructNMapInfoGhstride(entry_ninfo), strides);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetGlobalCSRank( hypre_BoxMapEntry *entry,
                                      hypre_Index        index,
                                      int               *rank_ptr )
{
   hypre_SStructMapInfo *entry_info;
   hypre_Index           imin;
   hypre_Index           imax;
   hypre_Index           strides;
   int                   offset;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxMapEntryGetExtents(entry, imin, imax);
   offset = hypre_SStructMapInfoOffset(entry_info);

   hypre_SStructMapEntryGetCSRstrides(entry, strides);

   *rank_ptr = offset +
      (hypre_IndexD(index, 2) - hypre_IndexD(imin, 2)) * strides[2] +
      (hypre_IndexD(index, 1) - hypre_IndexD(imin, 1)) * strides[1] +
      (hypre_IndexD(index, 0) - hypre_IndexD(imin, 0)) * strides[0];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 a way to get the rank when you are in the presence of ghosts
 * It could have been a function pointer but this is safer. It computes
 * the ghost rank by using ghoffset, ghstrides and imin is modified
 *--------------------------------------------------------------------------*/


int
hypre_SStructMapEntryGetGlobalGhrank( hypre_BoxMapEntry *entry,
                                      hypre_Index        index,
                                      int               *rank_ptr )
{
   hypre_SStructMapInfo *entry_info;
   hypre_Index           imin;
   hypre_Index           imax;
   hypre_Index           ghstrides;
   int                   ghoffset;
   int                   *numghost = hypre_BoxMapEntryNumGhost(entry);
   int                   d;
   int                   info_type;
   

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxMapEntryGetExtents(entry, imin, imax);
   ghoffset = hypre_SStructMapInfoGhoffset(entry_info);
   info_type = hypre_SStructMapInfoType(entry_info);
  

   hypre_SStructMapEntryGetGhstrides(entry, ghstrides);

   /* GEC shifting the imin according to the ghosts when you have a default info
    * When you have a neighbor info, you do not need to shift the imin since
    * the ghoffset for neighbor info has factored in the ghost presence during
    * the neighbor info assemble phase   */

   if (info_type == hypre_SSTRUCT_MAP_INFO_DEFAULT)
   {
      for (d = 0; d < 3; d++)
      {
         imin[d] -= numghost[2*d];
      }
   }
   
   *rank_ptr = ghoffset +
      (hypre_IndexD(index, 2) - hypre_IndexD(imin, 2)) * ghstrides[2] +
      (hypre_IndexD(index, 1) - hypre_IndexD(imin, 1)) * ghstrides[1] +
      (hypre_IndexD(index, 0) - hypre_IndexD(imin, 0)) * ghstrides[0];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetProcess( hypre_BoxMapEntry *entry,
                                 int               *proc_ptr )
{
   hypre_SStructMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   *proc_ptr = hypre_SStructMapInfoProc(entry_info);

   return hypre_error_flag;
}

int
hypre_SStructMapEntryGetBoxnum( hypre_BoxMapEntry *entry,
                                int               *boxnum_ptr )
{
   hypre_SStructMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   *boxnum_ptr = hypre_SStructMapInfoBoxnum(entry_info);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mapping Notes:
 *
 *   coord maps Box index-space to NborBox index-space.  That is, `coord[d]' is
 *   the dimension in the NborBox index-space, and `d' is the dimension in the
 *   Box index-space.
 *
 *   dir also works on the Box index-space.  That is, `dir[d]' is the direction
 *   (positive or negative) of dimension `coord[d]' in the NborBox index-space,
 *   relative to the positive direction of dimension `d' in the Box index-space.
 *
 *--------------------------------------------------------------------------*/

int
hypre_SStructBoxToNborBox( hypre_Box   *box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   int *imin = hypre_BoxIMin(box);
   int *imax = hypre_BoxIMax(box);
   int  nbor_imin[3];
   int  nbor_imax[3];

   int  d, nd;

   for (d = 0; d < 3; d++)
   {
      nd = coord[d];
      nbor_imin[nd] = nbor_index[nd] + (imin[d] - index[d]) * dir[d];
      nbor_imax[nd] = nbor_index[nd] + (imax[d] - index[d]) * dir[d];
   }

   for (d = 0; d < 3; d++)
   {
      imin[d] = hypre_min(nbor_imin[d], nbor_imax[d]);
      imax[d] = hypre_max(nbor_imin[d], nbor_imax[d]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * See "Mapping Notes" in comment for `hypre_SStructBoxToNborBox'.
 *--------------------------------------------------------------------------*/

int
hypre_SStructNborBoxToBox( hypre_Box   *nbor_box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   int *nbor_imin = hypre_BoxIMin(nbor_box);
   int *nbor_imax = hypre_BoxIMax(nbor_box);
   int  imin[3];
   int  imax[3];

   int  d, nd;

   for (d = 0; d < 3; d++)
   {
      nd = coord[d];
      imin[d] = index[d] + (nbor_imin[nd] - nbor_index[nd]) * dir[d];
      imax[d] = index[d] + (nbor_imax[nd] - nbor_index[nd]) * dir[d];
   }

   for (d = 0; d < 3; d++)
   {
      nbor_imin[d] = hypre_min(imin[d], imax[d]);
      nbor_imax[d] = hypre_max(imin[d], imax[d]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *
 * Assumptions:
 *
 * 1. Variables and variable types are the same on neighboring parts
 * 2. Variable types are listed in order as follows:
 *       Face - XFACE, YFACE, ZFACE
 *       Edge - XEDGE, YEDGE, ZEDGE
 * 3. If the coordinate transformation is not the identity, then all ndim
 *    variable types must exist on the grid.
 *
 *--------------------------------------------------------------------------*/

int
hypre_SStructVarToNborVar( hypre_SStructGrid  *grid,
                           int                 part,
                           int                 var,
                           int                *coord,
                           int                *nbor_var_ptr)
{
   hypre_SStructPGrid     *pgrid   = hypre_SStructGridPGrid(grid, part);
   HYPRE_SStructVariable   vartype = hypre_SStructPGridVarType(pgrid, var);

   switch(vartype)
   {
      case HYPRE_SSTRUCT_VARIABLE_XFACE:
      case HYPRE_SSTRUCT_VARIABLE_XEDGE:
         *nbor_var_ptr = var + (coord[0]  );
         break;
      case HYPRE_SSTRUCT_VARIABLE_YFACE:
      case HYPRE_SSTRUCT_VARIABLE_YEDGE:
         *nbor_var_ptr = var + (coord[1]-1);
         break;
      case HYPRE_SSTRUCT_VARIABLE_ZFACE:
      case HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         *nbor_var_ptr = var + (coord[2]-2);
         break;
      default:
         *nbor_var_ptr = var;
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC0902 a function that will set the ghost in each of the sgrids
 *
 *--------------------------------------------------------------------------*/
int
hypre_SStructGridSetNumGhost( hypre_SStructGrid  *grid, int *num_ghost )
{
   int                   nparts = hypre_SStructGridNParts(grid);
   int                   nvars ;
   int                   part  ;
   int                   var  ;
   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;

   for (part = 0; part < nparts; part++)
   {

     pgrid = hypre_SStructGridPGrid(grid, part);
     nvars = hypre_SStructPGridNVars(pgrid);
     
     for ( var = 0; var < nvars; var++)
     {
       sgrid = hypre_SStructPGridSGrid(pgrid, var);
       hypre_StructGridSetNumGhost(sgrid, num_ghost);
     }

   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 a function that will select the right way to calculate the rank
 * depending on the matrix type. It is an extension to the usual GetGlobalRank
 *
 *--------------------------------------------------------------------------*/
int
hypre_SStructMapEntryGetGlobalRank(hypre_BoxMapEntry   *entry,
                                   hypre_Index          index,
                                   int              *rank_ptr,
                                   int                    type)
{
   if (type == HYPRE_PARCSR)
   {
      hypre_SStructMapEntryGetGlobalCSRank(entry,index,rank_ptr);
   }
   if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
   {
      hypre_SStructMapEntryGetGlobalGhrank(entry,index,rank_ptr);
   }

   return hypre_error_flag;
}  

/*--------------------------------------------------------------------------
 * GEC1002 a function that will select the right way to calculate the strides
 * depending on the matrix type. It is an extension to the usual strides
 *
 *--------------------------------------------------------------------------*/
int
hypre_SStructMapEntryGetStrides(hypre_BoxMapEntry   *entry,
                                hypre_Index          strides,
                                   int                 type)
{
   if (type == HYPRE_PARCSR)
   {
      hypre_SStructMapEntryGetCSRstrides(entry,strides);
   }
   if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
   {
      hypre_SStructMapEntryGetGhstrides(entry,strides);
   }

   return hypre_error_flag;
}  

/*--------------------------------------------------------------------------
 *  A function to determine the local variable box numbers that underlie
 *  a cellbox with local box number boxnum. Only returns local box numbers
 *  of myproc.
 *--------------------------------------------------------------------------*/
int
hypre_SStructBoxNumMap(hypre_SStructGrid        *grid,
                       int                       part,
                       int                       boxnum,
                       int                     **num_varboxes_ptr,
                       int                    ***map_ptr)
{
   hypre_SStructPGrid    *pgrid   = hypre_SStructGridPGrid(grid, part);
   hypre_StructGrid      *cellgrid= hypre_SStructPGridCellSGrid(pgrid);
   hypre_StructGrid      *vargrid;
   hypre_BoxArray        *boxes;
   hypre_Box             *cellbox, vbox, *box, intersect_box;
   HYPRE_SStructVariable *vartypes= hypre_SStructPGridVarTypes(pgrid);

   int                    ndim    = hypre_SStructGridNDim(grid);
   int                    nvars   = hypre_SStructPGridNVars(pgrid);
   hypre_Index            varoffset;

   int                   *num_boxes;
   int                  **var_boxnums;
   int                   *temp;

   int                    i, j, k, var;

   cellbox= hypre_StructGridBox(cellgrid, boxnum);

  /* ptrs to store var_box map info */
   num_boxes  = hypre_CTAlloc(int, nvars);
   var_boxnums= hypre_TAlloc(int *, nvars);

  /* intersect the cellbox with the var_boxes */
   for (var= 0; var< nvars; var++)
   {
      vargrid= hypre_SStructPGridSGrid(pgrid, var);
      boxes  = hypre_StructGridBoxes(vargrid);
      temp   = hypre_CTAlloc(int, hypre_BoxArraySize(boxes));

     /* map cellbox to a variable box */
      hypre_CopyBox(cellbox, &vbox);

      i= vartypes[var];
      hypre_SStructVariableGetOffset((hypre_SStructVariable) i,
                                      ndim, varoffset);
      hypre_SubtractIndex(hypre_BoxIMin(&vbox), varoffset,
                          hypre_BoxIMin(&vbox));

     /* loop over boxes to see if they intersect with vbox */
      hypre_ForBoxI(i, boxes)
      {
         box= hypre_BoxArrayBox(boxes, i);
         hypre_IntersectBoxes(&vbox, box, &intersect_box);
         if (hypre_BoxVolume(&intersect_box))
         {
            temp[i]++;
            num_boxes[var]++;
         }
      }

     /* record local var box numbers */
      if (num_boxes[var])
      {
         var_boxnums[var]= hypre_TAlloc(int, num_boxes[var]);
      }
      else
      {
         var_boxnums[var]= NULL;
      }

      j= 0;
      k= hypre_BoxArraySize(boxes);
      for (i= 0; i< k; i++)
      {
         if (temp[i])
         {
            var_boxnums[var][j]= i;
            j++;
         }
      }
      hypre_TFree(temp);

   }  /* for (var= 0; var< nvars; var++) */

  *num_varboxes_ptr= num_boxes;
  *map_ptr= var_boxnums;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  A function to extract all the local var box numbers underlying the
 *  cellgrid boxes.
 *--------------------------------------------------------------------------*/
int
hypre_SStructCellGridBoxNumMap(hypre_SStructGrid        *grid,
                               int                       part,
                               int                    ***num_varboxes_ptr,
                               int                   ****map_ptr)
{
   hypre_SStructPGrid    *pgrid    = hypre_SStructGridPGrid(grid, part);
   hypre_StructGrid      *cellgrid = hypre_SStructPGridCellSGrid(pgrid);
   hypre_BoxArray        *cellboxes= hypre_StructGridBoxes(cellgrid);
   
   int                  **num_boxes;
   int                 ***var_boxnums;

   int                    i, ncellboxes;

   ncellboxes = hypre_BoxArraySize(cellboxes);

   num_boxes  = hypre_TAlloc(int *, ncellboxes);
   var_boxnums= hypre_TAlloc(int **, ncellboxes);

   hypre_ForBoxI(i, cellboxes)
   {
       hypre_SStructBoxNumMap(grid,
                              part,
                              i,
                             &num_boxes[i],
                             &var_boxnums[i]);
   }

  *num_varboxes_ptr= num_boxes;
  *map_ptr= var_boxnums;

   return hypre_error_flag;
}
