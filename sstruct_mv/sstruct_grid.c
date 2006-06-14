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
   int ierr = 0;
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

   return ierr;
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
   int                  ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridDestroy( hypre_SStructPGrid *pgrid )
{
   int ierr = 0;

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

   return ierr;
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

   return ( HYPRE_StructGridSetExtents(sgrid, ilower, iupper) );
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetCellSGrid
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridSetCellSGrid( hypre_SStructPGrid  *pgrid,
                                hypre_StructGrid    *cell_sgrid )
{
   int                     ierr = 0;

   hypre_SStructPGridCellSGrid(pgrid) = cell_sgrid;
   hypre_SStructPGridCellSGridDone(pgrid) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetVariables
 *--------------------------------------------------------------------------*/

int hypre_SStructPGridSetVariables( hypre_SStructPGrid    *pgrid,
                                    int                    nvars,
                                    HYPRE_SStructVariable *vartypes )
{
   int                     ierr = 0;

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

   return ierr;
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
   int                     ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridSetPNeighbor
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridSetPNeighbor( hypre_SStructPGrid  *pgrid,
                                hypre_Box           *pneighbor_box )
{
   int ierr = 0;

   hypre_AppendBox(pneighbor_box, hypre_SStructPGridPNeighbors(pgrid));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPGridAssemble
 *--------------------------------------------------------------------------*/

int
hypre_SStructPGridAssemble( hypre_SStructPGrid  *pgrid )
{
   int                  ierr = 0;

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
   hypre_BoxNeighbors    *hood;
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

   hood = hypre_StructGridNeighbors(cell_sgrid);
   hood_boxes       = hypre_BoxNeighborsBoxes(hood);
   hood_first_local = hypre_BoxNeighborsFirstLocal(hood);
   hood_num_local   = hypre_BoxNeighborsNumLocal(hood);

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

   return ierr;
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

   return 0;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridAssembleMaps( hypre_SStructGrid *grid )
{
   int ierr = 0;

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
            hypre_SStructMapInfoBox(entry_info)    = local_boxnums[b];

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

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridAssembleNBorMaps( hypre_SStructGrid *grid )
{
   int ierr = 0;

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
   int                        nbor_part;
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
            hypre_CopyIndex(hypre_SStructNeighborILower(vneighbor),
                            nbor_ilower);
            /*
             * Note that this assumes that the entire neighbor box
             * actually lives on the global grid.  This was insured by
             * intersecting the neighbor boxes with the global grid.
             * We also assume that each neighbour box intersects
             * with only one box on the neighbouring processor. This
             * should be the case since we only have one map_entry.
             */
            hypre_SStructGridFindMapEntry(grid, nbor_part, nbor_ilower, var,
                                          &map_entry);

            hypre_BoxMapEntryGetExtents(map_entry,
                                        hypre_BoxIMin(box),
                                        hypre_BoxIMax(box));
            hypre_SStructMapEntryGetProcess(map_entry, &nbor_proc);
            hypre_SStructMapEntryGetBox(map_entry, &nbor_boxnum);

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
            hypre_SStructMapInfoBox(entry_ninfo)= nbor_boxnum;
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
            stride[c[0]] *= d[0];
            stride[c[1]] *= d[1];
            stride[c[2]] *= d[2];

            /* GEC1002 expanding the ghostbox to compute the strides based on ghosts vector  */

            hypre_CopyBox(box, ghostbox);
            hypre_BoxExpand(ghostbox,num_ghost);
            ghstride = hypre_SStructNMapInfoGhstride(entry_ninfo);
            ghstride[c[0]] = 1;
            ghstride[c[1]] = hypre_BoxSizeD(ghostbox, 0);
            ghstride[c[2]] = hypre_BoxSizeD(ghostbox, 1) * ghstride[c[1]];
            ghstride[c[0]] *= d[0];
            ghstride[c[1]] *= d[1];
            ghstride[c[2]] *= d[2];
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

   return ierr;
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
   int ierr = 0;

   hypre_BoxMapFindEntry(hypre_SStructGridMap(grid, part, var),
                         index, entry_ptr);

   return ierr;
}

int
hypre_SStructGridBoxProcFindMapEntry( hypre_SStructGrid  *grid,
                                      int                 part,
                                      int                 var,
                                      int                 box,
                                      int                 proc,
                                      hypre_BoxMapEntry **entry_ptr )
{
   int ierr = 0;

   hypre_BoxMapFindBoxProcEntry(hypre_SStructGridMap(grid, part, var),
                                box, proc, entry_ptr);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetCSRstrides( hypre_BoxMapEntry *entry,
                                   hypre_Index        strides )
{
   int ierr = 0;
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

   return ierr;
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
   int ierr = 0;
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

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetGlobalCSRank( hypre_BoxMapEntry *entry,
                                      hypre_Index        index,
                                      int               *rank_ptr )
{
   int ierr = 0;

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

   return ierr;
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
   int ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetProcess( hypre_BoxMapEntry *entry,
                                 int               *proc_ptr )
{
   int ierr = 0;

   hypre_SStructMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   *proc_ptr = hypre_SStructMapInfoProc(entry_info);

   return ierr;
}

int
hypre_SStructMapEntryGetBox( hypre_BoxMapEntry *entry,
                             int               *box_ptr )
{
   int ierr = 0;

   hypre_SStructMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   *box_ptr = hypre_SStructMapInfoBox(entry_info);

   return ierr;
}

/*--------------------------------------------------------------------------
 * Mapping Notes:
 *
 *   coord maps Box index-space to NBorBox index-space.  That is,
 *   `coord[d]' is the dimension in the NBorBox index-space, and
 *   `d' is the dimension in the Box index-space.
 *
 *   dir works on the NBorBox index-space.  That is, `dir[coord[d]]' is
 *   the direction (positive or negative) of dimension `coord[d]' in
 *   the NBorBox index-space, relative to the positive direction of
 *   dimension `d' in the Box index-space.
 *
 *--------------------------------------------------------------------------*/

int
hypre_SStructBoxToNBorBox( hypre_Box   *box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   int ierr = 0;

   int *imin = hypre_BoxIMin(box);
   int *imax = hypre_BoxIMax(box);
   int  nbor_imin[3];
   int  nbor_imax[3];

   int  d, nd;

   for (d = 0; d < 3; d++)
   {
      nd = coord[d];
      nbor_imin[nd] = nbor_index[nd] + (imin[d] - index[d]) * dir[nd];
      nbor_imax[nd] = nbor_index[nd] + (imax[d] - index[d]) * dir[nd];
   }

   for (d = 0; d < 3; d++)
   {
      imin[d] = hypre_min(nbor_imin[d], nbor_imax[d]);
      imax[d] = hypre_max(nbor_imin[d], nbor_imax[d]);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * See "Mapping Notes" in comment for `hypre_SStructBoxToNBorBox'.
 *--------------------------------------------------------------------------*/

int
hypre_SStructNBorBoxToBox( hypre_Box   *nbor_box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   int ierr = 0;

   int *nbor_imin = hypre_BoxIMin(nbor_box);
   int *nbor_imax = hypre_BoxIMax(nbor_box);
   int  imin[3];
   int  imax[3];

   int  d, nd;

   for (d = 0; d < 3; d++)
   {
      nd = coord[d];
      imin[d] = index[d] + (nbor_imin[nd] - nbor_index[nd]) * dir[nd];
      imax[d] = index[d] + (nbor_imax[nd] - nbor_index[nd]) * dir[nd];
   }

   for (d = 0; d < 3; d++)
   {
      nbor_imin[d] = hypre_min(imin[d], imax[d]);
      nbor_imax[d] = hypre_max(imin[d], imax[d]);
   }

   return ierr;
}


        

/*--------------------------------------------------------------------------
 * GEC0902 a function that will set the ghost in each of the sgrids
 *
 *--------------------------------------------------------------------------*/
int
hypre_SStructGridSetNumGhost( hypre_SStructGrid  *grid, int *num_ghost )
{
   int                  ierr = 0;

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
   return ierr;
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
  int ierr = 0;

  if (type == HYPRE_PARCSR)
  {
    hypre_SStructMapEntryGetGlobalCSRank(entry,index,rank_ptr);
  }
  if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
  {
    hypre_SStructMapEntryGetGlobalGhrank(entry,index,rank_ptr);
  }

  return ierr;
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
   int ierr = 0;
  
  if (type == HYPRE_PARCSR)
  {
    hypre_SStructMapEntryGetCSRstrides(entry,strides);
  }
  if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
  {
    hypre_SStructMapEntryGetGhstrides(entry,strides);
  }

  return ierr;
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
   int ierr = 0;
  
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

   return ierr;
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
   int ierr = 0;
  
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

   return ierr;
}
