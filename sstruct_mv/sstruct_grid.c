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

   hypre_SStructPGridComm(pgrid)     = comm;
   hypre_SStructPGridNDim(pgrid)     = ndim;
   hypre_SStructPGridNVars(pgrid)    = 0;
   hypre_SStructPGridVarTypes(pgrid) = NULL;
   
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
    *-------------------------------------------------------------*/

   for (var = 0; var < nvars; var++)
   {
      sgrid = hypre_SStructPGridSGrid(pgrid, var);
      hypre_SStructPGridLocalSize(pgrid)  += hypre_StructGridLocalSize(sgrid);
      hypre_SStructPGridGlobalSize(pgrid) += hypre_StructGridGlobalSize(sgrid);
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
   int                      **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor   ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor     *vneighbor;
   hypre_BoxMap            ***maps;
   hypre_SStructMapInfo    ***info;
   hypre_SStructNMapInfo   ***ninfo;
   hypre_SStructPGrid        *pgrid;
   int                        nvars;
   hypre_StructGrid          *sgrid;
   hypre_Box                 *bounding_box;

   int                       *offsets;
   hypre_SStructMapInfo      *entry_info;
   hypre_SStructNMapInfo     *entry_ninfo;
   hypre_BoxArray            *boxes;
   hypre_Box                 *box;
   int                       *procs;
   int                        first_local;

   hypre_BoxMapEntry         *map_entry;
   hypre_Box                 *nbor_box;
   int                        nbor_part;
   hypre_Index                nbor_ilower;
   int                        nbor_offset;
   int                        nbor_proc;

   int                        nprocs, myproc;
   int                        proc, part, var, b;

   /*------------------------------------------------------
    * Build map info for grid boxes
    *------------------------------------------------------*/

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myproc);
   offsets = hypre_TAlloc(int, nprocs + 1);
   offsets[0] = 0;
   MPI_Allgather(&local_size, 1, MPI_INT, &offsets[1], 1, MPI_INT, comm);
   for (proc = 1; proc < (nprocs + 1); proc++)
   {
      offsets[proc] += offsets[proc-1];
   }
   hypre_SStructGridStartRank(grid) = offsets[myproc];

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

         hypre_BoxMapCreate(hypre_BoxArraySize(boxes) + nvneighbors[part][var],
                            hypre_BoxIMin(bounding_box),
                            hypre_BoxIMax(bounding_box),
                            &maps[part][var]);
         info[part][var] = hypre_TAlloc(hypre_SStructMapInfo,
                                        hypre_BoxArraySize(boxes));

         for (b = 0; b < hypre_BoxArraySize(boxes); b++)
         {
            box = hypre_BoxArrayBox(boxes, b);
            proc = procs[b];
            
            entry_info = &info[part][var][b];
            hypre_SStructMapInfoType(entry_info) =
               hypre_SSTRUCT_MAP_INFO_DEFAULT;
            hypre_SStructMapInfoProc(entry_info)   = proc;
            hypre_SStructMapInfoOffset(entry_info) = offsets[proc];
            
            hypre_BoxMapAddEntry(maps[part][var],
                                 hypre_BoxIMin(box),
                                 hypre_BoxIMax(box),
                                 entry_info);

            offsets[proc] += hypre_BoxVolume(box);
         }

         hypre_BoxArrayDestroy(boxes);
         hypre_TFree(procs);

         hypre_BoxMapAssemble(maps[part][var]);
      }
   }
   hypre_TFree(offsets);

   hypre_SStructGridMaps(grid) = maps;
   hypre_SStructGridInfo(grid) = info;

   /*------------------------------------------------------
    * Add neighbor boxes to maps and re-assemble
    *------------------------------------------------------*/

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
            hypre_SStructGridFindMapEntry(grid, nbor_part, nbor_ilower, var,
                                          &map_entry);
            hypre_SStructMapEntryGetProcess(map_entry, &nbor_proc);
            hypre_SStructMapEntryGetGlobalRank(map_entry, nbor_ilower,
                                               &nbor_offset);

            entry_ninfo = &ninfo[part][var][b];
            hypre_SStructNMapInfoType(entry_ninfo) =
               hypre_SSTRUCT_MAP_INFO_NEIGHBOR;
            hypre_SStructNMapInfoProc(entry_ninfo)   = nbor_proc;
            hypre_SStructNMapInfoOffset(entry_ninfo) = nbor_offset;
            hypre_SStructNMapInfoPart(entry_ninfo)   = nbor_part;
            hypre_CopyIndex(nbor_ilower,
                            hypre_SStructNMapInfoILower(entry_ninfo));
            hypre_CopyIndex(hypre_SStructNeighborCoord(vneighbor),
                            hypre_SStructNMapInfoCoord(entry_ninfo));
            hypre_CopyIndex(hypre_SStructNeighborDir(vneighbor),
                            hypre_SStructNMapInfoDir(entry_ninfo));

            hypre_BoxMapAddEntry(maps[part][var],
                                 hypre_BoxIMin(nbor_box),
                                 hypre_BoxIMax(nbor_box),
                                 entry_ninfo);
         }

         hypre_BoxMapAssemble(maps[part][var]);
      }
   }

   hypre_SStructGridNInfo(grid) = ninfo;

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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructMapEntryGetGlobalRank( hypre_BoxMapEntry *entry,
                                    hypre_Index        index,
                                    int               *rank_ptr )
{
   int ierr = 0;

   hypre_Index              imin;
   hypre_Index              imax;
   hypre_SStructMapInfo    *entry_info;
   int                      offset, s[3];

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxMapEntryGetExtents(entry, imin, imax);
   offset = hypre_SStructMapInfoOffset(entry_info);

   if (hypre_SStructMapInfoType(entry_info) == hypre_SSTRUCT_MAP_INFO_DEFAULT)
   {
      s[0] = 1;
      s[1] = hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1;
      s[2] = hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1;
      s[2] *= s[1];
   }
   else
   {
      hypre_SStructNMapInfo *entry_ninfo;
      hypre_Index            c, d;

      entry_ninfo = (hypre_SStructNMapInfo *) entry_info;
      hypre_CopyIndex(hypre_SStructNMapInfoCoord(entry_ninfo), c);
      hypre_CopyIndex(hypre_SStructNMapInfoDir(entry_ninfo), d);

      s[c[0]] = 1;
      s[c[1]] = hypre_IndexD(imax, c[0]) - hypre_IndexD(imin, c[0]) + 1;
      s[c[2]] = hypre_IndexD(imax, c[1]) - hypre_IndexD(imin, c[1]) + 1;
      s[c[2]] *= s[c[1]];
      s[0] *= d[c[0]];
      s[1] *= d[c[1]];
      s[2] *= d[c[2]];
   }

   *rank_ptr = offset +
      (hypre_IndexD(index, 2) - hypre_IndexD(imin, 2)) * s[2] +
      (hypre_IndexD(index, 1) - hypre_IndexD(imin, 1)) * s[1] +
      (hypre_IndexD(index, 0) - hypre_IndexD(imin, 0)) * s[0];

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

