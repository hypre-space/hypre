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
   
   hypre_SStructPGridLocalSize(pgrid)  = 0;
   hypre_SStructPGridGlobalSize(pgrid) = 0;
   
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

   hypre_StructGrid      *cell_sgrid;
   hypre_StructGrid      *sgrid;
   hypre_BoxArray        *iboxarray;
   hypre_BoxNeighbors    *hood;
   hypre_BoxArray        *hood_boxes;
   int                    hood_first_local;
   int                    hood_num_local;
   hypre_Box             *box;
   hypre_Box             *nbox;
   hypre_Box             *dbox;
   hypre_BoxArray        *boxes;
   hypre_BoxArray        *diff_boxes;
   hypre_BoxArray        *new_diff_boxes;
   hypre_BoxArray        *tmp_boxes;
   hypre_BoxArray        *tmp_boxes_ref;
   hypre_Index            varoffset;

   int                    t, var, i, j, k;

   /*-------------------------------------------------------------
    * set up the uniquely distributed sgrids for each vartype
    *-------------------------------------------------------------*/

   cell_sgrid = hypre_SStructPGridCellSGrid(pgrid);
   HYPRE_StructGridAssemble(cell_sgrid);

   hood = hypre_StructGridNeighbors(cell_sgrid);
   hood_boxes       = hypre_BoxNeighborsBoxes(hood);
   hood_first_local = hypre_BoxNeighborsFirstLocal(hood);
   hood_num_local   = hypre_BoxNeighborsNumLocal(hood);

   box  = hypre_BoxCreate();
   nbox = hypre_BoxCreate();
   diff_boxes     = hypre_BoxArrayCreate(0);
   new_diff_boxes = hypre_BoxArrayCreate(0);
   tmp_boxes      = hypre_BoxArrayCreate(0);

   for (var = 0; var < nvars; var++)
   {
      t = vartypes[var];

      if ((t > 0) && (sgrids[t] == NULL))
      {
         HYPRE_StructGridCreate(comm, ndim, &sgrid);
         boxes = hypre_BoxArrayCreate(0);
         hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                        ndim, varoffset);

         for (i = hood_first_local;
              i < (hood_first_local + hood_num_local); i++)
         {
            hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, i), box);
            hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
            hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
            hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);

            /* compute diff_boxes = (box - neighbors with smaller ID) */
            hypre_BoxArraySetSize(diff_boxes, 0);
            hypre_AppendBox(box, diff_boxes);
            for (j = 0; j < i; j++)
            {
               hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, j), nbox);
               hypre_BoxIMinX(nbox) -= hypre_IndexX(varoffset);
               hypre_BoxIMinY(nbox) -= hypre_IndexY(varoffset);
               hypre_BoxIMinZ(nbox) -= hypre_IndexZ(varoffset);

               /* compute new_diff_boxes = (diff_boxes - nbox) */
               hypre_BoxArraySetSize(new_diff_boxes, 0);
               hypre_ForBoxI(k, diff_boxes)
                  {
                     dbox = hypre_BoxArrayBox(diff_boxes, k);
                     hypre_SubtractBoxes(dbox, nbox, tmp_boxes);
                     hypre_AppendBoxArray(tmp_boxes, new_diff_boxes);
                  }

               /* swap diff_boxes and new_diff_boxes */
               tmp_boxes_ref  = new_diff_boxes;
               new_diff_boxes = diff_boxes;
               diff_boxes     = tmp_boxes_ref;
            }

            hypre_AppendBoxArray(diff_boxes, boxes);
         }

         hypre_StructGridSetBoxes(sgrid, boxes);
         HYPRE_StructGridAssemble(sgrid);

         sgrids[t] = sgrid;
      }            
   }

   hypre_BoxArrayDestroy(diff_boxes);
   hypre_BoxArrayDestroy(new_diff_boxes);
   hypre_BoxArrayDestroy(tmp_boxes);
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(nbox);

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

   MPI_Comm                   comm       = hypre_SStructGridComm(grid);
   int                        nparts     = hypre_SStructGridNParts(grid);
   int                        local_size = hypre_SStructGridLocalSize(grid);
   hypre_BoxMap            ***maps;
   hypre_SStructBoxMapInfo ***info;
   hypre_SStructPGrid        *pgrid;
   int                        nvars;
   hypre_StructGrid          *sgrid;
   hypre_Box                 *bounding_box;

   int                       *offsets;
   hypre_SStructBoxMapInfo   *entry_info;
   hypre_BoxArray            *boxes;
   hypre_Box                 *box;
   int                       *procs;
   int                        first_local;

   int                        nprocs, myproc;
   int                        proc, part, var, b;

   /*------------------------------------------------------
    * Compute starting ranks
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
   info = hypre_TAlloc(hypre_SStructBoxMapInfo **, nparts);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      maps[part] = hypre_TAlloc(hypre_BoxMap *, nvars);
      info[part] = hypre_TAlloc(hypre_SStructBoxMapInfo *, nvars);
      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);

         /* NOTE: With neighborhood info from user, don't need all gather */
         hypre_GatherAllBoxes(comm, hypre_StructGridBoxes(sgrid),
                              &boxes, &procs, &first_local);
         bounding_box = hypre_StructGridBoundingBox(sgrid);

         hypre_BoxMapCreate(hypre_BoxArraySize(boxes),
                            hypre_BoxIMin(bounding_box),
                            hypre_BoxIMax(bounding_box),
                            &maps[part][var]);
         info[part][var] = hypre_TAlloc(hypre_SStructBoxMapInfo,
                                        hypre_BoxArraySize(boxes));

         for (b = 0; b < hypre_BoxArraySize(boxes); b++)
         {
            box = hypre_BoxArrayBox(boxes, b);
            proc = procs[b];
            
            entry_info = &info[part][var][b];
            hypre_SStructBoxMapInfoProc(entry_info)   = proc;
            hypre_SStructBoxMapInfoOffset(entry_info) = offsets[proc];
            
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
   hypre_SStructGridMaps(grid) = maps;
   hypre_SStructGridInfo(grid) = info;

   hypre_TFree(offsets);

   return ierr;
}

/*--------------------------------------------------------------------------
 * This routine return a NULL 'entry_ptr' if an entry is not found
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
hypre_SStructBoxMapEntryGetGlobalRank( hypre_BoxMapEntry *entry,
                                       hypre_Index        index,
                                       int               *rank_ptr )
{
   int ierr = 0;

   hypre_Index              imin;
   hypre_Index              imax;
   hypre_SStructBoxMapInfo *entry_info;
   int                      offset, stride2, stride1;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxMapEntryGetExtents(entry, imin, imax);
   offset = hypre_SStructBoxMapInfoOffset(entry_info);
   stride1 = (hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1);
   stride2 = (hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1) * stride1;

   *rank_ptr = offset +
      (hypre_IndexD(index, 2) - hypre_IndexD(imin, 2)) * stride2 +
      (hypre_IndexD(index, 1) - hypre_IndexD(imin, 1)) * stride1 +
      (hypre_IndexD(index, 0) - hypre_IndexD(imin, 0));

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_SStructBoxMapEntryGetProcess( hypre_BoxMapEntry *entry,
                                    int               *proc_ptr )
{
   int ierr = 0;

   hypre_SStructBoxMapInfo *entry_info;

   hypre_BoxMapEntryGetInfo(entry, (void **) &entry_info);
   *proc_ptr = hypre_SStructBoxMapInfoProc(entry_info);

   return ierr;
}

