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
      hypre_SStructPGridVTSGrid(pgrid, t) = NULL;
      hypre_SStructPGridVTIGrid(pgrid, t) = NULL;
   }
   HYPRE_StructGridCreate(comm, ndim, &sgrid);
   hypre_SStructPGridCellSGrid(pgrid) = sgrid;
   /* hypre_StructGridRef(sgrid, &hypre_SStructPGridCellIGrid(pgrid)); */
   
   for (t = 0; t < 8; t++)
   {
      hypre_SStructPGridVTMap(pgrid, t) = NULL;
   }
   hypre_SStructPGridOffsets(pgrid)   = NULL;
   hypre_SStructPGridStartRank(pgrid) = 0;
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
   /* hypre_StructGrid **igrids; */
   hypre_StructMap  **maps;
   int                t;

   if (pgrid)
   {
      sgrids   = hypre_SStructPGridSGrids(pgrid);
      /* igrids   = hypre_SStructPGridIGrids(pgrid); */
      maps     = hypre_SStructPGridMaps(pgrid);
      hypre_TFree(hypre_SStructPGridVarTypes(pgrid));
      for (t = 0; t < 8; t++)
      {
         HYPRE_StructGridDestroy(sgrids[t]);
         /* HYPRE_StructGridDestroy(igrids[t]); */
         hypre_StructMapDestroy(maps[t]);
      }
      hypre_TFree(hypre_SStructPGridOffsets(pgrid));
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

   MPI_Comm               comm     = hypre_SStructPGridComm(pgrid);
   int                    ndim     = hypre_SStructPGridNDim(pgrid);
   int                    nvars    = hypre_SStructPGridNVars(pgrid);
   HYPRE_SStructVariable *vartypes = hypre_SStructPGridVarTypes(pgrid);
   hypre_StructGrid     **sgrids   = hypre_SStructPGridSGrids(pgrid);
   /* hypre_StructGrid     **igrids   = hypre_SStructPGridIGrids(pgrid); */
   hypre_StructMap      **maps     = hypre_SStructPGridMaps(pgrid);
   int                   *offsets;
   int                    start_rank;

   hypre_StructGrid      *cell_sgrid;
   hypre_StructGrid      *sgrid;
   /* hypre_StructGrid      *igrid; */
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
   int                    offset;

   int  t, var, i, j, k;

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
         hypre_SStructVariableGetOffset(t, ndim, varoffset);

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
    * compute igrids with the same number of boxes as the sgrids
    *-------------------------------------------------------------*/

#if 0 /* TODO */
   for (t = 1; t < 8; t++)
   {
      if (sgrids[t] != NULL)
      {
         hypre_StructGridCreate(comm, ndim, &igrid);
         sgrid = hypre_SStructPGridVTSGrid(pgrid, t);
         hypre_SStructVariableGetOffset(t, ndim, varoffset);

         /* compute hood_boxes */
         hood = hypre_StructGridNeighbors(sgrid);
         hood_boxes = hypre_BoxArrayDuplicate(hypre_BoxNeighborsBoxes(hood));
         hypre_ForBoxI(i, hood_boxes)
            {
               /* grow the boxes */
               box = hypre_BoxArrayBox(hood_boxes, i);
               hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
               hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
               hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
               hypre_BoxIMaxX(box) += hypre_IndexX(varoffset);
               hypre_BoxIMaxY(box) += hypre_IndexY(varoffset);
               hypre_BoxIMaxZ(box) += hypre_IndexZ(varoffset);
            }

         /* set neighborhood */
         hypre_StructGridSetHood(igrid, hood_boxes,
                                 hypre_BoxNeighborsProcs(hood),
                                 hypre_BoxNeighborsIDs(hood),
                                 hypre_BoxNeighborsFirstLocal(hood),
                                 hypre_BoxNeighborsNumLocal(hood),
                                 hypre_BoxNeighborsNumPeriodic(hood),
                                 hypre_StructGridBoundingBox(sgrid));

         hypre_StructGridSetHoodInfo(igrid,
                                     hypre_StructGridMaxDistance(sgrid));
         hypre_StructGridSetPeriodic(igrid,
                                     hypre_StructGridPeriodic(sgrid));

         hypre_StructGridAssemble(igrid);

         igrids[t] = igrid;
      }
   }
#endif

   /*-------------------------------------------------------------
    * Set up the struct maps
    *-------------------------------------------------------------*/

   for (t = 0; t < 8; t++)
   {
      if (sgrids[t] != NULL)
      {
         hypre_StructMapCreate(sgrids[t], &maps[t]);
      }
   }

   /*-------------------------------------------------------------
    * Set up the offsets and start rank
    *-------------------------------------------------------------*/

   offsets = hypre_CTAlloc(int, nvars);
   offset = 0;
   start_rank = 0;
   for (var = 0; var < nvars; var++)
   {
      offsets[var] = offset;
      offset += hypre_StructGridGlobalSize(sgrids[vartypes[var]]);
      start_rank += hypre_StructMapStartRank(maps[vartypes[var]]);
   }
   hypre_SStructPGridOffsets(pgrid)   = offsets;
   hypre_SStructPGridStartRank(pgrid) = start_rank;

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
 * hypre_SStructGridIndexToBox
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridIndexToBox( hypre_SStructGrid  *grid,
                             int                 part,
                             hypre_Index         index,
                             int                 var,
                             int                *box_ptr )
{
   int  ierr = 0;

   hypre_SStructPGrid *pgrid = hypre_SStructGridPGrid(grid, part);
   hypre_StructMap    *map   = hypre_SStructPGridMap(pgrid, var);

   hypre_StructMapIndexToBox(map, index, box_ptr);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructGridSVarIndexToRank
 *--------------------------------------------------------------------------*/

int
hypre_SStructGridSVarIndexToRank( hypre_SStructGrid  *grid,
                                  int                 box,
                                  int                 part,
                                  hypre_Index         index,
                                  int                 var,
                                  int                *rank_ptr )
{
   int  ierr = 0;

   hypre_SStructPGrid *pgrid = hypre_SStructGridPGrid(grid, part);
   hypre_StructMap    *map   = hypre_SStructPGridMap(pgrid, var);

   hypre_StructMapIndexToRank(map, box, index, rank_ptr);
   *rank_ptr += hypre_SStructPGridOffset(pgrid, var);
   *rank_ptr += hypre_SStructGridOffset(grid, part);

   return ierr;
}

