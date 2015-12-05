/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.30 $
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVariableGetOffset( HYPRE_SStructVariable  vartype,
                                HYPRE_Int              ndim,
                                hypre_Index            varoffset )
{
   HYPRE_Int d;

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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridCreate( MPI_Comm             comm,
                          HYPRE_Int            ndim,
                          hypre_SStructPGrid **pgrid_ptr )
{
   hypre_SStructPGrid  *pgrid;
   hypre_StructGrid    *sgrid;
   HYPRE_Int            t;

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
   hypre_SStructPGridPNborOffsets(pgrid) = NULL;

   hypre_SStructPGridLocalSize(pgrid)  = 0;
   hypre_SStructPGridGlobalSize(pgrid) = 0;

   /* GEC0902 ghost addition to the grid    */
   hypre_SStructPGridGhlocalSize(pgrid)   = 0;
   
   hypre_ClearIndex(hypre_SStructPGridPeriodic(pgrid));

   *pgrid_ptr = pgrid;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridDestroy( hypre_SStructPGrid *pgrid )
{
   hypre_StructGrid **sgrids;
   hypre_BoxArray   **iboxarrays;
   HYPRE_Int          t;

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
      hypre_TFree(hypre_SStructPGridPNborOffsets(pgrid));
      hypre_TFree(pgrid);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridSetExtents( hypre_SStructPGrid  *pgrid,
                              hypre_Index          ilower,
                              hypre_Index          iupper )
{
   hypre_StructGrid *sgrid = hypre_SStructPGridCellSGrid(pgrid);

   HYPRE_StructGridSetExtents(sgrid, ilower, iupper);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridSetCellSGrid( hypre_SStructPGrid  *pgrid,
                                hypre_StructGrid    *cell_sgrid )
{
   hypre_SStructPGridCellSGrid(pgrid) = cell_sgrid;
   hypre_SStructPGridCellSGridDone(pgrid) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPGridSetVariables( hypre_SStructPGrid    *pgrid,
                                    HYPRE_Int              nvars,
                                    HYPRE_SStructVariable *vartypes )
{
   hypre_SStructVariable  *new_vartypes;
   HYPRE_Int               i;

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
 * Like hypre_SStructPGridSetVariables, but do one variable at a time.
 * Nevertheless, you still must provide nvars, for memory allocation.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructPGridSetVariable( hypre_SStructPGrid    *pgrid,
                                   HYPRE_Int              var,
                                   HYPRE_Int              nvars,
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridSetPNeighbor( hypre_SStructPGrid  *pgrid,
                                hypre_Box           *pneighbor_box,
                                hypre_Index          pnbor_offset )
{
   hypre_BoxArray  *pneighbors    = hypre_SStructPGridPNeighbors(pgrid);
   hypre_Index     *pnbor_offsets = hypre_SStructPGridPNborOffsets(pgrid);
   HYPRE_Int        size          = hypre_BoxArraySize(pneighbors);
   HYPRE_Int        memchunk      = 10;

   hypre_AppendBox(pneighbor_box, pneighbors);
   if ((size % memchunk) == 0)
   {
      pnbor_offsets = hypre_TReAlloc(pnbor_offsets, hypre_Index, (size + memchunk));
      hypre_SStructPGridPNborOffsets(pgrid) = pnbor_offsets;
   }
   hypre_CopyIndex(pnbor_offset, pnbor_offsets[size]);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * 11/06 AHB - modified to use the box manager
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPGridAssemble( hypre_SStructPGrid  *pgrid )
{
   MPI_Comm               comm          = hypre_SStructPGridComm(pgrid);
   HYPRE_Int              ndim          = hypre_SStructPGridNDim(pgrid);
   HYPRE_Int              nvars         = hypre_SStructPGridNVars(pgrid);
   HYPRE_SStructVariable *vartypes      = hypre_SStructPGridVarTypes(pgrid);
   hypre_StructGrid     **sgrids        = hypre_SStructPGridSGrids(pgrid);
   hypre_BoxArray       **iboxarrays    = hypre_SStructPGridIBoxArrays(pgrid);
   hypre_BoxArray        *pneighbors    = hypre_SStructPGridPNeighbors(pgrid);
   hypre_Index           *pnbor_offsets = hypre_SStructPGridPNborOffsets(pgrid);
   hypre_IndexRef         periodic      = hypre_SStructPGridPeriodic(pgrid);

   hypre_StructGrid      *cell_sgrid;
   hypre_IndexRef         cell_imax;
   hypre_StructGrid      *sgrid;
   hypre_BoxArray        *iboxarray;
   hypre_BoxManager      *boxman;
   hypre_BoxArray        *hood_boxes;
   HYPRE_Int              hood_first_local;
   HYPRE_Int              hood_num_local;
   hypre_BoxArray        *nbor_boxes;
   hypre_BoxArray        *diff_boxes;
   hypre_BoxArray        *tmp_boxes;
   hypre_BoxArray        *boxes;
   hypre_Box             *box;
   hypre_Index            varoffset;
   HYPRE_Int              pneighbors_size, vneighbors_size;

   HYPRE_Int              t, var, i, j, d, valid;

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

   /* Add one since hood_first_local can be -1 */
   nbor_boxes =
      hypre_BoxArrayCreate(pneighbors_size + hood_first_local + hood_num_local + 1);
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
         vneighbors_size = 0;
         for (i = 0; i < pneighbors_size; i++)
         {
            box = hypre_BoxArrayBox(nbor_boxes, vneighbors_size);
            hypre_CopyBox(hypre_BoxArrayBox(pneighbors, i), box);
            hypre_SStructCellBoxToVarBox(box, pnbor_offsets[i], varoffset, &valid);
            /* only add pneighbor boxes for valid variable types*/
            if (valid)
            {
               vneighbors_size++;
            }
         }
         for (i = 0; i < (hood_first_local + hood_num_local); i++)
         {
            box = hypre_BoxArrayBox(nbor_boxes, vneighbors_size + i);
            hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, i), box);
            hypre_BoxIMinX(box) -= hypre_IndexX(varoffset);
            hypre_BoxIMinY(box) -= hypre_IndexY(varoffset);
            hypre_BoxIMinZ(box) -= hypre_IndexZ(varoffset);
         }

         /* boxes = (local boxes - neighbors with smaller ID - vneighbors) */
         for (i = 0; i < hood_num_local; i++)
         {
            j = vneighbors_size + hood_first_local + i;
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridRef( hypre_SStructGrid  *grid,
                      hypre_SStructGrid **grid_ref)
{
   hypre_SStructGridRefCount(grid) ++;
   *grid_ref = grid;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This replaces hypre_SStructGridAssembleMaps
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridAssembleBoxManagers( hypre_SStructGrid *grid )
{
   MPI_Comm                   comm        = hypre_SStructGridComm(grid);
   HYPRE_Int                  nparts      = hypre_SStructGridNParts(grid);
   HYPRE_Int                  local_size  = hypre_SStructGridLocalSize(grid);
   hypre_BoxManager        ***managers;
   hypre_SStructBoxManInfo    info_obj;
   hypre_SStructPGrid        *pgrid;
   HYPRE_Int                  nvars;
   hypre_StructGrid          *sgrid;
   hypre_Box                 *bounding_box;

   HYPRE_Int                 offsets[2];

   hypre_SStructBoxManInfo   *entry_info;


   hypre_BoxManEntry         *all_entries, *entry;
   HYPRE_Int                  num_entries;
   hypre_IndexRef             entry_imin;
   hypre_IndexRef             entry_imax;

   HYPRE_Int                  nprocs, myproc, proc;
   HYPRE_Int                  part, var, b, local_ct;

   hypre_Box                 *ghostbox, *box;
   HYPRE_Int                 * num_ghost;
   HYPRE_Int                  ghoffsets[2];
   HYPRE_Int                  ghlocal_size  = hypre_SStructGridGhlocalSize(grid);

   HYPRE_Int                  info_size;
   HYPRE_Int                  box_offset, ghbox_offset;

  
   /*------------------------------------------------------
    * Build box manager info for grid boxes
    *------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &nprocs);
   hypre_MPI_Comm_rank(comm, &myproc);

  
   /*find offset and ghost offsets */
   {
      HYPRE_Int scan_recv;
      
      /* offsets */

      hypre_MPI_Scan(&local_size, &scan_recv, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      /* first point in my range */ 
      offsets[0] = scan_recv - local_size;
      /* first point in next proc's range */
      offsets[1] = scan_recv;

      hypre_SStructGridStartRank(grid) = offsets[0];

      /* ghost offsets */
      hypre_MPI_Scan(&ghlocal_size, &scan_recv, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      /* first point in my range */ 
      ghoffsets[0] = scan_recv - ghlocal_size;
      /* first point in next proc's range */
      ghoffsets[1] = scan_recv;

      hypre_SStructGridGhstartRank(grid) = ghoffsets[0];
   }

   /* allocate a box manager for each part and variable -
      copy the local box info from the underlying sgrid boxmanager*/ 

   managers = hypre_TAlloc(hypre_BoxManager **, nparts);

   /* first offsets */
   box_offset =  offsets[0];
   ghbox_offset =  ghoffsets[0];

   info_size = sizeof(hypre_SStructBoxManInfo);
 
   /* storage for the entry info is allocated and kept in the box
      manager - so here we just write over the info_obj and then
      it is copied in AddEntry */
   entry_info = &info_obj;

   /* this is the same for all the info objects */
   hypre_SStructBoxManInfoType(entry_info) =
      hypre_SSTRUCT_BOXMAN_INFO_DEFAULT;
 
   box = hypre_BoxCreate();
   ghostbox = hypre_BoxCreate();

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      managers[part] = hypre_TAlloc(hypre_BoxManager *, nvars);

      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);

         /* get all the entires from the sgrid. for the local boxes,
          * we will calculate the info and add to the box manager -
          * the rest we will gather (becuz we cannot calculate the
          * info for them */

         hypre_BoxManGetAllEntries( hypre_StructGridBoxMan(sgrid), 
                                    &num_entries, &all_entries);


         bounding_box = hypre_StructGridBoundingBox(sgrid);     


         /* need to create a box manager and then later give it the
            bounding box for gather entries call */
         
         hypre_BoxManCreate(hypre_BoxManNumMyEntries(hypre_StructGridBoxMan(sgrid)), 
                            info_size, 
                            hypre_StructGridDim(sgrid),
                            bounding_box,  
                            hypre_StructGridComm(sgrid),
                            &managers[part][var]);

	 /* each sgrid has num_ghost */

         num_ghost = hypre_StructGridNumGhost(sgrid);
         hypre_BoxManSetNumGhost(managers[part][var], num_ghost);

         /* loop through the all of the entries - for the local boxes
          * populate the info object and add to Box Manager- recall
          * that all of the boxes array belong to the calling proc */
       
         local_ct = 0;
         for (b = 0; b < num_entries; b++)
         {
            
            entry = &all_entries[b];

            proc = hypre_BoxManEntryProc(entry);

            entry_imin = hypre_BoxManEntryIMin(entry);
            entry_imax = hypre_BoxManEntryIMax(entry);
            hypre_BoxSetExtents( box, entry_imin, entry_imax );

            if (proc == myproc)
            {
               
                hypre_SStructBoxManInfoOffset(entry_info) = box_offset;
                hypre_SStructBoxManInfoGhoffset(entry_info) = ghbox_offset;
                hypre_BoxManAddEntry(managers[part][var],
                                     entry_imin,
                                     entry_imax,
                                     myproc, local_ct,
                                     entry_info);

                /* update offset */
                box_offset += hypre_BoxVolume(box);

                /* expanding box to compute expanded volume for ghost
                 * calculation */
                hypre_CopyBox(box, ghostbox);
                hypre_BoxExpand(ghostbox,num_ghost);         
                
                /* update offset */
                ghbox_offset += hypre_BoxVolume(ghostbox); 

                local_ct++;

            }
            else /* not a local box */
            {
               
               hypre_BoxManGatherEntries( managers[part][var],  
                                          entry_imin,
                                          entry_imax);
            }
         }
         
         /* call the assemble later */

      } /* end of variable loop */
   } /* end of part loop */


#if  HYPRE_NO_GLOBAL_PARTITION
   {
      /* need to do a gather entries on neighbor information so
         that we have what we need for the NborBoxManagers function */
      
      /* these neighbor boxes are much parger than the data that we
         care about, so first we need to intersect them with the grid
         and just pass the intersected box into the Box Manager */
      

      hypre_SStructNeighbor    *vneighbor;
      HYPRE_Int                 b, i;
      hypre_Box                *vbox;
      HYPRE_Int               **nvneighbors = hypre_SStructGridNVNeighbors(grid);
      hypre_SStructNeighbor  ***vneighbors  = hypre_SStructGridVNeighbors(grid);
      HYPRE_Int                *coord, *dir;
      hypre_Index               imin0, imin1;
      HYPRE_Int                 nbor_part, nbor_var;
      hypre_IndexRef            max_distance;
      hypre_Box                *grow_box;
      hypre_Box                *int_box;
      hypre_Box                *nbor_box;
      hypre_BoxManager         *box_man;
      hypre_BoxArray           *local_boxes;
     
      grow_box = hypre_BoxCreate();
      int_box = hypre_BoxCreate();
      nbor_box =  hypre_BoxCreate();

      local_boxes = hypre_BoxArrayCreate(0);

      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);

         for (var = 0; var < nvars; var++)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, var);
            max_distance = hypre_StructGridMaxDistance(sgrid);
  
            /* now loop through my boxes, grow them, and intersect
             * with all of the neighbors */
           
            box_man = hypre_StructGridBoxMan(sgrid);
            hypre_BoxManGetLocalEntriesBoxes(box_man, local_boxes);
            
            hypre_ForBoxI(i, local_boxes)
            {
               hypre_CopyBox(hypre_BoxArrayBox(local_boxes, i), grow_box); 
               hypre_BoxExpandConstantDim(grow_box, max_distance);
              
               /* loop through neighbors */
               for (b = 0; b < nvneighbors[part][var]; b++)
               {
                  vneighbor = &vneighbors[part][var][b];
                  vbox = hypre_SStructNeighborBox(vneighbor);

                  /* grow neighbor box by 1 to account for shared parts */
                  hypre_CopyBox(vbox, nbor_box);
                  hypre_BoxExpandConstant(nbor_box, 1);

                  nbor_part = hypre_SStructNeighborPart(vneighbor);
              
                  coord = hypre_SStructNeighborCoord(vneighbor);
                  dir   = hypre_SStructNeighborDir(vneighbor);
        
                  /* find intersection of neighbor and my local box */
                  hypre_IntersectBoxes(grow_box, nbor_box, int_box); 
                  if (hypre_BoxVolume(int_box) > 0)
                  {
                     hypre_CopyIndex(hypre_BoxIMin(vbox), imin0);
                     hypre_CopyIndex(hypre_SStructNeighborILower(vneighbor), imin1);

                     /* map int_box to neighbor part index space */
                     hypre_SStructBoxToNborBox(int_box, imin0, imin1, coord, dir);
                     hypre_SStructVarToNborVar(grid, part, var, coord, &nbor_var);
                    
                     hypre_BoxManGatherEntries( managers[nbor_part][nbor_var], 
                                                hypre_BoxIMin(int_box), hypre_BoxIMax(int_box));
                  }
               } /* end neighbor loop */
            } /* end local box loop */
         }
      }
      hypre_BoxDestroy(grow_box);
      hypre_BoxDestroy(nbor_box);
      hypre_BoxArrayDestroy(local_boxes);
   }
#endif


   /* now call the assembles */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         hypre_BoxManAssemble(managers[part][var]);
      }
   }
   

   hypre_BoxDestroy(ghostbox);
   hypre_BoxDestroy(box);
   

   hypre_SStructGridBoxManagers(grid) = managers;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This replaces hypre_SStructGridAssembleNborMaps
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridAssembleNborBoxManagers( hypre_SStructGrid *grid )
{
   HYPRE_Int                    nparts      = hypre_SStructGridNParts(grid);
   HYPRE_Int                  **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor     ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor       *vneighbor;
   hypre_SStructPGrid          *pgrid;
   HYPRE_Int                    nvars;
   hypre_StructGrid            *sgrid;

   hypre_BoxManager          ***nbor_managers;
   hypre_SStructBoxManNborInfo *nbor_info;
   hypre_BoxManEntry          **entries;
   HYPRE_Int                    nentries;

   hypre_BoxArrayArray         *nbor_boxes;
   hypre_BoxArray              *nbor_boxa;
   hypre_BoxArray              *sub_boxa;
   hypre_BoxArray              *tmp_boxa;
   hypre_Box                   *nbor_box, *box, *int_box, *ghbox;
   HYPRE_Int                   *coord, *dir, *box_imin;
   hypre_Index                  imin0, imin1;
   HYPRE_Int                    nbor_offset, nbor_ghoffset;
   HYPRE_Int                    nbor_proc, nbor_boxnum, nbor_part, nbor_var;
   hypre_Index                  nbor_ilower;
   HYPRE_Int                    c[3], *num_ghost, *stride, *ghstride;
   HYPRE_Int                    part, var, b, bb, i, d, sub_part, info_size;

   hypre_Box                    *bounding_box;

   /*------------------------------------------------------
    * Create a box manager for the neighbor boxes 
    *------------------------------------------------------*/
   bounding_box = hypre_BoxCreate();

   nbor_box = hypre_BoxCreate();
   box = hypre_BoxCreate();
   int_box = hypre_BoxCreate();
   ghbox = hypre_BoxCreate();
   /* nbor_info is copied into the box manager */
   nbor_info = hypre_TAlloc(hypre_SStructBoxManNborInfo, 1);

   nbor_managers = hypre_TAlloc(hypre_BoxManager **, nparts);

   info_size = sizeof(hypre_SStructBoxManNborInfo);


   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      nbor_managers[part] = hypre_TAlloc(hypre_BoxManager *, nvars);

      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);
         /*we need to know the size of the bounding box (only matter
          * for the AP case - not used in the non-AP box manager*/
#if HYPRE_NO_GLOBAL_PARTITION
         {
            MPI_Comm     comm        = hypre_SStructGridComm(grid);
            hypre_Box   *vbox;
            hypre_Index  min_index, max_index;
            HYPRE_Int    d, dim  = hypre_SStructGridNDim(grid);
            HYPRE_Int    sendbuf6[6], recvbuf6[6];
            hypre_CopyBox( hypre_StructGridBoundingBox(sgrid),bounding_box); 
            hypre_CopyToCleanIndex( hypre_BoxIMin(bounding_box), dim, min_index);
            hypre_CopyToCleanIndex( hypre_BoxIMax(bounding_box), dim, max_index);

            for (b = 0; b < nvneighbors[part][var]; b++)
            {
                
               vneighbor = &vneighbors[part][var][b];
               vbox = hypre_SStructNeighborBox(vneighbor);
               /* find min and max box extents */  
               for (d = 0; d < dim; d++)
               {
                  hypre_IndexD(min_index, d) =
                     hypre_min( hypre_IndexD(min_index, d), 
                                hypre_BoxIMinD(bounding_box, d));
                  hypre_IndexD(max_index, d) =
                     hypre_max( hypre_IndexD(max_index, d), 
                                hypre_BoxIMaxD(bounding_box, d));
               } 
            }
            /* this is based on local info - all procs need to have
             * the same bounding box!  */
            hypre_BoxSetExtents( bounding_box, min_index, max_index);
             
            /* communication needed for the bounding box */
            /* pack buffer */
            for (d = 0; d < 3; d++) 
            {
               sendbuf6[d] = hypre_BoxIMinD(bounding_box, d);
               sendbuf6[d+3] = -hypre_BoxIMaxD(bounding_box, d);
            }
            hypre_MPI_Allreduce(sendbuf6, recvbuf6, 6, HYPRE_MPI_INT, hypre_MPI_MIN, comm);
            /* unpack buffer */
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
               hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d+3];
            }
         }
#else
         hypre_CopyBox( hypre_StructGridBoundingBox(sgrid), bounding_box); 
#endif
         /* Here we want to create a new manager for the neighbor information
          * (instead of adding to the current and reassembling).  This uses a
          * lower bound for the actual box manager size. */
        
         hypre_BoxManCreate(nvneighbors[part][var], info_size,
                            hypre_StructGridDim(sgrid),
                            hypre_StructGridBoundingBox(sgrid),
                            hypre_StructGridComm(sgrid),
                            &nbor_managers[part][var]);

         /* Put the vneighbor boxes into the nbor_boxes BoxArrayArray */
         nbor_boxes = hypre_BoxArrayArrayCreate(nvneighbors[part][var]);
         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            hypre_AppendBox(hypre_SStructNeighborBox(vneighbor), nbor_boxa);
         }

         /* Make sure that the nbor_boxes don't overlap */
         tmp_boxa = hypre_BoxArrayCreate(0);
         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];
            nbor_part = hypre_SStructNeighborPart(vneighbor);
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);
            for (i = 0; i < b; i++)
            {
               vneighbor = &vneighbors[part][var][i];
               sub_part = hypre_SStructNeighborPart(vneighbor);
               /* Only subtract boxes on the same neighbor part */
               if (nbor_part == sub_part)
               {
                  sub_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, i);
                  hypre_SubtractBoxArrays(nbor_boxa, sub_boxa, tmp_boxa);
               }
            }
         }
         hypre_BoxArrayDestroy(tmp_boxa);

         /* Compute entries and add to the neighbor box manager */
         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];
            nbor_boxa = hypre_BoxArrayArrayBoxArray(nbor_boxes, b);

            nbor_part = hypre_SStructNeighborPart(vneighbor);
            hypre_CopyIndex(
               hypre_BoxIMin(hypre_SStructNeighborBox(vneighbor)), imin0);
            hypre_CopyIndex(hypre_SStructNeighborILower(vneighbor), imin1);
            coord = hypre_SStructNeighborCoord(vneighbor);
            dir   = hypre_SStructNeighborDir(vneighbor);

            hypre_ForBoxI(bb, nbor_boxa)
            {
               /* Intersect neighbor boxes with appropriate PGrid */
               hypre_CopyBox(hypre_BoxArrayBox(nbor_boxa, bb), nbor_box);

               /* map to neighbor part index space */
               hypre_SStructBoxToNborBox(nbor_box, imin0, imin1, coord, dir);
               hypre_SStructVarToNborVar(grid, part, var, coord, &nbor_var);

               hypre_SStructGridIntersect(grid, nbor_part, nbor_var, nbor_box, 0,
                                          &entries, &nentries);

               for (i = 0; i < nentries; i++)
               {
                  hypre_BoxManEntryGetExtents(
                     entries[i], hypre_BoxIMin(box), hypre_BoxIMax(box));
                  hypre_IntersectBoxes(nbor_box, box, int_box);

                  /* map back from neighbor part index space */
                  hypre_SStructNborBoxToBox(int_box, imin0, imin1, coord, dir);

                  box_imin = hypre_BoxIMin(int_box);
                  for (d = 0; d < 3; d++)
                  {
                     nbor_ilower[coord[d]] =
                        imin1[coord[d]] + (box_imin[d] - imin0[d]) * dir[d];
                  }

                  hypre_SStructBoxManEntryGetProcess(entries[i], &nbor_proc);
                  hypre_SStructBoxManEntryGetBoxnum(entries[i], &nbor_boxnum);
                  hypre_SStructBoxManEntryGetGlobalCSRank(
                     entries[i], nbor_ilower, &nbor_offset);
                  hypre_SStructBoxManEntryGetGlobalGhrank(
                     entries[i], nbor_ilower, &nbor_ghoffset);
                  num_ghost = hypre_BoxManEntryNumGhost(entries[i]);

                  /* Set up the neighbor info. */
                  hypre_SStructBoxManInfoType(nbor_info) =
                     hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR;
                  hypre_SStructBoxManInfoOffset(nbor_info) = nbor_offset;
                  hypre_SStructBoxManInfoGhoffset(nbor_info) = nbor_ghoffset;
                  hypre_SStructBoxManNborInfoProc(nbor_info) = nbor_proc;
                  hypre_SStructBoxManNborInfoBoxnum(nbor_info) = nbor_boxnum;
                  hypre_SStructBoxManNborInfoPart(nbor_info) = nbor_part;
                  hypre_CopyIndex(
                     nbor_ilower, hypre_SStructBoxManNborInfoILower(nbor_info));
                  hypre_CopyIndex(
                     coord, hypre_SStructBoxManNborInfoCoord(nbor_info));
                  hypre_CopyIndex(
                     dir, hypre_SStructBoxManNborInfoDir(nbor_info));
                  /* This computes strides in the local index-space, so they
                   * may be negative.  Want `c' to map from the neighbor
                   * index-space back. */
                  c[coord[0]] = 0;
                  c[coord[1]] = 1;
                  c[coord[2]] = 2;
                  stride = hypre_SStructBoxManNborInfoStride(nbor_info);
                  stride[c[0]] = 1;
                  stride[c[1]] = hypre_BoxSizeD(box, 0);
                  stride[c[2]] = hypre_BoxSizeD(box, 1) * stride[c[1]];
                  stride[c[0]] *= dir[c[0]];
                  stride[c[1]] *= dir[c[1]];
                  stride[c[2]] *= dir[c[2]];
                  hypre_CopyBox(box, ghbox);
                  hypre_BoxExpand(ghbox, num_ghost);
                  ghstride = hypre_SStructBoxManNborInfoGhstride(nbor_info);
                  ghstride[c[0]] = 1;
                  ghstride[c[1]] = hypre_BoxSizeD(ghbox, 0);
                  ghstride[c[2]] = hypre_BoxSizeD(ghbox, 1) * ghstride[c[1]];
                  ghstride[c[0]] *= dir[c[0]];
                  ghstride[c[1]] *= dir[c[1]];
                  ghstride[c[2]] *= dir[c[2]];

                  /* Here the ids need to be unique.  Cannot use the boxnum.
                     A negative number lets the box manager assign the id. */
                  hypre_BoxManAddEntry(nbor_managers[part][var],
                                       hypre_BoxIMin(int_box),
                                       hypre_BoxIMax(int_box),
                                       nbor_proc, -1, nbor_info);

               } /* end of entries loop */

               hypre_TFree(entries);

            } /* end of ForBoxI loop */

         } /* end of vneighbor box loop */

         hypre_BoxArrayArrayDestroy(nbor_boxes);
         hypre_BoxManAssemble(nbor_managers[part][var]);

      } /* end of variables loop */

   } /* end of part loop */
   
   hypre_SStructGridNborBoxManagers(grid) = nbor_managers;

   hypre_TFree(nbor_info);
   hypre_BoxDestroy(nbor_box);
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(ghbox);

   hypre_BoxDestroy(bounding_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine computes the inter-part communication information for updating
 * shared variable data.
 *
 * It grows each local box according to vartype and intersects with the BoxManager
 * to get map entries.  Then, for each of the neighbor-type entries, it grows
 * either the local box or the neighbor box based on which one is the "owner"
 * (the part number determines this).
 *
 * NEW Approach
 *
 * Loop over the vneighbor boxes.  Let pi = my part and pj = vneighbor part.
 * The part with the smaller ID owns the data, so (pi < pj) means that shared
 * vneighbor data overlaps with pi's data and pj's ghost, and (pi > pj) means
 * that shared vneighbor data overlaps with pj's data and pi's ghost.
 *
 * Intersect each vneighbor box with the BoxManager for the owner part (either
 * pi or pj) and intersect a grown vneighbor box with the BoxManager for the
 * non-owner part.  This produces two lists of boxes on the two different parts
 * that share data.  The remainder of the routine loops over these two lists,
 * intersecting the boxes appropriately with the vneighbor box to determine send
 * and receive communication info.  For convenience, the information is put into
 * a 4D "matrix" based on pi, pj, vi (variable on part pi), and vj.  The upper
 * "triangle" (given by pi < pj) stores the send information and the lower
 * triangle stores the receive information.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridCreateCommInfo( hypre_SStructGrid  *grid )
{
   HYPRE_Int                ndim = hypre_SStructGridNDim(grid);
   HYPRE_Int                nparts = hypre_SStructGridNParts(grid);
   hypre_SStructPGrid     **pgrids = hypre_SStructGridPGrids(grid);
   HYPRE_Int              **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor ***vneighbors  = hypre_SStructGridVNeighbors(grid);
   hypre_SStructNeighbor   *vneighbor;
   hypre_SStructCommInfo  **vnbor_comm_info;
   HYPRE_Int                vnbor_ncomms;
   hypre_SStructCommInfo   *comm_info;
   HYPRE_SStructVariable   *vartypes;
   hypre_Index              varoffset;

   typedef struct
   {
      hypre_BoxArrayArray    *boxes;
      hypre_BoxArrayArray    *rboxes;
      HYPRE_Int             **procs;
      HYPRE_Int             **rboxnums;
      HYPRE_Int             **transforms;
      HYPRE_Int              *num_transforms; /* reference to num transforms */
      hypre_Index            *coords;
      hypre_Index            *dirs;

   } CInfo;

   hypre_IndexRef           coord, dir;

   CInfo                  **cinfo_a;  /* array of size (nparts^2)(maxvars^2) */
   CInfo                   *cinfo, *send_cinfo, *recv_cinfo;
   HYPRE_Int                cinfoi, cinfoj, maxvars;
   hypre_BoxArray          *cbox_a;
   hypre_BoxArray          *crbox_a;
   HYPRE_Int               *cproc_a;
   HYPRE_Int               *crboxnum_a;
   HYPRE_Int               *ctransform_a;
   HYPRE_Int               *cnum_transforms;
   hypre_Index             *ccoords;
   hypre_Index             *cdirs;

   hypre_SStructPGrid      *pgrid;
   hypre_BoxManager        *boxman;

   hypre_BoxManEntry      **pi_entries, **pj_entries;
   hypre_BoxManEntry       *pi_entry,    *pj_entry;
   HYPRE_Int                npi_entries,  npj_entries;

   hypre_Box               *vn_box, *pi_box, *pj_box, *int_box, *int_rbox;
   hypre_Index              imin0, imin1;

   HYPRE_Int                nvars, size, pi_proc, myproc;
   HYPRE_Int                pi, pj, vi, vj, ei, ej, ni, bi, ti;

   hypre_MPI_Comm_rank(hypre_SStructGridComm(grid), &myproc);

   vn_box = hypre_BoxCreate();
   pi_box = hypre_BoxCreate();
   pj_box = hypre_BoxCreate();
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
   /* for each part */
   for (pi = 0; pi < nparts; pi++)
   {
      pgrid  = pgrids[pi];
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);

      /* for each variable */
      for (vi = 0; vi < nvars; vi++)
      {
         hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);

         /* for each vneighbor box */
         for (ni = 0; ni < nvneighbors[pi][vi]; ni++)
         {
            vneighbor = &vneighbors[pi][vi][ni];
            hypre_CopyIndex(
               hypre_BoxIMin(hypre_SStructNeighborBox(vneighbor)), imin0);
            hypre_CopyIndex(hypre_SStructNeighborILower(vneighbor), imin1);
            coord = hypre_SStructNeighborCoord(vneighbor);
            dir   = hypre_SStructNeighborDir(vneighbor);

            pj = hypre_SStructNeighborPart(vneighbor);
            hypre_SStructVarToNborVar(grid, pi, vi, coord, &vj);

            /* intersect with boxman for part pi */

            boxman = hypre_SStructGridBoxManager(grid, pi, vi);
            hypre_CopyBox(hypre_SStructNeighborBox(vneighbor), vn_box);
            /* if pi is not the owner, grow the vneighbor box */
            if (pi > pj)
            {
               hypre_BoxIMinX(vn_box) -= hypre_IndexX(varoffset);
               hypre_BoxIMinY(vn_box) -= hypre_IndexY(varoffset);
               hypre_BoxIMinZ(vn_box) -= hypre_IndexZ(varoffset);
               hypre_BoxIMaxX(vn_box) += hypre_IndexX(varoffset);
               hypre_BoxIMaxY(vn_box) += hypre_IndexY(varoffset);
               hypre_BoxIMaxZ(vn_box) += hypre_IndexZ(varoffset);
            }
            hypre_SStructGridIntersect(grid, pi, vi, vn_box, 0,
                                       &pi_entries, &npi_entries);

            /* intersect with boxman for part pj */

            boxman = hypre_SStructGridBoxManager(grid, pj, vj);
            hypre_CopyBox(hypre_SStructNeighborBox(vneighbor), vn_box);
            /* if pj is not the owner, grow the vneighbor box */
            if (pj > pi)
            {
               hypre_BoxIMinX(vn_box) -= hypre_IndexX(varoffset);
               hypre_BoxIMinY(vn_box) -= hypre_IndexY(varoffset);
               hypre_BoxIMinZ(vn_box) -= hypre_IndexZ(varoffset);
               hypre_BoxIMaxX(vn_box) += hypre_IndexX(varoffset);
               hypre_BoxIMaxY(vn_box) += hypre_IndexY(varoffset);
               hypre_BoxIMaxZ(vn_box) += hypre_IndexZ(varoffset);
            }
            /* map vneighbor box to part pj index space */
            hypre_SStructBoxToNborBox(vn_box, imin0, imin1, coord, dir);
            hypre_SStructGridIntersect(grid, pj, vj, vn_box, 0,
                                       &pj_entries, &npj_entries);

            /* loop over pi and pj entries */
            for (ei = 0; ei < npi_entries; ei++)
            {
               pi_entry = pi_entries[ei];
               /* only concerned by pi boxes on my processor */
               hypre_SStructBoxManEntryGetProcess(pi_entry, &pi_proc);
               if (pi_proc != myproc)
               {
                  continue;
               }
               hypre_BoxManEntryGetExtents(
                  pi_entry, hypre_BoxIMin(pi_box), hypre_BoxIMax(pi_box));

               /* if pi is not the owner, grow pi_box to compute recv boxes */
               if (pi > pj)
               {
                  hypre_BoxIMinX(pi_box) -= hypre_IndexX(varoffset);
                  hypre_BoxIMinY(pi_box) -= hypre_IndexY(varoffset);
                  hypre_BoxIMinZ(pi_box) -= hypre_IndexZ(varoffset);
                  hypre_BoxIMaxX(pi_box) += hypre_IndexX(varoffset);
                  hypre_BoxIMaxY(pi_box) += hypre_IndexY(varoffset);
                  hypre_BoxIMaxZ(pi_box) += hypre_IndexZ(varoffset);
               }

               for (ej = 0; ej < npj_entries; ej++)
               {
                  pj_entry = pj_entries[ej];
                  hypre_BoxManEntryGetExtents(
                     pj_entry, hypre_BoxIMin(pj_box), hypre_BoxIMax(pj_box));
                  /* map pj_box to part pi index space */
                  hypre_SStructNborBoxToBox(pj_box, imin0, imin1, coord, dir);

                  /* if pj is not the owner, grow pj_box to compute send boxes */
                  if (pj > pi)
                  {
                     hypre_BoxIMinX(pj_box) -= hypre_IndexX(varoffset);
                     hypre_BoxIMinY(pj_box) -= hypre_IndexY(varoffset);
                     hypre_BoxIMinZ(pj_box) -= hypre_IndexZ(varoffset);
                     hypre_BoxIMaxX(pj_box) += hypre_IndexX(varoffset);
                     hypre_BoxIMaxY(pj_box) += hypre_IndexY(varoffset);
                     hypre_BoxIMaxZ(pj_box) += hypre_IndexZ(varoffset);
                  }

                  /* intersect the vneighbor box with the pi and pj boxes */
                  hypre_CopyBox(hypre_SStructNeighborBox(vneighbor), int_box);
                  hypre_IntersectBoxes(int_box, pi_box, int_box);
                  hypre_IntersectBoxes(int_box, pj_box, int_box);

                  /* if there is an intersection, compute communication info */
                  if (hypre_BoxVolume(int_box))
                  {
                     cinfoi = (((pi)*maxvars + vi)*nparts + pj)*maxvars + vj;
                     cinfoj = (((pj)*maxvars + vj)*nparts + pi)*maxvars + vi;

                     /* allocate CommInfo arguments if needed */
                     if (cinfo_a[cinfoi] == NULL)
                     {
                        HYPRE_Int  i_num_boxes = hypre_StructGridNumBoxes(
                           hypre_SStructPGridSGrid(pgrids[pi], vi));
                        HYPRE_Int  j_num_boxes = hypre_StructGridNumBoxes(
                           hypre_SStructPGridSGrid(pgrids[pj], vj));

                        cnum_transforms = hypre_CTAlloc(HYPRE_Int, 1);
                        ccoords = hypre_CTAlloc(hypre_Index, nvneighbors[pi][vi]);
                        cdirs   = hypre_CTAlloc(hypre_Index, nvneighbors[pi][vi]);

                        cinfo = hypre_TAlloc(CInfo, 1);
                        (cinfo->boxes) = hypre_BoxArrayArrayCreate(i_num_boxes);
                        (cinfo->rboxes) = hypre_BoxArrayArrayCreate(i_num_boxes);
                        (cinfo->procs) = hypre_CTAlloc(HYPRE_Int *, i_num_boxes);
                        (cinfo->rboxnums) = hypre_CTAlloc(HYPRE_Int *, i_num_boxes);
                        (cinfo->transforms) = hypre_CTAlloc(HYPRE_Int *, i_num_boxes);
                        (cinfo->num_transforms) = cnum_transforms;
                        (cinfo->coords) = ccoords;
                        (cinfo->dirs) = cdirs;
                        cinfo_a[cinfoi] = cinfo;

                        cinfo = hypre_TAlloc(CInfo, 1);
                        (cinfo->boxes) = hypre_BoxArrayArrayCreate(j_num_boxes);
                        (cinfo->rboxes) = hypre_BoxArrayArrayCreate(j_num_boxes);
                        (cinfo->procs) = hypre_CTAlloc(HYPRE_Int *, j_num_boxes);
                        (cinfo->rboxnums) = hypre_CTAlloc(HYPRE_Int *, j_num_boxes);
                        (cinfo->transforms) = hypre_CTAlloc(HYPRE_Int *, j_num_boxes);
                        (cinfo->num_transforms) = cnum_transforms;
                        (cinfo->coords) = ccoords;
                        (cinfo->dirs) = cdirs;
                        cinfo_a[cinfoj] = cinfo;

                        vnbor_ncomms++;
                     }

                     cinfo = cinfo_a[cinfoi];

                     hypre_SStructBoxManEntryGetBoxnum(pi_entry, &bi);

                     cbox_a = hypre_BoxArrayArrayBoxArray((cinfo->boxes), bi);
                     crbox_a = hypre_BoxArrayArrayBoxArray((cinfo->rboxes), bi);
                     size = hypre_BoxArraySize(cbox_a);
                     if (size == 0)
                     {
                        /* use npj_entries as an upper bound */
                        (cinfo->procs[bi]) = hypre_CTAlloc(HYPRE_Int, npj_entries);
                        (cinfo->rboxnums[bi]) = hypre_CTAlloc(HYPRE_Int, npj_entries);
                        (cinfo->transforms[bi]) = hypre_CTAlloc(HYPRE_Int, npj_entries);
                     }
                     cproc_a = (cinfo->procs[bi]);
                     crboxnum_a = (cinfo->rboxnums[bi]);
                     ctransform_a = (cinfo->transforms[bi]);
                     cnum_transforms = (cinfo->num_transforms);
                     ccoords = (cinfo->coords);
                     cdirs = (cinfo->dirs);

                     /* map intersection box to part pj index space */
                     hypre_CopyBox(int_box, int_rbox);
                     hypre_SStructBoxToNborBox(int_rbox, imin0, imin1, coord, dir);
                           
                     hypre_AppendBox(int_box, cbox_a);
                     hypre_AppendBox(int_rbox, crbox_a);
                     hypre_SStructBoxManEntryGetProcess(pj_entry, &cproc_a[size]);
                     hypre_SStructBoxManEntryGetBoxnum(pj_entry, &crboxnum_a[size]);
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
                     ctransform_a[size] = ti;

                  } /* end of if intersection box */
               } /* end of ej entries loop */
            } /* end of ei entries loop */
            hypre_TFree(pj_entries);
            hypre_TFree(pi_entries);
         } /* end of ni vneighbor box loop */
      } /* end of vi variable loop */
   } /* end of pi part loop */

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
                     (send_cinfo->boxes), (recv_cinfo->boxes),
                     (send_cinfo->procs), (recv_cinfo->procs),
                     (send_cinfo->rboxnums), (recv_cinfo->rboxnums),
                     (send_cinfo->rboxes), (recv_cinfo->rboxes),
                     0, &hypre_SStructCommInfoCommInfo(comm_info));
                  hypre_CommInfoSetTransforms(
                     hypre_SStructCommInfoCommInfo(comm_info),
                     *(send_cinfo->num_transforms),
                     (send_cinfo->coords), (send_cinfo->dirs),
                     (send_cinfo->transforms), (recv_cinfo->transforms));
                  hypre_TFree(send_cinfo->num_transforms);
                  
                  hypre_SStructCommInfoSendPart(comm_info) = pi;
                  hypre_SStructCommInfoRecvPart(comm_info) = pj;
                  hypre_SStructCommInfoSendVar(comm_info) = vi;
                  hypre_SStructCommInfoRecvVar(comm_info) = vj;
                  
                  vnbor_comm_info[vnbor_ncomms] = comm_info;
#if 0
{
   /* debugging print */
   hypre_BoxArrayArray *boxaa;
   hypre_BoxArray      *boxa;
   hypre_Box           *box;
   HYPRE_Int            i, j, **procs, **rboxs;

   boxaa = (comm_info->comm_info->send_boxes);
   procs = (comm_info->comm_info->send_processes);
   rboxs = (comm_info->comm_info->send_rboxnums);
   hypre_ForBoxArrayI(i, boxaa)
   {
      hypre_printf("%d: ncomm = %d, send box = %d, (proc, rbox) =",
             myproc, vnbor_ncomms, i);
      boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
      hypre_ForBoxI(j, boxa)
      {
         box = hypre_BoxArrayBox(boxa, j);
         hypre_printf(" (%d,%d; %d,%d,%d x %d,%d,%d)", procs[i][j], rboxs[i][j],
                hypre_BoxIMinX(box), hypre_BoxIMinY(box), hypre_BoxIMinZ(box),
                hypre_BoxIMaxX(box), hypre_BoxIMaxY(box), hypre_BoxIMaxZ(box));
      }
      hypre_printf("\n");
   }
   boxaa = (comm_info->comm_info->recv_boxes);
   procs = (comm_info->comm_info->recv_processes);
   rboxs = (comm_info->comm_info->recv_rboxnums);
   hypre_ForBoxArrayI(i, boxaa)
   {
      hypre_printf("%d: ncomm = %d, recv box = %d, (proc, rbox) =",
             myproc, vnbor_ncomms, i);
      boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
      hypre_ForBoxI(j, boxa)
      {
         box = hypre_BoxArrayBox(boxa, j);
         hypre_printf(" (%d,%d; %d,%d,%d x %d,%d,%d)", procs[i][j], rboxs[i][j],
                hypre_BoxIMinX(box), hypre_BoxIMinY(box), hypre_BoxIMinZ(box),
                hypre_BoxIMaxX(box), hypre_BoxIMaxY(box), hypre_BoxIMaxZ(box));
      }
      hypre_printf("\n");
   }
   fflush(stdout);
}
#endif
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
   hypre_BoxDestroy(vn_box);
   hypre_BoxDestroy(pi_box);
   hypre_BoxDestroy(pj_box);
   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(int_rbox);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine returns a NULL 'entry_ptr' if an entry is not found
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridFindBoxManEntry( hypre_SStructGrid  *grid,
                                  HYPRE_Int           part,
                                  hypre_Index         index,
                                  HYPRE_Int           var,
                                  hypre_BoxManEntry **entry_ptr )
{

   HYPRE_Int nentries;

   hypre_BoxManEntry **entries;
   
   hypre_BoxManIntersect (  hypre_SStructGridBoxManager(grid, part, var),
                            index, 
                            index,
                            &entries, 
                            &nentries);

   /* we should only get a single entry returned */
   if (nentries > 1)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      *entry_ptr = NULL;
   }
   else if (nentries == 0)
   {
      *entry_ptr = NULL;
   }
   else
   {
      *entry_ptr = entries[0];
      /* remove the entries array (allocated in the intersect routine) */
      hypre_TFree(entries); 
   }
   
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridFindNborBoxManEntry( hypre_SStructGrid  *grid,
                                      HYPRE_Int           part,
                                      hypre_Index         index,
                                      HYPRE_Int           var,
                                      hypre_BoxManEntry **entry_ptr )
{

   HYPRE_Int nentries;

   hypre_BoxManEntry **entries;
   

   hypre_BoxManIntersect (  hypre_SStructGridNborBoxManager(grid, part, var),
                            index, 
                            index,
                            &entries, 
                            &nentries);

   /* we should only get a single entry returned */
   if (nentries >  1)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      *entry_ptr = NULL;
   }
   else if (nentries == 0)
   {
      *entry_ptr = NULL;
   }
   else
   {
      *entry_ptr = entries[0];
      hypre_TFree(entries); 
   }
   
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridBoxProcFindBoxManEntry( hypre_SStructGrid  *grid,
                                         HYPRE_Int           part,
                                         HYPRE_Int           var,
                                         HYPRE_Int           box,
                                         HYPRE_Int           proc,
                                         hypre_BoxManEntry **entry_ptr )
{
   hypre_BoxManGetEntry(hypre_SStructGridBoxManager(grid, part, var),
                        proc, box, entry_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetCSRstrides(  hypre_BoxManEntry *entry,
                                        hypre_Index        strides )
{
   hypre_SStructBoxManInfo *entry_info;

   hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);

   if (hypre_SStructBoxManInfoType(entry_info) == hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
   {
      hypre_Index  imin;
      hypre_Index  imax;

      hypre_BoxManEntryGetExtents(entry, imin, imax);

      strides[0] = 1;
      strides[1] = hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1;
      strides[2] = hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1;
      strides[2] *= strides[1];
   }
   else
   {
      hypre_SStructBoxManNborInfo *entry_ninfo;

      entry_ninfo = (hypre_SStructBoxManNborInfo *) entry_info;

      hypre_CopyIndex(hypre_SStructBoxManNborInfoStride(entry_ninfo), strides);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 addition for a ghost stride calculation
 * same function except that you modify imin, imax with the ghost and
 * when the info is type nmapinfo you pull the ghoststrides.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetGhstrides( hypre_BoxManEntry *entry,
                                      hypre_Index        strides )
{
   hypre_SStructBoxManInfo *entry_info;
   HYPRE_Int               *numghost;
   HYPRE_Int                d ;

   hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);

   if (hypre_SStructBoxManInfoType(entry_info) == hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
   {
      hypre_Index  imin;
      hypre_Index  imax;

      hypre_BoxManEntryGetExtents(entry, imin, imax);

      /* getting the ghost from the mapentry to modify imin, imax */

      numghost = hypre_BoxManEntryNumGhost(entry);

      for (d = 0; d < 3; d++)
      { 
	imax[d] += numghost[2*d+1];
        imin[d] -= numghost[2*d];
      }  

      /* imin, imax modified now and calculation identical.  */

      strides[0] = 1;
      strides[1] = hypre_IndexD(imax, 0) - hypre_IndexD(imin, 0) + 1;
      strides[2] = hypre_IndexD(imax, 1) - hypre_IndexD(imin, 1) + 1;
      strides[2] *= strides[1];
   }
   else
   {
      hypre_SStructBoxManNborInfo *entry_ninfo;
      /* now get the ghost strides using the macro   */
      entry_ninfo = (hypre_SStructBoxManNborInfo *) entry_info;
      hypre_CopyIndex(hypre_SStructBoxManNborInfoGhstride(entry_ninfo), strides);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetGlobalCSRank(  hypre_BoxManEntry *entry,
                                          hypre_Index        index,
                                          HYPRE_Int         *rank_ptr )
{
   hypre_SStructBoxManInfo *entry_info;
   hypre_Index              imin;
   hypre_Index              imax;
   hypre_Index              strides;
   HYPRE_Int                offset;

   hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxManEntryGetExtents(entry, imin, imax);
   offset = hypre_SStructBoxManInfoOffset(entry_info);

   hypre_SStructBoxManEntryGetCSRstrides(entry, strides);

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

HYPRE_Int
hypre_SStructBoxManEntryGetGlobalGhrank( hypre_BoxManEntry *entry,
                                         hypre_Index        index,
                                         HYPRE_Int         *rank_ptr )
{
   hypre_SStructBoxManInfo  *entry_info;
   hypre_Index               imin;
   hypre_Index               imax;
   hypre_Index               ghstrides;
   HYPRE_Int                 ghoffset;
   HYPRE_Int                 *numghost = hypre_BoxManEntryNumGhost(entry);
   HYPRE_Int                 d;
   HYPRE_Int                 info_type;
   
   hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);
   hypre_BoxManEntryGetExtents(entry, imin, imax);
   ghoffset = hypre_SStructBoxManInfoGhoffset(entry_info);
   info_type = hypre_SStructBoxManInfoType(entry_info);

   hypre_SStructBoxManEntryGetGhstrides(entry, ghstrides);

   /* GEC shifting the imin according to the ghosts when you have a default info
    * When you have a neighbor info, you do not need to shift the imin since
    * the ghoffset for neighbor info has factored in the ghost presence during
    * the neighbor info assemble phase   */

   if (info_type == hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
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

HYPRE_Int
hypre_SStructBoxManEntryGetProcess( hypre_BoxManEntry *entry,
                                 HYPRE_Int         *proc_ptr )
{

   *proc_ptr = hypre_BoxManEntryProc(entry);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * For neighbors, the boxnum is in the info, otherwise it is the same
 * as the id.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetBoxnum( hypre_BoxManEntry *entry,
                                   HYPRE_Int         *id_ptr )
{

   hypre_SStructBoxManNborInfo *info;

   hypre_BoxManEntryGetInfo(entry, (void **) &info);
   
   if (hypre_SStructBoxManInfoType(info) ==
       hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR)
   /* get from the info object */
   {
      *id_ptr = hypre_SStructBoxManNborInfoBoxnum(info);
   }
   else /* use id from the entry */
   {
      *id_ptr = hypre_BoxManEntryId(entry);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetPart( hypre_BoxManEntry *entry,
                                 HYPRE_Int          part,
                                 HYPRE_Int         *part_ptr )
{
   hypre_SStructBoxManNborInfo *info;

   hypre_BoxManEntryGetInfo(entry, (void **) &info);
   
   if (hypre_SStructBoxManInfoType(info) == hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR)
   {
      *part_ptr = hypre_SStructBoxManNborInfoPart(info);
   }
   else
   {
      *part_ptr = part;
   }

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

HYPRE_Int
hypre_SStructBoxToNborBox( hypre_Box   *box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   HYPRE_Int *imin = hypre_BoxIMin(box);
   HYPRE_Int *imax = hypre_BoxIMax(box);
   HYPRE_Int  nbor_imin[3];
   HYPRE_Int  nbor_imax[3];

   HYPRE_Int  d, nd;

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

HYPRE_Int
hypre_SStructNborBoxToBox( hypre_Box   *nbor_box,
                           hypre_Index  index,
                           hypre_Index  nbor_index,
                           hypre_Index  coord,
                           hypre_Index  dir )
{
   HYPRE_Int *nbor_imin = hypre_BoxIMin(nbor_box);
   HYPRE_Int *nbor_imax = hypre_BoxIMax(nbor_box);
   HYPRE_Int  imin[3];
   HYPRE_Int  imax[3];

   HYPRE_Int  d, nd;

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

HYPRE_Int
hypre_SStructVarToNborVar( hypre_SStructGrid  *grid,
                           HYPRE_Int           part,
                           HYPRE_Int           var,
                           HYPRE_Int          *coord,
                           HYPRE_Int          *nbor_var_ptr)
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
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructGridSetNumGhost( hypre_SStructGrid  *grid, HYPRE_Int *num_ghost )
{
   HYPRE_Int             nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int             nvars ;
   HYPRE_Int             part  ;
   HYPRE_Int             var  ;
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetGlobalRank( hypre_BoxManEntry *entry,
                                       hypre_Index        index,
                                       HYPRE_Int          *rank_ptr,
                                       HYPRE_Int          type)
{
   if (type == HYPRE_PARCSR)
   {
      hypre_SStructBoxManEntryGetGlobalCSRank(entry,index, rank_ptr);
   }
   if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
   {
      hypre_SStructBoxManEntryGetGlobalGhrank(entry, index, rank_ptr);
   }

   return hypre_error_flag;
}  

/*--------------------------------------------------------------------------
 * GEC1002 a function that will select the right way to calculate the strides
 * depending on the matrix type. It is an extension to the usual strides
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxManEntryGetStrides(hypre_BoxManEntry   *entry,
                                   hypre_Index          strides,
                                   HYPRE_Int           type)
{
   if (type == HYPRE_PARCSR)
   {
      hypre_SStructBoxManEntryGetCSRstrides(entry, strides);
   }
   if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT)
   {
      hypre_SStructBoxManEntryGetGhstrides(entry, strides);
   }

   return hypre_error_flag;
}  

/*--------------------------------------------------------------------------
 *  A function to determine the local variable box numbers that underlie
 *  a cellbox with local box number boxnum. Only returns local box numbers
 *  of myproc.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructBoxNumMap(hypre_SStructGrid        *grid,
                       HYPRE_Int                 part,
                       HYPRE_Int                 boxnum,
                       HYPRE_Int               **num_varboxes_ptr,
                       HYPRE_Int              ***map_ptr)
{
   hypre_SStructPGrid    *pgrid   = hypre_SStructGridPGrid(grid, part);
   hypre_StructGrid      *cellgrid= hypre_SStructPGridCellSGrid(pgrid);
   hypre_StructGrid      *vargrid;
   hypre_BoxArray        *boxes;
   hypre_Box             *cellbox, vbox, *box, intersect_box;
   HYPRE_SStructVariable *vartypes= hypre_SStructPGridVarTypes(pgrid);

   HYPRE_Int              ndim    = hypre_SStructGridNDim(grid);
   HYPRE_Int              nvars   = hypre_SStructPGridNVars(pgrid);
   hypre_Index            varoffset;

   HYPRE_Int             *num_boxes;
   HYPRE_Int            **var_boxnums;
   HYPRE_Int             *temp;

   HYPRE_Int              i, j, k, var;

   cellbox= hypre_StructGridBox(cellgrid, boxnum);

  /* ptrs to store var_box map info */
   num_boxes  = hypre_CTAlloc(HYPRE_Int, nvars);
   var_boxnums= hypre_TAlloc(HYPRE_Int *, nvars);

  /* intersect the cellbox with the var_boxes */
   for (var= 0; var< nvars; var++)
   {
      vargrid= hypre_SStructPGridSGrid(pgrid, var);
      boxes  = hypre_StructGridBoxes(vargrid);
      temp   = hypre_CTAlloc(HYPRE_Int, hypre_BoxArraySize(boxes));

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
         var_boxnums[var]= hypre_TAlloc(HYPRE_Int, num_boxes[var]);
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

HYPRE_Int
hypre_SStructCellGridBoxNumMap(hypre_SStructGrid        *grid,
                               HYPRE_Int                 part,
                               HYPRE_Int              ***num_varboxes_ptr,
                               HYPRE_Int             ****map_ptr)
{
   hypre_SStructPGrid    *pgrid    = hypre_SStructGridPGrid(grid, part);
   hypre_StructGrid      *cellgrid = hypre_SStructPGridCellSGrid(pgrid);
   hypre_BoxArray        *cellboxes= hypre_StructGridBoxes(cellgrid);
   
   HYPRE_Int            **num_boxes;
   HYPRE_Int           ***var_boxnums;

   HYPRE_Int              i, ncellboxes;

   ncellboxes = hypre_BoxArraySize(cellboxes);

   num_boxes  = hypre_TAlloc(HYPRE_Int *, ncellboxes);
   var_boxnums= hypre_TAlloc(HYPRE_Int **, ncellboxes);

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

/*--------------------------------------------------------------------------
 * Converts a cell-based box with offset to a variable-based box.  The argument
 * valid is a boolean that specifies the status of the conversion.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructCellBoxToVarBox( hypre_Box   *box,
                              hypre_Index  offset,
                              hypre_Index  varoffset,
                              HYPRE_Int   *valid )
{
   hypre_IndexRef imin = hypre_BoxIMin(box);
   hypre_IndexRef imax = hypre_BoxIMax(box);
   HYPRE_Int  d, off;

   *valid = 1;
   for (d = 0; d < 3; d++)
   {
      off = hypre_IndexD(offset, d);
      if ( (hypre_IndexD(varoffset, d) == 0) && (off != 0) )
      {
         *valid = 0;
         break;
      }
      if (off < 0)
      {
         hypre_IndexD(imin, d) -= 1;
         hypre_IndexD(imax, d) -= 1;
      }
      else if (off == 0)
      {
         hypre_IndexD(imin, d) -= hypre_IndexD(varoffset, d);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Intersects with either the grid's boxman or neighbor boxman.
 *
 * action = 0   intersect only with my box manager
 * action = 1   intersect only with my neighbor box manager
 * action < 0   intersect with both box managers
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGridIntersect( hypre_SStructGrid   *grid,
                            HYPRE_Int            part,
                            HYPRE_Int            var,
                            hypre_Box           *box,
                            HYPRE_Int            action,
                            hypre_BoxManEntry ***entries_ptr, 
                            HYPRE_Int           *nentries_ptr )
{
   hypre_BoxManEntry **entries, **tentries;
   HYPRE_Int           nentries, ntentries, i;
   hypre_BoxManager   *boxman;

   if (action < 0)
   {
      boxman = hypre_SStructGridBoxManager(grid, part, var);
      hypre_BoxManIntersect(boxman, hypre_BoxIMin(box), hypre_BoxIMax(box),
                            &entries, &nentries);
      boxman = hypre_SStructGridNborBoxManager(grid, part, var);
      hypre_BoxManIntersect(boxman, hypre_BoxIMin(box), hypre_BoxIMax(box),
                            &tentries, &ntentries);
      entries = hypre_TReAlloc(entries, hypre_BoxManEntry *,
                               (nentries + ntentries));
      for (i = 0; i < ntentries; i++)
      {
         entries[nentries + i] = tentries[i];
      }
      nentries += ntentries;
      hypre_TFree(tentries);
   }
   else
   {
      if (action == 0)
      {
         boxman = hypre_SStructGridBoxManager(grid, part, var);
      }
      else if (action == 1)
      {
         boxman = hypre_SStructGridNborBoxManager(grid, part, var);
      }
      hypre_BoxManIntersect(boxman, hypre_BoxIMin(box), hypre_BoxIMax(box),
                            &entries, &nentries);
   }

   *entries_ptr  = entries;
   *nentries_ptr = nentries;

   return hypre_error_flag;
}

