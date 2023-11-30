/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Not sure about performace yet, so leaving the '#if 1' blocks below.
 *
 ******************************************************************************/

/******************************************************************************
 *  FAC composite level interpolation.
 *  Identity interpolation of values away from underlying refinement patches;
 *  linear inside patch.
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_FacSemiInterpData data structure
 *--------------------------------------------------------------------------*/
typedef struct
{
   HYPRE_Int             nvars;
   HYPRE_Int             ndim;
   hypre_Index           stride;

   hypre_SStructPVector *recv_cvectors;
   HYPRE_Int           **recv_boxnum_map;   /* mapping between the boxes of the
                                               recv_grid and the given grid */
   hypre_BoxArrayArray **identity_arrayboxes;
   hypre_BoxArrayArray **ownboxes;
   HYPRE_Int          ***own_cboxnums;

   hypre_CommPkg       **interlevel_comm;
   hypre_CommPkg       **gnodes_comm_pkg;

   HYPRE_Real          **weights;

} hypre_FacSemiInterpData2;

/*--------------------------------------------------------------------------
 * hypre_FacSemiInterpCreate
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FacSemiInterpCreate2( void **fac_interp_vdata_ptr )
{
   HYPRE_Int                 ierr = 0;
   hypre_FacSemiInterpData2  *fac_interp_data;

   fac_interp_data = hypre_CTAlloc(hypre_FacSemiInterpData2,  1, HYPRE_MEMORY_HOST);
   *fac_interp_vdata_ptr = (void *) fac_interp_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FacSemiInterpDestroy
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FacSemiInterpDestroy2( void *fac_interp_vdata)
{
   HYPRE_Int                 ierr = 0;

   hypre_FacSemiInterpData2 *fac_interp_data = (hypre_FacSemiInterpData2 *)fac_interp_vdata;
   HYPRE_Int                 i, j, size;

   if (fac_interp_data)
   {
      hypre_SStructPVectorDestroy(fac_interp_data-> recv_cvectors);

      for (i = 0; i < (fac_interp_data-> nvars); i++)
      {
         hypre_TFree(fac_interp_data -> recv_boxnum_map[i], HYPRE_MEMORY_HOST);
         hypre_BoxArrayArrayDestroy(fac_interp_data -> identity_arrayboxes[i]);

         size = hypre_BoxArrayArraySize(fac_interp_data -> ownboxes[i]);
         hypre_BoxArrayArrayDestroy(fac_interp_data -> ownboxes[i]);
         for (j = 0; j < size; j++)
         {
            hypre_TFree(fac_interp_data -> own_cboxnums[i][j], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(fac_interp_data -> own_cboxnums[i], HYPRE_MEMORY_HOST);

         hypre_CommPkgDestroy(fac_interp_data -> gnodes_comm_pkg[i]);
         hypre_CommPkgDestroy(fac_interp_data -> interlevel_comm[i]);

      }
      hypre_TFree(fac_interp_data -> recv_boxnum_map, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_interp_data -> identity_arrayboxes, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_interp_data -> ownboxes, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_interp_data -> own_cboxnums, HYPRE_MEMORY_HOST);

      hypre_TFree(fac_interp_data -> gnodes_comm_pkg, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_interp_data -> interlevel_comm, HYPRE_MEMORY_HOST);

      for (i = 0; i < (fac_interp_data -> ndim); i++)
      {
         hypre_TFree(fac_interp_data -> weights[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(fac_interp_data -> weights, HYPRE_MEMORY_HOST);

      hypre_TFree(fac_interp_data, HYPRE_MEMORY_HOST);
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FacSemiInterpSetup2:
 * Note that an intermediate coarse SStruct_PVector is used in interpolating
 * the interlevel communicated data (coarse data). The data in these
 * intermediate vectors will be interpolated to the fine grid.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FacSemiInterpSetup2( void                 *fac_interp_vdata,
                           hypre_SStructVector  *e,
                           hypre_SStructPVector *ec,
                           hypre_Index           rfactors)
{
   HYPRE_Int                 ierr = 0;

   hypre_FacSemiInterpData2 *fac_interp_data = (hypre_FacSemiInterpData2 *)fac_interp_vdata;
   HYPRE_Int                 part_fine = 1;
   HYPRE_Int                 part_crse = 0;

   hypre_CommPkg           **gnodes_comm_pkg;
   hypre_CommPkg           **interlevel_comm;
   hypre_CommInfo           *comm_info;

   hypre_SStructPVector     *recv_cvectors;
   hypre_SStructPGrid       *recv_cgrid;
   HYPRE_Int               **recv_boxnum_map;
   hypre_SStructGrid        *temp_grid;

   hypre_SStructPGrid       *pgrid;

   hypre_SStructPVector     *ef = hypre_SStructVectorPVector(e, part_fine);
   hypre_StructVector       *e_var, *s_rc, *s_cvector;

   hypre_BoxArrayArray     **identity_arrayboxes;
   hypre_BoxArrayArray     **ownboxes;

   hypre_BoxArrayArray     **send_boxes, *send_rboxes;
   HYPRE_Int              ***send_processes;
   HYPRE_Int              ***send_remote_boxnums;

   hypre_BoxArrayArray     **recv_boxes, *recv_rboxes;
   HYPRE_Int              ***recv_processes;
   HYPRE_Int              ***recv_remote_boxnums;

   hypre_BoxArray           *boxarray;
   hypre_BoxArray           *tmp_boxarray, *intersect_boxes;
   hypre_Box                 box, scaled_box;
   HYPRE_Int              ***own_cboxnums;

   hypre_BoxManager         *boxman1;
   hypre_BoxManEntry       **boxman_entries;
   HYPRE_Int                 nboxman_entries;

   HYPRE_Int                 nvars = hypre_SStructPVectorNVars(ef);
   HYPRE_Int                 vars;

   hypre_Index               zero_index, index;
   hypre_Index               ilower, iupper;
   HYPRE_Int                *num_ghost;

   HYPRE_Int                 ndim, i, j, k, fi, ci;
   HYPRE_Int                 cnt1, cnt2;
   HYPRE_Int                 proc, myproc, tot_procs;
   HYPRE_Int                 num_values;

   HYPRE_Real              **weights;
   HYPRE_Real                refine_factors_2recp[3];
   hypre_Index               refine_factors_half;

   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myproc);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &tot_procs);

   ndim = hypre_SStructPGridNDim(hypre_SStructPVectorPGrid(ef));
   hypre_SetIndex3(zero_index, 0, 0, 0);

   hypre_BoxInit(&box, ndim);
   hypre_BoxInit(&scaled_box, ndim);

   /*------------------------------------------------------------------------
    * Intralevel communication structures-
    * A communication pkg must be created for each StructVector. Stencils
    * are needed in creating the packages- we are assuming that the same
    * stencil pattern for each StructVector, i.e., linear interpolation for
    * each variable.
    *------------------------------------------------------------------------*/
   gnodes_comm_pkg = hypre_CTAlloc(hypre_CommPkg *,  nvars, HYPRE_MEMORY_HOST);
   for (vars = 0; vars < nvars; vars++)
   {
      e_var = hypre_SStructPVectorSVector(ec, vars);
      num_ghost = hypre_StructVectorNumGhost(e_var);

      hypre_CreateCommInfoFromNumGhost(hypre_StructVectorGrid(e_var),
                                       num_ghost, &comm_info);
      hypre_CommPkgCreate(comm_info,
                          hypre_StructVectorDataSpace(e_var),
                          hypre_StructVectorDataSpace(e_var),
                          1, NULL, 0, hypre_StructVectorComm(e_var),
                          &gnodes_comm_pkg[vars]);
      hypre_CommInfoDestroy(comm_info);
   }

   (fac_interp_data -> ndim)           = ndim;
   (fac_interp_data -> nvars)          = nvars;
   (fac_interp_data -> gnodes_comm_pkg) = gnodes_comm_pkg;
   hypre_CopyIndex(rfactors, (fac_interp_data -> stride));

   /*------------------------------------------------------------------------
    * Interlevel communication structures.
    *
    * Algorithm for identity_boxes: For each cbox on this processor, refine
    * it and intersect it with the fmap.
    *    (cbox - all coarsened fmap_intersect boxes)= identity chunks
    * for cbox.
    *
    * Algorithm for own_boxes (fullwgted boxes on this processor): For each
    * fbox, coarsen it and boxmap intersect it with cmap.
    *   (cmap_intersect boxes on myproc)= ownboxes
    * for this fbox.
    *
    * Algorithm for recv_box: For each fbox, coarsen it and boxmap intersect
    * it with cmap.
    *   (cmap_intersect boxes off_proc)= unstretched recv_boxes.
    * These boxes are stretched by one in each direction so that the ghostlayer
    * is also communicated. However, the recv_grid will consists of the
    * unstretched boxes so that overlapping does not occur.
    *--------------------------------------------------------------------------*/
   identity_arrayboxes = hypre_CTAlloc(hypre_BoxArrayArray *,  nvars, HYPRE_MEMORY_HOST);

   pgrid = hypre_SStructPVectorPGrid(ec);
   hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = rfactors[i] - 1;
   }

   tmp_boxarray = hypre_BoxArrayCreate(0, ndim);
   for (vars = 0; vars < nvars; vars++)
   {
      boxman1 = hypre_SStructGridBoxManager(hypre_SStructVectorGrid(e),
                                            part_fine, vars);
      boxarray = hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));
      identity_arrayboxes[vars] = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray), ndim);

      hypre_ForBoxI(ci, boxarray)
      {
         box = *hypre_BoxArrayBox(boxarray, ci);
         hypre_AppendBox(&box,
                         hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));

         hypre_StructMapCoarseToFine(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         hypre_StructMapCoarseToFine(hypre_BoxIMax(&box), index,
                                     rfactors, hypre_BoxIMax(&scaled_box));

         hypre_BoxManIntersect(boxman1, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

         intersect_boxes = hypre_BoxArrayCreate(0, ndim);
         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&box, ilower, iupper);
            hypre_IntersectBoxes(&box, &scaled_box, &box);

            /* contract this refined box so that only the coarse nodes on this
               processor will be subtracted. */
            for (j = 0; j < ndim; j++)
            {
               k = hypre_BoxIMin(&box)[j] % rfactors[j];
               if (k)
               {
                  hypre_BoxIMin(&box)[j] += rfactors[j] - k;
               }
            }

            hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                        rfactors, hypre_BoxIMin(&box));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                        rfactors, hypre_BoxIMax(&box));
            hypre_AppendBox(&box, intersect_boxes);
         }

         hypre_SubtractBoxArrays(hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci),
                                 intersect_boxes, tmp_boxarray);
         hypre_MinUnionBoxes(hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));

         hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);
         hypre_BoxArrayDestroy(intersect_boxes);
      }
   }
   hypre_BoxArrayDestroy(tmp_boxarray);
   fac_interp_data -> identity_arrayboxes = identity_arrayboxes;

   /*--------------------------------------------------------------------------
    * fboxes are coarsened. For each coarsened fbox, we need a boxarray of
    * recvboxes or ownboxes.
    *--------------------------------------------------------------------------*/
   ownboxes = hypre_CTAlloc(hypre_BoxArrayArray *,  nvars, HYPRE_MEMORY_HOST);
   own_cboxnums = hypre_CTAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);

   recv_boxes = hypre_CTAlloc(hypre_BoxArrayArray *,  nvars, HYPRE_MEMORY_HOST);
   recv_processes = hypre_CTAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);

   /* dummy pointer for CommInfoCreate */
   recv_remote_boxnums = hypre_CTAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);
   hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }

   for (vars = 0; vars < nvars; vars++)
   {
      boxman1 = hypre_SStructGridBoxManager(hypre_SStructVectorGrid(e),
                                            part_crse, vars);
      pgrid = hypre_SStructPVectorPGrid(ef);
      boxarray = hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));

      ownboxes[vars] = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray), ndim);
      own_cboxnums[vars] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);
      recv_boxes[vars]    = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray), ndim);
      recv_processes[vars] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);
      recv_remote_boxnums[vars] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(boxarray),
                                                HYPRE_MEMORY_HOST);

      hypre_ForBoxI(fi, boxarray)
      {
         box = *hypre_BoxArrayBox(boxarray, fi);

         /*--------------------------------------------------------------------
          * Adjust this box so that only the coarse nodes inside the fine box
          * are extracted.
          *--------------------------------------------------------------------*/
         for (j = 0; j < ndim; j++)
         {
            k = hypre_BoxIMin(&box)[j] % rfactors[j];
            if (k)
            {
               hypre_BoxIMin(&box)[j] += rfactors[j] - k;
            }
         }

         hypre_StructMapFineToCoarse(hypre_BoxIMin(&box), zero_index,
                                     rfactors, hypre_BoxIMin(&scaled_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&box), zero_index,
                                     rfactors, hypre_BoxIMax(&scaled_box));

         hypre_BoxManIntersect(boxman1, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &boxman_entries, &nboxman_entries);

         cnt1 = 0; cnt2 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc == myproc)
            {
               cnt1++;
            }
            else
            {
               cnt2++;
            }
         }

         own_cboxnums[vars][fi]  = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
         recv_processes[vars][fi] = hypre_CTAlloc(HYPRE_Int,  cnt2, HYPRE_MEMORY_HOST);
         recv_remote_boxnums[vars][fi] = hypre_CTAlloc(HYPRE_Int,  cnt2, HYPRE_MEMORY_HOST);

         cnt1 = 0; cnt2 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&box, ilower, iupper);
            hypre_IntersectBoxes(&box, &scaled_box, &box);

            hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc == myproc)
            {
               hypre_AppendBox(&box,
                               hypre_BoxArrayArrayBoxArray(ownboxes[vars], fi));
               hypre_SStructBoxManEntryGetBoxnum(boxman_entries[i],
                                                 &own_cboxnums[vars][fi][cnt1]);
               cnt1++;
            }
            else
            {
               /* extend the box so all the required data for interpolation is recvd. */
               hypre_SubtractIndexes(hypre_BoxIMin(&box), index, 3,
                                     hypre_BoxIMin(&box));
               hypre_AddIndexes(hypre_BoxIMax(&box), index, 3, hypre_BoxIMax(&box));

               hypre_AppendBox(&box,
                               hypre_BoxArrayArrayBoxArray(recv_boxes[vars], fi));
               recv_processes[vars][fi][cnt2] = proc;
               cnt2++;
            }
         }
         hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);
      }  /* hypre_ForBoxI(fi, boxarray) */
   }     /* for (vars= 0; vars< nvars; vars++) */

   (fac_interp_data -> ownboxes) = ownboxes;
   (fac_interp_data -> own_cboxnums) = own_cboxnums;

   /*--------------------------------------------------------------------------
    * With the recv'ed boxes form a SStructPGrid and a SStructGrid. The
    * SStructGrid is needed to generate a box_manager (so that a local box ordering
    * for the remote_boxnums are obtained). Record the recv_boxnum/fbox_num
    * mapping. That is, we interpolate a recv_box l to a fine box m, generally
    * l != m since the recv_grid and fgrid do not agree.
    *--------------------------------------------------------------------------*/
   HYPRE_SStructGridCreate(hypre_SStructPVectorComm(ec),
                           ndim, 1, &temp_grid);
   hypre_SStructPGridCreate(hypre_SStructPVectorComm(ec), ndim, &recv_cgrid);
   recv_boxnum_map = hypre_CTAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);

   cnt2 = 0;
   hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }
   for (vars = 0; vars < nvars; vars++)
   {
      cnt1 = 0;
      hypre_ForBoxArrayI(i, recv_boxes[vars])
      {
         boxarray = hypre_BoxArrayArrayBoxArray(recv_boxes[vars], i);
         cnt1 += hypre_BoxArraySize(boxarray);
      }
      recv_boxnum_map[vars] = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);

      cnt1 = 0;
      hypre_ForBoxArrayI(i, recv_boxes[vars])
      {
         boxarray = hypre_BoxArrayArrayBoxArray(recv_boxes[vars], i);
         hypre_ForBoxI(j, boxarray)
         {
            box = *hypre_BoxArrayBox(boxarray, j);

            /* contract the box its actual size. */
            hypre_AddIndexes(hypre_BoxIMin(&box), index, 3, hypre_BoxIMin(&box));
            hypre_SubtractIndexes(hypre_BoxIMax(&box), index, 3,
                                  hypre_BoxIMax(&box));

            hypre_SStructPGridSetExtents(recv_cgrid,
                                         hypre_BoxIMin(&box),
                                         hypre_BoxIMax(&box));

            HYPRE_SStructGridSetExtents(temp_grid, 0,
                                        hypre_BoxIMin(&box),
                                        hypre_BoxIMax(&box));

            recv_boxnum_map[vars][cnt1] = i; /* record the fbox num. i */
            cnt1++;
            cnt2++;
         }
      }
   }

   /*------------------------------------------------------------------------
    * When there are no boxes to communicate, set the temp_grid to have a
    * box of size zero. This is needed so that this SStructGrid can be
    * assembled. This is done only when this only one processor.
    *------------------------------------------------------------------------*/
   if (cnt2 == 0)
   {
      /* min_index > max_index so that the box has volume zero. */
      hypre_BoxSetExtents(&box, index, zero_index);
      hypre_SStructPGridSetExtents(recv_cgrid,
                                   hypre_BoxIMin(&box),
                                   hypre_BoxIMax(&box));

      HYPRE_SStructGridSetExtents(temp_grid, 0,
                                  hypre_BoxIMin(&box),
                                  hypre_BoxIMax(&box));
   }

   HYPRE_SStructGridSetVariables(temp_grid, 0,
                                 hypre_SStructPGridNVars(pgrid),
                                 hypre_SStructPGridVarTypes(pgrid));
   HYPRE_SStructGridAssemble(temp_grid);
   hypre_SStructPGridSetVariables(recv_cgrid, nvars,
                                  hypre_SStructPGridVarTypes(pgrid) );
   hypre_SStructPGridAssemble(recv_cgrid);

   hypre_SStructPVectorCreate(hypre_SStructPGridComm(recv_cgrid), recv_cgrid,
                              &recv_cvectors);
   hypre_SStructPVectorInitialize(recv_cvectors);
   hypre_SStructPVectorAssemble(recv_cvectors);

   fac_interp_data -> recv_cvectors  = recv_cvectors;
   fac_interp_data -> recv_boxnum_map = recv_boxnum_map;

   /* pgrid recv_cgrid no longer needed. */
   hypre_SStructPGridDestroy(recv_cgrid);

   /*------------------------------------------------------------------------
    * Send_boxes.
    * Algorithm for send_boxes: For each cbox on this processor, box_map
    * intersect it with temp_grid's map.
    *   (intersection boxes off-proc)= send_boxes for this cbox.
    * Note that the send_boxes will be stretched to include the ghostlayers.
    * This guarantees that all the data required for linear interpolation
    * will be on the processor. Also, note that the remote_boxnums are
    * with respect to the recv_cgrid box numbering.
    *--------------------------------------------------------------------------*/
   send_boxes = hypre_CTAlloc(hypre_BoxArrayArray *,  nvars, HYPRE_MEMORY_HOST);
   send_processes = hypre_CTAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);
   send_remote_boxnums = hypre_CTAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);

   hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }
   for (vars = 0; vars < nvars; vars++)
   {
      /*-------------------------------------------------------------------
       * send boxes: intersect with temp_grid that has all the recv boxes-
       * These local box_nums may not be the same as the local box_nums of
       * the coarse grid.
       *-------------------------------------------------------------------*/
      boxman1 = hypre_SStructGridBoxManager(temp_grid, 0, vars);
      pgrid = hypre_SStructPVectorPGrid(ec);
      boxarray = hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, vars));

      send_boxes[vars] = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxarray), ndim);
      send_processes[vars] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(boxarray), HYPRE_MEMORY_HOST);
      send_remote_boxnums[vars] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(boxarray),
                                                HYPRE_MEMORY_HOST);

      hypre_ForBoxI(ci, boxarray)
      {
         box = *hypre_BoxArrayBox(boxarray, ci);
         hypre_BoxSetExtents(&scaled_box, hypre_BoxIMin(&box), hypre_BoxIMax(&box));

         hypre_BoxManIntersect(boxman1, hypre_BoxIMin(&scaled_box),
                               hypre_BoxIMax(&scaled_box), &boxman_entries, &nboxman_entries);

         cnt1 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc != myproc)
            {
               cnt1++;
            }
         }
         send_processes[vars][ci]     = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
         send_remote_boxnums[vars][ci] = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);

         cnt1 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            hypre_BoxSetExtents(&box, ilower, iupper);
            hypre_IntersectBoxes(&box, &scaled_box, &box);

            hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc != myproc)
            {
               /* strech the box */
               hypre_SubtractIndexes(hypre_BoxIMin(&box), index, 3,
                                     hypre_BoxIMin(&box));
               hypre_AddIndexes(hypre_BoxIMax(&box), index, 3, hypre_BoxIMax(&box));

               hypre_AppendBox(&box,
                               hypre_BoxArrayArrayBoxArray(send_boxes[vars], ci));

               send_processes[vars][ci][cnt1] = proc;
               hypre_SStructBoxManEntryGetBoxnum(
                  boxman_entries[i], &send_remote_boxnums[vars][ci][cnt1]);
               cnt1++;
            }
         }

         hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);
      }  /* hypre_ForBoxI(ci, boxarray) */
   }    /* for (vars= 0; vars< nvars; vars++) */

   /*--------------------------------------------------------------------------
    * Can disgard temp_grid now- only needed it's box_man info,
    *--------------------------------------------------------------------------*/
   HYPRE_SStructGridDestroy(temp_grid);

   /*--------------------------------------------------------------------------
    * Can create the interlevel_comm.
    *--------------------------------------------------------------------------*/
   interlevel_comm = hypre_CTAlloc(hypre_CommPkg *,  nvars, HYPRE_MEMORY_HOST);

   num_values = 1;
   for (vars = 0; vars < nvars; vars++)
   {
      s_rc = hypre_SStructPVectorSVector(ec, vars);

      s_cvector = hypre_SStructPVectorSVector(recv_cvectors, vars);
      send_rboxes = hypre_BoxArrayArrayDuplicate(send_boxes[vars]);
      recv_rboxes = hypre_BoxArrayArrayDuplicate(recv_boxes[vars]);

      hypre_CommInfoCreate(send_boxes[vars], recv_boxes[vars],
                           send_processes[vars], recv_processes[vars],
                           send_remote_boxnums[vars], recv_remote_boxnums[vars],
                           send_rboxes, recv_rboxes, 1, &comm_info);

      hypre_CommPkgCreate(comm_info,
                          hypre_StructVectorDataSpace(s_rc),
                          hypre_StructVectorDataSpace(s_cvector),
                          num_values, NULL, 0,
                          hypre_StructVectorComm(s_rc),
                          &interlevel_comm[vars]);
      hypre_CommInfoDestroy(comm_info);
   }
   hypre_TFree(send_boxes, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_boxes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_processes, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_processes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_remote_boxnums, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_remote_boxnums, HYPRE_MEMORY_HOST);

   (fac_interp_data -> interlevel_comm) = interlevel_comm;

   /* interpolation weights */
   weights = hypre_TAlloc(HYPRE_Real *,  ndim, HYPRE_MEMORY_HOST);
   for (i = 0; i < ndim; i++)
   {
      weights[i] = hypre_CTAlloc(HYPRE_Real,  rfactors[i] + 1, HYPRE_MEMORY_HOST);
   }

   hypre_ClearIndex(refine_factors_half);
   /*   hypre_ClearIndex(refine_factors_2recp);*/
   for (i = 0; i < ndim; i++)
   {
      refine_factors_half[i] = rfactors[i] / 2;
      refine_factors_2recp[i] = 1.0 / (2.0 * rfactors[i]);
   }

   for (i = 0; i < ndim; i++)
   {
      for (j = 0; j <= refine_factors_half[i]; j++)
      {
         weights[i][j] = refine_factors_2recp[i] * (rfactors[i] + 2 * j - 1.0);
      }

      for (j = (refine_factors_half[i] + 1); j <= rfactors[i]; j++)
      {
         weights[i][j] = refine_factors_2recp[i] * (2 * j - rfactors[i] - 1.0);
      }
   }
   (fac_interp_data -> weights) = weights;


   return ierr;
}

HYPRE_Int
hypre_FAC_IdentityInterp2(void                 *  fac_interp_vdata,
                          hypre_SStructPVector *  xc,
                          hypre_SStructVector  *  e)
{
   hypre_FacSemiInterpData2 *interp_data = (hypre_FacSemiInterpData2 *)fac_interp_vdata;
   hypre_BoxArrayArray     **identity_boxes = interp_data-> identity_arrayboxes;

   HYPRE_Int               part_crse = 0;

   HYPRE_Int               ierr     = 0;

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection).
    * The pgrid of xc is the same as the part_csre pgrid of e.
    *-----------------------------------------------------------------------*/
   hypre_SStructPartialPCopy(xc,
                             hypre_SStructVectorPVector(e, part_crse),
                             identity_boxes);

   return ierr;
}

/*-------------------------------------------------------------------------
 * Linear interpolation. Interpolate the vector first by interpolating the
 * values in ownboxes and then values in recv_cvectors (the interlevel
 * communicated data).
 *-------------------------------------------------------------------------*/
HYPRE_Int
hypre_FAC_WeightedInterp2(void                  *fac_interp_vdata,
                          hypre_SStructPVector  *xc,
                          hypre_SStructVector   *e_parts)
{
   HYPRE_Int ierr = 0;

   hypre_FacSemiInterpData2 *interp_data = (hypre_FacSemiInterpData2 *)fac_interp_vdata;

   hypre_CommPkg          **comm_pkg       = interp_data-> gnodes_comm_pkg;
   hypre_CommPkg          **interlevel_comm = interp_data-> interlevel_comm;
   hypre_SStructPVector    *recv_cvectors  = interp_data-> recv_cvectors;
   HYPRE_Int              **recv_boxnum_map = interp_data-> recv_boxnum_map;
   hypre_BoxArrayArray    **ownboxes       = interp_data-> ownboxes;
   HYPRE_Int             ***own_cboxnums   = interp_data-> own_cboxnums;
   HYPRE_Real             **weights        = interp_data-> weights;
   HYPRE_Int                ndim           = interp_data-> ndim;

   hypre_CommHandle       *comm_handle;

   hypre_IndexRef          stride;  /* refinement factors */

   hypre_SStructPVector   *e;

   hypre_StructGrid       *fgrid;
   hypre_BoxArray         *fgrid_boxes;
   hypre_Box              *fbox;
   hypre_BoxArrayArray    *own_cboxes;
   hypre_BoxArray         *own_abox;
   hypre_Box              *ownbox;
   HYPRE_Int             **var_boxnums;
   HYPRE_Int              *cboxnums;

   hypre_Box              *xc_dbox;
   hypre_Box              *e_dbox;

   hypre_Box               refined_box, intersect_box;


   hypre_StructVector     *xc_var;
   hypre_StructVector     *e_var;
   hypre_StructVector     *recv_var;

   HYPRE_Real           ***xcp;
   HYPRE_Real           ***ep;

   hypre_Index             loop_size, lindex;
   hypre_Index             start, start_offset;
   hypre_Index             startc;
   hypre_Index             stridec;
   hypre_Index             refine_factors;
   hypre_Index             refine_factors_half;
   hypre_Index             intersect_size;
   hypre_Index             zero_index, temp_index1, temp_index2;

   HYPRE_Int               fi, bi;
   HYPRE_Int               nvars, var;

   HYPRE_Int               i, j, k, offset_ip1, offset_jp1, offset_kp1;
   HYPRE_Int               ishift, jshift = 0, kshift = 0;
   HYPRE_Int               ptr_ishift, ptr_jshift, ptr_kshift;
   HYPRE_Int               imax, jmax, kmax;
   HYPRE_Int               jsize, ksize;

   HYPRE_Int               part_fine = 1;

   HYPRE_Real              xweight1, xweight2;
   HYPRE_Real              yweight1, yweight2 = 0.0;
   HYPRE_Real              zweight1, zweight2 = 0.0;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   hypre_BoxInit(&refined_box, ndim);
   hypre_BoxInit(&intersect_box, ndim);

   stride = (interp_data -> stride);

   hypre_SetIndex3(zero_index, 0, 0, 0);
   hypre_SetIndex3(lindex, 0, 0, 0);
   hypre_CopyIndex(stride, refine_factors);
   for (i = ndim; i < 3; i++)
   {
      refine_factors[i] = 1;
   }
   hypre_SetIndex3(stridec, 1, 1, 1);
   for (i = 0; i < ndim; i++)
   {
      refine_factors_half[i] = refine_factors[i] / 2;
   }

   /*-----------------------------------------------------------------------
    * Compute e in the refined patch. But first communicate the coarse
    * data. Will need a ghostlayer communication on the given level and an
    * interlevel communication between levels.
    *-----------------------------------------------------------------------*/
   nvars =  hypre_SStructPVectorNVars(xc);
   for (var = 0; var < nvars; var++)
   {
      xc_var = hypre_SStructPVectorSVector(xc, var);
      hypre_InitializeCommunication(comm_pkg[var],
                                    hypre_StructVectorData(xc_var),
                                    hypre_StructVectorData(xc_var), 0, 0,
                                    &comm_handle);
      hypre_FinalizeCommunication(comm_handle);

      if (recv_cvectors != NULL)
      {
         recv_var = hypre_SStructPVectorSVector(recv_cvectors, var);
         hypre_InitializeCommunication(interlevel_comm[var],
                                       hypre_StructVectorData(xc_var),
                                       hypre_StructVectorData(recv_var), 0, 0,
                                       &comm_handle);
         hypre_FinalizeCommunication(comm_handle);
      }
   }

   e =  hypre_SStructVectorPVector(e_parts, part_fine);

   /*-----------------------------------------------------------------------
    * Allocate memory for the data pointers. Assuming linear interpolation.
    * We stride through the refinement patch by the refinement factors, and
    * so we must have pointers to the intermediate fine nodes=> ep will
    * be size refine_factors[2]*refine_factors[1]. This holds for all
    * dimensions since refine_factors[i]= 1 for i>= ndim.
    * Note that we need 3 coarse nodes per coordinate direction for the
    * interpolating. This is dimensional dependent:
    *   ndim= 3     kplane= 0,1,2 & jplane= 0,1,2    **ptr size [3][3]
    *   ndim= 2     kplane= 0     & jplane= 0,1,2    **ptr size [1][3]
    *   ndim= 1     kplane= 0     & jplane= 0        **ptr size [1][1]
    *-----------------------------------------------------------------------*/
   ksize = 3;
   jsize = 3;
   if (ndim < 3)
   {
      ksize = 1;
   }
   if (ndim < 2)
   {
      jsize = 1;
   }

   xcp  = hypre_TAlloc(HYPRE_Real **,  ksize, HYPRE_MEMORY_HOST);
   ep   = hypre_TAlloc(HYPRE_Real **,  refine_factors[2], HYPRE_MEMORY_HOST);

   for (k = 0; k < refine_factors[2]; k++)
   {
      ep[k] = hypre_TAlloc(HYPRE_Real *,  refine_factors[1], HYPRE_MEMORY_HOST);
   }

   for (k = 0; k < ksize; k++)
   {
      xcp[k] = hypre_TAlloc(HYPRE_Real *,  jsize, HYPRE_MEMORY_HOST);
   }

   for (var = 0; var < nvars; var++)
   {
      xc_var = hypre_SStructPVectorSVector(xc, var);
      e_var = hypre_SStructPVectorSVector(e, var);

      fgrid      = hypre_StructVectorGrid(e_var);
      fgrid_boxes = hypre_StructGridBoxes(fgrid);

      own_cboxes = ownboxes[var];
      var_boxnums = own_cboxnums[var];

      /*--------------------------------------------------------------------
       * Interpolate the own_box coarse grid values.
       *--------------------------------------------------------------------*/
      hypre_ForBoxI(fi, fgrid_boxes)
      {
         fbox = hypre_BoxArrayBox(fgrid_boxes, fi);

         e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e_var), fi);
         own_abox = hypre_BoxArrayArrayBoxArray(own_cboxes, fi);
         cboxnums = var_boxnums[fi];

         /*--------------------------------------------------------------------
          * Get the ptrs for the fine struct_vectors.
          *--------------------------------------------------------------------*/
         for (k = 0; k < refine_factors[2]; k++)
         {
            for (j = 0; j < refine_factors[1]; j++)
            {
               hypre_SetIndex3(temp_index1, 0, j, k);
               ep[k][j] = hypre_StructVectorBoxData(e_var, fi) +
                          hypre_BoxOffsetDistance(e_dbox, temp_index1);
            }
         }

         hypre_ForBoxI(bi, own_abox)
         {
            ownbox = hypre_BoxArrayBox(own_abox, bi);
            hypre_StructMapCoarseToFine(hypre_BoxIMin(ownbox), zero_index,
                                        refine_factors, hypre_BoxIMin(&refined_box));
            hypre_ClearIndex(temp_index1);
            for (j = 0; j < ndim; j++)
            {
               temp_index1[j] = refine_factors[j] - 1;
            }
            hypre_StructMapCoarseToFine(hypre_BoxIMax(ownbox), temp_index1,
                                        refine_factors, hypre_BoxIMax(&refined_box));
            hypre_IntersectBoxes(fbox, &refined_box, &intersect_box);

            xc_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(xc_var),
                                        cboxnums[bi]);

            /*-----------------------------------------------------------------
             * Get ptrs for the crse struct_vectors. For linear interpolation
             * and arbitrary refinement factors, we need to point to the correct
             * coarse grid nodes. Note that the ownboxes were created so that
             * only the coarse nodes inside a fbox are contained in ownbox.
             * Since we loop over the fine intersect box, we need to refine
             * ownbox.
             *-----------------------------------------------------------------*/
            hypre_CopyIndex(hypre_BoxIMin(&intersect_box), start);
            hypre_CopyIndex(hypre_BoxIMax(&intersect_box), intersect_size);
            for (i = 0; i < 3; i++)
            {
               intersect_size[i] -= (start[i] - 1);
            }

            /*------------------------------------------------------------------
             * The fine intersection box may not be divisible by the refinement
             * factor. This means that the interpolated coarse nodes and their
             * wieghts must be carefully determined. We accomplish this using the
             * offset away from a fine index that is divisible by the factor.
             * Because the ownboxes were created so that only coarse nodes
             * completely in the fbox are included, start is always divisible
             * by refine_factors. We do the calculation anyways for future changes.
             *------------------------------------------------------------------*/
            hypre_ClearIndex(start_offset);
            for (i = 0; i < ndim; i++)
            {
               start_offset[i] = start[i] % refine_factors[i];
            }

            ptr_kshift = 0;
            if ( (start[2] % refine_factors[2] < refine_factors_half[2]) && ndim == 3 )
            {
               ptr_kshift = -1;
            }

            ptr_jshift = 0;
            if ( start[1] % refine_factors[1] < refine_factors_half[1] && ndim >= 2 )
            {
               ptr_jshift = -1;
            }

            ptr_ishift = 0;
            if ( start[0] % refine_factors[0] < refine_factors_half[0] )
            {
               ptr_ishift = -1;
            }

            for (k = 0; k < ksize; k++)
            {
               for (j = 0; j < jsize; j++)
               {
                  hypre_SetIndex3(temp_index2, ptr_ishift, j + ptr_jshift, k + ptr_kshift);
                  xcp[k][j] = hypre_StructVectorBoxData(xc_var, cboxnums[bi]) +
                              hypre_BoxOffsetDistance(xc_dbox, temp_index2);
               }
            }

            hypre_CopyIndex(hypre_BoxIMin(ownbox), startc);
            hypre_BoxGetSize(ownbox, loop_size);

            hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                      e_dbox,  start,  stride,  ei,
                                      xc_dbox, startc, stridec, xci);
            {
               /*--------------------------------------------------------
                * Linear interpolation. Determine the weights and the
                * correct coarse grid values to be weighted. All fine
                * values in an agglomerated coarse cell or in the remainder
                * agglomerated coarse cells are determined. The upper
                * extents are needed.
                *--------------------------------------------------------*/
               zypre_BoxLoopGetIndex(lindex);
               imax = hypre_min( (intersect_size[0] - lindex[0] * stride[0]),
                                 refine_factors[0] );
               jmax = hypre_min( (intersect_size[1] - lindex[1] * stride[1]),
                                 refine_factors[1]);
               kmax = hypre_min( (intersect_size[2] - lindex[2] * stride[2]),
                                 refine_factors[2]);

               for (k = 0; k < kmax; k++)
               {
                  if (ndim == 3)
                  {
                     offset_kp1 = start_offset[2] + k + 1;

                     if (ptr_kshift == -1)
                     {
                        if (offset_kp1 <= refine_factors_half[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 1;
                           if (offset_kp1 >  refine_factors_half[2] &&
                               offset_kp1 <= refine_factors[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1 - refine_factors[2]];
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }
                     else
                     {
                        if (offset_kp1 > refine_factors_half[2] &&
                            offset_kp1 <= refine_factors[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 0;
                           offset_kp1 -= refine_factors[2];
                           if (offset_kp1 > 0 && offset_kp1 <= refine_factors_half[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1];
                              kshift  = 1;
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }
                  }     /* if (ndim == 3) */

                  for (j = 0; j < jmax; j++)
                  {
                     if (ndim >= 2)
                     {
                        offset_jp1 = start_offset[1] + j + 1;

                        if (ptr_jshift == -1)
                        {
                           if (offset_jp1 <= refine_factors_half[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 1;
                              if (offset_jp1 >  refine_factors_half[1] &&
                                  offset_jp1 <= refine_factors[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1 - refine_factors[1]];
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }

                        else
                        {
                           if (offset_jp1 > refine_factors_half[1] &&
                               offset_jp1 <= refine_factors[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 0;
                              offset_jp1 -= refine_factors[1];
                              if (offset_jp1 > 0 && offset_jp1 <= refine_factors_half[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1];
                                 jshift  = 1;
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }
                     }     /* if (ndim >= 2) */

                     for (i = 0; i < imax; i++)
                     {
                        offset_ip1 = start_offset[0] + i + 1;

                        if (ptr_ishift == -1)
                        {
                           if (offset_ip1 <= refine_factors_half[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 1;
                              if (offset_ip1 >  refine_factors_half[0] &&
                                  offset_ip1 <= refine_factors[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1 - refine_factors[0]];
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        else
                        {
                           if (offset_ip1 > refine_factors_half[0] &&
                               offset_ip1 <= refine_factors[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 0;
                              offset_ip1 -= refine_factors[0];
                              if (offset_ip1 > 0 && offset_ip1 <= refine_factors_half[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1];
                                 ishift  = 1;
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        if (ndim == 3)
                        {
                           ep[k][j][ei + i] = zweight1 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift + 1][ishift + xci + 1]) )
                                              + zweight2 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift + 1][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift + 1][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift + 1][ishift + xci + 1]) );
                        }
                        else if (ndim == 2)
                        {
                           ep[0][j][ei + i] = yweight1 * (
                                                 xweight1 * xcp[0][jshift][ishift + xci] +
                                                 xweight2 * xcp[0][jshift][ishift + xci + 1]);
                           ep[0][j][ei + i] += yweight2 * (
                                                  xweight1 * xcp[0][jshift + 1][ishift + xci] +
                                                  xweight2 * xcp[0][jshift + 1][ishift + xci + 1]);
                        }
                        else
                        {
                           ep[0][0][ei + i] = xweight1 * xcp[0][0][ishift + xci] +
                                              xweight2 * xcp[0][0][ishift + xci + 1];
                        }
                     }      /* for (i= 0; i< imax; i++) */
                  }         /* for (j= 0; j< jmax; j++) */
               }            /* for (k= 0; k< kmax; k++) */
            }
            hypre_SerialBoxLoop2End(ei, xci);

         }/* hypre_ForBoxI(bi, own_abox) */
      }   /* hypre_ForBoxArray(fi, fgrid_boxes) */

      /*--------------------------------------------------------------------
       * Interpolate the off-processor coarse grid values. These are the
       * recv_cvector values. We will use the ownbox ptrs.
       * recv_vector is non-null even when it has a grid with zero-volume
       * boxes.
       *--------------------------------------------------------------------*/
      recv_var = hypre_SStructPVectorSVector(recv_cvectors, var);
      own_abox = hypre_StructGridBoxes(hypre_StructVectorGrid(recv_var));
      cboxnums = recv_boxnum_map[var];

      hypre_ForBoxI(bi, own_abox)
      {
         ownbox = hypre_BoxArrayBox(own_abox, bi);

         /*check for boxes of volume zero- i.e., recv_cvectors is really null.*/
         if (hypre_BoxVolume(ownbox))
         {
            xc_dbox = hypre_BoxArrayBox(
                         hypre_StructVectorDataSpace(recv_var), bi);

            fi = cboxnums[bi];
            fbox  = hypre_BoxArrayBox(fgrid_boxes, fi);
            e_dbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(e_var), fi);

            /*------------------------------------------------------------------
             * Get the ptrs for the fine struct_vectors.
             *------------------------------------------------------------------*/
            for (k = 0; k < refine_factors[2]; k++)
            {
               for (j = 0; j < refine_factors[1]; j++)
               {
                  hypre_SetIndex3(temp_index1, 0, j, k);
                  ep[k][j] = hypre_StructVectorBoxData(e_var, fi) +
                             hypre_BoxOffsetDistance(e_dbox, temp_index1);
               }
            }

            hypre_StructMapCoarseToFine(hypre_BoxIMin(ownbox), zero_index,
                                        refine_factors, hypre_BoxIMin(&refined_box));
            hypre_ClearIndex(temp_index1);
            for (j = 0; j < ndim; j++)
            {
               temp_index1[j] = refine_factors[j] - 1;
            }
            hypre_StructMapCoarseToFine(hypre_BoxIMax(ownbox), temp_index1,
                                        refine_factors, hypre_BoxIMax(&refined_box));
            hypre_IntersectBoxes(fbox, &refined_box, &intersect_box);

            /*-----------------------------------------------------------------
             * Get ptrs for the crse struct_vectors. For linear interpolation
             * and arbitrary refinement factors, we need to point to the correct
             * coarse grid nodes. Note that the ownboxes were created so that
             * only the coarse nodes inside a fbox are contained in ownbox.
             * Since we loop over the fine intersect box, we need to refine
             * ownbox.
             *-----------------------------------------------------------------*/
            hypre_CopyIndex(hypre_BoxIMin(&intersect_box), start);
            hypre_CopyIndex(hypre_BoxIMax(&intersect_box), intersect_size);
            for (i = 0; i < 3; i++)
            {
               intersect_size[i] -= (start[i] - 1);
            }

            /*------------------------------------------------------------------
             * The fine intersection box may not be divisible by the refinement
             * factor. This means that the interpolated coarse nodes and their
             * weights must be carefully determined. We accomplish this using the
             * offset away from a fine index that is divisible by the factor.
             * Because the ownboxes were created so that only coarse nodes
             * completely in the fbox are included, start is always divisible
             * by refine_factors. We do the calculation anyways for future changes.
             *------------------------------------------------------------------*/
            hypre_ClearIndex(start_offset);
            for (i = 0; i < ndim; i++)
            {
               start_offset[i] = start[i] % refine_factors[i];
            }

            ptr_kshift = 0;
            if ((start[2] % refine_factors[2] < refine_factors_half[2]) && ndim == 3)
            {
               ptr_kshift = -1;
            }

            ptr_jshift = 0;
            if ((start[1] % refine_factors[1] < refine_factors_half[1]) && ndim >= 2)
            {
               ptr_jshift = -1;
            }

            ptr_ishift = 0;
            if ( start[0] % refine_factors[0] < refine_factors_half[0] )
            {
               ptr_ishift = -1;
            }

            for (k = 0; k < ksize; k++)
            {
               for (j = 0; j < jsize; j++)
               {
                  hypre_SetIndex3(temp_index2,
                                  ptr_ishift, j + ptr_jshift, k + ptr_kshift);
                  xcp[k][j] = hypre_StructVectorBoxData(recv_var, bi) +
                              hypre_BoxOffsetDistance(xc_dbox, temp_index2);
               }
            }

            hypre_CopyIndex(hypre_BoxIMin(ownbox), startc);
            hypre_BoxGetSize(ownbox, loop_size);

            hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                      e_dbox,  start,  stride,  ei,
                                      xc_dbox, startc, stridec, xci);
            {
               /*--------------------------------------------------------
                * Linear interpolation. Determine the weights and the
                * correct coarse grid values to be weighted. All fine
                * values in an agglomerated coarse cell or in the remainder
                * agglomerated coarse cells are determined. The upper
                * extents are needed.
                *--------------------------------------------------------*/
               zypre_BoxLoopGetIndex(lindex);
               imax = hypre_min( (intersect_size[0] - lindex[0] * stride[0]),
                                 refine_factors[0] );
               jmax = hypre_min( (intersect_size[1] - lindex[1] * stride[1]),
                                 refine_factors[1]);
               kmax = hypre_min( (intersect_size[2] - lindex[2] * stride[2]),
                                 refine_factors[2]);

               for (k = 0; k < kmax; k++)
               {
                  if (ndim == 3)
                  {
                     offset_kp1 = start_offset[2] + k + 1;

                     if (ptr_kshift == -1)
                     {
                        if (offset_kp1 <= refine_factors_half[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 1;
                           if (offset_kp1 >  refine_factors_half[2] &&
                               offset_kp1 <= refine_factors[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1 - refine_factors[2]];
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }

                     else
                     {
                        if (offset_kp1 > refine_factors_half[2] &&
                            offset_kp1 <= refine_factors[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 0;
                           offset_kp1 -= refine_factors[2];
                           if (offset_kp1 > 0 && offset_kp1 <= refine_factors_half[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1];
                              kshift  = 1;
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }
                  }     /* if (ndim == 3) */

                  for (j = 0; j < jmax; j++)
                  {
                     if (ndim >= 2)
                     {
                        offset_jp1 = start_offset[1] + j + 1;

                        if (ptr_jshift == -1)
                        {
                           if (offset_jp1 <= refine_factors_half[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 1;
                              if (offset_jp1 >  refine_factors_half[1] &&
                                  offset_jp1 <= refine_factors[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1 - refine_factors[1]];
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }

                        else
                        {
                           if (offset_jp1 > refine_factors_half[1] &&
                               offset_jp1 <= refine_factors[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 0;
                              offset_jp1 -= refine_factors[1];
                              if (offset_jp1 > 0 && offset_jp1 <= refine_factors_half[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1];
                                 jshift  = 1;
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }
                     }  /* if (ndim >= 2) */

                     for (i = 0; i < imax; i++)
                     {
                        offset_ip1 = start_offset[0] + i + 1;

                        if (ptr_ishift == -1)
                        {
                           if (offset_ip1 <= refine_factors_half[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 1;
                              if (offset_ip1 >  refine_factors_half[0] &&
                                  offset_ip1 <= refine_factors[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1 - refine_factors[0]];
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        else
                        {
                           if (offset_ip1 > refine_factors_half[0] &&
                               offset_ip1 <= refine_factors[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 0;
                              offset_ip1 -= refine_factors[0];
                              if (offset_ip1 > 0 && offset_ip1 <= refine_factors_half[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1];
                                 ishift  = 1;
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }


                        if (ndim == 3)
                        {
                           ep[k][j][ei + i] = zweight1 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift + 1][ishift + xci + 1]) )
                                              + zweight2 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift + 1][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift + 1][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift + 1][ishift + xci + 1]) );
                        }
                        else if (ndim == 2)
                        {
                           ep[0][j][ei + i] = yweight1 * (
                                                 xweight1 * xcp[0][jshift][ishift + xci] +
                                                 xweight2 * xcp[0][jshift][ishift + xci + 1]);
                           ep[0][j][ei + i] += yweight2 * (
                                                  xweight1 * xcp[0][jshift + 1][ishift + xci] +
                                                  xweight2 * xcp[0][jshift + 1][ishift + xci + 1]);
                        }

                        else
                        {
                           ep[0][0][ei + i] = xweight1 * xcp[0][0][ishift + xci] +
                                              xweight2 * xcp[0][0][ishift + xci + 1];
                        }

                     }      /* for (i= 0; i< imax; i++) */
                  }         /* for (j= 0; j< jmax; j++) */
               }            /* for (k= 0; k< kmax; k++) */
            }
            hypre_SerialBoxLoop2End(ei, xci);

         }  /* if (hypre_BoxVolume(ownbox)) */
      }     /* hypre_ForBoxI(bi, own_abox) */
   }         /* for (var= 0; var< nvars; var++)*/

   for (k = 0; k < ksize; k++)
   {
      hypre_TFree(xcp[k], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(xcp, HYPRE_MEMORY_HOST);

   for (k = 0; k < refine_factors[2]; k++)
   {
      hypre_TFree(ep[k], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ep, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/
   return ierr;
}
