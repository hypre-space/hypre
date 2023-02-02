/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_AMR_RAP:  Forms the coarse operators for all amr levels.
 * Given an amr composite matrix, the coarse grid operator is produced.
 * Nesting of amr levels is not assumed. Communication of chunks of the
 * coarse grid operator is performed.
 *
 * Note: The sstruct_grid of A and fac_A are the same. These are kept the
 * same so that the row ranks are the same. However, the generated
 * coarse-grid operators are re-distributed so that each processor has its
 * operator on its grid.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_AMR_RAP( hypre_SStructMatrix  *A,
               hypre_Index          *rfactors,
               hypre_SStructMatrix **fac_A_ptr )
{

   MPI_Comm                     comm         = hypre_SStructMatrixComm(A);
   HYPRE_Int                    ndim         = hypre_SStructMatrixNDim(A);
   HYPRE_Int                    nparts       = hypre_SStructMatrixNParts(A);
   hypre_SStructGraph          *graph        = hypre_SStructMatrixGraph(A);
   HYPRE_IJMatrix               ij_A         = hypre_SStructMatrixIJMatrix(A);
   HYPRE_Int                    matrix_type  = hypre_SStructMatrixObjectType(A);

   hypre_SStructGrid           *grid         = hypre_SStructGraphGrid(graph);
   HYPRE_Int                    nUventries   = hypre_SStructGraphNUVEntries(graph);
   HYPRE_Int                   *iUventries   = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry       **Uventries    = hypre_SStructGraphUVEntries(graph);
   hypre_SStructUVEntry        *Uventry;
   HYPRE_Int                    nUentries;

   hypre_CommPkg               *amrA_comm_pkg;
   hypre_CommHandle            *comm_handle;

   hypre_SStructMatrix         *fac_A;
   hypre_SStructPMatrix        *pmatrix, *fac_pmatrix;
   hypre_StructMatrix          *smatrix, *fac_smatrix;
   hypre_Box                   *smatrix_dbox, *fac_smatrix_dbox;
   HYPRE_Real                  *smatrix_vals, *fac_smatrix_vals;

   hypre_SStructGrid           *fac_grid;
   hypre_SStructGraph          *fac_graph;
   hypre_SStructPGrid          *f_pgrid, *c_pgrid;
   hypre_StructGrid            *fgrid, *cgrid;
   hypre_BoxArray              *grid_boxes, *cgrid_boxes;
   hypre_Box                   *grid_box;
   hypre_Box                    scaled_box;

   hypre_SStructPGrid          *temp_pgrid;
   hypre_SStructStencil       **temp_sstencils;
   hypre_SStructPMatrix        *temp_pmatrix;

   hypre_SStructOwnInfoData  ***owninfo;
   hypre_SStructRecvInfoData   *recvinfo;
   hypre_SStructSendInfoData   *sendinfo;
   hypre_BoxArrayArray         *own_composite_cboxes, *own_boxes;
   hypre_BoxArray              *own_composite_cbox;
   HYPRE_Int                  **own_cboxnums;

   hypre_BoxManager            *fboxman, *cboxman;
   hypre_BoxManEntry           *boxman_entry;
   hypre_Index                  ilower;

   HYPRE_Real                  *values;
   HYPRE_Int                   *ncols, tot_cols;
   HYPRE_BigInt                *rows, *cols;

   hypre_SStructStencil        *stencils;
   hypre_Index                  stencil_shape, loop_size;
   HYPRE_Int                    stencil_size, *stencil_vars;

   hypre_Index                  index, stride, zero_index;
   HYPRE_Int                    nvars, var1, var2, part, cbox;
   HYPRE_Int                    i, j, k, size;

   HYPRE_Int                    myid;
   HYPRE_Int                    ierr = 0;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_ClearIndex(zero_index);

   hypre_BoxInit(&scaled_box, ndim);

   hypre_SStructGraphRef(graph, &fac_graph);
   fac_grid = hypre_SStructGraphGrid(fac_graph);
   HYPRE_SStructMatrixCreate(comm, fac_graph, &fac_A);
   HYPRE_SStructMatrixInitialize(fac_A);

   /*--------------------------------------------------------------------------
    * Copy all A's unstructured data and structured data that are not processed
    * into fac_A. Since the grids are the same for both matrices, the ranks
    * are also the same. Thus, the rows, cols, etc. for the IJ_matrix are
    * the same.
    *--------------------------------------------------------------------------*/
   ncols = hypre_CTAlloc(HYPRE_Int,  nUventries, HYPRE_MEMORY_HOST);
   rows = hypre_CTAlloc(HYPRE_BigInt,  nUventries, HYPRE_MEMORY_HOST);

   tot_cols = 0;
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      tot_cols += hypre_SStructUVEntryNUEntries(Uventry);
   }
   cols = hypre_CTAlloc(HYPRE_BigInt,  tot_cols, HYPRE_MEMORY_HOST);

   k    = 0;
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      part   = hypre_SStructUVEntryPart(Uventry);
      hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);
      var1     = hypre_SStructUVEntryVar(Uventry);
      nUentries = hypre_SStructUVEntryNUEntries(Uventry);

      ncols[i] = nUentries;
      hypre_SStructGridFindBoxManEntry(grid, part, index, var1, &boxman_entry);
      hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rows[i], matrix_type);

      for (j = 0; j < nUentries; j++)
      {
         cols[k++] = hypre_SStructUVEntryToRank(Uventry, j);
      }
   }

   values = hypre_CTAlloc(HYPRE_Real,  tot_cols, HYPRE_MEMORY_HOST);
   HYPRE_IJMatrixGetValues(ij_A, nUventries, ncols, rows, cols, values);

   HYPRE_IJMatrixSetValues(hypre_SStructMatrixIJMatrix(fac_A), nUventries,
                           ncols, (const HYPRE_BigInt *) rows, (const HYPRE_BigInt *) cols,
                           (const HYPRE_Real *) values);
   hypre_TFree(ncols, HYPRE_MEMORY_HOST);
   hypre_TFree(rows, HYPRE_MEMORY_HOST);
   hypre_TFree(cols, HYPRE_MEMORY_HOST);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   owninfo = hypre_CTAlloc(hypre_SStructOwnInfoData  **,  nparts, HYPRE_MEMORY_HOST);
   for (part = (nparts - 1); part > 0; part--)
   {
      f_pgrid = hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = hypre_SStructGridPGrid(fac_grid, part - 1);

      nvars  = hypre_SStructPGridNVars(f_pgrid);
      owninfo[part] = hypre_CTAlloc(hypre_SStructOwnInfoData   *,  nvars, HYPRE_MEMORY_HOST);

      for (var1 = 0; var1 < nvars; var1++)
      {
         fboxman = hypre_SStructGridBoxManager(fac_grid, part, var1);
         cboxman = hypre_SStructGridBoxManager(fac_grid, part - 1, var1);

         fgrid = hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid = hypre_SStructPGridSGrid(c_pgrid, var1);

         owninfo[part][var1] = hypre_SStructOwnInfo(fgrid, cgrid, cboxman, fboxman,
                                                    rfactors[part]);
      }
   }

   hypre_SetIndex3(stride, 1, 1, 1);
   for (part = (nparts - 1); part > 0; part--)
   {
      f_pgrid = hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = hypre_SStructGridPGrid(fac_grid, part - 1);
      nvars  = hypre_SStructPGridNVars(f_pgrid);

      for (var1 = 0; var1 < nvars; var1++)
      {
         fgrid     = hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid     = hypre_SStructPGridSGrid(c_pgrid, var1);
         grid_boxes = hypre_StructGridBoxes(fgrid);

         stencils = hypre_SStructGraphStencil(graph, part, var1);
         stencil_size = hypre_SStructStencilSize(stencils);
         stencil_vars = hypre_SStructStencilVars(stencils);

         if (part == (nparts - 1)) /* copy all fine data */
         {
            pmatrix    = hypre_SStructMatrixPMatrix(A, part);
            fac_pmatrix = hypre_SStructMatrixPMatrix(fac_A, part);
            hypre_ForBoxI(i, grid_boxes)
            {
               grid_box = hypre_BoxArrayBox(grid_boxes, i);
               hypre_BoxGetSize(grid_box, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(grid_box), ilower);

               for (j = 0; j < stencil_size; j++)
               {
                  var2       = stencil_vars[j];
                  smatrix    = hypre_SStructPMatrixSMatrix(pmatrix, var1, var2);
                  fac_smatrix = hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  smatrix_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     hypre_BoxArrayBox(hypre_StructMatrixDataSpace(fac_smatrix), i);

                  hypre_CopyIndex(hypre_SStructStencilEntry(stencils, j), stencil_shape);
                  smatrix_vals = hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                         i,
                                                                         stencil_shape);
                  fac_smatrix_vals = hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                                             i,
                                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (j = 0; j < stencil_size; j++) */
            }     /* hypre_ForBoxI(i, grid_boxes) */
         }        /* if (part == (nparts-1)) */

         /*----------------------------------------------------------------------
          *  Copy all coarse data not underlying a fbox and on this processor-
          *  i.e., the own_composite_cbox data.
          *----------------------------------------------------------------------*/
         pmatrix    = hypre_SStructMatrixPMatrix(A, part - 1);
         fac_pmatrix = hypre_SStructMatrixPMatrix(fac_A, part - 1);

         own_composite_cboxes = hypre_SStructOwnInfoDataCompositeCBoxes(owninfo[part][var1]);

         stencils = hypre_SStructGraphStencil(graph, part - 1, var1);
         stencil_size = hypre_SStructStencilSize(stencils);
         stencil_vars = hypre_SStructStencilVars(stencils);

         hypre_ForBoxArrayI(i, own_composite_cboxes)
         {
            own_composite_cbox = hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i);
            hypre_ForBoxI(j, own_composite_cbox)
            {
               grid_box = hypre_BoxArrayBox(own_composite_cbox, j);
               hypre_BoxGetSize(grid_box, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(grid_box), ilower);

               for (k = 0; k < stencil_size; k++)
               {
                  var2       = stencil_vars[k];
                  smatrix    = hypre_SStructPMatrixSMatrix(pmatrix, var1, var2);
                  fac_smatrix = hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  smatrix_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     hypre_BoxArrayBox(hypre_StructMatrixDataSpace(fac_smatrix), i);

                  hypre_CopyIndex(hypre_SStructStencilEntry(stencils, k), stencil_shape);
                  smatrix_vals = hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                         i,
                                                                         stencil_shape);
                  fac_smatrix_vals = hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                                             i,
                                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (k = 0; k< stencil_size; k++) */
            }      /* hypre_ForBoxI(j, own_composite_cbox) */
         }          /* hypre_ForBoxArrayI(i, own_composite_cboxes) */

      }  /* for (var1= 0; var1< nvars; var1++) */
   }     /* for (part= (nparts-1); part> 0; part--) */

   /*--------------------------------------------------------------------------
    * All possible data has been copied into fac_A- i.e., the original amr
    * composite operator. Now we need to coarsen away the fboxes and the
    * interface connections.
    *
    * Algo.:
    *   Loop from the finest amr_level to amr_level 1
    *   {
    *      1) coarsen the cf connections to get stencil connections from
    *         the coarse nodes to the coarsened fbox nodes.
    *      2) coarsen the fboxes and the fc connections. These are coarsened
    *         into a temp SStruct_PMatrix whose grid is the coarsened fgrid.
    *      3) copy all coarsened data that belongs on this processor and
    *         communicate any that belongs to another processor.
    *   }
    *--------------------------------------------------------------------------*/
   for (part = (nparts - 1); part >= 1; part--)
   {
      hypre_AMR_CFCoarsen(A, fac_A, rfactors[part], part);

      /*-----------------------------------------------------------------------
       *  Create the temp SStruct_PMatrix for coarsening away the level= part
       *  boxes.
       *-----------------------------------------------------------------------*/
      f_pgrid = hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = hypre_SStructGridPGrid(fac_grid, part - 1);
      grid_boxes = hypre_SStructPGridCellIBoxArray(f_pgrid);

      hypre_SStructPGridCreate(hypre_SStructGridComm(f_pgrid),
                               ndim, &temp_pgrid);

      /*coarsen the fboxes.*/
      for (i = 0; i < hypre_BoxArraySize(grid_boxes); i++)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(grid_box), zero_index,
                                     rfactors[part], hypre_BoxIMin(&scaled_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(grid_box), zero_index,
                                     rfactors[part], hypre_BoxIMax(&scaled_box));

         hypre_SStructPGridSetExtents(temp_pgrid,
                                      hypre_BoxIMin(&scaled_box),
                                      hypre_BoxIMax(&scaled_box));
      }

      nvars  = hypre_SStructPGridNVars(f_pgrid);
      hypre_SStructPGridSetVariables(temp_pgrid, nvars,
                                     hypre_SStructPGridVarTypes(f_pgrid));
      hypre_SStructPGridAssemble(temp_pgrid);

      /* reference the sstruct_stencil of fac_pmatrix- to be used in temp_pmatrix */
      temp_sstencils = hypre_CTAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);
      fac_pmatrix = hypre_SStructMatrixPMatrix(fac_A, part - 1);
      for (i = 0; i < nvars; i++)
      {
         hypre_SStructStencilRef(hypre_SStructPMatrixStencil(fac_pmatrix, i),
                                 &temp_sstencils[i]);
      }

      hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(fac_pmatrix),
                                 temp_pgrid,
                                 temp_sstencils,
                                 &temp_pmatrix);
      hypre_SStructPMatrixInitialize(temp_pmatrix);

      hypre_AMR_FCoarsen(A, fac_A, temp_pmatrix, rfactors[part], part);

      /*-----------------------------------------------------------------------
       * Extract the own_box data (boxes of coarsen data of this processor).
       *-----------------------------------------------------------------------*/
      fac_pmatrix = hypre_SStructMatrixPMatrix(fac_A, part - 1);
      for (var1 = 0; var1 < nvars; var1++)
      {
         stencils = hypre_SStructGraphStencil(graph, part - 1, var1);
         stencil_size = hypre_SStructStencilSize(stencils);
         stencil_vars = hypre_SStructStencilVars(stencils);

         own_boxes = hypre_SStructOwnInfoDataOwnBoxes(owninfo[part][var1]);
         own_cboxnums = hypre_SStructOwnInfoDataOwnBoxNums(owninfo[part][var1]);
         size = hypre_SStructOwnInfoDataSize(owninfo[part][var1]);

         /* loop over all the cbox chunks */
         for (i = 0; i < size; i++)
         {
            cgrid_boxes = hypre_BoxArrayArrayBoxArray(own_boxes, i);
            hypre_ForBoxI(j, cgrid_boxes)
            {
               grid_box = hypre_BoxArrayBox(cgrid_boxes, j);
               hypre_BoxGetSize(grid_box, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(grid_box), ilower);

               cbox = own_cboxnums[i][j];

               for (k = 0; k < stencil_size; k++)
               {
                  var2 = stencil_vars[k];
                  smatrix = hypre_SStructPMatrixSMatrix(temp_pmatrix, var1, var2);
                  fac_smatrix = hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  /*---------------------------------------------------------------
                   * note: the cbox number of the temp_grid is the same as the
                   * fbox number, whereas the cbox numbers of the fac_grid is in
                   * own_cboxnums- i.e., numbers i & cbox, respectively.
                   *---------------------------------------------------------------*/
                  smatrix_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     hypre_BoxArrayBox(hypre_StructMatrixDataSpace(fac_smatrix), cbox);

                  hypre_CopyIndex(hypre_SStructStencilEntry(stencils, k), stencil_shape);
                  smatrix_vals =
                     hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                             i,
                                                             stencil_shape);
                  fac_smatrix_vals =
                     hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                             cbox,
                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (k = 0; k < stencil_size; k++) */
            }     /* hypre_ForBoxI(j, cgrid_boxes) */
         }        /* for (i= 0; i< size; i++) */

         hypre_SStructOwnInfoDataDestroy(owninfo[part][var1]);
      }           /* for (var1= 0; var1< nvars; var1++) */

      hypre_TFree(owninfo[part], HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       * Communication of off-process coarse data. A communication pkg is
       * needed. Thus, compute the communication info- sendboxes, recvboxes,
       * etc.
       *-----------------------------------------------------------------------*/
      for (var1 = 0; var1 < nvars; var1++)
      {
         fboxman = hypre_SStructGridBoxManager(fac_grid, part, var1);
         cboxman = hypre_SStructGridBoxManager(fac_grid, part - 1, var1);

         fgrid = hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid = hypre_SStructPGridSGrid(c_pgrid, var1);

         sendinfo = hypre_SStructSendInfo(fgrid, cboxman, rfactors[part]);
         recvinfo = hypre_SStructRecvInfo(cgrid, fboxman, rfactors[part]);

         /*-------------------------------------------------------------------
          * need to check this for more than one variable- are the comm. info
          * for this sgrid okay for cross-variable matrices?
          *-------------------------------------------------------------------*/
         for (var2 = 0; var2 < nvars; var2++)
         {
            fac_smatrix = hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);
            smatrix    = hypre_SStructPMatrixSMatrix(temp_pmatrix, var1, var2);

            hypre_SStructAMRInterCommunication(sendinfo,
                                               recvinfo,
                                               hypre_StructMatrixDataSpace(smatrix),
                                               hypre_StructMatrixDataSpace(fac_smatrix),
                                               hypre_StructMatrixNumValues(smatrix),
                                               comm,
                                               &amrA_comm_pkg);

            hypre_InitializeCommunication(amrA_comm_pkg,
                                          hypre_StructMatrixData(smatrix),
                                          hypre_StructMatrixData(fac_smatrix), 0, 0,
                                          &comm_handle);
            hypre_FinalizeCommunication(comm_handle);

            hypre_CommPkgDestroy(amrA_comm_pkg);
         }

         hypre_SStructSendInfoDataDestroy(sendinfo);
         hypre_SStructRecvInfoDataDestroy(recvinfo);

      }  /* for (var1= 0; var1< nvars; var1++) */

      hypre_SStructPGridDestroy(temp_pgrid);
      hypre_SStructPMatrixDestroy(temp_pmatrix);

   }  /* for (part= 0; part< nparts; part++) */

   hypre_TFree(owninfo, HYPRE_MEMORY_HOST);

   HYPRE_SStructMatrixAssemble(fac_A);

   *fac_A_ptr = fac_A;
   return ierr;
}
