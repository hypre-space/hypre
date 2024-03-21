/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   j, k (only where they are listed at the end of SMP_PRIVATE)
 *
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

hypre_IJMatrix *
hypre_Maxwell_PNedelec( hypre_SStructGrid    *fgrid_edge,
                        hypre_SStructGrid    *cgrid_edge,
                        hypre_Index           rfactor    )
{
   MPI_Comm               comm = (fgrid_edge->  comm);

   HYPRE_IJMatrix         edge_Edge;

   hypre_SStructPGrid    *p_cgrid, *p_fgrid;
   hypre_StructGrid      *var_cgrid,  *var_fgrid;
   hypre_BoxArray        *cboxes, *fboxes, *box_array;
   hypre_Box             *cbox, *fbox, *cellbox, *vbox, copy_box;

   hypre_BoxArray       **contract_fedgeBoxes;
   hypre_Index          **Edge_cstarts, **upper_shifts, **lower_shifts;
   HYPRE_Int            **cfbox_mapping, **fcbox_mapping;

   hypre_BoxManEntry     *entry;
   HYPRE_BigInt           rank, rank2;
   HYPRE_BigInt           start_rank1, start_rank2;

   HYPRE_Int              nedges;

   HYPRE_BigInt          *iedgeEdge;
   HYPRE_BigInt          *jedge_Edge;

   HYPRE_Real            *vals_edgeEdge;
   HYPRE_Real             fCedge_ratio;
   HYPRE_Int             *ncols_edgeEdge;

   hypre_Index            cindex;
   hypre_Index            findex;
   hypre_Index            var_index, *boxoffset, *suboffset;
   hypre_Index            loop_size, start, cstart, stride, hi_index, lindex;
   hypre_Index            ishift, jshift, kshift, zero_index, one_index;
   HYPRE_Int              n_boxoffsets;

   HYPRE_Int              nparts = hypre_SStructGridNParts(fgrid_edge);
   HYPRE_Int              ndim  = hypre_SStructGridNDim(fgrid_edge);

   HYPRE_SStructVariable *vartypes, *Edge_vartypes;
   hypre_Index           *varoffsets;
   HYPRE_Int             *vartype_map;
   HYPRE_Int              matrix_type = HYPRE_PARCSR;

   HYPRE_Int              nvars, Edge_nvars, part, var;
   HYPRE_Int              tot_vars = 8;

   HYPRE_Int              t, i, j, k, m, n, size;
   HYPRE_BigInt           l, p;

   HYPRE_BigInt           ilower, iupper;
   HYPRE_BigInt           jlower, jupper;
   HYPRE_BigInt         **lower_ranks, **upper_ranks;

   HYPRE_Int           ***n_CtoVbox, ****CtoVboxnums;
   HYPRE_Int             *num_vboxes, **vboxnums;

   HYPRE_Int              trueV = 1;
   HYPRE_Int              falseV = 0;
   HYPRE_Int              row_in;

   HYPRE_Int              myproc;

   hypre_BoxInit(&copy_box, ndim);

   hypre_MPI_Comm_rank(comm, &myproc);
   hypre_SetIndex3(ishift, 1, 0, 0);
   hypre_SetIndex3(jshift, 0, 1, 0);
   hypre_SetIndex3(kshift, 0, 0, 1);
   hypre_SetIndex(zero_index, 0);
   hypre_SetIndex(one_index, 1);
   hypre_SetIndex(lindex, 0);

   /* set rfactor[2]= 1 if ndim=2. */
   if (ndim == 2)
   {
      rfactor[2] = 1;
   }

   /*-------------------------------------------------------------------
    * Find the coarse-fine connection pattern, i.e., the topology
    * needed to create the interpolation operators.
    * These connections are determined using the cell-centred grids.
    * Note that we are assuming the variable type enumeration
    * given in hypre_SStructVariable_enum.
    *
    * We consider both 2-d and 3-d cases. In 2-d, the edges are faces.
    * We will continue to call them edges, but use the face variable
    * enumeration.
    *-------------------------------------------------------------------*/
   varoffsets = hypre_CTAlloc(hypre_Index,  tot_vars, HYPRE_MEMORY_HOST);

   /* total of 8 variable types. Create a mapping between user enumeration
      to hypre enumeration. Only need for edge grids. */
   vartype_map = hypre_CTAlloc(HYPRE_Int,  tot_vars, HYPRE_MEMORY_HOST);

   part = 0;
   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   vartypes = hypre_SStructPGridVarTypes(p_cgrid);

   for (i = 0; i < nvars; i++)
   {
      t = vartypes[i];
      hypre_SStructVariableGetOffset((hypre_SStructVariable) t,
                                     ndim, varoffsets[t]);
      switch (t)
      {
         case 2:
         {
            vartype_map[2] = i;
            break;
         }

         case 3:
         {
            vartype_map[3] = i;
            break;
         }

         case 5:
         {
            vartype_map[5] = i;
            break;
         }

         case 6:
         {
            vartype_map[6] = i;
            break;
         }

         case 7:
         {
            vartype_map[7] = i;
            break;
         }
      }
   }

   /* local sizes */
   nedges   = 0;
   for (part = 0; part < nparts; part++)
   {
      /* same for 2-d & 3-d, assuming that fgrid_edge= fgrid_face in input */
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);    /* edge fgrid */
      nvars   = hypre_SStructPGridNVars(p_fgrid);

      for (var = 0; var < nvars; var++)
      {
         var_fgrid = hypre_SStructPGridSGrid(p_fgrid, var);
         nedges  += hypre_StructGridLocalSize(var_fgrid);
      }
   }

   /*--------------------------------------------------------------------------
    *  Form mappings between the c & f box numbers. Note that a cbox
    *  can land inside only one fbox since the latter was contracted. Without
    *  the extraction, a cbox can land in more than 1 fboxes (e.g., cbox
    *  boundary extending into other fboxes).
    *--------------------------------------------------------------------------*/
   cfbox_mapping = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   fcbox_mapping = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   for (i = 0; i < nparts; i++)
   {
      p_fgrid  = hypre_SStructGridPGrid(fgrid_edge, i);
      var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      j        = hypre_BoxArraySize(fboxes);
      fcbox_mapping[i] = hypre_CTAlloc(HYPRE_Int,  j, HYPRE_MEMORY_HOST);

      p_cgrid  = hypre_SStructGridPGrid(cgrid_edge, i);
      var_cgrid = hypre_SStructPGridCellSGrid(p_cgrid);
      cboxes   = hypre_StructGridBoxes(var_cgrid);
      j        = hypre_BoxArraySize(fboxes);
      cfbox_mapping[i] = hypre_CTAlloc(HYPRE_Int,  j, HYPRE_MEMORY_HOST);

      /* assuming if i1 > i2 and (box j1) is coarsened from (box i1)
         and (box j2) from (box i2), then j1 > j2. */
      k = 0;
      hypre_ForBoxI(j, fboxes)
      {
         fbox = hypre_BoxArrayBox(fboxes, j);
         hypre_CopyBox(fbox, &copy_box);
         hypre_ProjectBox(&copy_box, zero_index, rfactor);
         hypre_StructMapFineToCoarse(hypre_BoxIMin(&copy_box), zero_index,
                                     rfactor, hypre_BoxIMin(&copy_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(&copy_box), zero_index,
                                     rfactor, hypre_BoxIMax(&copy_box));

         /* since the ordering of the cboxes was determined by the fbox
            ordering, we only have to check if the first cbox in the
            list intersects with copy_box. If not, this fbox vanished in the
            coarsening. Note that this gives you the correct interior cbox. */
         cbox = hypre_BoxArrayBox(cboxes, k);
         hypre_IntersectBoxes(&copy_box, cbox, &copy_box);
         if (hypre_BoxVolume(&copy_box))
         {
            cfbox_mapping[i][k] = j;
            fcbox_mapping[i][j] = k;
            k++;
         }  /* if (hypre_BoxVolume(&copy_box)) */
      }     /* hypre_ForBoxI(j, fboxes) */
   }        /* for (i= 0; i< nparts; i++) */

   /* variable rank bounds for this processor */
   n_CtoVbox   = hypre_TAlloc(HYPRE_Int **,  nparts, HYPRE_MEMORY_HOST);
   CtoVboxnums = hypre_TAlloc(HYPRE_Int ***,  nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructCellGridBoxNumMap(fgrid_edge, part, &n_CtoVbox[part],
                                     &CtoVboxnums[part]);
   }

   /* variable rank bounds for this processor */
   lower_ranks = hypre_TAlloc(HYPRE_BigInt *,  nparts, HYPRE_MEMORY_HOST);
   upper_ranks = hypre_TAlloc(HYPRE_BigInt *,  nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      p_fgrid  = hypre_SStructGridPGrid(fgrid_edge, part);
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);

      lower_ranks[part] = hypre_CTAlloc(HYPRE_BigInt,  Edge_nvars, HYPRE_MEMORY_HOST);
      upper_ranks[part] = hypre_CTAlloc(HYPRE_BigInt,  Edge_nvars, HYPRE_MEMORY_HOST);
      for (t = 0; t < Edge_nvars; t++)
      {
         var_fgrid = hypre_SStructPGridSGrid(p_fgrid, t);
         box_array = hypre_StructGridBoxes(var_fgrid);

         fbox     = hypre_BoxArrayBox(box_array, 0);
         hypre_CopyIndex(hypre_BoxIMin(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &lower_ranks[part][t],
                                               matrix_type);

         fbox = hypre_BoxArrayBox(box_array, hypre_BoxArraySize(box_array) - 1);
         hypre_CopyIndex(hypre_BoxIMax(fbox), findex);
         hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t,
                                          &entry);
         hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &upper_ranks[part][t],
                                               matrix_type);
      }
   }

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information.
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */

   /* edge_Edge. Same for 2-d and 3-d. */
   /* lower rank */
   start_rank1 = hypre_SStructGridStartRank(fgrid_edge);
   start_rank2 = hypre_SStructGridStartRank(cgrid_edge);
   ilower     = start_rank1;
   jlower     = start_rank2;

   /* upper rank */
   part = nparts - 1;
   p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_fgrid);
   var_fgrid = hypre_SStructPGridSGrid(p_fgrid, nvars - 1);
   fboxes   = hypre_StructGridBoxes(var_fgrid);
   fbox    = hypre_BoxArrayBox(fboxes, hypre_BoxArraySize(fboxes) - 1);

   hypre_SStructGridBoxProcFindBoxManEntry(fgrid_edge, part, nvars - 1,
                                           hypre_BoxArraySize(fboxes) - 1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(fbox), &iupper);

   p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part);
   nvars   = hypre_SStructPGridNVars(p_cgrid);
   var_cgrid = hypre_SStructPGridSGrid(p_cgrid, nvars - 1);
   cboxes   = hypre_StructGridBoxes(var_cgrid);
   cbox    = hypre_BoxArrayBox(cboxes, hypre_BoxArraySize(cboxes) - 1);

   hypre_SStructGridBoxProcFindBoxManEntry(cgrid_edge, part, nvars - 1,
                                           hypre_BoxArraySize(cboxes) - 1, myproc, &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(cbox), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &edge_Edge);
   HYPRE_IJMatrixSetObjectType(edge_Edge, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(edge_Edge);

   /*-----------------------------------------------------------------------
    * edge_Edge, the actual interpolation matrix.
    * For each fine edge row, we need to know if it is a edge,
    * boundary edge, or face edge. Knowing this allows us to determine the
    * structure and weights of the interpolation matrix.
    * We assume that a coarse edge interpolates only to fine edges in or on
    * an agglomerate. That is, fine edges with indices that do were
    * truncated do not get interpolated to.
    * Scheme: Loop over fine edge grid. For each fine edge ijk,
    *     1) map it to a fine cell with the fine edge at the lower end
    *        of the box,e.g. x_edge[ijk] -> cell[i,j+1,k+1].
    *     2) coarsen the fine cell to obtain a coarse cell. Determine the
    *        location of the fine edge with respect to the coarse edges
    *        of this cell. Coarsening needed only when determining the
    *        column rank.
    * Need to distinguish between 2-d and 3-d.
    *-----------------------------------------------------------------------*/

   /* count the row/col connections */
   iedgeEdge     = hypre_CTAlloc(HYPRE_BigInt,  nedges, HYPRE_MEMORY_HOST);
   ncols_edgeEdge = hypre_CTAlloc(HYPRE_Int,  nedges, HYPRE_MEMORY_HOST);

   /* get the contracted boxes */
   contract_fedgeBoxes = hypre_TAlloc(hypre_BoxArray *,  nparts, HYPRE_MEMORY_HOST);
   Edge_cstarts = hypre_TAlloc(hypre_Index *,  nparts, HYPRE_MEMORY_HOST);
   upper_shifts = hypre_TAlloc(hypre_Index *,  nparts, HYPRE_MEMORY_HOST);
   lower_shifts = hypre_TAlloc(hypre_Index *,  nparts, HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      p_fgrid  = hypre_SStructGridPGrid(fgrid_edge, part);
      var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);

      /* fill up the contracted box_array */
      contract_fedgeBoxes[part] = hypre_BoxArrayCreate(0, ndim);
      Edge_cstarts[part] = hypre_TAlloc(hypre_Index,  hypre_BoxArraySize(fboxes), HYPRE_MEMORY_HOST);
      upper_shifts[part] = hypre_TAlloc(hypre_Index,  hypre_BoxArraySize(fboxes), HYPRE_MEMORY_HOST);
      lower_shifts[part] = hypre_TAlloc(hypre_Index,  hypre_BoxArraySize(fboxes), HYPRE_MEMORY_HOST);

      hypre_ForBoxI(i, fboxes)
      {
         fbox = hypre_BoxArrayBox(fboxes, i);

         /* contract the fbox to correspond to the correct cbox */
         cbox = hypre_BoxContraction(fbox, var_fgrid, rfactor);
         hypre_AppendBox(cbox, contract_fedgeBoxes[part]);

         /* record the offset mapping between the coarse cell index and
            the fine cell index */
         hypre_ClearIndex(upper_shifts[part][i]);
         hypre_ClearIndex(lower_shifts[part][i]);
         for (k = 0; k < ndim; k++)
         {
            m = hypre_BoxIMin(cbox)[k];
            p = m % rfactor[k];
            if (p > 0 && m > 0)
            {
               upper_shifts[part][i][k] = p - 1;
               lower_shifts[part][i][k] = p - rfactor[k];
            }
            else
            {
               upper_shifts[part][i][k] = rfactor[k] - p - 1;
               lower_shifts[part][i][k] = -p;
            }
         }

         /* record the cstarts of the cbox */
         hypre_ProjectBox(cbox, zero_index, rfactor);
         hypre_CopyIndex(hypre_BoxIMin(cbox), Edge_cstarts[part][i]);
         hypre_StructMapFineToCoarse(Edge_cstarts[part][i], zero_index, rfactor,
                                     Edge_cstarts[part][i]);

         hypre_BoxDestroy(cbox);
      }

   }  /* for (part= 0; part< nparts; part++) */

   /*-----------------------------------------------------------------------
    * loop first over the fedges aligning with the agglomerate coarse edges.
    * Will loop over the face & interior edges separately also.
    *-----------------------------------------------------------------------*/
   j = 0;
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var         = Edge_vartypes[t];
         var_fgrid   = hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array   = hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets = ndim - 1;
         boxoffset   = hypre_CTAlloc(hypre_Index,  n_boxoffsets, HYPRE_MEMORY_HOST);
         suboffset   = hypre_CTAlloc(hypre_Index,  n_boxoffsets, HYPRE_MEMORY_HOST);
         switch (var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1] */
            {
               hypre_SetIndex3(stride, rfactor[0], 1, 1);
               hypre_CopyIndex(varoffsets[2], var_index);

               /* boxoffset shrink in the i direction */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1] */
            {
               hypre_SetIndex3(stride, 1, rfactor[1], 1);
               hypre_CopyIndex(varoffsets[3], var_index);

               /* boxoffset shrink in the j direction */
               hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]] */
            {
               hypre_SetIndex3(stride, 1, rfactor[1], rfactor[2]);
               hypre_CopyIndex(varoffsets[5], var_index);

               /* boxoffset shrink in the j & k directions */
               hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               hypre_SetIndex3(suboffset[0], 0, 1, 0);
               hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]] */
            {
               hypre_SetIndex3(stride, rfactor[0], 1, rfactor[2]);
               hypre_CopyIndex(varoffsets[6], var_index);

               /* boxoffset shrink in the i & k directions */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);
               hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 0, 1);
               break;
            }

            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1] */
            {
               hypre_SetIndex3(stride, rfactor[0], rfactor[1], 1);
               hypre_CopyIndex(varoffsets[7], var_index);

               /* boxoffset shrink in the i & j directions */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(boxoffset[1], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);
               hypre_SetIndex3(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 1, 0);
               break;
            }
         }

         hypre_ForBoxI(i, fboxes)
         {
            cellbox = hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes = n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            /* adjust the project cellbox to the variable box */
            hypre_CopyBox(cellbox, &copy_box);

            /* the adjusted variable box may be bigger than the actually
               variable box- variables that are shared may lead to smaller
               variable boxes than the SubtractIndex produces. If the box
               has to be decreased, then we decrease it by (rfactor[j]-1)
               in the appropriate direction.
               Check the location of the shifted lower box index. */
            for (k = 0; k < n_boxoffsets; k++)
            {
               hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), suboffset[k], 3,
                                     findex);
               row_in = falseV;
               for (p = 0; p < num_vboxes[t]; p++)
               {
                  vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (hypre_IndexInBox(findex, vbox))
                  {
                     hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                     row_in = trueV;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[k], 3,
                                   hypre_BoxIMin(&copy_box));
               }
            }

            hypre_BoxGetSize(&copy_box, loop_size);
            hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);
            /* extend the loop_size so that upper boundary of the box are reached. */
            hypre_AddIndexes(loop_size, hi_index, 3, loop_size);

            hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

            hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                      &copy_box, start, stride, m);
            {
               zypre_BoxLoopGetIndex(lindex);
               hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
               for (k = 0; k < 3; k++)
               {
                  findex[k] *= stride[k];
               }
               hypre_AddIndexes(findex, start, 3, findex);

               hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &p, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ( (p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t]) )
               {
                  iedgeEdge[j] = p;
                  ncols_edgeEdge[j] = 1;
                  j++;
               }
            }
            hypre_SerialBoxLoop1End(m);

         }   /* hypre_ForBoxI */

         hypre_TFree(boxoffset, HYPRE_MEMORY_HOST);
         hypre_TFree(suboffset, HYPRE_MEMORY_HOST);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /*-----------------------------------------------------------------------
    * Record the row ranks for the face edges. Only for 3-d.
    * Loop over the face edges.
    *-----------------------------------------------------------------------*/
   if (ndim == 3)
   {
      for (part = 0; part < nparts; part++)
      {
         p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
         Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes = contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset = hypre_TAlloc(hypre_Index,  ndim, HYPRE_MEMORY_HOST);
         for (t = 0; t < ndim; t++)
         {
            hypre_ClearIndex(boxoffset[t]);
            hypre_IndexD(boxoffset[t], t) = rfactor[t] - 1;
         }

         for (t = 0; t < Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid = hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array = hypre_StructGridBoxes(var_fgrid);

            /* to reduce comparison, take the switch outside of the loop */
            switch (var)
            {
               case 5:
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the contracted cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;
                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /************************************************************
                         * Loop over the Z_Face x_edges.
                         ************************************************************/
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);

                              /* still row l may be outside the processor */
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);
                     loop_size[1]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /* Y_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z_Face direction to
                        cover upper boundary Z_Faces. */
                     loop_size[2]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        hypre_AddIndexes(findex, start, 3, findex);

                        /* Z_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Z_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* X_Face */
                     hypre_CopyBox(cellbox, &copy_box);

                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        hypre_AddIndexes(findex, start, 3, findex);

                        /* X_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }

                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be interior, X or Y_Face, or Z_Edge */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X_Face direction */
                     loop_size[0]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        hypre_AddIndexes(findex, start, 3, findex);

                        /* X_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* X_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* Y_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         hypre_BoxIMin(&copy_box));
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        for (k = 0; k < 3; k++)
                        {
                           findex[k] *= rfactor[k];
                        }

                        hypre_AddIndexes(findex, start, 3, findex);

                        /* Y_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 iedgeEdge[j] = l;

                                 /* Y_Face. Two coarse Edge connections. */
                                 ncols_edgeEdge[j] = 2;
                                 j++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */

                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */

         hypre_TFree(boxoffset, HYPRE_MEMORY_HOST);
      }  /* for (part= 0; part< nparts; part++) */
   }     /* if (ndim == 3) */

   for (part = 0; part < nparts; part++)
   {
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid = hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = hypre_StructGridBoxes(var_fgrid);

         /* to reduce comparison, take the switch outside of the loop */
         switch (var)
         {
            case 2:
            {
               /* 2-d x_face = x_edge, can be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);

                  /* adjust the contract cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /*hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[0]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[0] += p;
                        for (n = 0; n < rfactor[1]; n++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                 matrix_type);
                           iedgeEdge[j] = l;

                           /* lies interior of Face. Two coarse Edge connection. */
                           ncols_edgeEdge[j] = 2;
                           j++;

                           var_index[1]++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  hypre_SerialBoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 3:
            {
               /* 2-d y_face = y_edge, can be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }

                     hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[1] += p;
                        for (n = 0; n < rfactor[0]; n++)
                        {
                           hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                 matrix_type);
                           iedgeEdge[j] = l;

                           /* lies interior of Face. Two coarse Edge connection. */
                           ncols_edgeEdge[j] = 2;
                           j++;

                           var_index[0]++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  hypre_SerialBoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 5:
            {
               /* 3-d x_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[2] += p;
                        for (n = 1; n < rfactor[1]; n++)
                        {
                           var_index[1]++;
                           for (k = 0; k < rfactor[0]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[0]++;
                           }  /* for (k= 0; k< rfactor[0]; k++) */

                           /* reset var_index[0] to the initial index for next k loop */
                           var_index[0] -= rfactor[0];

                        }  /* for (n= 1; n< rfactor[1]; n++) */

                        /* reset var_index[1] to the initial index for next n loop */
                        var_index[1] -= (rfactor[1] - 1);
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_SerialBoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 6:
            {
               /* 3-d y_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[2] += p;
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k = 0; k < rfactor[1]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[1]++;
                           }  /* for (k= 0; k< rfactor[1]; k++) */

                           /* reset var_index[1] to the initial index for next k loop */
                           var_index[1] -= rfactor[1];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0] -= (rfactor[0] - 1);
                     }  /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_SerialBoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */

               break;
            }

            case 7:
            {
               /* 3-d z_edge, can be only interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, m);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
                     for (k = 0; k < 3; k++)
                     {
                        findex[k] *= rfactor[k];
                     }
                     hypre_AddIndexes(findex, start, 3, findex);

                     /* get interior edges */
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        hypre_CopyIndex(findex, var_index);
                        var_index[1] += p;
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           var_index[0]++;
                           for (k = 0; k < rfactor[2]; k++)
                           {
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              iedgeEdge[j] = l;

                              /* Interior. Four coarse Edge connections. */
                              ncols_edgeEdge[j] = 4;
                              j++;

                              var_index[2]++;
                           }  /* for (k= 0; k< rfactor[2]; k++) */

                           /* reset var_index[2] to the initial index for next k loop */
                           var_index[2] -= rfactor[2];

                        }  /* for (n= 1; n< rfactor[0]; n++) */

                        /* reset var_index[0] to the initial index for next n loop */
                        var_index[0] -= (rfactor[0] - 1);
                     }  /* for (p= 1; p< rfactor[1]; p++) */
                  }
                  hypre_SerialBoxLoop1End(m);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   k = 0;
   j = 0;
   for (i = 0; i < nedges; i++)
   {
      if (ncols_edgeEdge[i])
      {
         k += ncols_edgeEdge[i];
         j++;
      }
   }
   vals_edgeEdge = hypre_CTAlloc(HYPRE_Real,  k, HYPRE_MEMORY_HOST);
   jedge_Edge    = hypre_CTAlloc(HYPRE_BigInt,  k, HYPRE_MEMORY_HOST);

   /* update nedges so that the true number of rows is set */
   size = j;

   /*********************************************************************
    * Fill up the edge_Edge interpolation matrix. Interpolation weights
    * are determined differently for each type of fine edges.
    *********************************************************************/

   /* loop over fedges aligning with the agglomerate coarse edges first. */
   k = 0;
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid = hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = hypre_StructGridBoxes(var_fgrid);

         n_boxoffsets = ndim - 1;
         boxoffset   = hypre_CTAlloc(hypre_Index,  n_boxoffsets, HYPRE_MEMORY_HOST);
         suboffset   = hypre_CTAlloc(hypre_Index,  n_boxoffsets, HYPRE_MEMORY_HOST);
         switch (var)
         {
            case 2: /* 2-d: x_face (vertical edges), stride=[rfactor[0],1,1]
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               hypre_SetIndex3(stride, rfactor[0], 1, 1);
               fCedge_ratio = 1.0 / rfactor[1];

               /* boxoffset shrink in the i direction */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 0, 0);
               break;
            }

            case 3: /* 2-d: y_face (horizontal edges), stride=[1,rfactor[1],1]
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               hypre_SetIndex3(stride, 1, rfactor[1], 1);
               fCedge_ratio = 1.0 / rfactor[0];

               /* boxoffset shrink in the j direction */
               hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(suboffset[0], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 0, 1, 0);
               break;
            }

            case 5: /* 3-d: x_edge, stride=[1,rfactor[1],rfactor[2]]
                       fCedge_ratio= 1.0/rfactor[0] */
            {
               hypre_SetIndex3(stride, 1, rfactor[1], rfactor[2]);
               fCedge_ratio = 1.0 / rfactor[0];

               /* boxoffset shrink in the j & k directions */
               hypre_SetIndex3(boxoffset[0], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               hypre_SetIndex3(suboffset[0], 0, 1, 0);
               hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 0, 1, 1);
               break;
            }

            case 6: /* 3-d: y_edge, stride=[rfactor[0],1,rfactor[2]]
                       fCedge_ratio= 1.0/rfactor[1] */
            {
               hypre_SetIndex3(stride, rfactor[0], 1, rfactor[2]);
               fCedge_ratio = 1.0 / rfactor[1];

               /* boxoffset shrink in the i & k directions */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(boxoffset[1], 0, 0, rfactor[2] - 1);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);
               hypre_SetIndex3(suboffset[1], 0, 0, 1);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 0, 1);
               break;
            }
            case 7: /* 3-d: z_edge, stride=[rfactor[0],rfactor[1],1]
                       fCedge_ratio= 1.0/rfactor[2] */
            {
               hypre_SetIndex3(stride, rfactor[0], rfactor[1], 1);
               fCedge_ratio = 1.0 / rfactor[2];

               /* boxoffset shrink in the i & j directions */
               hypre_SetIndex3(boxoffset[0], rfactor[0] - 1, 0, 0);
               hypre_SetIndex3(boxoffset[1], 0, rfactor[1] - 1, 0);
               hypre_SetIndex3(suboffset[0], 1, 0, 0);
               hypre_SetIndex3(suboffset[1], 0, 1, 0);

               /* extend loop_size by one in the stride direction */
               hypre_SetIndex3(hi_index, 1, 1, 0);
               break;
            }
            default:
            {
               fCedge_ratio = 1.0;
            }
         }

         hypre_ForBoxI(i, fboxes)
         {
            cellbox = hypre_BoxArrayBox(fboxes, i);

            /* vboxes inside the i'th cellbox */
            num_vboxes = n_CtoVbox[part][i];
            vboxnums  = CtoVboxnums[part][i];

            hypre_CopyIndex(Edge_cstarts[part][i], cstart);

            /* adjust the contracted cellbox to the variable box.
               Note that some of the fboxes may be skipped because they
               vanish. */
            hypre_CopyBox(cellbox, &copy_box);

            for (j = 0; j < n_boxoffsets; j++)
            {
               hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), suboffset[j], 3,
                                     findex);
               row_in = falseV;
               for (p = 0; p < num_vboxes[t]; p++)
               {
                  vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);

                  if (hypre_IndexInBox(findex, vbox))
                  {
                     hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                     row_in = trueV;
                     break;
                  }
               }
               /* not in any vbox */
               if (!row_in)
               {
                  hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[j], 3,
                                   hypre_BoxIMin(&copy_box));

                  /* also modify cstart */
                  hypre_AddIndexes(boxoffset[j], one_index, 3, boxoffset[j]);
                  hypre_StructMapFineToCoarse(boxoffset[j], zero_index, rfactor,
                                              boxoffset[j]);
                  hypre_AddIndexes(cstart, boxoffset[j], 3, cstart);
               }
            }

            hypre_BoxGetSize(&copy_box, loop_size);
            hypre_StructMapFineToCoarse(loop_size, zero_index, stride,
                                        loop_size);

            /* extend the loop_size so that upper boundary of the box are reached. */
            hypre_AddIndexes(loop_size, hi_index, 3, loop_size);

            hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

            /* note that the correct cbox corresponding to this non-vanishing
               fbox is used. */
            hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                      &copy_box, start, stride, m);
            {
               zypre_BoxLoopGetIndex(lindex);
               hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);
               for (j = 0; j < 3; j++)
               {
                  findex[j] *= stride[j];
               }

               /* make sure that we do have the fine row corresponding to findex */
               hypre_AddIndexes(findex, start, 3, findex);
               hypre_SStructGridFindBoxManEntry(fgrid_edge, part, findex, t, &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, findex, &p, matrix_type);

               /* still row p may be outside the processor- check to make sure in */
               if ( (p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t]) )
               {
                  hypre_SubtractIndexes(findex, start, 3, findex);

                  /* determine where the edge lies- coarsening required. */
                  hypre_StructMapFineToCoarse(findex, zero_index, rfactor,
                                              cindex);
                  hypre_AddIndexes(cindex, cstart, 3, cindex);

                  /* lies on coarse Edge. Coarse Edge connection:
                     var_index= cindex - subtract_index.*/
                  hypre_SubtractIndexes(cindex, varoffsets[var], 3, var_index);

                  hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                   t, &entry);
                  hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                        matrix_type);
                  jedge_Edge[k] = l;
                  vals_edgeEdge[k] = fCedge_ratio;

                  k++;
               }  /* if ((p <= upper_ranks[part][t]) && (p >= lower_ranks[part][t])) */
            }
            hypre_SerialBoxLoop1End(m);
         }   /* hypre_ForBoxI */

         hypre_TFree(boxoffset, HYPRE_MEMORY_HOST);
         hypre_TFree(suboffset, HYPRE_MEMORY_HOST);
      }  /* for (t= 0; t< nvars; t++) */
   }     /* for (part= 0; part< nparts; part++) */

   /* generate the face interpolation weights/info. Only for 3-d */
   if (ndim == 3)
   {
      for (part = 0; part < nparts; part++)
      {
         p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
         Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
         Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);
         p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

         /* note that fboxes are the contracted CELL boxes. Will get the correct
            variable grid extents. */
         fboxes = contract_fedgeBoxes[part];

         /* may need to shrink a given box in some boxoffset directions */
         boxoffset = hypre_TAlloc(hypre_Index,  ndim, HYPRE_MEMORY_HOST);
         for (t = 0; t < ndim; t++)
         {
            hypre_ClearIndex(boxoffset[t]);
            hypre_IndexD(boxoffset[t], t) = rfactor[t] - 1;
         }

         for (t = 0; t < Edge_nvars; t++)
         {
            var      = Edge_vartypes[t];
            var_fgrid =  hypre_SStructPGridVTSGrid(p_fgrid, var);
            box_array = hypre_StructGridBoxes(var_fgrid);

            switch (var)
            {
               case 5:
               {
                  /* 3-d x_edge, can be Y or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         x_edge-> Z_Face & Y_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndexes(cstart, kshift, 3, cstart);
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * Z_Face (i,j,k-1). Two like-var coarse Edge connections.
                         * x_Edge (i,j,k-1), (i,j-1,k-1)
                         ******************************************************/
                        hypre_SubtractIndexes(cindex, kshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Z_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);

                              /* still row l may be outside the processor */
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[1] * rfactor[0]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[0] * (1.0 - (HYPRE_Real) n / rfactor[1]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* Y plane direction */
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         hypre_BoxIMin(&copy_box));

                        /* modify cstart */
                        hypre_AddIndexes(cstart, jshift, 3, cstart);
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * Y_Face. Two coarse Edge connections.
                         * x_Edge (i,j-1,k), (i,j-1,k-1)
                         ******************************************************/
                        hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of x_edges making up the Y_Face */
                        for (p = 0; p < rfactor[0]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[0] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[0] * (1.0 - (HYPRE_Real) n / rfactor[2]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[0]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 6:
               {
                  /* 3-d y_edge, can be X or Z_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         y_edge-> X_Face & Z_Face:
                      *  Z_Face- contract in the z direction only if the
                      *          processor interface is in the z direction
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      ******************************************************/

                     /* Z_Face */
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[2], 3,
                                         hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        hypre_AddIndexes(cstart, kshift, 3, cstart);
                     }

                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the Z plane direction */
                     loop_size[2]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * Z_Face (i,j,k-1). Two like-var coarse Edge connections.
                         * y_Edge (i,j,k-1), (i-1,j,k-1)
                         ******************************************************/
                        hypre_SubtractIndexes(cindex, kshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, ishift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Z_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[0] * rfactor[1]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[1] * (1.0 - (HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* X_Face */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);

                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        hypre_AddIndexes(cstart, ishift, 3, cstart);
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), kshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[0]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);
                        /******************************************************
                         * X_Face. Two coarse Edge connections.
                         * y_Edge (i-1,j,k), (i-1,j,k-1)
                         ******************************************************/
                        hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the X_Face */
                        for (p = 0; p < rfactor[1]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[1] += p;
                           for (n = 1; n < rfactor[2]; n++)
                           {
                              var_index[2]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[1] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[1] * (1.0 - (HYPRE_Real) n / rfactor[2]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[2]; n++) */
                        }     /* for (p= 0; p< rfactor[1]; p++) */

                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

               case 7:
               {
                  /* 3-d z_edge, can be X or Y_Face */
                  hypre_ForBoxI(i, fboxes)
                  {
                     cellbox = hypre_BoxArrayBox(fboxes, i);

                     /* vboxes inside the i'th cellbox */
                     num_vboxes = n_CtoVbox[part][i];
                     vboxnums  = CtoVboxnums[part][i];

                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     /* adjust the project cellbox to the variable box */
                     hypre_CopyBox(cellbox, &copy_box);

                     /******************************************************
                      * Check the location of the shifted lower box index:
                      *         z_edge-> X_Face & Y_Face:
                      *  X_Face- contract in the x direction if the processor
                      *          interface is in the x direction.
                      *  Y_Face- contract in the y direction if the processor
                      *          interface is in the y direction.
                      ******************************************************/
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[0], 3,
                                         hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        hypre_AddIndexes(cstart, ishift, 3, cstart);
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     /* increase the loop_size by one in the X plane direction */
                     loop_size[0]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);

                        /******************************************************
                         * ranks for coarse edges. Fine edges of agglomerate
                         * connect to these coarse edges.
                         * X_Face. Two coarse Edge connections.
                         * z_Edge (i-1,j,k), (i-1,j-1,k)
                         ******************************************************/
                        hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of z_edges making up the X_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[1]; n++)
                           {
                              var_index[1]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[2] * (1.0 - (HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[1]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */
                     }
                     hypre_SerialBoxLoop1End(m);

                     /* Y plane */
                     hypre_CopyBox(cellbox, &copy_box);
                     hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), jshift, 3,
                                           findex);
                     /* loop over all the vboxes to see if findex is inside */
                     row_in = falseV;
                     for (p = 0; p < num_vboxes[t]; p++)
                     {
                        vbox = hypre_BoxArrayBox(box_array, vboxnums[t][p]);
                        if (hypre_IndexInBox(findex, vbox))
                        {
                           hypre_CopyIndex(findex, hypre_BoxIMin(&copy_box));
                           row_in = trueV;
                           break;
                        }
                     }
                     /* not in any vbox */
                     if (!row_in)
                     {
                        hypre_AddIndexes(hypre_BoxIMin(&copy_box), boxoffset[1], 3,
                                         hypre_BoxIMin(&copy_box));
                        /* modify cstart */
                        hypre_AddIndexes(cstart, jshift, 3, cstart);
                     }
                     hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), ishift, 3,
                                           hypre_BoxIMin(&copy_box));

                     hypre_BoxGetSize(&copy_box, loop_size);
                     hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                                 loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                     loop_size[1]++;

                     hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                               &copy_box, start, rfactor, m);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                        /* because of rfactor striding, cindex= findex. But adjust
                           by cstart to get actually coarse edge. */
                        hypre_CopyIndex(findex, cindex);
                        hypre_AddIndexes(cindex, cstart, 3, cindex);

                        /* Will need the actual fine indices. */
                        for (l = 0; l < ndim; l++)
                        {
                           findex[l] *= rfactor[l];
                        }
                        hypre_AddIndexes(findex, start, 3, findex);
                        /**********************************************************
                         * Y_Face (i,j-1,k). Two like-var coarse Edge connections.
                         * z_Edge (i,j-1,k), (i-1,j-1,k)
                         **********************************************************/
                        hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank2,
                                                              matrix_type);

                        hypre_SubtractIndexes(var_index, ishift, 3, var_index);
                        hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                         t, &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                              matrix_type);

                        /* loop over the strips of y_edges making up the Y_Face */
                        for (p = 0; p < rfactor[2]; p++)
                        {
                           hypre_CopyIndex(findex, var_index);
                           var_index[2] += p;
                           for (n = 1; n < rfactor[0]; n++)
                           {
                              var_index[0]++;
                              hypre_SStructGridFindBoxManEntry(fgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &l,
                                                                    matrix_type);
                              if ((l <= upper_ranks[part][t]) &&
                                  (l >= lower_ranks[part][t]))
                              {
                                 jedge_Edge[k] = rank;
                                 vals_edgeEdge[k] = (HYPRE_Real) n / (rfactor[0] * rfactor[2]);
                                 k++;

                                 jedge_Edge[k] = rank2;
                                 vals_edgeEdge[k] = 1.0 / rfactor[2] * (1.0 - (HYPRE_Real) n / rfactor[0]);
                                 k++;
                              }
                           }  /* for (n= 1; n< rfactor[0]; n++) */
                        }     /* for (p= 0; p< rfactor[2]; p++) */

                     }
                     hypre_SerialBoxLoop1End(m);
                  }  /* hypre_ForBoxI(i, fboxes) */
                  break;
               }

            }  /* switch */
         }     /* for (t= 0; t< Edge_nvars; t++) */
         hypre_TFree(boxoffset, HYPRE_MEMORY_HOST);
      }  /* for (part= 0; part< nparts; part++) */
   }     /* if (ndim == 3) */

   /* generate the interior interpolation weights/info */
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part); /* edge grid */
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);
      Edge_vartypes = hypre_SStructPGridVarTypes(p_fgrid);
      p_cgrid = hypre_SStructGridPGrid(cgrid_edge, part); /* Edge grid */

      /* note that fboxes are the contracted CELL boxes. Will get the correct
         variable grid extents. */
      fboxes = contract_fedgeBoxes[part];

      for (t = 0; t < Edge_nvars; t++)
      {
         var      = Edge_vartypes[t];
         var_fgrid =  hypre_SStructPGridVTSGrid(p_fgrid, var);
         box_array = hypre_StructGridBoxes(var_fgrid);

         switch (var)
         {
            case 2:
            {
               /* 2-d x_face = x_edge, can be interior or on X_Edge */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[0]; p++)
                     {
                        for (n = 0; n < rfactor[1]; n++)
                        {
                           hypre_CopyIndex(findex, cindex);
                           hypre_AddIndexes(cindex, cstart, 3, cindex);

                           /*interior of Face. Extract the two coarse Edge
                             (x_Edge ijk & (i-1,j,k)*/
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (HYPRE_Real) p / (rfactor[0] * rfactor[1]);
                           k++;

                           hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (HYPRE_Real) (rfactor[0] - p) / (rfactor[0] * rfactor[1]);
                           k++;
                        }  /* for (n= 0; n< rfactor[1]; n++) */
                     }     /* for (p= 1; p< rfactor[0]; p++) */

                  }
                  hypre_SerialBoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 3:
            {
               /* 2-d y_face = y_edge, can be interior or on Y_Edge */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /* hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        for (n = 0; n < rfactor[0]; n++)
                        {
                           hypre_CopyIndex(findex, cindex);
                           hypre_AddIndexes(cindex, cstart, 3, cindex);

                           /*lies interior of Face. Extract the two coarse Edge
                             (y_Edge ijk & (i,j-1,k). */
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (HYPRE_Real) p / (rfactor[0] * rfactor[1]);
                           k++;

                           hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                           hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                            t, &entry);
                           hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                 matrix_type);
                           jedge_Edge[k] = rank;
                           vals_edgeEdge[k] = (HYPRE_Real) (rfactor[1] - p) / (rfactor[0] * rfactor[1]);
                           k++;
                        }  /* for (n= 0; n< rfactor[0]; n++) */
                     }     /* for (p= 1; p< rfactor[1]; p++) */

                  }
                  hypre_SerialBoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 5:
            {
               /* 3-d x_edge, must be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /*hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        for (n = 1; n < rfactor[1]; n++)
                        {
                           for (m = 0; m < rfactor[0]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /***********************************************
                               * Interior.
                               * x_Edge ijk, (i,j-1,k), (i,j-1,k-1), (i,j,k-1)
                               ***********************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) p * n /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);

                              k++;

                              hypre_SubtractIndexes(cindex, jshift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) p * (rfactor[1] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) (rfactor[1] - n) * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_AddIndexes(var_index, jshift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) n * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[0]; m++) */
                        }     /* for (n= 1; n< rfactor[1]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */
                  }
                  hypre_SerialBoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 6:
            {
               /* 3-d y_edge, must be interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /*hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[2]; p++)
                     {
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           for (m = 0; m < rfactor[1]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /***********************************************
                               * Interior.
                               * y_Edge ijk, (i-1,j,k), (i-1,j,k-1), (i,j,k-1)
                               ***********************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) p * n /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) p * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_SubtractIndexes(var_index, kshift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) (rfactor[0] - n) * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_AddIndexes(var_index, ishift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) n * (rfactor[2] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[1]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[2]; p++) */

                  }
                  hypre_SerialBoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }

            case 7:
            {
               /* 3-d z_edge, only the interior */
               hypre_ForBoxI(i, fboxes)
               {
                  cellbox = hypre_BoxArrayBox(fboxes, i);
                  vbox   = hypre_BoxArrayBox(box_array, i);
                  hypre_CopyIndex(Edge_cstarts[part][i], cstart);

                  /* adjust the project cellbox to the variable box */
                  hypre_CopyBox(cellbox, &copy_box);
                  hypre_SubtractIndexes(hypre_BoxIMin(&copy_box), varoffsets[var], 3,
                                        hypre_BoxIMin(&copy_box));
                  /*hypre_IntersectBoxes(&copy_box, vbox, &copy_box);*/

                  hypre_BoxGetSize(&copy_box, loop_size);
                  hypre_StructMapFineToCoarse(loop_size, zero_index, rfactor,
                                              loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&copy_box), start);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            &copy_box, start, rfactor, r);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(findex, lindex[0], lindex[1], lindex[2]);

                     /*****************************************************
                      * Where the fine edge lies wrt the coarse edge:
                      * Since we stride by rfactor, lindex is
                      * the coarse index. No coarsening needed, i.e.,
                      * cindex= findex.
                      *
                      * Loop over the interior fine edges in an agglomerate.
                      *****************************************************/
                     for (p = 1; p < rfactor[1]; p++)
                     {
                        for (n = 1; n < rfactor[0]; n++)
                        {
                           for (m = 0; m < rfactor[2]; m++)
                           {
                              hypre_CopyIndex(findex, cindex);
                              hypre_AddIndexes(cindex, cstart, 3, cindex);

                              /*************************************************
                               * Interior.
                               * z_Edge ijk, (i-1,j,k), (i-1,j-1,k), (i,j-1,k)
                               *************************************************/
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, cindex,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, cindex, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) n * p /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_SubtractIndexes(cindex, ishift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) p * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_SubtractIndexes(var_index, jshift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) (rfactor[1] - p) * (rfactor[0] - n) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;

                              hypre_AddIndexes(var_index, ishift, 3, var_index);
                              hypre_SStructGridFindBoxManEntry(cgrid_edge, part, var_index,
                                                               t, &entry);
                              hypre_SStructBoxManEntryGetGlobalRank(entry, var_index, &rank,
                                                                    matrix_type);
                              jedge_Edge[k] = rank;
                              vals_edgeEdge[k] = (HYPRE_Real) n * (rfactor[1] - p) /
                                                 (rfactor[0] * rfactor[1] * rfactor[2]);
                              k++;
                           }  /* for (m= 0; m< rfactor[2]; m++) */
                        }     /* for (n= 1; n< rfactor[0]; n++) */
                     }        /* for (p= 1; p< rfactor[1]; p++) */
                  }
                  hypre_SerialBoxLoop1End(r);
               }  /* hypre_ForBoxI(i, fboxes) */
               break;
            }
         }  /* switch */
      }     /* for (t= 0; t< Edge_nvars; t++) */
   }        /* for (part= 0; part< nparts; part++) */

   HYPRE_IJMatrixSetValues(edge_Edge, size, ncols_edgeEdge,
                           (const HYPRE_BigInt*) iedgeEdge, (const HYPRE_BigInt*) jedge_Edge,
                           (const HYPRE_Real*) vals_edgeEdge);
   HYPRE_IJMatrixAssemble((HYPRE_IJMatrix) edge_Edge);

   hypre_TFree(ncols_edgeEdge, HYPRE_MEMORY_HOST);
   hypre_TFree(iedgeEdge, HYPRE_MEMORY_HOST);
   hypre_TFree(jedge_Edge, HYPRE_MEMORY_HOST);
   hypre_TFree(vals_edgeEdge, HYPRE_MEMORY_HOST);

   hypre_TFree(varoffsets, HYPRE_MEMORY_HOST);
   hypre_TFree(vartype_map, HYPRE_MEMORY_HOST);

   /* n_CtoVbox[part][cellboxi][var]  & CtoVboxnums[part][cellboxi][var][nvboxes] */
   for (part = 0; part < nparts; part++)
   {
      p_fgrid = hypre_SStructGridPGrid(fgrid_edge, part);
      Edge_nvars = hypre_SStructPGridNVars(p_fgrid);

      var_fgrid = hypre_SStructPGridCellSGrid(p_fgrid);
      fboxes   = hypre_StructGridBoxes(var_fgrid);
      hypre_ForBoxI(j, fboxes)
      {
         for (t = 0; t < Edge_nvars; t++)
         {
            hypre_TFree(CtoVboxnums[part][j][t], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(n_CtoVbox[part][j], HYPRE_MEMORY_HOST);
         hypre_TFree(CtoVboxnums[part][j], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(n_CtoVbox[part], HYPRE_MEMORY_HOST);
      hypre_TFree(CtoVboxnums[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(n_CtoVbox, HYPRE_MEMORY_HOST);
   hypre_TFree(CtoVboxnums, HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      hypre_BoxArrayDestroy(contract_fedgeBoxes[part]);
      hypre_TFree(Edge_cstarts[part], HYPRE_MEMORY_HOST);
      hypre_TFree(upper_shifts[part], HYPRE_MEMORY_HOST);
      hypre_TFree(lower_shifts[part], HYPRE_MEMORY_HOST);
      hypre_TFree(cfbox_mapping[part], HYPRE_MEMORY_HOST);
      hypre_TFree(fcbox_mapping[part], HYPRE_MEMORY_HOST);
      hypre_TFree(upper_ranks[part], HYPRE_MEMORY_HOST);
      hypre_TFree(lower_ranks[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(contract_fedgeBoxes, HYPRE_MEMORY_HOST);
   hypre_TFree(Edge_cstarts, HYPRE_MEMORY_HOST);
   hypre_TFree(upper_shifts, HYPRE_MEMORY_HOST);
   hypre_TFree(lower_shifts, HYPRE_MEMORY_HOST);
   hypre_TFree(cfbox_mapping, HYPRE_MEMORY_HOST);
   hypre_TFree(fcbox_mapping, HYPRE_MEMORY_HOST);
   hypre_TFree(upper_ranks, HYPRE_MEMORY_HOST);
   hypre_TFree(lower_ranks, HYPRE_MEMORY_HOST);

   return (hypre_IJMatrix *) edge_Edge;
}
