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
 *   i, nrows (only where they are listed at the end of SMP_PRIVATE)
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_Maxwell_Grad.c
 *   Forms a node-to-edge gradient operator. Looping over the
 *   edge grid so that each processor fills up only its own rows. Each
 *   processor will have its processor interface nodal ranks.
 *   Loops over two types of boxes, interior of grid boxes and boundary
 *   of boxes. Algo:
 *       find all nodal and edge physical boundary points and set
 *       the appropriate flag to be 0 at a boundary dof.
 *       set -1's in value array
 *       for each edge box,
 *       for interior
 *       {
 *          connect edge ijk (row) to nodes (col) connected to this edge
 *          and change -1 to 1 if needed;
 *       }
 *       for boundary layers
 *       {
 *          if edge not on the physical boundary connect only the nodes
 *          that are not on the physical boundary
 *       }
 *       set parcsr matrix with values;
 *
 * Note that the nodes that are on the processor interface can be
 * on the physical boundary. But the off-proc edges connected to this
 * type of node will be a physical boundary edge.
 *
 *--------------------------------------------------------------------------*/
hypre_ParCSRMatrix *
hypre_Maxwell_Grad(hypre_SStructGrid *grid)
{
   MPI_Comm               comm = (grid ->  comm);

   HYPRE_IJMatrix         T_grad;
   hypre_ParCSRMatrix    *parcsr_grad;
   HYPRE_Int              matrix_type = HYPRE_PARCSR;

   hypre_SStructGrid     *node_grid, *edge_grid;

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *var_grid;
   hypre_BoxArray        *boxes, *tmp_box_array1, *tmp_box_array2;
   hypre_BoxArray        *edge_boxes, *cell_boxes;
   hypre_Box             *box, *cell_box;
   hypre_Box              layer, interior_box;
   hypre_Box             *box_piece;

   hypre_BoxManager      *boxman;
   hypre_BoxManEntry     *entry;

   HYPRE_BigInt          *inode, *jedge;
   HYPRE_Int              nrows, nnodes, *nflag, *eflag, *ncols;
   HYPRE_Real            *vals;

   hypre_Index            index;
   hypre_Index            loop_size, start, lindex;
   hypre_Index            shift, shift2;
   hypre_Index           *offsets, *varoffsets;

   HYPRE_Int              nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int              ndim  = hypre_SStructGridNDim(grid);

   HYPRE_SStructVariable  vartype_node, *vartype_edges;
   HYPRE_SStructVariable *vartypes;

   HYPRE_Int              nvars, part;

   HYPRE_BigInt           m;
   HYPRE_Int              i, j, k, n, d;
   HYPRE_Int             *direction, ndirection;

   HYPRE_BigInt           ilower, iupper;
   HYPRE_BigInt           jlower, jupper;

   HYPRE_BigInt           start_rank1, start_rank2, rank;
   HYPRE_Int              myproc;

   HYPRE_MemoryLocation   memory_location;

   hypre_BoxInit(&layer, ndim);
   hypre_BoxInit(&interior_box, ndim);

   hypre_MPI_Comm_rank(comm, &myproc);

   hypre_ClearIndex(shift);
   hypre_SetIndex(shift, -1);
   hypre_SetIndex(lindex, 0);

   /* To get the correct ranks, separate node & edge grids must be formed.
      Note that the edge vars must be ordered the same way as is in grid.*/
   HYPRE_SStructGridCreate(comm, ndim, nparts, &node_grid);
   HYPRE_SStructGridCreate(comm, ndim, nparts, &edge_grid);

   vartype_node = HYPRE_SSTRUCT_VARIABLE_NODE;
   vartype_edges = hypre_TAlloc(HYPRE_SStructVariable,  ndim, HYPRE_MEMORY_HOST);

   /* Assuming the same edge variable types on all parts */
   pgrid   = hypre_SStructGridPGrid(grid, 0);
   vartypes = hypre_SStructPGridVarTypes(pgrid);
   nvars   = hypre_SStructPGridNVars(pgrid);

   k = 0;
   for (i = 0; i < nvars; i++)
   {
      j = vartypes[i];
      switch (j)
      {
         case 2:
         {
            vartype_edges[k] = HYPRE_SSTRUCT_VARIABLE_XFACE;
            k++;
            break;
         }

         case 3:
         {
            vartype_edges[k] = HYPRE_SSTRUCT_VARIABLE_YFACE;
            k++;
            break;
         }

         case 5:
         {
            vartype_edges[k] = HYPRE_SSTRUCT_VARIABLE_XEDGE;
            k++;
            break;
         }

         case 6:
         {
            vartype_edges[k] = HYPRE_SSTRUCT_VARIABLE_YEDGE;
            k++;
            break;
         }

         case 7:
         {
            vartype_edges[k] = HYPRE_SSTRUCT_VARIABLE_ZEDGE;
            k++;
            break;
         }

      }  /* switch(j) */
   }     /* for (i= 0; i< nvars; i++) */

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      var_grid = hypre_SStructPGridCellSGrid(pgrid) ;

      boxes = hypre_StructGridBoxes(var_grid);
      hypre_ForBoxI(j, boxes)
      {
         box = hypre_BoxArrayBox(boxes, j);
         HYPRE_SStructGridSetExtents(node_grid, part,
                                     hypre_BoxIMin(box), hypre_BoxIMax(box));
         HYPRE_SStructGridSetExtents(edge_grid, part,
                                     hypre_BoxIMin(box), hypre_BoxIMax(box));
      }
      HYPRE_SStructGridSetVariables(node_grid, part, 1, &vartype_node);
      HYPRE_SStructGridSetVariables(edge_grid, part, ndim, vartype_edges);
   }
   HYPRE_SStructGridAssemble(node_grid);
   HYPRE_SStructGridAssemble(edge_grid);

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information.
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */

   /* Grad: node(col) -> edge(row). Same for 2-d and 3-d */
   /* lower rank */
   part = 0;
   i   = 0;

   hypre_SStructGridBoxProcFindBoxManEntry(edge_grid, part, 0, i, myproc, &entry);
   pgrid   = hypre_SStructGridPGrid(edge_grid, part);
   var_grid = hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &ilower);

   hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0, i, myproc, &entry);
   pgrid   = hypre_SStructGridPGrid(node_grid, part);
   var_grid = hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, 0);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMin(box), &jlower);

   /* upper rank */
   part = nparts - 1;

   pgrid   = hypre_SStructGridPGrid(edge_grid, part);
   nvars   = hypre_SStructPGridNVars(pgrid);
   var_grid = hypre_SStructPGridSGrid(pgrid, nvars - 1);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes) - 1);

   hypre_SStructGridBoxProcFindBoxManEntry(edge_grid, part, nvars - 1,
                                           hypre_BoxArraySize(boxes) - 1, myproc,
                                           &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &iupper);

   pgrid   = hypre_SStructGridPGrid(node_grid, part);
   nvars   = hypre_SStructPGridNVars(pgrid);
   var_grid = hypre_SStructPGridSGrid(pgrid, nvars - 1);
   boxes   = hypre_StructGridBoxes(var_grid);
   box     = hypre_BoxArrayBox(boxes, hypre_BoxArraySize(boxes) - 1);

   hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, nvars - 1,
                                           hypre_BoxArraySize(boxes) - 1, myproc,
                                           &entry);
   hypre_SStructBoxManEntryGetGlobalCSRank(entry, hypre_BoxIMax(box), &jupper);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &T_grad);
   HYPRE_IJMatrixSetObjectType(T_grad, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(T_grad);

   memory_location = hypre_IJMatrixMemoryLocation(T_grad);

   /*------------------------------------------------------------------------------
    * fill up the parcsr matrix.
    *------------------------------------------------------------------------------*/

   /* count the no. of rows. Make sure repeated nodes along the boundaries are counted.*/
   nrows = 0;
   nnodes = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(edge_grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (m = 0; m < nvars; m++)
      {
         var_grid = hypre_SStructPGridSGrid(pgrid, m);
         boxes   = hypre_StructGridBoxes(var_grid);
         hypre_ForBoxI(j, boxes)
         {
            box = hypre_BoxArrayBox(boxes, j);
            /* make slightly bigger to handle any shared nodes */
            hypre_CopyBox(box, &layer);
            hypre_AddIndexes(hypre_BoxIMin(&layer), shift, 3, hypre_BoxIMin(&layer));
            hypre_SubtractIndexes(hypre_BoxIMax(&layer), shift, 3, hypre_BoxIMax(&layer));
            nrows += hypre_BoxVolume(&layer);
         }
      }

      pgrid = hypre_SStructGridPGrid(node_grid, part);
      var_grid = hypre_SStructPGridSGrid(pgrid, 0); /* only one variable grid */
      boxes   = hypre_StructGridBoxes(var_grid);
      hypre_ForBoxI(j, boxes)
      {
         box = hypre_BoxArrayBox(boxes, j);
         /* make slightly bigger to handle any shared nodes */
         hypre_CopyBox(box, &layer);
         hypre_AddIndexes(hypre_BoxIMin(&layer), shift, 3, hypre_BoxIMin(&layer));
         hypre_SubtractIndexes(hypre_BoxIMax(&layer), shift, 3, hypre_BoxIMax(&layer));
         nnodes += hypre_BoxVolume(&layer);
      }
   }

   eflag = hypre_CTAlloc(HYPRE_Int,  nrows, HYPRE_MEMORY_HOST);
   nflag = hypre_CTAlloc(HYPRE_Int,  nnodes, HYPRE_MEMORY_HOST);

   /* Set eflag to have the number of nodes connected to an edge (2) and
      nflag to have the number of edges connect to a node. */
   for (i = 0; i < nrows; i++)
   {
      eflag[i] = 2;
   }
   j = 2 * ndim;
   for (i = 0; i < nnodes; i++)
   {
      nflag[i] = j;
   }

   /* Determine physical boundary points. Get the rank and set flag[rank]= 0.
      This will boundary dof, i.e., flag[rank]= 0 will flag a boundary dof. */

   start_rank1 = hypre_SStructGridStartRank(node_grid);
   start_rank2 = hypre_SStructGridStartRank(edge_grid);
   for (part = 0; part < nparts; part++)
   {
      /* node flag */
      pgrid   = hypre_SStructGridPGrid(node_grid, part);
      var_grid = hypre_SStructPGridSGrid(pgrid, 0);
      boxes   = hypre_StructGridBoxes(var_grid);
      boxman     = hypre_SStructGridBoxManager(node_grid, part, 0);

      hypre_ForBoxI(j, boxes)
      {
         box = hypre_BoxArrayBox(boxes, j);
         hypre_BoxManGetEntry(boxman, myproc, j, &entry);
         i = hypre_BoxVolume(box);

         tmp_box_array1 = hypre_BoxArrayCreate(0, ndim);
         hypre_BoxBoundaryG(box, var_grid, tmp_box_array1);

         for (m = 0; m < hypre_BoxArraySize(tmp_box_array1); m++)
         {
            box_piece = hypre_BoxArrayBox(tmp_box_array1, m);
            if (hypre_BoxVolume(box_piece) < i)
            {
               hypre_BoxGetSize(box_piece, loop_size);
               hypre_CopyIndex(hypre_BoxIMin(box_piece), start);

               hypre_SerialBoxLoop0Begin(ndim, loop_size);
               {
                  zypre_BoxLoopGetIndex(lindex);
                  hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                  hypre_AddIndexes(index, start, 3, index);

                  hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                        &rank, matrix_type);
                  nflag[rank - start_rank1] = 0;
               }
               hypre_SerialBoxLoop0End();
            }  /* if (hypre_BoxVolume(box_piece) < i) */

         }  /* for (m= 0; m< hypre_BoxArraySize(tmp_box_array1); m++) */
         hypre_BoxArrayDestroy(tmp_box_array1);

      }  /* hypre_ForBoxI(j, boxes) */

      /*-----------------------------------------------------------------
       * edge flag. Since we want only the edges that completely lie
       * on a boundary, whereas the boundary extraction routines mark
       * edges that touch the boundary, we need to call the boundary
       * routines in appropriate directions:
       *    2-d horizontal edges (y faces)- search in j directions
       *    2-d vertical edges (x faces)  - search in i directions
       *    3-d x edges                   - search in j,k directions
       *    3-d y edges                   - search in i,k directions
       *    3-d z edges                   - search in i,j directions
       *-----------------------------------------------------------------*/
      pgrid    = hypre_SStructGridPGrid(edge_grid, part);
      nvars    = hypre_SStructPGridNVars(pgrid);
      direction = hypre_TAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST); /* only two directions at most */
      for (m = 0; m < nvars; m++)
      {
         var_grid = hypre_SStructPGridSGrid(pgrid, m);
         boxes   = hypre_StructGridBoxes(var_grid);
         boxman  = hypre_SStructGridBoxManager(edge_grid, part, m);

         j = vartype_edges[m];
         switch (j)
         {
            case 2: /* x faces, 2d */
            {
               ndirection  = 1;
               direction[0] = 0;
               break;
            }

            case 3: /* y faces, 2d */
            {
               ndirection  = 1;
               direction[0] = 1;
               break;
            }

            case 5: /* x edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 1;
               direction[1] = 2;
               break;
            }

            case 6: /* y edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 0;
               direction[1] = 2;
               break;
            }

            case 7: /* z edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 0;
               direction[1] = 1;
               break;
            }

            default:
            {
               ndirection = 0;
            }
         }  /* switch(j) */

         hypre_ForBoxI(j, boxes)
         {
            box = hypre_BoxArrayBox(boxes, j);
            hypre_BoxManGetEntry(boxman, myproc, j, &entry);
            i = hypre_BoxVolume(box);

            for (d = 0; d < ndirection; d++)
            {
               tmp_box_array1 = hypre_BoxArrayCreate(0, ndim);
               tmp_box_array2 = hypre_BoxArrayCreate(0, ndim);
               hypre_BoxBoundaryDG(box, var_grid, tmp_box_array1,
                                   tmp_box_array2, direction[d]);

               for (k = 0; k < hypre_BoxArraySize(tmp_box_array1); k++)
               {
                  box_piece = hypre_BoxArrayBox(tmp_box_array1, k);
                  if (hypre_BoxVolume(box_piece) < i)
                  {
                     hypre_BoxGetSize(box_piece, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(box_piece), start);

                     hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        hypre_AddIndexes(index, start, 3, index);

                        hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                              &rank, matrix_type);
                        eflag[rank - start_rank2] = 0;
                     }
                     hypre_SerialBoxLoop0End();
                  }  /* if (hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< hypre_BoxArraySize(tmp_box_array1); k++) */

               hypre_BoxArrayDestroy(tmp_box_array1);

               for (k = 0; k < hypre_BoxArraySize(tmp_box_array2); k++)
               {
                  box_piece = hypre_BoxArrayBox(tmp_box_array2, k);
                  if (hypre_BoxVolume(box_piece) < i)
                  {
                     hypre_BoxGetSize(box_piece, loop_size);
                     hypre_CopyIndex(hypre_BoxIMin(box_piece), start);

                     hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        hypre_AddIndexes(index, start, 3, index);

                        hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                              &rank, matrix_type);
                        eflag[rank - start_rank2] = 0;
                     }
                     hypre_SerialBoxLoop0End();
                  }  /* if (hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< hypre_BoxArraySize(tmp_box_array2); k++) */
               hypre_BoxArrayDestroy(tmp_box_array2);
            }  /* for (d= 0; d< ndirection; d++) */

         }  /* hypre_ForBoxI(j, boxes) */
      }     /* for (m= 0; m< nvars; m++) */

      hypre_TFree(direction, HYPRE_MEMORY_HOST);
   }  /* for (part= 0; part< nparts; part++) */

   /* set vals. Will have more memory than is needed- extra allotted
      for repeated nodes. */
   inode = hypre_CTAlloc(HYPRE_BigInt, nrows, memory_location);
   ncols = hypre_CTAlloc(HYPRE_Int, nrows, memory_location);

   /* each row can have at most two columns */
   k = 2 * nrows;
   jedge = hypre_CTAlloc(HYPRE_BigInt, k, memory_location);
   vals = hypre_TAlloc(HYPRE_Real, k, memory_location);
   for (i = 0; i < k; i++)
   {
      vals[i] = -1.0;
   }

   /* to get the correct col connection to each node, we need to offset
      index ijk. Determine these. Assuming the same var ordering for each
      part. Note that these are not the variable offsets. */
   offsets   = hypre_TAlloc(hypre_Index,  ndim, HYPRE_MEMORY_HOST);
   varoffsets = hypre_TAlloc(hypre_Index,  ndim, HYPRE_MEMORY_HOST);
   for (i = 0; i < ndim; i++)
   {
      j = vartype_edges[i];
      hypre_SStructVariableGetOffset(vartype_edges[i], ndim, varoffsets[i]);
      switch (j)
      {
         case 2:
         {
            hypre_SetIndex3(offsets[i], 0, 1, 0);
            break;
         }

         case 3:
         {
            hypre_SetIndex3(offsets[i], 1, 0, 0);
            break;
         }

         case 5:
         {
            hypre_SetIndex3(offsets[i], 1, 0, 0);
            break;
         }

         case 6:
         {
            hypre_SetIndex3(offsets[i], 0, 1, 0);
            break;
         }

         case 7:
         {
            hypre_SetIndex3(offsets[i], 0, 0, 1);
            break;
         }
      }   /*  switch(j) */
   }     /* for (i= 0; i< ndim; i++) */

   nrows = 0; i = 0;
   for (part = 0; part < nparts; part++)
   {
      /* grab boxarray for node rank extracting later */
      pgrid       = hypre_SStructGridPGrid(node_grid, part);
      var_grid    = hypre_SStructPGridSGrid(pgrid, 0);

      /* grab edge structures */
      pgrid     = hypre_SStructGridPGrid(edge_grid, part);

      /* the cell-centred reference box is used to get the correct
         interior edge box. For parallel distribution of the edge
         grid, simple contraction of the edge box does not get the
         correct interior edge box. Need to contract the cell box. */
      var_grid = hypre_SStructPGridCellSGrid(pgrid);
      cell_boxes = hypre_StructGridBoxes(var_grid);

      nvars     = hypre_SStructPGridNVars(pgrid);
      for (n = 0; n < nvars; n++)
      {
         var_grid  = hypre_SStructPGridSGrid(pgrid, n);
         edge_boxes = hypre_StructGridBoxes(var_grid);

         hypre_ForBoxI(j, edge_boxes)
         {
            box = hypre_BoxArrayBox(edge_boxes, j);
            cell_box = hypre_BoxArrayBox(cell_boxes, j);

            hypre_CopyBox(cell_box, &interior_box);

            /* shrink the cell_box to get the interior cell_box. All
               edges in the interior box should be on this proc. */
            hypre_SubtractIndexes(hypre_BoxIMin(&interior_box), shift, 3,
                                  hypre_BoxIMin(&interior_box));

            hypre_AddIndexes(hypre_BoxIMax(&interior_box), shift, 3,
                             hypre_BoxIMax(&interior_box));

            /* offset this to the variable interior box */
            hypre_CopyBox(&interior_box, &layer);
            hypre_SubtractIndexes(hypre_BoxIMin(&layer), varoffsets[n], 3,
                                  hypre_BoxIMin(&layer));

            hypre_BoxGetSize(&layer, loop_size);
            hypre_CopyIndex(hypre_BoxIMin(&layer), start);

            /* Interior box- loop over each edge and find the row rank and
               then the column ranks for the connected nodes. Change the
               appropriate values to 1. */
            hypre_SerialBoxLoop0Begin(ndim, loop_size);
            {
               zypre_BoxLoopGetIndex(lindex);
               hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
               hypre_AddIndexes(index, start, 3, index);

               /* edge ijk connected to nodes ijk & ijk-offsets. Interior edges
                  and so no boundary edges to consider. */
               hypre_SStructGridFindBoxManEntry(edge_grid, part, index, n,
                                                &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               inode[nrows] = m;

               hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i] = m;
               vals[i] = 1.0; /* change only this connection */
               i++;

               hypre_SubtractIndexes(index, offsets[n], 3, index);
               hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                &entry);
               hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i] = m;
               i++;

               ncols[nrows] = 2;
               nrows++;
            }
            hypre_SerialBoxLoop0End();

            /* now the boundary layers. To cases to consider: is the
               edge totally on the boundary or is the edge connected
               to the boundary. Need to check eflag & nflag. */
            for (d = 0; d < ndim; d++)
            {
               /*shift the layer box in the correct direction and distance.
                 distance= hypre_BoxIMax(box)[d]-hypre_BoxIMin(box)[d]+1-1
                 = hypre_BoxIMax(box)[d]-hypre_BoxIMin(box)[d] */
               hypre_ClearIndex(shift2);
               shift2[d] = hypre_BoxIMax(box)[d] - hypre_BoxIMin(box)[d];

               /* ndirection= 0 negative; ndirection= 1 positive */
               for (ndirection = 0; ndirection < 2; ndirection++)
               {
                  hypre_CopyBox(box, &layer);

                  if (ndirection)
                  {
                     hypre_BoxShiftPos(&layer, shift2);
                  }
                  else
                  {
                     hypre_BoxShiftNeg(&layer, shift2);
                  }

                  hypre_IntersectBoxes(box, &layer, &layer);
                  hypre_BoxGetSize(&layer, loop_size);
                  hypre_CopyIndex(hypre_BoxIMin(&layer), start);

                  hypre_SerialBoxLoop0Begin(ndim, loop_size);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                     hypre_AddIndexes(index, start, 3, index);

                     /* edge ijk connects to nodes ijk & ijk+offsets. */
                     hypre_SStructGridFindBoxManEntry(edge_grid, part, index, n,
                                                      &entry);
                     hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                           matrix_type);

                     /* check if the edge lies on the boundary & if not
                        check if the connecting node is on the boundary. */
                     if (eflag[m - start_rank2])
                     {
                        inode[nrows] = m;
                        /* edge not completely on the boundary. One connecting
                           node must be in the interior. */
                        hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                         &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                              matrix_type);

                        /* check if node on my processor. If not, the node must
                           be in the interior (draw a diagram to see this). */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m - start_rank1]) /* interior node */
                           {
                              jedge[i] = m;
                              vals[i] = 1.0;
                              i++;

                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i] = m;
                           vals[i] = 1.0;
                           i++;

                           ncols[nrows]++;
                        }

                        /* ijk+offsets */
                        hypre_SubtractIndexes(index, offsets[n], 3, index);
                        hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                         &entry);
                        hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                              matrix_type);
                        /* boundary checks again */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m - start_rank1]) /* interior node */
                           {
                              jedge[i] = m;
                              i++;
                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i] = m;
                           i++;
                           ncols[nrows]++;
                        }

                        nrows++; /* must have at least one node connection */
                     }  /* if (eflag[m-start_rank2]) */

                  }
                  hypre_SerialBoxLoop0End();
               }  /* for (ndirection= 0; ndirection< 2; ndirection++) */
            }     /* for (d= 0; d< ndim; d++) */

         }  /* hypre_ForBoxI(j, boxes) */
      }     /* for (n= 0; n< nvars; n++) */
   }        /* for (part= 0; part< nparts; part++) */

   hypre_TFree(offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(varoffsets, HYPRE_MEMORY_HOST);
   hypre_TFree(vartype_edges, HYPRE_MEMORY_HOST);
   HYPRE_SStructGridDestroy(node_grid);
   HYPRE_SStructGridDestroy(edge_grid);

   HYPRE_IJMatrixSetValues(T_grad, nrows, ncols,
                           (const HYPRE_BigInt*) inode, (const HYPRE_BigInt*) jedge,
                           (const HYPRE_Real*) vals);
   HYPRE_IJMatrixAssemble(T_grad);

   hypre_TFree(eflag, HYPRE_MEMORY_HOST);
   hypre_TFree(nflag, HYPRE_MEMORY_HOST);
   hypre_TFree(ncols, memory_location);
   hypre_TFree(inode, memory_location);
   hypre_TFree(jedge, memory_location);
   hypre_TFree(vals, memory_location);

   parcsr_grad = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(T_grad);
   HYPRE_IJMatrixSetObjectType(T_grad, -1);
   HYPRE_IJMatrixDestroy(T_grad);

   return  parcsr_grad;
}
