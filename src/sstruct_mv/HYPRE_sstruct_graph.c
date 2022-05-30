/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphCreate( MPI_Comm             comm,
                          HYPRE_SStructGrid    grid,
                          HYPRE_SStructGraph  *graph_ptr )
{
   hypre_SStructGraph     *graph;
   HYPRE_Int               nparts;
   hypre_SStructStencil ***stencils;
   hypre_SStructPGrid    **pgrids;
   HYPRE_Int              *fem_nsparse;
   HYPRE_Int             **fem_sparse_i;
   HYPRE_Int             **fem_sparse_j;
   HYPRE_Int             **fem_entries;
   HYPRE_Int               nvars;
   HYPRE_Int               part, var;

   graph = hypre_TAlloc(hypre_SStructGraph,  1, HYPRE_MEMORY_HOST);

   hypre_SStructGraphComm(graph) = comm;
   hypre_SStructGraphNDim(graph) = hypre_SStructGridNDim(grid);
   hypre_SStructGridRef(grid, &hypre_SStructGraphGrid(graph));
   hypre_SStructGridRef(grid, &hypre_SStructGraphDomainGrid(graph));
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructGraphNParts(graph) = nparts;
   pgrids = hypre_SStructGridPGrids(grid);
   stencils = hypre_TAlloc(hypre_SStructStencil **,  nparts, HYPRE_MEMORY_HOST);
   fem_nsparse  = hypre_TAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);
   fem_sparse_i = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   fem_sparse_j = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   fem_entries  = hypre_TAlloc(HYPRE_Int *,  nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      stencils[part]  = hypre_TAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);
      fem_nsparse[part]  = 0;
      fem_sparse_i[part] = NULL;
      fem_sparse_j[part] = NULL;
      fem_entries[part]  = NULL;
      for (var = 0; var < nvars; var++)
      {
         stencils[part][var] = NULL;
      }
   }
   hypre_SStructGraphStencils(graph)   = stencils;
   hypre_SStructGraphFEMNSparse(graph) = fem_nsparse;
   hypre_SStructGraphFEMSparseJ(graph) = fem_sparse_i;
   hypre_SStructGraphFEMSparseI(graph) = fem_sparse_j;
   hypre_SStructGraphFEMEntries(graph) = fem_entries;

   hypre_SStructGraphNUVEntries(graph) = 0;
   hypre_SStructGraphIUVEntries(graph) = NULL;
   hypre_SStructGraphUVEntries(graph)  = NULL;
   hypre_SStructGraphUVESize(graph)    = 0;
   hypre_SStructGraphUEMaxSize(graph)  = 0;
   hypre_SStructGraphUVEOffsets(graph) = NULL;

   hypre_SStructGraphRefCount(graph)   = 1;
   hypre_SStructGraphObjectType(graph) = HYPRE_SSTRUCT;

   hypre_SStructGraphEntries(graph)    = NULL;
   hypre_SStructNGraphEntries(graph)   = 0;
   hypre_SStructAGraphEntries(graph)   = 0;

   *graph_ptr = graph;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphDestroy( HYPRE_SStructGraph graph )
{
   HYPRE_Int                 nparts;
   hypre_SStructPGrid      **pgrids;
   hypre_SStructStencil   ***stencils;
   HYPRE_Int                *fem_nsparse;
   HYPRE_Int               **fem_sparse_i;
   HYPRE_Int               **fem_sparse_j;
   HYPRE_Int               **fem_entries;
   HYPRE_Int                 nUventries;
   HYPRE_Int                *iUventries;
   hypre_SStructUVEntry    **Uventries;
   hypre_SStructUVEntry     *Uventry;
   HYPRE_BigInt            **Uveoffsets;
   hypre_SStructGraphEntry **graph_entries;
   HYPRE_Int                 nvars;
   HYPRE_Int                 part, var, i;

   if (graph)
   {
      hypre_SStructGraphRefCount(graph) --;
      if (hypre_SStructGraphRefCount(graph) == 0)
      {
         nparts   = hypre_SStructGraphNParts(graph);
         pgrids   = hypre_SStructGraphPGrids(graph);
         stencils = hypre_SStructGraphStencils(graph);
         fem_nsparse  = hypre_SStructGraphFEMNSparse(graph);
         fem_sparse_i = hypre_SStructGraphFEMSparseJ(graph);
         fem_sparse_j = hypre_SStructGraphFEMSparseI(graph);
         fem_entries  = hypre_SStructGraphFEMEntries(graph);
         nUventries = hypre_SStructGraphNUVEntries(graph);
         iUventries = hypre_SStructGraphIUVEntries(graph);
         Uventries  = hypre_SStructGraphUVEntries(graph);
         Uveoffsets = hypre_SStructGraphUVEOffsets(graph);
         for (part = 0; part < nparts; part++)
         {
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               HYPRE_SStructStencilDestroy(stencils[part][var]);
            }
            hypre_TFree(stencils[part], HYPRE_MEMORY_HOST);
            hypre_TFree(fem_sparse_i[part], HYPRE_MEMORY_HOST);
            hypre_TFree(fem_sparse_j[part], HYPRE_MEMORY_HOST);
            hypre_TFree(fem_entries[part], HYPRE_MEMORY_HOST);
            hypre_TFree(Uveoffsets[part], HYPRE_MEMORY_HOST);
         }
         HYPRE_SStructGridDestroy(hypre_SStructGraphGrid(graph));
         HYPRE_SStructGridDestroy(hypre_SStructGraphDomainGrid(graph));
         hypre_TFree(stencils, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_nsparse, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_sparse_i, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_sparse_j, HYPRE_MEMORY_HOST);
         hypre_TFree(fem_entries, HYPRE_MEMORY_HOST);
         /* RDF: THREAD? */
         for (i = 0; i < nUventries; i++)
         {
            Uventry = Uventries[iUventries[i]];
            if (Uventry)
            {
               hypre_TFree(hypre_SStructUVEntryUEntries(Uventry), HYPRE_MEMORY_HOST);
               hypre_TFree(Uventry, HYPRE_MEMORY_HOST);
            }
            Uventries[iUventries[i]] = NULL;
         }
         hypre_TFree(iUventries, HYPRE_MEMORY_HOST);
         hypre_TFree(Uventries, HYPRE_MEMORY_HOST);
         hypre_TFree(Uveoffsets, HYPRE_MEMORY_HOST);
         graph_entries = hypre_SStructGraphEntries(graph);
         for (i = 0; i < hypre_SStructNGraphEntries(graph); i++)
         {
            hypre_TFree(graph_entries[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(graph_entries, HYPRE_MEMORY_HOST);
         hypre_TFree(graph, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphSetDomainGrid( HYPRE_SStructGraph graph,
                                 HYPRE_SStructGrid  domain_grid)
{
   /* This should only decrement a reference counter */
   HYPRE_SStructGridDestroy(hypre_SStructGraphDomainGrid(graph));
   hypre_SStructGridRef(domain_grid, &hypre_SStructGraphDomainGrid(graph));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphSetStencil( HYPRE_SStructGraph   graph,
                              HYPRE_Int            part,
                              HYPRE_Int            var,
                              HYPRE_SStructStencil stencil )
{
   hypre_SStructStencilRef(stencil, &hypre_SStructGraphStencil(graph, part, var));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphSetFEM( HYPRE_SStructGraph graph,
                          HYPRE_Int          part )
{
   if (!hypre_SStructGraphFEMPNSparse(graph, part))
   {
      hypre_SStructGraphFEMPNSparse(graph, part) = -1;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphSetFEMSparsity( HYPRE_SStructGraph  graph,
                                  HYPRE_Int           part,
                                  HYPRE_Int           nsparse,
                                  HYPRE_Int          *sparsity )
{
   HYPRE_Int          *fem_sparse_i;
   HYPRE_Int          *fem_sparse_j;
   HYPRE_Int           s;

   hypre_SStructGraphFEMPNSparse(graph, part) = nsparse;
   fem_sparse_i = hypre_TAlloc(HYPRE_Int,  nsparse, HYPRE_MEMORY_HOST);
   fem_sparse_j = hypre_TAlloc(HYPRE_Int,  nsparse, HYPRE_MEMORY_HOST);
   for (s = 0; s < nsparse; s++)
   {
      fem_sparse_i[s] = sparsity[2 * s];
      fem_sparse_j[s] = sparsity[2 * s + 1];
   }
   hypre_SStructGraphFEMPSparseI(graph, part) = fem_sparse_i;
   hypre_SStructGraphFEMPSparseJ(graph, part) = fem_sparse_j;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *
 *   Now we just keep track of calls to this function and do all the "work"
 *   in the assemble.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphAddEntries( HYPRE_SStructGraph   graph,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Int            to_part,
                              HYPRE_Int           *to_index,
                              HYPRE_Int            to_var )
{
   hypre_SStructGrid        *grid      = hypre_SStructGraphGrid(graph);
   HYPRE_Int                 ndim      = hypre_SStructGridNDim(grid);

   hypre_SStructGraphEntry **entries   = hypre_SStructGraphEntries(graph);
   hypre_SStructGraphEntry  *new_entry;

   HYPRE_Int                 n_entries = hypre_SStructNGraphEntries(graph);
   HYPRE_Int                 a_entries = hypre_SStructAGraphEntries(graph);

   /* check storage */
   if (!a_entries)
   {
      a_entries = 1000;
      entries = hypre_TAlloc(hypre_SStructGraphEntry *,  a_entries, HYPRE_MEMORY_HOST);

      hypre_SStructAGraphEntries(graph) = a_entries;
      hypre_SStructGraphEntries(graph) = entries;
   }
   else if (n_entries >= a_entries)
   {
      a_entries += 1000;
      entries = hypre_TReAlloc(entries,  hypre_SStructGraphEntry *,  a_entries, HYPRE_MEMORY_HOST);

      hypre_SStructAGraphEntries(graph) = a_entries;
      hypre_SStructGraphEntries(graph) = entries;
   }

   /*save parameters to a new entry */

   new_entry = hypre_TAlloc(hypre_SStructGraphEntry,  1, HYPRE_MEMORY_HOST);

   hypre_SStructGraphEntryPart(new_entry) = part;
   hypre_SStructGraphEntryToPart(new_entry) = to_part;

   hypre_SStructGraphEntryVar(new_entry) = var;
   hypre_SStructGraphEntryToVar(new_entry) = to_var;

   hypre_CopyToCleanIndex(index, ndim, hypre_SStructGraphEntryIndex(new_entry));
   hypre_CopyToCleanIndex(
      to_index, ndim, hypre_SStructGraphEntryToIndex(new_entry));

   entries[n_entries] = new_entry;

   /* update count */
   n_entries++;
   hypre_SStructNGraphEntries(graph) = n_entries;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine mainly computes the column numbers for the non-stencil
 * graph entries (i.e., those created by GraphAddEntries calls).
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphAssemble( HYPRE_SStructGraph graph )
{

   MPI_Comm                  comm        = hypre_SStructGraphComm(graph);
   HYPRE_Int                 ndim        = hypre_SStructGraphNDim(graph);
   hypre_SStructGrid        *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid        *dom_grid    = hypre_SStructGraphDomainGrid(graph);
   HYPRE_Int                 nparts      = hypre_SStructGraphNParts(graph);
   hypre_SStructStencil   ***stencils    = hypre_SStructGraphStencils(graph);
   HYPRE_Int                 nUventries;
   HYPRE_Int                *iUventries;
   hypre_SStructUVEntry    **Uventries;
   HYPRE_Int                 Uvesize;
   HYPRE_BigInt            **Uveoffsets;
   HYPRE_Int                 type        = hypre_SStructGraphObjectType(graph);
   hypre_SStructGraphEntry **add_entries = hypre_SStructGraphEntries(graph);
   HYPRE_Int                 n_add_entries = hypre_SStructNGraphEntries(graph);

   hypre_SStructPGrid       *pgrid;
   hypre_StructGrid         *sgrid;
   HYPRE_Int                 nvars;
   hypre_BoxArray           *boxes;
   hypre_Box                *box;
   HYPRE_Int                 vol, d;

   hypre_SStructGraphEntry  *new_entry;
   hypre_SStructUVEntry     *Uventry;
   HYPRE_Int                 nUentries;
   hypre_SStructUEntry      *Uentries;
   HYPRE_Int                 to_part;
   hypre_IndexRef            to_index;
   HYPRE_Int                 to_var;
   HYPRE_Int                 to_boxnum;
   HYPRE_Int                 to_proc;
   HYPRE_BigInt              Uverank, rank;
   hypre_BoxManEntry        *boxman_entry;

   HYPRE_Int                 nprocs, myproc;
   HYPRE_Int                 part, var;
   hypre_IndexRef            index;
   HYPRE_Int                 i, j;

   /* may need to re-do box managers for the AP*/
   hypre_BoxManager        ***managers = hypre_SStructGridBoxManagers(grid);
   hypre_BoxManager        ***new_managers = NULL;
   hypre_BoxManager          *orig_boxman;
   hypre_BoxManager          *new_boxman;

   HYPRE_Int                  global_n_add_entries;
   HYPRE_Int                  is_gather, k;

   hypre_BoxManEntry         *all_entries, *entry;
   HYPRE_Int                  num_entries;
   void                      *info;
   hypre_Box                 *bbox, *new_box;
   hypre_Box               ***new_gboxes, *new_gbox;
   HYPRE_Int                 *num_ghost;

   /*---------------------------------------------------------
    *  If AP, then may need to redo the box managers
    *
    *  Currently using bounding boxes based on the indexes in add_entries to
    *  determine which boxes to gather in the box managers.  We refer to these
    *  bounding boxes as "gather boxes" here (new_gboxes).  This should work
    *  well in most cases, but it does have the potential to cause lots of grid
    *  boxes to be gathered (hence lots of communication).
    *
    *  A better algorithm would use more care in computing gather boxes that
    *  aren't "too big", while not computing "too many" either (which can also
    *  be slow).  One approach might be to compute an octree with leaves that
    *  have the same volume as the maximum grid box volume.  The leaves would
    *  then serve as the gather boxes.  The number of gather boxes would then be
    *  on the order of the number of local grid boxes (assuming the add_entries
    *  are local, which is generally how they should be used).
    *---------------------------------------------------------*/

   new_box = hypre_BoxCreate(ndim);

   /* if any processor has added entries, then all need to participate */

   hypre_MPI_Allreduce(&n_add_entries, &global_n_add_entries,
                       1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);

   if (global_n_add_entries > 0 )
   {
      /* create new managers */
      new_managers = hypre_TAlloc(hypre_BoxManager **,  nparts, HYPRE_MEMORY_HOST);
      new_gboxes = hypre_TAlloc(hypre_Box **,  nparts, HYPRE_MEMORY_HOST);

      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);

         new_managers[part] = hypre_TAlloc(hypre_BoxManager *,  nvars, HYPRE_MEMORY_HOST);
         new_gboxes[part] = hypre_TAlloc(hypre_Box *,  nvars, HYPRE_MEMORY_HOST);

         for (var = 0; var < nvars; var++)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, var);

            orig_boxman = managers[part][var];
            bbox =  hypre_BoxManBoundingBox(orig_boxman);

            hypre_BoxManCreate(hypre_BoxManNEntries(orig_boxman),
                               hypre_BoxManEntryInfoSize(orig_boxman),
                               hypre_StructGridNDim(sgrid), bbox,
                               hypre_StructGridComm(sgrid),
                               &new_managers[part][var]);
            /* create gather box with flipped bounding box extents */
            new_gboxes[part][var] = hypre_BoxCreate(ndim);
            hypre_BoxSetExtents(new_gboxes[part][var],
                                hypre_BoxIMax(bbox), hypre_BoxIMin(bbox));


            /* need to set the num ghost for new manager also */
            num_ghost = hypre_StructGridNumGhost(sgrid);
            hypre_BoxManSetNumGhost(new_managers[part][var], num_ghost);
         }
      } /* end loop over parts */

      /* now go through the local add entries */
      for (j = 0; j < n_add_entries; j++)
      {
         new_entry = add_entries[j];

         /* check part, var, index, to_part, to_var, to_index */
         for (k = 0; k < 2; k++)
         {
            switch (k)
            {
               case 0:
                  part =  hypre_SStructGraphEntryPart(new_entry);
                  var = hypre_SStructGraphEntryVar(new_entry);
                  index = hypre_SStructGraphEntryIndex(new_entry);
                  break;
               case 1:
                  part =  hypre_SStructGraphEntryToPart(new_entry) ;
                  var =  hypre_SStructGraphEntryToVar(new_entry);
                  index = hypre_SStructGraphEntryToIndex(new_entry);
                  break;
            }

            /* if the index is not within the bounds of the struct grid bounding
               box (which has been set in the box manager) then there should not
               be a coupling here (doesn't make sense) */

            new_boxman = new_managers[part][var];
            new_gbox = new_gboxes[part][var];
            bbox =  hypre_BoxManBoundingBox(new_boxman);

            if (hypre_IndexInBox(index, bbox) != 0)
            {
               /* compute new gather box extents based on index */
               for (d = 0; d < ndim; d++)
               {
                  hypre_BoxIMinD(new_gbox, d) =
                     hypre_min(hypre_BoxIMinD(new_gbox, d), hypre_IndexD(index, d));
                  hypre_BoxIMaxD(new_gbox, d) =
                     hypre_max(hypre_BoxIMaxD(new_gbox, d), hypre_IndexD(index, d));
               }
            }
         }
      }

      /* Now go through the managers and if gather has been called (on any
         processor) then populate the new manager with the entries from the old
         manager and then assemble and delete the old manager. */
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);

         for (var = 0; var < nvars; var++)
         {
            new_boxman = new_managers[part][var];
            new_gbox = new_gboxes[part][var];

            /* call gather if non-empty gather box */
            if (hypre_BoxVolume(new_gbox) > 0)
            {
               hypre_BoxManGatherEntries(
                  new_boxman, hypre_BoxIMin(new_gbox), hypre_BoxIMax(new_gbox));
            }

            /* check to see if gather was called by some processor */
            hypre_BoxManGetGlobalIsGatherCalled(new_boxman, comm, &is_gather);
            if (is_gather)
            {
               /* copy orig boxman information to the new boxman*/

               orig_boxman = managers[part][var];

               hypre_BoxManGetAllEntries(orig_boxman, &num_entries, &all_entries);

               for (j = 0; j < num_entries; j++)
               {
                  entry = &all_entries[j];

                  hypre_BoxManEntryGetInfo(entry, &info);

                  hypre_BoxManAddEntry(new_boxman,
                                       hypre_BoxManEntryIMin(entry),
                                       hypre_BoxManEntryIMax(entry),
                                       hypre_BoxManEntryProc(entry),
                                       hypre_BoxManEntryId(entry),
                                       info);
               }

               /* call assemble for new boxmanager*/
               hypre_BoxManAssemble(new_boxman);

               /* TEMP for testing
                  if (hypre_BoxManNEntries(new_boxman) != num_entries)
                  {
                  hypre_MPI_Comm_rank(comm, &myproc);
                  hypre_printf("myid = %d, new_entries = %d, old entries = %d\n", myproc, hypre_BoxManNEntries(new_boxman), num_entries);
                  } */

               /* destroy old manager */
               hypre_BoxManDestroy (managers[part][var]);
            }
            else /* no gather called */
            {
               /*leave the old manager (so destroy the new one)  */
               hypre_BoxManDestroy(new_boxman);

               /*copy the old to the new */
               new_managers[part][var] = managers[part][var];
            }

            hypre_BoxDestroy(new_gboxes[part][var]);
         } /* end of var loop */
         hypre_TFree(managers[part], HYPRE_MEMORY_HOST);
         hypre_TFree(new_gboxes[part], HYPRE_MEMORY_HOST);
      } /* end of part loop */
      hypre_TFree(managers, HYPRE_MEMORY_HOST);
      hypre_TFree(new_gboxes, HYPRE_MEMORY_HOST);

      /* assign the new ones */
      hypre_SStructGridBoxManagers(grid) = new_managers;
   }

   /* clean up */
   hypre_BoxDestroy(new_box);

   /* end of AP stuff */

   hypre_MPI_Comm_size(comm, &nprocs);
   hypre_MPI_Comm_rank(comm, &myproc);

   /*---------------------------------------------------------
    * Set up UVEntries and iUventries
    *---------------------------------------------------------*/

   /* first set up Uvesize and Uveoffsets */

   Uvesize = 0;
   Uveoffsets = hypre_TAlloc(HYPRE_BigInt *,  nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      Uveoffsets[part] = hypre_TAlloc(HYPRE_BigInt,  nvars, HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         Uveoffsets[part][var] = Uvesize;
         sgrid = hypre_SStructPGridSGrid(pgrid, var);
         boxes = hypre_StructGridBoxes(sgrid);
         hypre_ForBoxI(i, boxes)
         {
            box = hypre_BoxArrayBox(boxes, i);
            vol = 1;
            for (d = 0; d < ndim; d++)
            {
               vol *= (hypre_BoxSizeD(box, d) + 2);
            }
            Uvesize += vol;
         }
      }
   }
   hypre_SStructGraphUVESize(graph)    = Uvesize;
   hypre_SStructGraphUVEOffsets(graph) = Uveoffsets;

   /* now set up nUventries, iUventries, and Uventries */

   iUventries = hypre_TAlloc(HYPRE_Int,  n_add_entries, HYPRE_MEMORY_HOST);
   Uventries = hypre_CTAlloc(hypre_SStructUVEntry *,  Uvesize, HYPRE_MEMORY_HOST);
   hypre_SStructGraphIUVEntries(graph) = iUventries;
   hypre_SStructGraphUVEntries(graph)  = Uventries;

   nUventries = 0;

   /* go through each entry that was added */
   for (j = 0; j < n_add_entries; j++)
   {
      new_entry = add_entries[j];

      part =  hypre_SStructGraphEntryPart(new_entry);
      var = hypre_SStructGraphEntryVar(new_entry);
      index = hypre_SStructGraphEntryIndex(new_entry);
      to_part =  hypre_SStructGraphEntryToPart(new_entry) ;
      to_var =  hypre_SStructGraphEntryToVar(new_entry);
      to_index = hypre_SStructGraphEntryToIndex(new_entry);

      /* compute location (rank) for Uventry */
      hypre_SStructGraphGetUVEntryRank(graph, part, var, index, &Uverank);

      if (Uverank > -1)
      {
         iUventries[nUventries] = Uverank;

         if (Uventries[Uverank] == NULL)
         {
            Uventry = hypre_TAlloc(hypre_SStructUVEntry,  1, HYPRE_MEMORY_HOST);
            hypre_SStructUVEntryPart(Uventry) = part;
            hypre_CopyIndex(index, hypre_SStructUVEntryIndex(Uventry));
            hypre_SStructUVEntryVar(Uventry) = var;
            hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);
            hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rank, type);
            hypre_SStructUVEntryRank(Uventry) = rank;
            nUentries = 1;
            Uentries = hypre_TAlloc(hypre_SStructUEntry,  nUentries, HYPRE_MEMORY_HOST);
         }
         else
         {
            Uventry = Uventries[Uverank];
            nUentries = hypre_SStructUVEntryNUEntries(Uventry) + 1;
            Uentries = hypre_SStructUVEntryUEntries(Uventry);
            Uentries = hypre_TReAlloc(Uentries,  hypre_SStructUEntry,  nUentries, HYPRE_MEMORY_HOST);
         }
         hypre_SStructUVEntryNUEntries(Uventry) = nUentries;
         hypre_SStructUVEntryUEntries(Uventry)  = Uentries;
         hypre_SStructGraphUEMaxSize(graph) =
            hypre_max(hypre_SStructGraphUEMaxSize(graph), nUentries);

         i = nUentries - 1;
         hypre_SStructUVEntryToPart(Uventry, i) = to_part;
         hypre_CopyIndex(to_index, hypre_SStructUVEntryToIndex(Uventry, i));
         hypre_SStructUVEntryToVar(Uventry, i) = to_var;

         hypre_SStructGridFindBoxManEntry(
            dom_grid, to_part, to_index, to_var, &boxman_entry);
         hypre_SStructBoxManEntryGetBoxnum(boxman_entry, &to_boxnum);
         hypre_SStructUVEntryToBoxnum(Uventry, i) = to_boxnum;
         hypre_SStructBoxManEntryGetProcess(boxman_entry, &to_proc);
         hypre_SStructUVEntryToProc(Uventry, i) = to_proc;
         hypre_SStructBoxManEntryGetGlobalRank(
            boxman_entry, to_index, &rank, type);
         hypre_SStructUVEntryToRank(Uventry, i) = rank;

         Uventries[Uverank] = Uventry;

         nUventries++;
         hypre_SStructGraphNUVEntries(graph) = nUventries;

         hypre_SStructGraphUVEntries(graph) = Uventries;
      }
   } /* end of loop through add entries */

   /*---------------------------------------------------------
    * Set up the FEM stencil information
    *---------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      /* only do this if SetFEM was called */
      if (hypre_SStructGraphFEMPNSparse(graph, part))
      {
         HYPRE_Int     fem_nsparse  = hypre_SStructGraphFEMPNSparse(graph, part);
         HYPRE_Int    *fem_sparse_i = hypre_SStructGraphFEMPSparseI(graph, part);
         HYPRE_Int    *fem_sparse_j = hypre_SStructGraphFEMPSparseJ(graph, part);
         HYPRE_Int    *fem_entries  = hypre_SStructGraphFEMPEntries(graph, part);
         HYPRE_Int     fem_nvars    = hypre_SStructGridFEMPNVars(grid, part);
         HYPRE_Int    *fem_vars     = hypre_SStructGridFEMPVars(grid, part);
         hypre_Index  *fem_offsets  = hypre_SStructGridFEMPOffsets(grid, part);
         hypre_Index   offset;
         HYPRE_Int     s, iv, jv, d, nvars, entry;
         HYPRE_Int    *stencil_sizes;
         hypre_Index **stencil_offsets;
         HYPRE_Int   **stencil_vars;

         nvars = hypre_SStructPGridNVars(hypre_SStructGridPGrid(grid, part));

         /* build default full sparsity pattern if nothing set by user */
         if (fem_nsparse < 0)
         {
            fem_nsparse = fem_nvars * fem_nvars;
            fem_sparse_i = hypre_TAlloc(HYPRE_Int,  fem_nsparse, HYPRE_MEMORY_HOST);
            fem_sparse_j = hypre_TAlloc(HYPRE_Int,  fem_nsparse, HYPRE_MEMORY_HOST);
            s = 0;
            for (i = 0; i < fem_nvars; i++)
            {
               for (j = 0; j < fem_nvars; j++)
               {
                  fem_sparse_i[s] = i;
                  fem_sparse_j[s] = j;
                  s++;
               }
            }
            hypre_SStructGraphFEMPNSparse(graph, part) = fem_nsparse;
            hypre_SStructGraphFEMPSparseI(graph, part) = fem_sparse_i;
            hypre_SStructGraphFEMPSparseJ(graph, part) = fem_sparse_j;
         }

         fem_entries = hypre_CTAlloc(HYPRE_Int,  fem_nsparse, HYPRE_MEMORY_HOST);
         hypre_SStructGraphFEMPEntries(graph, part) = fem_entries;

         stencil_sizes   = hypre_CTAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);
         stencil_offsets = hypre_CTAlloc(hypre_Index *,  nvars, HYPRE_MEMORY_HOST);
         stencil_vars    = hypre_CTAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);
         for (iv = 0; iv < nvars; iv++)
         {
            stencil_offsets[iv] = hypre_CTAlloc(hypre_Index,  fem_nvars * fem_nvars, HYPRE_MEMORY_HOST);
            stencil_vars[iv]    = hypre_CTAlloc(HYPRE_Int,  fem_nvars * fem_nvars, HYPRE_MEMORY_HOST);
         }

         for (s = 0; s < fem_nsparse; s++)
         {
            i = fem_sparse_i[s];
            j = fem_sparse_j[s];
            iv = fem_vars[i];
            jv = fem_vars[j];

            /* shift off-diagonal offset by diagonal */
            for (d = 0; d < ndim; d++)
            {
               offset[d] = fem_offsets[j][d] - fem_offsets[i][d];
            }

            /* search stencil_offsets */
            for (entry = 0; entry < stencil_sizes[iv]; entry++)
            {
               /* if offset is already in the stencil, break */
               if ( hypre_IndexesEqual(offset, stencil_offsets[iv][entry], ndim)
                    && (jv == stencil_vars[iv][entry]) )
               {
                  break;
               }
            }
            /* if this is a new stencil offset, add it to the stencil */
            if (entry == stencil_sizes[iv])
            {
               for (d = 0; d < ndim; d++)
               {
                  stencil_offsets[iv][entry][d] = offset[d];
               }
               stencil_vars[iv][entry] = jv;
               stencil_sizes[iv]++;
            }

            fem_entries[s] = entry;
         }

         /* set up the stencils */
         for (iv = 0; iv < nvars; iv++)
         {
            HYPRE_SStructStencilDestroy(stencils[part][iv]);
            HYPRE_SStructStencilCreate(ndim, stencil_sizes[iv],
                                       &stencils[part][iv]);
            for (entry = 0; entry < stencil_sizes[iv]; entry++)
            {
               HYPRE_SStructStencilSetEntry(stencils[part][iv], entry,
                                            stencil_offsets[iv][entry],
                                            stencil_vars[iv][entry]);
            }
         }

         /* free up temporary stuff */
         for (iv = 0; iv < nvars; iv++)
         {
            hypre_TFree(stencil_offsets[iv], HYPRE_MEMORY_HOST);
            hypre_TFree(stencil_vars[iv], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(stencil_sizes, HYPRE_MEMORY_HOST);
         hypre_TFree(stencil_offsets, HYPRE_MEMORY_HOST);
         hypre_TFree(stencil_vars, HYPRE_MEMORY_HOST);
      }
   }

   /*---------------------------------------------------------
    * Sort the iUventries array and eliminate duplicates.
    *---------------------------------------------------------*/

   if (nUventries > 1)
   {
      hypre_qsort0(iUventries, 0, nUventries - 1);

      j = 1;
      for (i = 1; i < nUventries; i++)
      {
         if (iUventries[i] > iUventries[i - 1])
         {
            iUventries[j] = iUventries[i];
            j++;
         }
      }
      nUventries = j;
      hypre_SStructGraphNUVEntries(graph) = nUventries;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphSetObjectType( HYPRE_SStructGraph  graph,
                                 HYPRE_Int           type )
{
   hypre_SStructGraphObjectType(graph) = type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphPrint( FILE *file, HYPRE_SStructGraph graph )
{
   HYPRE_Int                 type = hypre_SStructGraphObjectType(graph);
   HYPRE_Int                 ndim = hypre_SStructGraphNDim(graph);
   HYPRE_Int                 nentries = hypre_SStructNGraphEntries(graph);
   hypre_SStructGraphEntry **entries = hypre_SStructGraphEntries(graph);
   HYPRE_Int                 part, to_part;
   HYPRE_Int                 var, to_var;
   hypre_IndexRef            index, to_index;

   HYPRE_Int                 i;

   /* Print auxiliary info */
   hypre_fprintf(file, "GraphSetObjectType: %d\n", type);

   /* Print SStructGraphEntry info */
   hypre_fprintf(file, "GraphNumEntries: %d", nentries);
   for (i = 0; i < nentries; i++)
   {
      part = hypre_SStructGraphEntryPart(entries[i]);
      var = hypre_SStructGraphEntryVar(entries[i]);
      index = hypre_SStructGraphEntryIndex(entries[i]);
      to_part = hypre_SStructGraphEntryToPart(entries[i]);
      to_var = hypre_SStructGraphEntryToVar(entries[i]);
      to_index = hypre_SStructGraphEntryToIndex(entries[i]);

      hypre_fprintf(file, "\nGraphAddEntries: %d %d ", part, var);
      hypre_IndexPrint(file, ndim, index);
      hypre_fprintf(file, " %d %d ", to_part, to_var);
      hypre_IndexPrint(file, ndim, to_index);
   }
   hypre_fprintf(file, "\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphRead( FILE                  *file,
                        HYPRE_SStructGrid      grid,
                        HYPRE_SStructStencil **stencils,
                        HYPRE_SStructGraph    *graph_ptr )
{
   MPI_Comm                  comm = hypre_SStructGridComm(grid);
   HYPRE_Int                 nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int                 ndim = hypre_SStructGridNDim(grid);

   HYPRE_SStructGraph        graph;
   hypre_SStructGraphEntry **entries;
   hypre_SStructPGrid       *pgrid;
   HYPRE_Int                 nentries;
   HYPRE_Int                 a_entries;
   HYPRE_Int                 part, to_part;
   HYPRE_Int                 var, to_var;
   hypre_Index               index, to_index;

   HYPRE_Int                 type;
   HYPRE_Int                 nvars;
   HYPRE_Int                 i;

   /* Create graph */
   HYPRE_SStructGraphCreate(comm, grid, &graph);

   /* Read auxiliary info */
   hypre_fscanf(file, "GraphSetObjectType: %d\n", &type);
   HYPRE_SStructGraphSetObjectType(graph, type);

   /* Set stencils */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         HYPRE_SStructGraphSetStencil(graph, part, var, stencils[part][var]);
      }
   }

   /* TODO: HYPRE_SStructGraphSetFEM */
   /* TODO: HYPRE_SStructGraphSetFEMSparsity */

   /* Read SStructGraphEntry info */
   hypre_fscanf(file, "GraphNumEntries: %d", &nentries);
   a_entries = nentries + 1;
   hypre_SStructAGraphEntries(graph) = a_entries;
   entries = hypre_CTAlloc(hypre_SStructGraphEntry *, a_entries, HYPRE_MEMORY_HOST);
   hypre_SStructGraphEntries(graph) = entries;
   for (i = 0; i < nentries; i++)
   {
      hypre_fscanf(file, "\nGraphAddEntries: %d %d ", &part, &var);
      hypre_IndexRead(file, ndim, index);
      hypre_fscanf(file, " %d %d ", &to_part, &to_var);
      hypre_IndexRead(file, ndim, to_index);

      HYPRE_SStructGraphAddEntries(graph, part, index, var, to_part, to_index, to_var);
   }
   hypre_fscanf(file, "\n");

   *graph_ptr = graph;

   return hypre_error_flag;
}
