/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.20 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "headers.h"

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

   graph = hypre_TAlloc(hypre_SStructGraph, 1);

   hypre_SStructGraphComm(graph) = comm;
   hypre_SStructGraphNDim(graph) = hypre_SStructGridNDim(grid);
   hypre_SStructGridRef(grid, &hypre_SStructGraphGrid(graph));
   hypre_SStructGridRef(grid, &hypre_SStructGraphDomainGrid(graph));
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructGraphNParts(graph) = nparts;
   pgrids = hypre_SStructGridPGrids(grid);
   stencils = hypre_TAlloc(hypre_SStructStencil **, nparts);
   fem_nsparse  = hypre_TAlloc(HYPRE_Int, nparts);
   fem_sparse_i = hypre_TAlloc(HYPRE_Int *, nparts);
   fem_sparse_j = hypre_TAlloc(HYPRE_Int *, nparts);
   fem_entries  = hypre_TAlloc(HYPRE_Int *, nparts);
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      stencils[part]  = hypre_TAlloc(hypre_SStructStencil *, nvars);
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

   hypre_SStructGraphNUVEntries(graph)  = 0;
   hypre_SStructGraphAUVEntries(graph)  = 0;
   hypre_SStructGraphIUVEntries(graph)  = NULL;

   hypre_SStructGraphUVEntries(graph)   = NULL;
   hypre_SStructGraphTotUEntries(graph) = 0;
   hypre_SStructGraphRefCount(graph)    = 1;
   hypre_SStructGraphObjectType(graph)  = HYPRE_SSTRUCT;

   hypre_SStructGraphEntries(graph)     = NULL;
   hypre_SStructNGraphEntries(graph)    = 0;
   hypre_SStructAGraphEntries(graph)    = 0;
   
   *graph_ptr = graph;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphDestroy( HYPRE_SStructGraph graph )
{
   HYPRE_Int               nparts;
   hypre_SStructPGrid    **pgrids;
   hypre_SStructStencil ***stencils;
   HYPRE_Int              *fem_nsparse;
   HYPRE_Int             **fem_sparse_i;
   HYPRE_Int             **fem_sparse_j;
   HYPRE_Int             **fem_entries;
   HYPRE_Int               nUventries;
   HYPRE_Int              *iUventries;
 
   hypre_SStructUVEntry  **Uventries;
   hypre_SStructUVEntry   *Uventry;
   HYPRE_Int               nvars;
   HYPRE_Int               part, var, i;

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
         for (part = 0; part < nparts; part++)
         {
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               HYPRE_SStructStencilDestroy(stencils[part][var]);
            }
            hypre_TFree(stencils[part]);
            hypre_TFree(fem_sparse_i[part]);
            hypre_TFree(fem_sparse_j[part]);
            hypre_TFree(fem_entries[part]);
         }
         HYPRE_SStructGridDestroy(hypre_SStructGraphGrid(graph));
         HYPRE_SStructGridDestroy(hypre_SStructGraphDomainGrid(graph));
         hypre_TFree(stencils);
         hypre_TFree(fem_nsparse);
         hypre_TFree(fem_sparse_i);
         hypre_TFree(fem_sparse_j);
         hypre_TFree(fem_entries);
         /* RDF: THREAD? */
         for (i = 0; i < nUventries; i++)
         {
            Uventry = Uventries[iUventries[i]];
            if (Uventry)
            {
               hypre_TFree(hypre_SStructUVEntryUEntries(Uventry));
               hypre_TFree(Uventry);
            }
            Uventries[iUventries[i]] = NULL;
         }
         hypre_TFree(iUventries);
         hypre_TFree(Uventries);
         hypre_TFree(graph);
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
   fem_sparse_i = hypre_TAlloc(HYPRE_Int, nsparse);
   fem_sparse_j = hypre_TAlloc(HYPRE_Int, nsparse);
   for (s = 0; s < nsparse; s++)
   {
      fem_sparse_i[s] = sparsity[2*s];
      fem_sparse_j[s] = sparsity[2*s+1];
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
      entries = hypre_CTAlloc(hypre_SStructGraphEntry *, a_entries);

      hypre_SStructAGraphEntries(graph) = a_entries;
      hypre_SStructGraphEntries(graph) = entries;
   }
   else if (n_entries >= a_entries)
   {
      a_entries += 1000;
      entries = hypre_TReAlloc(entries, hypre_SStructGraphEntry *, a_entries);
   
      hypre_SStructAGraphEntries(graph) = a_entries;
      hypre_SStructGraphEntries(graph) = entries;
   }
   
   /*save parameters to a new entry */

   new_entry = hypre_TAlloc(hypre_SStructGraphEntry, 1);

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
 * graph entries (i.e., those created by GraphAddEntries calls).  The
 * routine will compute as many of these numbers on-process, but if
 * the information needed to compute a column is not stored locally,
 * it will be computed off-process instead.
 *
 * RDF: Is this off-process code really needed?
 *
 * To do this, column info is first requested from other processes
 * (tag=1 communications).  While waiting for this info, requests from
 * other processes are filled (tag=2).  Simultaneously, a fanin/fanout
 * procedure (tag=0) is invoked to determine when to stop: each
 * process participates in the send portion of the fanin once it has
 * received all of its requested column data and once it has completed
 * its receive portion of the fanin; each process then participates in
 * the fanout.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGraphAssemble( HYPRE_SStructGraph graph )
{

   MPI_Comm               comm        = hypre_SStructGraphComm(graph);
   hypre_SStructGrid     *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid     *dom_grid    = hypre_SStructGraphDomainGrid(graph);
   HYPRE_Int              nparts      = hypre_SStructGraphNParts(graph);
   hypre_SStructStencil ***stencils    = hypre_SStructGraphStencils(graph);
   HYPRE_Int              nUventries;
   HYPRE_Int             *iUventries;
   hypre_SStructUVEntry **Uventries;
   
   HYPRE_Int              type        = hypre_SStructGraphObjectType(graph);


   hypre_SStructUVEntry  *Uventry;
   HYPRE_Int              nUentries;
   HYPRE_Int              to_part;
   hypre_IndexRef         to_index;
   HYPRE_Int              to_var;
   HYPRE_Int              to_boxnum;
   HYPRE_Int              to_proc;
   HYPRE_Int              rank;
   hypre_BoxManEntry     *boxman_entry;

   HYPRE_Int              nprocs, myproc;
   
   HYPRE_Int              i, j;

   /* data from add entries calls */
   hypre_SStructGraphEntry **add_entries = hypre_SStructGraphEntries(graph);
   HYPRE_Int                 n_add_entries = hypre_SStructNGraphEntries(graph);
   hypre_SStructGraphEntry  *new_entry;
   HYPRE_Int                 part, var;
   hypre_IndexRef            index;
   hypre_Index               cindex;
   HYPRE_Int                 startrank;
   HYPRE_Int                 boxnum;
   HYPRE_Int                 aUventries;
   HYPRE_Int                 ndim       = hypre_SStructGridNDim(grid);
   hypre_SStructUEntry      *Uentries;

#if  HYPRE_NO_GLOBAL_PARTITION

   /* may need to re-do box managers for the AP*/
   hypre_BoxManager        ***managers = hypre_SStructGridBoxManagers(grid);
   hypre_BoxManager        ***new_managers = NULL;
   hypre_BoxManager          *orig_boxman;
   hypre_BoxManager          *new_boxman;
   
   HYPRE_Int                  global_n_add_entries;
   HYPRE_Int                  is_gather;
   
   hypre_SStructPGrid        *pgrid;
   HYPRE_Int                  nvars;
   hypre_BoxManEntry         *all_entries, *entry;
   HYPRE_Int                  num_entries;
   void                      *info;
   hypre_Box                 *bbox, *new_box;
   hypre_StructGrid          *sgrid;
   HYPRE_Int                 *num_ghost;

   /*---------------------------------------------------------
    *  If AP, then may need to redo the box managers
    *---------------------------------------------------------*/

   new_box = hypre_BoxCreate();
   
   /* if any processor has added entries, then all need to participate */

   hypre_MPI_Allreduce(&n_add_entries, &global_n_add_entries,
                       1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
 
   if (global_n_add_entries > 0 )
   {
      /* create new managers */
      new_managers = hypre_TAlloc(hypre_BoxManager **, nparts);

      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);
         
         new_managers[part] = hypre_TAlloc(hypre_BoxManager *, nvars);
         
         for (var = 0; var < nvars; var++)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, var);
       
            orig_boxman = managers[part][var];
            
            hypre_BoxManCreate(hypre_BoxManNEntries(orig_boxman), 
                               hypre_BoxManEntryInfoSize(orig_boxman), 
                               hypre_StructGridDim(sgrid),
                               hypre_BoxManBoundingBox(orig_boxman),  
                               hypre_StructGridComm(sgrid),
                               &new_managers[part][var]);

            /* need to set the num ghost for new manager also */
            num_ghost = hypre_StructGridNumGhost(sgrid);
            hypre_BoxManSetNumGhost(new_managers[part][var], num_ghost);
         }
      } /* end loop over parts */

      /* now go through the local add entries */
      for (j = 0; j < n_add_entries; j++)
      {
         new_entry = add_entries[j];

         /* check part, var, index */
         part =  hypre_SStructGraphEntryPart(new_entry);
         var = hypre_SStructGraphEntryVar(new_entry);
         index = hypre_SStructGraphEntryIndex(new_entry);

         /* if the index is not within the bounds of the struct grid
            bounding box (which has been set in the box manager) then
            there should noit be a coupling here (doens't make
            sense */

         new_boxman = new_managers[part][var];

         bbox =  hypre_BoxManBoundingBox(new_boxman);
         
         if (hypre_IndexInBoxP(index,bbox) != 0)
         {
            hypre_BoxManGatherEntries(new_boxman,index, index);
         }
         
         /* now repeat the check for to_part, to_var, to_index */
         to_part =  hypre_SStructGraphEntryToPart(new_entry) ;
         to_var =  hypre_SStructGraphEntryToVar(new_entry);
         to_index = hypre_SStructGraphEntryToIndex(new_entry);

         new_boxman = new_managers[to_part][to_var];
 
         bbox =  hypre_BoxManBoundingBox(new_boxman);
         
         if (hypre_IndexInBoxP(to_index,bbox) != 0)
         {
            hypre_BoxManGatherEntries(new_boxman,to_index, to_index);
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
            hypre_BoxManGetGlobalIsGatherCalled(new_boxman, comm, &is_gather);
            if (is_gather)
            {
               /* Gather has been called on at least 1 proc - copy
                * orig boxman information to the new boxman*/
               
               orig_boxman = managers[part][var];

               hypre_BoxManGetAllEntries(orig_boxman, &num_entries, &all_entries);
               
               for (j=0; j< num_entries; j++)
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
            
         } /* end of var loop */
         hypre_TFree(managers[part]);
      } /* end of part loop */
      hypre_TFree(managers);
   
      /* assign the new ones */
      hypre_SStructGridBoxManagers(grid) = new_managers;
   }

   /* clean up */
   hypre_BoxDestroy(new_box);

   /* end of AP stuff */
#endif

   hypre_MPI_Comm_size(comm, &nprocs);
   hypre_MPI_Comm_rank(comm, &myproc);

   /*---------------------------------------------------------
    * First we do the work that was previously in the AddEntries:
    * set up the UVEntry and iUventries
    *---------------------------------------------------------*/

   /* allocate proper storage */
   aUventries = hypre_max(hypre_SStructGridGhlocalSize(grid), n_add_entries);

   iUventries = hypre_TAlloc(HYPRE_Int, aUventries);
   Uventries = hypre_CTAlloc(hypre_SStructUVEntry *, aUventries);

   hypre_SStructGraphAUVEntries(graph) = aUventries;
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
      hypre_CopyToCleanIndex(index, ndim, cindex);

      hypre_SStructGridFindBoxManEntry(grid, part, cindex, var, &boxman_entry);
    
      /* GEC0203 getting the rank */ 
      hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, cindex, &rank, type);
      
      /* GEC 0902 filling up the iUventries with local ghrank
       * since HYPRE_SSTRUCT is chosen */

      if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT) 
      { 
         startrank = hypre_SStructGridGhstartRank(grid);
      }
      if (type == HYPRE_PARCSR)
      {
         startrank = hypre_SStructGridStartRank(grid);
      }

      rank -= startrank;

      iUventries[nUventries] = rank;

      if (Uventries[rank] == NULL)
      {
         Uventry = hypre_TAlloc(hypre_SStructUVEntry, 1);
         hypre_SStructUVEntryPart(Uventry) = part;
         hypre_CopyToCleanIndex(index, ndim, hypre_SStructUVEntryIndex(Uventry));
         hypre_SStructUVEntryVar(Uventry) = var;
         hypre_SStructBoxManEntryGetBoxnum(boxman_entry, &boxnum);
         hypre_SStructUVEntryBoxnum(Uventry) = boxnum;
         nUentries = 1;
         Uentries = hypre_TAlloc(hypre_SStructUEntry, nUentries);
      }
      else
      {
         Uventry = Uventries[rank];
         nUentries = hypre_SStructUVEntryNUEntries(Uventry) + 1;
         Uentries = hypre_SStructUVEntryUEntries(Uventry);
         Uentries = hypre_TReAlloc(Uentries, hypre_SStructUEntry, nUentries);
      }
      hypre_SStructUVEntryNUEntries(Uventry) = nUentries;
      hypre_SStructUVEntryUEntries(Uventry)  = Uentries;

      i = nUentries - 1;
      hypre_SStructUVEntryToPart(Uventry, i) = to_part;
      hypre_CopyToCleanIndex(to_index, ndim,
                             hypre_SStructUVEntryToIndex(Uventry, i));
      hypre_SStructUVEntryToVar(Uventry, i) = to_var;
      
      hypre_CopyToCleanIndex(to_index, ndim, cindex);
           
      hypre_SStructGridFindBoxManEntry(
         dom_grid, to_part, cindex, to_var, &boxman_entry);
      hypre_SStructBoxManEntryGetBoxnum(boxman_entry, &to_boxnum);
      hypre_SStructUVEntryToBoxnum(Uventry, i) = to_boxnum;
      hypre_SStructBoxManEntryGetProcess(boxman_entry, &to_proc);
      hypre_SStructUVEntryToProc(Uventry, i)= to_proc;
      
      Uventries[rank] = Uventry; /* GEC1102 where rank labels Uventries */
      
      nUventries++;
      hypre_SStructGraphNUVEntries(graph) = nUventries;

      hypre_SStructGraphUVEntries(graph) = Uventries;
      
      hypre_SStructGraphTotUEntries(graph) ++;

      /*free each add entry after copying */
      hypre_TFree(new_entry);

   }/* end of loop through add entries */
   
   /* free the storage for the add entires */
   hypre_TFree(add_entries);
   
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
            fem_sparse_i = hypre_TAlloc(HYPRE_Int, fem_nsparse);
            fem_sparse_j = hypre_TAlloc(HYPRE_Int, fem_nsparse);
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

         fem_entries = hypre_CTAlloc(HYPRE_Int, fem_nsparse);
         hypre_SStructGraphFEMPEntries(graph, part) = fem_entries;

         stencil_sizes   = hypre_CTAlloc(HYPRE_Int, nvars);
         stencil_offsets = hypre_CTAlloc(hypre_Index *, nvars);
         stencil_vars    = hypre_CTAlloc(HYPRE_Int *, nvars);
         for (iv = 0; iv < nvars; iv++)
         {
            stencil_offsets[iv] = hypre_CTAlloc(hypre_Index, fem_nvars*fem_nvars);
            stencil_vars[iv]    = hypre_CTAlloc(HYPRE_Int, fem_nvars*fem_nvars);
         }

         for (s = 0; s < fem_nsparse; s++)
         {
            i = fem_sparse_i[s];
            j = fem_sparse_j[s];
            iv = fem_vars[i];
            jv = fem_vars[j];

            /* shift off-diagonal offset by diagonal */
            for (d = 0; d < 3; d++)
            {
               offset[d] = fem_offsets[j][d] - fem_offsets[i][d];
            }

            /* search stencil_offsets */
            for (entry = 0; entry < stencil_sizes[iv]; entry++)
            {
               /* if offset is already in the stencil, break */
               if ( (offset[0] == stencil_offsets[iv][entry][0]) &&
                    (offset[1] == stencil_offsets[iv][entry][1]) &&
                    (offset[2] == stencil_offsets[iv][entry][2]) &&
                    (jv == stencil_vars[iv][entry]) )
               {
                  break;
               }
            }
            /* if this is a new stencil offset, add it to the stencil */
            if (entry == stencil_sizes[iv])
            {
               stencil_offsets[iv][entry][0] = offset[0];
               stencil_offsets[iv][entry][1] = offset[1];
               stencil_offsets[iv][entry][2] = offset[2];
               stencil_vars[iv][entry]       = jv;
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
            hypre_TFree(stencil_offsets[iv]);
            hypre_TFree(stencil_vars[iv]);
         }
         hypre_TFree(stencil_sizes);
         hypre_TFree(stencil_offsets);
         hypre_TFree(stencil_vars);
      }
   }

   /*---------------------------------------------------------
    * Sort the iUventries array and eliminate duplicates.
    *---------------------------------------------------------*/

   if (nUventries > 1)
   {
      qsort0(iUventries, 0, nUventries - 1);

      j = 1;
      for (i = 1; i < nUventries; i++)
      {
         if (iUventries[i] > iUventries[i-1])
         {
            iUventries[j] = iUventries[i];
            j++;
         }
      }
      nUventries = j;
      hypre_SStructGraphNUVEntries(graph) = nUventries;
   }

   /*---------------------------------------------------------
    * Compute non-stencil column numbers (if possible), and
    * start building requests for needed off-process info.
    *---------------------------------------------------------*/
  
   /* RDF: THREAD? */
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      nUentries = hypre_SStructUVEntryNUEntries(Uventry);
      for (j = 0; j < nUentries; j++)
      {
         to_part  = hypre_SStructUVEntryToPart(Uventry, j);
         to_index = hypre_SStructUVEntryToIndex(Uventry, j);
         to_var   = hypre_SStructUVEntryToVar(Uventry, j);

         /*---------------------------------------------------------
          * used in future? The to_boxnum corresponds to the first
          * map_entry on the map_entry link list.
          *---------------------------------------------------------*/

         to_boxnum = hypre_SStructUVEntryToBoxnum(Uventry, j);
         to_proc   = hypre_SStructUVEntryToProc(Uventry, j);
         hypre_SStructGridBoxProcFindBoxManEntry(
            dom_grid, to_part, to_var, to_boxnum, to_proc, &boxman_entry);
         if (boxman_entry != NULL)
         {
            /* compute ranks locally */
            hypre_SStructBoxManEntryGetGlobalRank(
               boxman_entry, to_index, &rank, type);          
            hypre_SStructUVEntryRank(Uventry, j) = rank;
         }
         else   
         {
            /* This should not happen (TO DO: take out print statement) */
            hypre_printf("Error in HYPRE_SStructGraphAssemble, my id = %d\n",
                         myproc);
         }
      }
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
