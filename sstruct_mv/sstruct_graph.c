/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Member functions for hypre_SStructGraph class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructGraphRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructGraphRef( hypre_SStructGraph  *graph,
                       hypre_SStructGraph **graph_ref )
{
   hypre_SStructGraphRefCount(graph) ++;
   *graph_ref = graph;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SStructGraphFindUVEntry
 *
 * NOTE: This may search an Octree in the future.
 *--------------------------------------------------------------------------*/

int
hypre_SStructGraphFindUVEntry( hypre_SStructGraph    *graph,
                               int                    part,
                               hypre_Index            index,
                               int                    var,
                               hypre_SStructUVEntry **Uventry_ptr )
{
   int ierr = 0;

   hypre_SStructUVEntry **Uventries = hypre_SStructGraphUVEntries(graph);
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   int                   type       = hypre_SStructGraphObjectType(graph);
   hypre_BoxMapEntry     *map_entry;
   int                    rank;

   hypre_SStructGridFindMapEntry(grid, part, index, var, &map_entry);
   hypre_SStructMapEntryGetGlobalRank(map_entry, index, &rank, type);

   if (type == HYPRE_SSTRUCT || type ==  HYPRE_STRUCT)
   {
    rank -= hypre_SStructGridGhstartRank(grid);
   }
   if (type == HYPRE_PARCSR)
   {
    rank -= hypre_SStructGridStartRank(grid);
   }
 
   *Uventry_ptr = Uventries[rank];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructGraphFindBoxEndpt
 *
 * Computes the local Uventries index for the endpt of a box. This index
 * can be used to localize a search for Uventries of a box.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes
 *--------------------------------------------------------------------------*/

int
hypre_SStructGraphFindBoxEndpt(hypre_SStructGraph    *graph,
                               int                    part,
                               int                    var,
                               int                    proc,
                               int                    endpt,
                               int                    boxi)
{
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   int                    type      = hypre_SStructGraphObjectType(graph);
   hypre_BoxMap          *map;
   hypre_BoxMapEntry     *map_entry;
   hypre_StructGrid      *sgrid;
   hypre_Box             *box;
   int                    rank;

   map= hypre_SStructGridMap(grid, part, var);
   hypre_BoxMapFindBoxProcEntry(map, boxi, proc, &map_entry);

   sgrid= hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
   box  = hypre_StructGridBox(sgrid, boxi);

   /* get the global rank of the endpt corner of box boxi */
   if (endpt < 1)
   {
       hypre_SStructMapEntryGetGlobalRank(map_entry, hypre_BoxIMin(box), &rank,
                                          type);
   }

   else
   {
       hypre_SStructMapEntryGetGlobalRank(map_entry, hypre_BoxIMax(box), &rank,
                                          type);
   }

   if (type == HYPRE_SSTRUCT || type ==  HYPRE_STRUCT)
   {
    rank -= hypre_SStructGridGhstartRank(grid);
   }
   if (type == HYPRE_PARCSR)
   {
    rank -= hypre_SStructGridStartRank(grid);
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * hypre_SStructGraphFindSGridEndpts
 *
 * Computes the local Uventries index for the start or end of each box of
 * a given sgrid.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes
 *--------------------------------------------------------------------------*/

int
hypre_SStructGraphFindSGridEndpts(hypre_SStructGraph    *graph,
                                  int                    part,
                                  int                    var,
                                  int                    proc,
                                  int                    endpt,
                                  int                   *endpts)
{
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   hypre_StructGrid      *sgrid;
   hypre_BoxArray        *boxes;
   int                    i;

   sgrid= hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
   boxes= hypre_StructGridBoxes(sgrid);

   /* get the endpts using hypre_SStructGraphFindBoxEndpt */
   for (i= 0; i< hypre_BoxArraySize(boxes); i++)
   {
      endpts[i]= hypre_SStructGraphFindBoxEndpt(graph, part, var, proc, endpt, i);
   }

   return 0;
}

