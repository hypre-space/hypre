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
   hypre_BoxMapEntry     *map_entry;
   int                    rank;

   hypre_SStructGridFindMapEntry(grid, part, index, var, &map_entry);
   hypre_SStructBoxMapEntryGetGlobalRank(map_entry, index, &rank);
   rank -= hypre_SStructGridStartRank(grid);

   *Uventry_ptr = Uventries[rank];

   return ierr;
}

