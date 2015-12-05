/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Member functions for hypre_SStructGraph class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGraphRef( hypre_SStructGraph  *graph,
                       hypre_SStructGraph **graph_ref )
{
   hypre_SStructGraphRefCount(graph) ++;
   *graph_ref = graph;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: This may search an Octree in the future.
 *
 * 9/09 AB - modified to use the box manager
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGraphFindUVEntry( hypre_SStructGraph    *graph,
                               HYPRE_Int              part,
                               hypre_Index            index,
                               HYPRE_Int              var,
                               hypre_SStructUVEntry **Uventry_ptr )
{
   hypre_SStructUVEntry **Uventries = hypre_SStructGraphUVEntries(graph);
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   HYPRE_Int             type       = hypre_SStructGraphObjectType(graph);
   hypre_BoxManEntry     *boxman_entry;
   HYPRE_Int              rank;


   
   /* Should we be checking the neighbor box manager also ?*/

   hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);
   hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rank, type);

   if (type == HYPRE_SSTRUCT || type ==  HYPRE_STRUCT)
   {
    rank -= hypre_SStructGridGhstartRank(grid);
   }
   if (type == HYPRE_PARCSR)
   {
    rank -= hypre_SStructGridStartRank(grid);
   }
 
   *Uventry_ptr = Uventries[rank];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the local Uventries index for the endpt of a box. This index
 * can be used to localize a search for Uventries of a box.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes

 * 9/09 AB - modified to use the box manager
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGraphFindBoxEndpt(hypre_SStructGraph    *graph,
                               HYPRE_Int              part,
                               HYPRE_Int              var,
                               HYPRE_Int              proc,
                               HYPRE_Int              endpt,
                               HYPRE_Int              boxi)
{
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   HYPRE_Int              type      = hypre_SStructGraphObjectType(graph);
   hypre_BoxManager      *boxman;
   hypre_BoxManEntry     *boxman_entry;
   hypre_StructGrid      *sgrid;
   hypre_Box             *box;
   HYPRE_Int              rank;

   /* Should we be checking the neighbor box manager also ?*/

   boxman= hypre_SStructGridBoxManager(grid, part, var);
   hypre_BoxManGetEntry(boxman, proc, boxi, &boxman_entry);

   sgrid= hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
   box  = hypre_StructGridBox(sgrid, boxi);

   /* get the global rank of the endpt corner of box boxi */
   if (endpt < 1)
   {
       hypre_SStructBoxManEntryGetGlobalRank(
          boxman_entry, hypre_BoxIMin(box), &rank, type);
   }

   else
   {
       hypre_SStructBoxManEntryGetGlobalRank(
          boxman_entry, hypre_BoxIMax(box), &rank, type);
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
 * Computes the local Uventries index for the start or end of each box of
 * a given sgrid.
 *      endpt= 0   start of boxes
 *      endpt= 1   end of boxes
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGraphFindSGridEndpts(hypre_SStructGraph    *graph,
                                  HYPRE_Int              part,
                                  HYPRE_Int              var,
                                  HYPRE_Int              proc,
                                  HYPRE_Int              endpt,
                                  HYPRE_Int             *endpts)
{
   hypre_SStructGrid     *grid      = hypre_SStructGraphGrid(graph);
   hypre_StructGrid      *sgrid;
   hypre_BoxArray        *boxes;
   HYPRE_Int              i;

   sgrid= hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
   boxes= hypre_StructGridBoxes(sgrid);

   /* get the endpts using hypre_SStructGraphFindBoxEndpt */
   for (i= 0; i< hypre_BoxArraySize(boxes); i++)
   {
      endpts[i]= hypre_SStructGraphFindBoxEndpt(graph, part, var, proc, endpt, i);
   }

   return hypre_error_flag;
}

