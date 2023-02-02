/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_SStructGraph class.
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

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
 * Uventries are stored in an array indexed via a local rank that comes from an
 * ordering of the local grid boxes with ghost zones added.  Since a grid index
 * may intersect multiple grid boxes, the box with the smallest boxnum is used.
 *
 * RDF: Consider using another "local" BoxManager to optimize.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructGraphGetUVEntryRank( hypre_SStructGraph    *graph,
                                  HYPRE_Int              part,
                                  HYPRE_Int              var,
                                  hypre_Index            index,
                                  HYPRE_BigInt          *rank )
{
   HYPRE_Int              ndim  = hypre_SStructGraphNDim(graph);
   hypre_SStructGrid     *grid  = hypre_SStructGraphGrid(graph);
   hypre_SStructPGrid    *pgrid = hypre_SStructGridPGrid(grid, part);
   hypre_StructGrid      *sgrid = hypre_SStructPGridSGrid(pgrid, var);
   hypre_BoxArray        *boxes = hypre_StructGridBoxes(sgrid);
   hypre_Box             *box;
   HYPRE_Int              i, d, vol, found;


   *rank = hypre_SStructGraphUVEOffset(graph, part, var);
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      found = 1;
      for (d = 0; d < ndim; d++)
      {
         if ( (hypre_IndexD(index, d) < (hypre_BoxIMinD(box, d) - 1)) ||
              (hypre_IndexD(index, d) > (hypre_BoxIMaxD(box, d) + 1)) )
         {
            /* not in this box */
            found = 0;
            break;
         }
      }
      if (found)
      {
         vol = 0;
         for (d = (ndim - 1); d > -1; d--)
         {
            vol = vol * (hypre_BoxSizeD(box, d) + 2) +
                  (hypre_IndexD(index, d) - hypre_BoxIMinD(box, d) + 1);
         }
         *rank += (HYPRE_BigInt)vol;
         return hypre_error_flag;
      }
      else
      {
         vol = 1;
         for (d = 0; d < ndim; d++)
         {
            vol *= (hypre_BoxSizeD(box, d) + 2);
         }
         *rank += (HYPRE_BigInt)vol;
      }
   }

   /* a value of -1 indicates that the index was not found */
   *rank = -1;

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
   HYPRE_BigInt           rank;

   /* Should we be checking the neighbor box manager also ?*/

   boxman = hypre_SStructGridBoxManager(grid, part, var);
   hypre_BoxManGetEntry(boxman, proc, boxi, &boxman_entry);

   sgrid = hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
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

   sgrid = hypre_SStructPGridSGrid(hypre_SStructGridPGrid(grid, part), var);
   boxes = hypre_StructGridBoxes(sgrid);

   /* get the endpts using hypre_SStructGraphFindBoxEndpt */
   for (i = 0; i < hypre_BoxArraySize(boxes); i++)
   {
      endpts[i] = hypre_SStructGraphFindBoxEndpt(graph, part, var, proc, endpt, i);
   }

   return hypre_error_flag;
}

