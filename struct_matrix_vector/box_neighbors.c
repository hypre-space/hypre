/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for the hypre_BoxNeighbors class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewRankLink:
 *--------------------------------------------------------------------------*/

hypre_RankLink *
hypre_NewRankLink( int  rank )
{
   hypre_RankLink  *rank_link;

   rank_link = hypre_TAlloc(hypre_RankLink, 1);

   hypre_RankLinkRank(rank_link) = rank;
   hypre_RankLinkNext(rank_link) = NULL;

   return rank_link;
}

/*--------------------------------------------------------------------------
 * hypre_FreeRankLink:
 *--------------------------------------------------------------------------*/

int
hypre_FreeRankLink( hypre_RankLink  *rank_link )
{
   int  ierr = 0;

   if (rank_link)
   {
      hypre_TFree(rank_link);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NewBoxNeighbors:
 *
 * Finds boxes that are "near" the box given by `box_rank', where near
 * is defined by max_distance.
 *--------------------------------------------------------------------------*/

hypre_BoxNeighbors *
hypre_NewBoxNeighbors( hypre_BoxArray  *boxes,
                       int              box_rank,
                       int              max_distance )
{
   hypre_BoxNeighbors  *neighbors;
   hypre_RankLink      *rank_link;

   hypre_Box           *box;
   hypre_Box           *neighbor_box;

   int                  distance;
   int                  distance_index[3];

   int                  diff;
   int                  i, d;

   /*-----------------------------------------------------------------------
    * Initialize the hypre_BoxNeighbors structure
    *-----------------------------------------------------------------------*/

   neighbors = hypre_CTAlloc(hypre_BoxNeighbors, 1);

   hypre_BoxNeighborsBoxes(neighbors)       = boxes;
   hypre_BoxNeighborsBoxRank(neighbors)     = box_rank;
   hypre_BoxNeighborsMaxDistance(neighbors) = max_distance;

   /*-----------------------------------------------------------------------
    * Find neighboring boxes
    *-----------------------------------------------------------------------*/

   box = hypre_BoxArrayBox(boxes, box_rank);
   hypre_ForBoxI(i, boxes)
      {
         if (i != box_rank)
         {
            neighbor_box = hypre_BoxArrayBox(boxes, i);

            /* compute distance info */
            distance = 0;
            for (d = 0; d < 3; d++)
            {
               distance_index[d] = 0;

               diff = hypre_BoxIMinD(neighbor_box, d) - hypre_BoxIMaxD(box, d);
               if (diff > 0)
               {
                  distance_index[d] = 1;
                  distance = max(distance, diff);
               }

               diff = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(neighbor_box, d);
               if (diff > 0)
               {
                  distance_index[d] = -1;
                  distance = max(distance, diff);
               }
            }

            /* create new rank_link */
            if (distance <= max_distance)
            {
               rank_link = hypre_NewRankLink(i);
               hypre_RankLinkNext(rank_link) =
                  hypre_BoxNeighborsRankLink(neighbors,
                                             distance_index[0],
                                             distance_index[1],
                                             distance_index[2]);
               hypre_BoxNeighborsRankLink(neighbors,
                                          distance_index[0],
                                          distance_index[1],
                                          distance_index[2]) = rank_link;
            }
         }
      }

   return neighbors;
}

/*--------------------------------------------------------------------------
 * hypre_FreeBoxNeighbors:
 *--------------------------------------------------------------------------*/

int
hypre_FreeBoxNeighbors( hypre_BoxNeighbors  *neighbors )
{
   hypre_RankLink  *rank_link;
   hypre_RankLink  *next_rank_link;

   int              i, j, k;

   int              ierr = 0;

   if (neighbors)
   {
      for (k = -1; k <= 1; k++)
      {
         for (j = -1; j <= 1; j++)
         {
            for (i = -1; i <= 1; i++)
            {
               rank_link = hypre_BoxNeighborsRankLink(neighbors, i, j, k);
               while (rank_link)
               {
                  next_rank_link = hypre_RankLinkNext(rank_link);
                  hypre_FreeRankLink(rank_link);
                  rank_link = next_rank_link;
               }
            }
         }
      }
      hypre_TFree(neighbors);
   }

   return ierr;
}
