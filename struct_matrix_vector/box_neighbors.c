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
 * hypre_RankLinkCreate:
 *--------------------------------------------------------------------------*/

hypre_RankLink *
hypre_RankLinkCreate( int  rank )
{
   hypre_RankLink  *rank_link;

   rank_link = hypre_TAlloc(hypre_RankLink, 1);

   hypre_RankLinkRank(rank_link) = rank;
   hypre_RankLinkNext(rank_link) = NULL;

   return rank_link;
}

/*--------------------------------------------------------------------------
 * hypre_RankLinkDestroy:
 *--------------------------------------------------------------------------*/

int
hypre_RankLinkDestroy( hypre_RankLink  *rank_link )
{
   int  ierr = 0;

   if (rank_link)
   {
      hypre_TFree(rank_link);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BoxNeighborsCreate:
 *
 * Finds boxes that are "near" the boxes given by `local_ranks',
 * where near is defined by max_distance.
 *
 * Note: This routine does not consider a box to be a neighbor of itself,
 * and will not include it in the list of neighboring boxes.
 *--------------------------------------------------------------------------*/

hypre_BoxNeighbors *
hypre_BoxNeighborsCreate( int             *local_ranks,
                          int              num_local,
                          hypre_BoxArray  *boxes,
                          int             *processes,
                          int              max_distance )
{
   hypre_BoxArray      *new_boxes;
   int                 *new_processes;
   int                 *new_boxes_ranks;
   int                  num_new_boxes;

   hypre_BoxNeighbors  *neighbors;
   hypre_RankLink      *rank_link;

   hypre_Box           *local_box;
   hypre_Box           *neighbor_box;

   int                  distance;
   int                  distance_index[3];

   int                  diff;
   int                  i, j, d;

   /*---------------------------------------------
    * Find neighboring boxes
    *---------------------------------------------*/

   neighbors = hypre_CTAlloc(hypre_BoxNeighbors, 1);
   hypre_BoxNeighborsRankLinks(neighbors) =
      hypre_CTAlloc(hypre_RankLinkArray, num_local);

   new_boxes_ranks = hypre_TAlloc(int, hypre_BoxArraySize(boxes) + 1);
   num_new_boxes = 0;
   new_boxes_ranks[num_new_boxes] = -1;
   hypre_ForBoxI(i, boxes)
      {
         for (j = 0; j < num_local; j++)
         {
            if (i != local_ranks[j])
            {
               local_box = hypre_BoxArrayBox(boxes, local_ranks[j]);
               neighbor_box = hypre_BoxArrayBox(boxes, i);

               /* compute distance info */
               distance = 0;
               for (d = 0; d < 3; d++)
               {
                  distance_index[d] = 0;

                  diff = hypre_BoxIMinD(neighbor_box, d) -
                     hypre_BoxIMaxD(local_box, d);
                  if (diff > 0)
                  {
                     distance_index[d] = 1;
                     distance = hypre_max(distance, diff);
                  }

                  diff = hypre_BoxIMinD(local_box, d) -
                     hypre_BoxIMaxD(neighbor_box, d);
                  if (diff > 0)
                  {
                     distance_index[d] = -1;
                     distance = hypre_max(distance, diff);
                  }
               }

               /* create new rank_link */
               if (distance <= max_distance)
               {
                  new_boxes_ranks[num_new_boxes] = i;

                  rank_link = hypre_RankLinkCreate(num_new_boxes);
                  hypre_RankLinkNext(rank_link) =
                     hypre_BoxNeighborsRankLink(neighbors, j,
                                                distance_index[0],
                                                distance_index[1],
                                                distance_index[2]);
                  hypre_BoxNeighborsRankLink(neighbors, j,
                                             distance_index[0],
                                             distance_index[1],
                                             distance_index[2]) = rank_link;
               }
            }
         }

         if (new_boxes_ranks[num_new_boxes] > -1)
         {
            num_new_boxes++;
            new_boxes_ranks[num_new_boxes] = -1;
         }
      }

   /*---------------------------------------------
    * Create new_boxes and new_processes
    *---------------------------------------------*/

   new_boxes_ranks = hypre_TReAlloc(new_boxes_ranks, int, num_new_boxes);
   new_boxes = hypre_BoxArrayCreate(num_new_boxes);
   for (i = 0; i < num_new_boxes; i++)
   {
      neighbor_box = hypre_BoxArrayBox(boxes, new_boxes_ranks[i]);
      hypre_CopyBox(neighbor_box, hypre_BoxArrayBox(new_boxes, i));
      new_boxes_ranks[i] = processes[new_boxes_ranks[i]];
   }
   new_processes = new_boxes_ranks;

   /*---------------------------------------------
    * Create neighbors
    *---------------------------------------------*/

   hypre_BoxNeighborsNumLocal(neighbors)    = num_local;
   hypre_BoxNeighborsBoxes(neighbors)       = new_boxes;
   hypre_BoxNeighborsProcesses(neighbors)   = new_processes;
   hypre_BoxNeighborsMaxDistance(neighbors) = max_distance;
   
   return neighbors;
}

/*--------------------------------------------------------------------------
 * hypre_BoxNeighborsDestroy:
 *--------------------------------------------------------------------------*/

int
hypre_BoxNeighborsDestroy( hypre_BoxNeighbors  *neighbors )
{
   hypre_RankLink  *rank_link;
   hypre_RankLink  *next_rank_link;

   int              b, i, j, k;

   int              ierr = 0;

   if (neighbors)
   {
      for (b = 0; b < hypre_BoxNeighborsNumLocal(neighbors); b++)
      {
         for (k = -1; k <= 1; k++)
         {
            for (j = -1; j <= 1; j++)
            {
               for (i = -1; i <= 1; i++)
               {
                  rank_link =
                     hypre_BoxNeighborsRankLink(neighbors, b, i, j, k);
                  while (rank_link)
                  {
                     next_rank_link = hypre_RankLinkNext(rank_link);
                     hypre_RankLinkDestroy(rank_link);
                     rank_link = next_rank_link;
                  }
               }
            }
         }
      }
      hypre_BoxArrayDestroy(hypre_BoxNeighborsBoxes(neighbors));
      hypre_TFree(hypre_BoxNeighborsProcesses(neighbors));
      hypre_TFree(hypre_BoxNeighborsRankLinks(neighbors));
      hypre_TFree(neighbors);
   }

   return ierr;
}
