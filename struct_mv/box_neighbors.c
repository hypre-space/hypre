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

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
int        my_rank;
hypre_Box *box;
static int debug_count = 0;
#endif

/*--------------------------------------------------------------------------
 * hypre_RankLinkCreate:
 *--------------------------------------------------------------------------*/

int
hypre_RankLinkCreate( int              rank,
                      hypre_RankLink **rank_link_ptr)
{
   hypre_RankLink  *rank_link;

   rank_link = hypre_TAlloc(hypre_RankLink, 1);

   hypre_RankLinkRank(rank_link) = rank;
   hypre_RankLinkNext(rank_link) = NULL;

   *rank_link_ptr = rank_link;

   return 0;
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
 *--------------------------------------------------------------------------*/

int
hypre_BoxNeighborsCreate( hypre_BoxArray      *boxes,
                          int                 *procs,
                          int                 *ids,
                          int                  first_local,
                          int                  num_local,
                          int                  num_periodic,
                          hypre_BoxNeighbors **neighbors_ptr )
{
   hypre_BoxNeighbors  *neighbors;

   neighbors = hypre_CTAlloc(hypre_BoxNeighbors, 1);
   hypre_BoxNeighborsRankLinks(neighbors) =
      hypre_CTAlloc(hypre_RankLinkArray, num_local);

   hypre_BoxNeighborsBoxes(neighbors)           = boxes;
   hypre_BoxNeighborsProcs(neighbors)           = procs;
   hypre_BoxNeighborsIDs(neighbors)             = ids;
   hypre_BoxNeighborsFirstLocal(neighbors)      = first_local;
   hypre_BoxNeighborsNumLocal(neighbors)        = num_local;
   hypre_BoxNeighborsNumPeriodic(neighbors)     = num_periodic;
   
   *neighbors_ptr = neighbors;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_BoxNeighborsAssemble:
 *
 * Finds boxes that are "near" the local boxes,
 * where near is defined by `max_distance'.
 *
 * Note: A box is not a neighbor of itself, but it will appear
 * in the `boxes' BoxArray.
 *
 * Note: The box ids remain in increasing order, and the box procs
 * remain in non-decreasing order.
 *
 * Note: All boxes on my processor remain in the neighborhood.  However,
 * they may not be a neighbor of any local box.
 *
 *--------------------------------------------------------------------------*/

int
hypre_BoxNeighborsAssemble( hypre_BoxNeighbors *neighbors,
                            int                 max_distance,
                            int                 prune )
{
   hypre_BoxArray      *boxes;
   int                 *procs;
   int                 *ids;
   int                  first_local;
   int                  num_local;
   int                  num_periodic;

   int                  keep_box;
   int                  num_boxes;

   hypre_RankLink      *rank_link;

   hypre_Box           *local_box;
   hypre_Box           *neighbor_box;

   int                  distance;
   int                  distance_index[3];

   int                  diff;
   int                  i, j, d, ilocal, inew;

   int                  ierr = 0;

   /*---------------------------------------------
    * Find neighboring boxes
    *---------------------------------------------*/

   boxes           = hypre_BoxNeighborsBoxes(neighbors);
   procs           = hypre_BoxNeighborsProcs(neighbors);
   ids             = hypre_BoxNeighborsIDs(neighbors);
   first_local     = hypre_BoxNeighborsFirstLocal(neighbors);
   num_local       = hypre_BoxNeighborsNumLocal(neighbors);
   num_periodic    = hypre_BoxNeighborsNumPeriodic(neighbors);

   /*---------------------------------------------
    * Find neighboring boxes
    *---------------------------------------------*/

   inew = 0;
   num_boxes = 0;
   hypre_ForBoxI(i, boxes)
      {
         keep_box = 0;
         for (j = 0; j < num_local + num_periodic; j++)
         {
            ilocal = first_local + j;
            if (i != ilocal)
            {
               local_box = hypre_BoxArrayBox(boxes, ilocal);
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
                  keep_box = 1;

                  if (j < num_local)
                  {
                     hypre_RankLinkCreate(num_boxes, &rank_link);
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
            else
            {
               keep_box = 1;
            }
         }

         if (prune)
         {
            /* use procs array to store which boxes to keep */
            if (keep_box)
            {
               procs[i] = -procs[i];
               if (inew < i)
               {
                  procs[inew] = i;
               }
               inew = i + 1;
               
               num_boxes++;
            }
         }
         else
         {
            /* keep all of the boxes */
            num_boxes++;
         }
      }

   if (prune)
   {
      i = 0;
      for (inew = 0; inew < num_boxes; inew++)
      {
         if (procs[i] > 0)
         {
            i = procs[i];
         }
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i),
                       hypre_BoxArrayBox(boxes, inew));
         procs[inew] = -procs[i];
         ids[inew]   = ids[i];
         if (i == first_local)
         {
            first_local = inew;
         }

         i++;
      }
   }

   hypre_BoxArraySetSize(boxes, num_boxes);
   hypre_BoxNeighborsFirstLocal(neighbors) = first_local;

#if DEBUG
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   sprintf(filename, "zneighbors.%05d", my_rank);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "\n\n%d\n\n", debug_count++);
   fprintf(file, "num_boxes = %d\n", num_boxes);
   for (i = 0; i < num_boxes; i++)
   {
      box = hypre_BoxArrayBox(boxes, i);
      fprintf(file, "(%d,%d,%d) X (%d,%d,%d) ; (%d,%d); %d\n",
              hypre_BoxIMinX(box),hypre_BoxIMinY(box),hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box),hypre_BoxIMaxY(box),hypre_BoxIMaxZ(box),
              procs[i], ids[i], hypre_BoxVolume(box));
   }
   fprintf(file, "first_local  = %d\n", first_local);
   fprintf(file, "num_local    = %d\n", num_local);
   fprintf(file, "num_periodic = %d\n", num_periodic);
   fprintf(file, "max_distance = %d\n", max_distance);

   fprintf(file, "\n");

   fflush(file);
   fclose(file);
#endif
   
   return ierr;
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
      hypre_TFree(hypre_BoxNeighborsProcs(neighbors));
      hypre_TFree(hypre_BoxNeighborsIDs(neighbors));
      hypre_TFree(hypre_BoxNeighborsRankLinks(neighbors));
      hypre_TFree(neighbors);
   }

   return ierr;
}
