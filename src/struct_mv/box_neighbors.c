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
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


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
                      int              prank,
                      hypre_RankLink **rank_link_ptr)
{
   hypre_RankLink  *rank_link;

   rank_link = hypre_TAlloc(hypre_RankLink, 1);

   hypre_RankLinkRank(rank_link)  = rank;
   hypre_RankLinkPRank(rank_link) = prank;
   hypre_RankLinkNext(rank_link)  = NULL;

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
                          hypre_BoxNeighbors **neighbors_ptr )
{
   hypre_BoxNeighbors  *neighbors;
   int                 *boxnums;

   neighbors = hypre_CTAlloc(hypre_BoxNeighbors, 1);
   hypre_BoxNeighborsRankLinks(neighbors) =
      hypre_CTAlloc(hypre_RankLink *, num_local);

   hypre_BoxNeighborsBoxes(neighbors)      = boxes;
   hypre_BoxNeighborsProcs(neighbors)      = procs;
   hypre_ComputeBoxnums(boxes, procs, &boxnums);
   hypre_BoxNeighborsBoxnums(neighbors)    = boxnums;
   hypre_BoxNeighborsIDs(neighbors)        = ids;
   hypre_BoxNeighborsFirstLocal(neighbors) = first_local;
   hypre_BoxNeighborsNumLocal(neighbors)   = num_local;

   *neighbors_ptr = neighbors;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_BoxNeighborsAssemble:
 *
 * Add "periodic boxes" to the 'boxes' BoxArray, then find boxes that
 * are "near" (defined by 'max_distance') the local boxes.  The ranks
 * (in the 'boxes' array) of the nearby boxes of each local box are
 * stored in a linked list (called 'rank_links') for fast access in
 * other algorithms.
 *
 * If 'prune' is turned on, then only the folowing boxes are kept in
 * the 'boxes' array: the local boxes, the nearby boxes, and all other
 * boxes on the same processor as the nearby ones.
 *
 * NOTE: A box is not a neighbor of itself.
 *
 * NOTE: Periodic boxes are not assigned unique ids.  The generating
 * box and its associated processor/ID information can be found with
 * index 'i % num_boxes', where 'i' is the index for the periodic box.
 *
 *--------------------------------------------------------------------------*/

int
hypre_BoxNeighborsAssemble( hypre_BoxNeighbors *neighbors,
                            hypre_Index         periodic,
                            int                 max_distance,
                            int                 prune )
{
   hypre_BoxArray      *boxes;
   int                 *procs;
   int                 *boxnums;
   int                 *ids;
   int                  first_local;
   int                  num_local;
   int                  num_periods;
   hypre_Index         *pshifts;

   hypre_IndexRef       pshift;
   int                  period;
   int                  keep_box;
   int                  num_boxes;

   hypre_RankLink     **rank_links;
   hypre_RankLink      *rank_link;

   hypre_Box           *local_box;
   hypre_Box           *neighbor_box;

   int                  distance;
   int                  diff, firstproc, firstproci;
   int                  i, j, p, d, ilocal, inew, ii;

   int                  px = hypre_IndexX(periodic);
   int                  py = hypre_IndexY(periodic);
   int                  pz = hypre_IndexZ(periodic);
                        
   int                  i_periodic = px ? 1 : 0;
   int                  j_periodic = py ? 1 : 0;
   int                  k_periodic = pz ? 1 : 0;

   int                  ierr = 0;

   /*--------------------------------------------------
    * Create periodic boxes
    *--------------------------------------------------*/

   boxes     = hypre_BoxNeighborsBoxes(neighbors);
   procs     = hypre_BoxNeighborsProcs(neighbors);
   boxnums   = hypre_BoxNeighborsBoxnums(neighbors);
   ids       = hypre_BoxNeighborsIDs(neighbors);
   num_boxes = hypre_BoxArraySize(boxes);

   num_periods = (1+2*i_periodic) * (1+2*j_periodic) * (1+2*k_periodic);
   pshifts = hypre_CTAlloc(hypre_Index, num_periods);

   if( num_periods > 1 )
   {
      int  ip, jp, kp;

      hypre_BoxArraySetSize(boxes, num_periods*num_boxes);
      procs = hypre_TReAlloc(procs, int, num_periods*num_boxes);

      p = 1;
      for (ip = -i_periodic; ip <= i_periodic; ip++)
      {
         for (jp = -j_periodic; jp <= j_periodic; jp++)
         {
            for (kp = -k_periodic; kp <= k_periodic; kp++)
            {
               if( !(ip == 0 && jp == 0 && kp == 0) )
               {
                  pshift = pshifts[p];
                  hypre_SetIndex(pshift, ip*px, jp*py, kp*pz);

                  for (i = 0; i < num_boxes; i++)
                  {
                     inew = i + p*num_boxes;
                     local_box = hypre_BoxArrayBox(boxes, inew);
                     hypre_CopyBox(hypre_BoxArrayBox(boxes, i), local_box);
                     hypre_BoxShiftPos(local_box, pshift);

                     procs[inew] = procs[i];
                  }

                  p++;
               }
            }
         }
      }
   }
   
   hypre_BoxNeighborsBoxes(neighbors)      = boxes;
   hypre_BoxNeighborsProcs(neighbors)      = procs;
   hypre_CopyIndex(periodic, hypre_BoxNeighborsPeriodic(neighbors));
   hypre_BoxNeighborsNumPeriods(neighbors) = num_periods;
   hypre_BoxNeighborsPShifts(neighbors)    = pshifts;

   /*-----------------------------------------------------------------
    * Find neighboring boxes:
    *
    * Keep boxes that are nearby plus all other boxes on the same
    * processor as the nearby boxes.
    *-----------------------------------------------------------------*/

   period      = num_boxes;
   first_local = hypre_BoxNeighborsFirstLocal(neighbors);
   num_local   = hypre_BoxNeighborsNumLocal(neighbors);

   rank_links = hypre_BoxNeighborsRankLinks(neighbors);

   inew = 0;
   num_boxes = 0;
   firstproc = -1;
   for (i = 0; i < period; i++)
   {
      /* keep all boxes with same process number as nearby boxes */
      if (procs[i] > firstproc)
      {
         firstproci = i;
         firstproc  = procs[i];
         keep_box  = 0;
      }

      /* loop over box i and its periodic boxes; use ii */
      for (p = 0; p < num_periods; p++)
      {
         ii = i + p*period;

         for (j = 0; j < num_local; j++, ilocal++)
         {
            ilocal = first_local + j;

            if (ii == ilocal)
            {
               keep_box = 1;
            }
            else
            {
               local_box = hypre_BoxArrayBox(boxes, ilocal);
               neighbor_box = hypre_BoxArrayBox(boxes, ii);

               /* compute distance info */
               distance = 0;
               for (d = 0; d < 3; d++)
               {
                  diff = hypre_BoxIMinD(neighbor_box, d) -
                     hypre_BoxIMaxD(local_box, d);
                  if (diff > 0)
                  {
                     distance = hypre_max(distance, diff);
                  }

                  diff = hypre_BoxIMinD(local_box, d) -
                     hypre_BoxIMaxD(neighbor_box, d);
                  if (diff > 0)
                  {
                     distance = hypre_max(distance, diff);
                  }
               }

               /* if close enough, keep the box */
               if (distance <= max_distance)
               {
                  /* adjust for earlier boxes with same process number */
                  if (!keep_box)
                  {
                     num_boxes += (i - firstproci);
                     keep_box = 1;
                  }

                  /* create new rank_link and prepend to the list */
                  hypre_RankLinkCreate(num_boxes, p, &rank_link);
                  hypre_RankLinkNext(rank_link) = rank_links[j];
                  rank_links[j] = rank_link;
               }
            }
         }
      }

      /*-----------------------------------------------------------------
       * If pruning is on, use procs array to mark the boxes to keep
       *-----------------------------------------------------------------*/

      if (prune)
      {
         /* use procs array to store which boxes to keep */
         if (keep_box)
         {
            if (inew < firstproci)
            {
               procs[inew] = firstproci;
            }
            inew = i + 1;

            for (ii = firstproci; ii < inew; ii++)
            {
               procs[ii] = -procs[ii];
            }
            firstproci = inew;
               
            num_boxes++;
         }
      }
      else
      {
         /* keep all of the boxes */
         num_boxes++;
      }
   }

   /*-----------------------------------------------------------------
    * Prune the array of neighbor boxes
    *-----------------------------------------------------------------*/

   if (prune)
   {
      for (p = 0; p < num_periods; p++)
      {
         i = 0;
         for (inew = 0; inew < num_boxes; inew++)
         {
            if (procs[i] > 0)
            {
               i = procs[i];
            }
            hypre_CopyBox(hypre_BoxArrayBox(boxes, i + p*period),
                          hypre_BoxArrayBox(boxes, inew + p*num_boxes));
            if (p == 0)
            {
               boxnums[inew] = boxnums[i];
               ids[inew] = ids[i];
            }
            
            i++;
         }
      }

      i = 0;
      for (inew = 0; inew < num_boxes; inew++)
      {
         if (procs[i] > 0)
         {
            i = procs[i];
         }
         procs[inew] = -procs[i];
         if (i == first_local)
         {
            first_local = inew;
         }
         
         i++;
      }
      for (p = 1; p < num_periods; p++)
      {
         for (inew = 0; inew < num_boxes; inew++)
         {
            procs[inew + p*num_boxes] = procs[inew];
         }
      }
   }

   num_boxes *= num_periods;
   hypre_BoxArraySetSize(boxes, num_boxes);
   hypre_BoxNeighborsFirstLocal(neighbors) = first_local;

#if DEBUG
{
   int  b, k, n;

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
   fprintf(file, "periodic     = %d %d %d\n",
           periodic[0], periodic[1], periodic[2]);
   fprintf(file, "num_periods  = %d\n", num_periods);
   fprintf(file, "max_distance = %d\n", max_distance);
   fprintf(file, "prune        = %d\n", prune);

   fprintf(file, "rank_links:\n");
   num_boxes /= num_periods;
   for (b = 0; b < hypre_BoxNeighborsNumLocal(neighbors); b++)
   {
      for (k = -1; k <= 1; k++)
      {
         for (j = -1; j <= 1; j++)
         {
            for (i = -1; i <= 1; i++)
            {
               rank_link = hypre_BoxNeighborsRankLink(neighbors, b, i, j, k);
               while (rank_link)
               {
                  n = hypre_RankLinkRank(rank_link) +
                     hypre_RankLinkPRank(rank_link)*num_boxes;
                  fprintf(file, "(%d : %d, %d, %d) = %d\n",
                          b, i, j, k, n);
                  
                  rank_link = hypre_RankLinkNext(rank_link);
               }
            }
         }
      }
   }

   fprintf(file, "\n");

   fflush(file);
   fclose(file);
}
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

   int              b;

   int              ierr = 0;

   if (neighbors)
   {
      for (b = 0; b < hypre_BoxNeighborsNumLocal(neighbors); b++)
      {
         rank_link = hypre_BoxNeighborsRankLink(neighbors, b);
         while (rank_link)
         {
            next_rank_link = hypre_RankLinkNext(rank_link);
            hypre_RankLinkDestroy(rank_link);
            rank_link = next_rank_link;
         }
      }
      hypre_BoxArrayDestroy(hypre_BoxNeighborsBoxes(neighbors));
      hypre_TFree(hypre_BoxNeighborsProcs(neighbors));
      hypre_TFree(hypre_BoxNeighborsBoxnums(neighbors));
      hypre_TFree(hypre_BoxNeighborsIDs(neighbors));
      hypre_TFree(hypre_BoxNeighborsPShifts(neighbors));
      hypre_TFree(hypre_BoxNeighborsRankLinks(neighbors));
      hypre_TFree(neighbors);
   }

   return ierr;
}
