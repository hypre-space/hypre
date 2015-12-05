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
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/


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
 * NEW hypre_BoxNeighborsCreate (for use with "no global partition" option)
 *      here we assign the box nums - we don't compute.
 *      also the ids are the same as the boxnums (no longer globally unique)    
 *      AHB 6/05     
 *--------------------------------------------------------------------------*/


int
hypre_BoxNeighborsCreateWithAP( hypre_BoxArray      *boxes,
                                int                 *procs,
                                int                 *boxnums,
                                int                  first_local,
                                int                  num_local,
                                hypre_Index         *pshifts,
                                hypre_BoxNeighbors **neighbors_ptr )
{
   hypre_BoxNeighbors  *neighbors;
   int                 *ids, i;

   neighbors = hypre_CTAlloc(hypre_BoxNeighbors, 1);
   hypre_BoxNeighborsRankLinks(neighbors) =
      hypre_CTAlloc(hypre_RankLink *, num_local);

   hypre_BoxNeighborsBoxes(neighbors)      = boxes;
   hypre_BoxNeighborsProcs(neighbors)      = procs;
 
   hypre_BoxNeighborsBoxnums(neighbors)    = boxnums;

   /*these ids are not used anymore - just make them
     the same as the boxnums - eventually get rid of these */
   ids = hypre_CTAlloc(int,  hypre_BoxArraySize(boxes));
   hypre_ForBoxI(i, boxes)
   {     
      ids [i] = boxnums[i];
   }
   hypre_BoxNeighborsIDs(neighbors)        = ids;
 

   hypre_BoxNeighborsFirstLocal(neighbors) = first_local;
   hypre_BoxNeighborsNumLocal(neighbors)   = num_local;


   hypre_BoxNeighborsPShifts(neighbors)    = pshifts;



   *neighbors_ptr = neighbors;

   return 0;
}



/*--------------------------------------------------------------------------
 * hypre_BoxNeighborsAssembleWithAP:
 *
 * NOte: Now we do not keep nearby boxes with the same proc id (AB 10/04)
 * as a "neighbor" box
 *
 * Add "periodic boxes" to the 'boxes' BoxArray, then find boxes that
 * are "near" (defined by 'max_distance') the local boxes.  The ranks
 * (in the 'boxes' array) of the nearby boxes of each local box are
 * stored in a linked list (called 'rank_links') for fast access in
 * other algorithms.
 *
 *
 * NOTE: A box is not a neighbor of itself.
 *
 * NOTE: Periodic boxes are not assigned unique ids.  The generating
 * box and its associated processor/ID information can be found with
 * index 'i % num_boxes', where 'i' is the index for the periodic box.
 *
 *--------------------------------------------------------------------------*/

int
hypre_BoxNeighborsAssembleWithAP( hypre_BoxNeighbors *neighbors,
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
   int                  keep_box;
   int                  num_boxes, num_nonperiodic_boxes;

   hypre_RankLink     **rank_links;
   hypre_RankLink      *rank_link;

   hypre_Box           *local_box;
   hypre_Box           *neighbor_box;

   int                  distance;
   int                  diff;
   int                  i, j, p, d, ilocal, inew;

   int                  px = hypre_IndexX(periodic);
   int                  py = hypre_IndexY(periodic);
   int                  pz = hypre_IndexZ(periodic);
                        
   int                  i_periodic = px ? 1 : 0;
   int                  j_periodic = py ? 1 : 0;
   int                  k_periodic = pz ? 1 : 0;

   int                  ierr = 0;

   int            *tmp_p;
   
   int             first_local_orig, check_loc;;
   


   /*--------------------------------------------------
    * Create periodic boxes
    *--------------------------------------------------*/

   /* periodic boxes are given box numbers also */


   /*prune = 0;*/   

   boxes     = hypre_BoxNeighborsBoxes(neighbors);
   procs     = hypre_BoxNeighborsProcs(neighbors);
   boxnums   = hypre_BoxNeighborsBoxnums(neighbors); /* boxnums and ids are 
                                                        the same values */
   ids       = hypre_BoxNeighborsIDs(neighbors);
   num_boxes = hypre_BoxArraySize(boxes);

   num_nonperiodic_boxes = num_boxes;
   

   num_periods = (1+2*i_periodic) * (1+2*j_periodic) * (1+2*k_periodic);
   pshifts = hypre_BoxNeighborsPShifts(neighbors);
   
   
   tmp_p = hypre_CTAlloc(int, num_boxes);

   
   if( num_periods > 1 )
   {

      hypre_BoxArraySetSize(boxes, num_periods*num_boxes);
      procs = hypre_TReAlloc(procs, int, num_periods*num_boxes);
      /* 11/19 */
      boxnums = hypre_TReAlloc(boxnums, int, num_periods*num_boxes);
      ids =  hypre_TReAlloc(ids, int, num_periods*num_boxes); 
      tmp_p =  hypre_TReAlloc(tmp_p, int, num_periods*num_boxes);      

      /* spshifts are already set */
      for (p = 1; p < num_periods; p++) 
      {
          pshift = pshifts[p];
          for (i = 0; i < num_boxes; i++)
          {
             inew = i + p*num_boxes;
             local_box = hypre_BoxArrayBox(boxes, inew);
             hypre_CopyBox(hypre_BoxArrayBox(boxes, i), local_box);
             hypre_BoxShiftPos(local_box, pshift);
             procs[inew] = procs[i];
             /* 11/19 */
             /* give periodic boxes a boxnum and id also */
             boxnums[inew] = boxnums[i];
             ids[inew] = ids[i];
             tmp_p[inew] = p;
             

          }
      }
      

   } /* end of make periodic boxes */
   

   hypre_BoxNeighborsBoxnums(neighbors) = boxnums;
   hypre_BoxNeighborsIDs(neighbors)= ids;

   hypre_BoxNeighborsBoxes(neighbors)      = boxes;
   hypre_BoxNeighborsProcs(neighbors)      = procs;
   hypre_CopyIndex(periodic, hypre_BoxNeighborsPeriodic(neighbors));
   hypre_BoxNeighborsNumPeriods(neighbors) = num_periods;


   /*-----------------------------------------------------------------
    * Find neighboring boxes:
    *
    * Keep boxes that are nearby. - Don't treat periodic differently!
    *-----------------------------------------------------------------*/

    
   rank_links = hypre_BoxNeighborsRankLinks(neighbors);
   first_local = hypre_BoxNeighborsFirstLocal(neighbors);
   num_local   = hypre_BoxNeighborsNumLocal(neighbors);
   inew = 0;
   num_boxes = 0;
   
   first_local_orig = first_local;
   
   
   
   /* loop over all potential neighbor boxes */          
   hypre_ForBoxI(i, boxes)
   {
      keep_box = 0;
      neighbor_box = hypre_BoxArrayBox(boxes, i);
    
      /*moved from above 1/11*/
      if (i < num_nonperiodic_boxes)
      {
         tmp_p[i] = 0;
      }
      
      /* 1/19/05 - this added because we are copying boxes
         in the loop - so local boxes coupld be separated */
      check_loc = 0;
      
      if ((first_local < first_local_orig) &&  (i < (first_local_orig + num_local)))
      {
         check_loc = 1;
      }
          
    
      /*check each of my local boxes against this box*/
      for (j = 0; j < num_local; j++)
       {
          ilocal = first_local + j;

          /* 1/19/05 - recalculate if the local boxes are separated*/  
          if (check_loc  && ilocal >= num_boxes)          
          {
             ilocal = first_local_orig + j;
          }
          
  
          if (i == ilocal)  /* if neighbor box and local box are the same box */
          {
               keep_box = 1;
          }
          else
          {
             local_box = hypre_BoxArrayBox(boxes, ilocal);
             
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
                keep_box = 1;
                  
                p = tmp_p[i]; /* periodic number for neighbor box */
                
                
                /* create new rank_link and prepend to the list */
                hypre_RankLinkCreate(num_boxes, p, &rank_link);
                hypre_RankLinkNext(rank_link) = rank_links[j];
                rank_links[j] = rank_link;


             }
          } /*end of "if neighbor and my box are the same box "*/
       }  /* end of local boxes loop */ 
       
       if (prune)
       {
        
          if (keep_box)
          {
           

  
             /* copy now - avoid an extra loop */  
            hypre_CopyBox(hypre_BoxArrayBox(boxes, i),
                           hypre_BoxArrayBox(boxes, num_boxes));
            boxnums[num_boxes] = boxnums[i];
            ids[num_boxes] = ids[i];
            procs[num_boxes] = procs[i];

            if (i == first_local)
            {
               first_local = num_boxes;
            }


            num_boxes++;

          }
       }
       else
       {
          /* keep all of the boxes */
          num_boxes++;
       }

    }/*end of loop through each box */
    
    /*-----------------------------------------------------------------
     * Prune the array of neighbor boxes
     *-----------------------------------------------------------------*/



  hypre_BoxArraySetSize(boxes, num_boxes);
  hypre_BoxNeighborsFirstLocal(neighbors) = first_local;


  hypre_TFree(tmp_p);

   
   return ierr;
}



