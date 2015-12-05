/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.24 $
 ***********************************************************************EHEADER*/

#define TIME_DEBUG 0

#if TIME_DEBUG
static HYPRE_Int s_coarsen_num = 0;
#endif


#include "headers.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
static HYPRE_Int debug_count = 0;
#endif

/*--------------------------------------------------------------------------
 * hypre_StructMapFineToCoarse
 *
 * NOTE: findex and cindex are indexes on the fine and coarse index space, and
 * do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMapFineToCoarse( hypre_Index findex,
                             hypre_Index index,
                             hypre_Index stride,
                             hypre_Index cindex )
{
   hypre_IndexX(cindex) =
      (hypre_IndexX(findex) - hypre_IndexX(index)) / hypre_IndexX(stride);
   hypre_IndexY(cindex) =
      (hypre_IndexY(findex) - hypre_IndexY(index)) / hypre_IndexY(stride);
   hypre_IndexZ(cindex) =
      (hypre_IndexZ(findex) - hypre_IndexZ(index)) / hypre_IndexZ(stride);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_StructMapCoarseToFine
 *
 * NOTE: findex and cindex are indexes on the fine and coarse index space, and
 * do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMapCoarseToFine( hypre_Index cindex,
                             hypre_Index index,
                             hypre_Index stride,
                             hypre_Index findex ) 
{
   hypre_IndexX(findex) =
      hypre_IndexX(cindex) * hypre_IndexX(stride) + hypre_IndexX(index);
   hypre_IndexY(findex) =
      hypre_IndexY(cindex) * hypre_IndexY(stride) + hypre_IndexY(index);
   hypre_IndexZ(findex) =
      hypre_IndexZ(cindex) * hypre_IndexZ(stride) + hypre_IndexZ(index);

   return 0;
}

#define hypre_StructCoarsenBox(box, index, stride) \
hypre_ProjectBox(box, index, stride);\
hypre_StructMapFineToCoarse(hypre_BoxIMin(box), index, stride,\
                            hypre_BoxIMin(box));\
hypre_StructMapFineToCoarse(hypre_BoxIMax(box), index, stride,\
                            hypre_BoxIMax(box))

/*--------------------------------------------------------------------------
 * New version of hypre_StructCoarsen that uses the BoxManager (AHB 12/06)
 *
 * This routine coarsens the grid, 'fgrid', by the coarsening factor, 'stride',
 * using the index mapping in 'hypre_StructMapFineToCoarse'.
 *  
 *  1.  A coarse grid is created with boxes that result from coarsening the fine
 *  grid boxes, bounding box, and periodicity information.
 *
 *  2. If "sufficient" neighbor information exists in the fine grid to be
 *  transferred to the coarse grid, then the coarse grid box manager can be
 *  created by simply coarsening all of the entries in the fine grid manager.
 *  ("Sufficient" is determined by checking max_distance in the fine grid.)
 *
 *  3.  Otherwise, neighbor information will be collected during the
 *  StructGridAssemble according to the choosen value of max_distance for the
 *  coarse grid.
 *
 *   4. We do not need a separate version for the assumed partition case
 *
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructCoarsen( hypre_StructGrid  *fgrid,
                     hypre_Index        index,
                     hypre_Index        stride,
                     HYPRE_Int          prune,
                     hypre_StructGrid **cgrid_ptr )
{
   hypre_StructGrid *cgrid;

   MPI_Comm          comm;
   HYPRE_Int         dim;

   hypre_BoxArray   *my_boxes;

   hypre_Index       periodic;
   hypre_Index       ilower, iupper;

   hypre_Box        *box;
   hypre_Box        *new_box;
   hypre_Box        *bounding_box;

   HYPRE_Int         i, j, myid, count;
   HYPRE_Int         info_size, max_nentries;
   HYPRE_Int         num_entries;
   HYPRE_Int        *fids, *cids;
   hypre_Index       new_dist;
   hypre_IndexRef    max_distance;
   HYPRE_Int         proc, id;
   HYPRE_Int         coarsen_factor, known;
   HYPRE_Int         num, last_proc;
#if 0
   hypre_StructAssumedPart *fap = NULL, *cap = NULL;
#endif
   hypre_BoxManager   *fboxman, *cboxman;

   hypre_BoxManEntry *entries;
   hypre_BoxManEntry  *entry;
     
   void               *entry_info = NULL;
 
#if TIME_DEBUG  
   HYPRE_Int tindex;
   char new_title[80];
   hypre_sprintf(new_title,"Coarsen.%d",s_coarsen_num);
   tindex = hypre_InitializeTiming(new_title);
   s_coarsen_num++;

   hypre_BeginTiming(tindex);
#endif

   /* get relevant information from the fine grid */
   fids = hypre_StructGridIDs(fgrid);
   fboxman = hypre_StructGridBoxMan(fgrid);
   comm  = hypre_StructGridComm(fgrid);
   dim   = hypre_StructGridDim(fgrid);
   max_distance = hypre_StructGridMaxDistance(fgrid);
   
   /* initial */
   hypre_MPI_Comm_rank(comm, &myid );

   /* create new coarse grid */
   hypre_StructGridCreate(comm, dim, &cgrid);

   /* coarsen my boxes and create the coarse grid ids (same as fgrid) */
   my_boxes = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(fgrid));
   cids = hypre_TAlloc(HYPRE_Int,  hypre_BoxArraySize(my_boxes));
   for (i = 0; i < hypre_BoxArraySize(my_boxes); i++)
   {
      box = hypre_BoxArrayBox(my_boxes, i);
      hypre_StructCoarsenBox(box, index, stride);
      cids[i] = fids[i];
   }
   
   /* prune? */
   /* zero volume boxes are needed when forming P and P^T */ 
   if (prune)
   {
      count = 0;    
      hypre_ForBoxI(i, my_boxes)
      {
         box = hypre_BoxArrayBox(my_boxes, i);
         if (hypre_BoxVolume(box))
         {
            hypre_CopyBox(box, hypre_BoxArrayBox(my_boxes, count));
            cids[count] = cids[i];
            count++;
         }
      }
      hypre_BoxArraySetSize(my_boxes, count);
   }

   /* set coarse grid boxes */
   hypre_StructGridSetBoxes(cgrid, my_boxes);

   /* set coarse grid ids */ 
   hypre_StructGridSetIDs(cgrid, cids);

   /* adjust periodicity and set for the coarse grid */
   hypre_CopyIndex(hypre_StructGridPeriodic(fgrid), periodic);
   for (i = 0; i < dim; i++)
   {
      hypre_IndexD(periodic,i) /= hypre_IndexD(stride,i);
   }
   hypre_StructGridSetPeriodic(cgrid, periodic);

   /* Check the max_distance value of the fine grid to determine whether we will
      need to re-gather information in the assemble.  If we need to re-gather,
      then the max_distance will be set to (0,0,0).  Either way, we will create
      and populate the box manager with the information from the fine grid.

      Note: if all global info is already known for a grid, the we do not need
      to re-gather regardless of the max_distance values. */

   for (i = 0; i < dim; i++)
   {
      coarsen_factor = hypre_IndexD(stride,i); 
      hypre_IndexD(new_dist, i) = hypre_IndexD(max_distance,i)/coarsen_factor;
   }
   for (i = dim; i < 3; i++)
   {
      hypre_IndexD(new_dist, i) = 2;
   }
   
   hypre_BoxManGetAllGlobalKnown (fboxman, &known );


   if ( hypre_IndexGTESize(new_dist, 2) || known) /* large enough - don't need
                                                   * to re-gather */
   {
      /* update new max distance value */  
      if (!known) /* only need to change if global info is not known */
         hypre_StructGridSetMaxDistance(cgrid, new_dist);
   }
   else  /* not large enough - set max_distance to 0 - neighbor info will be
            collected during the assemble */
   {
      hypre_ClearIndex(new_dist);
      hypre_StructGridSetMaxDistance(cgrid, new_dist);
   }

   /* update the new bounding box */
   bounding_box = hypre_BoxDuplicate(hypre_StructGridBoundingBox(fgrid));
   hypre_StructCoarsenBox(bounding_box, index, stride);
   
   hypre_StructGridSetBoundingBox(cgrid, bounding_box);
   
   /* create a box manager for the coarse grid */ 
   info_size = hypre_BoxManEntryInfoSize(fboxman);
   max_nentries =  hypre_BoxManMaxNEntries(fboxman);
   hypre_BoxManCreate(max_nentries, info_size, dim, bounding_box, 
                      comm, &cboxman);
   
   hypre_BoxDestroy(bounding_box);
   
   /* update all global known */
   hypre_BoxManSetAllGlobalKnown(cboxman, known );
   
   /* now get the entries from the fgrid box manager, coarsen, and add to the
      coarse grid box manager (note: my boxes have already been coarsened) */
   
   hypre_BoxManGetAllEntries( fboxman , &num_entries, &entries); 

   new_box = hypre_BoxCreate();
   num = 0;
   last_proc = -1;

   /* entries are sorted by (proc, id) pairs - may not have entries for all
      processors, but for each processor represented, we do have all of its
      boxes.  We will keep them sorted in the new box manager - to avoid
      re-sorting */
   for (i = 0; i < num_entries; i++)
   {
      entry = &entries[i];
      proc = hypre_BoxManEntryProc(entry);
      
      if  (proc != myid) /* not my boxes */ 
      {
         hypre_BoxManEntryGetExtents(entry, ilower, iupper);
         hypre_BoxSetExtents(new_box, ilower, iupper);
         hypre_StructCoarsenBox(new_box, index, stride);
         id =  hypre_BoxManEntryId(entry);
         /* if there is pruning we need to adjust the ids if any boxes drop out
            (we want these ids sequential - no gaps) - and zero boxes are not
            kept in the box manager */
         if (prune)
         {  
            if (proc != last_proc) 
            {
               num = 0;
               last_proc = proc;
            }
            if (hypre_BoxVolume(new_box))
            {
               
               hypre_BoxManAddEntry( cboxman, hypre_BoxIMin(new_box) ,
                                     hypre_BoxIMax(new_box), proc, num,
                                     entry_info);
               num++;
            }
         }
         else /* no pruning - just use id (note that size zero boxes will not be
                 saved in the box manager, so we will have gaps in the box
                 numbers) */
         {
            hypre_BoxManAddEntry( cboxman, hypre_BoxIMin(new_box) ,
                                  hypre_BoxIMax(new_box), proc, id,
                                  entry_info);
         }
      } 
      else /* my boxes */
           /* add my coarse grid boxes to the coarse grid box manager (have
              already been pruned if necessary) - re-number the entry ids to be
              sequential (this is the box number, really) */
      {
         if (proc != last_proc) /* just do this once (the first myid) */
         {
            hypre_ForBoxI(j, my_boxes)
            {
               box = hypre_BoxArrayBox(my_boxes, j);
               hypre_BoxManAddEntry( cboxman, hypre_BoxIMin(box),
                                     hypre_BoxIMax(box), myid, j,
                                     entry_info );
            }
            last_proc = proc;
         }
      }
   } /* loop through entries */
   
   /* these entries are sorted */
   hypre_BoxManSetIsEntriesSort(cboxman, 1 );
   
   hypre_BoxDestroy(new_box);

#if 0   
   /* if there is an assumed partition in the fg, then coarsen those boxes as
      well and add to cg */
    hypre_BoxManGetAssumedPartition ( fboxman, &fap);
    
    if (fap)
    {
       /* coarsen fap to get cap */ 

       /* set cap */  
       hypre_BoxManSetAssumedPartition (cboxman, cap);
    }
#endif

   /* assign new box manager */
   hypre_StructGridSetBoxManager(cgrid, cboxman);
      
   /* finally... assemble the new coarse grid */
   hypre_StructGridAssemble(cgrid);

   /* return the coarse grid */   
   *cgrid_ptr = cgrid;

#if TIME_DEBUG
   hypre_EndTiming(tindex);
#endif

   return hypre_error_flag;
}

#undef hypre_StructCoarsenBox



/*--------------------------------------------------------------------------
 * hypre_Merge
 *
 * Merge the integers in the sorted 'arrays'.  The routine returns the
 * (array, array_index) pairs for each entry in the merged list.  The
 * end of the pairs is indicated by the first '-1' entry in 'mergei'.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Merge( HYPRE_Int   **arrays,
             HYPRE_Int    *sizes,
             HYPRE_Int     size,
             HYPRE_Int   **mergei_ptr,
             HYPRE_Int   **mergej_ptr )
{
   HYPRE_Int ierr = 0;

   HYPRE_Int  *mergei;
   HYPRE_Int  *mergej;

   HYPRE_Int   num, i;
   HYPRE_Int   lastval;

   struct linkstruct
   {
      HYPRE_Int  i;
      HYPRE_Int  j;
      struct linkstruct  *next;

   } *list, *first, *link, *next;

   num = 0;
   for (i = 0; i < size; i++)
   {
      num += sizes[i];
   }
   mergei = hypre_TAlloc(HYPRE_Int, num+1);
   mergej = hypre_TAlloc(HYPRE_Int, num+1);

   if (num > 0)
   {
      /* Create the sorted linked list (temporarily use merge arrays) */

      num = 0;
      for (i = 0; i < size; i++)
      {
         if (sizes[i] > 0)
         {
            mergei[num] = arrays[i][0];
            mergej[num] = i;
            num++;
         }
      }

      hypre_qsort2i(mergei, mergej, 0, (num-1));

      list = hypre_TAlloc(struct linkstruct, num);
      first = list;
      link = &first[0];
      link->i = mergej[0];
      link->j = 0;
      for (i = 1; i < num; i++)
      {
         link->next = &first[i];
         link       = (link -> next);
         link->i    = mergej[i];
         link->j    = 0;
      }
      link->next = NULL;

      /* merge the arrays using the sorted linked list */

      num = 0;
      lastval = arrays[first->i][first->j] - 1;
      while (first != NULL)
      {
         /* put unique values in the merged list */
         if ( arrays[first->i][first->j] > lastval )
         {
            mergei[num] = first->i;
            mergej[num] = first->j;
            lastval = arrays[first->i][first->j];
            num++;
         }

         /* find the next value, while keeping the list sorted */
         first->j += 1;
         next = first->next;
         if ( !((first->j) < sizes[first->i]) )
         {
            /* pop 'first' from the list */
            first = first->next;
         }
         else if (next != NULL)
         {
            if ( arrays[first->i][first->j] > 
                 arrays[next ->i][next ->j] )
            {
               /* find new place in the list for 'first' */
               link = next;
               next = link->next;
               while (next != NULL)
               {
                  if ( arrays[first->i][first->j] <
                       arrays[next ->i][next ->j] )
                  {
                     break;
                  }
                  link = next;
                  next = link->next;
               }

               /* put 'first' after 'link' and reset 'first' */
               next = first;
               first = first->next;
               next->next = link->next;
               link->next = next;
            }
         }
      }
   }

   mergei[num] = -1;
   mergej[num] = -1;

   hypre_TFree(list);
   
   *mergei_ptr = mergei;
   *mergej_ptr = mergej;

   return ierr;
}

