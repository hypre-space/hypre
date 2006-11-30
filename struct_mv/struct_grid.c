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
 * Member functions for hypre_StructGrid class.
 *
 *****************************************************************************/

#include "headers.h"

#define DEBUG 0

#if DEBU
char       filename[255];
FILE      *file;
int        my_rank;
#endif

/*--------------------------------------------------------------------------
 * hypre_StructGridCreate
 *--------------------------------------------------------------------------*/

int
hypre_StructGridCreate( MPI_Comm           comm,
                        int                dim,
                        hypre_StructGrid **grid_ptr)
{
   hypre_StructGrid    *grid;
   int                 i;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   hypre_StructGridComm(grid)        = comm;
   hypre_StructGridDim(grid)         = dim;
   hypre_StructGridBoxes(grid)       = hypre_BoxArrayCreate(0);
   hypre_StructGridIDs(grid)         = NULL;
   hypre_StructGridMaxDistance(grid) = 8;
   hypre_StructGridBoundingBox(grid) = NULL;
   hypre_StructGridLocalSize(grid)   = 0;
   hypre_StructGridGlobalSize(grid)  = 0;
   hypre_SetIndex(hypre_StructGridPeriodic(grid), 0, 0, 0);
   hypre_StructGridRefCount(grid)     = 1;
   hypre_StructGridBoxMan(grid)       = NULL;
   
   hypre_StructGridNumPeriods(grid)   = 1;
   hypre_StructGridPShifts(grid)     = NULL;
   
   hypre_StructGridGhlocalSize(grid)  = 0;
   for (i = 0; i < 6; i++)
   {
     hypre_StructGridNumGhost(grid)[i] = 1;
   }

   *grid_ptr = grid;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridRef
 *--------------------------------------------------------------------------*/

int
hypre_StructGridRef( hypre_StructGrid  *grid,
                     hypre_StructGrid **grid_ref)
{
   hypre_StructGridRefCount(grid) ++;
   *grid_ref = grid;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridDestroy( hypre_StructGrid *grid )
{

   if (grid)
   {
      hypre_StructGridRefCount(grid) --;
      if (hypre_StructGridRefCount(grid) == 0)
      {
         hypre_BoxDestroy(hypre_StructGridBoundingBox(grid));
         hypre_TFree(hypre_StructGridIDs(grid));
         hypre_BoxArrayDestroy(hypre_StructGridBoxes(grid));

         hypre_BoxManDestroy(hypre_StructGridBoxMan(grid));
         hypre_TFree( hypre_StructGridPShifts(grid));

         hypre_TFree(grid);
      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructGridSetPeriodic
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetPeriodic( hypre_StructGrid  *grid,
                             hypre_Index        periodic)
{
   hypre_CopyIndex(periodic, hypre_StructGridPeriodic(grid));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetExtents
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetExtents( hypre_StructGrid  *grid,
                            hypre_Index        ilower,
                            hypre_Index        iupper )
{
   hypre_Box   *box;

   box = hypre_BoxCreate();
   hypre_BoxSetExtents(box, ilower, iupper);
   hypre_AppendBox(box, hypre_StructGridBoxes(grid));
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoxes
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetBoxes( hypre_StructGrid *grid,
                          hypre_BoxArray   *boxes )
{

   hypre_TFree(hypre_StructGridBoxes(grid));
   hypre_StructGridBoxes(grid) = boxes;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoundingBox
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetBoundingBox( hypre_StructGrid *grid,
                                hypre_Box   *new_bb )
{

   hypre_BoxDestroy(hypre_StructGridBoundingBox(grid));
   hypre_StructGridBoundingBox(grid) = hypre_BoxDuplicate(new_bb);
   
    return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_StructGridSetIDs
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetIDs( hypre_StructGrid *grid,
                          int   *ids )
{
   int ierr = 0;

   hypre_TFree(hypre_StructGridIDs(grid));
   hypre_StructGridIDs(grid) = ids;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoxManager
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetBoxManager( hypre_StructGrid *grid,
                               hypre_BoxManager *boxman )
{

   hypre_TFree(hypre_StructGridBoxMan(grid));
   hypre_StructGridBoxMan(grid) = boxman;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetMaxDistance
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetMaxDistance( hypre_StructGrid *grid,
                                int dist )
{

   hypre_StructGridMaxDistance(grid) = dist;

   return hypre_error_flag;
}



/*--------------------------------------------------------------------------
 * New - hypre_StructGridAssemble
 * AHB 9/06
 * New assemble routine that uses the BoxManager structure 
 *
 *   Notes:
 *   1. No longer need a different assemble for the assumed partition case
     2. if this is called from StructCoarsen, then the Box Manager has
        already been created, and ids have been set
 *
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridAssemble( hypre_StructGrid *grid )
{

   int d, k, p, i;
   
   int is_boxman;
   int size, ghostsize;
   int num_local_boxes;
   int myid, num_procs;
   int global_size;
   int max_nentries;
   int info_size;
   int num_periods;
   
   int *ids = NULL;
   int px, py, pz;
   int i_periodic, j_periodic, k_periodic;
 

   int  sendbuf6[6], recvbuf6[6];
         
   hypre_Box           *box;
   hypre_Box  *ghostbox;
   hypre_Box  *grow_box;
   hypre_Box  *periodic_box;
   hypre_Box  *result_box;
  
   hypre_Index min_index, max_index;
   hypre_Index *pshifts;

   hypre_IndexRef pshift;

   void *entry_info = NULL;

   /*  initialize info from the grid */
   MPI_Comm             comm         = hypre_StructGridComm(grid);
   int                  dim          = hypre_StructGridDim(grid);
   hypre_BoxArray      *local_boxes  = hypre_StructGridBoxes(grid);
   int                  max_distance = hypre_StructGridMaxDistance(grid);
   hypre_Box           *bounding_box = hypre_StructGridBoundingBox(grid);
   hypre_IndexRef       periodic     = hypre_StructGridPeriodic(grid);
   hypre_BoxManager    *boxman       = hypre_StructGridBoxMan(grid); 
   int                  *numghost    = hypre_StructGridNumGhost(grid);

  
   /* other initializations */
   num_local_boxes = hypre_BoxArraySize(local_boxes);

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);


   /* has the box manager been created? */
   if (boxman == NULL)
   {
      is_boxman = 0;
   }
   else 
   {
      is_boxman = 1;
   }
   
   /* are the ids known? (these may have been set in coarsen)  - if not we need
      to set them */
   if (hypre_StructGridIDs(grid) == NULL)
   {
      ids = hypre_CTAlloc(int, num_local_boxes);
      for (i=0; i< num_local_boxes; i++)
      {
         ids[i] = i;
      }
      hypre_StructGridIDs(grid) = ids;
   }
   else
   {
      ids = hypre_StructGridIDs(grid);
   }
   

   /******** calculate the periodicity information ****************/

     /*first determine the  periodic info */
   px = hypre_IndexX(periodic);
   py = hypre_IndexY(periodic);
   pz = hypre_IndexZ(periodic);
   
   /* how many periodic shifts? */
   i_periodic = px ? 1 : 0;
   j_periodic = py ? 1 : 0;
   k_periodic = pz ? 1 : 0;
   
   num_periods = (1+2*i_periodic) * (1+2*j_periodic) * (1+2*k_periodic);
   
   /* determine the shifting needed for periodic boxes */  
   pshifts = hypre_CTAlloc(hypre_Index, num_periods);
   pshift = pshifts[0];
   hypre_ClearIndex(pshift);
   if (num_periods > 1)
   {
      int  ip, jp, kp;
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
                  p++;
               }
            }
         }
      }
   }

   hypre_StructGridNumPeriods(grid) = num_periods;
   hypre_StructGridPShifts(grid) = pshifts;
   

   /********calculate local size and the ghost size **************/
   size = 0;
   ghostsize = 0;
   ghostbox = hypre_BoxCreate();

   hypre_ForBoxI(i, local_boxes)
   {
      box = hypre_BoxArrayBox(local_boxes, i);
      size +=  hypre_BoxVolume(box);
  
      hypre_CopyBox(box, ghostbox);
      hypre_BoxExpand(ghostbox, numghost);    
      ghostsize += hypre_BoxVolume(ghostbox); 
   }
   
   hypre_StructGridLocalSize(grid) = size;
   hypre_StructGridGhlocalSize(grid) = ghostsize;
   hypre_BoxDestroy(ghostbox);


   /* if the box manager has been created then we don't need to do the
    * following (because it was done through the coarsening routine) */
   if (!is_boxman)
   {
    
      /*************** set the global size *****************/

      MPI_Allreduce(&size, &global_size, 1, MPI_INT, MPI_SUM, comm);
      hypre_StructGridGlobalSize(grid) = global_size; /* this int
                                                       * could
                                                       * overflow!*/
      
  
      /*************** set bounding box ***********/

      bounding_box = hypre_BoxCreate();

      if (num_local_boxes) 
      {
         /* initialize min and max index*/
         box = hypre_BoxArrayBox(local_boxes, 0);
         for (d = 0; d < dim; d++)
         {
            hypre_IndexD(min_index, d) =  hypre_BoxIMinD(box, d);
            hypre_IndexD(max_index, d) =  hypre_BoxIMaxD(box, d);
         }

         hypre_ForBoxI(i, local_boxes)
         {
            box = hypre_BoxArrayBox(local_boxes, i);


            /* find min and max box extents */  
            for (d = 0; d < dim; d++)
            {
               hypre_IndexD(min_index, d) = hypre_min( hypre_IndexD(min_index, d), 
                                                       hypre_BoxIMinD(box, d));
               hypre_IndexD(max_index, d) = hypre_max( hypre_IndexD(max_index, d), 
                                                       hypre_BoxIMaxD(box, d));
            }
         }
         /*set bounding box (this is still based on local info only) */
         hypre_BoxSetExtents(bounding_box, min_index, max_index);

      }
      else /* no boxes owned*/
      {
         /* initialize min and max */ 
         for (d = 0; d < dim; d++)
         {
            hypre_BoxIMinD(bounding_box, d) = pow(2,30) ; 
            hypre_BoxIMaxD(bounding_box, d) = -pow(2,30);
         }
      }
      /* set the extra dimensions of the bounding box to zero */
      for (d = dim; d < 3; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = 0;
         hypre_BoxIMaxD(bounding_box, d) = 0;
      }

      /* communication needed for the bounding box */
      /* pack buffer */
      for (d = 0; d < 3; d++) 
      {
         sendbuf6[d] = hypre_BoxIMinD(bounding_box, d);
         sendbuf6[d+3] = -hypre_BoxIMaxD(bounding_box, d);
      }
      MPI_Allreduce(sendbuf6, recvbuf6, 6, MPI_INT, MPI_MIN, comm);
      /* unpack buffer */
      for (d = 0; d < 3; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
         hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d+3];
      }

      hypre_StructGridBoundingBox(grid) = bounding_box; 

      /***************create a box manager *****************/
      max_nentries =  num_local_boxes + 20;
      info_size = 0; /* we don't need an info object */
      hypre_BoxManCreate(max_nentries, info_size, dim, bounding_box, 
                         comm, &boxman);
      
      /********populate the box manager with my local boxes and 
               gather neighbor information                      ******/

      grow_box = hypre_BoxCreate();
      result_box = hypre_BoxCreate();
      periodic_box = hypre_BoxCreate();

       
      /* now loop through each local box */
      hypre_ForBoxI(i, local_boxes)
      {
         box = hypre_BoxArrayBox(local_boxes, i);
         /* add entry for each local box (the id is the boxnum, and
          should be sequential */         
         hypre_BoxManAddEntry( boxman, hypre_BoxIMin(box),
                               hypre_BoxIMax(box), myid, i,
                               entry_info );
 
       
         /* now expand box by max_distance or larger and gather entries */
         hypre_CopyBox(box ,grow_box);     
         hypre_BoxExpandConstant(grow_box, max_distance);
         hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(grow_box), 
                                   hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k=1; k < num_periods; k++) /* k=0 is original box */
         {
            hypre_CopyBox(grow_box, periodic_box);
            pshift = pshifts[k];
            hypre_BoxShiftPos(periodic_box, pshift);

            /* see if the shifted box intersects the domain */  
            hypre_IntersectBoxes(periodic_box, bounding_box, result_box);  
            /* if so, call gather entries */
            if (hypre_BoxVolume(result_box) > 0)
            {
               hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(periodic_box), 
                                         hypre_BoxIMax(periodic_box));
            }
         }
      }/* end of for each local box */
      
      hypre_BoxDestroy(periodic_box);
      hypre_BoxDestroy(grow_box);
      hypre_BoxDestroy(result_box);

      
   } /* end of if (!is_boxman) */
   else if (max_distance == 0) /* boxman was created (by coarsen), but we need
                                  to collect additional neighbor info */
   {

      /* pick a new max distance and set in grid*/  
      max_distance = 8;
      hypre_StructGridMaxDistance(grid) = max_distance;

      grow_box = hypre_BoxCreate();
      result_box = hypre_BoxCreate();
      periodic_box = hypre_BoxCreate();

      /* now loop through each local box */
      hypre_ForBoxI(i, local_boxes)
      {
         box = hypre_BoxArrayBox(local_boxes, i);
       
         /* now expand box by max_distance or larger and gather entries */
         hypre_CopyBox(box ,grow_box);     
         hypre_BoxExpandConstant(grow_box, max_distance);
         hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(grow_box), 
                                   hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k=1; k < num_periods; k++) /* k=0 is original box */
         {
            hypre_CopyBox(grow_box, periodic_box);
            pshift = pshifts[k];
            hypre_BoxShiftPos(periodic_box, pshift);

            /* see if the shifted box intersects the domain */  
            hypre_IntersectBoxes(periodic_box, bounding_box, result_box);  
            /* if so, call gather entries */
            if (hypre_BoxVolume(result_box) > 0)
            {
               hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(periodic_box), 
                                         hypre_BoxIMax(periodic_box));
            }
         }
      }/* end of for each local box */
      
      hypre_BoxDestroy(periodic_box);
      hypre_BoxDestroy(grow_box);
      hypre_BoxDestroy(result_box);

   }
   
   /***************Assemble the box manager *****************/
   
   hypre_BoxManAssemble(boxman);
   
   hypre_StructGridBoxMan(grid) = boxman;
   
   return hypre_error_flag;
}




/*--------------------------------------------------------------------------
 * hypre_GatherAllBoxes
 *--------------------------------------------------------------------------*/

int
hypre_GatherAllBoxes(MPI_Comm         comm,
                     hypre_BoxArray  *boxes,
                     hypre_BoxArray **all_boxes_ptr,
                     int            **all_procs_ptr,
                     int             *first_local_ptr)
{
   hypre_BoxArray    *all_boxes;
   int               *all_procs;
   int                first_local;
   int                all_boxes_size;

   hypre_Box         *box;
   hypre_Index        imin;
   hypre_Index        imax;
                     
   int                num_all_procs, my_rank;
                     
   int               *sendbuf;
   int                sendcount;
   int               *recvbuf;
   int               *recvcounts;
   int               *displs;
   int                recvbuf_size;
                     
   int                i, p, b, d;
   int                ierr = 0;

   /*-----------------------------------------------------
    * Accumulate the box info
    *-----------------------------------------------------*/
   
   MPI_Comm_size(comm, &num_all_procs);
   MPI_Comm_rank(comm, &my_rank);

   /* compute recvcounts and displs */
   sendcount = 7*hypre_BoxArraySize(boxes);
   recvcounts = hypre_SharedTAlloc(int, num_all_procs);
   displs = hypre_TAlloc(int, num_all_procs);
   MPI_Allgather(&sendcount, 1, MPI_INT,
                 recvcounts, 1, MPI_INT, comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (p = 1; p < num_all_procs; p++)
   {
      displs[p] = displs[p-1] + recvcounts[p-1];
      recvbuf_size += recvcounts[p];
   }

   /* allocate sendbuf and recvbuf */
   sendbuf = hypre_TAlloc(int, sendcount);
   recvbuf = hypre_SharedTAlloc(int, recvbuf_size);

   /* put local box extents and process number into sendbuf */
   i = 0;
   for (b = 0; b < hypre_BoxArraySize(boxes); b++)
   {
      sendbuf[i++] = my_rank;

      box = hypre_BoxArrayBox(boxes, b);
      for (d = 0; d < 3; d++)
      {
         sendbuf[i++] = hypre_BoxIMinD(box, d);
         sendbuf[i++] = hypre_BoxIMaxD(box, d);
      }
   }

   /* get global grid info */
   MPI_Allgatherv(sendbuf, sendcount, MPI_INT,
                  recvbuf, recvcounts, displs, MPI_INT, comm);

   /* sort recvbuf by process rank? */

   /*-----------------------------------------------------
    * Create all_boxes, etc.
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / 7;
   all_boxes   = hypre_BoxArrayCreate(all_boxes_size);
   all_procs   = hypre_TAlloc(int, all_boxes_size);
   first_local = -1;
   i = 0;
   b = 0;
   box = hypre_BoxCreate();
   while (i < recvbuf_size)
   {
      all_procs[b] = recvbuf[i++];
      for (d = 0; d < 3; d++)
      {
         hypre_IndexD(imin, d) = recvbuf[i++];
         hypre_IndexD(imax, d) = recvbuf[i++];
      }
      hypre_BoxSetExtents(box, imin, imax);
      hypre_CopyBox(box, hypre_BoxArrayBox(all_boxes, b));

      if ((first_local < 0) && (all_procs[b] == my_rank))
      {
         first_local = b;
      }

      b++;
   }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------
    * Return
    *-----------------------------------------------------*/

   hypre_TFree(sendbuf);
   hypre_SharedTFree(recvbuf);
   hypre_SharedTFree(recvcounts);
   hypre_TFree(displs);

   *all_boxes_ptr   = all_boxes;
   *all_procs_ptr   = all_procs;
   *first_local_ptr = first_local;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ComputeBoxnums
 *
 * It is assumed that, for any process number in 'procs', all of that
 * processes local boxes appear in the 'boxes' array.
 *
 * It is assumed that the boxes in 'boxes' are ordered by associated
 * process number then by their local ordering on that process.
 *
 *--------------------------------------------------------------------------*/

int
hypre_ComputeBoxnums(hypre_BoxArray *boxes,
                     int            *procs,
                     int           **boxnums_ptr)
{

   int               *boxnums;
   int                num_boxes;
   int                p, b, boxnum;

   /*-----------------------------------------------------
    *-----------------------------------------------------*/

   num_boxes = hypre_BoxArraySize(boxes);
   boxnums = hypre_TAlloc(int, num_boxes);

   p = -1;
   for(b = 0; b < num_boxes; b++)
   {
      /* start boxnum count at zero for each new process */
      if (procs[b] != p)
      {
         boxnum = 0;
         p = procs[b];
      }
      boxnums[b] = boxnum;
      boxnum++;
   }

   *boxnums_ptr = boxnums;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridPrint
 *--------------------------------------------------------------------------*/
 
int
hypre_StructGridPrint( FILE             *file,
                       hypre_StructGrid *grid )
{

   hypre_BoxArray  *boxes;
   hypre_Box       *box;

   int              i;

   fprintf(file, "%d\n", hypre_StructGridDim(grid));

   boxes = hypre_StructGridBoxes(grid);
   fprintf(file, "%d\n", hypre_BoxArraySize(boxes));
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         fprintf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
                 i,
                 hypre_BoxIMinX(box),
                 hypre_BoxIMinY(box),
                 hypre_BoxIMinZ(box),
                 hypre_BoxIMaxX(box),
                 hypre_BoxIMaxY(box),
                 hypre_BoxIMaxZ(box));
      }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridRead
 *--------------------------------------------------------------------------*/
 
int
hypre_StructGridRead( MPI_Comm           comm,
                      FILE              *file,
                      hypre_StructGrid **grid_ptr )
{

   hypre_StructGrid *grid;

   hypre_Index       ilower;
   hypre_Index       iupper;

   int               dim;
   int               num_boxes;
               
   int               i, idummy;

   fscanf(file, "%d\n", &dim);
   hypre_StructGridCreate(comm, dim, &grid);

   fscanf(file, "%d\n", &num_boxes);
   for (i = 0; i < num_boxes; i++)
   {
      fscanf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
             &idummy,
             &hypre_IndexX(ilower),
             &hypre_IndexY(ilower),
             &hypre_IndexZ(ilower),
             &hypre_IndexX(iupper),
             &hypre_IndexY(iupper),
             &hypre_IndexZ(iupper));

      hypre_StructGridSetExtents(grid, ilower, iupper);
   }

   hypre_StructGridAssemble(grid);

   *grid_ptr = grid;

   return hypre_error_flag;
}

/*------------------------------------------------------------------------------
 * GEC0902  hypre_StructGridSetNumGhost
 *
 * the purpose is to set num ghost in the structure grid. It is identical
 * to the function that is used in the structure vector entity.
 *-----------------------------------------------------------------------------*/

int
hypre_StructGridSetNumGhost( hypre_StructGrid *grid, int  *num_ghost )
{
  int  i;
  
  for (i = 0; i < 6; i++)
  {
    hypre_StructGridNumGhost(grid)[i] = num_ghost[i];
  }

  return hypre_error_flag;
}
