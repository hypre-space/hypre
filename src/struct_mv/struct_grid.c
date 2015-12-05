/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.23 $
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
HYPRE_Int  my_rank;
#endif

static HYPRE_Int time_index = 0;

/*--------------------------------------------------------------------------
 * hypre_StructGridCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructGridCreate( MPI_Comm           comm,
                        HYPRE_Int          dim,
                        hypre_StructGrid **grid_ptr)
{
   hypre_StructGrid    *grid;
   HYPRE_Int           i;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   hypre_StructGridComm(grid)        = comm;
   hypre_StructGridDim(grid)         = dim;
   hypre_StructGridBoxes(grid)       = hypre_BoxArrayCreate(0);
   hypre_StructGridIDs(grid)         = NULL;

   hypre_SetIndex(hypre_StructGridMaxDistance(grid),8, 8, 8);

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

HYPRE_Int
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

HYPRE_Int 
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

HYPRE_Int 
hypre_StructGridSetPeriodic( hypre_StructGrid  *grid,
                             hypre_Index        periodic)
{
   hypre_CopyIndex(periodic, hypre_StructGridPeriodic(grid));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetExtents
 *--------------------------------------------------------------------------*/

HYPRE_Int 
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

HYPRE_Int 
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

HYPRE_Int 
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

HYPRE_Int 
hypre_StructGridSetIDs( hypre_StructGrid *grid,
                          HYPRE_Int   *ids )
{
   HYPRE_Int ierr = 0;

   hypre_TFree(hypre_StructGridIDs(grid));
   hypre_StructGridIDs(grid) = ids;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoxManager
 *--------------------------------------------------------------------------*/

HYPRE_Int 
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

HYPRE_Int 
hypre_StructGridSetMaxDistance( hypre_StructGrid *grid,
                                hypre_Index dist )
{

   hypre_CopyIndex(dist, hypre_StructGridMaxDistance(grid));

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

HYPRE_Int 
hypre_StructGridAssemble( hypre_StructGrid *grid )
{

   HYPRE_Int d, k, p, i;
   
   HYPRE_Int is_boxman;
   HYPRE_Int size, ghostsize;
   HYPRE_Int num_local_boxes;
   HYPRE_Int myid, num_procs;
   HYPRE_Int global_size;
   HYPRE_Int max_nentries;
   HYPRE_Int info_size;
   HYPRE_Int num_periods;
   
   HYPRE_Int *ids = NULL;
   HYPRE_Int px, py, pz;
   HYPRE_Int i_periodic, j_periodic, k_periodic;
 

   HYPRE_Int  sendbuf6[6], recvbuf6[6];
         
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
   HYPRE_Int            dim          = hypre_StructGridDim(grid);
   hypre_BoxArray      *local_boxes  = hypre_StructGridBoxes(grid);
   hypre_IndexRef       max_distance = hypre_StructGridMaxDistance(grid);
   hypre_Box           *bounding_box = hypre_StructGridBoundingBox(grid);
   hypre_IndexRef       periodic     = hypre_StructGridPeriodic(grid);
   hypre_BoxManager    *boxman       = hypre_StructGridBoxMan(grid); 
   HYPRE_Int            *numghost    = hypre_StructGridNumGhost(grid);

   
   if (!time_index)
      time_index = hypre_InitializeTiming("StructGridAssemble");

   hypre_BeginTiming(time_index);


   /* other initializations */
   num_local_boxes = hypre_BoxArraySize(local_boxes);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);


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
      ids = hypre_CTAlloc(HYPRE_Int, num_local_boxes);
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
      HYPRE_Int  ip, jp, kp;
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

      hypre_MPI_Allreduce(&size, &global_size, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      hypre_StructGridGlobalSize(grid) = global_size; /* TO DO: this HYPRE_Int could
                                                       * overflow! (used for calc flops) */
      
  
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
      hypre_MPI_Allreduce(sendbuf6, recvbuf6, 6, HYPRE_MPI_INT, hypre_MPI_MIN, comm);
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
         hypre_BoxExpandConstantDim(grow_box, max_distance);
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
   else if ( hypre_IndexZero(max_distance) ) /* boxman was created (by coarsen), but we need
                                  to collect additional neighbor info */
   {

      /* pick a new max distance and set in grid*/  
      hypre_SetIndex(hypre_StructGridMaxDistance(grid), 2, 2, 2);
      max_distance =  hypre_StructGridMaxDistance(grid);
      

      grow_box = hypre_BoxCreate();
      result_box = hypre_BoxCreate();
      periodic_box = hypre_BoxCreate();

      /* now loop through each local box */
      hypre_ForBoxI(i, local_boxes)
      {
         box = hypre_BoxArrayBox(local_boxes, i);
       
         /* now expand box by max_distance or larger and gather entries */
         hypre_CopyBox(box ,grow_box);     
         hypre_BoxExpandConstantDim(grow_box, max_distance);
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
   

   hypre_EndTiming(time_index);

   return hypre_error_flag;
}




/*--------------------------------------------------------------------------
 * hypre_GatherAllBoxes
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GatherAllBoxes(MPI_Comm         comm,
                     hypre_BoxArray  *boxes,
                     hypre_BoxArray **all_boxes_ptr,
                     HYPRE_Int      **all_procs_ptr,
                     HYPRE_Int       *first_local_ptr)
{
   hypre_BoxArray    *all_boxes;
   HYPRE_Int         *all_procs;
   HYPRE_Int          first_local;
   HYPRE_Int          all_boxes_size;

   hypre_Box         *box;
   hypre_Index        imin;
   hypre_Index        imax;
                     
   HYPRE_Int          num_all_procs, my_rank;
                     
   HYPRE_Int         *sendbuf;
   HYPRE_Int          sendcount;
   HYPRE_Int         *recvbuf;
   HYPRE_Int         *recvcounts;
   HYPRE_Int         *displs;
   HYPRE_Int          recvbuf_size;
                     
   HYPRE_Int          i, p, b, d;
   HYPRE_Int          ierr = 0;

   /*-----------------------------------------------------
    * Accumulate the box info
    *-----------------------------------------------------*/
   
   hypre_MPI_Comm_size(comm, &num_all_procs);
   hypre_MPI_Comm_rank(comm, &my_rank);

   /* compute recvcounts and displs */
   sendcount = 7*hypre_BoxArraySize(boxes);
   recvcounts = hypre_SharedTAlloc(HYPRE_Int, num_all_procs);
   displs = hypre_TAlloc(HYPRE_Int, num_all_procs);
   hypre_MPI_Allgather(&sendcount, 1, HYPRE_MPI_INT,
                 recvcounts, 1, HYPRE_MPI_INT, comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (p = 1; p < num_all_procs; p++)
   {
      displs[p] = displs[p-1] + recvcounts[p-1];
      recvbuf_size += recvcounts[p];
   }

   /* allocate sendbuf and recvbuf */
   sendbuf = hypre_TAlloc(HYPRE_Int, sendcount);
   recvbuf = hypre_SharedTAlloc(HYPRE_Int, recvbuf_size);

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
   hypre_MPI_Allgatherv(sendbuf, sendcount, HYPRE_MPI_INT,
                  recvbuf, recvcounts, displs, HYPRE_MPI_INT, comm);

   /* sort recvbuf by process rank? */

   /*-----------------------------------------------------
    * Create all_boxes, etc.
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / 7;
   all_boxes   = hypre_BoxArrayCreate(all_boxes_size);
   all_procs   = hypre_TAlloc(HYPRE_Int, all_boxes_size);
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

HYPRE_Int
hypre_ComputeBoxnums(hypre_BoxArray *boxes,
                     HYPRE_Int      *procs,
                     HYPRE_Int     **boxnums_ptr)
{

   HYPRE_Int         *boxnums;
   HYPRE_Int          num_boxes;
   HYPRE_Int          p, b, boxnum;

   /*-----------------------------------------------------
    *-----------------------------------------------------*/

   num_boxes = hypre_BoxArraySize(boxes);
   boxnums = hypre_TAlloc(HYPRE_Int, num_boxes);

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
 
HYPRE_Int
hypre_StructGridPrint( FILE             *file,
                       hypre_StructGrid *grid )
{

   hypre_BoxArray  *boxes;
   hypre_Box       *box;

   HYPRE_Int        i;

   hypre_fprintf(file, "%d\n", hypre_StructGridDim(grid));

   boxes = hypre_StructGridBoxes(grid);
   hypre_fprintf(file, "%d\n", hypre_BoxArraySize(boxes));
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         hypre_fprintf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
                 i,
                 hypre_BoxIMinX(box),
                 hypre_BoxIMinY(box),
                 hypre_BoxIMinZ(box),
                 hypre_BoxIMaxX(box),
                 hypre_BoxIMaxY(box),
                 hypre_BoxIMaxZ(box));
      }
   hypre_fprintf(file, "\nPeriodic: %d %d %d\n",
           hypre_StructGridPeriodic(grid)[0],
           hypre_StructGridPeriodic(grid)[1],
           hypre_StructGridPeriodic(grid)[2]);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridRead
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_StructGridRead( MPI_Comm           comm,
                      FILE              *file,
                      hypre_StructGrid **grid_ptr )
{

   hypre_StructGrid *grid;

   hypre_Index       ilower;
   hypre_Index       iupper;

   HYPRE_Int         dim;
   HYPRE_Int         num_boxes;
               
   HYPRE_Int         i, idummy;

   hypre_fscanf(file, "%d\n", &dim);
   hypre_StructGridCreate(comm, dim, &grid);

   hypre_fscanf(file, "%d\n", &num_boxes);
   for (i = 0; i < num_boxes; i++)
   {
      hypre_fscanf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
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

HYPRE_Int
hypre_StructGridSetNumGhost( hypre_StructGrid *grid, HYPRE_Int  *num_ghost )
{
  HYPRE_Int  i;
  
  for (i = 0; i < 6; i++)
  {
    hypre_StructGridNumGhost(grid)[i] = num_ghost[i];
  }

  return hypre_error_flag;
}
