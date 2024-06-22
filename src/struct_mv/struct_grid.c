/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_StructGrid class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

#define DEBUG 0

#if DEBUG
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
                        HYPRE_Int          ndim,
                        hypre_StructGrid **grid_ptr)
{
   hypre_StructGrid    *grid;
   HYPRE_Int           i;

   grid = hypre_TAlloc(hypre_StructGrid,  1, HYPRE_MEMORY_HOST);

   hypre_StructGridComm(grid)        = comm;
   hypre_StructGridNDim(grid)        = ndim;
   hypre_StructGridBoxes(grid)       = hypre_BoxArrayCreate(0, ndim);
   hypre_StructGridIDs(grid)         = NULL;

   hypre_SetIndex(hypre_StructGridMaxDistance(grid), 8);

   hypre_StructGridBoundingBox(grid) = NULL;
   hypre_StructGridLocalSize(grid)   = 0;
   hypre_StructGridGlobalSize(grid)  = 0;
   hypre_SetIndex(hypre_StructGridPeriodic(grid), 0);
   hypre_StructGridRefCount(grid)     = 1;
   hypre_StructGridBoxMan(grid)       = NULL;

   hypre_StructGridNumPeriods(grid)   = 1;
   hypre_StructGridPShifts(grid)     = NULL;

   hypre_StructGridGhlocalSize(grid)  = 0;
   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructGridNumGhost(grid)[i] = 1;
   }

   *grid_ptr = grid;

   return hypre_error_flag;
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

   return hypre_error_flag;
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
         hypre_TFree(hypre_StructGridIDs(grid), HYPRE_MEMORY_HOST);
         hypre_BoxArrayDestroy(hypre_StructGridBoxes(grid));

         hypre_BoxManDestroy(hypre_StructGridBoxMan(grid));
         hypre_TFree( hypre_StructGridPShifts(grid), HYPRE_MEMORY_HOST);

         hypre_TFree(grid, HYPRE_MEMORY_HOST);
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

   box = hypre_BoxCreate(hypre_StructGridNDim(grid));
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

   hypre_TFree(hypre_StructGridBoxes(grid), HYPRE_MEMORY_HOST);
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
   hypre_TFree(hypre_StructGridIDs(grid), HYPRE_MEMORY_HOST);
   hypre_StructGridIDs(grid) = ids;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoxManager
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructGridSetBoxManager( hypre_StructGrid *grid,
                               hypre_BoxManager *boxman )
{

   hypre_TFree(hypre_StructGridBoxMan(grid), HYPRE_MEMORY_HOST);
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
 *   2. if this is called from StructCoarsen, then the Box Manager has already
 *   been created, and ids have been set
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
   HYPRE_BigInt global_size;
   HYPRE_Int max_nentries;
   HYPRE_Int info_size;
   HYPRE_Int num_periods;

   HYPRE_Int *ids = NULL;
   HYPRE_Int  iperiodic, notcenter;

   HYPRE_Int  sendbuf6[2 * HYPRE_MAXDIM], recvbuf6[2 * HYPRE_MAXDIM];

   hypre_Box  *box;
   hypre_Box  *ghostbox;
   hypre_Box  *grow_box;
   hypre_Box  *periodic_box;
   hypre_Box  *result_box;

   hypre_Index min_index, max_index, loop_size;
   hypre_Index *pshifts;
   hypre_IndexRef pshift;

   void *entry_info = NULL;

   /*  initialize info from the grid */
   MPI_Comm             comm         = hypre_StructGridComm(grid);
   HYPRE_Int            ndim         = hypre_StructGridNDim(grid);
   hypre_BoxArray      *local_boxes  = hypre_StructGridBoxes(grid);
   hypre_IndexRef       max_distance = hypre_StructGridMaxDistance(grid);
   hypre_Box           *bounding_box = hypre_StructGridBoundingBox(grid);
   hypre_IndexRef       periodic     = hypre_StructGridPeriodic(grid);
   hypre_BoxManager    *boxman       = hypre_StructGridBoxMan(grid);
   HYPRE_Int           *numghost     = hypre_StructGridNumGhost(grid);

   if (!time_index)
   {
      time_index = hypre_InitializeTiming("StructGridAssemble");
   }

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
      ids = hypre_CTAlloc(HYPRE_Int,  num_local_boxes, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_local_boxes; i++)
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

   box = hypre_BoxCreate(ndim);
   for (d = 0; d < ndim; d++)
   {
      iperiodic = hypre_IndexD(periodic, d) ? 1 : 0;
      hypre_BoxIMinD(box, d) = -iperiodic;
      hypre_BoxIMaxD(box, d) =  iperiodic;
   }
   num_periods = hypre_BoxVolume(box);

   pshifts = hypre_CTAlloc(hypre_Index,  num_periods, HYPRE_MEMORY_HOST);
   pshift = pshifts[0];
   hypre_SetIndex(pshift, 0);
   if (num_periods > 1)
   {
      p = 1;
      hypre_BoxGetSize(box, loop_size);
      hypre_SerialBoxLoop0Begin(ndim, loop_size);
      {
         pshift = pshifts[p];
         zypre_BoxLoopGetIndex(pshift);
         hypre_AddIndexes(pshift, hypre_BoxIMin(box), ndim, pshift);
         notcenter = 0;
         for (d = 0; d < ndim; d++)
         {
            hypre_IndexD(pshift, d) *= hypre_IndexD(periodic, d);
            if (hypre_IndexD(pshift, d))
            {
               notcenter = 1;
            }
         }
         if (notcenter)
         {
            p++;
         }
      }
      hypre_SerialBoxLoop0End();
   }
   hypre_BoxDestroy(box);

   hypre_StructGridNumPeriods(grid) = num_periods;
   hypre_StructGridPShifts(grid)    = pshifts;

   /********calculate local size and the ghost size **************/

   size = 0;
   ghostsize = 0;
   ghostbox = hypre_BoxCreate(ndim);

   hypre_ForBoxI(i, local_boxes)
   {
      box = hypre_BoxArrayBox(local_boxes, i);
      size +=  hypre_BoxVolume(box);

      hypre_CopyBox(box, ghostbox);
      hypre_BoxGrowByArray(ghostbox, numghost);
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

      HYPRE_BigInt big_size = (HYPRE_BigInt)size;
      hypre_MPI_Allreduce(&big_size, &global_size, 1, HYPRE_MPI_BIG_INT,
                          hypre_MPI_SUM, comm);
      hypre_StructGridGlobalSize(grid) = global_size; /* TO DO: this HYPRE_Int
                                                       * could overflow! (used
                                                       * to calc flops) */

      /*************** set bounding box ***********/

      bounding_box = hypre_BoxCreate(ndim);

      if (num_local_boxes)
      {
         /* initialize min and max index*/
         box = hypre_BoxArrayBox(local_boxes, 0);
         for (d = 0; d < ndim; d++)
         {
            hypre_IndexD(min_index, d) =  hypre_BoxIMinD(box, d);
            hypre_IndexD(max_index, d) =  hypre_BoxIMaxD(box, d);
         }

         hypre_ForBoxI(i, local_boxes)
         {
            box = hypre_BoxArrayBox(local_boxes, i);


            /* find min and max box extents */
            for (d = 0; d < ndim; d++)
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
         for (d = 0; d < ndim; d++)
         {
            hypre_BoxIMinD(bounding_box, d) =  hypre_pow2(30);
            hypre_BoxIMaxD(bounding_box, d) = -hypre_pow2(30);
         }
      }
      /* set the extra dimensions of the bounding box to zero */
      for (d = ndim; d < HYPRE_MAXDIM; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = 0;
         hypre_BoxIMaxD(bounding_box, d) = 0;
      }

      /* communication needed for the bounding box */
      /* pack buffer */
      for (d = 0; d < ndim; d++)
      {
         sendbuf6[d] = hypre_BoxIMinD(bounding_box, d);
         sendbuf6[d + ndim] = -hypre_BoxIMaxD(bounding_box, d);
      }
      hypre_MPI_Allreduce(sendbuf6, recvbuf6, 2 * ndim, HYPRE_MPI_INT,
                          hypre_MPI_MIN, comm);
      /* unpack buffer */
      for (d = 0; d < ndim; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
         hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d + ndim];
      }

      hypre_StructGridBoundingBox(grid) = bounding_box;

      /*************** create a box manager *****************/
      max_nentries =  num_local_boxes + 20;
      info_size = 0; /* we don't need an info object */
      hypre_BoxManCreate(max_nentries, info_size, ndim, bounding_box,
                         comm, &boxman);

      /******** populate the box manager with my local boxes and gather neighbor
                information  ******/

      grow_box = hypre_BoxCreate(ndim);
      result_box = hypre_BoxCreate(ndim);
      periodic_box = hypre_BoxCreate(ndim);

      /* now loop through each local box */
      hypre_ForBoxI(i, local_boxes)
      {
         box = hypre_BoxArrayBox(local_boxes, i);
         /* add entry for each local box (the id is the boxnum, and should be
            sequential */
         hypre_BoxManAddEntry( boxman, hypre_BoxIMin(box), hypre_BoxIMax(box),
                               myid, i, entry_info );

         /* now expand box by max_distance or larger and gather entries */
         hypre_CopyBox(box, grow_box);
         hypre_BoxGrowByIndex(grow_box, max_distance);
         hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(grow_box),
                                   hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k = 1; k < num_periods; k++) /* k=0 is original box */
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

   /* boxman was created, but need to get additional neighbor info */
   else if ( hypre_IndexEqual(max_distance, 0, ndim) )
   {
      /* pick a new max distance and set in grid*/
      hypre_SetIndex(hypre_StructGridMaxDistance(grid), 2);
      max_distance =  hypre_StructGridMaxDistance(grid);

      grow_box = hypre_BoxCreate(ndim);
      result_box = hypre_BoxCreate(ndim);
      periodic_box = hypre_BoxCreate(ndim);

      /* now loop through each local box */
      hypre_ForBoxI(i, local_boxes)
      {
         box = hypre_BoxArrayBox(local_boxes, i);

         /* now expand box by max_distance or larger and gather entries */
         hypre_CopyBox(box, grow_box);
         hypre_BoxGrowByIndex(grow_box, max_distance);
         hypre_BoxManGatherEntries(boxman, hypre_BoxIMin(grow_box),
                                   hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k = 1; k < num_periods; k++) /* k=0 is original box */
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
                     HYPRE_Int        ndim,
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
   HYPRE_Int          item_size;

   HYPRE_Int          i, p, b, d;

   /*-----------------------------------------------------
    * Accumulate the box info
    *-----------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_all_procs);
   hypre_MPI_Comm_rank(comm, &my_rank);

   /* compute recvcounts and displs */
   item_size = 2 * ndim + 1;
   sendcount = item_size * hypre_BoxArraySize(boxes);
   recvcounts =  hypre_TAlloc(HYPRE_Int,  num_all_procs, HYPRE_MEMORY_HOST);
   displs = hypre_TAlloc(HYPRE_Int,  num_all_procs, HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(&sendcount, 1, HYPRE_MPI_INT,
                       recvcounts, 1, HYPRE_MPI_INT, comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (p = 1; p < num_all_procs; p++)
   {
      displs[p] = displs[p - 1] + recvcounts[p - 1];
      recvbuf_size += recvcounts[p];
   }

   /* allocate sendbuf and recvbuf */
   sendbuf = hypre_TAlloc(HYPRE_Int,  sendcount, HYPRE_MEMORY_HOST);
   recvbuf =  hypre_TAlloc(HYPRE_Int,  recvbuf_size, HYPRE_MEMORY_HOST);

   /* put local box extents and process number into sendbuf */
   i = 0;
   for (b = 0; b < hypre_BoxArraySize(boxes); b++)
   {
      sendbuf[i++] = my_rank;

      box = hypre_BoxArrayBox(boxes, b);
      for (d = 0; d < ndim; d++)
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
   all_boxes_size = recvbuf_size / item_size;
   all_boxes   = hypre_BoxArrayCreate(all_boxes_size, ndim);
   all_procs   = hypre_TAlloc(HYPRE_Int,  all_boxes_size, HYPRE_MEMORY_HOST);
   first_local = -1;
   i = 0;
   b = 0;
   box = hypre_BoxCreate(ndim);
   while (i < recvbuf_size)
   {
      all_procs[b] = recvbuf[i++];
      for (d = 0; d < ndim; d++)
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

   hypre_TFree(sendbuf, HYPRE_MEMORY_HOST);
   hypre_TFree(recvbuf, HYPRE_MEMORY_HOST);
   hypre_TFree(recvcounts, HYPRE_MEMORY_HOST);
   hypre_TFree(displs, HYPRE_MEMORY_HOST);

   *all_boxes_ptr   = all_boxes;
   *all_procs_ptr   = all_procs;
   *first_local_ptr = first_local;

   return hypre_error_flag;
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
   HYPRE_Int          p, b;
   HYPRE_Int          boxnum = 0;

   /*-----------------------------------------------------
    *-----------------------------------------------------*/

   num_boxes = hypre_BoxArraySize(boxes);
   boxnums = hypre_TAlloc(HYPRE_Int,  num_boxes, HYPRE_MEMORY_HOST);

   p = -1;
   for (b = 0; b < num_boxes; b++)
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

   HYPRE_Int        i, d, ndim;

   ndim = hypre_StructGridNDim(grid);
   hypre_fprintf(file, "%d\n", ndim);

   boxes = hypre_StructGridBoxes(grid);
   hypre_fprintf(file, "%d\n", hypre_BoxArraySize(boxes));

   /* Print lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      hypre_fprintf(file, "%d:  (%d", i, hypre_BoxIMinD(box, 0));
      for (d = 1; d < ndim; d++)
      {
         hypre_fprintf(file, ", %d", hypre_BoxIMinD(box, d));
      }
      hypre_fprintf(file, ")  x  (%d", hypre_BoxIMaxD(box, 0));
      for (d = 1; d < ndim; d++)
      {
         hypre_fprintf(file, ", %d", hypre_BoxIMaxD(box, d));
      }
      hypre_fprintf(file, ")\n");
   }
   /* Print line of the form: "Periodic: %d %d %d\n" */
   hypre_fprintf(file, "\nPeriodic:");
   for (d = 0; d < ndim; d++)
   {
      hypre_fprintf(file, " %d", hypre_StructGridPeriodic(grid)[d]);
   }
   hypre_fprintf(file, "\n");

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
   hypre_IndexRef    periodic;

   HYPRE_Int         ndim;
   HYPRE_Int         num_boxes;

   HYPRE_Int         i, d, idummy;

   hypre_fscanf(file, "%d\n", &ndim);
   hypre_StructGridCreate(comm, ndim, &grid);

   hypre_fscanf(file, "%d\n", &num_boxes);

   /* Read lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   for (i = 0; i < num_boxes; i++)
   {
      hypre_fscanf(file, "%d:  (%d", &idummy, &hypre_IndexD(ilower, 0));
      for (d = 1; d < ndim; d++)
      {
         hypre_fscanf(file, ", %d", &hypre_IndexD(ilower, d));
      }
      hypre_fscanf(file, ")  x  (%d", &hypre_IndexD(iupper, 0));
      for (d = 1; d < ndim; d++)
      {
         hypre_fscanf(file, ", %d", &hypre_IndexD(iupper, d));
      }
      hypre_fscanf(file, ")\n");

      hypre_StructGridSetExtents(grid, ilower, iupper);
   }

   periodic = hypre_StructGridPeriodic(grid);

   /* Read line of the form: "Periodic: %d %d %d\n" */
   hypre_fscanf(file, "Periodic:");
   for (d = 0; d < ndim; d++)
   {
      hypre_fscanf(file, " %d", &hypre_IndexD(periodic, d));
   }
   hypre_fscanf(file, "\n");

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
   HYPRE_Int  i, ndim = hypre_StructGridNDim(grid);

   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructGridNumGhost(grid)[i] = num_ghost[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridGetMaxBoxSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructGridGetMaxBoxSize( hypre_StructGrid *grid )
{
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   HYPRE_Int         i, max_box_size = 0;

   boxes = hypre_StructGridBoxes(grid);
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(hypre_StructGridBoxes(grid), i);
      max_box_size = hypre_max(max_box_size, hypre_BoxVolume(box));
   }

   return max_box_size;
}

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int
hypre_StructGridSetDataLocation( HYPRE_StructGrid grid, HYPRE_MemoryLocation data_location )
{
   hypre_StructGridDataLocation(grid) = data_location;

   return hypre_error_flag;
}

#endif
