/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
static int debug_count = 0;
#endif

/*--------------------------------------------------------------------------
 * hypre_StructMapFineToCoarse
 *
 * NOTE: findex and cindex are indexes on the fine and coarse index space,
 * and do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

int
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
 * NOTE: findex and cindex are indexes on the fine and coarse index space,
 * and do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

int
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

#if 1

/*--------------------------------------------------------------------------
 * hypre_StructCoarsen
 *
 * This routine coarsens the grid, 'fgrid', by the coarsening factor,
 * 'stride', using the index mapping in 'hypre_StructMapFineToCoarse'.
 * The basic algorithm is as follows:
 *
 * 1. Coarsen the neighborhood boxes (hood boxes).
 *
 * 2. Exchange coarsened hood boxes with all neighbor processes.
 *
 * 3. Merge the coarsened hood boxes from my neighbor processes with
 * my local coarsened hood boxes.  The result is a list of coarse grid
 * hood boxes that is sorted and unique.
 *
 * 4. If the boolean variable, 'prune', is nonzero, eliminate boxes of
 * size zero from the coarse grid hood boxes.
 *
 * 5. Use the coarse grid hood boxes to construct the coarse grid.
 *
 * NOTES:
 *
 * 1. All non-periodic neighborhood info is sent.
 *
 * 2. Neighborhoods must contain all (non-periodic) boxes associated
 * with the process where it lives.  The neighbor class routines
 * insure this.
 *
 * 3. Process numbers for non-periodic boxes must appear in the hood
 * in non-decreasing order, and box IDs must be unique and appear in
 * increasing order.
 *
 * 4. This routine will work only if the coarsening factor is <= 2.
 * To extend this algorithm to work with larger coarsening factors,
 * more than one exchange of neighbor information will be needed after
 * each process coarsens its own neighborhood.
 *
 *--------------------------------------------------------------------------*/

#define hypre_StructCoarsenBox(box, index, stride) \
hypre_ProjectBox(box, index, stride);\
hypre_StructMapFineToCoarse(hypre_BoxIMin(box), index, stride,\
                            hypre_BoxIMin(box));\
hypre_StructMapFineToCoarse(hypre_BoxIMax(box), index, stride,\
                            hypre_BoxIMax(box))

int
hypre_StructCoarsen( hypre_StructGrid  *fgrid,
                     hypre_Index        index,
                     hypre_Index        stride,
                     int                prune,
                     hypre_StructGrid **cgrid_ptr )
{
   int ierr = 0;

   hypre_StructGrid   *cgrid;
                      
   MPI_Comm            comm;
   int                 dim;
   hypre_BoxNeighbors *neighbors;
   hypre_BoxArray     *hood_boxes;
   int                 num_hood;
   int                *hood_procs;
   int                *hood_ids;
   int                 first_local;
   int                 num_local;
   int                 id_period;
   int                 num_periods;
   int                 max_distance;
   hypre_Box          *bounding_box;
   hypre_Index         periodic;

   MPI_Request        *send_requests;
   MPI_Status         *send_status;
   int                *send_buffer;
   int                 send_size;
   MPI_Request        *recv_requests;
   MPI_Status         *recv_status;
   int               **recv_buffers;
   int                *recv_sizes;
   int                 my_rank;

   int                *comm_procs;
   int                 num_comms;
                      
   int               **arrays;
   int                *sizes;
   int                 size;
   int                *mergei;
   int                *mergej;

   hypre_BoxArray     *new_hood_boxes;
   int                 new_num_hood;
   int                *new_hood_procs;
   int                *new_hood_ids;
   int                 new_first_local;
   int                 new_num_local;

   hypre_Box          *box;
   hypre_Index         imin;
   hypre_Index         imax;
   int                 alloc_size;

   int                 i, j, d;
   int                 jj;

   /*-----------------------------------------
    * Copy needed info from fgrid
    *-----------------------------------------*/

   comm         = hypre_StructGridComm(fgrid);
   dim          = hypre_StructGridDim(fgrid);
   neighbors    = hypre_StructGridNeighbors(fgrid);
   hood_boxes   = hypre_BoxNeighborsBoxes(neighbors);
   hood_procs   = hypre_BoxNeighborsProcs(neighbors);
   hood_ids     = hypre_BoxNeighborsIDs(neighbors);
   num_hood     = hypre_BoxArraySize(hood_boxes);
   id_period    = hypre_BoxNeighborsIDPeriod(neighbors);
   num_periods  = hypre_BoxNeighborsNumPeriods(neighbors);

   /* adjust num_hood to focus only on non-periodic boxes */
   num_hood /= num_periods;

   /* make a copy of hood_boxes, hood_procs, and hood_ids */
   new_hood_boxes = hypre_BoxArrayCreate(num_hood);
   new_hood_procs = hypre_TAlloc(int, num_hood);
   new_hood_ids   = hypre_TAlloc(int, num_hood);
   for (i = 0; i < num_hood; i++)
   {
      hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, i),
                    hypre_BoxArrayBox(new_hood_boxes, i));
      new_hood_procs[i] = hood_procs[i];
      new_hood_ids[i] = hood_ids[i];
   }
   hood_boxes = new_hood_boxes;
   hood_procs = new_hood_procs;
   hood_ids   = new_hood_ids;

   first_local  = hypre_BoxNeighborsFirstLocal(neighbors);
   num_local    = hypre_BoxNeighborsNumLocal(neighbors);

   max_distance = hypre_StructGridMaxDistance(fgrid);
   bounding_box = hypre_BoxDuplicate(hypre_StructGridBoundingBox(fgrid));
   hypre_CopyIndex(hypre_StructGridPeriodic(fgrid), periodic);

   MPI_Comm_rank(comm, &my_rank);

#if DEBUG
   sprintf(filename, "zcoarsen.%05d", my_rank);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");
   fprintf(file, "\n\n%d\n\n", debug_count++);
   fprintf(file, "num_hood = %d\n", num_hood);
   for (i = 0; i < num_hood; i++)
   {
      box = hypre_BoxArrayBox(hood_boxes, i);
      fprintf(file, "(%d,%d,%d) X (%d,%d,%d) ; (%d,%d); %d\n",
              hypre_BoxIMinX(box),hypre_BoxIMinY(box),hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box),hypre_BoxIMaxY(box),hypre_BoxIMaxZ(box),
              hood_procs[i], hood_ids[i], hypre_BoxVolume(box));
   }
   fprintf(file, "first_local  = %d\n", first_local);
   fprintf(file, "num_local    = %d\n", num_local);
#endif

   /*-----------------------------------------
    * Coarsen bounding box
    *-----------------------------------------*/

   hypre_StructCoarsenBox(bounding_box, index, stride);

   /*-----------------------------------------
    * Coarsen neighborhood boxes
    *-----------------------------------------*/

   for (i = 0; i < num_hood; i++)
   {
      box = hypre_BoxArrayBox(hood_boxes, i);
      hypre_StructCoarsenBox(box, index, stride);
   }

   /*-----------------------------------------
    * Determine send and receive procs.
    *
    * NOTE: We communicate with all neighbor
    * processes, hence comm_procs is used for
    * both sends and receives.
    *
    * NOTE: This relies on the fact that the
    * process numbers for the non-periodic
    * boxes are in non-decreasing order.
    *-----------------------------------------*/

   comm_procs = NULL;
   num_comms  = 0;
   for (i = 0; i < num_hood; i++)
   {
      if (hood_procs[i] != my_rank)
      {
         if (num_comms == 0)
         {
            comm_procs = hypre_TAlloc(int, num_hood);
            comm_procs[num_comms] = hood_procs[i];
            num_comms++;
         }
         else if (hood_procs[i] > hood_procs[i-1])
         {
            comm_procs[num_comms] = hood_procs[i];
            num_comms++;
         }
      }
   }

#if DEBUG
   fprintf(file, "num_comms = %d\n", num_comms);
   for (i = 0; i < num_comms; i++)
   {
      fprintf(file, "%d ", comm_procs[i]);
   }
   fprintf(file, "\n");

   fflush(file);
   fclose(file);
#endif

   /*-----------------------------------------
    * Exchange neighbor info with other procs
    *-----------------------------------------*/

   if (num_comms)
   {
      /* neighbor size info - post receives */

      recv_requests = hypre_TAlloc(MPI_Request, num_comms);
      recv_status   = hypre_TAlloc(MPI_Status, num_comms);

      recv_sizes = hypre_TAlloc(int, num_comms);
      for (i = 0; i < num_comms; i++)
      {
         MPI_Irecv(&recv_sizes[i], 1, MPI_INT,
                   comm_procs[i], 0, comm, &recv_requests[i]);
      }

      /* neighbor size info - post sends */

      send_requests = hypre_TAlloc(MPI_Request, num_comms);
      send_status   = hypre_TAlloc(MPI_Status, num_comms);

      send_size = 8 * num_hood;
      for (i = 0; i < num_comms; i++)
      {
         MPI_Isend(&send_size, 1, MPI_INT,
                   comm_procs[i], 0, comm, &send_requests[i]);
      }

      /* neighbor size info - complete receives */
      MPI_Waitall(num_comms, recv_requests, recv_status);

      /* neighbor size info - complete sends */
      MPI_Waitall(num_comms, send_requests, send_status);
   }

   /*-----------------------------------------*/

   if (num_comms)
   {
      /* neighbor info - post receives */

      recv_buffers = hypre_TAlloc(int *, num_comms);
      for (i = 0; i < num_comms; i++)
      {
         recv_buffers[i] = hypre_SharedTAlloc(int, recv_sizes[i]);
         MPI_Irecv(recv_buffers[i], recv_sizes[i], MPI_INT,
                   comm_procs[i], 0, comm, &recv_requests[i]);
      }

      /* neighbor info - post sends */

      /* pack the send buffer */
      send_buffer = hypre_SharedTAlloc(int, send_size);
      for (j = 0; j < num_hood; j++)
      {
         send_buffer[j] = hood_ids[j];
         jj = num_hood + 7*j;
         send_buffer[jj++] = hood_procs[j];
         box = hypre_BoxArrayBox(hood_boxes, j);
         for (d = 0; d < 3; d++)
         {
            send_buffer[jj++] = hypre_BoxIMinD(box, d);
            send_buffer[jj++] = hypre_BoxIMaxD(box, d);
         }
      }

      for (i = 0; i < num_comms; i++)
      {
         MPI_Isend(send_buffer, send_size, MPI_INT,
                   comm_procs[i], 0, comm, &send_requests[i]);
      }

      /* neighbor info - complete receives */
      MPI_Waitall(num_comms, recv_requests, recv_status);

      hypre_TFree(recv_requests);
      hypre_TFree(recv_status);

      /* neighbor info - complete sends */
      MPI_Waitall(num_comms, send_requests, send_status);

      hypre_TFree(send_requests);
      hypre_TFree(send_status);
      hypre_TFree(send_buffer);
   }

   /*-----------------------------------------
    * Unpack the recv buffers to create
    * new neighborhood info
    *-----------------------------------------*/

   if (num_comms)
   {
      /* merge the buffers according to ID */
      size   = num_comms + 1;
      arrays = hypre_TAlloc(int *, size);
      sizes  = hypre_TAlloc(int, size);
      for (i = 0; i < num_comms; i++)
      {
         arrays[i] = recv_buffers[i];
         sizes[i]  = recv_sizes[i] / 8;
      }
      arrays[num_comms] = hood_ids;
      sizes[num_comms]  = num_hood;

      hypre_Merge(arrays, sizes, size, &mergei, &mergej);

      alloc_size = num_hood;
      new_hood_boxes = hypre_BoxArrayCreate(alloc_size);
      hypre_BoxArraySetSize(new_hood_boxes, 0);
      new_hood_procs = hypre_TAlloc(int, alloc_size);
      new_hood_ids   = hypre_TAlloc(int, alloc_size);

      box = hypre_BoxCreate();

      new_num_hood = 0;
      while (mergei[new_num_hood] > -1)
      {
         i = mergei[new_num_hood];
         j = mergej[new_num_hood];

         if (new_num_hood == alloc_size)
         {
            alloc_size += num_hood;
            new_hood_procs = hypre_TReAlloc(new_hood_procs, int, alloc_size);
            new_hood_ids   = hypre_TReAlloc(new_hood_ids,   int, alloc_size);
         }

         if (i == num_comms)
         {
            /* get data from my neighborhood */
            new_hood_procs[new_num_hood] = hood_procs[j];
            new_hood_ids[new_num_hood]   = hood_ids[j];
            hypre_AppendBox(hypre_BoxArrayBox(hood_boxes, j),
                            new_hood_boxes);
         }
         else
         {
            /* get data from recv buffer neighborhoods */
            new_hood_ids[new_num_hood] = recv_buffers[i][j];
            jj = sizes[i] + 7*j;
            new_hood_procs[new_num_hood] = recv_buffers[i][jj++];
            for (d = 0; d < 3; d++)
            {
               hypre_IndexD(imin, d) = recv_buffers[i][jj++];
               hypre_IndexD(imax, d) = recv_buffers[i][jj++];
            }
            hypre_BoxSetExtents(box, imin, imax);
            hypre_AppendBox(box, new_hood_boxes);
         }

         if (new_hood_ids[new_num_hood] == hood_ids[first_local])
         {
            new_first_local = new_num_hood;
         }

         new_num_hood++;
      }

      hypre_TFree(arrays);
      hypre_TFree(sizes);
      hypre_TFree(mergei);
      hypre_TFree(mergej);

      for (i = 0; i < num_comms; i++)
      {
         hypre_TFree(recv_buffers[i]);
      }
      hypre_TFree(recv_buffers);
      hypre_TFree(recv_sizes);

      hypre_BoxDestroy(box);

      hypre_BoxArrayDestroy(hood_boxes);
      hypre_TFree(hood_procs);
      hypre_TFree(hood_ids);

      hood_boxes  = new_hood_boxes;
      num_hood    = new_num_hood;
      hood_procs  = new_hood_procs;
      hood_ids    = new_hood_ids;
      first_local = new_first_local;
   }

   hypre_TFree(comm_procs);

   /*-----------------------------------------
    * Eliminate boxes of size zero
    *-----------------------------------------*/

   if (prune)
   {
      j = 0;
      new_first_local = -1;
      new_num_local = 0;
      for (i = 0; i < num_hood; i++)
      {
         box = hypre_BoxArrayBox(hood_boxes, i);
         if ( hypre_BoxVolume(box) )
         {
            hypre_CopyBox(box, hypre_BoxArrayBox(hood_boxes, j));
            hood_procs[j] = hood_procs[i];
            hood_ids[j]   = hood_ids[i];
            if ((i >= first_local) &&
                (i <  first_local + num_local))
            {
               if (new_first_local == -1)
               {
                  new_first_local = j;
               }
               new_num_local++;
            }
            j++;
         }
      }
      num_hood = j;
      hypre_BoxArraySetSize(hood_boxes, num_hood);
      first_local  = new_first_local;
      num_local    = new_num_local;
   }

   /*-----------------------------------------
    * Build the coarse grid
    *-----------------------------------------*/

   hypre_StructGridCreate(comm, dim, &cgrid);

   /* set neighborhood */
   hypre_StructGridSetHood(cgrid, hood_boxes, hood_procs, hood_ids,
                           first_local, num_local, id_period, bounding_box);

   hypre_StructGridSetHoodInfo(cgrid, max_distance);

   hypre_StructGridSetPeriodic(cgrid, periodic);

   hypre_StructGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return ierr;
}

#undef hypre_StructCoarsenBox

#else

/*--------------------------------------------------------------------------
 * hypre_StructCoarsen    - OLD
 *--------------------------------------------------------------------------*/

int
hypre_StructCoarsen( hypre_StructGrid  *fgrid,
                     hypre_Index        index,
                     hypre_Index        stride,
                     int                prune,
                     hypre_StructGrid **cgrid_ptr )
{
   int ierr = 0;

   hypre_StructGrid *cgrid;

   MPI_Comm          comm  = hypre_StructGridComm(fgrid);
   int               dim   = hypre_StructGridDim(fgrid);
   hypre_BoxArray   *boxes;

   hypre_Box        *box;
                    
   int               i, d;

   hypre_StructGridCreate(comm, dim, &cgrid);

   /* coarsen boxes */
   boxes = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(fgrid));
   hypre_ProjectBoxArray(boxes, index, stride);
   for (i = 0; i < hypre_BoxArraySize(boxes); i++)
   {
      box = hypre_BoxArrayBox(boxes, i);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(box), index, stride,
                                  hypre_BoxIMin(box));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(box), index, stride,
                                  hypre_BoxIMax(box));
   }

   /* set boxes */
   hypre_StructGridSetBoxes(cgrid, boxes);

   hypre_StructGridSetPeriodic(cgrid, hypre_StructGridPeriodic(fgrid));

   hypre_StructGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return ierr;
}

#endif


/*--------------------------------------------------------------------------
 * hypre_Merge
 *
 * Merge the integers in the sorted 'arrays'.  The routine returns the
 * (array, array_index) pairs for each entry in the merged list.  The
 * end of the pairs is indicated by the first '-1' entry in 'mergei'.
 *
 *--------------------------------------------------------------------------*/

int
hypre_Merge( int   **arrays,
             int    *sizes,
             int     size,
             int   **mergei_ptr,
             int   **mergej_ptr )
{
   int ierr = 0;

   int  *mergei;
   int  *mergej;

   int   num, i;
   int   lastval;

   struct linkstruct
   {
      int  i;
      int  j;
      struct linkstruct  *next;

   } *list, *first, *link, *next;

   num = 0;
   for (i = 0; i < size; i++)
   {
      num += sizes[i];
   }
   mergei = hypre_TAlloc(int, num+1);
   mergej = hypre_TAlloc(int, num+1);

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

