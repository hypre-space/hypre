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
 * hypre_StructCoarsen    - NEW
 *
 * This routine coarsens the grid, 'fgrid', by the coarsening factor,
 * 'stride', using the index mapping in 'hypre_StructMapFineToCoarse'.
 * The basic algorithm is as follows:
 *
 * 1. Coarsen the neighborhood boxes.
 *
 * 2. Loop through neighborhood boxes, and compute the minimum
 * positive outside xyz distances from local boxes to neighbor boxes.
 * If some xyz distance is less than desired, receive neighborhood
 * information from the neighbor box processor.
 *
 * 3. Loop through neighborhood boxes, and compute the minimum
 * positive outside xyz distances from neighbor boxes to local boxes.
 * If some xyz distance is less than desired, send neighborhood
 * information to the neighbor box processor.
 *
 * 4. If the boolean variable, 'prune', is nonzero, eliminate boxes of
 * size zero from the coarse grid.
 *
 * Notes:
 *
 * 1. All neighborhood info is sent.
 *
 * 2. Positive outside difference, d, is defined as follows:
 *
 *               |<---- d ---->|
 *         ------        ------
 *        |      |      |      |
 *        | box1 |      | box2 | 
 *        |      |      |      |
 *         ------        ------
 *
 * 3. Neighborhoods must contain all boxes associated with the
 * processor where it lives.  In particular, "periodic boxes", (i.e.,
 * those boxes that were shifted to handle periodicity) associated
 * with local boxes should remain in the neighborhood.  The neighbor
 * class routines insure this.
 *
 * 4. Processor numbers must appear in non-decreasing order in the
 * neighborhood box array, and IDs must be unique and appear in
 * increasing order.
 *
 * 5. Neighborhood information only needs to be exchanged the first
 * time a box boundary moves within the max_distance perimeter.
 *
 * 6. Boxes of size zero must also be considered when determining
 * neighborhood information exchanges.
 *
 * 7. This routine will work only if the coarsening factor is <= 2.
 * To extend this algorithm to work with larger coarsening factors,
 * more than one exchange of neighbor information will be needed after
 * each processor coarsens its own neighborhood.
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
   int                 num_periodic;
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

   int                *send_procs;
   int                *recv_procs;
   int                 num_sends;
   int                 num_recvs;
                      
   hypre_BoxArray     *new_hood_boxes;
   int                 new_num_hood;
   int                *new_hood_procs;
   int                *new_hood_ids;
   int                 new_first_local;
   int                 new_num_local;
   int                 new_num_periodic;

   hypre_Box          *box;
   hypre_Box          *local_box;
   hypre_Box          *neighbor_box;
   hypre_Box          *local_cbox;
   hypre_Box          *neighbor_cbox;
   hypre_Index         imin;
   hypre_Index         imax;
   int                 alloc_size;

   double              perimeter_count, cperimeter_count;
   /*double              diff, distance, perimeter_count, cperimeter_count;*/
                      
   int                *iarray;
   int                *jrecv;
   int                 i, j, d, ilocal;
   int                 data_id, min_id, jj;

   /*-----------------------------------------
    * Copy needed info from fgrid
    *-----------------------------------------*/

   comm         = hypre_StructGridComm(fgrid);
   dim          = hypre_StructGridDim(fgrid);
   neighbors    = hypre_StructGridNeighbors(fgrid);
   hood_boxes   = hypre_BoxArrayDuplicate(hypre_BoxNeighborsBoxes(neighbors));
   num_hood     = hypre_BoxArraySize(hood_boxes);

   iarray  = hypre_BoxNeighborsProcs(neighbors);
   hood_procs = hypre_TAlloc(int, num_hood);
   for (i = 0; i < num_hood; i++)
   {
      hood_procs[i] = iarray[i];
   }

   iarray = hypre_BoxNeighborsIDs(neighbors);
   hood_ids  = hypre_TAlloc(int, num_hood);
   for (i = 0; i < num_hood; i++)
   {
      hood_ids[i] = iarray[i];
   }

   first_local  = hypre_BoxNeighborsFirstLocal(neighbors);
   num_local    = hypre_BoxNeighborsNumLocal(neighbors);
   num_periodic = hypre_BoxNeighborsNumPeriodic(neighbors);

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
   fprintf(file, "num_periodic = %d\n", num_periodic);
#endif

   /*-----------------------------------------
    * Coarsen bounding box
    *-----------------------------------------*/

   hypre_StructCoarsenBox(bounding_box, index, stride);

   /*-----------------------------------------
    * Coarsen neighborhood boxes & determine
    * send / recv procs
    *
    * NOTE: Currently, this always communicates
    * with all neighboring processes.
    *-----------------------------------------*/

   local_cbox = hypre_BoxCreate();
   neighbor_cbox = hypre_BoxCreate();

   num_recvs = 0;
   num_sends = 0;
   recv_procs = NULL;
   send_procs = NULL;
   for (i = 0; i < num_hood; i++)
   {
      if (hood_procs[i] != my_rank)
      {
         for (j = 0; j < num_local; j++)
         {
            ilocal = first_local + j;

            local_box    = hypre_BoxArrayBox(hood_boxes, ilocal);
            neighbor_box = hypre_BoxArrayBox(hood_boxes, i);

            /* coarsen boxes being considered */
            hypre_CopyBox(local_box, local_cbox);
            hypre_StructCoarsenBox(local_cbox, index, stride);
            hypre_CopyBox(neighbor_box, neighbor_cbox);
            hypre_StructCoarsenBox(neighbor_cbox, index, stride);

            /*-----------------------
             * Receive info?
             *-----------------------*/

/* always communicate */
#if 0
            perimeter_count = 0;
            cperimeter_count = 0;
            for (d = 0; d < 3; d++)
            {
               distance = max_distance;
               diff = hypre_BoxIMaxD(neighbor_box, d) -
                  hypre_BoxIMaxD(local_box, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               diff = hypre_BoxIMinD(local_box, d) -
                  hypre_BoxIMinD(neighbor_box, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               if (distance < max_distance)
               {
                  perimeter_count++;
               }

               distance = max_distance;
               diff = hypre_BoxIMaxD(neighbor_cbox, d) -
                  hypre_BoxIMaxD(local_cbox, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               diff = hypre_BoxIMinD(local_cbox, d) -
                  hypre_BoxIMinD(neighbor_cbox, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               if (distance < max_distance)
               {
                  cperimeter_count++;
               }
            }
#else
            perimeter_count = 0;
            cperimeter_count = 1;
#endif
            if (cperimeter_count > perimeter_count)
            {
               if (num_recvs == 0)
               {
                  recv_procs = hypre_TAlloc(int, num_hood);
                  recv_procs[num_recvs] = hood_procs[i];
                  num_recvs++;
               }
               else if (hood_procs[i] != recv_procs[num_recvs-1])
               {
                  recv_procs[num_recvs] = hood_procs[i];
                  num_recvs++;
               }
            }

            /*-----------------------
             * Send info?
             *-----------------------*/

/* always communicate */
#if 0
            perimeter_count = 0;
            cperimeter_count = 0;
            for (d = 0; d < 3; d++)
            {
               distance = max_distance;
               diff = hypre_BoxIMaxD(local_box, d) -
                  hypre_BoxIMaxD(neighbor_box, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               diff = hypre_BoxIMinD(neighbor_box, d) -
                  hypre_BoxIMinD(local_box, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               if (distance < max_distance)
               {
                  perimeter_count++;
               }

               distance = max_distance;
               diff = hypre_BoxIMaxD(local_cbox, d) -
                  hypre_BoxIMaxD(neighbor_cbox, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               diff = hypre_BoxIMinD(neighbor_cbox, d) -
                  hypre_BoxIMinD(local_cbox, d);
               if (diff > 0)
               {
                  distance = hypre_min(distance, diff);
               }
               if (distance < max_distance)
               {
                  cperimeter_count++;
               }
            }
#else
            perimeter_count = 0;
            cperimeter_count = 1;
#endif
            if (cperimeter_count > perimeter_count)
            {
               if (num_sends == 0)
               {
                  send_procs = hypre_TAlloc(int, num_hood);
                  send_procs[num_sends] = hood_procs[i];
                  num_sends++;
               }
               else if (hood_procs[i] != send_procs[num_sends-1])
               {
                  send_procs[num_sends] = hood_procs[i];
                  num_sends++;
               }
            }
         }
      }
   }

   hypre_BoxDestroy(local_cbox);
   hypre_BoxDestroy(neighbor_cbox);

   /* coarsen neighborhood boxes */
   for (i = 0; i < num_hood; i++)
   {
      box = hypre_BoxArrayBox(hood_boxes, i);
      hypre_StructCoarsenBox(box, index, stride);
   }

#if DEBUG
   fprintf(file, "num_recvs = %d\n", num_recvs);
   for (i = 0; i < num_recvs; i++)
   {
      fprintf(file, "%d ", recv_procs[i]);
   }
   fprintf(file, "\n");
   fprintf(file, "num_sends = %d\n", num_sends);
   for (i = 0; i < num_sends; i++)
   {
      fprintf(file, "%d ", send_procs[i]);
   }
   fprintf(file, "\n");

   fflush(file);
   fclose(file);
#endif

   /*-----------------------------------------
    * Exchange neighbor info with other procs
    *-----------------------------------------*/

   /* neighbor size info - post receives */
   if (num_recvs)
   {
      recv_requests = hypre_TAlloc(MPI_Request, num_recvs);
      recv_status   = hypre_TAlloc(MPI_Status, num_recvs);

      recv_sizes = hypre_TAlloc(int, num_recvs);
      for (i = 0; i < num_recvs; i++)
      {
         MPI_Irecv(&recv_sizes[i], 1, MPI_INT,
                   recv_procs[i], 0, comm, &recv_requests[i]);
      }
   }

   /* neighbor size info - post sends */
   if (num_sends)
   {
      send_requests = hypre_TAlloc(MPI_Request, num_sends);
      send_status   = hypre_TAlloc(MPI_Status, num_sends);

      send_size = 8 * hypre_BoxArraySize(hood_boxes);
      for (i = 0; i < num_sends; i++)
      {
         MPI_Isend(&send_size, 1, MPI_INT,
                   send_procs[i], 0, comm, &send_requests[i]);
      }
   }

   /* neighbor size info - complete receives */
   if (num_recvs)
   {
      MPI_Waitall(num_recvs, recv_requests, recv_status);
   }

   /* neighbor size info - complete sends */
   if (num_sends)
   {
      MPI_Waitall(num_sends, send_requests, send_status);
   }

   /*-----------------------------------------*/

   /* neighbor info - post receives */
   if (num_recvs)
   {
      recv_buffers = hypre_TAlloc(int *, num_recvs);
      for (i = 0; i < num_recvs; i++)
      {
         recv_buffers[i] = hypre_SharedTAlloc(int, recv_sizes[i]);
         MPI_Irecv(recv_buffers[i], recv_sizes[i], MPI_INT,
                   recv_procs[i], 0, comm, &recv_requests[i]);
      }
   }

   /* neighbor info - post sends */
   if (num_sends)
   {
      /* pack the send buffer */
      send_buffer = hypre_SharedTAlloc(int, send_size);
      j = 0;
      for (i = 0; i < num_hood; i++)
      {
         send_buffer[j++] = hood_ids[i];
         send_buffer[j++] = hood_procs[i];
         box = hypre_BoxArrayBox(hood_boxes, i);
         for (d = 0; d < 3; d++)
         {
            send_buffer[j++] = hypre_BoxIMinD(box, d);
            send_buffer[j++] = hypre_BoxIMaxD(box, d);
         }
      }

      for (i = 0; i < num_sends; i++)
      {
         MPI_Isend(send_buffer, send_size, MPI_INT,
                   send_procs[i], 0, comm, &send_requests[i]);
      }
   }

   /* neighbor info - complete receives */
   if (num_recvs)
   {
      MPI_Waitall(num_recvs, recv_requests, recv_status);

      hypre_TFree(recv_requests);
      hypre_TFree(recv_status);
   }

   /* neighbor info - complete sends */
   if (num_sends)
   {
      MPI_Waitall(num_sends, send_requests, send_status);

      hypre_TFree(send_requests);
      hypre_TFree(send_status);
      hypre_TFree(send_buffer);
   }

   /*-----------------------------------------
    * Unpack the recv buffers to create
    * new neighborhood info
    *-----------------------------------------*/

   if (num_recvs)
   {
      alloc_size = num_hood;
      new_hood_boxes = hypre_BoxArrayCreate(alloc_size);
      hypre_BoxArraySetSize(new_hood_boxes, 0);
      new_hood_procs = hypre_TAlloc(int, alloc_size);
      new_hood_ids   = hypre_TAlloc(int, alloc_size);

      box = hypre_BoxCreate();

      j = 0;
      jrecv = hypre_CTAlloc(int, num_recvs);
      new_num_hood = 0;
      while (1)
      {
         data_id = -2;

         /* inspect neighborhood */
         if (j < num_hood)
         {
            if (data_id == -2)
            {
               min_id  = hood_ids[j];
               data_id = -1;
            }
            else if (hood_ids[j] < min_id)
            {
               min_id = hood_ids[j];
               data_id = -1;
            }
            else if (hood_ids[j] == min_id)
            {
               j++;
            }
         }

         /* inspect recv buffer neighborhoods */
         for (i = 0; i < num_recvs; i++)
         {
            jj = jrecv[i];
            if (jj < recv_sizes[i])
            {
               if (data_id == -2)
               {
                  min_id  = recv_buffers[i][jj];
                  data_id = i;
               }
               else if (recv_buffers[i][jj] < min_id)
               {
                  min_id = recv_buffers[i][jj];
                  data_id = i;
               }
               else if (recv_buffers[i][jj] == min_id)
               {
                  jrecv[i] += 8;
               }
            }
         }

         /* put data into new neighborhood structures */
         if (data_id > -2)
         {
            if (new_num_hood == alloc_size)
            {
               alloc_size += num_hood;
               new_hood_procs =
                  hypre_TReAlloc(new_hood_procs, int, alloc_size);
               new_hood_ids =
                  hypre_TReAlloc(new_hood_ids,   int, alloc_size);
            }

            if (data_id == -1)
            {
               /* get data from neighborhood */
               new_hood_procs[new_num_hood] = hood_procs[j];
               new_hood_ids[new_num_hood]   = hood_ids[j];
               hypre_AppendBox(hypre_BoxArrayBox(hood_boxes, j),
                               new_hood_boxes);
               if (j == first_local)
               {
                  new_first_local = new_num_hood;
               }

               j++;
            }
            else
            {
               /* get data from recv buffer neighborhoods */
               jj = jrecv[data_id];
               new_hood_ids[new_num_hood]   = recv_buffers[data_id][jj++];
               new_hood_procs[new_num_hood] = recv_buffers[data_id][jj++];
               for (d = 0; d < 3; d++)
               {
                  hypre_IndexD(imin, d) = recv_buffers[data_id][jj++];
                  hypre_IndexD(imax, d) = recv_buffers[data_id][jj++];
               }
               hypre_BoxSetExtents(box, imin, imax);
               hypre_AppendBox(box, new_hood_boxes);
               jrecv[data_id] = jj;
            }

            new_num_hood++;
         }
         else
         {
            break;
         }
      }

      for (i = 0; i < num_recvs; i++)
      {
         hypre_TFree(recv_buffers[i]);
      }
      hypre_TFree(recv_buffers);
      hypre_TFree(recv_sizes);

      hypre_BoxDestroy(box);
      hypre_TFree(jrecv);

      hypre_BoxArrayDestroy(hood_boxes);
      hypre_TFree(hood_procs);
      hypre_TFree(hood_ids);

      hood_boxes  = new_hood_boxes;
      num_hood    = new_num_hood;
      hood_procs  = new_hood_procs;
      hood_ids    = new_hood_ids;
      first_local = new_first_local;
   }

   hypre_TFree(send_procs);
   hypre_TFree(recv_procs);

   /*-----------------------------------------
    * Eliminate boxes of size zero
    *-----------------------------------------*/

   if (prune)
   {
      j = 0;
      new_first_local = -1;
      new_num_local = 0;
      new_num_periodic = 0;
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
            else if ((i >= first_local + num_local) &&
                     (i <  first_local + num_local + num_periodic))
            {
               new_num_periodic++;
            }
            j++;
         }
      }
      num_hood = j;
      hypre_BoxArraySetSize(hood_boxes, num_hood);
      first_local  = new_first_local;
      num_local    = new_num_local;
      num_periodic = new_num_periodic;
   }

   /*-----------------------------------------
    * Build the coarse grid
    *-----------------------------------------*/

   hypre_StructGridCreate(comm, dim, &cgrid);

   /* set neighborhood */
   hypre_StructGridSetHood(cgrid, hood_boxes, hood_procs, hood_ids,
                           first_local, num_local, num_periodic, bounding_box);

   hypre_StructGridSetHoodInfo(cgrid, max_distance);

   /* set periodicity */
   for (d = 0; d < dim; d++)
   {
      if (hypre_IndexD(periodic, d) > 0)
      {
         hypre_IndexD(periodic, d) =
            hypre_IndexD(periodic, d) / hypre_IndexD(stride, d);
      }
   }
   hypre_StructGridSetPeriodic(cgrid, periodic);

   hypre_StructGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return ierr;
}

#undef hypre_StructCoarsenBox

#else

/*--------------------------------------------------------------------------
 * hypre_StructCoarsen    - TEMPORARY
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
   hypre_Index       periodic;

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

   /* set periodicity */
   hypre_CopyIndex(hypre_StructGridPeriodic(fgrid), periodic);
   for (d = 0; d < dim; d++)
   {
      if (hypre_IndexD(periodic, d) > 0)
      {
         hypre_IndexD(periodic, d) =
            hypre_IndexD(periodic, d) / hypre_IndexD(stride, d);
      }
   }
   hypre_StructGridSetPeriodic(cgrid, periodic);

   hypre_StructGridAssemble(cgrid);

   *cgrid_ptr = cgrid;

   return ierr;
}

#endif
