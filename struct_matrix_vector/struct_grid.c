/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_StructGrid class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructGridCreate
 *--------------------------------------------------------------------------*/

int
hypre_StructGridCreate( MPI_Comm           comm,
                        int                dim,
                        hypre_StructGrid **grid_ptr)
{
   hypre_StructGrid    *grid;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   hypre_StructGridComm(grid)        = comm;
   hypre_StructGridDim(grid)         = dim;
   hypre_StructGridBoxes(grid)       = hypre_BoxArrayCreate(0);
   hypre_StructGridIDs(grid)         = NULL;
   hypre_StructGridNeighbors(grid)   = NULL;
   hypre_StructGridMaxDistance(grid) = 2;
   hypre_StructGridBoundingBox(grid) = NULL;
   hypre_StructGridLocalSize(grid)   = 0;
   hypre_StructGridGlobalSize(grid)  = 0;
   hypre_SetIndex(hypre_StructGridPeriodic(grid), 0, 0, 0);
   hypre_StructGridRefCount(grid)     = 1;

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
   int ierr = 0;

   if (grid)
   {
      hypre_StructGridRefCount(grid) --;
      if (hypre_StructGridRefCount(grid) == 0)
      {
         hypre_BoxDestroy(hypre_StructGridBoundingBox(grid));
         hypre_BoxNeighborsDestroy(hypre_StructGridNeighbors(grid));
         hypre_TFree(hypre_StructGridIDs(grid));
         hypre_BoxArrayDestroy(hypre_StructGridBoxes(grid));
         hypre_TFree(grid);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetHoodInfo
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetHoodInfo( hypre_StructGrid  *grid,
                             int                max_distance )
{
   int          ierr = 0;

   hypre_StructGridMaxDistance(grid) = max_distance;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetPeriodic
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetPeriodic( hypre_StructGrid  *grid,
                             hypre_Index        periodic)
{
   int          ierr = 0;

   hypre_CopyIndex(periodic, hypre_StructGridPeriodic(grid));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetExtents
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetExtents( hypre_StructGrid  *grid,
                            hypre_Index        ilower,
                            hypre_Index        iupper )
{
   int          ierr = 0;
   hypre_Box   *box;

   box = hypre_BoxCreate();
   hypre_BoxSetExtents(box, ilower, iupper);
   hypre_AppendBox(box, hypre_StructGridBoxes(grid));
   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetBoxes
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetBoxes( hypre_StructGrid *grid,
                          hypre_BoxArray   *boxes )
{
   int ierr = 0;

   hypre_TFree(hypre_StructGridBoxes(grid));
   hypre_StructGridBoxes(grid) = boxes;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridSetHood
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridSetHood( hypre_StructGrid *grid,
                         hypre_BoxArray   *hood_boxes,
                         int              *hood_procs,
                         int              *hood_ids,
                         int               first_local,
                         int               num_local,
                         int               num_periodic,
                         hypre_Box        *bounding_box )
{
   int                  ierr = 0;
                       
   hypre_BoxArray      *boxes;
   int                 *ids;
   hypre_BoxNeighbors  *neighbors;

   int                  i, ilocal;

   boxes = hypre_BoxArrayCreate(num_local);
   ids = hypre_TAlloc(int, num_local);
   for (i = 0; i < num_local; i++)
   {
      ilocal = first_local + i;
      hypre_CopyBox(hypre_BoxArrayBox(hood_boxes, ilocal),
                    hypre_BoxArrayBox(boxes, i));
      ids[i] = hood_ids[ilocal];
   }
   hypre_TFree(hypre_StructGridBoxes(grid));
   hypre_TFree(hypre_StructGridIDs(grid));
   hypre_StructGridBoxes(grid) = boxes;
   hypre_StructGridIDs(grid)   = ids;

   hypre_BoxNeighborsCreate(hood_boxes, hood_procs, hood_ids,
                            first_local, num_local, num_periodic, &neighbors);
   hypre_StructGridNeighbors(grid) = neighbors;

   hypre_BoxDestroy(hypre_StructGridBoundingBox(grid));
   hypre_StructGridBoundingBox(grid) = bounding_box;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridAssemble
 *
 * NOTE: Box ids are set here.  They are globally unique, and appear
 * in increasing order.
 *
 * NOTE: Box procs are set here.  They appear in non-decreasing order.
 *
 * NOTE: The boxes in `all_boxes' appear as follows, for example:
 *
 *   proc:     0 0 0 0 1 1 2 2 2 2 ...
 *   ID:       0 1 2 3 4 5 6 7 8 9 ...
 *   periodic:     * *   *     * *
 *
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridAssemble( hypre_StructGrid *grid )
{
   int                  ierr = 0;

   hypre_BoxArray      *boxes;
   hypre_Box           *box;
   int                  size;
   int                  prune;
   int                  i;

   boxes = hypre_StructGridBoxes(grid);
   prune = 1;

   if (hypre_StructGridNeighbors(grid) == NULL)
   {
      MPI_Comm            comm = hypre_StructGridComm(grid);
      int                 dim  = hypre_StructGridDim(grid);
      int                *ids;
      hypre_BoxNeighbors *neighbors;
      hypre_Box          *bounding_box;
                         
      hypre_BoxArray     *all_boxes;
      int                *all_procs;
      int                *all_ids;
      int                 first_local;
      int                 num_local;
      int                 num_periodic;
                         
      int                 d, idmin, idmax;

      /* gather grid box info */
      hypre_GatherAllBoxes(comm, boxes, &all_boxes, &all_procs, &first_local);
      num_local = hypre_BoxArraySize(boxes);

      /* set bounding box */
      bounding_box = hypre_BoxCreate();
      for (d = 0; d < dim; d++)
      {
         idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), d);
         idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), d);
         hypre_ForBoxI(i, all_boxes)
            {
               box = hypre_BoxArrayBox(all_boxes, i);
               idmin = hypre_min(idmin, hypre_BoxIMinD(box, d));
               idmax = hypre_max(idmax, hypre_BoxIMaxD(box, d));
            }
         hypre_BoxIMinD(bounding_box, d) = idmin;
         hypre_BoxIMaxD(bounding_box, d) = idmax;
      }
      for (d = dim; d < 3; d++)
      {
         hypre_BoxIMinD(bounding_box, d) = 0;
         hypre_BoxIMaxD(bounding_box, d) = 0;
      }
      hypre_StructGridBoundingBox(grid) = bounding_box;

      /* set global size */
      size = 0;
      hypre_ForBoxI(i, all_boxes)
         {
            box = hypre_BoxArrayBox(all_boxes, i);
            size += hypre_BoxVolume(box);
         }
      hypre_StructGridGlobalSize(grid) = size;

      /* modify all_boxes as required for periodicity */
      hypre_StructGridPeriodicAllBoxes(grid, &all_boxes, &all_procs,
                                       &first_local, &num_periodic);

      /* set all_ids */
      all_ids = hypre_TAlloc(int, hypre_BoxArraySize(all_boxes));
      hypre_ForBoxI(i, all_boxes)
         {
            all_ids[i] = i;
         }

      /* set neighbors */
      hypre_BoxNeighborsCreate(all_boxes, all_procs, all_ids,
                               first_local, num_local, num_periodic,
                               &neighbors);
      hypre_StructGridNeighbors(grid) = neighbors;

      /* set ids */
      ids = hypre_TAlloc(int, hypre_BoxArraySize(boxes));
      hypre_ForBoxI(i, boxes)
         {
            ids[i] = all_ids[first_local + i];
         }
      hypre_StructGridIDs(grid) = ids;

      prune = 1;
   }

   hypre_BoxNeighborsAssemble(hypre_StructGridNeighbors(grid),
                              hypre_StructGridMaxDistance(grid), prune);

   /* compute local size */
   size = 0;
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         size += hypre_BoxVolume(box);
      }
   hypre_StructGridLocalSize(grid) = size;

   return ierr;
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
                     
   int                i, p, b, ab, d;
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
    * Create all_boxes, all_procs, and first_local
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / 7;
   all_boxes = hypre_BoxArrayCreate(all_boxes_size);
   all_procs = hypre_TAlloc(int, all_boxes_size);
   first_local = -1;
   i  = 0;
   p  = 0;
   ab = 0;
   box = hypre_BoxCreate();
   while (i < recvbuf_size)
   {
      all_procs[p] = recvbuf[i++];
      for (d = 0; d < 3; d++)
      {
         hypre_IndexD(imin, d) = recvbuf[i++];
         hypre_IndexD(imax, d) = recvbuf[i++];
      }
      hypre_BoxSetExtents(box, imin, imax);
      hypre_CopyBox(box, hypre_BoxArrayBox(all_boxes, ab));
      ab++;

      if ((first_local < 0) && (all_procs[p] == my_rank))
      {
         first_local = p;
      }

      p++;
   }
   hypre_BoxDestroy(box);

   /*-----------------------------------------------------
    * Return
    *-----------------------------------------------------*/

   hypre_TFree(sendbuf);
   hypre_SharedTFree(recvbuf);
   hypre_SharedTFree(recvcounts);
   hypre_TFree(displs);

   *all_boxes_ptr = all_boxes;
   *all_procs_ptr = all_procs;
   *first_local_ptr = first_local;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridPrint
 *--------------------------------------------------------------------------*/
 
int
hypre_StructGridPrint( FILE             *file,
                       hypre_StructGrid *grid )
{
   int              ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridRead
 *--------------------------------------------------------------------------*/
 
int
hypre_StructGridRead( MPI_Comm           comm,
                      FILE              *file,
                      hypre_StructGrid **grid_ptr )
{
   int ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridPeriodicAllBoxes
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridPeriodicAllBoxes( hypre_StructGrid  *grid,
                                  hypre_BoxArray   **all_boxes_ptr,
                                  int              **all_procs_ptr,
                                  int               *first_local_ptr,
                                  int               *num_periodic_ptr )
{
   int               ierr = 0;

   int               new_num_periodic = 0;

   int               px = hypre_IndexX(hypre_StructGridPeriodic(grid));
   int               py = hypre_IndexY(hypre_StructGridPeriodic(grid));
   int               pz = hypre_IndexZ(hypre_StructGridPeriodic(grid));

   int               i_periodic = 0;
   int               j_periodic = 0;
   int               k_periodic = 0;
      
   if (px != 0)
      i_periodic = 1;
   if (py != 0)
      j_periodic = 1;
   if (pz != 0)
      k_periodic = 1;

   if( !(i_periodic == 0 && j_periodic == 0 && k_periodic == 0) )
   {
      hypre_BoxArray   *new_all_boxes;
      int              *new_all_procs;
      int               new_first_local;

      hypre_BoxArray   *all_boxes   = *all_boxes_ptr;
      int              *all_procs   = *all_procs_ptr;
      int               first_local = *first_local_ptr;
      int               num_local;
      int               num_periodic;

      hypre_Box        *box;
      
      int               num_all, new_num_all;
      int               i, inew, ip, jp, kp;
      int               first_i, first_inew;

      num_all  = hypre_BoxArraySize(all_boxes);
      new_num_all = num_all * ((1+2*i_periodic) *
                               (1+2*j_periodic) *
                               (1+2*k_periodic));

      new_all_boxes = hypre_BoxArrayCreate(new_num_all);
      new_all_procs = hypre_TAlloc(int, new_num_all);

      /* add boxes required for periodicity */
      i = 0;
      inew = 0;
      while (i < num_all)
      {
         first_i    = i;
         first_inew = inew;

         for (i = first_i; i < num_all; i++)
         {
            if (all_procs[i] != all_procs[first_i])
            {
               break;
            }

            hypre_CopyBox(hypre_BoxArrayBox(all_boxes, i),
                          hypre_BoxArrayBox(new_all_boxes, inew));
            new_all_procs[inew] = all_procs[i];

            inew++;
         }
         num_local = i - first_i;

         for (ip = -i_periodic; ip <= i_periodic; ip++)
         {
            for (jp = -j_periodic; jp <= j_periodic; jp++)
            {
               for (kp = -k_periodic; kp <= k_periodic; kp++)
               {
                  if( !(ip == 0 && jp == 0 && kp == 0) )
                  {
                     for (i = first_i; i < (first_i + num_local); i++)
                     {
                        box = hypre_BoxArrayBox(new_all_boxes, inew);
                        hypre_CopyBox(hypre_BoxArrayBox(all_boxes, i), box);
                        
                        /* shift box */
                        hypre_BoxIMinD(box, 0) =
                           hypre_BoxIMinD(box, 0) + (ip * px);
                        hypre_BoxIMinD(box, 1) =
                           hypre_BoxIMinD(box, 1) + (jp * py);
                        hypre_BoxIMinD(box, 2) =
                           hypre_BoxIMinD(box, 2) + (kp * pz);
                        hypre_BoxIMaxD(box, 0) =
                           hypre_BoxIMaxD(box, 0) + (ip * px);
                        hypre_BoxIMaxD(box, 1) =
                           hypre_BoxIMaxD(box, 1) + (jp * py);
                        hypre_BoxIMaxD(box, 2) =
                           hypre_BoxIMaxD(box, 2) + (kp * pz);

                        new_all_procs[inew] = all_procs[i];

                        inew++;
                     }
                  }
               }
            }
         }
         num_periodic = inew - first_inew - num_local;

         if (first_i == first_local)
         {
            new_first_local  = first_inew;
            new_num_periodic = num_periodic;
         }
      }

      hypre_BoxArraySetSize(new_all_boxes, inew);

      hypre_BoxArrayDestroy(all_boxes);
      hypre_TFree(all_procs);

      *all_boxes_ptr   = new_all_boxes;
      *all_procs_ptr   = new_all_procs;
      *first_local_ptr = new_first_local;
   }

   *num_periodic_ptr = new_num_periodic;

   return ierr;
}
