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
 * hypre_NewStructGrid
 *--------------------------------------------------------------------------*/

hypre_StructGrid *
hypre_NewStructGrid( MPI_Comm  comm,
                     int       dim  )
{
   hypre_StructGrid    *grid;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   hypre_StructGridComm(grid)        = comm;
   hypre_StructGridBoxes(grid)       = NULL;
   hypre_StructGridDim(grid)         = dim;
   hypre_StructGridGlobalSize(grid)  = 0;
   hypre_StructGridLocalSize(grid)   = 0;
   hypre_StructGridNeighbors(grid)   = NULL;
   hypre_StructGridMaxDistance(grid) = 5;
   hypre_SetIndex(hypre_StructGridPeriodic(grid), 0, 0, 0);
   hypre_StructGridAllBoxes(grid)     = NULL;
   hypre_StructGridProcesses(grid)    = NULL;
   hypre_StructGridBoxRanks(grid)     = NULL;
   hypre_StructGridBaseAllBoxes(grid) = NULL;
   hypre_SetIndex(hypre_StructGridPIndex(grid), 0, 0, 0);
   hypre_SetIndex(hypre_StructGridPStride(grid), 1, 1, 1);
   hypre_StructGridAlloced(grid) = 0;

   return grid;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructGrid
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructGrid( hypre_StructGrid *grid )
{
   int ierr = 0;

   if (grid)
   {
      if (hypre_StructGridAlloced(grid))
      {
         hypre_FreeBoxArray(hypre_StructGridAllBoxes(grid));
         hypre_TFree(hypre_StructGridProcesses(grid));
         hypre_TFree(hypre_StructGridBoxRanks(grid));
         hypre_FreeBoxArray(hypre_StructGridBaseAllBoxes(grid));
      }
      hypre_FreeBoxNeighbors(hypre_StructGridNeighbors(grid));
      hypre_FreeBoxArray(hypre_StructGridBoxes(grid));
      hypre_TFree(grid);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridPeriodic
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructGridPeriodic( hypre_StructGrid  *grid,
                             hypre_Index        periodic)
{
   int          ierr = 0;

   hypre_CopyIndex(periodic, hypre_StructGridPeriodic(grid));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridExtents
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructGridExtents( hypre_StructGrid  *grid,
                            hypre_Index        ilower,
                            hypre_Index        iupper )
{
   int          ierr = 0;
   hypre_Box   *box;

   if (hypre_StructGridBoxes(grid) == NULL)
   {
      hypre_StructGridBoxes(grid) = hypre_NewBoxArray(0);
   }

   box = hypre_NewBox();
   hypre_SetBoxExtents(box, ilower, iupper);
   hypre_AppendBox(box, hypre_StructGridBoxes(grid));
   hypre_FreeBox(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridBoxes
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructGridBoxes( hypre_StructGrid  *grid,
                          hypre_BoxArray    *boxes )
{
   int ierr = 0;

   hypre_StructGridBoxes(grid) = boxes;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridGlobalInfo
 *--------------------------------------------------------------------------*/

int 
hypre_SetStructGridGlobalInfo( hypre_StructGrid  *grid,
                               hypre_BoxArray    *all_boxes,
                               int               *processes,
                               int               *box_ranks,
                               hypre_BoxArray    *base_all_boxes,
                               hypre_Index        pindex,
                               hypre_Index        pstride        )
{
   int ierr = 0;

   hypre_StructGridAllBoxes(grid)     = all_boxes;
   hypre_StructGridProcesses(grid)    = processes;
   hypre_StructGridBoxRanks(grid)     = box_ranks;
   hypre_StructGridBaseAllBoxes(grid) = base_all_boxes;
   hypre_CopyIndex(pindex,  hypre_StructGridPIndex(grid));
   hypre_CopyIndex(pstride, hypre_StructGridPStride(grid));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructGrid
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleStructGrid( hypre_StructGrid *grid )
{
   int                  ierr = 0;

   MPI_Comm             comm  = hypre_StructGridComm(grid);
   hypre_BoxArray      *boxes = hypre_StructGridBoxes(grid);
   int                  global_size;
   int                  local_size;
   hypre_BoxNeighbors  *neighbors;

   hypre_BoxArray      *all_boxes;
   int                 *processes;
   int                 *box_ranks;

   hypre_Box           *box;
   int                  i;
                     
   boxes = hypre_StructGridBoxes(grid);

   if (hypre_StructGridAllBoxes(grid) == NULL)
   {
      hypre_GatherAllBoxes(comm, boxes,
                           &hypre_StructGridAllBoxes(grid),
                           &hypre_StructGridProcesses(grid),
                           &hypre_StructGridBoxRanks(grid));
      hypre_StructGridBaseAllBoxes(grid) =
         hypre_DuplicateBoxArray(hypre_StructGridAllBoxes(grid));
      hypre_StructGridAlloced(grid) = 1;
   }

   all_boxes = hypre_StructGridAllBoxes(grid);
   processes = hypre_StructGridProcesses(grid);
   box_ranks = hypre_StructGridBoxRanks(grid);

   /* compute global_size */
   global_size = 0;
   hypre_ForBoxI(i, all_boxes)
      {
         box = hypre_BoxArrayBox(all_boxes, i);
         global_size += hypre_BoxVolume(box);
      }
   hypre_StructGridGlobalSize(grid) = global_size;

   /* compute local_size */
   local_size = 0;
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         local_size += hypre_BoxVolume(box);
      }
   hypre_StructGridLocalSize(grid) = local_size;

   /* compute neighbors */
   neighbors = hypre_StructGridNeighbors(grid);
   hypre_FreeBoxNeighbors(hypre_StructGridNeighbors(grid));
   hypre_StructGridNeighbors(grid) =
      hypre_NewBoxNeighbors( box_ranks, hypre_BoxArraySize(boxes),
                             all_boxes, processes,
                             hypre_StructGridMaxDistance(grid));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GatherAllBoxes
 *--------------------------------------------------------------------------*/

int
hypre_GatherAllBoxes( MPI_Comm         comm,
                      hypre_BoxArray  *boxes,
                      hypre_BoxArray **all_boxes_ptr,
                      int            **processes_ptr,
                      int            **box_ranks_ptr )
{
   hypre_BoxArray    *all_boxes;
   int               *processes;
   int               *box_ranks;
   int                all_boxes_size;

   hypre_Box         *box;
   hypre_Index        imin;
   hypre_Index        imax;
                     
   int                num_procs, my_rank;
                     
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
   
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_rank);

   /* compute recvcounts and displs */
   sendcount = 7*hypre_BoxArraySize(boxes);
   recvcounts = hypre_SharedTAlloc(int, num_procs);
   displs = hypre_TAlloc(int, num_procs);
   MPI_Allgather(&sendcount, 1, MPI_INT,
		 recvcounts, 1, MPI_INT, comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (p = 1; p < num_procs; p++)
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
    * Create all_boxes, processes, and box_ranks
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / 7;
   all_boxes = hypre_NewBoxArray(all_boxes_size);
   processes = hypre_TAlloc(int, all_boxes_size);
   box_ranks = hypre_TAlloc(int, hypre_BoxArraySize(boxes));
   i  = 0;
   p  = 0;
   b  = 0;
   ab = 0;
   box = hypre_NewBox();
   while (i < recvbuf_size)
   {
      processes[p] = recvbuf[i++];
      for (d = 0; d < 3; d++)
      {
	 hypre_IndexD(imin, d) = recvbuf[i++];
	 hypre_IndexD(imax, d) = recvbuf[i++];
      }
      hypre_SetBoxExtents(box, imin, imax);
      hypre_CopyBox(box, hypre_BoxArrayBox(all_boxes, ab));
      ab++;

      if (processes[p] == my_rank)
      {
         box_ranks[b] = p;
         b++;
      }

      p++;
   }
   hypre_FreeBox(box);

   /*-----------------------------------------------------
    * Return
    *-----------------------------------------------------*/

   hypre_TFree(sendbuf);
   hypre_SharedTFree(recvbuf);
   hypre_SharedTFree(recvcounts);
   hypre_TFree(displs);

   *all_boxes_ptr = all_boxes;
   *processes_ptr = processes;
   *box_ranks_ptr = box_ranks;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructGrid
 *--------------------------------------------------------------------------*/
 
int
hypre_PrintStructGrid( FILE             *file,
                       hypre_StructGrid *grid )
{
   int                ierr = 0;

   hypre_BoxArray    *boxes;
   hypre_Box         *box;

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
 * hypre_ReadStructGrid
 *--------------------------------------------------------------------------*/
 
hypre_StructGrid *
hypre_ReadStructGrid( MPI_Comm  comm,
                      FILE     *file )
{
   hypre_StructGrid *grid;

   hypre_Index       ilower;
   hypre_Index       iupper;

   int               dim;
   int               num_boxes;
               
   int               i, idummy;

   fscanf(file, "%d\n", &dim);
   grid = hypre_NewStructGrid(comm, dim);

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

      hypre_SetStructGridExtents(grid, ilower, iupper);
   }

   hypre_AssembleStructGrid(grid);

   return grid;
}

