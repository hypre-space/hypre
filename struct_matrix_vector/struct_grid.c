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
hypre_NewStructGrid( MPI_Comm *comm,
		   int       dim  )
{
   hypre_StructGrid    *grid;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   hypre_StructGridComm(grid)      = comm;
   hypre_StructGridAllBoxes(grid)  = NULL;
   hypre_StructGridProcesses(grid) = NULL;
   hypre_StructGridBoxes(grid)     = hypre_NewBoxArray();
   hypre_StructGridBoxRanks(grid)  = NULL;
   hypre_StructGridDim(grid)       = dim;
   hypre_StructGridGlobalSize(grid)= 0;
   hypre_StructGridLocalSize(grid) = 0;

   return grid;
}

/*--------------------------------------------------------------------------
 * hypre_NewAssembledStructGrid
 *--------------------------------------------------------------------------*/

hypre_StructGrid *
hypre_NewAssembledStructGrid( MPI_Comm      *comm,
                            int            dim,
                            hypre_BoxArray  *all_boxes,
                            int           *processes )
{
   hypre_StructGrid    *grid;

   hypre_BoxArray      *boxes;
   hypre_Box           *box;
   int               *box_ranks;
   int                box_volume;
   int                global_size;
   int                local_size;

   int                i, j, my_rank;

   grid = hypre_TAlloc(hypre_StructGrid, 1);

   MPI_Comm_rank(*comm, &my_rank);

   global_size = 0;
   local_size = 0;
   boxes = hypre_NewBoxArray();
   box_ranks = hypre_TAlloc(int, hypre_BoxArraySize(all_boxes));

   j = 0;
   hypre_ForBoxI(i, all_boxes)
   {
      box = hypre_BoxArrayBox(all_boxes, i);
      box_volume = hypre_BoxVolume(box);

      if (processes[i] == my_rank)
      {
	 hypre_AppendBox(box, boxes);
	 box_ranks[j++] = i;
         local_size += box_volume;
      }

      global_size += box_volume;
   }
   box_ranks = hypre_TReAlloc(box_ranks, int, hypre_BoxArraySize(boxes));

   hypre_StructGridAllBoxes(grid)  = all_boxes;
   hypre_StructGridProcesses(grid) = processes;
   hypre_StructGridBoxes(grid)     = boxes;
   hypre_StructGridBoxRanks(grid)  = box_ranks;
   hypre_StructGridDim(grid)       = dim;
   hypre_StructGridGlobalSize(grid)= global_size;
   hypre_StructGridLocalSize(grid) = local_size;
   hypre_StructGridComm(grid)      = comm;

   return grid;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructGrid
 *--------------------------------------------------------------------------*/

void 
hypre_FreeStructGrid( hypre_StructGrid *grid )
{
   hypre_FreeBoxArray(hypre_StructGridAllBoxes(grid));
   hypre_TFree(hypre_StructGridProcesses(grid));

   /* this box array points to grid boxes in all_boxes */
   hypre_FreeBoxArrayShell(hypre_StructGridBoxes(grid));

   hypre_TFree(hypre_StructGridBoxRanks(grid));

   hypre_TFree(grid);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
hypre_SetStructGridExtents( hypre_StructGrid  *grid,
			  hypre_Index        ilower,
			  hypre_Index        iupper )
{
   hypre_Box   *box;

   box = hypre_NewBox(ilower, iupper);

   hypre_AppendBox(box, hypre_StructGridBoxes(grid));
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
hypre_AssembleStructGrid( hypre_StructGrid *grid )
{
   MPI_Comm       *comm = hypre_StructGridComm(grid);

   hypre_StructGrid *new_grid;
   hypre_BoxArray   *all_boxes;
   int            *processes;
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
                  
   hypre_Index       imin;
   hypre_Index       imax;
                  
   int             num_procs, my_rank;
                  
   int            *sendbuf;
   int             sendcount;
   int            *recvbuf;
   int            *recvcounts;
   int            *displs;
   int             recvbuf_size;
                  
   int             i, j, b, d;

   boxes = hypre_StructGridBoxes(grid);

   MPI_Comm_size(*comm, &num_procs);
   MPI_Comm_rank(*comm, &my_rank);

   /* allocate sendbuf */
   sendcount = 7*hypre_BoxArraySize(boxes);
   sendbuf = hypre_TAlloc(int, sendcount);

   /* compute recvcounts and displs */
   recvcounts = hypre_TAlloc(int, num_procs);
   displs = hypre_TAlloc(int, num_procs);
   MPI_Allgather(&sendcount, 1, MPI_INT,
		 recvcounts, 1, MPI_INT, *comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i-1] + recvcounts[i-1];
      recvbuf_size += recvcounts[i];
   }

   /* allocate recvbuf */
   recvbuf = hypre_TAlloc(int, recvbuf_size);

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
		  recvbuf, recvcounts, displs, MPI_INT, *comm);

   /* sort recvbuf by process rank? */

   /* unpack recvbuf grid info */
   all_boxes = hypre_NewBoxArray();
   processes = hypre_TAlloc(int, (recvbuf_size / 7));
   i = 0;
   j = 0;
   while (i < recvbuf_size)
   {
      processes[j++] = recvbuf[i++];

      for (d = 0; d < 3; d++)
      {
	 hypre_IndexD(imin, d) = recvbuf[i++];
	 hypre_IndexD(imax, d) = recvbuf[i++];
      }

      box = hypre_NewBox(imin, imax);
      hypre_AppendBox(box, all_boxes);
   }

   hypre_FreeBoxArray(boxes);
   hypre_TFree(sendbuf);
   hypre_TFree(recvcounts);
   hypre_TFree(displs);
   hypre_TFree(recvbuf);

   /* complete the grid structure */
   new_grid = hypre_NewAssembledStructGrid(hypre_StructGridComm(grid),
                                         hypre_StructGridDim(grid),
                                         all_boxes, processes);
   hypre_StructGridAllBoxes(grid)  = hypre_StructGridAllBoxes(new_grid);
   hypre_StructGridProcesses(grid) = hypre_StructGridProcesses(new_grid);
   hypre_StructGridBoxes(grid)     = hypre_StructGridBoxes(new_grid);
   hypre_StructGridBoxRanks(grid)  = hypre_StructGridBoxRanks(new_grid);
   hypre_StructGridGlobalSize(grid)= hypre_StructGridGlobalSize(new_grid);
   hypre_StructGridLocalSize(grid) = hypre_StructGridLocalSize(new_grid);
   hypre_TFree(new_grid);
}

/*--------------------------------------------------------------------------
 * hypre_PrintStructGrid
 *--------------------------------------------------------------------------*/
 
void
hypre_PrintStructGrid( FILE           *file,
                     hypre_StructGrid *grid )
{
   hypre_BoxArray    *boxes;
   hypre_Box         *box;

   int              i;

   fprintf(file, "%d\n", hypre_StructGridDim(grid));

   boxes = hypre_StructGridBoxes(grid);
   fprintf(file, "%d\n", hypre_BoxArraySize(boxes));
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      fprintf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n", i,
              hypre_BoxIMinX(box), hypre_BoxIMinY(box), hypre_BoxIMinZ(box),
              hypre_BoxIMaxX(box), hypre_BoxIMaxY(box), hypre_BoxIMaxZ(box));
   }
}

/*--------------------------------------------------------------------------
 * hypre_ReadStructGrid
 *--------------------------------------------------------------------------*/
 
hypre_StructGrid *
hypre_ReadStructGrid( MPI_Comm *comm, FILE *file )
{
   hypre_StructGrid *grid;

   hypre_Index       ilower;
   hypre_Index       iupper;

   int             dim;
   int             num_boxes;

   int             i, idummy;

   fscanf(file, "%d\n", &dim);
   grid = hypre_NewStructGrid(comm, dim);

   fscanf(file, "%d\n", &num_boxes);
   for (i = 0; i < num_boxes; i++)
   {
      fscanf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n", &idummy,
             &hypre_IndexX(ilower), &hypre_IndexY(ilower), &hypre_IndexZ(ilower),
             &hypre_IndexX(iupper), &hypre_IndexY(iupper), &hypre_IndexZ(iupper));

      hypre_SetStructGridExtents(grid, ilower, iupper);
   }

   hypre_AssembleStructGrid(grid);

   return grid;
}
