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
 * Member functions for zzz_StructGrid class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructGrid
 *--------------------------------------------------------------------------*/

zzz_StructGrid *
zzz_NewStructGrid( MPI_Comm *comm,
		   int       dim  )
{
   zzz_StructGrid    *grid;

   grid = zzz_TAlloc(zzz_StructGrid, 1);

   zzz_StructGridComm(grid)      = comm;
   zzz_StructGridAllBoxes(grid)  = NULL;
   zzz_StructGridProcesses(grid) = NULL;
   zzz_StructGridBoxes(grid)     = zzz_NewBoxArray();
   zzz_StructGridBoxRanks(grid)  = NULL;
   zzz_StructGridDim(grid)       = dim;
   zzz_StructGridGlobalSize(grid)= 0;
   zzz_StructGridLocalSize(grid) = 0;

   return grid;
}

/*--------------------------------------------------------------------------
 * zzz_NewAssembledStructGrid
 *--------------------------------------------------------------------------*/

zzz_StructGrid *
zzz_NewAssembledStructGrid( MPI_Comm      *comm,
                            int            dim,
                            zzz_BoxArray  *all_boxes,
                            int           *processes )
{
   zzz_StructGrid    *grid;

   zzz_BoxArray      *boxes;
   zzz_Box           *box;
   int               *box_ranks;
   int                box_volume;
   int                global_size;
   int                local_size;

   int                i, j, my_rank;

   grid = zzz_TAlloc(zzz_StructGrid, 1);

   MPI_Comm_rank(*comm, &my_rank);

   global_size = 0;
   local_size = 0;
   boxes = zzz_NewBoxArray();
   box_ranks = zzz_TAlloc(int, zzz_BoxArraySize(all_boxes));

   j = 0;
   zzz_ForBoxI(i, all_boxes)
   {
      box = zzz_BoxArrayBox(all_boxes, i);
      box_volume = zzz_BoxVolume(box);

      if (processes[i] == my_rank)
      {
	 zzz_AppendBox(box, boxes);
	 box_ranks[j++] = i;
         local_size += box_volume;
      }

      global_size += box_volume;
   }
   box_ranks = zzz_TReAlloc(box_ranks, int, zzz_BoxArraySize(boxes));

   zzz_StructGridAllBoxes(grid)  = all_boxes;
   zzz_StructGridProcesses(grid) = processes;
   zzz_StructGridBoxes(grid)     = boxes;
   zzz_StructGridBoxRanks(grid)  = box_ranks;
   zzz_StructGridDim(grid)       = dim;
   zzz_StructGridGlobalSize(grid)= global_size;
   zzz_StructGridLocalSize(grid) = local_size;
   zzz_StructGridComm(grid)      = comm;

   return grid;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructGrid
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructGrid( zzz_StructGrid *grid )
{
   zzz_FreeBoxArray(zzz_StructGridAllBoxes(grid));
   zzz_TFree(zzz_StructGridProcesses(grid));

   /* this box array points to grid boxes in all_boxes */
   zzz_FreeBoxArrayShell(zzz_StructGridBoxes(grid));

   zzz_TFree(zzz_StructGridBoxRanks(grid));

   zzz_TFree(grid);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
zzz_SetStructGridExtents( zzz_StructGrid  *grid,
			  zzz_Index        ilower,
			  zzz_Index        iupper )
{
   zzz_Box   *box;

   box = zzz_NewBox(ilower, iupper);

   zzz_AppendBox(box, zzz_StructGridBoxes(grid));
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
zzz_AssembleStructGrid( zzz_StructGrid *grid )
{
   MPI_Comm       *comm = zzz_StructGridComm(grid);

   zzz_StructGrid *new_grid;
   zzz_BoxArray   *all_boxes;
   int            *processes;
   zzz_BoxArray   *boxes;
   zzz_Box        *box;
                  
   zzz_Index       imin;
   zzz_Index       imax;
                  
   int             num_procs, my_rank;
                  
   int            *sendbuf;
   int             sendcount;
   int            *recvbuf;
   int            *recvcounts;
   int            *displs;
   int             recvbuf_size;
                  
   int             i, j, b, d;

   boxes = zzz_StructGridBoxes(grid);

   MPI_Comm_size(*comm, &num_procs);
   MPI_Comm_rank(*comm, &my_rank);

   /* allocate sendbuf */
   sendcount = 7*zzz_BoxArraySize(boxes);
   sendbuf = zzz_TAlloc(int, sendcount);

   /* compute recvcounts and displs */
   recvcounts = zzz_TAlloc(int, num_procs);
   displs = zzz_TAlloc(int, num_procs);
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
   recvbuf = zzz_TAlloc(int, recvbuf_size);

   /* put local box extents and process number into sendbuf */
   i = 0;
   for (b = 0; b < zzz_BoxArraySize(boxes); b++)
   {
      sendbuf[i++] = my_rank;

      box = zzz_BoxArrayBox(boxes, b);
      for (d = 0; d < 3; d++)
      {
	 sendbuf[i++] = zzz_BoxIMinD(box, d);
	 sendbuf[i++] = zzz_BoxIMaxD(box, d);
      }
   }

   /* get global grid info */
   MPI_Allgatherv(sendbuf, sendcount, MPI_INT,
		  recvbuf, recvcounts, displs, MPI_INT, *comm);

   /* sort recvbuf by process rank? */

   /* unpack recvbuf grid info */
   all_boxes = zzz_NewBoxArray();
   processes = zzz_TAlloc(int, (recvbuf_size / 7));
   i = 0;
   j = 0;
   while (i < recvbuf_size)
   {
      processes[j++] = recvbuf[i++];

      for (d = 0; d < 3; d++)
      {
	 zzz_IndexD(imin, d) = recvbuf[i++];
	 zzz_IndexD(imax, d) = recvbuf[i++];
      }

      box = zzz_NewBox(imin, imax);
      zzz_AppendBox(box, all_boxes);
   }

   zzz_FreeBoxArray(boxes);
   zzz_TFree(sendbuf);
   zzz_TFree(recvcounts);
   zzz_TFree(displs);
   zzz_TFree(recvbuf);

   /* complete the grid structure */
   new_grid = zzz_NewAssembledStructGrid(zzz_StructGridComm(grid),
                                         zzz_StructGridDim(grid),
                                         all_boxes, processes);
   zzz_StructGridAllBoxes(grid)  = zzz_StructGridAllBoxes(new_grid);
   zzz_StructGridProcesses(grid) = zzz_StructGridProcesses(new_grid);
   zzz_StructGridBoxes(grid)     = zzz_StructGridBoxes(new_grid);
   zzz_StructGridBoxRanks(grid)  = zzz_StructGridBoxRanks(new_grid);
   zzz_StructGridGlobalSize(grid)= zzz_StructGridGlobalSize(new_grid);
   zzz_StructGridLocalSize(grid) = zzz_StructGridLocalSize(new_grid);
   zzz_TFree(new_grid);
}

/*--------------------------------------------------------------------------
 * zzz_PrintStructGrid
 *--------------------------------------------------------------------------*/
 
void
zzz_PrintStructGrid( FILE           *file,
                     zzz_StructGrid *grid )
{
   zzz_BoxArray    *boxes;
   zzz_Box         *box;

   int              i;

   fprintf(file, "%d\n", zzz_StructGridDim(grid));

   boxes = zzz_StructGridBoxes(grid);
   fprintf(file, "%d\n", zzz_BoxArraySize(boxes));
   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);
      fprintf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n", i,
              zzz_BoxIMinX(box), zzz_BoxIMinY(box), zzz_BoxIMinZ(box),
              zzz_BoxIMaxX(box), zzz_BoxIMaxY(box), zzz_BoxIMaxZ(box));
   }
}

/*--------------------------------------------------------------------------
 * zzz_ReadStructGrid
 *--------------------------------------------------------------------------*/
 
zzz_StructGrid *
zzz_ReadStructGrid( MPI_Comm *comm, FILE *file )
{
   zzz_StructGrid *grid;

   zzz_Index       ilower;
   zzz_Index       iupper;

   int             dim;
   int             num_boxes;

   int             i, idummy;

   fscanf(file, "%d\n", &dim);
   grid = zzz_NewStructGrid(comm, dim);

   fscanf(file, "%d\n", &num_boxes);
   for (i = 0; i < num_boxes; i++)
   {
      fscanf(file, "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n", &idummy,
             &zzz_IndexX(ilower), &zzz_IndexY(ilower), &zzz_IndexZ(ilower),
             &zzz_IndexX(iupper), &zzz_IndexY(iupper), &zzz_IndexZ(iupper));

      zzz_SetStructGridExtents(grid, ilower, iupper);
   }

   zzz_AssembleStructGrid(grid);

   return grid;
}
