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
zzz_NewStructGrid( int dim )
{
   zzz_StructGrid    *grid;

   int      i, size;

   grid = talloc(zzz_StructGrid, 1);

   zzz_StructGridAllBoxes(grid)  = NULL;
   zzz_StructGridProcesses(grid) = NULL;
   zzz_StructGridBoxes(grid)     = zzz_NewBoxArray();
   zzz_StructGridBoxRanks(grid)  = NULL;
   zzz_StructGridDim(grid)       = dim;
   zzz_StructGridGlobalSize(grid)= 0;
   zzz_StructGridLocalSize(grid) = 0;

   return (grid);
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructGrid
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructGrid( zzz_StructGrid *grid )
{
   zzz_FreeBoxArray(zzz_StructGridAllBoxes(grid));
   tfree(zzz_StructGridProcesses(grid));

   /* this box array points to grid boxes in all_boxes */
   zzz_BoxArraySize(zzz_StructGridBoxes(grid)) = 0;
   zzz_FreeBoxArray(zzz_StructGridBoxes(grid));

   tfree(zzz_StructGridBoxRanks(grid));

   tfree(grid);
}

/*--------------------------------------------------------------------------
 * zzz_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
zzz_SetStructGridExtents( zzz_StructGrid  *grid,
		    zzz_Index *ilower,
		    zzz_Index *iupper )
{
   zzz_Index *imin;
   zzz_Index *imax;

   zzz_Box   *box;
   int        d;

   imin = zzz_NewIndex();
   imax = zzz_NewIndex();
   for (d = 0; d < zzz_StructGridDim(grid); d++)
   {
      zzz_IndexD(imin, d) = ilower[d];
      zzz_IndexD(imax, d) = iupper[d];
   }
   for (d = zzz_StructGridDim(grid); d < 3; d++)
   {
      zzz_IndexD(imin, d) = 0;
      zzz_IndexD(imax, d) = 0;
   }

   box = zzz_NewBox(imin, imax);

   zzz_AppendBox(box, zzz_StructGridBoxes(grid));
}

/*--------------------------------------------------------------------------
 * zzz_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
zzz_AssembleStructGrid( zzz_StructGrid *grid )
{
   zzz_BoxArray  *boxes;
   zzz_Box       *box;

   zzz_Index     *imin;
   zzz_Index     *imax;

   int            num_procs, my_rank;

   int           *sendbuf;
   int            sendcount;
   int           *recvbuf;
   int           *recvcounts;
   int           *displs;
   int            recvbuf_size;

   int            i, j, k, b, d;

   boxes = zzz_StructGridBoxes(grid);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* allocate sendbuf */
   sendcount = 7*zzz_BoxArraySize(boxes);
   sendbuf = talloc(int, sendcount);

   /* compute recvcounts and displs */
   recvcounts = talloc(int, num_procs);
   displs = talloc(int, num_procs);
   MPI_Allgather(&sendcount, 1, MPI_INT,
		 recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i-1] + recvcounts[i-1];
      recvbuf_size += recvcounts[i];
   }

   /* allocate recvbuf */
   recvbuf = talloc(int, recvbuf_size);

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
		  recvbuf, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

   /* sort recvbuf by process rank? */

   /* use recvbuf info to set up grid structure */
   zzz_StructGridAllBoxes(grid) = zzz_NewBoxArray();
   zzz_StructGridProcesses(grid) = talloc(int, (recvbuf_size / 7));
   zzz_StructGridBoxes(grid) = zzz_NewBoxArray();
   zzz_StructGridBoxRanks(grid) = talloc(int, zzz_BoxArraySize(boxes));
   zzz_StructGridGlobalSize(grid) = 0;
   zzz_StructGridLocalSize(grid) = 0;
   i = 0;
   j = 0;
   k = 0;
   while (i < recvbuf_size)
   {
      zzz_StructGridProcess(grid, j++) = recvbuf[i++];

      imin = zzz_NewIndex();
      imax = zzz_NewIndex();
      for (d = 0; d < 3; d++)
      {
	 zzz_IndexD(imin, d) = recvbuf[i++];
	 zzz_IndexD(imax, d) = recvbuf[i++];
      }

      box = zzz_NewBox(imin, imax);
      zzz_AppendBox(box, zzz_StructGridAllBoxes(grid));
      if (zzz_StructGridProcess(grid, j-1) == my_rank)
      {
	 zzz_AppendBox(box, zzz_StructGridBoxes(grid));
	 zzz_StructGridBoxRank(grid, k++) = j-1;
         zzz_StructGridLocalSize(grid) += zzz_BoxTotalSize(box);
      }

      zzz_StructGridGlobalSize(grid) += zzz_BoxTotalSize(box);
   }

   zzz_FreeBoxArray(boxes);

   tfree(sendbuf);
   tfree(recvcounts);
   tfree(displs);
   tfree(recvbuf);
}


