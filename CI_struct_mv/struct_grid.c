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
hypre_NewStructGrid( MPI_Comm context, int dim )
{
   hypre_StructGrid    *grid;

   int      i, size;

   grid = hypre_CTAlloc(hypre_StructGrid, 1);

   hypre_StructGridContext(grid)  = context;
   hypre_StructGridAllBoxes(grid)  = NULL;
   hypre_StructGridProcesses(grid) = NULL;
   hypre_StructGridBoxes(grid)     = hypre_NewBoxArray();
   hypre_StructGridBoxRanks(grid)  = NULL;
   hypre_StructGridDim(grid)       = dim;
   hypre_StructGridGlobalSize(grid)= 0;
   hypre_StructGridLocalSize(grid) = 0;

   return (grid);
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
   hypre_BoxArraySize(hypre_StructGridBoxes(grid)) = 0;
   hypre_FreeBoxArray(hypre_StructGridBoxes(grid));

   hypre_TFree(hypre_StructGridBoxRanks(grid));

   hypre_TFree(grid);
}

/*--------------------------------------------------------------------------
 * hypre_SetStructGridExtents
 *--------------------------------------------------------------------------*/

void 
hypre_SetStructGridExtents( hypre_StructGrid  *grid,
		    hypre_Index *ilower,
		    hypre_Index *iupper )
{
   hypre_Index *imin;
   hypre_Index *imax;

   hypre_Box   *box;
   int        d;

   imin = hypre_NewIndex();
   imax = hypre_NewIndex();
   for (d = 0; d < hypre_StructGridDim(grid); d++)
   {
      hypre_IndexD(imin, d) = ilower[d];
      hypre_IndexD(imax, d) = iupper[d];
   }
   for (d = hypre_StructGridDim(grid); d < 3; d++)
   {
      hypre_IndexD(imin, d) = 0;
      hypre_IndexD(imax, d) = 0;
   }

   box = hypre_NewBox(imin, imax);

   hypre_AppendBox(box, hypre_StructGridBoxes(grid));
}

/*--------------------------------------------------------------------------
 * hypre_AssembleStructGrid
 *--------------------------------------------------------------------------*/

void 
hypre_AssembleStructGrid( hypre_StructGrid *grid )
{
   hypre_BoxArray  *boxes;
   hypre_Box       *box;

   hypre_Index     *imin;
   hypre_Index     *imax;

   int            num_procs, my_rank;

   int           *sendbuf;
   int            sendcount;
   int           *recvbuf;
   int           *recvcounts;
   int           *displs;
   int            recvbuf_size;

   int            i, j, k, b, d;

   boxes = hypre_StructGridBoxes(grid);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* allocate sendbuf */
   sendcount = 7*hypre_BoxArraySize(boxes);
   sendbuf = hypre_CTAlloc(int, sendcount);

   /* compute recvcounts and displs */
   recvcounts = hypre_CTAlloc(int, num_procs);
   displs = hypre_CTAlloc(int, num_procs);
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
   recvbuf = hypre_CTAlloc(int, recvbuf_size);

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
		  recvbuf, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

   /* sort recvbuf by process rank? */

   /* use recvbuf info to set up grid structure */
   hypre_StructGridAllBoxes(grid) = hypre_NewBoxArray();
   hypre_StructGridProcesses(grid) = hypre_CTAlloc(int, (recvbuf_size / 7));
   hypre_StructGridBoxes(grid) = hypre_NewBoxArray();
   hypre_StructGridBoxRanks(grid) = hypre_CTAlloc(int, hypre_BoxArraySize(boxes));
   hypre_StructGridGlobalSize(grid) = 0;
   hypre_StructGridLocalSize(grid) = 0;
   i = 0;
   j = 0;
   k = 0;
   while (i < recvbuf_size)
   {
      hypre_StructGridProcess(grid, j++) = recvbuf[i++];

      imin = hypre_NewIndex();
      imax = hypre_NewIndex();
      for (d = 0; d < 3; d++)
      {
	 hypre_IndexD(imin, d) = recvbuf[i++];
	 hypre_IndexD(imax, d) = recvbuf[i++];
      }

      box = hypre_NewBox(imin, imax);
      hypre_AppendBox(box, hypre_StructGridAllBoxes(grid));
      if (hypre_StructGridProcess(grid, j-1) == my_rank)
      {
	 hypre_AppendBox(box, hypre_StructGridBoxes(grid));
	 hypre_StructGridBoxRank(grid, k++) = j-1;
         hypre_StructGridLocalSize(grid) += hypre_BoxTotalSize(box);
      }

      hypre_StructGridGlobalSize(grid) += hypre_BoxTotalSize(box);
   }

   hypre_FreeBoxArray(boxes);

   hypre_TFree(sendbuf);
   hypre_TFree(recvcounts);
   hypre_TFree(displs);
   hypre_TFree(recvbuf);
}


