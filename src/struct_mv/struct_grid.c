/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.11 $
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
int        my_rank;
#endif

/*--------------------------------------------------------------------------
 * hypre_StructGridCreate
 *--------------------------------------------------------------------------*/

int
hypre_StructGridCreate( MPI_Comm           comm,
                        int                dim,
                        hypre_StructGrid **grid_ptr)
{
   hypre_StructGrid    *grid;
   int                 i;

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

   /* additional defaults for the grid ghosts GEC0902  */
   
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
                            first_local, num_local, &neighbors);
   hypre_StructGridNeighbors(grid) = neighbors;

   hypre_BoxDestroy(hypre_StructGridBoundingBox(grid));
   hypre_StructGridBoundingBox(grid) = bounding_box;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructGridAssemble
 *
 * NOTE: Box ids are set here for the non-periodic boxes.  They are
 * globally unique, and appear in increasing order.  The periodic
 * boxes are definedin BoxNeighborsAssemble.
 *
 * NOTE: Box procs are set here.  They appear in non-decreasing order
 * for the non-periodic boxes.
 *
 *--------------------------------------------------------------------------*/

int 
hypre_StructGridAssemble( hypre_StructGrid *grid )
{
   int                  ierr = 0;

   MPI_Comm             comm         = hypre_StructGridComm(grid);
   int                  dim          = hypre_StructGridDim(grid);
   hypre_BoxArray      *boxes        = hypre_StructGridBoxes(grid);
   int                 *ids;
   hypre_BoxNeighbors  *neighbors    = hypre_StructGridNeighbors(grid);
   int                  max_distance = hypre_StructGridMaxDistance(grid);
   hypre_Box           *bounding_box = hypre_StructGridBoundingBox(grid);
   hypre_IndexRef       periodic     = hypre_StructGridPeriodic(grid);

   hypre_Box           *box;
   hypre_BoxArray      *all_boxes;
   int                 *all_procs;
   int                 *all_ids;
   int                  first_local;
   int                  num_local;
   int                  size;
   int                  prune;
   int                  i, d, idmin, idmax;
   /*  GEC  new declarations for the ghost size local  */
   int                  *numghost;
   int                   ghostsize;
   hypre_Box            *ghostbox;

   prune = 1;

   if (neighbors == NULL)
   {
      /* gather grid box info */
      hypre_GatherAllBoxes(comm, boxes, &all_boxes, &all_procs, &first_local);

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
   }

   if (neighbors == NULL)
   {
      /* set all_ids */
      all_ids = hypre_TAlloc(int, hypre_BoxArraySize(all_boxes));
      hypre_ForBoxI(i, all_boxes)
         {
            all_ids[i] = i;
         }

      /* set neighbors */
      num_local = hypre_BoxArraySize(boxes);
      hypre_BoxNeighborsCreate(all_boxes, all_procs, all_ids,
                               first_local, num_local, &neighbors);
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

   hypre_BoxNeighborsAssemble(neighbors, periodic, max_distance, prune);

   /* compute local size */
  
   size = 0;
   ghostsize = 0;
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         size += hypre_BoxVolume(box);         
      }

   hypre_StructGridLocalSize(grid) = size;

 /* GEC0902 expand the box to include the ghosts. Create, copy and expand
  * the ghostbox and finally inserting into the ghlocalsize. As a reminder
  * the boxes variable is the localboxes of the grid (owned by the processor)  
  */

   numghost = hypre_StructGridNumGhost(grid) ;
   ghostsize = 0;
   ghostbox = hypre_BoxCreate();
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         hypre_CopyBox(box, ghostbox);
         hypre_BoxExpand(ghostbox, numghost);         

	 /*        for (d = 0; d < 3; d++)
	  * {
	  *    hypre_BoxIminD(ghostbox, d) -= numghost[2*d];
	  *    hypre_BoxImaxD(ghostbox, d) += numghost[2*d + 1];
	  *  }                                           */

        ghostsize += hypre_BoxVolume(ghostbox);        

      }
   
   hypre_StructGridGhlocalSize(grid) = ghostsize;
   hypre_BoxDestroy(ghostbox);

#if DEBUG
{
   hypre_BoxNeighbors *neighbors;
   int                *procs, *boxnums;
   int                 num_neighbors, i;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   sprintf(filename, "zgrid.%05d", my_rank);

   if ((file = fopen(filename, "a")) == NULL)
   {
      printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   fprintf(file, "\n\n============================\n\n");

   hypre_StructGridPrint(file, grid);

   neighbors = hypre_StructGridNeighbors(grid);
   num_neighbors = hypre_BoxArraySize(hypre_BoxNeighborsBoxes(neighbors));
   procs   = hypre_BoxNeighborsProcs(neighbors);
   boxnums = hypre_BoxNeighborsBoxnums(neighbors);
   fprintf(file, "num_neighbors = %d\n", num_neighbors);
   for (i = 0; i < num_neighbors; i++)
   {
      fprintf(file, "%d: (%d, %d)\n", i, procs[i], boxnums[i]);
   }

   fflush(file);
   fclose(file);
}
#endif
   
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
                     
   int                i, p, b, d;
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
    * Create all_boxes, etc.
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / 7;
   all_boxes   = hypre_BoxArrayCreate(all_boxes_size);
   all_procs   = hypre_TAlloc(int, all_boxes_size);
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

int
hypre_ComputeBoxnums(hypre_BoxArray *boxes,
                     int            *procs,
                     int           **boxnums_ptr)
{
   int                ierr = 0;

   int               *boxnums;
   int                num_boxes;
   int                p, b, boxnum;

   /*-----------------------------------------------------
    *-----------------------------------------------------*/

   num_boxes = hypre_BoxArraySize(boxes);
   boxnums = hypre_TAlloc(int, num_boxes);

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

#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_StructGridAssembleWithAP(grid);
#else
   hypre_StructGridAssemble(grid);
#endif
   *grid_ptr = grid;

   return ierr;
}

/*------------------------------------------------------------------------------
 * GEC0902  hypre_StructGridSetNumGhost
 *
 * the purpose is to set num ghost in the structure grid. It is identical
 * to the function that is used in the structure vector entity.
 *-----------------------------------------------------------------------------*/

int
hypre_StructGridSetNumGhost( hypre_StructGrid *grid, int  *num_ghost )
{
  int  ierr = 0;
  int  i;
  
  for (i = 0; i < 6; i++)
  {
    hypre_StructGridNumGhost(grid)[i] = num_ghost[i];
  }

  return ierr;
}
