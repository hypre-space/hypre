/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 * 
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for a given
 * grid-stencil computation.  These patterns are defined by
 * intersecting the data dependencies of each box (including data
 * dependencies within the box) with its neighbor boxes.
 *
 * A consistent ordering of the boxes in the send/recv data regions is
 * returned (given by the send_order and recv_order arrays).  That is,
 * the ordering of the boxes on process p for receives from process q
 * is guaranteed to be the same as the ordering of the boxes on
 * process q for sends to process p.
 *
 * The routine uses a grow-the-box-and-intersect-with-neighbors style
 * algorithm, which helps in generating pre-sorted information and
 * eliminates the need for the UnionBoxes routine which was used in
 * previous implementations.
 *
 * 1. The basic algorithm:
 *
 * The algorithm relies on the assumption that the nearby neighbors of
 * each local box are accessed as a single pre-sorted (by ID) list.
 * The basic algorithm is as follows, with one additional optimization
 * discussed below that helps to minimize the number of communications
 * that are done with neighbors (e.g., consider a 7-pt stencil and the
 * difference between doing 26 communications versus 6):
 *
 * To compute send/recv regions, do
 * 
 *   for i = local box
 *   {
 *      // receives
 *      for j = neighbor box of i
 *      {
 *         gbox = grow box i according to stencil
 *         intersect gbox with box j and add to recv region
 *      }
 * 
 *      // sends
 *      for j = neighbor box of i
 *      {
 *         gbox = grow box j according to stencil
 *         intersect gbox with box i and add to send region
 *      }
 *   }
 * 
 *   sort the send boxes by j index first (this can be done cheaply)
 * 
 * 2. Optimization on basic algorithm:
 * 
 * Before looping over the neighbors in the above algorithm, do a
 * preliminary sweep through the neighbors to select a subset of
 * neighbors to do the intersections with.  To select the subset,
 * compute a so-called "distance index" and check the corresponding
 * entry in the so-called "stencil grid" to decide whether or not to
 * use the box.
 * 
 * The "stencil grid" is a 3x3x3 grid in 3D that is built from the
 * stencil as follows:
 * 
 *   // assume for simplicity that i,j,k are -1, 0, or 1
 *   for each stencil entry (i,j,k)
 *   {
 *      mark all stencil grid entries in (1,1,1) x (1+i,1+j,1+k)
 *      // here (1,1,1) is the "center" entry in the stencil grid
 *   }
 * 
 * 3. Complications with periodicity:
 * 
 * When periodicity is on, it is possible to have a box-pair region
 * (the description of a communication pattern between two boxes) that
 * consists of more than one box.  To produce a consistent ordering on
 * these lists of boxes, the algorithm runs through the periodic part
 * of the recv boxes in one direction and the periodic part of the
 * send boxes in the opposite direction.  For example, consider a
 * box-pair region of size 3 for boxes i and j.  The following are
 * equivalent (here, each box is tagged according to the
 * (i,period),(j,period) pair that produced it):
 * 
 *   from i's point of view              from j's point of view
 *       (i,0),(j,0)                          (i,0),(j,0)
 *       (i,0),(j,1)                          (i,2),(j,0)
 *       (i,0),(j,2)                          (i,1),(j,0)
 * 
 * NOTE: It is assumed that the grids neighbor information is
 * sufficiently large.
 *
 * NOTE: No concept of data ownership is assumed.  As a result,
 * redundant communication patterns can be produced when the grid
 * boxes overlap.
 *
 *--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromStencil( hypre_StructGrid      *grid,
                                 hypre_StructStencil   *stencil,
                                 hypre_BoxArrayArray  **send_boxes_ptr,
                                 hypre_BoxArrayArray  **recv_boxes_ptr,
                                 int                 ***send_procs_ptr,
                                 int                 ***recv_procs_ptr,
                                 int                  **send_order_ptr,
                                 int                  **recv_order_ptr )
{
   int                    ierr = 0;

   hypre_BoxArrayArray   *send_boxes;
   hypre_BoxArrayArray   *recv_boxes;
   int                  **send_procs;
   int                  **recv_procs;
   int                   *send_order;
   int                   *recv_order;
                       
   hypre_BoxArray        *boxes     = hypre_StructGridBoxes(grid);
   int                    num_boxes = hypre_BoxArraySize(boxes);
                       
   hypre_BoxNeighbors    *neighbors      = hypre_StructGridNeighbors(grid);
   hypre_BoxArray        *neighbor_boxes = hypre_BoxNeighborsBoxes(neighbors);
   int                   *neighbor_procs = hypre_BoxNeighborsProcs(neighbors);
   int                    num_hood;
                       
   hypre_Index           *stencil_shape;
   hypre_IndexRef         stencil_offset;
                       
   hypre_Box             *box;
   hypre_Box             *neighbor_box;
   hypre_Box             *grow_box;
   hypre_Box             *int_box;

   int                    stencil_grid[3][3][3];
   int                    grow[3][2];
                       
   hypre_BoxArray        *send_box_array;
   hypre_BoxArray        *recv_box_array;
   int                    send_box_array_size;
   int                    recv_box_array_size;
   int                    send_order_size;
   int                    recv_order_size;
                       
   hypre_Box            **cboxes;
   hypre_Box             *cboxes_mem;
   int                   *cboxes_j;
   int                    num_cboxes;
   hypre_BoxArray       **cper_arrays;
   int                    cper_array_size;
   int                   *send_order_j;
                       
   int                    i, j, k, m, n;
   int                    s, d, lastj;
   int                    istart[3], istop[3];
   int                    sgindex[3];
                       
   /*------------------------------------------------------
    * Compute the "stencil grid" and "grow" information
    *------------------------------------------------------*/

   stencil_shape = hypre_StructStencilShape(stencil);

   for (k = 0; k < 3; k++)
   {
      for (j = 0; j < 3; j++)
      {
         for (i = 0; i < 3; i++)
         {
            stencil_grid[i][j][k] = 0;
         }
      }
   }

   for (d = 0; d < 3; d++)
   {
      grow[d][0] = 0;
      grow[d][1] = 0;
   }

   for (s = 0; s < hypre_StructStencilSize(stencil); s++)
   {
      stencil_offset = stencil_shape[s];

      for (d = 0; d < 3; d++)
      {
         m = stencil_offset[d];

         istart[d] = 1;
         istop[d]  = 1;

         if (m < 0)
         {
            istart[d] = 0;
            grow[d][0] = hypre_max(grow[d][0], -m);
         }
         else if (m > 0)
         {
            istop[d] = 2;
            grow[d][1] = hypre_max(grow[d][1],  m);
         }
      }

      for (k = istart[2]; k <= istop[2]; k++)
      {
         for (j = istart[1]; j <= istop[1]; j++)
         {
            for (i = istart[0]; i <= istop[0]; i++)
            {
               stencil_grid[i][j][k] = 1;
            }
         }
      }
   }
 
   /*------------------------------------------------------
    * Compute send/recv boxes and procs
    *------------------------------------------------------*/

   send_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   recv_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   send_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));
   recv_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));

   send_order_size = 0;
   recv_order_size = 0;

   grow_box = hypre_BoxCreate();
   int_box  = hypre_BoxCreate();

   num_hood = hypre_BoxArraySize(neighbor_boxes) /
      hypre_BoxNeighborsNumPeriods(neighbors);

   cboxes       = hypre_CTAlloc(hypre_Box *, num_hood);
   cboxes_mem   = hypre_CTAlloc(hypre_Box, num_hood);
   cboxes_j     = hypre_CTAlloc(int, num_hood);
   cper_arrays  = hypre_CTAlloc(hypre_BoxArray *, num_hood);
   send_order_j = hypre_CTAlloc(int, num_hood);

   for (i = 0; i < num_boxes; i++)
   {
      box = hypre_BoxArrayBox(boxes, i);

      /*------------------------------------------------
       * Compute recv_box_array for box i
       *------------------------------------------------*/

      /* grow box */
      hypre_CopyBox(box, grow_box);
      for (d = 0; d < 3; d++)
      {
         hypre_BoxIMinD(grow_box, d) -= grow[d][0];
         hypre_BoxIMaxD(grow_box, d) += grow[d][1];
      }

      lastj = -1;
      num_cboxes = 0;
      recv_box_array_size = 0;
      hypre_BeginBoxNeighborsLoop(k, neighbors, i)
         {
            neighbor_box = hypre_BoxArrayBox(neighbor_boxes, k);

            for (d = 0; d < 3; d++)
            {
               sgindex[d] = 1;
               
               s = hypre_BoxIMinD(neighbor_box, d) - hypre_BoxIMaxD(box, d);
               if (s > 0)
               {
                  sgindex[d] = 2;
               }
               s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(neighbor_box, d);
               if (s > 0)
               {
                  sgindex[d] = 0;
               }
            }

            if (stencil_grid[sgindex[0]][sgindex[1]][sgindex[2]])
            {
               /* intersect */
               hypre_IntersectBoxes(grow_box, neighbor_box, int_box);
               
               if (hypre_BoxVolume(int_box))
               {
                  j = k % num_hood;
                  
                  if (j != lastj)
                  {
                     cboxes_j[num_cboxes] = j;
                     num_cboxes++;
                     lastj = j;
                  }
                  recv_box_array_size++;
                  
                  if (k < num_hood)
                  {
                     /* the neighbor was not periodic */
                     cboxes[j] = &cboxes_mem[j];
                     hypre_CopyBox(int_box, cboxes[j]);
                  }
                  else
                  {
                     /* the neighbor was periodic */
                     if (cper_arrays[j] == NULL)
                     {
                        cper_arrays[j] = hypre_BoxArrayCreate(26);
                        hypre_BoxArraySetSize(cper_arrays[j], 0);
                     }
                     hypre_AppendBox(int_box, cper_arrays[j]);
                  }
               }
            }
         }
      hypre_EndBoxNeighborsLoop;

      recv_order_size += recv_box_array_size;

      /* create recv_box_array and recv_procs */
      recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      hypre_BoxArraySetSize(recv_box_array, recv_box_array_size);
      recv_procs[i] = hypre_CTAlloc(int, recv_box_array_size);
      n = 0;
      for (m = 0; m < num_cboxes; m++)
      {
         j = cboxes_j[m];

         /* add the non-periodic box */
         if (cboxes[j] != NULL)
         {
            recv_procs[i][n] = neighbor_procs[j];
            hypre_CopyBox(cboxes[j], hypre_BoxArrayBox(recv_box_array, n));
            n++;
            cboxes[j] = NULL;
         }

         /* add the periodic boxes */
         if (cper_arrays[j] != NULL)
         {
            cper_array_size = hypre_BoxArraySize(cper_arrays[j]);
            for (k = 0; k < cper_array_size; k++)
            {
               recv_procs[i][n] = neighbor_procs[j];
               hypre_CopyBox(hypre_BoxArrayBox(cper_arrays[j], k),
                             hypre_BoxArrayBox(recv_box_array, n));
               n++;
            }
            hypre_BoxArrayDestroy(cper_arrays[j]);
            cper_arrays[j] = NULL;
         }
      }

      /*------------------------------------------------
       * Compute send_box_array for box i
       *------------------------------------------------*/

      lastj = -1;
      num_cboxes = 0;
      send_box_array_size = 0;
      hypre_BeginBoxNeighborsLoop(k, neighbors, i)
         {
            neighbor_box = hypre_BoxArrayBox(neighbor_boxes, k);

            for (d = 0; d < 3; d++)
            {
               sgindex[d] = 1;
               
               s = hypre_BoxIMinD(box, d) - hypre_BoxIMaxD(neighbor_box, d);
               if (s > 0)
               {
                  sgindex[d] = 2;
               }
               s = hypre_BoxIMinD(neighbor_box, d) - hypre_BoxIMaxD(box, d);
               if (s > 0)
               {
                  sgindex[d] = 0;
               }
            }

            if (stencil_grid[sgindex[0]][sgindex[1]][sgindex[2]])
            {
               /* grow box and intersect */
               hypre_CopyBox(neighbor_box, grow_box);
               for (d = 0; d < 3; d++)
               {
                  hypre_BoxIMinD(grow_box, d) -= grow[d][0];
                  hypre_BoxIMaxD(grow_box, d) += grow[d][1];
               }
               hypre_IntersectBoxes(box, grow_box, int_box);

               if (hypre_BoxVolume(int_box))
               {
                  j = k % num_hood;

                  if (j != lastj)
                  {
                     cboxes_j[num_cboxes] = j;
                     num_cboxes++;
                     lastj = j;
                  }
                  send_box_array_size++;
                  send_order_j[j]++;

                  if (k < num_hood)
                  {
                     /* the neighbor was not periodic */
                     cboxes[j] = &cboxes_mem[j];
                     hypre_CopyBox(int_box, cboxes[j]);
                  }
                  else
                  {
                     /* the neighbor was periodic */
                     if (cper_arrays[j] == NULL)
                     {
                        cper_arrays[j] = hypre_BoxArrayCreate(26);
                        hypre_BoxArraySetSize(cper_arrays[j], 0);
                     }
                     hypre_AppendBox(int_box, cper_arrays[j]);
                  }
               }
            }
         }
      hypre_EndBoxNeighborsLoop;
      
      send_order_size += send_box_array_size;

      /* create send_box_array and send_procs */
      send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
      hypre_BoxArraySetSize(send_box_array, send_box_array_size);
      send_procs[i] = hypre_CTAlloc(int, send_box_array_size);
      n = 0;
      for (m = 0; m < num_cboxes; m++)
      {
         j = cboxes_j[m];

         /* add the non-periodic box */
         if (cboxes[j] != NULL)
         {
            /* temporarily use send_procs to store j index */
            send_procs[i][n] = j;
            hypre_CopyBox(cboxes[j], hypre_BoxArrayBox(send_box_array, n));
            n++;
            cboxes[j] = NULL;
         }

         /* add the periodic boxes (in reverse order) */
         if (cper_arrays[j] != NULL)
         {
            cper_array_size = hypre_BoxArraySize(cper_arrays[j]);
            for (k = (cper_array_size-1); k > -1; k--)
            {
               /* temporarily use send_procs to store j index */
               send_procs[i][n] = j;
               hypre_CopyBox(hypre_BoxArrayBox(cper_arrays[j], k),
                             hypre_BoxArrayBox(send_box_array, n));
               n++;
            }
            hypre_BoxArrayDestroy(cper_arrays[j]);
            cper_arrays[j] = NULL;
         }
      }
   }

   hypre_TFree(cboxes);
   hypre_TFree(cboxes_mem);
   hypre_TFree(cboxes_j);
   hypre_TFree(cper_arrays);

   hypre_BoxDestroy(grow_box);
   hypre_BoxDestroy(int_box);

   /*------------------------------------------------------
    * Set up recv_order array
    *------------------------------------------------------*/

   recv_order = hypre_TAlloc(int, 2*recv_order_size + 1);
   recv_order[2*recv_order_size] = -1;

   k = 0;
   hypre_ForBoxArrayI(i, recv_boxes)
      {
         recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         hypre_ForBoxI(n, recv_box_array)
            {
               recv_order[k++] = i;
               recv_order[k++] = n;
            }
      }

   /*------------------------------------------------------
    * Set up send_order array
    *------------------------------------------------------*/

   send_order = hypre_TAlloc(int, 2*send_order_size + 1);
   send_order[2*send_order_size] = -1;

   k = 0;
   for (j = 0; j < num_hood; j++)
   {
      if (send_order_j[j] > 0)
      {
         m = send_order_j[j];
         send_order_j[j] = k;
         k += m;
      }
      else
      {
         send_order_j[j] = -1;
      }
   }

   hypre_ForBoxArrayI(i, send_boxes)
      {
         send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         hypre_ForBoxI(n, send_box_array)
            {
               j = send_procs[i][n];
               send_procs[i][n] = neighbor_procs[j];

               k = 2*send_order_j[j];
               send_order[k++] = i;
               send_order[k++] = n;
               send_order_j[j]++;
            }
      }

   hypre_TFree(send_order_j);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_procs_ptr = send_procs;
   *recv_procs_ptr = recv_procs;
   *send_order_ptr = send_order;
   *recv_order_ptr = recv_order;

   return ierr;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for a given grid
 * based on a specified number of "ghost zones".  These patterns are
 * defined by building a stencil and calling CommInfoFromStencil.
 *--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromNumGhost( hypre_StructGrid      *grid,
                                  int                   *num_ghost,
                                  hypre_BoxArrayArray  **send_boxes_ptr,
                                  hypre_BoxArrayArray  **recv_boxes_ptr,
                                  int                 ***send_procs_ptr,
                                  int                 ***recv_procs_ptr,
                                  int                  **send_order_ptr,
                                  int                  **recv_order_ptr )
{
   int  ierr = 0;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   startstop[6], ii[3], i, d, size;

   stencil_shape = hypre_CTAlloc(hypre_Index, 27);
   for (i = 0; i < 6; i++)
   {
      startstop[i] = num_ghost[i] ? 1 : 0;
   }
   size = 0;
   for (ii[2] = -startstop[4]; ii[2] <= startstop[5]; ii[2]++)
   {
      for (ii[1] = -startstop[2]; ii[1] <= startstop[3]; ii[1]++)
      {
         for (ii[0] = -startstop[0]; ii[0] <= startstop[1]; ii[0]++)
         {
            for (d = 0; d < 3; d++)
            {
               if (ii[d] < 0)
               {
                  stencil_shape[size][d] = -num_ghost[2*d];
               }
               else if (ii[d] > 0)
               {
                  stencil_shape[size][d] =  num_ghost[2*d+1];
               }
            }
            size++;
         }
      }
   }
   stencil = hypre_StructStencilCreate(3, size, stencil_shape);

   hypre_CreateCommInfoFromStencil(grid, stencil,
                                   send_boxes_ptr, recv_boxes_ptr,
                                   send_procs_ptr, recv_procs_ptr,
                                   send_order_ptr, recv_order_ptr);
   
   hypre_StructStencilDestroy(stencil);

   return ierr;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for migrating data
 * from one grid distribution to another.
 *--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromGrids( hypre_StructGrid      *from_grid,
                               hypre_StructGrid      *to_grid,
                               hypre_BoxArrayArray  **send_boxes_ptr,
                               hypre_BoxArrayArray  **recv_boxes_ptr,
                               int                 ***send_procs_ptr,
                               int                 ***recv_procs_ptr,
                               int                  **send_order_ptr,
                               int                  **recv_order_ptr )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_procs;
   int                    **recv_procs;

   hypre_BoxArrayArray     *comm_boxes;
   int                    **comm_procs;
   hypre_BoxArray          *comm_box_array;
   hypre_Box               *comm_box;

   hypre_StructGrid        *local_grid;
   hypre_StructGrid        *remote_grid;

   hypre_BoxArray          *local_boxes;
   hypre_BoxArray          *remote_boxes;
   hypre_BoxArray          *remote_all_boxes;
   int                     *remote_all_procs;
   int                      remote_first_local;

   hypre_Box               *local_box;
   hypre_Box               *remote_box;

   int                      i, j, k, r;

   /*------------------------------------------------------
    * Set up communication info
    *------------------------------------------------------*/
 
   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         local_grid  = from_grid;
         remote_grid = to_grid;
         break;

         case 1:
         local_grid  = to_grid;
         remote_grid = from_grid;
         break;
      }

      /*---------------------------------------------------
       * Compute comm_boxes and comm_procs
       *---------------------------------------------------*/

      local_boxes  = hypre_StructGridBoxes(local_grid);
      remote_boxes = hypre_StructGridBoxes(remote_grid);
      hypre_GatherAllBoxes(hypre_StructGridComm(remote_grid), remote_boxes,
                           &remote_all_boxes,
                           &remote_all_procs,
                           &remote_first_local);

      comm_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(local_boxes));
      comm_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(local_boxes));

      comm_box = hypre_BoxCreate();
      hypre_ForBoxI(i, local_boxes)
         {
            local_box = hypre_BoxArrayBox(local_boxes, i);

            comm_box_array = hypre_BoxArrayArrayBoxArray(comm_boxes, i);
            comm_procs[i] =
               hypre_CTAlloc(int, hypre_BoxArraySize(remote_all_boxes));

            hypre_ForBoxI(j, remote_all_boxes)
               {
                  remote_box = hypre_BoxArrayBox(remote_all_boxes, j);

                  hypre_IntersectBoxes(local_box, remote_box, comm_box);
                  if (hypre_BoxVolume(comm_box))
                  {
                     k = hypre_BoxArraySize(comm_box_array);
                     comm_procs[i][k] = remote_all_procs[j];
                     
                     hypre_AppendBox(comm_box, comm_box_array);
                  }
               }

            comm_procs[i] =
               hypre_TReAlloc(comm_procs[i],
                              int, hypre_BoxArraySize(comm_box_array));
         }
      hypre_BoxDestroy(comm_box);

      hypre_BoxArrayDestroy(remote_all_boxes);
      hypre_TFree(remote_all_procs);

      switch(r)
      {
         case 0:
         send_boxes = comm_boxes;
         send_procs = comm_procs;
         break;

         case 1:
         recv_boxes = comm_boxes;
         recv_procs = comm_procs;
         break;
      }
   }

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_procs_ptr = send_procs;
   *recv_procs_ptr = recv_procs;

   return ierr;
}
