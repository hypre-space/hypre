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
 * hypre_NewCommInfoFromStencil:
 *--------------------------------------------------------------------------*/

int
hypre_NewCommInfoFromStencil( hypre_BoxArrayArray  **send_boxes_ptr,
                              hypre_BoxArrayArray  **recv_boxes_ptr,
                              int                 ***send_processes_ptr,
                              int                 ***recv_processes_ptr,
                              hypre_StructGrid      *grid,
                              hypre_StructStencil   *stencil            )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_processes;
   int                    **recv_processes;

   /* internal variables */
   hypre_BoxArray          *boxes = hypre_StructGridBoxes(grid);

   hypre_BoxNeighbors      *neighbors;
   hypre_BoxArray          *neighbor_boxes;
   int                     *neighbor_processes;
   hypre_Box               *neighbor_box;

   hypre_Index             *stencil_shape;
   hypre_IndexRef           stencil_offset;
   int                      stencil_max_offset;

   hypre_Box               *box;
   hypre_Box               *shift_box;
                         
   hypre_BoxArray          *send_box_array;
   hypre_BoxArray          *recv_box_array;
   int                      send_box_array_size;
   int                      recv_box_array_size;

   hypre_BoxArray         **cbox_arrays;
   int                     *cbox_arrays_i;
   int                      num_cbox_arrays;

   int                      i, j, k, m;
   int                      s, d;

   /* temporary work variables */
   hypre_BoxArray          *box_a0;
   hypre_Box               *box0;

   /*------------------------------------------------------
    * Determine neighbors:
    *------------------------------------------------------*/

   neighbors = hypre_StructGridNeighbors(grid);

   stencil_max_offset = hypre_StructStencilMaxOffset(stencil);
   if (stencil_max_offset > hypre_BoxNeighborsMaxDistance(neighbors))
   {
      hypre_StructGridMaxDistance(grid) = stencil_max_offset;
      hypre_AssembleStructGrid(grid, NULL, NULL, NULL);
      neighbors = hypre_StructGridNeighbors(grid);
   }

   /*------------------------------------------------------
    * Compute send/recv boxes and processes
    *------------------------------------------------------*/

   send_boxes = hypre_NewBoxArrayArray(hypre_BoxArraySize(boxes));
   recv_boxes = hypre_NewBoxArrayArray(hypre_BoxArraySize(boxes));
   send_processes = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));
   recv_processes = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));

   stencil_shape = hypre_StructStencilShape(stencil);

   neighbor_boxes = hypre_BoxNeighborsBoxes(neighbors);
   neighbor_processes = hypre_BoxNeighborsProcesses(neighbors);
   cbox_arrays =
      hypre_CTAlloc(hypre_BoxArray *, hypre_BoxArraySize(neighbor_boxes));
   cbox_arrays_i =
      hypre_CTAlloc(int, hypre_BoxArraySize(neighbor_boxes));

   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         shift_box = hypre_DuplicateBox(box);

         /*------------------------------------------------
          * Compute recv_box_array for box i
          *------------------------------------------------*/

         num_cbox_arrays = 0;
         for (s = 0; s < hypre_StructStencilSize(stencil); s++)
         {
            stencil_offset = stencil_shape[s];

            /* shift box by stencil_offset */
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(shift_box, d) =
                  hypre_BoxIMinD(box, d) + hypre_IndexD(stencil_offset, d);
               hypre_BoxIMaxD(shift_box, d) =
                  hypre_BoxIMaxD(box, d) + hypre_IndexD(stencil_offset, d);
            }
 
            hypre_BeginBoxNeighborsLoop(j, i, neighbors, stencil_offset)
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  box0 = hypre_IntersectBoxes(shift_box, neighbor_box);
                  if (box0)
                  {
                     if (cbox_arrays[j] == NULL)
                     {
                        cbox_arrays[j] = hypre_NewBoxArray(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     box_a0 = hypre_SubtractBoxes(box0, box);
                     hypre_AppendBoxArray(box_a0, cbox_arrays[j]);

                     hypre_FreeBox(box0);
                     hypre_FreeBoxArrayShell(box_a0);
                  }
               }
            hypre_EndBoxNeighborsLoop;
         }

         /* union the boxes in cbox_arrays */
         recv_box_array_size = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            box_a0 = hypre_UnionBoxArray(cbox_arrays[j]);
            hypre_FreeBoxArray(cbox_arrays[j]);
            cbox_arrays[j] = box_a0;
            recv_box_array_size += hypre_BoxArraySize(box_a0);
         }

         /* create recv_box_array and recv_processes */
         recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         recv_processes[i] = hypre_CTAlloc(int, recv_box_array_size);
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  recv_processes[i][hypre_BoxArraySize(recv_box_array)] =
                     neighbor_processes[j];
                  hypre_AppendBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                  recv_box_array);
               }
            hypre_FreeBoxArrayShell(cbox_arrays[j]);
            cbox_arrays[j] = NULL;
         }

         /*------------------------------------------------
          * Compute send_box_array for box i
          *------------------------------------------------*/

         num_cbox_arrays = 0;
         for (s = 0; s < hypre_StructStencilSize(stencil); s++)
         {
            stencil_offset = stencil_shape[s];

            /* transpose stencil_offset */
            for (d = 0; d < 3; d++)
            {
               hypre_IndexD(stencil_offset, d) =
                  -hypre_IndexD(stencil_offset, d);
            }

            /* shift box by transpose stencil_offset */
            for (d = 0; d < 3; d++)
            {
               hypre_BoxIMinD(shift_box, d) =
                  hypre_BoxIMinD(box, d) + hypre_IndexD(stencil_offset, d);
               hypre_BoxIMaxD(shift_box, d) =
                  hypre_BoxIMaxD(box, d) + hypre_IndexD(stencil_offset, d);
            }
 
            hypre_BeginBoxNeighborsLoop(j, i, neighbors, stencil_offset)
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  box0 = hypre_IntersectBoxes(shift_box, neighbor_box);
                  if (box0)
                  {
                     /* shift box0 back */
                     for (d = 0; d < 3; d++)
                     {
                        hypre_BoxIMinD(box0, d) -=
                           hypre_IndexD(stencil_offset, d);
                        hypre_BoxIMaxD(box0, d) -=
                           hypre_IndexD(stencil_offset, d);
                     }

                     if (cbox_arrays[j] == NULL)
                     {
                        cbox_arrays[j] = hypre_NewBoxArray(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     box_a0 = hypre_SubtractBoxes(box0, neighbor_box);
                     hypre_AppendBoxArray(box_a0, cbox_arrays[j]);

                     hypre_FreeBox(box0);
                     hypre_FreeBoxArrayShell(box_a0);
                  }
               }
            hypre_EndBoxNeighborsLoop;

            /* restore stencil_offset */
            for (d = 0; d < 3; d++)
            {
               hypre_IndexD(stencil_offset, d) =
                  -hypre_IndexD(stencil_offset, d);
            }
         }

         /* union the boxes in cbox_arrays */
         send_box_array_size = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            box_a0 = hypre_UnionBoxArray(cbox_arrays[j]);
            hypre_FreeBoxArray(cbox_arrays[j]);
            cbox_arrays[j] = box_a0;
            send_box_array_size += hypre_BoxArraySize(box_a0);
         }

         /* create send_box_array and send_processes */
         send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         send_processes[i] = hypre_CTAlloc(int, send_box_array_size);
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  send_processes[i][hypre_BoxArraySize(send_box_array)] =
                     neighbor_processes[j];
                  hypre_AppendBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                  send_box_array);
               }
            hypre_FreeBoxArrayShell(cbox_arrays[j]);
            cbox_arrays[j] = NULL;
         }

         hypre_FreeBox(shift_box);
      }

   hypre_TFree(cbox_arrays);
   hypre_TFree(cbox_arrays_i);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_processes_ptr = send_processes;
   *recv_processes_ptr = recv_processes;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NewCommInfoFromGrids:
 *--------------------------------------------------------------------------*/

int
hypre_NewCommInfoFromGrids( hypre_BoxArrayArray  **send_boxes_ptr,
                            hypre_BoxArrayArray  **recv_boxes_ptr,
                            int                 ***send_processes_ptr,
                            int                 ***recv_processes_ptr,
                            hypre_StructGrid      *from_grid,
                            hypre_StructGrid      *to_grid            )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_processes;
   int                    **recv_processes;

   hypre_BoxArrayArray     *comm_boxes;
   int                    **comm_processes;
   hypre_BoxArray          *comm_box_array;
   hypre_Box               *comm_box;

   hypre_StructGrid        *local_grid;
   hypre_StructGrid        *remote_grid;

   hypre_BoxArray          *local_boxes;
   hypre_BoxArray          *remote_boxes;
   hypre_BoxArray          *remote_all_boxes;
   int                     *remote_processes;
   int                     *remote_box_ranks;

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
       * Compute comm_boxes and comm_processes
       *---------------------------------------------------*/

      local_boxes  = hypre_StructGridBoxes(local_grid);
      remote_boxes = hypre_StructGridBoxes(remote_grid);
      hypre_GatherAllBoxes(hypre_StructGridComm(remote_grid), remote_boxes,
                           &remote_all_boxes,
                           &remote_processes,
                           &remote_box_ranks);

      comm_boxes = hypre_NewBoxArrayArray(hypre_BoxArraySize(local_boxes));
      comm_processes = hypre_CTAlloc(int *, hypre_BoxArraySize(local_boxes));

      hypre_ForBoxI(i, local_boxes)
         {
            local_box = hypre_BoxArrayBox(local_boxes, i);

            comm_box_array = hypre_BoxArrayArrayBoxArray(comm_boxes, i);
            comm_processes[i] =
               hypre_CTAlloc(int, hypre_BoxArraySize(remote_all_boxes));

            hypre_ForBoxI(j, remote_all_boxes)
               {
                  remote_box = hypre_BoxArrayBox(remote_all_boxes, j);

                  comm_box = hypre_IntersectBoxes(local_box, remote_box);
                  if (comm_box)
                  {
                     k = hypre_BoxArraySize(comm_box_array);
                     comm_processes[i][k] = remote_processes[j];
                     
                     hypre_AppendBox(comm_box, comm_box_array);
                  }
               }

            comm_processes[i] =
               hypre_TReAlloc(comm_processes[i],
                              int, hypre_BoxArraySize(comm_box_array));
         }

      hypre_FreeBoxArray(remote_all_boxes);
      hypre_TFree(remote_processes);
      hypre_TFree(remote_box_ranks);

      switch(r)
      {
         case 0:
         send_boxes     = comm_boxes;
         send_processes = comm_processes;
         break;

         case 1:
         recv_boxes     = comm_boxes;
         recv_processes = comm_processes;
         break;
      }
   }

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_processes_ptr = send_processes;
   *recv_processes_ptr = recv_processes;

   return ierr;
}
