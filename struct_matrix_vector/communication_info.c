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

/*==========================================================================*/
/*==========================================================================*/
/** Return descriptions of communications patterns for a given
grid-stencil computation.  These patterns are defined by intersecting
the data dependencies of each box (including data dependencies within
the box) with its neighbor boxes.

{\bf Note:} It is assumed that the grids neighbor information is
sufficiently large.

{\bf Note:} No concept of data ownership is assumed.  As a result,
problematic communications patterns can be produced when the grid
boxes overlap.  For example, it is likely that some boxes will have
send and receive patterns that overlap.

{\bf Input files:}
headers.h

@return Error code.

@param grid [IN]
  computational grid
@param stencil [IN]
  computational stencil
@param send_boxes_ptr [OUT]
  description of the grid data to be sent to other processors.
@param recv_boxes_ptr [OUT]
  description of the grid data to be received from other processors.
@param send_procs_ptr [OUT]
  processors that data is to be sent to.
@param recv_procs_ptr [OUT]
  processors that data is to be received from.

@see hypre_CreateComputeInfo */
/*--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromStencil( hypre_StructGrid      *grid,
                                 hypre_StructStencil   *stencil,
                                 hypre_BoxArrayArray  **send_boxes_ptr,
                                 hypre_BoxArrayArray  **recv_boxes_ptr,
                                 int                 ***send_procs_ptr,
                                 int                 ***recv_procs_ptr )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_procs;
   int                    **recv_procs;

   /* internal variables */
   hypre_BoxArray          *boxes = hypre_StructGridBoxes(grid);

   hypre_BoxNeighbors      *neighbors;
   hypre_BoxArray          *neighbor_boxes;
   int                     *neighbor_procs;
   hypre_Box               *neighbor_box;

   hypre_Index             *stencil_shape;
   hypre_IndexRef           stencil_offset;

   hypre_Box               *box;
   hypre_Box               *shift_box;
                         
   hypre_BoxArray          *send_box_array;
   hypre_BoxArray          *recv_box_array;
   int                      send_box_array_size;
   int                      recv_box_array_size;

   hypre_BoxArray         **cbox_arrays;
   int                     *cbox_arrays_i;
   int                      num_cbox_arrays;

   int                      i, j, k, m, n;
   int                      s, d;

   /* temporary work variables */
   hypre_Box               *box0;

   /*------------------------------------------------------
    * Determine neighbors:
    *------------------------------------------------------*/

   neighbors = hypre_StructGridNeighbors(grid);

   /*------------------------------------------------------
    * Compute send/recv boxes and procs
    *------------------------------------------------------*/

   send_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   recv_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   send_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));
   recv_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));

   stencil_shape = hypre_StructStencilShape(stencil);

   neighbor_boxes = hypre_BoxNeighborsBoxes(neighbors);
   neighbor_procs = hypre_BoxNeighborsProcs(neighbors);

   box0 = hypre_BoxCreate();
   shift_box = hypre_BoxCreate();

   cbox_arrays =
      hypre_CTAlloc(hypre_BoxArray *, hypre_BoxArraySize(neighbor_boxes));
   cbox_arrays_i =
      hypre_CTAlloc(int, hypre_BoxArraySize(neighbor_boxes));

   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         hypre_CopyBox(box, shift_box);

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
 
            hypre_BeginBoxNeighborsLoop(j, neighbors, i, stencil_offset)
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  hypre_IntersectBoxes(shift_box, neighbor_box, box0);
                  if (hypre_BoxVolume(box0))
                  {
                     if (cbox_arrays[j] == NULL)
                     {
                        cbox_arrays[j] = hypre_BoxArrayCreate(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     hypre_AppendBox(box0, cbox_arrays[j]);
                  }
               }
            hypre_EndBoxNeighborsLoop;
         }

         /* union the boxes in cbox_arrays */
         recv_box_array_size = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_UnionBoxes(cbox_arrays[j]);
            recv_box_array_size += hypre_BoxArraySize(cbox_arrays[j]);
         }

         /* create recv_box_array and recv_procs */
         recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         hypre_BoxArraySetSize(recv_box_array, recv_box_array_size);
         recv_procs[i] = hypre_CTAlloc(int, recv_box_array_size);
         n = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  recv_procs[i][n] = neighbor_procs[j];
                  hypre_CopyBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                hypre_BoxArrayBox(recv_box_array, n));
                  n++;
               }
            hypre_BoxArrayDestroy(cbox_arrays[j]);
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
 
            hypre_BeginBoxNeighborsLoop(j, neighbors, i, stencil_offset)
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  hypre_IntersectBoxes(shift_box, neighbor_box, box0);
                  if (hypre_BoxVolume(box0))
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
                        cbox_arrays[j] = hypre_BoxArrayCreate(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     hypre_AppendBox(box0, cbox_arrays[j]);
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
            hypre_UnionBoxes(cbox_arrays[j]);
            send_box_array_size += hypre_BoxArraySize(cbox_arrays[j]);
         }

         /* create send_box_array and send_procs */
         send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         hypre_BoxArraySetSize(send_box_array, send_box_array_size);
         send_procs[i] = hypre_CTAlloc(int, send_box_array_size);
         n = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  send_procs[i][n] = neighbor_procs[j];
                  hypre_CopyBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                hypre_BoxArrayBox(send_box_array, n));
                  n++;
               }
            hypre_BoxArrayDestroy(cbox_arrays[j]);
            cbox_arrays[j] = NULL;
         }
      }

   hypre_TFree(cbox_arrays);
   hypre_TFree(cbox_arrays_i);

   hypre_BoxDestroy(shift_box);
   hypre_BoxDestroy(box0);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_procs_ptr = send_procs;
   *recv_procs_ptr = recv_procs;

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Return descriptions of communications patterns for a given grid
with "ghost zones".  These patterns are defined by intersecting each
box, "grown" by the number of "ghost zones", with its neighbor boxes.

{\bf Note:} It is assumed that the grids neighbor information is
sufficiently large.

{\bf Input files:}
headers.h

@return Error code.

@param grid [IN]
  computational grid
@param num_ghost [IN]
  number of ghost zones in each direction
@param send_boxes_ptr [OUT]
  description of the grid data to be sent to other processors.
@param recv_boxes_ptr [OUT]
  description of the grid data to be received from other processors.
@param send_procs_ptr [OUT]
  processors that data is to be sent to.
@param recv_procs_ptr [OUT]
  processors that data is to be received from.

@see hypre_CreateCommInfoFromStencil */
/*--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromNumGhost( hypre_StructGrid      *grid,
                                  int                   *num_ghost,
                                  hypre_BoxArrayArray  **send_boxes_ptr,
                                  hypre_BoxArrayArray  **recv_boxes_ptr,
                                  int                 ***send_procs_ptr,
                                  int                 ***recv_procs_ptr )
{
   int                      ierr = 0;

   /* output variables */
   hypre_BoxArrayArray     *send_boxes;
   hypre_BoxArrayArray     *recv_boxes;
   int                    **send_procs;
   int                    **recv_procs;

   /* internal variables */
   hypre_BoxArray          *boxes = hypre_StructGridBoxes(grid);
   int                     *ids   = hypre_StructGridIDs(grid);

   hypre_BoxNeighbors      *neighbors;
   hypre_BoxArray          *neighbor_boxes;
   int                     *neighbor_procs;
   int                     *neighbor_ids;
   hypre_Box               *neighbor_box;

   hypre_Box               *box;
   hypre_Box               *grow_box;
                         
   hypre_BoxArray          *send_box_array;
   hypre_BoxArray          *recv_box_array;
   int                      send_box_array_size;
   int                      recv_box_array_size;

   hypre_BoxArray         **cbox_arrays;
   int                     *cbox_arrays_i;
   int                      num_cbox_arrays;

   int                      i, j, k, m, n, d;

   /* temporary work variables */
   hypre_Box               *box0;

   /*------------------------------------------------------
    * Determine neighbors:
    *------------------------------------------------------*/

   neighbors = hypre_StructGridNeighbors(grid);

   /*------------------------------------------------------
    * Compute send/recv boxes and procs
    *------------------------------------------------------*/

   send_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   recv_boxes = hypre_BoxArrayArrayCreate(hypre_BoxArraySize(boxes));
   send_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));
   recv_procs = hypre_CTAlloc(int *, hypre_BoxArraySize(boxes));

   neighbor_boxes = hypre_BoxNeighborsBoxes(neighbors);
   neighbor_procs = hypre_BoxNeighborsProcs(neighbors);
   neighbor_ids   = hypre_BoxNeighborsIDs(neighbors);

   box0 = hypre_BoxCreate();
   grow_box = hypre_BoxCreate();

   cbox_arrays =
      hypre_CTAlloc(hypre_BoxArray *, hypre_BoxArraySize(neighbor_boxes));
   cbox_arrays_i =
      hypre_CTAlloc(int, hypre_BoxArraySize(neighbor_boxes));

   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         hypre_CopyBox(box, grow_box);

         /*------------------------------------------------
          * Compute recv_box_array for box i
          *------------------------------------------------*/

         num_cbox_arrays = 0;

         /* grow box by num_ghost */
         for (d = 0; d < 3; d++)
         {
            hypre_BoxIMinD(grow_box, d) =
               hypre_BoxIMinD(box, d) - num_ghost[2*d];
            hypre_BoxIMaxD(grow_box, d) =
               hypre_BoxIMaxD(box, d) + num_ghost[2*d + 1];
         }
 
         hypre_ForBoxI(j, neighbor_boxes)
            {
               if (ids[i] != neighbor_ids[j])
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  hypre_IntersectBoxes(grow_box, neighbor_box, box0);
                  if (hypre_BoxVolume(box0))
                  {
                     if (cbox_arrays[j] == NULL)
                     {
                        cbox_arrays[j] = hypre_BoxArrayCreate(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     hypre_AppendBox(box0, cbox_arrays[j]);
                  }
               }
            }

         /* union the boxes in cbox_arrays */
         recv_box_array_size = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_UnionBoxes(cbox_arrays[j]);
            recv_box_array_size += hypre_BoxArraySize(cbox_arrays[j]);
         }

         /* create recv_box_array and recv_procs */
         recv_box_array = hypre_BoxArrayArrayBoxArray(recv_boxes, i);
         hypre_BoxArraySetSize(recv_box_array, recv_box_array_size);
         recv_procs[i] = hypre_CTAlloc(int, recv_box_array_size);
         n = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  recv_procs[i][n] = neighbor_procs[j];
                  hypre_CopyBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                hypre_BoxArrayBox(recv_box_array, n));
                  n++;
               }
            hypre_BoxArrayDestroy(cbox_arrays[j]);
            cbox_arrays[j] = NULL;
         }

         /*------------------------------------------------
          * Compute send_box_array for box i
          *------------------------------------------------*/

         num_cbox_arrays = 0;

         hypre_ForBoxI(j, neighbor_boxes)
            {
               if (ids[i] != neighbor_ids[j])
               {
                  neighbor_box = hypre_BoxArrayBox(neighbor_boxes, j);

                  /* grow neighbor box by num_ghost */
                  for (d = 0; d < 3; d++)
                  {
                     hypre_BoxIMinD(grow_box, d) =
                        hypre_BoxIMinD(neighbor_box, d) - num_ghost[2*d];
                     hypre_BoxIMaxD(grow_box, d) =
                        hypre_BoxIMaxD(neighbor_box, d) + num_ghost[2*d + 1];
                  }
 
                  hypre_IntersectBoxes(box, grow_box, box0);
                  if (hypre_BoxVolume(box0))
                  {
                     if (cbox_arrays[j] == NULL)
                     {
                        cbox_arrays[j] = hypre_BoxArrayCreate(0);
                        cbox_arrays_i[num_cbox_arrays] = j;
                        num_cbox_arrays++;
                     }
                     hypre_AppendBox(box0, cbox_arrays[j]);
                  }
               }
            }

         /* union the boxes in cbox_arrays */
         send_box_array_size = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_UnionBoxes(cbox_arrays[j]);
            send_box_array_size += hypre_BoxArraySize(cbox_arrays[j]);
         }

         /* create send_box_array and send_procs */
         send_box_array = hypre_BoxArrayArrayBoxArray(send_boxes, i);
         hypre_BoxArraySetSize(send_box_array, send_box_array_size);
         send_procs[i] = hypre_CTAlloc(int, send_box_array_size);
         n = 0;
         for (m = 0; m < num_cbox_arrays; m++)
         {
            j = cbox_arrays_i[m];
            hypre_ForBoxI(k, cbox_arrays[j])
               {
                  send_procs[i][n] = neighbor_procs[j];
                  hypre_CopyBox(hypre_BoxArrayBox(cbox_arrays[j], k),
                                hypre_BoxArrayBox(send_box_array, n));
                  n++;
               }
            hypre_BoxArrayDestroy(cbox_arrays[j]);
            cbox_arrays[j] = NULL;
         }
      }

   hypre_TFree(cbox_arrays);
   hypre_TFree(cbox_arrays_i);

   hypre_BoxDestroy(grow_box);
   hypre_BoxDestroy(box0);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_procs_ptr = send_procs;
   *recv_procs_ptr = recv_procs;

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Return descriptions of communications patterns for migrating data
from one grid distribution to another.

{\bf Input files:}
headers.h

@return Error code.

@param from_grid [IN]
  grid distribution to migrate data from.
@param to_grid [IN]
  grid distribution to migrate data to.
@param send_boxes_ptr [OUT]
  description of the grid data to be sent to other processors.
@param recv_boxes_ptr [OUT]
  description of the grid data to be received from other processors.
@param send_procs_ptr [OUT]
  processors that data is to be sent to.
@param recv_procs_ptr [OUT]
  processors that data is to be received from.

@see hypre_StructMatrixMigrate, hypre_StructVectorMigrate */
/*--------------------------------------------------------------------------*/

int
hypre_CreateCommInfoFromGrids( hypre_StructGrid      *from_grid,
                               hypre_StructGrid      *to_grid,
                               hypre_BoxArrayArray  **send_boxes_ptr,
                               hypre_BoxArrayArray  **recv_boxes_ptr,
                               int                 ***send_procs_ptr,
                               int                 ***recv_procs_ptr )
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
