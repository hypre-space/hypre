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
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_GetCommInfo:
 *--------------------------------------------------------------------------*/

void
zzz_GetCommInfo( zzz_BoxArrayArray  **send_boxes_ptr,
                 zzz_BoxArrayArray  **recv_boxes_ptr,
                 int               ***send_processes_ptr,
                 int               ***recv_processes_ptr,
                 zzz_StructGrid      *grid,
                 zzz_StructStencil   *stencil            )
{
   /* output variables */
   zzz_BoxArrayArray     *send_boxes;
   zzz_BoxArrayArray     *recv_boxes;
   int                  **send_processes;
   int                  **recv_processes;

   /* internal variables */
   zzz_BoxArray          *boxes;
   zzz_BoxArray          *all_boxes;
   int                   *processes;

   zzz_BoxArray          *neighbors;
   int                   *neighbor_ranks;
                         
   zzz_BoxArrayArray     *shift_boxes;
   zzz_BoxArrayArray     *shift_neighbors;
                          
   int                   *neighbor_processes;
   int                    num_neighbor_processes;

   int                    process;

   int                    r, p, i, j, k;

   /* temporary work variables */
   zzz_BoxArrayArray     *box_aa0;
   zzz_BoxArrayArray     *box_aa1;
                         
   zzz_BoxArray          *box_a0;
   zzz_BoxArray          *box_a1;
   zzz_BoxArray          *box_a2;
   zzz_BoxArray          *box_a3;
                         
   zzz_Box               *box0;
   zzz_Box               *box1;
   zzz_Box               *box2;

   int                  **process_aa;
   int                   *box_array_sizes;

   /*------------------------------------------------------
    * Extract needed grid info
    *------------------------------------------------------*/

   boxes     = zzz_StructGridBoxes(grid);
   all_boxes = zzz_StructGridAllBoxes(grid);
   processes = zzz_StructGridProcesses(grid);

   /*------------------------------------------------------
    * Determine neighbors
    *------------------------------------------------------*/

   zzz_FindBoxNeighbors(boxes, all_boxes, stencil,
                        &neighbors, &neighbor_ranks);

   /*------------------------------------------------------
    * Determine shift_boxes and shift_neighbors
    *------------------------------------------------------*/

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         box_a0 = boxes;
         break;

         case 1:
         box_a0 = neighbors;
         break;
      }

      box_aa0 = zzz_GrowBoxArrayByStencil(box_a0, stencil, 0);
      box_aa1 = zzz_NewBoxArrayArray(zzz_BoxArraySize(box_a0));

      zzz_ForBoxArrayI(i, box_aa0)
      {
         box_a1 = zzz_BoxArrayArrayBoxArray(box_aa0, i);

         zzz_ForBoxI(j, box_a1)
         {
            box_a2 = zzz_SubtractBoxes(zzz_BoxArrayBox(box_a1, j),
                                       zzz_BoxArrayBox(box_a0, i));
            zzz_AppendBoxArray(box_a2,
                               zzz_BoxArrayArrayBoxArray(box_aa1, i));
            zzz_FreeBoxArrayShell(box_a2);
         }
      }

      zzz_FreeBoxArrayArray(box_aa0);

      switch(r)
      {
         case 0:
         shift_boxes = box_aa1;
         break;

         case 1:
         shift_neighbors = box_aa1;
         break;
      }
   }

   /*------------------------------------------------------
    * Determine recv_boxes and send_boxes by intersecting.
    * Also keep track of processes.
    *------------------------------------------------------*/

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         box_aa0 = shift_boxes;
         box_a0  = neighbors;
         break;

         case 1:
         box_aa0 = shift_neighbors;
         box_a0  = boxes;
         break;
      }

      box_aa1 = zzz_NewBoxArrayArray(zzz_BoxArraySize(boxes));
      process_aa = ctalloc(int *, zzz_BoxArraySize(boxes));
      zzz_ForBoxI(i, boxes)
         process_aa[i] = ctalloc(int, zzz_BoxArraySize(neighbors));

      zzz_ForBoxI(i, box_a0)
      {
         box0 = zzz_BoxArrayBox(box_a0, i);

         zzz_ForBoxArrayI(j, box_aa0)
         {
            box_a2 = zzz_BoxArrayArrayBoxArray(box_aa0, j);

            zzz_ForBoxI(k, box_a2)
            {
               box1 = zzz_BoxArrayBox(box_a2, k);

               box2 = zzz_IntersectBoxes(box0, box1);
               if (box2)
               {
                  switch(r)
                  {
                     case 0:
                     box_a3 = zzz_BoxArrayArrayBoxArray(box_aa1, j);
                     process_aa[j][zzz_BoxArraySize(box_a3)] =
                        processes[neighbor_ranks[i]];
                     zzz_AppendBox(box2, box_a3);
                     break;

                     case 1:
                     box_a3 = zzz_BoxArrayArrayBoxArray(box_aa1, i);
                     process_aa[i][zzz_BoxArraySize(box_a3)] =
                        processes[neighbor_ranks[j]];
                     zzz_AppendBox(box2, box_a3);
                     break;
                  }
               }
            }
         }
      }

      switch(r)
      {
         case 0:
         recv_boxes = box_aa1;
         recv_processes = process_aa;
         break;

         case 1:
         send_boxes = box_aa1;
         send_processes = process_aa;
         break;
      }
   }

   zzz_FreeBoxArrayArray(shift_boxes);
   zzz_FreeBoxArrayArray(shift_neighbors);

   /*------------------------------------------------------
    * Union the send_boxes and recv_boxes by process
    *------------------------------------------------------*/

   /* determine neighbor_processes and num_neighbor_processes */
   neighbor_processes = talloc(int, zzz_BoxArraySize(neighbors));
   num_neighbor_processes = 0;
   zzz_ForBoxI(i, neighbors)
   {
      process = processes[neighbor_ranks[i]];
      box0 = zzz_BoxArrayBox(neighbors, i);

      for (p = 0; p < num_neighbor_processes; p++)
         if (process == neighbor_processes[p])
            break;
      if (p == num_neighbor_processes)
      {
         neighbor_processes[p] = process;
         num_neighbor_processes++;
      }
   }

   box_array_sizes = ctalloc(int, num_neighbor_processes);

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         box_aa0 = send_boxes;
         process_aa = send_processes;
         break;

         case 1:
         box_aa0 = recv_boxes;
         process_aa = recv_processes;
         break;
      }

      box_aa1 = zzz_NewBoxArrayArray(zzz_BoxArrayArraySize(box_aa0));

      zzz_ForBoxArrayI(i, box_aa0)
      {
         box_a0 = zzz_BoxArrayArrayBoxArray(box_aa0, i);
         box_a1 = zzz_BoxArrayArrayBoxArray(box_aa1, i);

         for (p = 0; p < num_neighbor_processes; p++)
         {
            box_a2 = zzz_NewBoxArray();
            zzz_ForBoxI(j, box_a0)
            {
               if (process_aa[i][j] == neighbor_processes[p])
                  zzz_AppendBox(zzz_BoxArrayBox(box_a0, j), box_a2);
            }

            box_a3 = zzz_UnionBoxArray(box_a2);
            zzz_AppendBoxArray(box_a3, box_a1);
            box_array_sizes[p] = zzz_BoxArraySize(box_a3);

            zzz_FreeBoxArrayShell(box_a2);
            zzz_FreeBoxArrayShell(box_a3);
         }

         /* fix process info */
         tfree(process_aa[i]);
         process_aa[i] = ctalloc(int, zzz_BoxArraySize(box_a1));
         j = 0;
         for (p = 0; p < num_neighbor_processes; p++)
         {
            for (k = 0; k < box_array_sizes[p]; k++)
            {
               process_aa[i][j] = neighbor_processes[p];
               j++;
            }
         }
      }

      zzz_FreeBoxArrayArray(box_aa0);

      switch(r)
      {
         case 0:
         send_boxes = box_aa1;
         break;

         case 1:
         recv_boxes = box_aa1;
         break;
      }
   }

   tfree(box_array_sizes);
   tfree(neighbor_processes);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   tfree(neighbor_ranks);
   zzz_FreeBoxArrayShell(neighbors);

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_processes_ptr = send_processes;
   *recv_processes_ptr = recv_processes;
}

/*--------------------------------------------------------------------------
 * zzz_GetSBoxType:
 *   Computes the MPI derived datatype for a communication SBox, `comm_box',
 *   imbedded in a data space Box, `data_box'.
 *--------------------------------------------------------------------------*/

void
zzz_GetSBoxType( zzz_SBox     *comm_sbox,
                 zzz_Box      *data_box,
                 int           num_values,
                 MPI_Datatype *comm_sbox_type                      )
{
   int  length_array[4];
   int  stride_array[4];

   MPI_Datatype *old_type;
   MPI_Datatype *new_type;
   MPI_Datatype *tmp_type;
             
   int  i, j, dim;

   /*------------------------------------------------------
    * Compute length_array, stride_array, and dim
    *------------------------------------------------------*/

   /* initialize length_array */
   for (i = 0; i < 3; i++)
      length_array[i] = zzz_SBoxSizeD(comm_sbox, i);
   length_array[3] = num_values;

   /* initialize stride_array */
   for (i = 0; i < 3; i++)
   {
      stride_array[i] = zzz_SBoxStrideD(comm_sbox, i);
      for (j = 0; j < i; j++)
         stride_array[i] *= zzz_BoxSizeD(data_box, j);
   }
   stride_array[3] = zzz_BoxVolume(data_box);

   /* eliminate dimensions with length_array = 1 */
   dim = 4;
   for(i = 0; i < dim; i++)
   {
      if(length_array[i] == 1)
      {
         for(j = i; j < 3; j++)
         {
            length_array[j] = length_array[j+1];
            stride_array[j] = stride_array[j+1];
         }
         dim--;
      }
   }

#if 0
   /* sort the array according to length_array (largest to smallest) */
   for (i = (dim-1); i > 0; i--)
      for (j = 0; j < i; j++)
	 if (length_array[j] < length_array[j+1])
	 {
	    i_tmp             = length_array[j];
	    length_array[j]   = length_array[j+1];
	    length_array[j+1] = i_tmp;

	    i_tmp             = stride_array[j];
	    stride_array[j]   = stride_array[j+1];
	    stride_array[j+1] = i_tmp;
	 }
#endif

   /* if every len was 1 we need to fix to communicate at least one */
   if(!dim)
      dim = 1;

   /*------------------------------------------------------
    * Compute comm_sbox_type
    *------------------------------------------------------*/

   old_type = ctalloc(MPI_Datatype, 1);
   new_type = ctalloc(MPI_Datatype, 1);

   MPI_Type_contiguous(1, MPI_DOUBLE, old_type);
   for (i = 0; i < (dim - 1); i++)
   {
      MPI_Type_hvector(length_array[i], 1,
                       (MPI_Aint)(stride_array[i]*sizeof(double)),
                       *old_type, new_type);

      MPI_Type_free(old_type);
      tmp_type = old_type;
      old_type = new_type;
      new_type = tmp_type;

   }
   MPI_Type_hvector(length_array[i], 1,
                    (MPI_Aint)(stride_array[i]*sizeof(double)),
                    *old_type, comm_sbox_type);
   MPI_Type_free(old_type);

   tfree(old_type);
   tfree(new_type);
}

/*--------------------------------------------------------------------------
 * zzz_NewCommPkg:
 *--------------------------------------------------------------------------*/

zzz_CommPkg *
zzz_NewCommPkg( zzz_SBoxArrayArray  *send_sboxes,
                zzz_SBoxArrayArray  *recv_sboxes,
                int                **send_processes,
                int                **recv_processes,
                zzz_BoxArray        *data_space,
                int                  num_values     )
{
   /* output variables */
   zzz_CommPkg         *comm_pkg;
                       
   int                  pkg_num_sends;
   int                  pkg_num_recvs;
                       
   int                 *pkg_send_processes;
   int                 *pkg_recv_processes;
                       
   MPI_Datatype        *pkg_send_types;
   MPI_Datatype        *pkg_recv_types;

   /* internal variables */
   int                  pkg_num_comms;
   int                 *pkg_comm_processes;
   MPI_Datatype        *pkg_comm_types;
   int                 *tmp_pkg_comm_processes;

   zzz_SBoxArrayArray  *comm_sboxes;
   zzz_SBoxArray       *comm_sbox_array;
   zzz_SBox            *comm_sbox;
   int                **comm_processes;

   int                  num_comm_sboxes;
   int                  max_num_comm_sboxes;
                     
   int                 *comm_sbox_block_lengths;
   MPI_Aint            *comm_sbox_displacements;
   MPI_Datatype        *comm_sbox_types;

   zzz_Box             *data_box;
   int                  data_box_offset;
                     
   int                  process;
   int                  r, p, i, j;
                
   /*------------------------------------------------------
    * Compute an upper bound for the number of send and
    * receive SBoxes, `max_num_comm_sboxes'.
    *------------------------------------------------------*/

   max_num_comm_sboxes = 0;

   num_comm_sboxes = 0;
   zzz_ForSBoxArrayI(i, send_sboxes)
      num_comm_sboxes +=
	 zzz_SBoxArraySize(zzz_SBoxArrayArraySBoxArray(send_sboxes, i));
   max_num_comm_sboxes = max(max_num_comm_sboxes, num_comm_sboxes);

   num_comm_sboxes = 0;
   zzz_ForSBoxArrayI(i, recv_sboxes)
      num_comm_sboxes +=
	 zzz_SBoxArraySize(zzz_SBoxArrayArraySBoxArray(recv_sboxes, i));
   max_num_comm_sboxes = max(max_num_comm_sboxes, num_comm_sboxes);

   /*------------------------------------------------------
    * compute pkg_num_sends, pkg_send_processes and
    *         pkg_num_recvs, pkg_recv_processes
    *------------------------------------------------------*/

   tmp_pkg_comm_processes = talloc(int, max_num_comm_sboxes);

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         comm_sboxes    = send_sboxes;
         comm_processes = send_processes;
         break;
 
         case 1:
         comm_sboxes    = recv_sboxes;
         comm_processes = recv_processes;
         break;
      }

      pkg_num_comms = 0;
      zzz_ForSBoxArrayI(i, comm_sboxes)
      {
         comm_sbox_array = zzz_SBoxArrayArraySBoxArray(comm_sboxes, i);
         zzz_ForSBoxI(j, comm_sbox_array)
         {
            process = comm_processes[i][j];
            for (p = 0; p < pkg_num_comms; p++)
               if (process == tmp_pkg_comm_processes[p])
                  break;
            if (p == pkg_num_comms)
               tmp_pkg_comm_processes[pkg_num_comms++] = process;
         }
      }

      pkg_comm_processes = talloc(int, pkg_num_comms);
      for (i = 0; i < pkg_num_comms; i++)
         pkg_comm_processes[i] = tmp_pkg_comm_processes[i];

      switch(r)
      {
         case 0:
         pkg_num_sends      = pkg_num_comms;
         pkg_send_processes = pkg_comm_processes;
         break;
 
         case 1:
         pkg_num_recvs      = pkg_num_comms;
         pkg_recv_processes = pkg_comm_processes;
         break;
      }
   }

   tfree(tmp_pkg_comm_processes);

   /*------------------------------------------------------
    * Set up pkg_send_types and pkg_recv_types
    *------------------------------------------------------*/

   comm_sbox_block_lengths = talloc(int, max_num_comm_sboxes);
   for (i = 0; i < max_num_comm_sboxes; i++)
      comm_sbox_block_lengths[i] = 1;
   comm_sbox_displacements = talloc(MPI_Aint, max_num_comm_sboxes);
   comm_sbox_types = talloc(MPI_Datatype, max_num_comm_sboxes);

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         pkg_num_comms       = pkg_num_sends;
         pkg_comm_processes  = pkg_send_processes;
         comm_sboxes         = send_sboxes;
         comm_processes      = send_processes;
         break;
 
         case 1:
         pkg_num_comms       = pkg_num_recvs;
         pkg_comm_processes  = pkg_recv_processes;
         comm_sboxes         = recv_sboxes;
         comm_processes      = recv_processes;
         break;
      }

      pkg_comm_types = talloc(MPI_Datatype, pkg_num_comms);

      for(p = 0; p < pkg_num_comms; p++)
      {
	 num_comm_sboxes = 0;
         data_box_offset = 0;
	 zzz_ForSBoxArrayI(i, comm_sboxes)
	 {
	    comm_sbox_array = zzz_SBoxArrayArraySBoxArray(comm_sboxes, i);
	    data_box = zzz_BoxArrayBox(data_space, i);

	    zzz_ForSBoxI(j, comm_sbox_array)
	    {
	       comm_sbox = zzz_SBoxArraySBox(comm_sbox_array, j);

	       if ((zzz_SBoxVolume(comm_sbox) != 0) &&
                   (comm_processes[i][j] == pkg_comm_processes[p]))
	       {
                  zzz_GetSBoxType(comm_sbox, data_box, num_values,
                                  &comm_sbox_types[num_comm_sboxes]);

                  comm_sbox_displacements[num_comm_sboxes] =
                     (zzz_BoxIndexRank(data_box, zzz_SBoxIMin(comm_sbox)) +
                      data_box_offset) * sizeof(double);

		  num_comm_sboxes++;
	       }
	    }

            data_box_offset += zzz_BoxVolume(data_box);
	 }

         MPI_Type_struct(num_comm_sboxes, comm_sbox_block_lengths,
                         comm_sbox_displacements, comm_sbox_types,
                         &pkg_comm_types[p]);
         MPI_Type_commit(&pkg_comm_types[p]);

         for (i = 0; i < num_comm_sboxes; i++)
            MPI_Type_free(&comm_sbox_types[i]);
      }

      switch(r)
      {
         case 0:
         pkg_send_types = pkg_comm_types;
         break;

         case 1:
         pkg_recv_types = pkg_comm_types;
         break;
      }
   }

   tfree(comm_sbox_block_lengths);
   tfree(comm_sbox_displacements);
   tfree(comm_sbox_types);

   /*------------------------------------------------------
    * Set up zzz_CommPkg
    *------------------------------------------------------*/

   comm_pkg = ctalloc(zzz_CommPkg, 1);

   zzz_CommPkgNumSends(comm_pkg)      = pkg_num_sends;
   zzz_CommPkgSendProcesses(comm_pkg) = pkg_send_processes;
   zzz_CommPkgSendTypes(comm_pkg)     = pkg_send_types;

   zzz_CommPkgNumRecvs(comm_pkg)      = pkg_num_recvs;
   zzz_CommPkgRecvProcesses(comm_pkg) = pkg_recv_processes;
   zzz_CommPkgRecvTypes(comm_pkg)     = pkg_recv_types;

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 * zzz_FreeCommPkg:
 *--------------------------------------------------------------------------*/

void
zzz_FreeCommPkg( zzz_CommPkg *comm_pkg )
{
   MPI_Datatype  *types;
   int            i;

   if (comm_pkg)
   {
      tfree(zzz_CommPkgSendProcesses(comm_pkg));
      types = zzz_CommPkgSendTypes(comm_pkg);
      for (i = 0; i < zzz_CommPkgNumSends(comm_pkg); i++)
         MPI_Type_free(&types[i]);
      tfree(types);
     
      tfree(zzz_CommPkgRecvProcesses(comm_pkg));
      types = zzz_CommPkgRecvTypes(comm_pkg);
      for (i = 0; i < zzz_CommPkgNumRecvs(comm_pkg); i++)
         MPI_Type_free(&types[i]);
      tfree(types);

      tfree(comm_pkg);
   }
}

/*--------------------------------------------------------------------------
 * zzz_NewCommHandle:
 *--------------------------------------------------------------------------*/

zzz_CommHandle *
zzz_NewCommHandle( int          num_requests,
                   MPI_Request *requests     )
{
   zzz_CommHandle *comm_handle;

   comm_handle = ctalloc(zzz_CommHandle, 1);

   zzz_CommHandleNumRequests(comm_handle) = num_requests;
   zzz_CommHandleRequests(comm_handle)    = requests;

   return comm_handle;
}

/*--------------------------------------------------------------------------
 * zzz_FreeCommHandle:
 *--------------------------------------------------------------------------*/

void
zzz_FreeCommHandle( zzz_CommHandle *comm_handle )
{
   if (comm_handle)
   {
      tfree(zzz_CommHandleRequests(comm_handle));
      tfree(comm_handle);
   }
}

/*--------------------------------------------------------------------------
 * zzz_InitializeCommunication:
 *--------------------------------------------------------------------------*/

zzz_CommHandle *
zzz_InitializeCommunication( zzz_CommPkg *comm_pkg,
                             double      *data     )
{
   int              num_requests;
   MPI_Request     *requests;

   void            *vdata;
   int              i, j;

   vdata = (void *) data;

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   num_requests  =
      zzz_CommPkgNumSends(comm_pkg) +
      zzz_CommPkgNumRecvs(comm_pkg);
   requests = ctalloc(MPI_Request, num_requests);

   j = 0;

   for(i = 0; i < zzz_CommPkgNumRecvs(comm_pkg); i++)
   {
      MPI_Irecv(vdata, 1,
                zzz_CommPkgRecvType(comm_pkg, i), 
                zzz_CommPkgRecvProcess(comm_pkg, i), 
		0, MPI_COMM_WORLD, &requests[j++]);
   }

   for(i = 0; i < zzz_CommPkgNumSends(comm_pkg); i++)
   {
      MPI_Isend(vdata, 1,
                zzz_CommPkgSendType(comm_pkg, i), 
                zzz_CommPkgSendProcess(comm_pkg, i), 
		0, MPI_COMM_WORLD, &requests[j++]);
   }

   return ( zzz_NewCommHandle(num_requests, requests) );
}

/*--------------------------------------------------------------------------
 * zzz_FinalizeCommunication:
 *--------------------------------------------------------------------------*/

void
zzz_FinalizeCommunication( zzz_CommHandle *comm_handle )
{
   MPI_Status *status;

   if (comm_handle)
   {
      if (zzz_CommHandleNumRequests(comm_handle))
      {
         status = ctalloc(MPI_Status, zzz_CommHandleNumRequests(comm_handle));

         MPI_Waitall(zzz_CommHandleNumRequests(comm_handle),
                     zzz_CommHandleRequests(comm_handle),
                     status);

         tfree(status);
      }
 
      zzz_FreeCommHandle(comm_handle);
   }
}


