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
                 int               ***send_box_ranks_ptr,
                 int               ***recv_box_ranks_ptr,
                 zzz_StructGrid      *grid,
                 zzz_StructStencil   *stencil            )
{
   /* output variables */
   zzz_BoxArrayArray     *send_boxes;
   zzz_BoxArrayArray     *recv_boxes;
   int                  **send_box_ranks;
   int                  **recv_box_ranks;

   /* internal variables */
   zzz_BoxArray          *boxes;
   zzz_BoxArray          *all_boxes;
   int                   *processes;

   zzz_BoxArray          *neighbors;
   int                   *neighbor_ranks;
   int                    num_neighbors;
                         
   zzz_BoxArrayArray     *shift_boxes;
   zzz_BoxArrayArray     *shift_neighbors;
                          
   int                    r, n, i, j, k;

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

   int                  **box_ranks;
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
    * Also keep track of communication box ranks.
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
      box_ranks = ctalloc(int *, zzz_BoxArraySize(boxes));
      zzz_ForBoxI(i, boxes)
         box_ranks[i] = ctalloc(int, zzz_BoxArraySize(neighbors));

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
                     box_ranks[j][zzz_BoxArraySize(box_a3)] =
                        neighbor_ranks[i];
                     zzz_AppendBox(box2, box_a3);
                     break;

                     case 1:
                     box_a3 = zzz_BoxArrayArrayBoxArray(box_aa1, i);
                     box_ranks[i][zzz_BoxArraySize(box_a3)] =
                        neighbor_ranks[j];
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
         recv_box_ranks = box_ranks;
         break;

         case 1:
         send_boxes = box_aa1;
         send_box_ranks = box_ranks;
         break;
      }
   }

   zzz_FreeBoxArrayArray(shift_boxes);
   zzz_FreeBoxArrayArray(shift_neighbors);

   /*------------------------------------------------------
    * Union the send_boxes and recv_boxes by communication
    * box ranks.
    *------------------------------------------------------*/

   num_neighbors = zzz_BoxArraySize(neighbors);
   box_array_sizes = ctalloc(int, num_neighbors);

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         box_aa0 = send_boxes;
         box_ranks = send_box_ranks;
         break;

         case 1:
         box_aa0 = recv_boxes;
         box_ranks = recv_box_ranks;
         break;
      }

      box_aa1 = zzz_NewBoxArrayArray(zzz_BoxArrayArraySize(box_aa0));

      zzz_ForBoxArrayI(i, box_aa0)
      {
         box_a0 = zzz_BoxArrayArrayBoxArray(box_aa0, i);
         box_a1 = zzz_BoxArrayArrayBoxArray(box_aa1, i);

         for (n = 0; n < num_neighbors; n++)
         {
            box_a2 = zzz_NewBoxArray();
            zzz_ForBoxI(j, box_a0)
            {
               if (box_ranks[i][j] == neighbor_ranks[n])
                  zzz_AppendBox(zzz_BoxArrayBox(box_a0, j), box_a2);
            }

            box_a3 = zzz_UnionBoxArray(box_a2);
            zzz_AppendBoxArray(box_a3, box_a1);
            box_array_sizes[n] = zzz_BoxArraySize(box_a3);

            zzz_FreeBoxArrayShell(box_a2);
            zzz_FreeBoxArrayShell(box_a3);
         }

         /* fix box rank info */
         tfree(box_ranks[i]);
         box_ranks[i] = ctalloc(int, zzz_BoxArraySize(box_a1));
         j = 0;
         for (n = 0; n < num_neighbors; n++)
         {
            for (k = 0; k < box_array_sizes[n]; k++)
            {
               box_ranks[i][j] = neighbor_ranks[n];
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

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   tfree(neighbor_ranks);
   zzz_FreeBoxArrayShell(neighbors);

   *send_boxes_ptr = send_boxes;
   *recv_boxes_ptr = recv_boxes;
   *send_box_ranks_ptr = send_box_ranks;
   *recv_box_ranks_ptr = recv_box_ranks;
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
                int                **send_sbox_ranks,
                int                **recv_sbox_ranks,
                zzz_StructGrid      *grid,
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

   int                 *box_ranks;
   int                 *processes;

   zzz_SBoxArrayArray  *comm_sboxes;
   zzz_SBoxArray       *comm_sbox_array;
   zzz_SBox            *comm_sbox;
   int                **comm_sbox_ranks;

   zzz_Box             *data_box;
   int                  data_box_offset;

   int                 *comm_process_flags;
   struct CommStruct
   {
      zzz_SBox         *comm_sbox;
      zzz_Box          *data_box;
      int               data_box_offset;
      int               orig;
      int               dest;
   };
   struct CommStruct  **comm_structs;
                     
   int                 *comm_origs;
   int                 *comm_dests;
   int                 *comm_sort;
   
   int                  num_comms;
   int                 *comm_block_lengths;
   MPI_Aint            *comm_displacements;
   MPI_Datatype        *comm_types;
                     
   int                  r, p, i, j;
   int                  num_procs;
   int                  tmp_int;
                
   /*---------------------------------------------------------
    * First time through, compute send package info.
    * Second time through, compute recv package info.
    *---------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );

   box_ranks = zzz_StructGridBoxRanks(grid);
   processes = zzz_StructGridProcesses(grid);

   for (r = 0; r < 2; r++)
   {
      switch(r)
      {
         case 0:
         comm_sboxes     = send_sboxes;
         comm_sbox_ranks = send_sbox_ranks;
         break;
 
         case 1:
         comm_sboxes     = recv_sboxes;
         comm_sbox_ranks = recv_sbox_ranks;
         break;
      }

      /*------------------------------------------------------
       * Loop over `comm_sboxes' and compute `comm_process_flags'.
       *------------------------------------------------------*/

      comm_process_flags = ctalloc(int, num_procs);

      pkg_num_comms = 0;
      zzz_ForSBoxArrayI(i, comm_sboxes)
      {
         comm_sbox_array = zzz_SBoxArrayArraySBoxArray(comm_sboxes, i);

         zzz_ForSBoxI(j, comm_sbox_array)
         {
            comm_sbox = zzz_SBoxArraySBox(comm_sbox_array, j);
            p = processes[comm_sbox_ranks[i][j]];

            if (zzz_SBoxVolume(comm_sbox) != 0)
            {
               comm_process_flags[p]++;
               if (comm_process_flags[p] == 1)
                  pkg_num_comms++;
            }
         }
      }

      /*------------------------------------------------------
       * Loop over `comm_sboxes' and compute `comm_structs'.
       *------------------------------------------------------*/

      comm_structs = ctalloc(struct CommStruct *, num_procs);

      data_box_offset = 0;
      zzz_ForSBoxArrayI(i, comm_sboxes)
      {
         comm_sbox_array = zzz_SBoxArrayArraySBoxArray(comm_sboxes, i);
         data_box = zzz_BoxArrayBox(data_space, i);

         zzz_ForSBoxI(j, comm_sbox_array)
         {
            comm_sbox = zzz_SBoxArraySBox(comm_sbox_array, j);
            p = processes[comm_sbox_ranks[i][j]];

            if (zzz_SBoxVolume(comm_sbox) != 0)
            {
               /* allocate CommStruct pointer */
               if (comm_structs[p] == NULL)
               {
                  comm_structs[p] =
                     ctalloc(struct CommStruct, comm_process_flags[p]);
                  comm_process_flags[p] = 0;
               }

               num_comms = comm_process_flags[p];

               comm_structs[p][num_comms].comm_sbox       = comm_sbox;
               comm_structs[p][num_comms].data_box        = data_box;
               comm_structs[p][num_comms].data_box_offset = data_box_offset;
               switch(r)
               {
                  case 0:
                  comm_structs[p][num_comms].orig = box_ranks[i];
                  comm_structs[p][num_comms].dest = comm_sbox_ranks[i][j];
                  break;
 
                  case 1:
                  comm_structs[p][num_comms].orig = comm_sbox_ranks[i][j];
                  comm_structs[p][num_comms].dest = box_ranks[i];
                  break;
               }

               comm_process_flags[p]++;
            }
         }

         data_box_offset += zzz_BoxVolume(data_box) * num_values;
      }

      /*------------------------------------------------------
       * Loop over comm_structs and build package info
       *------------------------------------------------------*/

      pkg_comm_processes = talloc(int, pkg_num_comms);
      pkg_comm_types     = talloc(MPI_Datatype, pkg_num_comms);

      pkg_num_comms = 0;
      for (p = 0; p < num_procs; p++)
      {
         if (comm_structs[p] != NULL)
         {
            num_comms = comm_process_flags[p];

            /* add process number to `pkg_comm_processes' */
            pkg_comm_processes[pkg_num_comms] = p;

            /* sort the comm_struct data                               */
            /* note: this bubble sort will maintain the original order */
            /*       for data with the same `orig' and `dest'          */
            comm_origs = talloc(int, num_comms);
            comm_dests = talloc(int, num_comms);
            comm_sort  = talloc(int, num_comms);
            for (i = 0; i < num_comms; i++)
            {
               comm_origs[i] = comm_structs[p][i].orig;
               comm_dests[i] = comm_structs[p][i].dest;
               comm_sort[i]  = i;
            }
            for (i = (num_comms - 1); i > 0; i--)
            {
               for (j = 0; j < i; j++)
               {
                  if ( (comm_dests[j] > comm_dests[j+1]) ||
                       ((comm_dests[j] == comm_dests[j+1]) &&
                        (comm_origs[j]  > comm_origs[j+1]))   )
                  {
                     tmp_int         = comm_origs[j];
                     comm_origs[j]   = comm_origs[j+1];
                     comm_origs[j+1] = tmp_int;

                     tmp_int         = comm_dests[j];
                     comm_dests[j]   = comm_dests[j+1];
                     comm_dests[j+1] = tmp_int;

                     tmp_int         = comm_sort[j];
                     comm_sort[j]    = comm_sort[j+1];
                     comm_sort[j+1]  = tmp_int;
                  }
               }
            }

            /* compute arguments for MPI_Type_struct routine */
            comm_block_lengths = talloc(int, num_comms);
            comm_displacements = talloc(MPI_Aint, num_comms);
            comm_types         = talloc(MPI_Datatype, num_comms);
            for (i = 0; i < num_comms; i++)
            {
               /* extract data from comm_struct */
               j = comm_sort[i];
               comm_sbox       = comm_structs[p][j].comm_sbox;
               data_box        = comm_structs[p][j].data_box;
               data_box_offset = comm_structs[p][j].data_box_offset;

               /* set block_lengths */
               comm_block_lengths[i] = 1;

               /* compute displacements */
               comm_displacements[i] = 
                  (zzz_BoxIndexRank(data_box, zzz_SBoxIMin(comm_sbox)) +
                   data_box_offset) * sizeof(double);

               /* compute types */
               zzz_GetSBoxType(comm_sbox, data_box, num_values,
                               &comm_types[i]);
            }

            /* create `pkg_comm_types' */
            MPI_Type_struct(num_comms, comm_block_lengths,
                            comm_displacements, comm_types,
                            &pkg_comm_types[pkg_num_comms]);
            MPI_Type_commit(&pkg_comm_types[pkg_num_comms]);

            pkg_num_comms++;

            /* free up memory */
            for (i = 0; i < num_comms; i++)
               MPI_Type_free(&comm_types[i]);
            tfree(comm_block_lengths);
            tfree(comm_displacements);
            tfree(comm_types);
            tfree(comm_origs);
            tfree(comm_dests);
            tfree(comm_sort);
            tfree(comm_structs[p]);
         }
      }

      tfree(comm_structs);
      tfree(comm_process_flags);

      switch(r)
      {
         case 0:
         pkg_num_sends      = pkg_num_comms;
         pkg_send_processes = pkg_comm_processes;
         pkg_send_types     = pkg_comm_types;
         break;
 
         case 1:
         pkg_num_recvs      = pkg_num_comms;
         pkg_recv_processes = pkg_comm_processes;
         pkg_recv_types     = pkg_comm_types;
         break;
      }
   }

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


