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
 * hypre_NewCommDataType
 *--------------------------------------------------------------------------*/
 
hypre_CommDataType *
hypre_NewCommDataType( hypre_SBox  *sbox,
                       hypre_Box   *data_box,
                       int          data_offset )
{
   hypre_CommDataType   *comm_data_type;
 
   comm_data_type = hypre_TAlloc(hypre_CommDataType, 1);
 
   hypre_CommDataTypeSBox(comm_data_type)       = sbox;
   hypre_CommDataTypeDataBox(comm_data_type)    = data_box;
   hypre_CommDataTypeDataOffset(comm_data_type) = data_offset;
 
   return comm_data_type;
}
 
/*--------------------------------------------------------------------------
 * hypre_FreeCommDataType
 *--------------------------------------------------------------------------*/
 
void
hypre_FreeCommDataType( hypre_CommDataType *comm_data_type )
{
   if (comm_data_type)
   {
      hypre_TFree(comm_data_type);
   }
}

/*--------------------------------------------------------------------------
 * hypre_NewSBoxType:
 *   Computes the MPI derived datatype for a communication SBox, `comm_box',
 *   imbedded in a data space Box, `data_box'.
 *--------------------------------------------------------------------------*/

void
hypre_NewSBoxType( hypre_SBox     *comm_sbox,
                   hypre_Box      *data_box,
                   int             num_values,
                   MPI_Datatype   *comm_sbox_type )
{
   int           length_array[4];
   int           stride_array[4];

   MPI_Datatype *old_type;
   MPI_Datatype *new_type;
   MPI_Datatype *tmp_type;
             
   int           i, j, dim;

   /*------------------------------------------------------
    * Compute length_array, stride_array, and dim
    *------------------------------------------------------*/

   /* initialize length_array */
   for (i = 0; i < 3; i++)
      length_array[i] = hypre_SBoxSizeD(comm_sbox, i);
   length_array[3] = num_values;

   /* initialize stride_array */
   for (i = 0; i < 3; i++)
   {
      stride_array[i] = hypre_SBoxStrideD(comm_sbox, i);
      for (j = 0; j < i; j++)
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
   }
   stride_array[3] = hypre_BoxVolume(data_box);

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

   if (dim == 1)
   {
      MPI_Type_hvector(length_array[0], 1,
                       (MPI_Aint)(stride_array[0]*sizeof(double)),
                       MPI_DOUBLE, comm_sbox_type);
   }
   else
   {
      old_type = hypre_CTAlloc(MPI_Datatype, 1);
      new_type = hypre_CTAlloc(MPI_Datatype, 1);

      MPI_Type_hvector(length_array[0], 1,
                       (MPI_Aint)(stride_array[0]*sizeof(double)),
                       MPI_DOUBLE, old_type);
      for (i = 1; i < (dim - 1); i++)
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

      hypre_TFree(old_type);
      hypre_TFree(new_type);
   }
}

/*--------------------------------------------------------------------------
 * hypre_SortCommDataTypes:
 *--------------------------------------------------------------------------*/

int
hypre_SortCommDataTypes( hypre_CommDataType  **comm_data_types,
                         int                   num_comm_data_types )
{
   hypre_CommDataType    *comm_data_type;
   hypre_SBox            *sbox;
   hypre_IndexRef         imin0, imin1;
   int                    swap;
   int                    i, j;
   int                    ierr = 0;
                      
   /*------------------------------------------------
    * Sort by imin:
    *
    * Note: this assumes that all sboxes describing
    * communications between any pair of processes
    * is distinct.
    *------------------------------------------------*/

   for (i = (num_comm_data_types - 1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         swap = 0;
         sbox = hypre_CommDataTypeSBox(comm_data_types[j]);
         imin0 = hypre_SBoxIMin(sbox);
         sbox = hypre_CommDataTypeSBox(comm_data_types[j+1]);
         imin1 = hypre_SBoxIMin(sbox);
         if ( hypre_IndexZ(imin0) > hypre_IndexZ(imin1) )
         {
            swap = 1;
         }
         else if ( hypre_IndexZ(imin0) == hypre_IndexZ(imin1) )
         {
            if ( hypre_IndexY(imin0) > hypre_IndexY(imin1) )
            {
               swap = 1;
            }
            else if ( hypre_IndexY(imin0) == hypre_IndexY(imin1) )
            {
               if ( hypre_IndexX(imin0) > hypre_IndexX(imin1) )
               {
                  swap = 1;
               }
            }
         }

         if (swap)
         {
            comm_data_type       = comm_data_types[j];
            comm_data_types[j]   = comm_data_types[j+1];
            comm_data_types[j+1] = comm_data_type;
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NewCommTypes:
 *--------------------------------------------------------------------------*/

int
hypre_NewCommTypes( hypre_SBoxArrayArray  *sboxes,
                    hypre_BoxArray        *data_space,
                    int                  **processes,
                    int                    num_values,
                    MPI_Comm               comm,
                    int                   *num_comms_ptr,
                    int                  **comm_processes_ptr,
                    MPI_Datatype         **comm_types_ptr,
                    int                   *num_copies_ptr,
                    hypre_CommDataType  ***copy_types_ptr)
{
   int                    num_comms;
   int                   *comm_processes;
   MPI_Datatype          *comm_types;
   int                    num_copies;
   hypre_CommDataType   **copy_types;
                       
   int                    num_comm_sboxes;
   int                   *comm_sbox_block_lengths;
   MPI_Aint              *comm_sbox_displacements;
   MPI_Datatype          *comm_sbox_types;
                       
   hypre_SBoxArray       *sbox_array;
   hypre_SBox            *sbox;

   hypre_Box             *data_box;
   int                    data_offset;

   hypre_CommDataType  ***comm_data_types;
   int                   *comm_data_types_sizes;
   hypre_CommDataType    *comm_data_type;
                       
   int                    p, i, j, m;
   int                    num_procs, my_proc;

   int                    ierr = 0;
                
   /*---------------------------------------------------------
    * Misc stuff
    *---------------------------------------------------------*/

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Loop over `sboxes' and compute `comm_data_types_sizes'.
    *------------------------------------------------------*/

   comm_data_types_sizes = hypre_CTAlloc(int, num_procs);

   num_comms = 0;
   hypre_ForSBoxArrayI(i, sboxes)
      {
         sbox_array = hypre_SBoxArrayArraySBoxArray(sboxes, i);

         hypre_ForSBoxI(j, sbox_array)
            {
               sbox = hypre_SBoxArraySBox(sbox_array, j);
               p = processes[i][j];

               if (hypre_SBoxVolume(sbox) != 0)
               {
                  comm_data_types_sizes[p]++;
                  if ((comm_data_types_sizes[p] == 1) && (p != my_proc))
                  {
                     num_comms++;
                  }
               }
            }
      }

   /*------------------------------------------------------
    * Loop over `sboxes' and compute `comm_data_types'
    * and `comm_processes'.
    *------------------------------------------------------*/

   comm_data_types = hypre_CTAlloc(hypre_CommDataType **, num_procs);
   comm_processes  = hypre_TAlloc(int, num_comms);

   m = 0;
   data_offset = 0;
   hypre_ForSBoxArrayI(i, sboxes)
      {
         sbox_array = hypre_SBoxArrayArraySBoxArray(sboxes, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         hypre_ForSBoxI(j, sbox_array)
            {
               sbox = hypre_SBoxArraySBox(sbox_array, j);
               p = processes[i][j];

               if (hypre_SBoxVolume(sbox) != 0)
               {
                  /* allocate CommStruct pointer */
                  if (comm_data_types[p] == NULL)
                  {
                     comm_data_types[p] =
                        hypre_CTAlloc(hypre_CommDataType *,
                                      comm_data_types_sizes[p]);
                     comm_data_types_sizes[p] = 0;

                     if (p != my_proc)
                     {
                        comm_processes[m] = p;
                        m++;
                     }
                  }

                  num_comm_sboxes = comm_data_types_sizes[p];

                  comm_data_types[p][num_comm_sboxes] =
                     hypre_NewCommDataType(sbox, data_box, data_offset);

                  comm_data_types_sizes[p]++;
               }
            }

         data_offset += hypre_BoxVolume(data_box) * num_values;
      }

   /*------------------------------------------------------
    * Loop over comm_data_types and build comm_types
    *------------------------------------------------------*/

   comm_types = hypre_TAlloc(MPI_Datatype, num_comms);

   for (m = 0; m < num_comms; m++)
   {
      p = comm_processes[m];
      num_comm_sboxes = comm_data_types_sizes[p];

      hypre_SortCommDataTypes(comm_data_types[p], num_comm_sboxes);

      comm_sbox_block_lengths = hypre_TAlloc(int, num_comm_sboxes);
      comm_sbox_displacements = hypre_TAlloc(MPI_Aint, num_comm_sboxes);
      comm_sbox_types = hypre_TAlloc(MPI_Datatype, num_comm_sboxes);
      for (i = 0; i < num_comm_sboxes; i++)
      {
         comm_data_type = comm_data_types[p][i];

         /* extract data from comm_struct */
         sbox        = hypre_CommDataTypeSBox(comm_data_type);
         data_box    = hypre_CommDataTypeDataBox(comm_data_type);
         data_offset = hypre_CommDataTypeDataOffset(comm_data_type);

         /* set block_lengths */
         comm_sbox_block_lengths[i] = 1;

         /* compute displacements */
         comm_sbox_displacements[i] = 
            (hypre_BoxIndexRank(data_box, hypre_SBoxIMin(sbox)) +
             data_offset) * sizeof(double);

         /* compute types */
         hypre_NewSBoxType(sbox, data_box, num_values,
                           &comm_sbox_types[i]);

         hypre_FreeCommDataType(comm_data_type);
      }

      /* create `comm_types' */
      MPI_Type_struct(num_comm_sboxes, comm_sbox_block_lengths,
                      comm_sbox_displacements, comm_sbox_types,
                      &comm_types[m]);
      MPI_Type_commit(&comm_types[m]);

         /* free up memory */
      for (i = 0; i < num_comm_sboxes; i++)
         MPI_Type_free(&comm_sbox_types[i]);
      hypre_TFree(comm_sbox_block_lengths);
      hypre_TFree(comm_sbox_displacements);
      hypre_TFree(comm_sbox_types);
      hypre_TFree(comm_data_types[p]);
   }

   /*------------------------------------------------------
    * Build copy_types
    *------------------------------------------------------*/

   if (comm_data_types[my_proc] != NULL)
   {
      num_comm_sboxes = comm_data_types_sizes[my_proc];
      hypre_SortCommDataTypes(comm_data_types[my_proc], num_comm_sboxes);

      num_copies = num_comm_sboxes;
      copy_types = comm_data_types[my_proc];
   }
   else
   {
      num_copies = 0;
      copy_types = NULL;
   }

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   hypre_TFree(comm_data_types);
   hypre_TFree(comm_data_types_sizes);

   *num_comms_ptr      = num_comms;
   *comm_processes_ptr = comm_processes;
   *comm_types_ptr     = comm_types;
   *num_copies_ptr     = num_copies;
   *copy_types_ptr     = copy_types;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NewCommPkg:
 *--------------------------------------------------------------------------*/

hypre_CommPkg *
hypre_NewCommPkg( hypre_SBoxArrayArray  *send_sboxes,
                  hypre_SBoxArrayArray  *recv_sboxes,
                  hypre_BoxArray        *send_data_space,
                  hypre_BoxArray        *recv_data_space,
                  int                  **send_processes,
                  int                  **recv_processes,
                  int                    num_values,
                  MPI_Comm               comm            )
{
   hypre_CommPkg         *comm_pkg;
                       
   int                    num_sends;
   int                   *send_procs;
   MPI_Datatype          *send_types;
   int                    num_recvs;
   int                   *recv_procs;
   MPI_Datatype          *recv_types;

   int                    num_copies_from;
   hypre_CommDataType   **copy_from_types;
   int                    num_copies_to;
   hypre_CommDataType   **copy_to_types;

   /*------------------------------------------------------
    * Put arguments into hypre_CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);

   hypre_CommPkgSendSBoxes(comm_pkg)    = send_sboxes;
   hypre_CommPkgRecvSBoxes(comm_pkg)    = recv_sboxes;
   hypre_CommPkgSendDataSpace(comm_pkg) = send_data_space;
   hypre_CommPkgRecvDataSpace(comm_pkg) = recv_data_space;
   hypre_CommPkgSendProcesses(comm_pkg) = send_processes;
   hypre_CommPkgRecvProcesses(comm_pkg) = recv_processes;
   hypre_CommPkgNumValues(comm_pkg)     = num_values;
   hypre_CommPkgComm(comm_pkg)          = comm;

   /*------------------------------------------------------
    * Set up communication information
    *------------------------------------------------------*/

   hypre_NewCommTypes(send_sboxes, send_data_space, send_processes,
                      num_values, comm,
                      &num_sends, &send_procs, &send_types,
                      &num_copies_from, &copy_from_types);

   hypre_CommPkgNumSends(comm_pkg)      = num_sends;
   hypre_CommPkgSendProcs(comm_pkg)     = send_procs;
   hypre_CommPkgSendTypes(comm_pkg)     = send_types;

   hypre_CommPkgNumCopiesFrom(comm_pkg) = num_copies_from;
   hypre_CommPkgCopyFromTypes(comm_pkg) = copy_from_types;

   hypre_NewCommTypes(recv_sboxes, recv_data_space, recv_processes,
                      num_values, comm,
                      &num_recvs, &recv_procs, &recv_types,
                      &num_copies_to, &copy_to_types);

   hypre_CommPkgNumRecvs(comm_pkg)      = num_recvs;
   hypre_CommPkgRecvProcs(comm_pkg)     = recv_procs;
   hypre_CommPkgRecvTypes(comm_pkg)     = recv_types;

   hypre_CommPkgNumCopiesTo(comm_pkg)   = num_copies_to;
   hypre_CommPkgCopyToTypes(comm_pkg)   = copy_to_types;

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 * hypre_FreeCommPkg:
 *--------------------------------------------------------------------------*/

void
hypre_FreeCommPkg( hypre_CommPkg *comm_pkg )
{
   MPI_Datatype  *types;
   int            i;

   if (comm_pkg)
   {
      hypre_ForSBoxArrayI(i, hypre_CommPkgSendSBoxes(comm_pkg))
         hypre_TFree(hypre_CommPkgSendProcesses(comm_pkg)[i]);
      hypre_ForSBoxArrayI(i, hypre_CommPkgRecvSBoxes(comm_pkg))
         hypre_TFree(hypre_CommPkgRecvProcesses(comm_pkg)[i]);
      hypre_TFree(hypre_CommPkgSendProcesses(comm_pkg));
      hypre_TFree(hypre_CommPkgRecvProcesses(comm_pkg));

      hypre_FreeSBoxArrayArray(hypre_CommPkgSendSBoxes(comm_pkg));
      hypre_FreeSBoxArrayArray(hypre_CommPkgRecvSBoxes(comm_pkg));

      hypre_TFree(hypre_CommPkgSendProcs(comm_pkg));
      types = hypre_CommPkgSendTypes(comm_pkg);
      for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
         MPI_Type_free(&types[i]);
      hypre_TFree(types);
     
      hypre_TFree(hypre_CommPkgRecvProcs(comm_pkg));
      types = hypre_CommPkgRecvTypes(comm_pkg);
      for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
         MPI_Type_free(&types[i]);
      hypre_TFree(types);

      for (i = 0; i < hypre_CommPkgNumCopiesFrom(comm_pkg); i++)
         hypre_FreeCommDataType(hypre_CommPkgCopyFromType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgCopyFromTypes(comm_pkg));

      for (i = 0; i < hypre_CommPkgNumCopiesTo(comm_pkg); i++)
         hypre_FreeCommDataType(hypre_CommPkgCopyToType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgCopyToTypes(comm_pkg));

      hypre_TFree(comm_pkg);
   }
}

/*--------------------------------------------------------------------------
 * hypre_NewCommHandle:
 *--------------------------------------------------------------------------*/

hypre_CommHandle *
hypre_NewCommHandle( int          num_requests,
                     MPI_Request *requests     )
{
   hypre_CommHandle *comm_handle;

   comm_handle = hypre_CTAlloc(hypre_CommHandle, 1);

   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;

   return comm_handle;
}

/*--------------------------------------------------------------------------
 * hypre_FreeCommHandle:
 *--------------------------------------------------------------------------*/

void
hypre_FreeCommHandle( hypre_CommHandle *comm_handle )
{
   if (comm_handle)
   {
      hypre_TFree(hypre_CommHandleRequests(comm_handle));
      hypre_TFree(comm_handle);
   }
}

/*--------------------------------------------------------------------------
 * hypre_InitializeCommunication:
 *--------------------------------------------------------------------------*/

hypre_CommHandle *
hypre_InitializeCommunication( hypre_CommPkg *comm_pkg,
                               double        *send_data,
                               double        *recv_data )
{
   MPI_Comm            comm       = hypre_CommPkgComm(comm_pkg);
   void               *send_vdata = (void *) send_data;
   void               *recv_vdata = (void *) recv_data;
                    
   int                 num_requests;
   MPI_Request        *requests;

   hypre_CommDataType *copy_from_type;
   hypre_CommDataType *copy_to_type;

   double             *from_dp;
   double             *to_dp;
   int                 from_i;
   int                 to_i;

   hypre_SBox         *sbox;
   hypre_Box          *from_data_box;
   hypre_Box          *to_data_box;

   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_IndexRef      stride;
                       
   int                 i, j;
   int                 loopi, loopj, loopk;

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   num_requests  =
      hypre_CommPkgNumSends(comm_pkg) +
      hypre_CommPkgNumRecvs(comm_pkg);
   requests = hypre_CTAlloc(MPI_Request, num_requests);

   j = 0;

   for(i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
   {
      MPI_Irecv(recv_vdata, 1,
                hypre_CommPkgRecvType(comm_pkg, i), 
                hypre_CommPkgRecvProc(comm_pkg, i), 
		0, comm, &requests[j++]);
   }

   for(i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
   {
      MPI_Isend(send_vdata, 1,
                hypre_CommPkgSendType(comm_pkg, i), 
                hypre_CommPkgSendProc(comm_pkg, i), 
		0, comm, &requests[j++]);
   }

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   for (i = 0; i < hypre_CommPkgNumCopiesFrom(comm_pkg); i++)
   {
      copy_from_type = hypre_CommPkgCopyFromType(comm_pkg, i);
      copy_to_type   = hypre_CommPkgCopyToType(comm_pkg, i);

      from_dp = send_data + hypre_CommDataTypeDataOffset(copy_from_type);
      to_dp   = recv_data + hypre_CommDataTypeDataOffset(copy_to_type);

      /* copy data only when necessary */
      if (to_dp != from_dp)
      {
         sbox          = hypre_CommDataTypeSBox(copy_from_type);
         from_data_box = hypre_CommDataTypeDataBox(copy_from_type);
         to_data_box   = hypre_CommDataTypeDataBox(copy_to_type);

         hypre_GetSBoxSize(sbox, loop_size);
         start  = hypre_SBoxIMin(sbox);
         stride = hypre_SBoxStride(sbox);
         hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
                        from_data_box, start, stride, from_i,
                        to_data_box,   start, stride, to_i,
                        {
                           to_dp[to_i] = from_dp[from_i];
                        });

      }
   }

   return ( hypre_NewCommHandle(num_requests, requests) );
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeCommunication:
 *--------------------------------------------------------------------------*/

void
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   MPI_Status *status;

   if (comm_handle)
   {
      if (hypre_CommHandleNumRequests(comm_handle))
      {
         status =
            hypre_CTAlloc(MPI_Status,
                          hypre_CommHandleNumRequests(comm_handle));

         MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                     hypre_CommHandleRequests(comm_handle),
                     status);

         hypre_TFree(status);
      }
 
      hypre_FreeCommHandle(comm_handle);
   }
}

