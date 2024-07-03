/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/*---------------------------------------------------------------------------------------
 * hypre_ParCSR(Persistent)CommHandleCreate sets up a communication handle,
 * posts receives and initiates sends. It always requires num_sends,
 * num_recvs, recv_procs and send_procs to be set in comm_pkg.
 * There are different options for job:
 * job = 1 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a Matvec,  it requires send_data
 *           and recv_data to be doubles, recv_vec_starts and
 *           send_map_starts need to be set in comm_pkg.
 * job = 2 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a MatvecT,  it requires send_data
 *           and recv_data to be doubles, recv_vec_starts and
 *           send_map_starts need to be set in comm_pkg.
 * job = 11: similar to job = 1, but exchanges data of type HYPRE_Int (not HYPRE_Complex),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * job = 12: similar to job = 2, but exchanges data of type HYPRE_Int (not HYPRE_Complex),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * job = 21: similar to job = 1, but exchanges data of type HYPRE_BigInt (not HYPRE_Complex),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * job = 22: similar to job = 2, but exchanges data of type HYPRE_BigInt (not HYPRE_Complex),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * default: ignores send_data and recv_data, requires send_mpi_types
 *           and recv_mpi_types to be set in comm_pkg.
 *           datatypes need to point to absolute
 *           addresses, e.g. generated using hypre_MPI_Address .
 *-------------------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommHandleAllocateBuffers( HYPRE_MemoryLocation    send_memory_location,
                                       HYPRE_MemoryLocation    recv_memory_location,
                                       HYPRE_Int               num_send_elems,
                                       HYPRE_Int               num_recv_elems,
                                       HYPRE_Int               size_of_elem,
                                       hypre_ParCSRCommHandle *comm_handle )
{
   hypre_MemoryLocation send_memory_alocation = hypre_GetActualMemLocation(send_memory_location);
   hypre_MemoryLocation recv_memory_alocation = hypre_GetActualMemLocation(recv_memory_location);

   if (!hypre_ParCSRCommHandleSendBuffer(comm_handle) && hypre_MPINeedHostBuffer(send_memory_alocation))
   {
      hypre_MemoryLocation location = hypre_HandleMPIHostBufferLocation(hypre_handle());
      hypre_ParCSRCommHandleSendBufferLocation(comm_handle) = location;
      hypre_ParCSRCommHandleSendBuffer(comm_handle) = _hypre_TAlloc(char, num_send_elems * size_of_elem, location);
   }

   if (!hypre_ParCSRCommHandleRecvBuffer(comm_handle) && hypre_MPINeedHostBuffer(recv_memory_alocation))
   {
      hypre_MemoryLocation location = hypre_HandleMPIHostBufferLocation(hypre_handle());
      hypre_ParCSRCommHandleRecvBufferLocation(comm_handle) = location;
      hypre_ParCSRCommHandleRecvBuffer(comm_handle) = _hypre_TAlloc(char, num_recv_elems * size_of_elem, location);
   }

   if (hypre_ParCSRCommHandlePersistent(comm_handle))
   {
      if (!hypre_ParCSRCommHandleSendData(comm_handle))
      {
         hypre_ParCSRCommHandleSendData(comm_handle) = hypre_TAlloc(char, num_send_elems * size_of_elem, send_memory_location);
      }

      if (!hypre_ParCSRCommHandleRecvData(comm_handle))
      {
         hypre_ParCSRCommHandleRecvData(comm_handle) = hypre_TAlloc(char, num_recv_elems * size_of_elem, recv_memory_location);
      }
   }

   hypre_MPICommSetSendLocation(hypre_ParCSRCommHandleComm(comm_handle), send_memory_alocation);
   hypre_MPICommSetSendBuffer(hypre_ParCSRCommHandleComm(comm_handle), hypre_ParCSRCommHandleSendBuffer(comm_handle));
   hypre_MPICommSetSendBufferLocation(hypre_ParCSRCommHandleComm(comm_handle), hypre_ParCSRCommHandleSendBufferLocation(comm_handle));

   hypre_MPICommSetRecvLocation(hypre_ParCSRCommHandleComm(comm_handle), recv_memory_alocation);
   hypre_MPICommSetRecvBuffer(hypre_ParCSRCommHandleComm(comm_handle), hypre_ParCSRCommHandleRecvBuffer(comm_handle));
   hypre_MPICommSetRecvBufferLocation(hypre_ParCSRCommHandleComm(comm_handle), hypre_ParCSRCommHandleRecvBufferLocation(comm_handle));

   return hypre_error_flag;
}

#if defined(HYPRE_USING_PERSISTENT_COMM)

/*------------------------------------------------------------------
 * hypre_ParCSRPersistentCommHandleCreate
 *
 * When send_data and recv_data are NULL, buffers are internally
 * allocated and CommHandle owns the buffer
 *------------------------------------------------------------------*/

hypre_ParCSRCommHandle*
hypre_ParCSRPersistentCommHandleCreate( HYPRE_Int            job,
                                        hypre_ParCSRCommPkg *comm_pkg,
                                        HYPRE_MemoryLocation send_memory_location,
                                        HYPRE_MemoryLocation recv_memory_location )
{
   return hypre_ParCSRCommHandleCreate_core(1, job, comm_pkg, send_memory_location, NULL, recv_memory_location, NULL);
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgGetPersistentCommHandle
 *------------------------------------------------------------------*/

hypre_ParCSRCommHandle*
hypre_ParCSRCommPkgGetPersistentCommHandle( HYPRE_Int            job,
                                            hypre_ParCSRCommPkg *comm_pkg,
                                            HYPRE_MemoryLocation send_memory_location,
                                            HYPRE_MemoryLocation recv_memory_location )
{
   CommPkgJobType type = hypre_ParCSRCommHandleGetJobType(job);

   if (!hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg, type))
   {
      /* data is owned by persistent comm handle */
      hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg, type) =
         hypre_ParCSRPersistentCommHandleCreate(job, comm_pkg, send_memory_location, recv_memory_location);
   }

   return hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg, type);
}


/*------------------------------------------------------------------
 * hypre_ParCSRPersistentCommHandleStart
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRPersistentCommHandleStart( hypre_ParCSRCommHandle *comm_handle )
{
   HYPRE_Int flag;
   hypre_MPI_Request_get_status(hypre_ParCSRCommHandleExtraRequest(comm_handle, 1), &flag, MPI_STATUS_IGNORE);

   if (hypre_ParCSRCommHandleNumRequests(comm_handle) > 0)
   {
      HYPRE_Int ret = hypre_MPI_Startall(hypre_ParCSRCommHandleNumRequests(comm_handle),
                                         hypre_ParCSRCommHandleRequests(comm_handle));
      if (hypre_MPI_SUCCESS != ret)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MPI error\n");
      }
   }

   return hypre_error_flag;
}

#endif // HYPRE_USING_PERSISTENT_COMM

/*------------------------------------------------------------------
 * hypre_ParCSRCommHandleCreate
 *------------------------------------------------------------------*/

hypre_ParCSRCommHandle*
hypre_ParCSRCommHandleCreate ( HYPRE_Int            job,
                               hypre_ParCSRCommPkg *comm_pkg,
                               void                *send_data,
                               void                *recv_data )
{
   return hypre_ParCSRCommHandleCreate_v2(job, comm_pkg, HYPRE_MEMORY_HOST, send_data,
                                          HYPRE_MEMORY_HOST, recv_data);
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommHandleCreate_v2
 *------------------------------------------------------------------*/
hypre_ParCSRCommHandle*
hypre_ParCSRCommHandleCreate_v2 ( HYPRE_Int            job,
                                  hypre_ParCSRCommPkg *comm_pkg,
                                  HYPRE_MemoryLocation send_memory_location,
                                  void                *send_data,
                                  HYPRE_MemoryLocation recv_memory_location,
                                  void                *recv_data )
{
   return hypre_ParCSRCommHandleCreate_core(0, job, comm_pkg, send_memory_location, send_data, recv_memory_location, recv_data);
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommHandleCreate_core
 *------------------------------------------------------------------*/

hypre_ParCSRCommHandle*
hypre_ParCSRCommHandleCreate_core ( HYPRE_Int            persistent,
                                    HYPRE_Int            job,
                                    hypre_ParCSRCommPkg *comm_pkg,
                                    HYPRE_MemoryLocation send_memory_location,
                                    void                *send_data,
                                    HYPRE_MemoryLocation recv_memory_location,
                                    void                *recv_data )
{
   hypre_GpuProfilingPushRange("hypre_ParCSRCommHandleCreate_core");

   HYPRE_Int               num_sends          = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int               num_recvs          = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm                comm_orig          = hypre_ParCSRCommPkgComm(comm_pkg);
   MPI_Comm                comm;
   HYPRE_Int               num_requests       = num_sends + num_recvs;
   hypre_MPI_Request      *requests           = hypre_CTAlloc(hypre_MPI_Request, num_requests, HYPRE_MEMORY_HOST);
   hypre_MPI_Datatype      mpi_dtype          = hypre_ParCSRCommHandleGetMPIDataType(job);
   hypre_ParCSRCommHandle *comm_handle        = hypre_CTAlloc(hypre_ParCSRCommHandle, 1, HYPRE_MEMORY_HOST);
   HYPRE_Int               num_extra_requests = persistent ? 2 : 1;
   hypre_MPI_Request      *extra_requests     = hypre_CTAlloc(hypre_MPI_Request, num_extra_requests, HYPRE_MEMORY_HOST);
   HYPRE_Int               num_send_elems     = 0;
   HYPRE_Int               num_recv_elems     = 0;
   HYPRE_Int               data_size;

   hypre_MPI_Comm_dup(comm_orig, &comm);
   hypre_MPI_Type_size(mpi_dtype, &data_size);

   hypre_ParCSRCommHandlePersistent(comm_handle) = persistent;
   hypre_ParCSRCommHandleComm(comm_handle) = comm;

   switch (hypre_ParCSRCommHandleGetJobType(job))
   {
      case HYPRE_COMM_PKG_JOB_COMPLEX:
      case HYPRE_COMM_PKG_JOB_INT:
      case HYPRE_COMM_PKG_JOB_BIGINT:
      {
         num_send_elems = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_recv_elems = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         hypre_ParCSRCommHandleAllocateBuffers(send_memory_location, recv_memory_location,
                                               num_send_elems, num_recv_elems, data_size, comm_handle);

         if (persistent)
         {
            hypre_MPI_Recv_init_Multiple(hypre_ParCSRCommHandleRecvData(comm_handle),
                                         num_recvs,
                                         hypre_ParCSRCommPkgRecvVecStarts(comm_pkg),
                                         NULL, mpi_dtype,
                                         hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                         0, comm, requests, &extra_requests[0]);

            hypre_MPI_Send_init_Multiple(hypre_ParCSRCommHandleSendData(comm_handle),
                                         num_sends,
                                         hypre_ParCSRCommPkgSendMapStarts(comm_pkg),
                                         NULL, mpi_dtype,
                                         hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                         0, comm, requests + num_recvs, &extra_requests[1]);
         }
         else
         {
            hypre_MPI_Irecv_Multiple(recv_data, num_recvs,
                                     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg),
                                     NULL, mpi_dtype,
                                     hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                     0, comm, requests, extra_requests);

            hypre_MPI_Isend_Multiple(send_data, num_sends,
                                     hypre_ParCSRCommPkgSendMapStarts(comm_pkg),
                                     NULL, mpi_dtype,
                                     hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                     0, comm, requests + num_recvs);
         }

         break;
      }

      case HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE:
      case HYPRE_COMM_PKG_JOB_INT_TRANSPOSE:
      case HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE:
      {
         num_send_elems = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         num_recv_elems = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         hypre_ParCSRCommHandleAllocateBuffers(send_memory_location, recv_memory_location,
                                               num_send_elems, num_recv_elems, data_size, comm_handle);

         if (persistent)
         {
            hypre_MPI_Recv_init_Multiple(hypre_ParCSRCommHandleRecvData(comm_handle),
                                         num_sends,
                                         hypre_ParCSRCommPkgSendMapStarts(comm_pkg),
                                         NULL, mpi_dtype,
                                         hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                         0, comm, requests, &extra_requests[0]);

            hypre_MPI_Send_init_Multiple(hypre_ParCSRCommHandleSendData(comm_handle),
                                         num_recvs,
                                         hypre_ParCSRCommPkgRecvVecStarts(comm_pkg),
                                         NULL, mpi_dtype,
                                         hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                         0, comm, requests + num_sends, &extra_requests[1]);
         }
         else
         {
            hypre_MPI_Irecv_Multiple(recv_data, num_sends,
                                     hypre_ParCSRCommPkgSendMapStarts(comm_pkg),
                                     NULL, mpi_dtype,
                                     hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                     0, comm, requests, extra_requests);

            hypre_MPI_Isend_Multiple(send_data, num_recvs,
                                     hypre_ParCSRCommPkgRecvVecStarts(comm_pkg),
                                     NULL, mpi_dtype,
                                     hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                     0, comm, requests + num_sends);
         }

         break;
      }

      default:
         break;
   }

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   hypre_ParCSRCommHandleCommPkg(comm_handle)          = comm_pkg;
   if (!persistent)
   {
      hypre_ParCSRCommHandleSendData(comm_handle)      = send_data;
      hypre_ParCSRCommHandleRecvData(comm_handle)      = recv_data;
   }
   hypre_ParCSRCommHandleSendLocation(comm_handle)     = send_memory_location;
   hypre_ParCSRCommHandleRecvLocation(comm_handle)     = recv_memory_location;
   hypre_ParCSRCommHandleNumRequests(comm_handle)      = num_requests;
   hypre_ParCSRCommHandleRequests(comm_handle)         = requests;
   hypre_ParCSRCommHandleNumExtraRequests(comm_handle) = num_extra_requests;
   hypre_ParCSRCommHandleExtraRequests(comm_handle)    = extra_requests;

   hypre_GpuProfilingPopRange();

   return ( comm_handle );
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommHandleWait
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommHandleWait( hypre_ParCSRCommHandle *comm_handle )
{
   if ( comm_handle == NULL )
   {
      return hypre_error_flag;
   }

   if (hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      HYPRE_Int ret = hypre_MPI_Waitall(hypre_ParCSRCommHandleNumRequests(comm_handle),
                                        hypre_ParCSRCommHandleRequests(comm_handle),
                                        hypre_MPI_STATUSES_IGNORE);
      if (hypre_MPI_SUCCESS != ret)
      {
         char errmsg[256];
         hypre_sprintf(errmsg, "MPI error %d in %s (%s, line %u)\n", ret, __FUNCTION__, __FILE__, __LINE__);
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      }
   }

   if (hypre_ParCSRCommHandlePersistent(comm_handle))
   {
      HYPRE_Int flag;
      hypre_MPI_Request_get_status(hypre_ParCSRCommHandleExtraRequest(comm_handle, 0), &flag, hypre_MPI_STATUS_IGNORE);
   }
   else
   {
      hypre_MPI_Wait(&hypre_ParCSRCommHandleExtraRequest(comm_handle, 0), MPI_STATUS_IGNORE);
   }

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommHandleDestroy
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommHandleDestroy( hypre_ParCSRCommHandle *comm_handle )
{
   if (!comm_handle)
   {
      return hypre_error_flag;
   }

   HYPRE_Int persistent = hypre_ParCSRCommHandlePersistent(comm_handle);

   if (!persistent)
   {
      hypre_ParCSRCommHandleWait(comm_handle);
   }

   _hypre_TFree(hypre_ParCSRCommHandleSendBuffer(comm_handle), hypre_ParCSRCommHandleSendBufferLocation(comm_handle));
   _hypre_TFree(hypre_ParCSRCommHandleRecvBuffer(comm_handle), hypre_ParCSRCommHandleRecvBufferLocation(comm_handle));


   if (persistent)
   {
      hypre_TFree(hypre_ParCSRCommHandleSendData(comm_handle), hypre_ParCSRCommHandleSendLocation(comm_handle));
      hypre_TFree(hypre_ParCSRCommHandleRecvData(comm_handle), hypre_ParCSRCommHandleRecvLocation(comm_handle));
   }

   hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle), HYPRE_MEMORY_HOST);

   if (persistent)
   {
      HYPRE_Int i;
      for (i = 0; i < hypre_ParCSRCommHandleNumExtraRequests(comm_handle); i++)
      {
         if (hypre_ParCSRCommHandleExtraRequest(comm_handle, i) != hypre_MPI_REQUEST_NULL)
         {
            hypre_MPI_Request_free(&hypre_ParCSRCommHandleExtraRequest(comm_handle, i));
         }
      }
   }

   hypre_TFree(hypre_ParCSRCommHandleExtraRequests(comm_handle), HYPRE_MEMORY_HOST);

   /* attributes should be deleted when the communicator is being freed */
   /*
   hypre_MPICommDeleteSendLocation(hypre_ParCSRCommHandleComm(comm_handle));
   hypre_MPICommDeleteRecvLocation(hypre_ParCSRCommHandleComm(comm_handle));
   hypre_MPICommDeleteSendBufferLocation(hypre_ParCSRCommHandleComm(comm_handle));
   hypre_MPICommDeleteRecvBufferLocation(hypre_ParCSRCommHandleComm(comm_handle));
   hypre_MPICommDeleteSendBuffer(hypre_ParCSRCommHandleComm(comm_handle));
   hypre_MPICommDeleteRecvBuffer(hypre_ParCSRCommHandleComm(comm_handle));
   */

   hypre_MPI_Comm_free(&hypre_ParCSRCommHandleComm(comm_handle));
   hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgCreate_core
 *
 * This function does all the communications and computations for
 * hypre_ParCSRCommPkgCreate(hypre_ParCSRMatrix *A) and
 * hypre_BooleanMatvecCommPkgCreate(hypre_ParCSRBooleanMatrix *A)
 *
 * To support both data types, it has hardly any data structures
 * other than HYPRE_Int*.
 *------------------------------------------------------------------*/

void
hypre_ParCSRCommPkgCreate_core(
   /* input args: */
   MPI_Comm   comm,
   HYPRE_BigInt *col_map_offd,
   HYPRE_BigInt  first_col_diag,
   HYPRE_BigInt *col_starts,
   HYPRE_Int  num_cols_diag,
   HYPRE_Int  num_cols_offd,
   /* pointers to output args: */
   HYPRE_Int  *p_num_recvs,
   HYPRE_Int **p_recv_procs,
   HYPRE_Int **p_recv_vec_starts,
   HYPRE_Int  *p_num_sends,
   HYPRE_Int **p_send_procs,
   HYPRE_Int **p_send_map_starts,
   HYPRE_Int **p_send_map_elmts
)
{
   HYPRE_Int    i, j;
   HYPRE_Int    num_procs, my_id, proc_num, num_elmts;
   HYPRE_Int    local_info;
   HYPRE_BigInt offd_col;
   HYPRE_BigInt *big_buf_data = NULL;
   HYPRE_Int    *proc_mark, *proc_add, *tmp, *recv_buf, *displs, *info;
   /* outputs: */
   HYPRE_Int  num_recvs, *recv_procs, *recv_vec_starts;
   HYPRE_Int  num_sends, *send_procs, *send_map_starts, *send_map_elmts;
   HYPRE_Int  ip, vec_start, vec_len, num_requests;

   hypre_MPI_Request *requests = NULL;
   hypre_MPI_Status *status = NULL;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   proc_mark = hypre_CTAlloc(HYPRE_Int,  num_procs, HYPRE_MEMORY_HOST);
   proc_add = hypre_CTAlloc(HYPRE_Int,  num_procs, HYPRE_MEMORY_HOST);
   info = hypre_CTAlloc(HYPRE_Int,  num_procs, HYPRE_MEMORY_HOST);

   /* ----------------------------------------------------------------------
    * determine which processors to receive from (set proc_mark) and num_recvs,
    * at the end of the loop proc_mark[i] contains the number of elements to be
    * received from Proc. i
    * ---------------------------------------------------------------------*/

   proc_num = 0;
   if (num_cols_offd)
   {
      offd_col = col_map_offd[0];
   }

   num_recvs = 0;
   for (i = 0; i < num_cols_offd; i++)
   {
      if (num_cols_diag)
      {
         proc_num = hypre_min(num_procs - 1, (HYPRE_Int)(offd_col / (HYPRE_BigInt)num_cols_diag));
      }

      while (col_starts[proc_num] > offd_col )
      {
         proc_num = proc_num - 1;
      }

      while (col_starts[proc_num + 1] - 1 < offd_col )
      {
         proc_num = proc_num + 1;
      }

      proc_mark[num_recvs] = proc_num;
      j = i;
      while (col_starts[proc_num + 1] > offd_col)
      {
         proc_add[num_recvs]++;
         if (j < num_cols_offd - 1)
         {
            j++;
            offd_col = col_map_offd[j];
         }
         else
         {
            j++;
            offd_col = col_starts[num_procs];
         }
      }
      num_recvs++;

      i = (j < num_cols_offd) ? (j - 1) : j;
   }

   local_info = 2 * num_recvs;

   hypre_MPI_Allgather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, comm);

   /* ----------------------------------------------------------------------
    * generate information to be sent: tmp contains for each recv_proc:
    * id of recv_procs, number of elements to be received for this processor,
    * indices of elements (in this order)
    * ---------------------------------------------------------------------*/

   displs = hypre_CTAlloc(HYPRE_Int,  num_procs + 1, HYPRE_MEMORY_HOST);
   for (i = 1; i < num_procs + 1; i++)
   {
      displs[i] = displs[i - 1] + info[i - 1];
   }
   recv_buf = hypre_CTAlloc(HYPRE_Int,  displs[num_procs], HYPRE_MEMORY_HOST);

   recv_procs = NULL;
   tmp = NULL;
   if (num_recvs)
   {
      recv_procs = hypre_CTAlloc(HYPRE_Int,  num_recvs, HYPRE_MEMORY_HOST);
      tmp = hypre_CTAlloc(HYPRE_Int,  local_info, HYPRE_MEMORY_HOST);
   }
   recv_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_recvs + 1, HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_recvs; i++)
   {
      num_elmts = proc_add[i];
      recv_procs[i] = proc_mark[i];
      recv_vec_starts[i + 1] = recv_vec_starts[i] + num_elmts;
      tmp[j++] = proc_mark[i];
      tmp[j++] = num_elmts;
   }

   hypre_MPI_Allgatherv(tmp, local_info, HYPRE_MPI_INT, recv_buf, info,
                        displs, HYPRE_MPI_INT, comm);

   /* ----------------------------------------------------------------------
    * determine num_sends and number of elements to be sent
    * ---------------------------------------------------------------------*/

   num_sends = 0;
   num_elmts = 0;
   proc_add[0] = 0;
   for (i = 0; i < num_procs; i++)
   {
      j = displs[i];
      while ( j < displs[i + 1])
      {
         if (recv_buf[j++] == my_id)
         {
            proc_mark[num_sends] = i;
            num_sends++;
            proc_add[num_sends] = proc_add[num_sends - 1] + recv_buf[j];
            break;
         }
         j++;
      }
   }

   /* ----------------------------------------------------------------------
    * determine send_procs and actual elements to be send (in send_map_elmts)
    * and send_map_starts whose i-th entry points to the beginning of the
    * elements to be send to proc. i
    * ---------------------------------------------------------------------*/

   send_procs = NULL;
   send_map_elmts = NULL;

   if (num_sends)
   {
      send_procs = hypre_CTAlloc(HYPRE_Int,  num_sends, HYPRE_MEMORY_HOST);
      send_map_elmts = hypre_CTAlloc(HYPRE_Int,  proc_add[num_sends], HYPRE_MEMORY_HOST);
      big_buf_data = hypre_CTAlloc(HYPRE_BigInt,  proc_add[num_sends], HYPRE_MEMORY_HOST);
   }
   send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
   num_requests = num_recvs + num_sends;
   if (num_requests)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request,  num_requests, HYPRE_MEMORY_HOST);
      status = hypre_CTAlloc(hypre_MPI_Status,  num_requests, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_sends; i++)
   {
      send_map_starts[i + 1] = proc_add[i + 1];
      send_procs[i] = proc_mark[i];
   }

   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i + 1] - vec_start;
      ip = send_procs[i];
      hypre_MPI_Irecv(&big_buf_data[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      ip, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i + 1] - vec_start;
      ip = recv_procs[i];
      hypre_MPI_Isend(&col_map_offd[vec_start], vec_len, HYPRE_MPI_BIG_INT,
                      ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      hypre_MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
   }

   if (num_sends)
   {
      for (i = 0; i < send_map_starts[num_sends]; i++)
      {
         send_map_elmts[i] = (HYPRE_Int)(big_buf_data[i] - first_col_diag);
      }
   }

   hypre_TFree(proc_add, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_mark, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(displs, HYPRE_MEMORY_HOST);
   hypre_TFree(info, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);

   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgCreate
 *
 * Creates the communication package with MPI collectives calls.
 *
 * Notes:
 *    1) This version does not use the assumed partition.
 *    2) comm_pkg must be allocated outside of this function
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommPkgCreate( MPI_Comm             comm,
                           HYPRE_BigInt        *col_map_offd,
                           HYPRE_BigInt         first_col_diag,
                           HYPRE_BigInt        *col_starts,
                           HYPRE_Int            num_cols_diag,
                           HYPRE_Int            num_cols_offd,
                           hypre_ParCSRCommPkg *comm_pkg )
{
   HYPRE_Int  num_sends;
   HYPRE_Int *send_procs;
   HYPRE_Int *send_map_starts;
   HYPRE_Int *send_map_elmts;

   HYPRE_Int  num_recvs;
   HYPRE_Int *recv_procs;
   HYPRE_Int *recv_vec_starts;

   hypre_ParCSRCommPkgCreate_core(comm, col_map_offd, first_col_diag,
                                  col_starts, num_cols_diag, num_cols_offd,
                                  &num_recvs, &recv_procs, &recv_vec_starts,
                                  &num_sends, &send_procs, &send_map_starts,
                                  &send_map_elmts);

   /* Fill the communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgCreateAndFill
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommPkgCreateAndFill( MPI_Comm              comm,
                                  HYPRE_Int             num_recvs,
                                  HYPRE_Int            *recv_procs,
                                  HYPRE_Int            *recv_vec_starts,
                                  HYPRE_Int             num_sends,
                                  HYPRE_Int            *send_procs,
                                  HYPRE_Int            *send_map_starts,
                                  HYPRE_Int            *send_map_elmts,
                                  hypre_ParCSRCommPkg **comm_pkg_ptr )
{
   hypre_ParCSRCommPkg  *comm_pkg;

   /* Allocate memory for comm_pkg if needed */
   if (*comm_pkg_ptr == NULL)
   {
      comm_pkg = hypre_TAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   }
   else
   {
      comm_pkg = *comm_pkg_ptr;
   }

   /* Set default info */
   hypre_ParCSRCommPkgNumComponents(comm_pkg)      = 1;
   hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) = NULL;
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_ParCSRCommPkgTmpData(comm_pkg)            = NULL;
   hypre_ParCSRCommPkgBufData(comm_pkg)            = NULL;
   hypre_ParCSRCommPkgMatrixE(comm_pkg)            = NULL;
#endif
#if defined(HYPRE_USING_PERSISTENT_COMM)
   HYPRE_Int i;

   for (i = 0; i < NUM_OF_COMM_PKG_JOB_TYPE; i++)
   {
      comm_pkg->persistent_comm_handles[i] = NULL;
   }
#endif

   /* Set input info */
   hypre_ParCSRCommPkgComm(comm_pkg)          = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

   /* Set output pointer */
   *comm_pkg_ptr = comm_pkg;

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRCommPkgUpdateVecStarts
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRCommPkgUpdateVecStarts( hypre_ParCSRCommPkg *comm_pkg,
                                    HYPRE_Int            num_components_in,
                                    HYPRE_Int            vecstride,
                                    HYPRE_Int            idxstride )
{
   HYPRE_Int     num_components  = hypre_ParCSRCommPkgNumComponents(comm_pkg);
   HYPRE_Int     num_sends       = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int     num_recvs       = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int    *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int    *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int    *send_map_elmts  = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   HYPRE_Int    *send_map_elmts_new;

   HYPRE_Int     i, j;

   hypre_assert(num_components > 0);

   if (num_components_in != num_components)
   {
      /* Update number of components in the communication package */
      hypre_ParCSRCommPkgNumComponents(comm_pkg) = num_components_in;

      /* Allocate send_maps_elmts */
      send_map_elmts_new = hypre_CTAlloc(HYPRE_Int,
                                         send_map_starts[num_sends] * num_components_in,
                                         HYPRE_MEMORY_HOST);

      /* Update send_maps_elmts */
      if (num_components_in > num_components)
      {
         if (num_components == 1)
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_components_in; j++)
               {
                  send_map_elmts_new[i * num_components_in + j] = send_map_elmts[i] * idxstride +
                                                                  j * vecstride;
               }
            }
         }
         else
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_components_in; j++)
               {
                  send_map_elmts_new[i * num_components_in + j] =
                     send_map_elmts[i * num_components] * idxstride + j * vecstride;
               }
            }
         }
      }
      else
      {
         /* num_components_in < num_components */
         if (num_components_in == 1)
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               send_map_elmts_new[i] = send_map_elmts[i * num_components];
            }
         }
         else
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_components_in; j++)
               {
                  send_map_elmts_new[i * num_components_in + j] =
                     send_map_elmts[i * num_components + j];
               }
            }
         }
      }
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts_new;

      /* Free memory */
      hypre_TFree(send_map_elmts, HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg), HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
      hypre_CSRMatrixDestroy(hypre_ParCSRCommPkgMatrixE(comm_pkg));
      hypre_ParCSRCommPkgMatrixE(comm_pkg) = NULL;
#endif

      /* Update send_map_starts */
      for (i = 0; i < num_sends + 1; i++)
      {
         send_map_starts[i] *= num_components_in / num_components;
      }

      /* Update recv_vec_starts */
      for (i = 0; i < num_recvs + 1; i++)
      {
         recv_vec_starts[i] *= num_components_in / num_components;
      }
   }

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_MatvecCommPkgCreate
 *
 * Generates the communication package for A using assumed partition
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A )
{
   MPI_Comm             comm  = hypre_ParCSRMatrixComm(A);
   hypre_IJAssumedPart *apart = hypre_ParCSRMatrixAssumedPartition(A);
   hypre_ParCSRCommPkg *comm_pkg;

   HYPRE_BigInt         first_col_diag  = hypre_ParCSRMatrixFirstColDiag(A);
   HYPRE_BigInt        *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int            num_cols_offd   = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_BigInt         global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Create the assumed partition and should own it */
   if (apart == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(A);
      hypre_ParCSRMatrixOwnsAssumedPartition(A) = 1;
      apart = hypre_ParCSRMatrixAssumedPartition(A);
   }

   /*-----------------------------------------------------------
    * setup commpkg
    *----------------------------------------------------------*/

   comm_pkg = hypre_TAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;
   hypre_ParCSRCommPkgCreateApart( comm, col_map_offd, first_col_diag,
                                   num_cols_offd, global_num_cols,
                                   apart,
                                   comm_pkg );

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_MatvecCommPkgDestroy
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_MatvecCommPkgDestroy( hypre_ParCSRCommPkg *comm_pkg )
{
#ifdef HYPRE_USING_PERSISTENT_COMM
   HYPRE_Int i;
   for (i = HYPRE_COMM_PKG_JOB_COMPLEX; i < NUM_OF_COMM_PKG_JOB_TYPE; ++i)
   {
      if (hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg, i))
      {
         hypre_ParCSRPersistentCommHandleDestroy(hypre_ParCSRCommPkgPersistentCommHandle(comm_pkg, i));
      }
   }
#endif

   if (hypre_ParCSRCommPkgNumSends(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendProcs(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(comm_pkg), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg), HYPRE_MEMORY_DEVICE);
   }
   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg), HYPRE_MEMORY_HOST);
   /* if (hypre_ParCSRCommPkgSendMPITypes(comm_pkg))
      hypre_TFree(hypre_ParCSRCommPkgSendMPITypes(comm_pkg), HYPRE_MEMORY_HOST); */
   if (hypre_ParCSRCommPkgNumRecvs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvProcs(comm_pkg), HYPRE_MEMORY_HOST);
   }
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg), HYPRE_MEMORY_HOST);
   /* if (hypre_ParCSRCommPkgRecvMPITypes(comm_pkg))
      hypre_TFree(hypre_ParCSRCommPkgRecvMPITypes(comm_pkg), HYPRE_MEMORY_HOST); */

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_TFree(hypre_ParCSRCommPkgTmpData(comm_pkg), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_ParCSRCommPkgBufData(comm_pkg), HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixDestroy(hypre_ParCSRCommPkgMatrixE(comm_pkg));
#endif

   hypre_TFree(comm_pkg, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_ParCSRFindExtendCommPkg
 *
 * AHB 11/06 : alternate to the extend function below - creates a
 * second comm pkg based on indices - this makes it easier to use the
 * global partition
 *
 * RL: renamed and moved it here
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRFindExtendCommPkg(MPI_Comm              comm,
                              HYPRE_BigInt          global_num,
                              HYPRE_BigInt          my_first,
                              HYPRE_Int             local_num,
                              HYPRE_BigInt         *starts,
                              hypre_IJAssumedPart  *apart,
                              HYPRE_Int             indices_len,
                              HYPRE_BigInt         *indices,
                              hypre_ParCSRCommPkg **extend_comm_pkg)
{
   HYPRE_UNUSED_VAR(local_num);
   HYPRE_UNUSED_VAR(starts);

   hypre_ParCSRCommPkg *new_comm_pkg = hypre_TAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_assert(apart != NULL);
   hypre_ParCSRCommPkgCreateApart(comm, indices, my_first, indices_len,
                                  global_num, apart, new_comm_pkg);

   *extend_comm_pkg = new_comm_pkg;

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_BuildCSRMatrixMPIDataType
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_BuildCSRMatrixMPIDataType( HYPRE_Int num_nonzeros,
                                 HYPRE_Int num_rows,
                                 HYPRE_Complex *a_data,
                                 HYPRE_Int *a_i,
                                 HYPRE_Int *a_j,
                                 hypre_MPI_Datatype *csr_matrix_datatype )
{
   HYPRE_Int            block_lens[3];
   hypre_MPI_Aint       displ[3];
   hypre_MPI_Datatype   types[3];

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_rows + 1;
   block_lens[2] = num_nonzeros;

   types[0] = HYPRE_MPI_COMPLEX;
   types[1] = HYPRE_MPI_INT;
   types[2] = HYPRE_MPI_INT;

   hypre_MPI_Address(a_data, &displ[0]);
   hypre_MPI_Address(a_i, &displ[1]);
   hypre_MPI_Address(a_j, &displ[2]);
   hypre_MPI_Type_struct(3, block_lens, displ, types, csr_matrix_datatype);
   hypre_MPI_Type_commit(csr_matrix_datatype);

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_BuildCSRMatrixMPIDataType
 *------------------------------------------------------------------*/

HYPRE_Int
hypre_BuildCSRJDataType( HYPRE_Int num_nonzeros,
                         HYPRE_Complex *a_data,
                         HYPRE_Int *a_j,
                         hypre_MPI_Datatype *csr_jdata_datatype )
{
   HYPRE_Int          block_lens[2];
   hypre_MPI_Aint     displs[2];
   hypre_MPI_Datatype types[2];

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_nonzeros;

   types[0] = HYPRE_MPI_COMPLEX;
   types[1] = HYPRE_MPI_INT;

   hypre_MPI_Address(a_data, &displs[0]);
   hypre_MPI_Address(a_j, &displs[1]);

   hypre_MPI_Type_struct(2, block_lens, displs, types, csr_jdata_datatype);
   hypre_MPI_Type_commit(csr_jdata_datatype);

   return hypre_error_flag;
}
