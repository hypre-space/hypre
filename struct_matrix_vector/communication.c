/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"

/*==========================================================================*/
/*==========================================================================*/
/** Create a communication package.  A grid-based description of a
communication exchange is passed in.  This description is then
compiled into an intermediate processor-based description of the
communication.  It may further be compiled into a form based on the
message-passing layer in the routine hypre\_CommitCommPkg.  This
proceeds as follows based on several compiler flags:

\begin{itemize}
\item If HYPRE\_COMM\_SIMPLE is defined, the intermediate
processor-based description is not compiled into a form based on
the message-passing layer.  This intermediate description is used
directly to pack and unpack buffers during the communications.
No MPI derived datatypes are used.
\item Else if HYPRE\_COMM\_VOLATILE is defined, the communication
package is not committed, and the intermediate processor-based
description is retained.  The package is committed at communication
time.
\item Else the communication package is committed, and the intermediate
processor-based description is freed up.
\end{itemize}

{\bf Note:}
The input boxes and processes are destroyed.

{\bf Input files:}
headers.h

@return Communication package.

@param send_boxes [IN]
  description of the grid data to be sent to other processors.
@param recv_boxes [IN]
  description of the grid data to be received from other processors.
@param send_data_space [IN]
  description of the stored grid data associated with the sends.
@param recv_data_space [IN]
  description of the stored grid data associated with the receives.
@param send_processes [IN]
  processors that data is to be sent to.
@param recv_processes [IN]
  processors that data is to be received from.
@param num_values [IN]
  number of data values to be sent for each grid index.
@param comm [IN]
  communicator.

@see hypre_CommPkgCreateInfo, hypre_CommPkgCommit, hypre_CommPkgDestroy */
/*--------------------------------------------------------------------------*/

  hypre_CommPkg *
  hypre_CommPkgCreate( hypre_BoxArrayArray   *send_boxes,
                       hypre_BoxArrayArray   *recv_boxes,
                       hypre_Index            send_stride,
                       hypre_Index            recv_stride,
                       hypre_BoxArray        *send_data_space,
                       hypre_BoxArray        *recv_data_space,
                       int                  **send_processes,
                       int                  **recv_processes,
                       int                    num_values,
                       MPI_Comm               comm,
                       hypre_Index            periodic            )
{
   hypre_CommPkg    *comm_pkg;
                  
   int               num_sends;
   int              *send_procs;
   hypre_CommType  **send_types;
   int               num_recvs;
   int              *recv_procs;
   hypre_CommType  **recv_types;

   hypre_CommType   *copy_from_type;
   hypre_CommType   *copy_to_type;

   int               i;

   /*------------------------------------------------------
    * Put arguments into hypre_CommPkg
    *------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);

   hypre_CommPkgNumValues(comm_pkg)     = num_values;
   hypre_CommPkgComm(comm_pkg)          = comm;

   /*------------------------------------------------------
    * Set up communication information
    *------------------------------------------------------*/

   hypre_CommPkgCreateInfo(send_boxes, send_stride,
                           send_data_space, send_processes,
                           num_values, comm, periodic,
                           &num_sends, &send_procs,
                           &send_types, &copy_from_type);

   hypre_CommPkgNumSends(comm_pkg)     = num_sends;
   hypre_CommPkgSendProcs(comm_pkg)    = send_procs;
   hypre_CommPkgSendTypes(comm_pkg)    = send_types;
   hypre_CommPkgCopyFromType(comm_pkg) = copy_from_type;

   hypre_CommPkgCreateInfo(recv_boxes, recv_stride,
                           recv_data_space, recv_processes,
                           num_values, comm,  periodic,
                           &num_recvs, &recv_procs,
                           &recv_types, &copy_to_type);

   hypre_CommPkgNumRecvs(comm_pkg)   = num_recvs;
   hypre_CommPkgRecvProcs(comm_pkg)  = recv_procs;
   hypre_CommPkgRecvTypes(comm_pkg)  = recv_types;
   hypre_CommPkgCopyToType(comm_pkg) = copy_to_type;

   /*------------------------------------------------------
    * Destroy the input boxes and processes
    *------------------------------------------------------*/

   hypre_ForBoxArrayI(i, send_boxes)
      hypre_TFree(send_processes[i]);
   hypre_BoxArrayArrayDestroy(send_boxes);
   hypre_TFree(send_processes);

   hypre_ForBoxArrayI(i, recv_boxes)
      hypre_TFree(recv_processes[i]);
   hypre_BoxArrayArrayDestroy(recv_boxes);
   hypre_TFree(recv_processes);

#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
#else
   hypre_CommPkgCommit(comm_pkg);

   /* free up comm types */
   for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
      hypre_CommTypeDestroy(hypre_CommPkgSendType(comm_pkg, i));
   hypre_TFree(hypre_CommPkgSendTypes(comm_pkg));
   for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
      hypre_CommTypeDestroy(hypre_CommPkgRecvType(comm_pkg, i));
   hypre_TFree(hypre_CommPkgRecvTypes(comm_pkg));
#endif

   return comm_pkg;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroy a communication package.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN/OUT]
  communication package.

@see hypre_CommPkgCreate */
/*--------------------------------------------------------------------------*/

int
hypre_CommPkgDestroy( hypre_CommPkg *comm_pkg )
{
   int ierr = 0;
#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
   int               i;
#else
#endif

   if (comm_pkg)
   {
#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
      /* free up comm types */
      for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
         hypre_CommTypeDestroy(hypre_CommPkgSendType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgSendTypes(comm_pkg));
      for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
         hypre_CommTypeDestroy(hypre_CommPkgRecvType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgRecvTypes(comm_pkg));
#else
      hypre_CommPkgUnCommit(comm_pkg);
#endif

      hypre_TFree(hypre_CommPkgSendProcs(comm_pkg));
      hypre_TFree(hypre_CommPkgRecvProcs(comm_pkg));

      hypre_CommTypeDestroy(hypre_CommPkgCopyFromType(comm_pkg));
      hypre_CommTypeDestroy(hypre_CommPkgCopyToType(comm_pkg));

      hypre_TFree(comm_pkg);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Initialize a non-blocking communication exchange.

\begin{itemize}
\item If HYPRE\_COMM\_SIMPLE is defined, the communication buffers are
created, the send buffer is manually packed, and the communication
requests are posted.  No MPI derived datatypes are used.
\item Else if HYPRE\_COMM\_VOLATILE is defined, the communication
package is committed, the communication requests are posted, then
the communication package is un-committed.
\item Else the communication requests are posted.
\end{itemize}

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN]
  communication package.
@param send_data [IN]
  reference pointer for the send data.
@param recv_data [IN]
  reference pointer for the recv data.
@param comm_handle [OUT]
  communication handle.

@see hypre_FinalizeCommunication, hypre_CommPkgCreate */
/*--------------------------------------------------------------------------*/

#if defined(HYPRE_COMM_SIMPLE)

int
hypre_InitializeCommunication( hypre_CommPkg     *comm_pkg,
                               double            *send_data,
                               double            *recv_data,
                               hypre_CommHandle **comm_handle_ptr )
{
   int                  ierr = 0;
                     
   hypre_CommHandle    *comm_handle;
                     
   int                  num_sends = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_CommPkgComm(comm_pkg);
                     
   int                  num_requests;
   MPI_Request         *requests;
   MPI_Status          *status;
   double             **send_buffers;
   double             **recv_buffers;
   int                 *send_sizes;
   int                 *recv_sizes;

   hypre_CommType      *send_type;
   hypre_CommTypeEntry *send_entry;
   hypre_CommType      *recv_type;
   hypre_CommTypeEntry *recv_entry;

   int                 *length_array;
   int                 *stride_array;

   double              *iptr, *jptr, *kptr, *lptr, *bptr;

   int                  i, j, k, ii, jj, kk, ll;
   int                  entry_size, total_size;
                      
   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status   = hypre_CTAlloc(MPI_Status, num_requests);

   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = hypre_TAlloc(double *, num_sends);
   send_sizes = hypre_TAlloc(int, num_sends);
   total_size = 0;
   for (i = 0; i < num_sends; i++)
   {
      send_type = hypre_CommPkgSendType(comm_pkg, i);

      send_sizes[i] = 0;
      for (j = 0; j < hypre_CommTypeNumEntries(send_type); j++)
      {
         send_entry = hypre_CommTypeCommEntry(send_type, j);
         length_array = hypre_CommTypeEntryLengthArray(send_entry);

         entry_size = 1;
         for (k = 0; k < 4; k++)
         {
            entry_size *= length_array[k];
         }
         send_sizes[i] += entry_size;
      }

      total_size += send_sizes[i];
   }
   if (num_sends > 0)
   {
      send_buffers[0] = hypre_SharedTAlloc(double, total_size);
      for (i = 1; i < num_sends; i++)
      {
         send_buffers[i] = send_buffers[i-1] + send_sizes[i-1];
      }
   }

   /* allocate recv buffers */
   recv_buffers = hypre_TAlloc(double *, num_recvs);
   recv_sizes = hypre_TAlloc(int, num_recvs);
   total_size = 0;
   for (i = 0; i < num_recvs; i++)
   {
      recv_type = hypre_CommPkgRecvType(comm_pkg, i);

      recv_sizes[i] = 0;
      for (j = 0; j < hypre_CommTypeNumEntries(recv_type); j++)
      {
         recv_entry = hypre_CommTypeCommEntry(recv_type, j);
         length_array = hypre_CommTypeEntryLengthArray(recv_entry);

         entry_size = 1;
         for (k = 0; k < 4; k++)
         {
            entry_size *= length_array[k];
         }
         recv_sizes[i] += entry_size;
      }

      total_size += recv_sizes[i];
   }
   if (num_recvs > 0)
   {
      recv_buffers[0] = hypre_SharedTAlloc(double, total_size);
      for (i = 1; i < num_recvs; i++)
      {
         recv_buffers[i] = recv_buffers[i-1] + recv_sizes[i-1];
      }
   }

   /*--------------------------------------------------------------------
    * pack send buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      send_type = hypre_CommPkgSendType(comm_pkg, i);

      bptr = (double *) send_buffers[i];
      for (j = 0; j < hypre_CommTypeNumEntries(send_type); j++)
      {
         send_entry = hypre_CommTypeCommEntry(send_type, j);
         length_array = hypre_CommTypeEntryLengthArray(send_entry);
         stride_array = hypre_CommTypeEntryStrideArray(send_entry);

         lptr = send_data + hypre_CommTypeEntryOffset(send_entry);
         for (ll = 0; ll < length_array[3]; ll++)
         {
            kptr = lptr;
            for (kk = 0; kk < length_array[2]; kk++)
            {
               jptr = kptr;
               for (jj = 0; jj < length_array[1]; jj++)
               {
                  if (stride_array[0] == 1)
                  {
                     memcpy(bptr, jptr, length_array[0]*sizeof(double));
                  }
                  else
                  {
                     iptr = jptr;
                     for (ii = 0; ii < length_array[0]; ii++)
                     {
                        bptr[ii] = *iptr;
                        iptr += stride_array[0];
                     }
                  }
                  bptr += length_array[0];
                  jptr += stride_array[1];
               }
               kptr += stride_array[2];
            }
            lptr += stride_array[3];
         }
      }
   }

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   j = 0;
   for(i = 0; i < num_recvs; i++)
   {
      MPI_Irecv(recv_buffers[i], recv_sizes[i], MPI_DOUBLE, 
                hypre_CommPkgRecvProc(comm_pkg, i), 
                0, comm, &requests[j++]);
   }
   for(i = 0; i < num_sends; i++)
   {
      MPI_Isend(send_buffers[i], send_sizes[i], MPI_DOUBLE, 
                hypre_CommPkgSendProc(comm_pkg, i), 
                0, comm, &requests[j++]);
   }

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;
   hypre_CommHandleStatus(comm_handle)      = status;
   hypre_CommHandleSendBuffers(comm_handle) = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle) = recv_buffers;
   hypre_CommHandleSendSizes(comm_handle)   = send_sizes;
   hypre_CommHandleRecvSizes(comm_handle)   = recv_sizes;

   *comm_handle_ptr = comm_handle;

   return ierr;
}

/*--------------------------------------------------------------------------*/

#else

int
hypre_InitializeCommunication( hypre_CommPkg     *comm_pkg,
                               double            *send_data,
                               double            *recv_data,
                               hypre_CommHandle **comm_handle_ptr )
{
   int                  ierr = 0;

   hypre_CommHandle    *comm_handle;

   int                  num_sends  = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs  = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = hypre_CommPkgComm(comm_pkg);
                      
   int                  num_requests;
   MPI_Request         *requests;
   MPI_Status          *status;
                     
   int                  i, j;
                      
   /*--------------------------------------------------------------------
    * allocate requests and status
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status   = hypre_CTAlloc(MPI_Status, num_requests);

   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

#if defined(HYPRE_COMM_VOLATILE)
   /* commit the communication package */
   hypre_CommPkgCommit(comm_pkg);
#else
#endif

   j = 0;
   for(i = 0; i < num_recvs; i++)
   {
      MPI_Irecv((void *)recv_data, 1,
                hypre_CommPkgRecvMPIType(comm_pkg, i), 
                hypre_CommPkgRecvProc(comm_pkg, i), 
                0, comm, &requests[j++]);
   }
   for(i = 0; i < num_sends; i++)
   {
      MPI_Isend((void *)send_data, 1,
                hypre_CommPkgSendMPIType(comm_pkg, i), 
                hypre_CommPkgSendProc(comm_pkg, i), 
                0, comm, &requests[j++]);
   }

#if defined(HYPRE_COMM_VOLATILE)
   /* un-commit the communication package */
   hypre_CommPkgUnCommit(comm_pkg);
#else
#endif

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_TAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;
   hypre_CommHandleStatus(comm_handle)      = status;

   *comm_handle_ptr = comm_handle;

   return ierr;
}

#endif

/*==========================================================================*/
/*==========================================================================*/
/** Finalize a communication exchange.  This routine blocks until all
of the communication requests are completed.

\begin{itemize}
\item If HYPRE\_COMM\_SIMPLE is defined, the communication requests
are completed, and the receive buffer is manually unpacked.
\item Else if HYPRE\_COMM\_VOLATILE is defined, the communication requests
are completed and the communication package is un-committed.
\item Else the communication requests are completed.
\end{itemize}

{\bf Input files:}
headers.h

@return Error code.

@param comm_handle [IN/OUT]
  communication handle.

@see hypre_InitializeCommunication, hypre_CommPkgCreate */
/*--------------------------------------------------------------------------*/

#if defined(HYPRE_COMM_SIMPLE)

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   
   int              ierr = 0;

   hypre_CommPkg   *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
   double         **send_buffers = hypre_CommHandleSendBuffers(comm_handle);
   double         **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);
   int             *send_sizes   = hypre_CommHandleSendSizes(comm_handle);
   int             *recv_sizes   = hypre_CommHandleRecvSizes(comm_handle);
   int              num_sends    = hypre_CommPkgNumSends(comm_pkg);
   int              num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   hypre_CommType      *recv_type;
   hypre_CommTypeEntry *recv_entry;

   int                 *length_array;
   int                 *stride_array;

   double              *iptr, *jptr, *kptr, *lptr, *bptr;

   int                  i, j, ii, jj, kk, ll;

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

   if (hypre_CommHandleNumRequests(comm_handle))
   {
      MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                  hypre_CommHandleRequests(comm_handle),
                  hypre_CommHandleStatus(comm_handle));
   }

   /*--------------------------------------------------------------------
    * unpack recv buffers
    *--------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      recv_type = hypre_CommPkgRecvType(comm_pkg, i);

      bptr = (double *) recv_buffers[i];
      for (j = 0; j < hypre_CommTypeNumEntries(recv_type); j++)
      {
         recv_entry = hypre_CommTypeCommEntry(recv_type, j);
         length_array = hypre_CommTypeEntryLengthArray(recv_entry);
         stride_array = hypre_CommTypeEntryStrideArray(recv_entry);

         lptr = hypre_CommHandleRecvData(comm_handle) +
            hypre_CommTypeEntryOffset(recv_entry);
         for (ll = 0; ll < length_array[3]; ll++)
         {
            kptr = lptr;
            for (kk = 0; kk < length_array[2]; kk++)
            {
               jptr = kptr;
               for (jj = 0; jj < length_array[1]; jj++)
               {
                  if (stride_array[0] == 1)
                  {
                     memcpy(jptr, bptr, length_array[0]*sizeof(double));
                  }
                  else
                  {
                     iptr = jptr;
                     for (ii = 0; ii < length_array[0]; ii++)
                     {
                        *iptr = bptr[ii];
                        iptr += stride_array[0];
                     }
                  }
                  bptr += length_array[0];
                  jptr += stride_array[1];
               }
               kptr += stride_array[2];
            }
            lptr += stride_array[3];
         }
      }
   }

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/

   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(hypre_CommHandleStatus(comm_handle));
   if (num_sends > 0)
   {
      hypre_SharedTFree(send_buffers[0]);
   }
   if (num_recvs > 0)
   {
      hypre_SharedTFree(recv_buffers[0]);
   }
   hypre_TFree(send_buffers);
   hypre_TFree(recv_buffers);
   hypre_TFree(send_sizes);
   hypre_TFree(recv_sizes);
   hypre_TFree(comm_handle);

   return ierr;
}

#else

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   int  ierr = 0;

   if (hypre_CommHandleNumRequests(comm_handle))
   {
      MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                  hypre_CommHandleRequests(comm_handle),
                  hypre_CommHandleStatus(comm_handle));
   }

   /*--------------------------------------------------------------------
    * Free up communication handle
    *--------------------------------------------------------------------*/

   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(hypre_CommHandleStatus(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

#endif

/*==========================================================================*/
/*==========================================================================*/
/** Execute local data exchanges.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN]
  communication package.
@param send_data [IN]
  reference pointer for the send data.
@param recv_data [IN]
  reference pointer for the recv data.

@see hypre_InitializeCommunication */
/*--------------------------------------------------------------------------*/

int
hypre_ExchangeLocalData( hypre_CommPkg *comm_pkg,
                         double        *send_data,
                         double        *recv_data )
{
   hypre_CommType      *copy_from_type;
   hypre_CommType      *copy_to_type;
   hypre_CommTypeEntry *copy_from_entry;
   hypre_CommTypeEntry *copy_to_entry;

   double              *from_dp;
   int                 *from_stride_array;
   int                  from_i;
   double              *to_dp;
   int                 *to_stride_array;
   int                  to_i;
                      
   int                 *length_array;
   int                  i0, i1, i2, i3;

   int                  i;
   int                  ierr = 0;

   /*--------------------------------------------------------------------
    * copy local data
    *--------------------------------------------------------------------*/

   copy_from_type = hypre_CommPkgCopyFromType(comm_pkg);
   copy_to_type   = hypre_CommPkgCopyToType(comm_pkg);

   for (i = 0; i < hypre_CommTypeNumEntries(copy_from_type); i++)
   {
      copy_from_entry = hypre_CommTypeCommEntry(copy_from_type, i);
      copy_to_entry   = hypre_CommTypeCommEntry(copy_to_type, i);

      from_dp = send_data + hypre_CommTypeEntryOffset(copy_from_entry);
      to_dp   = recv_data + hypre_CommTypeEntryOffset(copy_to_entry);

      /* copy data only when necessary */
      if (to_dp != from_dp)
      {
         length_array = hypre_CommTypeEntryLengthArray(copy_from_entry);

         from_stride_array = hypre_CommTypeEntryStrideArray(copy_from_entry);
         to_stride_array = hypre_CommTypeEntryStrideArray(copy_to_entry);

         for (i3 = 0; i3 < length_array[3]; i3++)
         {
            for (i2 = 0; i2 < length_array[2]; i2++)
            {
               for (i1 = 0; i1 < length_array[1]; i1++)
               {
                  from_i = (i3*from_stride_array[3] +
                            i2*from_stride_array[2] +
                            i1*from_stride_array[1]  );
                  to_i = (i3*to_stride_array[3] +
                          i2*to_stride_array[2] +
                          i1*to_stride_array[1]  );
                  for (i0 = 0; i0 < length_array[0]; i0++)
                  {
                     to_dp[to_i] = from_dp[from_i];

                     from_i += from_stride_array[0];
                     to_i += to_stride_array[0];
                  }
               }
            }
         }
      }
   }

   return ( ierr );
}

/*==========================================================================*/
/*==========================================================================*/
/** Create a communication type.

{\bf Input files:}
headers.h

@return Communication type.

@param comm_entries [IN]
  array of pointers to communication type entries.
@param num_entries [IN]
  number of elements in comm\_entries array.

@see hypre_CommTypeDestroy */
/*--------------------------------------------------------------------------*/

hypre_CommType *
hypre_CommTypeCreate( hypre_CommTypeEntry **comm_entries,
                      int                   num_entries  )
{
   hypre_CommType      *comm_type;

   comm_type = hypre_TAlloc(hypre_CommType, 1);

   hypre_CommTypeCommEntries(comm_type) = comm_entries;
   hypre_CommTypeNumEntries(comm_type)  = num_entries;

   return comm_type;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroy a communication type.

{\bf Input files:}
headers.h

@return Error code.

@param comm_type [IN]
  communication type.

@see hypre_CommTypeCreate */
/*--------------------------------------------------------------------------*/

int 
hypre_CommTypeDestroy( hypre_CommType *comm_type )
{
   int                   ierr = 0;
   hypre_CommTypeEntry  *comm_entry;
   int                   i;

   if (comm_type)
   {
      if ( hypre_CommTypeCommEntries(comm_type) != NULL )
      {
         for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
         {
            comm_entry = hypre_CommTypeCommEntry(comm_type, i);
            hypre_CommTypeEntryDestroy(comm_entry);
         }
      }

      hypre_TFree(hypre_CommTypeCommEntries(comm_type));
      hypre_TFree(comm_type);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Create a communication type entry.

{\bf Input files:}
headers.h

@return Communication type entry.

@param box [IN]
  description of the grid data to be communicated.
@param data_box [IN]
  description of the stored grid data.
@param num_values [IN]
  number of data values to be communicated for each grid index.
@param data_box_offset [IN]
  offset from some location in memory of the data associated with the
  imin index of data_box.

@see hypre_CommTypeEntryDestroy */
/*--------------------------------------------------------------------------*/

hypre_CommTypeEntry *
hypre_CommTypeEntryCreate( hypre_Box   *box,
                           hypre_Index  stride,
                           hypre_Box   *data_box,
                           int          num_values,
                           int          data_box_offset )
{
   hypre_CommTypeEntry  *comm_entry;

   int                  *length_array;
   int                  *stride_array;
                       
   hypre_Index           size;
   int                   i, j, dim;

   comm_entry = hypre_TAlloc(hypre_CommTypeEntry, 1);

   /*------------------------------------------------------
    * Set imin, imax, and offset
    *------------------------------------------------------*/

   hypre_CopyIndex(hypre_BoxIMin(box),
                   hypre_CommTypeEntryIMin(comm_entry));
   hypre_CopyIndex(hypre_BoxIMax(box),
                   hypre_CommTypeEntryIMax(comm_entry));

   hypre_CommTypeEntryOffset(comm_entry) =
      data_box_offset + hypre_BoxIndexRank(data_box, hypre_BoxIMin(box));

   /*------------------------------------------------------
    * Set length_array, stride_array, and dim
    *------------------------------------------------------*/

   length_array = hypre_CommTypeEntryLengthArray(comm_entry);
   stride_array = hypre_CommTypeEntryStrideArray(comm_entry);
 
   /* initialize length_array */
   hypre_BoxGetStrideSize(box, stride, size);
   for (i = 0; i < 3; i++)
      length_array[i] = hypre_IndexD(size, i);
   length_array[3] = num_values;

   /* initialize stride_array */
   for (i = 0; i < 3; i++)
   {
      stride_array[i] = hypre_IndexD(stride, i);
      for (j = 0; j < i; j++)
         stride_array[i] *= hypre_BoxSizeD(data_box, j);
   }
   stride_array[3] = hypre_BoxVolume(data_box);

   /* eliminate dimensions with length_array = 1 */
   dim = 4;
   i = 0;
   while (i < dim)
   {
      if(length_array[i] == 1)
      {
         for(j = i; j < (dim - 1); j++)
         {
            length_array[j] = length_array[j+1];
            stride_array[j] = stride_array[j+1];
         }
         length_array[dim - 1] = 1;
         stride_array[dim - 1] = 1;
         dim--;
      }
      else
      {
         i++;
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

   hypre_CommTypeEntryDim(comm_entry) = dim;
 
   return comm_entry;
}
 
/*==========================================================================*/
/*==========================================================================*/
/** Destroy a communication type entry.

{\bf Input files:}
headers.h

@return Error code.

@param comm_entry [IN/OUT]
  communication type entry.

@see hypre_CommTypeEntryCreate */
/*--------------------------------------------------------------------------*/

int
hypre_CommTypeEntryDestroy( hypre_CommTypeEntry *comm_entry )
{
   int ierr = 0;

   if (comm_entry)
   {
      hypre_TFree(comm_entry);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Compute a processor-based description of a communication from a
grid-based one.  Used to construct a communication package.

{\bf Input files:}
headers.h

@return Error code.

@param boxes [IN]
  description of the grid data to be communicated to other processors.
@param data_space [IN]
  description of the stored grid data associated with the communications.
@param processes [IN]
  processors that data is to be communicated with.
@param num_values [IN]
  number of data values to be communicated for each grid index.
@param comm [IN]
  communicator.
@param num_comms_ptr [OUT]
  number of communications.  The number of communications is defined
  by the number of processors involved in the communications, not
  counting ``my processor''.
@param comm_processes_ptr [OUT]
  processor 
ranks involved in the communications.
@param comm_types_ptr [OUT]
  inter-processor communication types.
@param copy_type_ptr [OUT]
  intra-processor communication type (copies).

@see hypre_CommPkgCreate, hypre_CommTypeSort */
/*--------------------------------------------------------------------------*/

int
hypre_CommPkgCreateInfo( hypre_BoxArrayArray   *boxes,
                         hypre_Index            stride,
                         hypre_BoxArray        *data_space,
                         int                  **processes,
                         int                    num_values,
                         MPI_Comm               comm,
                         hypre_Index            periodic,
                         int                   *num_comms_ptr,
                         int                  **comm_processes_ptr,
                         hypre_CommType      ***comm_types_ptr,
                         hypre_CommType       **copy_type_ptr)
{
   int                    num_comms;
   int                   *comm_processes;
   hypre_CommType       **comm_types;
   hypre_CommType        *copy_type;

   hypre_CommTypeEntry ***comm_entries;
   int                   *num_entries;
                    
   hypre_BoxArray        *box_array;
   hypre_Box             *box;
   hypre_Box             *data_box;
   int                    data_box_offset;
                        
   int                    i, j, p, m;
   int                    num_procs, my_proc;
                        
   int                    ierr = 0;
                
   /*---------------------------------------------------------
    * Misc stuff
    *---------------------------------------------------------*/

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &my_proc );

   /*------------------------------------------------------
    * Loop over boxes and compute num_entries.
    *------------------------------------------------------*/

   num_entries = hypre_CTAlloc(int, num_procs);

   num_comms = 0;
   hypre_ForBoxArrayI(i, boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(boxes, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               p = processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  num_entries[p]++;
                  if ((num_entries[p] == 1) && (p != my_proc))
                  {
                     num_comms++;
                  }
               }
            }
      }

   /*------------------------------------------------------
    * Loop over boxes and compute comm_entries
    * and comm_processes.
    *------------------------------------------------------*/

   comm_entries = hypre_CTAlloc(hypre_CommTypeEntry **, num_procs);
   comm_processes  = hypre_TAlloc(int, num_comms);

   m = 0;
   data_box_offset = 0;
   hypre_ForBoxArrayI(i, boxes)
      {
         box_array = hypre_BoxArrayArrayBoxArray(boxes, i);
         data_box = hypre_BoxArrayBox(data_space, i);

         hypre_ForBoxI(j, box_array)
            {
               box = hypre_BoxArrayBox(box_array, j);
               p = processes[i][j];

               if (hypre_BoxVolume(box) != 0)
               {
                  /* allocate comm_entries pointer */
                  if (comm_entries[p] == NULL)
                  {
                     comm_entries[p] =
                        hypre_CTAlloc(hypre_CommTypeEntry *, num_entries[p]);
                     num_entries[p] = 0;

                     if (p != my_proc)
                     {
                        comm_processes[m] = p;
                        m++;
                     }
                  }

                  comm_entries[p][num_entries[p]] =
                     hypre_CommTypeEntryCreate(box, stride, data_box,
                                               num_values, data_box_offset);

                  num_entries[p]++;
               }
            }

         data_box_offset += hypre_BoxVolume(data_box) * num_values;
      }

   /*------------------------------------------------------
    * Loop over comm_entries and build comm_types
    *------------------------------------------------------*/

   comm_types = hypre_TAlloc(hypre_CommType *, num_comms);

   for (m = 0; m < num_comms; m++)
   {
      p = comm_processes[m];
      comm_types[m] = hypre_CommTypeCreate(comm_entries[p], num_entries[p]);
      hypre_CommTypeSort(comm_types[m], periodic);
   }

   /*------------------------------------------------------
    * Build copy_type
    *------------------------------------------------------*/

   if (comm_entries[my_proc] != NULL)
   {
      p = my_proc;
      copy_type = hypre_CommTypeCreate(comm_entries[p], num_entries[p]);
      hypre_CommTypeSort(copy_type, periodic);
   }
   else
   {
      copy_type = hypre_CommTypeCreate(NULL, 0);
   }

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   hypre_TFree(comm_entries);
   hypre_TFree(num_entries);

   *num_comms_ptr      = num_comms;
   *comm_processes_ptr = comm_processes;
   *comm_types_ptr     = comm_types;
   *copy_type_ptr      = copy_type;

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Sort the entries of a communication type.  This routine is used to
maintain consistency in communications.

{\bf Input files:}
headers.h

{\bf Note:}
The entries are sorted by imin first.  Entries with common imin are
then sorted by imax.  This assumes that imin and imax define a unique
communication type.

@return Error code.

@param comm_type [IN/OUT]
  communication type to be sorted.

@see hypre_CommPkgCreateInfo */
/*--------------------------------------------------------------------------*/

int
hypre_CommTypeSort( hypre_CommType  *comm_type,
                    hypre_Index      periodic )
{
   hypre_CommTypeEntry  **comm_entries = hypre_CommTypeCommEntries(comm_type);
   int                    num_entries  = hypre_CommTypeNumEntries(comm_type);

   hypre_CommTypeEntry   *comm_entry;
   hypre_IndexRef         imin0, imin1;
   int                   *imax0, *imax1;
   int                    swap;
   int                    i, j, ii, jj;
   int                    ierr = 0;
                      
#if 1
   /*------------------------------------------------
    * Sort by imin:
    *------------------------------------------------*/

   for (i = (num_entries - 1); i > 0; i--)
   {
      for (j = 0; j < i; j++)
      {
         swap = 0;
         imin0 = hypre_CommTypeEntryIMin(comm_entries[j]);
         imin1 = hypre_CommTypeEntryIMin(comm_entries[j+1]);
         if ( hypre_IModPeriodZ(imin0, periodic) > 
              hypre_IModPeriodZ(imin1, periodic) )
         {
            swap = 1;
         }
         else if ( hypre_IModPeriodZ(imin0, periodic) == 
                   hypre_IModPeriodZ(imin1, periodic) )
         {
            if ( hypre_IModPeriodY(imin0, periodic) > 
                 hypre_IModPeriodY(imin1, periodic) )
            {
               swap = 1;
            }
            else if ( hypre_IModPeriodY(imin0, periodic) == 
                      hypre_IModPeriodY(imin1, periodic) )
            {
               if ( hypre_IModPeriodX(imin0, periodic) > 
                    hypre_IModPeriodX(imin1, periodic) )
               {
                  swap = 1;
               }
            }
         }

         if (swap)
         {
            comm_entry        = comm_entries[j];
            comm_entries[j]   = comm_entries[j+1];
            comm_entries[j+1] = comm_entry;
         }
      }
   }

   /*------------------------------------------------
    * Sort entries with common imin by imax:
    *------------------------------------------------*/

   for (ii = 0; ii < (num_entries - 1); ii = jj)
   {
      /* want jj where entries ii through jj-1 have common imin */
      imin0 = hypre_CommTypeEntryIMin(comm_entries[ii]);
      for (jj = (ii + 1); jj < num_entries; jj++)
      {
         imin1 = hypre_CommTypeEntryIMin(comm_entries[jj]);
         if ( ( hypre_IModPeriodX(imin0, periodic) !=
                hypre_IModPeriodX(imin1, periodic) ) ||
              ( hypre_IModPeriodY(imin0, periodic) != 
                hypre_IModPeriodY(imin1, periodic) ) ||
              ( hypre_IModPeriodZ(imin0, periodic) !=
                hypre_IModPeriodZ(imin1, periodic) ) )
         {
            break;
         }
      }

      /* sort entries ii through jj-1 by imax */
      for (i = (jj - 1); i > ii; i--)
      {
         for (j = ii; j < i; j++)
         {
            swap = 0;
            imax0 = hypre_CommTypeEntryIMax(comm_entries[j]);
            imax1 = hypre_CommTypeEntryIMax(comm_entries[j+1]);
            if ( hypre_IModPeriodZ(imax0, periodic) >
                 hypre_IModPeriodZ(imax1, periodic) )
            {
               swap = 1;
            }
            else if ( hypre_IModPeriodZ(imax0, periodic) ==
                      hypre_IModPeriodZ(imax1, periodic) )
            {
               if ( hypre_IModPeriodY(imax0, periodic) >
                    hypre_IModPeriodY(imax1, periodic) )
               {
                  swap = 1;
               }
               else if ( hypre_IModPeriodY(imax0, periodic) ==
                         hypre_IModPeriodY(imax1, periodic) )
               {
                  if ( hypre_IModPeriodX(imax0, periodic) >
                       hypre_IModPeriodX(imax1, periodic) )
                  {
                     swap = 1;
                  }
               }
            }

            if (swap)
            {
               comm_entry        = comm_entries[j];
               comm_entries[j]   = comm_entries[j+1];
               comm_entries[j+1] = comm_entry;
            }
         }
      }
   }
#endif

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Compile a communication package into a form based on the
message-passing layer.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN/OUT]
  communication package.

@see hypre_CommPkgCreate, hypre_InitializeCommunication,
  hypre_CommTypeBuildMPI, hypre_CommPkgUnCommit */
/*--------------------------------------------------------------------------*/

int
hypre_CommPkgCommit( hypre_CommPkg *comm_pkg )
{
   int  ierr = 0;

   /* create send MPI_Datatypes */
   hypre_CommPkgSendMPITypes(comm_pkg) =
      hypre_TAlloc(MPI_Datatype, hypre_CommPkgNumSends(comm_pkg));
   hypre_CommTypeBuildMPI(hypre_CommPkgNumSends(comm_pkg),
                          hypre_CommPkgSendProcs(comm_pkg),
                          hypre_CommPkgSendTypes(comm_pkg),
                          hypre_CommPkgSendMPITypes(comm_pkg));

   /* create recv MPI_Datatypes */
   hypre_CommPkgRecvMPITypes(comm_pkg) =
      hypre_TAlloc(MPI_Datatype, hypre_CommPkgNumRecvs(comm_pkg));
   hypre_CommTypeBuildMPI(hypre_CommPkgNumRecvs(comm_pkg),
                          hypre_CommPkgRecvProcs(comm_pkg),
                          hypre_CommPkgRecvTypes(comm_pkg),
                          hypre_CommPkgRecvMPITypes(comm_pkg));

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Destroy the message-passing-layer component of the communication
package.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN/OUT]
  communication package.

@see hypre_CommPkgCommit */
/*--------------------------------------------------------------------------*/

int
hypre_CommPkgUnCommit( hypre_CommPkg *comm_pkg )
{
   MPI_Datatype  *types;
   int            i;
   int            ierr = 0;

   if (comm_pkg)
   {
      types = hypre_CommPkgSendMPITypes(comm_pkg);
      if (types)
      {
         for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
            MPI_Type_free(&types[i]);
         hypre_TFree(types);
      }
     
      types = hypre_CommPkgRecvMPITypes(comm_pkg);
      if (types)
      {
         for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
            MPI_Type_free(&types[i]);
         hypre_TFree(types);
      }
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Create an MPI-based description of a communication from a
processor-based one.

{\bf Input files:}
headers.h

@return Error code.

@param num_comms [IN]
  number of communications.
@param comm_procs [IN]
  processor ranks involved in the communications.
@param comm_types [IN]
  processor-based communication types.
@param comm_mpi_types [OUT]
  MPI derived data-types.

@see hypre_CommPkgCommit, hypre_CommTypeEntryBuildMPI */
/*--------------------------------------------------------------------------*/

int
hypre_CommTypeBuildMPI( int               num_comms,
                        int              *comm_procs,
                        hypre_CommType  **comm_types,
                        MPI_Datatype     *comm_mpi_types )
{
   hypre_CommType       *comm_type;
   hypre_CommTypeEntry  *comm_entry;
   int                   num_entries;
   int                  *comm_entry_blocklengths;
   MPI_Aint             *comm_entry_displacements;
   MPI_Datatype         *comm_entry_mpi_types;
                           
   int                   m, i;
   int                   ierr = 0;

   for (m = 0; m < num_comms; m++)
   {
      comm_type = comm_types[m];

      num_entries = hypre_CommTypeNumEntries(comm_type);
      comm_entry_blocklengths = hypre_TAlloc(int, num_entries);
      comm_entry_displacements = hypre_TAlloc(MPI_Aint, num_entries);
      comm_entry_mpi_types = hypre_TAlloc(MPI_Datatype, num_entries);

      for (i = 0; i < num_entries; i++)
      {
         comm_entry = hypre_CommTypeCommEntry(comm_type, i);

         /* set blocklengths */
         comm_entry_blocklengths[i] = 1;

         /* compute displacements */
         comm_entry_displacements[i] =
            hypre_CommTypeEntryOffset(comm_entry) * sizeof(double);

         /* compute types */
         hypre_CommTypeEntryBuildMPI(comm_entry, &comm_entry_mpi_types[i]);
      }

      /* create `comm_mpi_types' */
      MPI_Type_struct(num_entries, comm_entry_blocklengths,
                      comm_entry_displacements, comm_entry_mpi_types,
                      &comm_mpi_types[m]);
      MPI_Type_commit(&comm_mpi_types[m]);

      /* free up memory */
      for (i = 0; i < num_entries; i++)
         MPI_Type_free(&comm_entry_mpi_types[i]);
      hypre_TFree(comm_entry_blocklengths);
      hypre_TFree(comm_entry_displacements);
      hypre_TFree(comm_entry_mpi_types);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/** Create an MPI-based description of a communication entry.

{\bf Input files:}
headers.h

@return Error code.

@param comm_entry [IN]
  communication entry.
@param comm_entry_mpi_type [OUT]
  MPI derived data-type.

@see hypre_CommTypeBuildMPI */
/*--------------------------------------------------------------------------*/

int
hypre_CommTypeEntryBuildMPI( hypre_CommTypeEntry *comm_entry,
                             MPI_Datatype        *comm_entry_mpi_type )
{
   int           dim          = hypre_CommTypeEntryDim(comm_entry);
   int          *length_array = hypre_CommTypeEntryLengthArray(comm_entry);
   int          *stride_array = hypre_CommTypeEntryStrideArray(comm_entry);

   MPI_Datatype *old_type;
   MPI_Datatype *new_type;
   MPI_Datatype *tmp_type;
             
   int           i;
   int           ierr = 0;

   if (dim == 1)
   {
      MPI_Type_hvector(length_array[0], 1,
                       (MPI_Aint)(stride_array[0]*sizeof(double)),
                       MPI_DOUBLE, comm_entry_mpi_type);
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
                       *old_type, comm_entry_mpi_type);
      MPI_Type_free(old_type);

      hypre_TFree(old_type);
      hypre_TFree(new_type);
   }

   return ierr;
}

