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
/**
Create a communication package.  A grid-based description of a
communication exchange is passed in.  This description is then
compiled into an intermediate processor-based description of the
communication.  It may further compiled into a form based on the
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

@see hypre_NewCommPkgInfo, hypre_CommitCommPkg, hypre_FreeCommPkg
*/
/*--------------------------------------------------------------------------*/

hypre_CommPkg *
hypre_NewCommPkg( hypre_BoxArrayArray   *send_boxes,
                  hypre_BoxArrayArray   *recv_boxes,
                  hypre_Index            send_stride,
                  hypre_Index            recv_stride,
                  hypre_BoxArray        *send_data_space,
                  hypre_BoxArray        *recv_data_space,
                  int                  **send_processes,
                  int                  **recv_processes,
                  int                    num_values,
                  MPI_Comm               comm            )
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

   hypre_NewCommPkgInfo(send_boxes, send_stride,
                        send_data_space, send_processes,
                        num_values, comm,
                        &num_sends, &send_procs,
                        &send_types, &copy_from_type);

   hypre_CommPkgNumSends(comm_pkg)     = num_sends;
   hypre_CommPkgSendProcs(comm_pkg)    = send_procs;
   hypre_CommPkgSendTypes(comm_pkg)    = send_types;
   hypre_CommPkgCopyFromType(comm_pkg) = copy_from_type;

   hypre_NewCommPkgInfo(recv_boxes, recv_stride,
                        recv_data_space, recv_processes,
                        num_values, comm,
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
   hypre_FreeBoxArrayArray(send_boxes);
   hypre_TFree(send_processes);

   hypre_ForBoxArrayI(i, recv_boxes)
      hypre_TFree(recv_processes[i]);
   hypre_FreeBoxArrayArray(recv_boxes);
   hypre_TFree(recv_processes);

#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
#else
   hypre_CommitCommPkg(comm_pkg);

   /* free up comm types */
   for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
      hypre_FreeCommType(hypre_CommPkgSendType(comm_pkg, i));
   hypre_TFree(hypre_CommPkgSendTypes(comm_pkg));
   for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
      hypre_FreeCommType(hypre_CommPkgRecvType(comm_pkg, i));
   hypre_TFree(hypre_CommPkgRecvTypes(comm_pkg));
#endif

   return comm_pkg;
}

/*==========================================================================*/
/*==========================================================================*/
/**
Destroy a communication package.

{\bf Input files:}
headers.h

@return Void.

@param comm_pkg [IN]
  communication package.

@see hypre_NewCommPkg
*/
/*--------------------------------------------------------------------------*/

void
hypre_FreeCommPkg( hypre_CommPkg *comm_pkg )
{
#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
   int               i;
#else
#endif

   if (comm_pkg)
   {
#if defined(HYPRE_COMM_SIMPLE) || defined(HYPRE_COMM_VOLATILE)
      /* free up comm types */
      for (i = 0; i < hypre_CommPkgNumSends(comm_pkg); i++)
         hypre_FreeCommType(hypre_CommPkgSendType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgSendTypes(comm_pkg));
      for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
         hypre_FreeCommType(hypre_CommPkgRecvType(comm_pkg, i));
      hypre_TFree(hypre_CommPkgRecvTypes(comm_pkg));
#else
      hypre_UnCommitCommPkg(comm_pkg);
#endif

      hypre_TFree(hypre_CommPkgSendProcs(comm_pkg));
      hypre_TFree(hypre_CommPkgRecvProcs(comm_pkg));

      hypre_FreeCommType(hypre_CommPkgCopyFromType(comm_pkg));
      hypre_FreeCommType(hypre_CommPkgCopyToType(comm_pkg));

      hypre_TFree(comm_pkg);
   }
}

/*==========================================================================*/
/*==========================================================================*/
/**
Initialize a non-blocking communication exchange.

\begin{itemize}
\item If HYPRE\_COMM\_SIMPLE is defined, the communication buffers
are created, the send buffer is manually packed, and the communication
requests are posted.  No MPI derived datatypes are used.
\item Else if HYPRE\_COMM\_VOLATILE is defined, the communication
package is committed, the communication requests are posted, then
the communication package is un-committed.
\item Else the communication requests are posted.
\end{itemize}

{\bf Input files:}
headers.h

@return Communication handle.

@param comm_pkg [IN]
  communication package.
@param send_data [IN]
  reference pointer for the send data.
@param recv_data [IN]
  reference pointer for the recv data.

@see hypre_FinalizeCommunication, hypre_NewCommPkg
*/
/*--------------------------------------------------------------------------*/

#if defined(HYPRE_COMM_SIMPLE)

hypre_CommHandle *
hypre_InitializeCommunication( hypre_CommPkg *comm_pkg,
                               double        *send_data,
                               double        *recv_data )
{
   int                  num_sends = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_CommPkgComm(comm_pkg);

   hypre_CommHandle    *comm_handle;
   int                  num_requests;
   MPI_Request         *requests;
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
   int                  entry_size;
                      
   /*--------------------------------------------------------------------
    * allocate buffers
    *--------------------------------------------------------------------*/

   /* allocate send buffers */
   send_buffers = hypre_TAlloc(double *, num_sends);
   send_sizes   = hypre_TAlloc(int, num_sends);
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

      send_buffers[i] = hypre_TAlloc(double, send_sizes[i]);
   }

   /* allocate recv buffers */
   recv_buffers = hypre_TAlloc(double *, num_recvs);
   recv_sizes   = hypre_TAlloc(int, num_recvs);
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

      recv_buffers[i] = hypre_SharedTAlloc(double, recv_sizes[i]);
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

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);

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

   hypre_TFree(send_sizes);
   hypre_TFree(recv_sizes);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;
   hypre_CommHandleSendBuffers(comm_handle) = send_buffers;
   hypre_CommHandleRecvBuffers(comm_handle) = recv_buffers;

   return ( comm_handle );
}

/*--------------------------------------------------------------------------*/

#else

hypre_CommHandle *
hypre_InitializeCommunication( hypre_CommPkg *comm_pkg,
                               double        *send_data,
                               double        *recv_data )
{
   int                  num_sends  = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs  = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm       = hypre_CommPkgComm(comm_pkg);
   void                *send_vdata = (void *) send_data;
   void                *recv_vdata = (void *) recv_data;
                      
   hypre_CommHandle    *comm_handle;
   int                  num_requests;
   MPI_Request         *requests;

   int                  i, j;
                      
   /*--------------------------------------------------------------------
    * post receives and initiate sends
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);

#if defined(HYPRE_COMM_VOLATILE)
   /* commit the communication package */
   hypre_CommitCommPkg(comm_pkg);
#else
#endif

   j = 0;
   for(i = 0; i < num_recvs; i++)
   {
      MPI_Irecv(recv_vdata, 1,
                hypre_CommPkgRecvMPIType(comm_pkg, i), 
                hypre_CommPkgRecvProc(comm_pkg, i), 
		0, comm, &requests[j++]);
   }
   for(i = 0; i < num_sends; i++)
   {
      MPI_Isend(send_vdata, 1,
                hypre_CommPkgSendMPIType(comm_pkg, i), 
                hypre_CommPkgSendProc(comm_pkg, i), 
		0, comm, &requests[j++]);
   }

#if defined(HYPRE_COMM_VOLATILE)
   /* un-commit the communication package */
   hypre_UnCommitCommPkg(comm_pkg);
#else
#endif

   hypre_ExchangeLocalData(comm_pkg, send_data, recv_data);

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;

   return ( comm_handle );
}

#endif

/*==========================================================================*/
/*==========================================================================*/
/**
Finalize a communication exchange.  This routine blocks until all of
the communication requests are completed.

\begin{itemize}
\item If HYPRE\_COMM\_SIMPLE is defined, the communication requests
are completed, the receive buffer is manually unpacked, and the
communication buffers are destroyed.
\item Else if HYPRE\_COMM\_VOLATILE is defined, the communication requests
are completed and the communication package is un-committed.
\item Else the communication requests are completed.
\end{itemize}

{\bf Input files:}
headers.h

@return Error code.

@param comm_handle [IN]
  communication handle.

@see hypre_InitializeCommunication, hypre_NewCommPkg
*/
/*--------------------------------------------------------------------------*/

#if defined(HYPRE_COMM_SIMPLE)

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   
   hypre_CommPkg   *comm_pkg     = hypre_CommHandleCommPkg(comm_handle);
   double         **send_buffers = hypre_CommHandleSendBuffers(comm_handle);
   double         **recv_buffers = hypre_CommHandleRecvBuffers(comm_handle);
   int              num_sends    = hypre_CommPkgNumSends(comm_pkg);
   int              num_recvs    = hypre_CommPkgNumRecvs(comm_pkg);

   MPI_Status          *status;

   hypre_CommType      *recv_type;
   hypre_CommTypeEntry *recv_entry;

   int                 *length_array;
   int                 *stride_array;

   double              *iptr, *jptr, *kptr, *lptr, *bptr;

   int                  i, j, ii, jj, kk, ll;
   int                  ierr = 0;

   /*--------------------------------------------------------------------
    * finish communications
    *--------------------------------------------------------------------*/

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
    * free up comm_handle
    *--------------------------------------------------------------------*/

   /* free up send/recv buffers */
   for (i = 0; i < num_sends; i++)
      hypre_TFree(send_buffers[i]);
   for (i = 0; i < num_recvs; i++)
      hypre_SharedTFree(recv_buffers[i]);
   hypre_TFree(send_buffers);
   hypre_TFree(recv_buffers);

   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

#else

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   MPI_Status *status;
   int         ierr = 0;

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

   /* free up comm_handle */
   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

#endif

/*==========================================================================*/
/*==========================================================================*/
/**
Execute local data exchanges.

{\bf Input files:}
headers.h

@return Error flag.

@param comm_pkg [IN]
  communication package.
@param send_data [IN]
  reference pointer for the send data.
@param recv_data [IN]
  reference pointer for the recv data.

@see hypre_InitializeCommunication
*/
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
/**
Create a communication type.

{\bf Input files:}
headers.h

@return Communication type.

@param comm_entries [IN]
  array of pointers to communication type entries.
@param num_entries [IN]
  number of elements in comm\_entries array.

@see hypre_FreeCommType
*/
/*--------------------------------------------------------------------------*/

hypre_CommType *
hypre_NewCommType( hypre_CommTypeEntry **comm_entries,
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
/**
Destroy a communication type.

{\bf Input files:}
headers.h

@return Void.

@param comm_type [IN]
  communication type.

@see hypre_NewCommType
*/
/*--------------------------------------------------------------------------*/

void 
hypre_FreeCommType( hypre_CommType *comm_type )
{
   hypre_CommTypeEntry  *comm_entry;
   int                   i;

   if (comm_type)
   {
      if ( hypre_CommTypeCommEntries(comm_type) != NULL )
      {
         for (i = 0; i < hypre_CommTypeNumEntries(comm_type); i++)
         {
            comm_entry = hypre_CommTypeCommEntry(comm_type, i);
            hypre_FreeCommTypeEntry(comm_entry);
         }
      }

      hypre_TFree(hypre_CommTypeCommEntries(comm_type));
      hypre_TFree(comm_type);
   }
}

/*==========================================================================*/
/*==========================================================================*/
/**
Create a communication type entry.

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

@see hypre_FreeCommTypeEntry
*/
/*--------------------------------------------------------------------------*/

hypre_CommTypeEntry *
hypre_NewCommTypeEntry( hypre_Box   *box,
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
   hypre_GetStrideBoxSize(box, stride, size);
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
/**
Destroy a communication type entry.

{\bf Input files:}
headers.h

@return Void.

@param comm_entry [IN]
  communication type entry.

@see hypre_NewCommTypeEntry
*/
/*--------------------------------------------------------------------------*/

void
hypre_FreeCommTypeEntry( hypre_CommTypeEntry *comm_entry )
{
   if (comm_entry)
   {
      hypre_TFree(comm_entry);
   }
}

/*==========================================================================*/
/*==========================================================================*/
/**
Compute a processor-based description of a communication from a
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

@see hypre_NewCommPkg, hypre_SortCommType
*/
/*--------------------------------------------------------------------------*/

int
hypre_NewCommPkgInfo( hypre_BoxArrayArray   *boxes,
                      hypre_Index            stride,
                      hypre_BoxArray        *data_space,
                      int                  **processes,
                      int                    num_values,
                      MPI_Comm               comm,
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
   int                   *nprocs_ptr, *myproc_ptr;
                        
   int                    ierr = 0;
                
   /*---------------------------------------------------------
    * Misc stuff
    *---------------------------------------------------------*/
   nprocs_ptr = hypre_SharedTAlloc(int, 1);
   myproc_ptr = hypre_SharedTAlloc(int, 1);

   MPI_Comm_size(comm, nprocs_ptr );
   MPI_Comm_rank(comm, myproc_ptr );

   num_procs = *nprocs_ptr;
   my_proc = *myproc_ptr;

   hypre_SharedTFree(nprocs_ptr);
   hypre_SharedTFree(myproc_ptr);
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
                     hypre_NewCommTypeEntry(box, stride, data_box,
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
      comm_types[m] = hypre_NewCommType(comm_entries[p], num_entries[p]);
      hypre_SortCommType(comm_types[m]);
   }

   /*------------------------------------------------------
    * Build copy_type
    *------------------------------------------------------*/

   if (comm_entries[my_proc] != NULL)
   {
      p = my_proc;
      copy_type = hypre_NewCommType(comm_entries[p], num_entries[p]);
      hypre_SortCommType(copy_type);
   }
   else
   {
      copy_type = hypre_NewCommType(NULL, 0);
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
/**
Sort the entries of a communication type.  This routine is used to
maintain consistency in communications.

{\bf Input files:}
headers.h

{\bf Note:}
The entries are sorted by imin first.  Entries with common imin's are
then sorted by imax.  This assumes that imin and imax define a unique
communication type.

@return Error code.

@param comm_type [IN/OUT]
  communication type to be sorted.

@see hypre_NewCommPkgInfo
*/
/*--------------------------------------------------------------------------*/

int
hypre_SortCommType( hypre_CommType  *comm_type )
{
   hypre_CommTypeEntry  **comm_entries = hypre_CommTypeCommEntries(comm_type);
   int                    num_entries  = hypre_CommTypeNumEntries(comm_type);

   hypre_CommTypeEntry   *comm_entry;
   hypre_IndexRef         imin0, imin1;
   int                   *imax0, *imax1;
   int                    swap;
   int                    i, j, ii, jj;
   int                    ierr = 0;
                      
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
            comm_entry        = comm_entries[j];
            comm_entries[j]   = comm_entries[j+1];
            comm_entries[j+1] = comm_entry;
         }
      }
   }

   /*------------------------------------------------
    * Sort entries with common imin's by imax:
    *------------------------------------------------*/

   for (ii = 0; ii < (num_entries - 1); ii = jj)
   {
      /* want jj where entries ii through jj-1 have common imin's */
      imin0 = hypre_CommTypeEntryIMin(comm_entries[ii]);
      for (jj = (ii + 1); jj < num_entries; jj++)
      {
         imin1 = hypre_CommTypeEntryIMin(comm_entries[jj]);
         if ( ( hypre_IndexX(imin0) != hypre_IndexX(imin1) ) ||
              ( hypre_IndexY(imin0) != hypre_IndexY(imin1) ) ||
              ( hypre_IndexZ(imin0) != hypre_IndexZ(imin1) ) )
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
            if ( hypre_IndexZ(imax0) > hypre_IndexZ(imax1) )
            {
               swap = 1;
            }
            else if ( hypre_IndexZ(imax0) == hypre_IndexZ(imax1) )
            {
               if ( hypre_IndexY(imax0) > hypre_IndexY(imax1) )
               {
                  swap = 1;
               }
               else if ( hypre_IndexY(imax0) == hypre_IndexY(imax1) )
               {
                  if ( hypre_IndexX(imax0) > hypre_IndexX(imax1) )
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

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/**
Compile a communication package into a form based on the
message-passing layer.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN/OUT]
  communication package.

@see hypre_NewCommPkg, hypre_InitializeCommunication,
  hypre_BuildCommMPITypes, hypre_UnCommitCommPkg
*/
/*--------------------------------------------------------------------------*/

int
hypre_CommitCommPkg( hypre_CommPkg *comm_pkg )
{
   int  ierr = 0;

   /* create send MPI_Datatype's */
   hypre_CommPkgSendMPITypes(comm_pkg) =
      hypre_SharedTAlloc(MPI_Datatype, hypre_CommPkgNumSends(comm_pkg));
   hypre_BuildCommMPITypes(hypre_CommPkgNumSends(comm_pkg),
                           hypre_CommPkgSendProcs(comm_pkg),
                           hypre_CommPkgSendTypes(comm_pkg),
                           hypre_CommPkgSendMPITypes(comm_pkg));

   /* create recv MPI_Datatype's */
   hypre_CommPkgRecvMPITypes(comm_pkg) =
      hypre_SharedTAlloc(MPI_Datatype, hypre_CommPkgNumRecvs(comm_pkg));
   hypre_BuildCommMPITypes(hypre_CommPkgNumRecvs(comm_pkg),
                           hypre_CommPkgRecvProcs(comm_pkg),
                           hypre_CommPkgRecvTypes(comm_pkg),
                           hypre_CommPkgRecvMPITypes(comm_pkg));

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/**
Destroy the message-passing-layer component of the communication package.

{\bf Input files:}
headers.h

@return Error code.

@param comm_pkg [IN/OUT]
  communication package.

@see hypre_CommitCommPkg
*/
/*--------------------------------------------------------------------------*/

int
hypre_UnCommitCommPkg( hypre_CommPkg *comm_pkg )
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
         hypre_SharedTFree(types);
      }
     
      types = hypre_CommPkgRecvMPITypes(comm_pkg);
      if (types)
      {
         for (i = 0; i < hypre_CommPkgNumRecvs(comm_pkg); i++)
            MPI_Type_free(&types[i]);
         hypre_SharedTFree(types);
      }
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/**
Create an MPI-based description of a communication from a
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

@see hypre_CommitCommPkg, hypre_BuildCommEntryMPIType
*/
/*--------------------------------------------------------------------------*/

int
hypre_BuildCommMPITypes( int               num_comms,
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
      comm_entry_mpi_types = hypre_SharedTAlloc(MPI_Datatype, num_entries);

      for (i = 0; i < num_entries; i++)
      {
         comm_entry = hypre_CommTypeCommEntry(comm_type, i);

         /* set blocklengths */
         comm_entry_blocklengths[i] = 1;

         /* compute displacements */
         comm_entry_displacements[i] =
            hypre_CommTypeEntryOffset(comm_entry) * sizeof(double);

         /* compute types */
         hypre_BuildCommEntryMPIType(comm_entry, &comm_entry_mpi_types[i]);
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
      hypre_SharedTFree(comm_entry_mpi_types);
   }

   return ierr;
}

/*==========================================================================*/
/*==========================================================================*/
/**
Create an MPI-based description of a communication entry.

{\bf Input files:}
headers.h

@return Error code.

@param comm_entry [IN]
  communication entry.
@param comm_entry_mpi_type [OUT]
  MPI derived data-type.

@see hypre_BuildCommMPITypes
*/
/*--------------------------------------------------------------------------*/

int
hypre_BuildCommEntryMPIType( hypre_CommTypeEntry *comm_entry,
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
      old_type = hypre_SharedCTAlloc(MPI_Datatype, 1);
      new_type = hypre_SharedCTAlloc(MPI_Datatype, 1);

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

      hypre_SharedTFree(old_type);
      hypre_SharedTFree(new_type);
   }

   return ierr;
}

