/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommTypeEntry:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBox  *sbox;
   hypre_Box   *data_box;
   int          sbox_offset;      /* offset for sbox imin */
   int          data_box_offset;  /* offset for data_box imin */

   int          dim;              /* dimension of the communication */
   int          length_array[4];
   int          stride_array[4];

} hypre_CommTypeEntry;

/*--------------------------------------------------------------------------
 * hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommTypeEntry  **comm_entries;
   int                    num_entries;

} hypre_CommType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                    num_values;
   MPI_Comm               comm;

   int                    num_sends;
   int                    num_recvs;
   int                   *send_procs;
   int                   *recv_procs;

   /* remote communication information */
   hypre_CommType       **send_types;
   hypre_CommType       **recv_types;
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

   /* local copy information */
   hypre_CommType        *copy_from_type;
   hypre_CommType        *copy_to_type;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_CommPkg  *comm_pkg;
   double         *send_data;
   double         *recv_data;

   int             num_requests;
   MPI_Request    *requests;

#if defined(HYPRE_COMM_SIMPLE)
   double        **send_buffers;
   double        **recv_buffers;
#endif

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommTypeEntry
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeEntrySBox(entry)          (entry -> sbox)
#define hypre_CommTypeEntryDataBox(entry)       (entry -> data_box)
#define hypre_CommTypeEntrySBoxOffset(entry)    (entry -> sbox_offset)
#define hypre_CommTypeEntryDataBoxOffset(entry) (entry -> data_box_offset)
#define hypre_CommTypeEntryDim(entry)           (entry -> dim)
#define hypre_CommTypeEntryLengthArray(entry)   (entry -> length_array)
#define hypre_CommTypeEntryStrideArray(entry)   (entry -> stride_array)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommTypeCommEntries(type)   (type -> comm_entries)
#define hypre_CommTypeCommEntry(type, i)  (type -> comm_entries[i])
#define hypre_CommTypeNumEntries(type)    (type -> num_entries)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgNumValues(comm_pkg)       (comm_pkg -> num_values)
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)
                                               
#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgSendProcs(comm_pkg)       (comm_pkg -> send_procs)
#define hypre_CommPkgSendProc(comm_pkg, i)     (comm_pkg -> send_procs[i])
#define hypre_CommPkgRecvProcs(comm_pkg)       (comm_pkg -> recv_procs)
#define hypre_CommPkgRecvProc(comm_pkg, i)     (comm_pkg -> recv_procs[i])

#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)     (comm_pkg -> send_types[i])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)     (comm_pkg -> recv_types[i])
#define hypre_CommPkgSendMPITypes(comm_pkg)    (comm_pkg -> send_mpi_types)
#define hypre_CommPkgSendMPIType(comm_pkg, i)  (comm_pkg -> send_mpi_types[i])
#define hypre_CommPkgRecvMPITypes(comm_pkg)    (comm_pkg -> recv_mpi_types)
#define hypre_CommPkgRecvMPIType(comm_pkg, i)  (comm_pkg -> recv_mpi_types[i])

#define hypre_CommPkgCopyFromType(comm_pkg)    (comm_pkg -> copy_from_type)
#define hypre_CommPkgCopyToType(comm_pkg)      (comm_pkg -> copy_to_type)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_CommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_CommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleRequest(comm_handle, i)  (comm_handle -> requests[(i)])
#if defined(HYPRE_COMM_SIMPLE)
#define hypre_CommHandleSendBuffers(comm_handle) (comm_handle -> send_buffers)
#define hypre_CommHandleRecvBuffers(comm_handle) (comm_handle -> recv_buffers)
#endif

#endif
