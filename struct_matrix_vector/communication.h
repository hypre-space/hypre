/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header info for communication
 *
 *****************************************************************************/

#ifndef hypre_COMMUNICATION_HEADER
#define hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_CommDataType:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBox         *sbox;
   hypre_Box          *data_box;
   int                 data_offset;

} hypre_CommDataType;

/*--------------------------------------------------------------------------
 * hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBoxArrayArray  *send_sboxes;
   hypre_SBoxArrayArray  *recv_sboxes;
   hypre_BoxArray        *send_data_space;
   hypre_BoxArray        *recv_data_space;
   int                  **send_processes;
   int                  **recv_processes;
   int                    num_values;
   MPI_Comm               comm;

   /* remote communication information */
   int                    num_sends;
   int                   *send_procs;
   MPI_Datatype          *send_types;
   int                    num_recvs;
   int                   *recv_procs;
   MPI_Datatype          *recv_types;

   /* local copy information */
   int                    num_copies_from;
   hypre_CommDataType   **copy_from_types;
   int                    num_copies_to;
   hypre_CommDataType   **copy_to_types;

} hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int           num_requests;
   MPI_Request  *requests;

} hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommDataType
 *--------------------------------------------------------------------------*/
 
#define hypre_CommDataTypeSBox(type)        (type -> sbox)
#define hypre_CommDataTypeDataBox(type)     (type -> data_box)
#define hypre_CommDataTypeDataOffset(type)  (type -> data_offset)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgSendSBoxes(comm_pkg)      (comm_pkg -> send_sboxes)
#define hypre_CommPkgRecvSBoxes(comm_pkg)      (comm_pkg -> recv_sboxes)
#define hypre_CommPkgSendDataSpace(comm_pkg)   (comm_pkg -> send_data_space)
#define hypre_CommPkgRecvDataSpace(comm_pkg)   (comm_pkg -> recv_data_space)
#define hypre_CommPkgSendProcesses(comm_pkg)   (comm_pkg -> send_processes)
#define hypre_CommPkgRecvProcesses(comm_pkg)   (comm_pkg -> recv_processes)
#define hypre_CommPkgNumValues(comm_pkg)       (comm_pkg -> num_values)
#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)
                                               
#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgSendProcs(comm_pkg)       (comm_pkg -> send_procs)
#define hypre_CommPkgSendProc(comm_pkg, i)     (comm_pkg -> send_procs[(i)])
#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)     (comm_pkg -> send_types[(i)])
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgRecvProcs(comm_pkg)       (comm_pkg -> recv_procs)
#define hypre_CommPkgRecvProc(comm_pkg, i)     (comm_pkg -> recv_procs[(i)])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)     (comm_pkg -> recv_types[(i)])
                                               
#define hypre_CommPkgNumCopiesFrom(comm_pkg)   (comm_pkg -> num_copies_from)
#define hypre_CommPkgCopyFromTypes(comm_pkg)   (comm_pkg -> copy_from_types)
#define hypre_CommPkgCopyFromType(comm_pkg, i) \
(comm_pkg -> copy_from_types[(i)])
#define hypre_CommPkgNumCopiesTo(comm_pkg)     (comm_pkg -> num_copies_to)
#define hypre_CommPkgCopyToTypes(comm_pkg)     (comm_pkg -> copy_to_types)
#define hypre_CommPkgCopyToType(comm_pkg, i) \
(comm_pkg -> copy_to_types[(i)])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_CommHandleRequest(comm_handle, i)  (comm_handle -> requests[(i)])

#endif
