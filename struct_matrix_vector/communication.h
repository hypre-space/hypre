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

#ifndef zzz_COMMUNICATION_HEADER
#define zzz_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   int            num_sends;
   int           *send_processes;
   MPI_Datatype  *send_types;

   int            num_recvs;
   int           *recv_processes;
   MPI_Datatype  *recv_types;

} zzz_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int           num_requests;
   MPI_Request  *requests;

} zzz_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_CommPkg
 *--------------------------------------------------------------------------*/
 
#define zzz_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define zzz_CommPkgSendProcesses(comm_pkg)   (comm_pkg -> send_processes)
#define zzz_CommPkgSendProcess(comm_pkg, i)  (comm_pkg -> send_processes[(i)])
#define zzz_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define zzz_CommPkgSendType(comm_pkg, i)     (comm_pkg -> send_types[(i)])
 
#define zzz_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define zzz_CommPkgRecvProcesses(comm_pkg)   (comm_pkg -> recv_processes)
#define zzz_CommPkgRecvProcess(comm_pkg, i)  (comm_pkg -> recv_processes[(i)])
#define zzz_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define zzz_CommPkgRecvType(comm_pkg, i)     (comm_pkg -> recv_types[(i)])

/*--------------------------------------------------------------------------
 * Accessor macros: zzz_CommHandle
 *--------------------------------------------------------------------------*/
 
#define zzz_CommHandleNumRequests(comm_handle)  (comm_handle -> num_requests)
#define zzz_CommHandleRequests(comm_handle)     (comm_handle -> requests)
#define zzz_CommHandleRequest(comm_handle, i)   (comm_handle -> requests[(i)])

#endif
