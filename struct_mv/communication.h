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
 * CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_SBoxArrayArray  *send_sboxes;
   hypre_SBoxArrayArray  *recv_sboxes;

   int                **send_box_ranks;
   int                **recv_box_ranks;

   MPI_Comm            *comm;

   int                  num_sends;
   int                 *send_processes;
   MPI_Datatype        *send_types;
                     
   int                  num_recvs;
   int                 *recv_processes;
   MPI_Datatype        *recv_types;

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
 * Accessor macros: hypre_CommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_CommPkgSendSBoxes(comm_pkg)      (comm_pkg -> send_sboxes)
#define hypre_CommPkgRecvSBoxes(comm_pkg)      (comm_pkg -> recv_sboxes)
                                             
#define hypre_CommPkgSendBoxRanks(comm_pkg)    (comm_pkg -> send_box_ranks)
#define hypre_CommPkgRecvBoxRanks(comm_pkg)    (comm_pkg -> recv_box_ranks)

#define hypre_CommPkgComm(comm_pkg)            (comm_pkg -> comm)

#define hypre_CommPkgNumSends(comm_pkg)        (comm_pkg -> num_sends)
#define hypre_CommPkgSendProcesses(comm_pkg)   (comm_pkg -> send_processes)
#define hypre_CommPkgSendProcess(comm_pkg, i)  (comm_pkg -> send_processes[(i)])
#define hypre_CommPkgSendTypes(comm_pkg)       (comm_pkg -> send_types)
#define hypre_CommPkgSendType(comm_pkg, i)     (comm_pkg -> send_types[(i)])
 
#define hypre_CommPkgNumRecvs(comm_pkg)        (comm_pkg -> num_recvs)
#define hypre_CommPkgRecvProcesses(comm_pkg)   (comm_pkg -> recv_processes)
#define hypre_CommPkgRecvProcess(comm_pkg, i)  (comm_pkg -> recv_processes[(i)])
#define hypre_CommPkgRecvTypes(comm_pkg)       (comm_pkg -> recv_types)
#define hypre_CommPkgRecvType(comm_pkg, i)     (comm_pkg -> recv_types[(i)])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_CommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_CommHandleNumRequests(comm_handle)  (comm_handle -> num_requests)
#define hypre_CommHandleRequests(comm_handle)     (comm_handle -> requests)
#define hypre_CommHandleRequest(comm_handle, i)   (comm_handle -> requests[(i)])

#endif
