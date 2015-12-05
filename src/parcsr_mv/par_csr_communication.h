/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#ifndef HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   HYPRE_Int                    num_sends;
   HYPRE_Int                   *send_procs;
   HYPRE_Int			 *send_map_starts;
   HYPRE_Int			 *send_map_elmts;

   HYPRE_Int                    num_recvs;
   HYPRE_Int                   *recv_procs;
   HYPRE_Int                   *recv_vec_starts;

   /* remote communication information */
   hypre_MPI_Datatype          *send_mpi_types;
   hypre_MPI_Datatype          *recv_mpi_types;

} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   HYPRE_Int             num_requests;
   hypre_MPI_Request    *requests;

} hypre_ParCSRCommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_ParCSRCommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */
