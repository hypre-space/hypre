/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/




#ifndef HYPRE_PAR_CSR_PMVCOMM_HEADER
#define HYPRE_PAR_CSR_PMVCOMM_HEADER

#include "_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommMultiHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	        *send_data;
   void 	        *recv_data;
   HYPRE_Int                  num_requests;
   hypre_MPI_Request          *requests;

} hypre_ParCSRCommMultiHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommMultiHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommMultiHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_ParCSRCommMultiHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_ParCSRCommMultiHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_ParCSRCommMultiHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_ParCSRCommMultiHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_ParCSRCommMultiHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

hypre_ParCSRCommMultiHandle *
hypre_ParCSRCommMultiHandleCreate ( HYPRE_Int 	          job,
			            hypre_ParCSRCommPkg   *comm_pkg,
                                    void                  *send_data, 
                                    void                  *recv_data, 
				    HYPRE_Int                   nvecs       );


HYPRE_Int
hypre_ParCSRCommMultiHandleDestroy(hypre_ParCSRCommMultiHandle *comm_handle);

#ifdef __cplusplus
}
#endif

#endif /* HYPRE_PAR_CSR_MULTICOMMUNICATION_HEADER */ 
