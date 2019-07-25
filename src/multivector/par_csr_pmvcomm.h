/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
