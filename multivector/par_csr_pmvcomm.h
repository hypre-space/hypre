/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
   int                  num_requests;
   MPI_Request          *requests;

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
hypre_ParCSRCommMultiHandleCreate ( int 	          job,
			            hypre_ParCSRCommPkg   *comm_pkg,
                                    void                  *send_data, 
                                    void                  *recv_data, 
				    int                   nvecs       );


int
hypre_ParCSRCommMultiHandleDestroy(hypre_ParCSRCommMultiHandle *comm_handle);

#ifdef __cplusplus
}
#endif

#endif /* HYPRE_PAR_CSR_MULTICOMMUNICATION_HEADER */ 
