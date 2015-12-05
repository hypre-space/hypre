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




#include "par_csr_pmvcomm.h"

#include "_hypre_parcsr_mv.h"

/*==========================================================================*/

hypre_ParCSRCommMultiHandle *
hypre_ParCSRCommMultiHandleCreate (HYPRE_Int 	               job,
			           hypre_ParCSRCommPkg *comm_pkg,
                                   void                *send_data, 
                                   void                *recv_data, 
				   HYPRE_Int                 num_vecs )
{
   HYPRE_Int                  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int                  num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_ParCSRCommPkgComm(comm_pkg);

   hypre_ParCSRCommMultiHandle *comm_handle;
   HYPRE_Int                         num_requests;
   hypre_MPI_Request                 *requests;

   HYPRE_Int                  i, j;
   HYPRE_Int			my_id, num_procs;
   HYPRE_Int			ip, vec_start, vec_len;
                  
   /*--------------------------------------------------------------------
    * hypre_Initialize sets up a communication handle,
    * posts receives and initiates sends. It always requires num_sends, 
    * num_recvs, recv_procs and send_procs to be set in comm_pkg.
    * There are different options for job:
    * job = 1 : is used to initialize communication exchange for the parts
    *		of vector needed to perform a Matvec,  it requires send_data 
    *		and recv_data to be doubles, recv_vec_starts and 
    *		send_map_starts need to be set in comm_pkg.
    * job = 2 : is used to initialize communication exchange for the parts
    *		of vector needed to perform a MatvecT,  it requires send_data 
    *		and recv_data to be doubles, recv_vec_starts and 
    *		send_map_starts need to be set in comm_pkg.
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests);
 
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   j = 0;
   switch (job)
   {
      case  1:
      {
	 double *d_send_data = (double *) send_data;
	 double *d_recv_data = (double *) recv_data;
   	 for (i = 0; i < num_recvs; i++)
   	 {
      	    ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      	    vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      	    vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      	    hypre_MPI_Irecv(&d_recv_data[vec_start*num_vecs], vec_len*num_vecs, 
                      hypre_MPI_DOUBLE, ip, 0, comm, &requests[j++]);
   	 }
   	 for (i = 0; i < num_sends; i++)
   	 {
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    hypre_MPI_Isend(&d_send_data[vec_start*num_vecs], vec_len*num_vecs,
                      hypre_MPI_DOUBLE, ip, 0, comm, &requests[j++]);
   	 }
	 break;
      }
      case  2:
      {
 	 double *d_send_data = (double *) send_data;
	 double *d_recv_data = (double *) recv_data;
   	 for (i = 0; i < num_sends; i++)
   	 {
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    hypre_MPI_Irecv(&d_recv_data[vec_start*num_vecs], vec_len*num_vecs,
                      hypre_MPI_DOUBLE, ip, 0, comm, &requests[j++]);
   	 }
   	 for (i = 0; i < num_recvs; i++)
   	 {
      	    ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      	    vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      	    vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      	    hypre_MPI_Isend(&d_send_data[vec_start*num_vecs], vec_len*num_vecs,
                      hypre_MPI_DOUBLE, ip, 0, comm, &requests[j++]);
   	 }
	 break;
      }
   }

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_ParCSRCommMultiHandle, 1);

   hypre_ParCSRCommMultiHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_ParCSRCommMultiHandleSendData(comm_handle)    = send_data;
   hypre_ParCSRCommMultiHandleRecvData(comm_handle)    = recv_data;
   hypre_ParCSRCommMultiHandleNumRequests(comm_handle) = num_requests;
   hypre_ParCSRCommMultiHandleRequests(comm_handle)    = requests;

   return (comm_handle);
}

HYPRE_Int
hypre_ParCSRCommMultiHandleDestroy(hypre_ParCSRCommMultiHandle *comm_handle)
{
   hypre_MPI_Status *status0;
   HYPRE_Int	ierr = 0;

   if (hypre_ParCSRCommMultiHandleNumRequests(comm_handle))
   {
      status0 = hypre_CTAlloc(hypre_MPI_Status,
                       hypre_ParCSRCommMultiHandleNumRequests(comm_handle));
      hypre_MPI_Waitall(hypre_ParCSRCommMultiHandleNumRequests(comm_handle),
                  hypre_ParCSRCommMultiHandleRequests(comm_handle), status0);
      hypre_TFree(status0);
   }

   hypre_TFree(hypre_ParCSRCommMultiHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

