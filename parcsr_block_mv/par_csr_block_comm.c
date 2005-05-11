/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
                                                                                                               
#include <HYPRE_config.h>
#include "utilities.h"
#include "../parcsr_mv/parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockCommHandleCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRCommHandle *
hypre_ParCSRBlockCommHandleCreate(int bnnz, hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data, void *recv_data )
{
   int      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm comm      = hypre_ParCSRCommPkgComm(comm_pkg);
   hypre_ParCSRCommHandle *comm_handle;
   int         num_requests;
   MPI_Request *requests;
   int    i, j, my_id, num_procs, ip, vec_start, vec_len;
   double *d_send_data = (double *) send_data;
   double *d_recv_data = (double *) recv_data;
                  
   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
 
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   j = 0;
   for (i = 0; i < num_recvs; i++)
   {
      ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      vec_len = (hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start)*bnnz;
      MPI_Irecv(&d_recv_data[vec_start*bnnz], vec_len, MPI_DOUBLE,
		ip, 0, comm, &requests[j++]);
  }
  for (i = 0; i < num_sends; i++)
  {
     vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     vec_len = (hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1)-vec_start)*bnnz;
     ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
     MPI_Isend(&d_send_data[vec_start*bnnz], vec_len, MPI_DOUBLE, ip, 0, comm, 
               &requests[j++]);
  }

  /*--------------------------------------------------------------------
   * set up comm_handle and return
   *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1);

   hypre_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_ParCSRCommHandleSendData(comm_handle)    = send_data;
   hypre_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   hypre_ParCSRCommHandleRequests(comm_handle)    = requests;
   return ( comm_handle );
}

int
hypre_ParCSRBlockCommHandleDestroy(hypre_ParCSRCommHandle *comm_handle)
{
   MPI_Status          *status0;
   int			ierr = 0;

   if ( comm_handle==NULL ) return ierr;
   if (hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      status0 = hypre_CTAlloc(MPI_Status,
                       hypre_ParCSRCommHandleNumRequests(comm_handle));
      MPI_Waitall(hypre_ParCSRCommHandleNumRequests(comm_handle),
                  hypre_ParCSRCommHandleRequests(comm_handle), status0);
      hypre_TFree(status0);
   }

   hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

