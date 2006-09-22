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



#include "headers.h"

/*==========================================================================*/

hypre_ParCSRCommHandle *
hypre_ParCSRCommHandleCreate ( int 	      job,
			       hypre_ParCSRCommPkg *comm_pkg,
                               void          *send_data, 
                               void          *recv_data )
{
   int                  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int                  num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_ParCSRCommPkgComm(comm_pkg);

   hypre_ParCSRCommHandle    *comm_handle;
   int                  num_requests;
   MPI_Request         *requests;

   int                  i, j;
   int			my_id, num_procs;
   int			ip, vec_start, vec_len;
                  
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
    * job = 11: similar to job = 1, but exchanges data of type int (not double),
    *		requires send_data and recv_data to be ints
    *		recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * job = 12: similar to job = 1, but exchanges data of type int (not double),
    *		requires send_data and recv_data to be ints
    *		recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * default: ignores send_data and recv_data, requires send_mpi_types
    *		and recv_mpi_types to be set in comm_pkg.
    *		datatypes need to point to absolute
    *		addresses, e.g. generated using MPI_Address . 
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
 
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

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
      		MPI_Irecv(&d_recv_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    MPI_Isend(&d_send_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
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
   	    MPI_Irecv(&d_recv_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Isend(&d_send_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
   	}
	break;
   }
   case  11:
   {
	int *i_send_data = (int *) send_data;
	int *i_recv_data = (int *) recv_data;
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Irecv(&i_recv_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    MPI_Isend(&i_send_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
	break;
   }
   case  12:
   {
	int *i_send_data = (int *) send_data;
	int *i_recv_data = (int *) recv_data;
   	for (i = 0; i < num_sends; i++)
   	{
	    vec_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
      	    ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
   	    MPI_Irecv(&i_recv_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Isend(&i_send_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
	break;
   }
   /* default :
   {
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i); 
      		MPI_Irecv(MPI_BOTTOM, 1, 
                	hypre_ParCSRCommPkgRecvMPIType(comm_pkg, i), 
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
      		ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i); 
      		MPI_Isend(MPI_BOTTOM, 1, 
                	hypre_ParCSRCommPkgSendMPIType(comm_pkg, i), 
			ip, 0, comm, &requests[j++]);
   	}
	break;
   } */
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
hypre_ParCSRCommHandleDestroy( hypre_ParCSRCommHandle *comm_handle )
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


/* hypre_MatCommPkgCreate_core does all the communications and computations for
       hypre_MatCommPkgCreate ( hypre_ParCSRMatrix *A)
 and   hypre_BoolMatCommPkgCreate ( hypre_ParCSRBooleanMatrix *A)
 To support both data types, it has hardly any data structures other than int*.

*/

void
hypre_MatvecCommPkgCreate_core (

/* input args: */
   MPI_Comm comm, int * col_map_offd, int first_col_diag, int * col_starts,
   int num_cols_diag, int num_cols_offd,
   int firstColDiag, int * colMapOffd,

   int data,  /* = 1 for a matrix with floating-point data, =0 for Boolean matrix */

/* pointers to output args: */
   int * p_num_recvs, int ** p_recv_procs, int ** p_recv_vec_starts,
   int * p_num_sends, int ** p_send_procs, int ** p_send_map_starts,
   int ** p_send_map_elmts

   )
{
   int	i, j;
   int	num_procs, my_id, proc_num, num_elmts;
   int	local_info, offd_col;
   int	*proc_mark, *proc_add, *tmp, *recv_buf, *displs, *info;
   /* outputs: */
   int  num_recvs, * recv_procs, * recv_vec_starts;
   int  num_sends, * send_procs, * send_map_starts, * send_map_elmts;
   int  ip, vec_start, vec_len, num_requests;

   MPI_Request *requests;
   MPI_Status *status; 

   MPI_Comm_size(comm, &num_procs);  
   MPI_Comm_rank(comm, &my_id);

   proc_mark = hypre_CTAlloc(int, num_procs);
   proc_add = hypre_CTAlloc(int, num_procs);
   info = hypre_CTAlloc(int, num_procs);

/* ----------------------------------------------------------------------
 * determine which processors to receive from (set proc_mark) and num_recvs,
 * at the end of the loop proc_mark[i] contains the number of elements to be
 * received from Proc. i
 * ---------------------------------------------------------------------*/

   for (i=0; i < num_procs; i++)
		proc_add[i] = 0;

   proc_num = 0;
   if (num_cols_offd) offd_col = col_map_offd[0];
   num_recvs=0;
   j = 0;
   for (i=0; i < num_cols_offd; i++)
   {
	if (num_cols_diag) proc_num = hypre_min(num_procs-1,offd_col / 
					num_cols_diag);
	while (col_starts[proc_num] > offd_col )
		proc_num = proc_num-1;
	while (col_starts[proc_num+1]-1 < offd_col )
		proc_num = proc_num+1;
	proc_mark[num_recvs] = proc_num;
	j = i;
	while (col_starts[proc_num+1] > offd_col)
	{
	   proc_add[num_recvs]++;
	   if (j < num_cols_offd-1) 
	   {
	      j++;
	      offd_col = col_map_offd[j];
	   }
	   else
	   {
	      j++;
	      offd_col = col_starts[num_procs];
	   }
	}
	num_recvs++;
	if (j < num_cols_offd) i = j-1;
	else i=j;
   }

   local_info = 2*num_recvs;
			
   MPI_Allgather(&local_info, 1, MPI_INT, info, 1, MPI_INT, comm); 

/* ----------------------------------------------------------------------
 * generate information to be sent: tmp contains for each recv_proc:
 * id of recv_procs, number of elements to be received for this processor,
 * indices of elements (in this order)
 * ---------------------------------------------------------------------*/

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
	displs[i] = displs[i-1]+info[i-1]; 
   recv_buf = hypre_CTAlloc(int, displs[num_procs]); 

   recv_procs = NULL;
   tmp = NULL;
   if (num_recvs)
   {
      recv_procs = hypre_CTAlloc(int, num_recvs);
      tmp = hypre_CTAlloc(int, local_info);
   }
   recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);


   j = 0;
   if (num_recvs) recv_vec_starts[0] = 0;
   for (i=0; i < num_recvs; i++)
   {
		num_elmts = proc_add[i];
		recv_procs[i] = proc_mark[i];
		recv_vec_starts[i+1] = recv_vec_starts[i]+num_elmts;
		tmp[j++] = proc_mark[i];
		tmp[j++] = num_elmts;
   }

   MPI_Allgatherv(tmp,local_info,MPI_INT,recv_buf,info,displs,MPI_INT,comm);
	

/* ----------------------------------------------------------------------
 * determine num_sends and number of elements to be sent
 * ---------------------------------------------------------------------*/

   num_sends = 0;
   num_elmts = 0;
   proc_add[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      j = displs[i];
      while ( j < displs[i+1])
      {
	 if (recv_buf[j++] == my_id)
	 {
	    proc_mark[num_sends] = i;
	    num_sends++;
	    proc_add[num_sends] = proc_add[num_sends-1]+recv_buf[j];
	    break;
	 }
	 j++;
      }	
   }
		
/* ----------------------------------------------------------------------
 * determine send_procs and actual elements to be send (in send_map_elmts)
 * and send_map_starts whose i-th entry points to the beginning of the 
 * elements to be send to proc. i
 * ---------------------------------------------------------------------*/

   send_procs = NULL;
   send_map_elmts = NULL;

   if (num_sends)
   {
      send_procs = hypre_CTAlloc(int, num_sends);
      send_map_elmts = hypre_CTAlloc(int, proc_add[num_sends]);
   }
   send_map_starts = hypre_CTAlloc(int, num_sends+1);
   num_requests = num_recvs+num_sends;
   if (num_requests)
   {
      requests = hypre_CTAlloc(MPI_Request, num_requests);
      status = hypre_CTAlloc(MPI_Status, num_requests);
   }

   if (num_sends) send_map_starts[0] = 0;
   for (i=0; i < num_sends; i++)
   {
      send_map_starts[i+1] = proc_add[i+1];
      send_procs[i] = proc_mark[i];
   }

   j=0;
   for (i=0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      MPI_Irecv(&send_map_elmts[vec_start], vec_len, MPI_INT,
                        ip, 0, comm, &requests[j++]);
   }
   for (i=0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      MPI_Isend(&col_map_offd[vec_start], vec_len, MPI_INT,
                        ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   if (num_sends)
   {
      for (i=0; i<send_map_starts[num_sends]; i++)
         send_map_elmts[i] -= first_col_diag;
   }

   hypre_TFree(proc_add);
   hypre_TFree(proc_mark); 
   hypre_TFree(tmp);
   hypre_TFree(recv_buf);
   hypre_TFree(displs);
   hypre_TFree(info);
 
   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;
}

/* ----------------------------------------------------------------------
 * hypre_MatvecCommPkgCreate
 * generates the comm_pkg for A 
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

int
hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A)
{
   hypre_ParCSRCommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
/*   MPI_Datatype         *recv_mpi_types;
   MPI_Datatype         *send_mpi_types;
*/
   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;
   
   int  *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   int  first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
   int  *col_starts = hypre_ParCSRMatrixColStarts(A);

   int	ierr = 0;
   int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   int	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));

   hypre_MatvecCommPkgCreate_core
      (
         comm, col_map_offd, first_col_diag, col_starts,
         num_cols_diag, num_cols_offd,
         first_col_diag, col_map_offd,
         1,
         &num_recvs, &recv_procs, &recv_vec_starts,
         &num_sends, &send_procs, &send_map_starts,
         &send_map_elmts
         );

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   /* hypre_ParCSRCommPkgRecvMPITypes(comm_pkg) = recv_mpi_types; */

   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;
   /* hypre_ParCSRCommPkgSendMPITypes(comm_pkg) = send_mpi_types; */

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;

   return ierr;
}

int
hypre_MatvecCommPkgDestroy(hypre_ParCSRCommPkg *comm_pkg)
{
   int ierr = 0;

   if (hypre_ParCSRCommPkgNumSends(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendProcs(comm_pkg));
      hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(comm_pkg));
   }
   hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg));
   /* if (hypre_ParCSRCommPkgSendMPITypes(comm_pkg))
      hypre_TFree(hypre_ParCSRCommPkgSendMPITypes(comm_pkg)); */ 
   if (hypre_ParCSRCommPkgNumRecvs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvProcs(comm_pkg));
   }
   hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg));
   /* if (hypre_ParCSRCommPkgRecvMPITypes(comm_pkg))
      hypre_TFree(hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)); */
   hypre_TFree(comm_pkg);

   return ierr;
}

int
hypre_BuildCSRMatrixMPIDataType(int num_nonzeros, int num_rows,
			double *a_data, int *a_i, int *a_j, 
			MPI_Datatype *csr_matrix_datatype)
{
   int		block_lens[3];
   MPI_Aint	displ[3];
   MPI_Datatype	types[3];
   int		ierr = 0;

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_rows+1;
   block_lens[2] = num_nonzeros;

   types[0] = MPI_DOUBLE;
   types[1] = MPI_INT;
   types[2] = MPI_INT;

   MPI_Address(a_data, &displ[0]);
   MPI_Address(a_i, &displ[1]);
   MPI_Address(a_j, &displ[2]);
   MPI_Type_struct(3,block_lens,displ,types,csr_matrix_datatype);
   MPI_Type_commit(csr_matrix_datatype);

   return ierr;
}

int
hypre_BuildCSRJDataType(int num_nonzeros,
                  double *a_data, int *a_j,
                  MPI_Datatype *csr_jdata_datatype)
{
   int          block_lens[2];
   MPI_Aint     displs[2];
   MPI_Datatype types[2];
   int          ierr = 0;
 
   block_lens[0] = num_nonzeros;
   block_lens[1] = num_nonzeros;
 
   types[0] = MPI_DOUBLE;
   types[1] = MPI_INT;
 
   MPI_Address(a_data, &displs[0]);
   MPI_Address(a_j, &displs[1]);
 
   MPI_Type_struct(2,block_lens,displs,types,csr_jdata_datatype);
   MPI_Type_commit(csr_jdata_datatype);
 
   return ierr;
}
