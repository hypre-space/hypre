#include "headers.h"

/*==========================================================================*/

hypre_CommHandle *
hypre_InitializeCommunication( int 	      job,
			       hypre_CommPkg *comm_pkg,
                               void          *send_data, 
                               void          *recv_data )
{
   int                  num_sends = hypre_CommPkgNumSends(comm_pkg);
   int                  num_recvs = hypre_CommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = hypre_CommPkgComm(comm_pkg);

   hypre_CommHandle    *comm_handle;
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
      		ip = hypre_CommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_CommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_CommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Irecv(&d_recv_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
	    vec_start = hypre_CommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_CommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      	    ip = hypre_CommPkgSendProc(comm_pkg, i); 
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
	    vec_start = hypre_CommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_CommPkgSendMapStart(comm_pkg, i+1) - vec_start;
      	    ip = hypre_CommPkgSendProc(comm_pkg, i); 
   	    MPI_Irecv(&d_recv_data[vec_start], vec_len, MPI_DOUBLE,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_CommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_CommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_CommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
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
      		ip = hypre_CommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_CommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_CommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Irecv(&i_recv_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
	    vec_start = hypre_CommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_CommPkgSendMapStart(comm_pkg, i+1)-vec_start;
      	    ip = hypre_CommPkgSendProc(comm_pkg, i); 
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
	    vec_start = hypre_CommPkgSendMapStart(comm_pkg, i);
	    vec_len = hypre_CommPkgSendMapStart(comm_pkg, i+1) - vec_start;
      	    ip = hypre_CommPkgSendProc(comm_pkg, i); 
   	    MPI_Irecv(&i_recv_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_CommPkgRecvProc(comm_pkg, i); 
      		vec_start = hypre_CommPkgRecvVecStart(comm_pkg,i);
      		vec_len = hypre_CommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
      		MPI_Isend(&i_send_data[vec_start], vec_len, MPI_INT,
			ip, 0, comm, &requests[j++]);
   	}
	break;
   }
   default :
   {
   	for (i = 0; i < num_recvs; i++)
   	{
      		ip = hypre_CommPkgRecvProc(comm_pkg, i); 
      		MPI_Irecv(MPI_BOTTOM, 1, 
                	hypre_CommPkgRecvMPIType(comm_pkg, i), 
			ip, 0, comm, &requests[j++]);
   	}
   	for (i = 0; i < num_sends; i++)
   	{
      		ip = hypre_CommPkgSendProc(comm_pkg, i); 
      		MPI_Isend(MPI_BOTTOM, 1, 
                	hypre_CommPkgSendMPIType(comm_pkg, i), 
			ip, 0, comm, &requests[j++]);
   	}
	break;
   }
   }
   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = hypre_CTAlloc(hypre_CommHandle, 1);

   hypre_CommHandleCommPkg(comm_handle)     = comm_pkg;
   hypre_CommHandleSendData(comm_handle)    = send_data;
   hypre_CommHandleRecvData(comm_handle)    = recv_data;
   hypre_CommHandleNumRequests(comm_handle) = num_requests;
   hypre_CommHandleRequests(comm_handle)    = requests;

   return ( comm_handle );
}

int
hypre_FinalizeCommunication( hypre_CommHandle *comm_handle )
{
   MPI_Status          *status0;
   int			ierr = 0;

   if (hypre_CommHandleNumRequests(comm_handle))
   {
      status0 = hypre_CTAlloc(MPI_Status,
                       hypre_CommHandleNumRequests(comm_handle));
      MPI_Waitall(hypre_CommHandleNumRequests(comm_handle),
                  hypre_CommHandleRequests(comm_handle), status0);
      hypre_TFree(status0);
   }

   hypre_TFree(hypre_CommHandleRequests(comm_handle));
   hypre_TFree(comm_handle);

   return ierr;
}

/* ----------------------------------------------------------------------
 * hypre_GenerateMatvecCommunicationInfo
 * generates the comm_pkg for A 
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d 
 * ---------------------------------------------------------------------*/

int
hypre_GenerateMatvecCommunicationInfo ( hypre_ParCSRMatrix *A)
{
   hypre_CommPkg	*comm_pkg;
   
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   MPI_Datatype         *recv_mpi_types;
   MPI_Datatype         *send_mpi_types;

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

   int	i, j, j2, k;
   int	*proc_mark, *tmp, *recv_buf, *displs, *info;
   int	num_procs, my_id, proc_num, num_elmts;
   int	local_info, index, index2, offset, offd_col;
   int	ierr = 0;
   int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   int	num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));

   MPI_Comm_size(comm, &num_procs);  
   MPI_Comm_rank(comm, &my_id);

   proc_mark = hypre_CTAlloc(int, num_procs);
   info = hypre_CTAlloc(int, num_procs);

/* ----------------------------------------------------------------------
 * determine which processors to receive from (set proc_mark) and num_recvs,
 * at the end of the loop proc_mark[i] contains the number of elements to be
 * received from Proc. i
 * ---------------------------------------------------------------------*/

   for (i=0; i < num_procs; i++)
		proc_mark[i] = 0;

   for (i=0; i < num_cols_offd; i++)
   {
	offd_col = col_map_offd[i];
	proc_num = 0;
	if (num_cols_diag) proc_num = hypre_min(num_procs-1,offd_col / num_cols_diag);
	while (col_starts[proc_num] > offd_col )
		proc_num = proc_num-1;
	while (col_starts[proc_num+1]-1 < offd_col )
		proc_num = proc_num+1;
	proc_mark[proc_num]++;
   }

   num_recvs=0;
   for (i=0; i < num_procs; i++)
	if (proc_mark[i]) num_recvs++;

   local_info = 2*num_recvs + num_cols_offd;
			
   MPI_Allgather(&local_info, 1, MPI_INT, info, 1, MPI_INT, comm); 

/* ----------------------------------------------------------------------
 * generate information to be send: tmp contains for each recv_proc:
 * id of recv_procs, number of elements to be received for this processor,
 * indices of elements (in this order)
 * ---------------------------------------------------------------------*/

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
	displs[i] = displs[i-1]+info[i-1];
   recv_procs = hypre_CTAlloc(int, num_recvs);
   recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   recv_mpi_types = hypre_CTAlloc(MPI_Datatype, num_recvs);
   recv_buf = hypre_CTAlloc(int, displs[num_procs]);
   tmp = hypre_CTAlloc(int, local_info);

   j = 0;
   j2 = 0;
   recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
	if (proc_mark[i])
	{
		recv_procs[j2] = i;
		recv_vec_starts[j2+1] = recv_vec_starts[j2]+proc_mark[i];
		MPI_Type_contiguous(proc_mark[i], MPI_DOUBLE,
			&recv_mpi_types[j2]);
		MPI_Type_commit(&recv_mpi_types[j2]);
		j2++;
		tmp[j++] = i;
		tmp[j++] = proc_mark[i];
		for (k=0; k < num_cols_offd; k++)
			if (col_map_offd[k] >= col_starts[i] && 
				col_map_offd[k] < col_starts[i+1])
				tmp[j++] = col_map_offd[k];
	}

   MPI_Allgatherv(tmp,local_info,MPI_INT,recv_buf,info,displs,MPI_INT,comm);
	

/* ----------------------------------------------------------------------
 * determine num_sends and number of elements to be send
 * ---------------------------------------------------------------------*/

   num_sends = 0;
   num_elmts = 0;
   for (i=0; i < num_procs; i++)
   {
	j = displs[i];
	while ( j < displs[i+1])
     	{
		if (recv_buf[j++] == my_id)
		{
			num_sends++;
			num_elmts += recv_buf[j];
			break;
		}
		j += recv_buf[j];
		j++;
	}	
   }
		
/* ----------------------------------------------------------------------
 * determine send_procs and actual elements to be send (in send_map_elmts)
 * and send_map_starts whose i-th entry points to the beginning of the 
 * elements to be send to proc. i
 * ---------------------------------------------------------------------*/

   send_procs = hypre_CTAlloc(int, num_sends);
   send_mpi_types = hypre_CTAlloc(MPI_Datatype, num_procs);
   send_map_starts = hypre_CTAlloc(int, num_sends+1);
   send_map_elmts = hypre_CTAlloc(int, num_elmts);
 
   index = 0;
   index2 = 0; 
   send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
	offset = first_col_diag;
	j = displs[i];
	while ( j < displs[i+1])
     	{
		if (recv_buf[j++] == my_id)
		{
			send_procs[index] = i;
			num_elmts = recv_buf[j++];
			MPI_Type_contiguous(num_elmts, MPI_DOUBLE,
					&send_mpi_types[index]);
			MPI_Type_commit(&send_mpi_types[index]);
			send_map_starts[index+1] = send_map_starts[index]
						+ num_elmts;
			index++;
			for (k = 0; k < num_elmts; k++)
				send_map_elmts[index2++] = recv_buf[j++]-offset;
			break;
		}
		j += recv_buf[j];
		j++;
	}	
   }
		
   comm_pkg = hypre_CTAlloc(hypre_CommPkg, 1);

   hypre_CommPkgComm(comm_pkg) = comm;

   hypre_CommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_CommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_CommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_CommPkgRecvMPITypes(comm_pkg) = recv_mpi_types;

   hypre_CommPkgNumSends(comm_pkg) = num_sends;
   hypre_CommPkgSendProcs(comm_pkg) = send_procs;
   hypre_CommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_CommPkgSendMapElmts(comm_pkg) = send_map_elmts;
   hypre_CommPkgSendMPITypes(comm_pkg) = send_mpi_types;

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;

   hypre_TFree(proc_mark); 
   hypre_TFree(tmp);
   hypre_TFree(recv_buf);
   hypre_TFree(displs);
   hypre_TFree(info);
 
   return ierr;
}

int
hypre_DestroyMatvecCommPkg(hypre_CommPkg *comm_pkg)
{
   int ierr = 0;

   hypre_TFree(hypre_CommPkgSendProcs(comm_pkg));
   hypre_TFree(hypre_CommPkgSendMapStarts(comm_pkg));
   hypre_TFree(hypre_CommPkgSendMapElmts(comm_pkg));
   hypre_TFree(hypre_CommPkgSendMPITypes(comm_pkg));
   hypre_TFree(hypre_CommPkgRecvProcs(comm_pkg));
   hypre_TFree(hypre_CommPkgRecvVecStarts(comm_pkg));
   hypre_TFree(hypre_CommPkgRecvMPITypes(comm_pkg));
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
