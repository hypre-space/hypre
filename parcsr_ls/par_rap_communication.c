/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"

int
hypre_GetCommPkgRTFromCommPkgA( hypre_ParCSRMatrix *RT,
			       	hypre_ParCSRMatrix *A)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(RT);
   hypre_CommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs_A = hypre_CommPkgNumRecvs(comm_pkg_A);
   int *recv_procs_A = hypre_CommPkgRecvProcs(comm_pkg_A);
   int num_sends_A = hypre_CommPkgNumSends(comm_pkg_A);
   int *send_procs_A = hypre_CommPkgSendProcs(comm_pkg_A);

   int num_recvs_RT;
   int *recv_procs_RT;   
   int *recv_vec_starts_RT;   
   int num_sends_RT;
   int *send_procs_RT;   
/*   int *send_map_starts_RT;   
   int *send_map_elmts_RT;   */

   int *col_map_offd_RT = hypre_ParCSRMatrixColMapOffd(RT);
   int *partitioning = hypre_ParCSRMatrixColStarts(RT);
   int num_cols_offd_RT = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd(RT));

   int i, j;
   int num_procs, my_id;
   int ierr = 0;
   int num_requests;
   int offd_col, proc_num;
 
   int *proc_mark;
   int *change_array;

   MPI_Request *requests;
   MPI_Status *status;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   proc_mark = hypre_CTAlloc(int, num_procs);

   for (i=0; i < num_procs; i++)
                proc_mark[i] = 0;
 
/*--------------------------------------------------------------------------
 * determine num_recvs, recv_procs and recv_vec_starts for RT
 *--------------------------------------------------------------------------*/

   proc_num = 0;
   for (i=0; i < num_cols_offd_RT; i++)
   {
        offd_col = col_map_offd_RT[i];
        while (partitioning[proc_num+1]-1 < offd_col )
                proc_num++;
        proc_mark[proc_num]++;
   }
 
   num_recvs_RT = 0;
   for (i=0; i < num_procs; i++)
        if (proc_mark[i]) num_recvs_RT++;

   recv_procs_RT = hypre_CTAlloc(int,num_recvs_RT);
   recv_vec_starts_RT = hypre_CTAlloc(int, num_recvs_RT+1);
 
   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i=0; i < num_procs; i++)
        if (proc_mark[i])
        {
                recv_procs_RT[j] = i;
                recv_vec_starts_RT[j+1] = recv_vec_starts_RT[j]+proc_mark[i];
                j++;
        }


/*--------------------------------------------------------------------------
 * send num_changes to recv_procs_A and receive change_array from send_procs_A
 *--------------------------------------------------------------------------*/

   num_requests = num_recvs_A+num_sends_A;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   change_array = hypre_CTAlloc(int, num_sends_A);

   j = 0;
   for (i=0; i < num_sends_A; i++)
	MPI_Irecv(&change_array[i],1,MPI_INT,send_procs_A[i],0,comm,
		&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
	MPI_Isend(&proc_mark[recv_procs_A[i]],1,MPI_INT,recv_procs_A[i],0,comm,
		&requests[j++]);
   
   MPI_Waitall(num_requests,requests,status);

   hypre_TFree(proc_mark);
   hypre_TFree(requests);
   hypre_TFree(status);
/*--------------------------------------------------------------------------
 * if change_array[i] is 0 , omit send_procs_A[i] in send_procs_RT
 *--------------------------------------------------------------------------*/

   num_sends_RT = 0;
   for (i=0; i < num_sends_A; i++)
	if (change_array[i]) 
	{
		num_sends_RT++;
	}

   send_procs_RT = hypre_CTAlloc(int, num_sends_RT);

   j = 0;
   for (i=0; i < num_sends_A; i++)
	if (change_array[i]) 
	{
		send_procs_RT[j++] = send_procs_A[i];
	}

/*--------------------------------------------------------------------------
 * generate send_map_starts and send_map_elmts
 *--------------------------------------------------------------------------*/

   hypre_ParCSRMatrixCommPkg(RT) = hypre_GenerateSendMapAndCommPkg(comm, 
			num_sends_RT, num_recvs_RT,
			recv_procs_RT, send_procs_RT, 
			recv_vec_starts_RT, RT);

/*   printf (" my_id %d num_sends %d num_recvs %d \n", my_id,
	num_sends_RT, num_recvs_RT);
   send_map_starts_RT = 
	hypre_CommPkgSendMapStarts(hypre_ParCSRMatrixCommPkg(RT));
   send_map_elmts_RT = 
	hypre_CommPkgSendMapElmts(hypre_ParCSRMatrixCommPkg(RT));
   for (i=0; i < num_sends_RT; i++)
   {
	printf (" send_procs %d send_map_starts %d\n", send_procs_RT[i],
	send_map_starts_RT[i+1]);
	for (j=send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j++)
		printf(" j %d send_map_elmts %d\n", j, send_map_elmts_RT[j]);
   }
   for (i=0; i < num_recvs_RT; i++)
   {
	printf (" recv_procs %d recv_vec_starts %d\n", recv_procs_RT[i],
	recv_vec_starts_RT[i+1]);
   }
*/
   
   hypre_TFree(change_array);

   return ierr;
}

hypre_CommPkg *
hypre_GenerateSendMapAndCommPkg(MPI_Comm comm, int num_sends, int num_recvs,
				int *recv_procs, int *send_procs,
				int *recv_vec_starts, hypre_ParCSRMatrix *A)
{
   int *send_map_starts;
   int *send_map_elmts;
   int i, j;
   int num_requests = num_sends+num_recvs;
   MPI_Request *requests;
   MPI_Status *status;
   int vec_len, vec_start;
   hypre_CommPkg *comm_pkg;
   int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);

/*--------------------------------------------------------------------------
 * generate send_map_starts and send_map_elmts
 *--------------------------------------------------------------------------*/
   requests = hypre_CTAlloc(MPI_Request,num_requests);
   status = hypre_CTAlloc(MPI_Status,num_requests);
   send_map_starts = hypre_CTAlloc(int, num_sends+1);
   j = 0;
   for (i=0; i < num_sends; i++)
	MPI_Irecv(&send_map_starts[i+1],1,MPI_INT,send_procs[i],0,comm,
		&requests[j++]);

   for (i=0; i < num_recvs; i++)
   {
	vec_len = recv_vec_starts[i+1] - recv_vec_starts[i];
	MPI_Isend(&vec_len,1,MPI_INT, recv_procs[i],0,comm,&requests[j++]);
   }
   
   MPI_Waitall(j,requests,status);
 
   send_map_starts[0] = 0; 
   for (i=0; i < num_sends; i++)
	send_map_starts[i+1] += send_map_starts[i]; 

   send_map_elmts = hypre_CTAlloc(int,send_map_starts[num_sends]);

   j = 0;
   for (i=0; i < num_sends; i++)
   {
	vec_start = send_map_starts[i];
	vec_len = send_map_starts[i+1]-vec_start;
	MPI_Irecv(&send_map_elmts[vec_start],vec_len,MPI_INT,
		send_procs[i],0,comm,&requests[j++]);
   }

   for (i=0; i < num_recvs; i++)
   {
	vec_start = recv_vec_starts[i];
	vec_len = recv_vec_starts[i+1] - vec_start;
	MPI_Isend(&col_map_offd[vec_start],vec_len,MPI_INT, 
		recv_procs[i],0,comm,&requests[j++]);
   }
   
   MPI_Waitall(j,requests,status);

   for (i=0; i < send_map_starts[num_sends]; i++)
	send_map_elmts[i] -= first_col_diag; 
	
   comm_pkg = hypre_CTAlloc(hypre_CommPkg,1);

   hypre_CommPkgComm(comm_pkg) = comm;
   hypre_CommPkgNumSends(comm_pkg) = num_sends;
   hypre_CommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_CommPkgSendProcs(comm_pkg) = send_procs;
   hypre_CommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_CommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_CommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_CommPkgSendMapElmts(comm_pkg) = send_map_elmts;

   hypre_TFree(status);
   hypre_TFree(requests);

   return comm_pkg;
}

int
hypre_GenerateRAPCommPkg( hypre_ParCSRMatrix *RAP,
			  hypre_ParCSRMatrix *A)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(RAP);
   hypre_CommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs_A = hypre_CommPkgNumRecvs(comm_pkg_A);
   int *recv_procs_A = hypre_CommPkgRecvProcs(comm_pkg_A);
   int num_sends_A = hypre_CommPkgNumSends(comm_pkg_A);
   int *send_procs_A = hypre_CommPkgSendProcs(comm_pkg_A);

   int num_recvs_RAP;
   int *recv_procs_RAP;   
   int *recv_vec_starts_RAP;   
   int num_sends_RAP;
   int *send_procs_RAP;   
/*   int *send_map_starts_RAP;   
   int *send_map_elmts_RAP;   */

   int *col_map_offd_RAP = hypre_ParCSRMatrixColMapOffd(RAP);
   int *partitioning = hypre_ParCSRMatrixRowStarts(RAP);
   int num_cols_offd_RAP = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd(RAP));

   int i, j, k, cnt;
   int num_procs, my_id;
   int ierr = 0;
   int num_requests;
   int offd_col, proc_num, change;
   int num_changes, total_num_procs, num_new_send_procs;
 
   int *proc_mark;
   int *work, *change_array, *changed_procs;
   int *proc_vec_starts, *recv_buf, *flag;
   int *send_starts, *send_list, *recv_starts;
   int *new_send_procs;

   MPI_Request *requests;
   MPI_Status *status;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   proc_mark = hypre_CTAlloc(int, num_procs);

   for (i=0; i < num_procs; i++)
                proc_mark[i] = 0;
 
/*--------------------------------------------------------------------------
 * determine num_recvs, recv_procs and recv_vec_starts for RAP
 *--------------------------------------------------------------------------*/

   proc_num = 0;
   for (i=0; i < num_cols_offd_RAP; i++)
   {
        offd_col = col_map_offd_RAP[i];
        while (partitioning[proc_num+1]-1 < offd_col )
                proc_num++;
        proc_mark[proc_num]++;
   }
 
   num_recvs_RAP = 0;
   for (i=0; i < num_procs; i++)
        if (proc_mark[i]) num_recvs_RAP++;

   recv_procs_RAP = hypre_CTAlloc(int,num_recvs_RAP);
   recv_vec_starts_RAP = hypre_CTAlloc(int, num_recvs_RAP+1);
 
   j = 0;
   recv_vec_starts_RAP[0] = 0;
   for (i=0; i < num_procs; i++)
        if (proc_mark[i])
        {
                recv_procs_RAP[j] = i;
                recv_vec_starts_RAP[j+1] = recv_vec_starts_RAP[j]+proc_mark[i];
                j++;
        }

   hypre_TFree(proc_mark);

/*--------------------------------------------------------------------------
 * determine if recv_procs_A differs from recv_procs_RAP
 *--------------------------------------------------------------------------*/

   work = hypre_CTAlloc(int,num_recvs_RAP+num_recvs_A);

   change = 0;
   j = 0;

   if (!num_recvs_RAP)
   {
	change = num_recvs_A;
	for (i=0; i < num_recvs_A; i++)
		work[i] = -recv_procs_A[i]-1;
   }
   else if (!num_recvs_A)
   {
	change = num_recvs_RAP;
	for (i=0; i < num_recvs_RAP; i++)
		work[i] = recv_procs_RAP[i]+1;
   }
   else	
   {
      for (i=0; i < num_recvs_RAP ; i++)
      {
	 if (recv_procs_A[j] == recv_procs_RAP[i])
	    j++;
	 else
	 {
	    if (recv_procs_A[j] > recv_procs_RAP[i])
		work[change++] = recv_procs_RAP[i]+1;
	    else
	    {
		work[change++] = -recv_procs_A[j]-1;
		j++;
		i--;
	    }
	 }
      }
      for (i=j; i < num_recvs_A; i++)
	 work[change++] = -recv_procs_A[i]-1;
   }
/*--------------------------------------------------------------------------
 * send num_changes to recv_procs_A and receive change_array from send_procs_A
 *--------------------------------------------------------------------------*/

   num_requests = num_recvs_A+num_sends_A;
   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   change_array = hypre_CTAlloc(int, num_sends_A);

   j = 0;
   for (i=0; i < num_sends_A; i++)
	MPI_Irecv(&change_array[i],1,MPI_INT,send_procs_A[i],0,comm,
		&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
	MPI_Isend(&change,1,MPI_INT,recv_procs_A[i],0,comm,
		&requests[j++]);
   
   MPI_Waitall(num_requests,requests,status);

/*--------------------------------------------------------------------------
 * if there was a change send and receive recv_procs_RAP
 *--------------------------------------------------------------------------*/

   num_changes = 0;
   total_num_procs = 0;
   for (i=0; i < num_sends_A; i++)
	if (change_array[i]) 
	{
		num_changes++;
		total_num_procs += change_array[i];
	}

   changed_procs = hypre_CTAlloc(int, num_changes);
   proc_vec_starts = hypre_CTAlloc(int, num_changes+1);

   j = 0;
   proc_vec_starts[0] = 0;
   for (i=0; i < num_sends_A; i++)
	if (change_array[i]) 
	{
		changed_procs[j++] = send_procs_A[i];
		proc_vec_starts[j] = proc_vec_starts[j-1]+change_array[i];
	}

   recv_buf = hypre_CTAlloc(int, total_num_procs);

   j = 0;
   for (i=0; i < num_changes; i++)
	MPI_Irecv(&recv_buf[proc_vec_starts[i]],proc_vec_starts[i+1]
		-proc_vec_starts[i],MPI_INT,changed_procs[i],0,comm,
		&requests[j++]);

   if (change)
   	for (i=0; i < num_recvs_A; i++)
		MPI_Isend(work,change,MPI_INT,
			recv_procs_A[i],0,comm,
			&requests[j++]);
   
   MPI_Waitall(j,requests,status);

/*--------------------------------------------------------------------------
 * Now examine recv_buf for changes in send_procs,
 * if for changed_procs[i] recv_buf contains -(my_id+1) changed_procs[i]
 * is not contained in send_procs_RAP, 
 * if recv_buf contains k+1 and k is in recv_procs_A , proc k needs to
 * add changed_procs[i] to its send_procs_RAP, i.e. 
 *--------------------------------------------------------------------------*/

   num_sends_RAP = num_sends_A;
   for (i=0; i < num_sends_A; i++)
	work[i] = 1;

   flag = hypre_CTAlloc(int, num_recvs_A);

   for (i=0; i < num_recvs_A; i++)
	flag[i] = 0;

   for (i=0; i < num_changes; i++)
	for (j=proc_vec_starts[i]; j < proc_vec_starts[i+1]; j++)
	{
	    if (recv_buf[j] == -(my_id+1))
	    {
		for (k=0; k < num_sends_A; k++)
		   if (send_procs_A[k] == changed_procs[i])
			{ work[k] = 0; break; }
		num_sends_RAP--;
	    }
 	    for (k=0; k < num_recvs_A; k++)
		if (recv_buf[j]-1 == recv_procs_A[k])
			flag[k]++;
	}

/*--------------------------------------------------------------------------
 * flag to recv_procs_A number of processors to be added to send_procs_RAP
 *--------------------------------------------------------------------------*/

   j = 0;
   for (i=0; i < num_sends_A; i++)
	MPI_Irecv(&change_array[i],1,MPI_INT,send_procs_A[i],0,comm,
		&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
	MPI_Isend(&flag[i],1,MPI_INT,recv_procs_A[i],0,comm,
		&requests[j++]);
   
   MPI_Waitall(num_requests,requests,status);

/*--------------------------------------------------------------------------
 * Now examine recv_buf for changes in send_procs,
 * if recv_buf contains k+1 and k is in recv_procs_A , proc k needs to
 * add changed_procs[i] to its send_procs_RAP, i.e. 
 *--------------------------------------------------------------------------*/

   send_starts = hypre_CTAlloc(int,num_recvs_A+1);
   recv_starts = hypre_CTAlloc(int,num_sends_A+1);

   cnt = 0;
   for (k=0; k < num_recvs_A; k++)
   {
  	send_starts[k] = cnt;
 	for (i=0; i < num_changes; i++)
  	{
		for (j=proc_vec_starts[i]; j < proc_vec_starts[i+1]; j++)
		{
			if (recv_buf[j]-1 == recv_procs_A[k]) cnt++;
  		}
  	}
   }
   send_starts[num_recvs_A] = cnt;

   send_list = hypre_CTAlloc(int,cnt);

   cnt = 0;
   for (k=0; k < num_recvs_A; k++)
   {
 	for (i=0; i < num_changes; i++)
		for (j=proc_vec_starts[i]; j < proc_vec_starts[i+1]; j++)
			if (recv_buf[j]-1 == recv_procs_A[k])
				send_list[cnt++] = changed_procs[i];	
   }

   recv_starts[0] = 0;
   for (k=0; k < num_sends_A; k++)
  	recv_starts[k+1] = recv_starts[k]+change_array[k];
   
/*--------------------------------------------------------------------------
 * flag to recv_procs_A processors to be added to send_procs_RAP
 *--------------------------------------------------------------------------*/
   
   num_new_send_procs = recv_starts[num_sends_A];
   new_send_procs = hypre_CTAlloc(int, num_new_send_procs);

   j = 0;
   for (i=0; i < num_sends_A; i++)
	if (change_array[i])
	MPI_Irecv(&new_send_procs[recv_starts[i]],change_array[i],MPI_INT,
		send_procs_A[i],0,comm,&requests[j++]);

   for (i=0; i < num_recvs_A; i++)
	if (flag[i])
	MPI_Isend(&send_list[send_starts[i]],flag[i],MPI_INT,
		recv_procs_A[i],0,comm,&requests[j++]);
   
   MPI_Waitall(j,requests,status);

/*--------------------------------------------------------------------------
 * generate send_procs_RAP and num_sends_RAP
 *--------------------------------------------------------------------------*/

   for (i = 0; i < num_new_send_procs ; i++)
   {
	while (new_send_procs[i] == -1) i++;
	for (j = i+1; j < num_new_send_procs; j++)
	{
		if (new_send_procs[j] == new_send_procs[i])
			new_send_procs[j] = -1;
	}
   }

   for (i=0; i < num_new_send_procs; i++)
	if (new_send_procs[i] != -1) num_sends_RAP++;

   send_procs_RAP = hypre_CTAlloc(int, num_sends_RAP);

   cnt = 0;
   for (i=0; i < num_sends_A; i++)
	if (work[i]) send_procs_RAP[cnt++] = send_procs_A[i];

   for (i=0; i < num_new_send_procs; i++)
	if (new_send_procs[i] != -1) send_procs_RAP[cnt++] = new_send_procs[i];

/*--------------------------------------------------------------------------
 * generate send_map_starts and send_map_elmts
 *--------------------------------------------------------------------------*/

   hypre_ParCSRMatrixCommPkg(RAP) = hypre_GenerateSendMapAndCommPkg(comm,
                        num_sends_RAP, num_recvs_RAP,
                        recv_procs_RAP, send_procs_RAP,
                        recv_vec_starts_RAP, RAP);

/*   send_map_starts_RAP = 
	hypre_CommPkgSendMapStarts(hypre_ParCSRMatrixCommPkg(RAP));
   send_map_elmts_RAP = 
	hypre_CommPkgSendMapElmts(hypre_ParCSRMatrixCommPkg(RAP));
   printf (" my_id %d num_sends %d num_recvs %d \n", my_id,
	num_sends_RAP, num_recvs_RAP);
   for (i=0; i < num_sends_RAP; i++)
   {
	printf (" send_procs %d send_map_starts %d\n", send_procs_RAP[i],
	send_map_starts_RAP[i+1]);
	for (j=send_map_starts_RAP[i]; j < send_map_starts_RAP[i+1]; j++)
		printf(" j %d send_map_elmts %d\n", j, send_map_elmts_RAP[j]);
   }
   for (i=0; i < num_recvs_RAP; i++)
   {
	printf (" recv_procs %d recv_vec_starts %d\n", recv_procs_RAP[i],
	recv_vec_starts_RAP[i+1]);
   }
*/
   
   hypre_TFree(work);
   hypre_TFree(flag);
   hypre_TFree(send_list);
   hypre_TFree(send_starts);
   hypre_TFree(recv_starts);
   hypre_TFree(recv_buf);
   hypre_TFree(proc_vec_starts);
   hypre_TFree(change_array);
   hypre_TFree(changed_procs);
   hypre_TFree(new_send_procs);
   hypre_TFree(status);
   hypre_TFree(requests);

   return ierr;
}
