/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/



#include "headers.h"
#include "aux_interp.h"

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
#if 0

/* AHB - this has been replaced by the function after this one - we should
   delete this */

/* Inserts nodes to position expected for CF_marker_offd and P_marker_offd.
 * This is different than the send/recv vecs
 * explanation: old offd nodes take up first chunk of CF_marker_offd, new offd 
 * nodes take up the second chunk 0f CF_marker_offd. */
void insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker, 
		      HYPRE_Int *node_add, HYPRE_Int num_cols_A_offd, 
		      HYPRE_Int full_off_procNodes, HYPRE_Int num_procs,
		      HYPRE_Int *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  HYPRE_Int i, j, start, index, ip, min, max;
  HYPRE_Int num_sends, back_shift, original_off;
  HYPRE_Int num_recvs;
  HYPRE_Int *recv_procs;
  HYPRE_Int *recv_vec_starts;
  HYPRE_Int *end_diag, *int_buf_data;

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
  recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   
  index = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
  if(index < full_off_procNodes)
    index = full_off_procNodes;
  int_buf_data = hypre_CTAlloc(HYPRE_Int, index);

  index = 0;
  for (i = 0; i < num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); 
	 j++)
      int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  /* Sort fine_to_coarse so that the original off processors are first*/
  back_shift = 0;
  end_diag = hypre_CTAlloc(HYPRE_Int, num_procs);
  if(num_recvs)
  {
    min = recv_procs[0];
    max = min;
    for(i = 0; i < num_recvs; i++)
    {
      ip = recv_procs[i];
      start = recv_vec_starts[i];
      original_off = (recv_vec_starts[i+1] - start) - node_add[ip];
      end_diag[ip] = start + original_off;
      for(j = start; j < end_diag[ip]; j++)
	int_buf_data[j-back_shift] = OUT_marker[j];
      back_shift += node_add[ip];
      if(ip < min) min = ip;
      if(ip > max) max = ip;
    }
    back_shift = 0;
    for(i = min; i <= max; i++)
    {
      for(j = 0; j < node_add[i]; j++)
	int_buf_data[back_shift+j+num_cols_A_offd] = 
	  OUT_marker[end_diag[i]+j];
      back_shift += node_add[i];
    }
    
    for(i = 0; i < full_off_procNodes; i++)
      OUT_marker[i] = int_buf_data[i];

    hypre_TFree(int_buf_data);
    hypre_TFree(end_diag);
  } 
  return;
} 
#endif
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker
  
   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd 
   nodes from extend_comm_pkg take up the second chunk 0f CF_marker_offd. */



HYPRE_Int alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, 
                          hypre_ParCSRCommPkg *extend_comm_pkg,
                          HYPRE_Int *IN_marker, 
                          HYPRE_Int full_off_procNodes,
                          HYPRE_Int *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  HYPRE_Int i, j, start, index, shift;

  HYPRE_Int num_sends, num_recvs;
  
  HYPRE_Int *recv_vec_starts;

  HYPRE_Int e_num_sends;

  HYPRE_Int *int_buf_data;
  HYPRE_Int *e_out_marker;
  

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

  e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


  index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

  int_buf_data = hypre_CTAlloc(HYPRE_Int, index);

  /* orig commpkg data*/
  index = 0;
  
  for (i = 0; i < num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); 
	 j++)
      int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  /* now do the extend commpkg */

  /* first we need to shift our position in the OUT_marker */
  shift = recv_vec_starts[num_recvs];
  e_out_marker = OUT_marker + shift;
  
  index = 0;

  for (i = 0; i < e_num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i+1); 
	 j++)
       int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, extend_comm_pkg, int_buf_data, 
					      e_out_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  hypre_TFree(int_buf_data);
    
  return hypre_error_flag;
} 



/* AHB 11/06 : alternate to the extend function below - creates a
 * second comm pkg based on found - this makes it easier to use the
 * global partition*/
HYPRE_Int
hypre_ParCSRFindExtendCommPkg(hypre_ParCSRMatrix *A, HYPRE_Int newoff, HYPRE_Int *found, 
                              hypre_ParCSRCommPkg **extend_comm_pkg)

{
   

   HYPRE_Int			num_sends;
   HYPRE_Int			*send_procs;
   HYPRE_Int			*send_map_starts;
   HYPRE_Int			*send_map_elmts;
 
   HYPRE_Int			num_recvs;
   HYPRE_Int			*recv_procs;
   HYPRE_Int			*recv_vec_starts;

   hypre_ParCSRCommPkg	*new_comm_pkg;

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
  /* use found instead of col_map_offd in A, and newoff instead 
      of num_cols_offd*/

#if HYPRE_NO_GLOBAL_PARTITION

   HYPRE_Int        row_start=0, row_end=0, col_start = 0, col_end = 0;
   HYPRE_Int        global_num_cols;
   hypre_IJAssumedPart   *apart;
   
   hypre_ParCSRMatrixGetLocalRange( A,
                                    &row_start, &row_end ,
                                    &col_start, &col_end );
   

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A); 

   /* Create the assumed partition */
   if  (hypre_ParCSRMatrixAssumedPartition(A) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(A);
   }

   apart = hypre_ParCSRMatrixAssumedPartition(A);
   
   hypre_NewCommPkgCreate_core( comm, found, first_col_diag, 
                                col_start, col_end, 
                                newoff, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elmts, apart);

#else   
   HYPRE_Int  *col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_Int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   
   hypre_MatvecCommPkgCreate_core
      (
         comm, found, first_col_diag, col_starts,
         num_cols_diag, newoff,
         first_col_diag, found,
         1,
         &num_recvs, &recv_procs, &recv_vec_starts,
         &num_sends, &send_procs, &send_map_starts,
         &send_map_elmts
         );

#endif

   new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(new_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = send_map_elmts;



   *extend_comm_pkg = new_comm_pkg;
   

   return hypre_error_flag;
   
}

#if 0
/* this has been replaced by the function above - we should delete
 * this one*/

/* Add new communication patterns for new offd nodes */

void
hypre_ParCSRCommExtendA(hypre_ParCSRMatrix *A, HYPRE_Int newoff, HYPRE_Int *found,
			HYPRE_Int *p_num_recvs, HYPRE_Int **p_recv_procs,
			HYPRE_Int **p_recv_vec_starts, HYPRE_Int *p_num_sends,
			HYPRE_Int **p_send_procs, HYPRE_Int **p_send_map_starts,
			HYPRE_Int **p_send_map_elmts, HYPRE_Int **p_node_add)
{
  hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
  HYPRE_Int num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
  HYPRE_Int *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
  HYPRE_Int *recv_vec_starts_A = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
  HYPRE_Int *col_starts = hypre_ParCSRMatrixColStarts(A);
  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
  HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);

  HYPRE_Int new_num_recv;
  HYPRE_Int *new_recv_proc;   
  HYPRE_Int *tmp_recv_proc;
  HYPRE_Int *tmp;
  HYPRE_Int *new_recv_vec_starts;   
  HYPRE_Int num_sends;
  HYPRE_Int num_elmts;
  HYPRE_Int *send_procs;   
  HYPRE_Int *send_map_starts;   
  HYPRE_Int *send_map_elmts;   
  HYPRE_Int *nodes_recv;

  HYPRE_Int i, j, k;
  HYPRE_Int vec_len, vec_start;
  HYPRE_Int num_procs, my_id;
  HYPRE_Int num_requests;
  HYPRE_Int local_info;
  HYPRE_Int *info;
  HYPRE_Int *displs;
  HYPRE_Int *recv_buf;
  HYPRE_Int ip;
  HYPRE_Int glob_col;
  HYPRE_Int proc_found;
  HYPRE_Int *proc_add;
  HYPRE_Int *proc_mark;
  HYPRE_Int *new_map;
  HYPRE_Int *new_map_starts;
  HYPRE_Int j1,index;
  HYPRE_Int total_cols = num_cols_A_offd + newoff;
  HYPRE_Int *node_addition;

  MPI_Comm comm = hypre_ParCSRMatrixComm(A);
  hypre_MPI_Request *requests;
  hypre_MPI_Status *status;
  
  hypre_MPI_Comm_size(comm,&num_procs);
  hypre_MPI_Comm_rank(comm,&my_id);
  
  new_num_recv = num_recvs_A;

  /* Allocate vectors for temporary variables */
  tmp_recv_proc = hypre_CTAlloc(HYPRE_Int, num_procs);
  nodes_recv = hypre_CTAlloc(HYPRE_Int, num_procs);
  proc_add = hypre_CTAlloc(HYPRE_Int, num_procs);
  proc_mark = hypre_CTAlloc(HYPRE_Int, num_procs);
  info = hypre_CTAlloc(HYPRE_Int, num_procs);

  /* Initialize the new node proc counter (already accounted for procs
   * will stay 0 in the loop */
  for(i = 0; i < new_num_recv; i++)
    tmp_recv_proc[i] = recv_procs_A[i];

  /* Set up full offd map, col_map_offd only has neighbor off diag entries.
   * We need neighbor of neighbor nodes as well.*/
  new_map = hypre_CTAlloc(HYPRE_Int, total_cols);
  new_map_starts = hypre_CTAlloc(HYPRE_Int, num_procs+1);
  node_addition = hypre_CTAlloc(HYPRE_Int, num_procs);
  for(i = 0; i < num_procs; i++)
  {
    nodes_recv[i] = 0;
    node_addition[i] = 0;
  }

  if(newoff)
  {
    i = 0;
    k = 0;
    j1 = 0;
    index = 0;
    while(i < num_procs)
    {
      new_map_starts[i] = index;
      while(col_map_offd[j1] < col_starts[i+1] && j1 < num_cols_A_offd)
      {
	new_map[index] = col_map_offd[j1];
	j1++;
	index++;
      }
      proc_found = index;
      for(j = 0; j < newoff; j++)
	if(i != my_id)
	  if(found[j] < col_starts[i+1] && found[j] >= col_starts[i])
	  {
	    new_map[index] = found[j];
	    node_addition[k]++;
	    index++;
	  }
      /* Sort new nodes at end */
      if(proc_found < index - 1)
	hypre_ssort(&new_map[index - (index-proc_found)],index-proc_found);
      k++;
      i++;
    }
    new_map_starts[num_procs] = index;
  }
  else
    for(i = 0; i < num_cols_A_offd; i++)
      new_map[i] = col_map_offd[i];

  /* Loop through the neighbor of neighbor nodes to determine proc
   * ownership. Add node to list of receives from that proc. */
  for(i = num_cols_A_offd; i < num_cols_A_offd + newoff; i++)
  { 
    glob_col = found[i-num_cols_A_offd];
    j = 1;
    while(j <= num_procs)
    {
      if(glob_col < col_starts[j])
      {
	proc_found = 0;
	k = 0;
	while(k < num_recvs_A)
	{
	  if(recv_procs_A[k] == j-1)
	  {
	    proc_found = 1;
	    nodes_recv[j-1]++;
	    k = num_recvs_A;
	  }
	  else
	    k++;
	}
	if(!proc_found)
	  while(k < new_num_recv)
	  {
	    if(tmp_recv_proc[k] == j-1)
	    {
	      proc_found = 1;
	      nodes_recv[j-1]++;
	      k = new_num_recv;
	    }
	    else
	      k++;
	  }
	if(!proc_found)
	{
	  tmp_recv_proc[new_num_recv] = j-1;
	  nodes_recv[j-1]++;
	  new_num_recv++;
	}
	j = num_procs + 1;
      }
      j++;
    }
  }
  
  new_recv_proc = hypre_CTAlloc(HYPRE_Int, new_num_recv);
  for(i = 0; i < new_num_recv; i++)
    new_recv_proc[i] = tmp_recv_proc[i];

  new_recv_vec_starts = hypre_CTAlloc(HYPRE_Int, new_num_recv+1);
  new_recv_vec_starts[0] = 0;
 
  /* Now tell new processors that they need to send some info and change
   * their send comm*/
  local_info = 2*new_num_recv;
  hypre_MPI_Allgather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, comm); 

  /* ----------------------------------------------------------------------
   * generate information to be sent: tmp contains for each recv_proc:
   * id of recv_procs, number of elements to be received for this processor,
   * indices of elements (in this order)
   * ---------------------------------------------------------------------*/
   displs = hypre_CTAlloc(HYPRE_Int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
	displs[i] = displs[i-1]+info[i-1]; 
   recv_buf = hypre_CTAlloc(HYPRE_Int, displs[num_procs]); 

   tmp = hypre_CTAlloc(HYPRE_Int, local_info);

   j = 0;
   /* Load old information if recv proc was already in comm */  
   for (i=0; i < num_recvs_A; i++)
   {
     num_elmts = recv_vec_starts_A[i+1]-recv_vec_starts_A[i] + 
       nodes_recv[new_recv_proc[i]];
     new_recv_vec_starts[i+1] = new_recv_vec_starts[i]+num_elmts;
     tmp[j++] = new_recv_proc[i];
     tmp[j++] = num_elmts;
   }
   /* Add information if recv proc was added */
   for (i=num_recvs_A; i < new_num_recv; i++)
   {
     num_elmts = nodes_recv[new_recv_proc[i]];
     new_recv_vec_starts[i+1] = new_recv_vec_starts[i]+num_elmts;
     tmp[j++] = new_recv_proc[i];
     tmp[j++] = num_elmts;
   }

   hypre_MPI_Allgatherv(tmp,local_info,HYPRE_MPI_INT,recv_buf,info,displs,HYPRE_MPI_INT,comm);
	
   hypre_TFree(tmp);

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
      send_procs = hypre_CTAlloc(HYPRE_Int, num_sends);
      send_map_elmts = hypre_CTAlloc(HYPRE_Int, proc_add[num_sends]);
   }
   send_map_starts = hypre_CTAlloc(HYPRE_Int, num_sends+1);
   num_requests = new_num_recv+num_sends;
   if (num_requests)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request, num_requests);
      status = hypre_CTAlloc(hypre_MPI_Status, num_requests);
   }

   if (num_sends) send_map_starts[0] = 0;
   for (i=0; i < num_sends; i++)
   {
      send_map_starts[i+1] = proc_add[i+1];
      send_procs[i] = proc_mark[i];
   }

   j=0;
   for (i=0; i < new_num_recv; i++)
   {
     vec_start = new_recv_vec_starts[i];
     vec_len = new_recv_vec_starts[i+1] - vec_start;
     ip = new_recv_proc[i];
     if(newoff)
       vec_start = new_map_starts[ip];
     hypre_MPI_Isend(&new_map[vec_start], vec_len, HYPRE_MPI_INT,
	       ip, 0, comm, &requests[j++]);
   }
   for (i=0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      hypre_MPI_Irecv(&send_map_elmts[vec_start], vec_len, HYPRE_MPI_INT,
                        ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      hypre_MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   if (num_sends)
     for (i=0; i<send_map_starts[num_sends]; i++)
       send_map_elmts[i] -= first_col_diag;
   
   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = new_num_recv;
   *p_recv_procs = new_recv_proc;
   *p_recv_vec_starts = new_recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;
   *p_node_add = node_addition;

   /* De-allocate memory */
   hypre_TFree(tmp_recv_proc);
   hypre_TFree(nodes_recv);   
   hypre_TFree(proc_add);
   hypre_TFree(proc_mark); 
   hypre_TFree(new_map);
   hypre_TFree(recv_buf);
   hypre_TFree(displs);
   hypre_TFree(info);
   hypre_TFree(new_map_starts);

   return;
}

#endif

/* sort for non-ordered arrays */
HYPRE_Int hypre_ssort(HYPRE_Int *data, HYPRE_Int n)
{
  HYPRE_Int i,si;               
  HYPRE_Int change = 0;
  
  if(n > 0)
    for(i = n-1; i > 0; i--){
      si = index_of_minimum(data,i+1);
      if(i != si)
      {
	swap_int(data, i, si);
	change = 1;
      }
    }                                                                       
  return change;
}

/* Auxilary function for hypre_ssort */
HYPRE_Int index_of_minimum(HYPRE_Int *data, HYPRE_Int n)
{
  HYPRE_Int answer;
  HYPRE_Int i;
                                                                               
  answer = 0;
  for(i = 1; i < n; i++)
    if(data[answer] < data[i])
      answer = i;
                                                                               
  return answer;
}
                                                                               
void swap_int(HYPRE_Int *data, HYPRE_Int a, HYPRE_Int b)
{
  HYPRE_Int temp;
                                                                               
  temp = data[a];
  data[a] = data[b];
  data[b] = temp;

  return;
}

/* Initialize CF_marker_offd, CF_marker, P_marker, P_marker_offd, tmp */
void initialize_vecs(HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc, HYPRE_Int *offd_ftc, 
		     HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF)
{
  HYPRE_Int i;

  /* Quicker initialization */
  if(offd_n < diag_n)
  {
    for(i = 0; i < offd_n; i++)
    {
      diag_ftc[i] = -1;
      offd_ftc[i] = -1;
      diag_pm[i] = -1;
      offd_pm[i] = -1;
      tmp_CF[i] = -1;
    }
    for(i = offd_n; i < diag_n; i++)
    { 
      diag_ftc[i] = -1;
      diag_pm[i] = -1;
    }
  }
  else
  {
    for(i = 0; i < diag_n; i++)
    {
      diag_ftc[i] = -1;
      offd_ftc[i] = -1;
      diag_pm[i] = -1;
      offd_pm[i] = -1;
      tmp_CF[i] = -1;
    }
    for(i = diag_n; i < offd_n; i++)
    { 
      offd_ftc[i] = -1;
      offd_pm[i] = -1;
      tmp_CF[i] = -1;
    }
  }
  return;
}

/* Find nodes that are offd and are not contained in original offd
 * (neighbors of neighbors) */
HYPRE_Int new_offd_nodes(HYPRE_Int **found, HYPRE_Int num_cols_A_offd, HYPRE_Int *A_ext_i, HYPRE_Int *A_ext_j, 
		   HYPRE_Int num_cols_S_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, 
		   HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j,
		   HYPRE_Int *CF_marker, hypre_ParCSRCommPkg *comm_pkg)
{
  HYPRE_Int i, i1, ii, j, ifound, kk, k1;
  HYPRE_Int got_loc, loc_col;

  HYPRE_Int min;

  HYPRE_Int size_offP;

  HYPRE_Int *tmp_found;
  HYPRE_Int *CF_marker_offd = NULL;
  HYPRE_Int *int_buf_data;
  HYPRE_Int newoff = 0;
  HYPRE_Int full_off_procNodes = 0;
  hypre_ParCSRCommHandle *comm_handle;
                                                                                                                                         
  CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);
  int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                hypre_ParCSRCommPkgNumSends(comm_pkg)));
  ii = 0;
  for (i=0; i < hypre_ParCSRCommPkgNumSends(comm_pkg); i++)
  {
      for (j=hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
                j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        int_buf_data[ii++]
          = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
  comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,int_buf_data,
        CF_marker_offd);
  hypre_ParCSRCommHandleDestroy(comm_handle);
  hypre_TFree(int_buf_data);

  size_offP = A_ext_i[num_cols_A_offd];
  tmp_found = hypre_CTAlloc(HYPRE_Int, size_offP);

  /* Find nodes that will be added to the off diag list */ 
  for (i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
    for (j = A_ext_i[i]; j < A_ext_i[i+1]; j++)
    {
      i1 = A_ext_j[j];
      if(i1 < col_1 || i1 >= col_n)
      {
	  ifound = hypre_BinarySearch(col_map_offd,i1,num_cols_A_offd);
	  if(ifound == -1)
	  {
	      tmp_found[newoff]=i1;
	      newoff++;
	  }
	  else
	  {
	      A_ext_j[j] = -ifound-1;
	  }
      }
    }
   }
  }
  /* Put found in monotone increasing order */
  if (newoff > 0)
  {
     qsort0(tmp_found,0,newoff-1);
     ifound = tmp_found[0];
     min = 1;
     for (i=1; i < newoff; i++)
     {
       if (tmp_found[i] > ifound)
       {
          ifound = tmp_found[i];
          tmp_found[min++] = ifound;
       }
     }
     newoff = min;
  }

  full_off_procNodes = newoff + num_cols_A_offd;
  /* Set column indices for Sop and A_ext such that offd nodes are
   * negatively indexed */
  for(i = 0; i < num_cols_S_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
     for(kk = Sop_i[i]; kk < Sop_i[i+1]; kk++)
     {
       k1 = Sop_j[kk];
       if(k1 < col_1 || k1 >= col_n)
       { 
	 if(newoff < num_cols_A_offd)
	 {  
	   got_loc = hypre_BinarySearch(tmp_found,k1,newoff);
	   if(got_loc > -1)
	     loc_col = got_loc + num_cols_A_offd;
	   else
	     loc_col = hypre_BinarySearch(col_map_offd,k1,
					  num_cols_A_offd);
	 }
	 else
	 {
	   loc_col = hypre_BinarySearch(col_map_offd,k1,
					num_cols_A_offd);
	   if(loc_col == -1)
	     loc_col = hypre_BinarySearch(tmp_found,k1,newoff) +
	       num_cols_A_offd;
	 }
	 if(loc_col < 0)
	 {
	   hypre_printf("Could not find node: STOP\n");
	   return(-1);
	 }
	 Sop_j[kk] = -loc_col - 1;
       }
     }
   }
  }
  for(i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
     for (kk = A_ext_i[i]; kk < A_ext_i[i+1]; kk++)
     {
       k1 = A_ext_j[kk];
       if(k1 > -1 && (k1 < col_1 || k1 >= col_n))
       {
	 got_loc = hypre_BinarySearch(tmp_found,k1,newoff);
	 loc_col = got_loc + num_cols_A_offd;
	 A_ext_j[kk] = -loc_col - 1;
       }
     }
   }
  }


  hypre_TFree(CF_marker_offd);
  

  *found = tmp_found;
 


  return newoff;
}
