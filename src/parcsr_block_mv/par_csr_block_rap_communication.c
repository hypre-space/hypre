/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/





#include "headers.h"


/* This function is the same as hypre_GetCommPkgRTFromCommPkgA, except that the
arguments are Block matrices.  We should change the code to take the commpkgs as input
(and a couple of other items) and then we would not need two functions. (Because
the commpkg is not different for a block matrix.) */



int
hypre_GetCommPkgBlockRTFromCommPkgBlockA( hypre_ParCSRBlockMatrix *RT,
			       	hypre_ParCSRBlockMatrix *A,
			       	int *fine_to_coarse_offd)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(RT);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRBlockMatrixCommPkg(A);
   int num_recvs_A = hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   int *recv_procs_A = hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   int *recv_vec_starts_A = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   int num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   int *send_procs_A = hypre_ParCSRCommPkgSendProcs(comm_pkg_A);

   hypre_ParCSRCommPkg *comm_pkg;
   int num_recvs_RT;
   int *recv_procs_RT;   
   int *recv_vec_starts_RT;   
   int num_sends_RT;
   int *send_procs_RT;   
   int *send_map_starts_RT;   
   int *send_map_elmts_RT;   

   int *col_map_offd_RT = hypre_ParCSRBlockMatrixColMapOffd(RT);
   int num_cols_offd_RT = hypre_CSRBlockMatrixNumCols( hypre_ParCSRMatrixOffd(RT));
   int first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(RT);

   int i, j;
   int vec_len, vec_start;
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

/*--------------------------------------------------------------------------
 * determine num_recvs, recv_procs and recv_vec_starts for RT
 *--------------------------------------------------------------------------*/

   proc_mark = hypre_CTAlloc(int, num_recvs_A);

   for (i=0; i < num_recvs_A; i++)
                proc_mark[i] = 0;

   proc_num = 0;
   num_recvs_RT = 0;
   if (num_cols_offd_RT)
   {
      for (i=0; i < num_recvs_A; i++)
      {
         for (j=recv_vec_starts_A[i]; j<recv_vec_starts_A[i+1]; j++)
         {
            offd_col = col_map_offd_RT[proc_num];
            if (offd_col == j)
            {
                proc_mark[i]++;
                proc_num++;
                if (proc_num == num_cols_offd_RT) break;
            }
         }
         if (proc_mark[i]) num_recvs_RT++;
         if (proc_num == num_cols_offd_RT) break;
      }
   }

   for (i=0; i < num_cols_offd_RT; i++)
      col_map_offd_RT[i] = fine_to_coarse_offd[col_map_offd_RT[i]];
 
   recv_procs_RT = hypre_CTAlloc(int,num_recvs_RT);
   recv_vec_starts_RT = hypre_CTAlloc(int, num_recvs_RT+1);
 
   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i=0; i < num_recvs_A; i++)
        if (proc_mark[i])
        {
                recv_procs_RT[j] = recv_procs_A[i];
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
	MPI_Isend(&proc_mark[i],1,MPI_INT,recv_procs_A[i],0,comm,
		&requests[j++]);
   
   MPI_Waitall(num_requests,requests,status);

   hypre_TFree(proc_mark);
   
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
   send_map_starts_RT = hypre_CTAlloc(int, num_sends_RT+1);

   j = 0;
   send_map_starts_RT[0] = 0;
   for (i=0; i < num_sends_A; i++)
      if (change_array[i]) 
      {
	 send_procs_RT[j] = send_procs_A[i];
	 send_map_starts_RT[j+1] = send_map_starts_RT[j]+change_array[i];
	 j++;
      }

/*--------------------------------------------------------------------------
 * generate send_map_elmts
 *--------------------------------------------------------------------------*/

   send_map_elmts_RT = hypre_CTAlloc(int,send_map_starts_RT[num_sends_RT]);

   j = 0;
   for (i=0; i < num_sends_RT; i++)
   {
	vec_start = send_map_starts_RT[i];
	vec_len = send_map_starts_RT[i+1]-vec_start;
	MPI_Irecv(&send_map_elmts_RT[vec_start],vec_len,MPI_INT,
		send_procs_RT[i],0,comm,&requests[j++]);
   }

   for (i=0; i < num_recvs_RT; i++)
   {
	vec_start = recv_vec_starts_RT[i];
	vec_len = recv_vec_starts_RT[i+1] - vec_start;
	MPI_Isend(&col_map_offd_RT[vec_start],vec_len,MPI_INT, 
		recv_procs_RT[i],0,comm,&requests[j++]);
   }
   
   MPI_Waitall(j,requests,status);

   for (i=0; i < send_map_starts_RT[num_sends_RT]; i++)
	send_map_elmts_RT[i] -= first_col_diag; 
	
   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends_RT;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs_RT;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs_RT;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs_RT;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts_RT;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts_RT;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts_RT;

   hypre_TFree(status);
   hypre_TFree(requests);

   hypre_ParCSRBlockMatrixCommPkg(RT) = comm_pkg;
   hypre_TFree(change_array);

   return ierr;
}

