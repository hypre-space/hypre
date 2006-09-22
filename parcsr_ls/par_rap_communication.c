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

int
hypre_GetCommPkgRTFromCommPkgA( hypre_ParCSRMatrix *RT,
			       	hypre_ParCSRMatrix *A,
			       	int *fine_to_coarse_offd)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(RT);
   hypre_ParCSRCommPkg *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);
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

   int *col_map_offd_RT = hypre_ParCSRMatrixColMapOffd(RT);
   int num_cols_offd_RT = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd(RT));
   int first_col_diag = hypre_ParCSRMatrixFirstColDiag(RT);

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

   hypre_ParCSRMatrixCommPkg(RT) = comm_pkg;
   hypre_TFree(change_array);

   return ierr;
}

int
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
   hypre_ParCSRCommPkg *comm_pkg;
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
	
   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts;

   hypre_TFree(status);
   hypre_TFree(requests);

   hypre_ParCSRMatrixCommPkg(A) = comm_pkg;
   return 0;
}
