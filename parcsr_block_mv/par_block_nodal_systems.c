/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "headers.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockCreateNodalA

   This is the block version of creating a nodal norm matrix.

   option: determine which type of "norm" (or other measurement) is used.

   1 = frobenius
   2 = sum of abs. value of all elements
   3 = largest element (positive or negative)
   4 = 1-norm
   5 = inf - norm
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBlockCreateNodalA(hypre_ParCSRBlockMatrix    *A,
                       int                    option,
                       hypre_ParCSRMatrix   **AN_ptr)
{
   MPI_Comm 	            comm         = hypre_ParCSRBlockMatrixComm(A);
   hypre_CSRBlockMatrix    *A_diag       = hypre_ParCSRBlockMatrixDiag(A);
   int                     *A_diag_i     = hypre_CSRBlockMatrixI(A_diag);
   double                  *A_diag_data  = hypre_CSRBlockMatrixData(A_diag);

   int                  block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   int                  bnnz = block_size*block_size;

   hypre_CSRBlockMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                     *A_offd_i        = hypre_CSRBlockMatrixI(A_offd);
   double                  *A_offd_data     = hypre_CSRBlockMatrixData(A_offd);
   int                     *A_diag_j        = hypre_CSRBlockMatrixJ(A_diag);
   int                     *A_offd_j        = hypre_CSRBlockMatrixJ(A_offd);

   int 		      *row_starts      = hypre_ParCSRBlockMatrixRowStarts(A);
   int 		      *col_map_offd    = hypre_ParCSRBlockMatrixColMapOffd(A);
   int 		       num_nonzeros_diag;
   int 		       num_nonzeros_offd = 0;
   int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *AN;
   hypre_CSRMatrix    *AN_diag;
   int                *AN_diag_i;
   int                *AN_diag_j=NULL;
   double             *AN_diag_data = NULL; 
   hypre_CSRMatrix    *AN_offd;
   int                *AN_offd_i;
   int                *AN_offd_j = NULL;
   double             *AN_offd_data = NULL; 
   int		      *col_map_offd_AN = NULL;
   int		      *row_starts_AN;

                 
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   int		       num_sends;
   int		       num_recvs;
   int		      *send_procs;
   int		      *send_map_starts;
   int		      *send_map_elmts;
   int		      *recv_procs;
   int		      *recv_vec_starts;

   hypre_ParCSRCommPkg *comm_pkg_AN = NULL;
   int		      *send_procs_AN = NULL;
   int		      *send_map_starts_AN = NULL;
   int		      *send_map_elmts_AN = NULL;
   int		      *recv_procs_AN = NULL;
   int		      *recv_vec_starts_AN = NULL;

   int                 i;
                      
   int                 ierr = 0;

   int		       num_procs;
   int		       cnt;
   int		       norm_type;

   int		       global_num_nodes;
   int		       num_nodes;

   double             tmp;
   


   MPI_Comm_size(comm,&num_procs);

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_BlockNewCommPkgCreate(A);
#else
      hypre_BlockMatvecCommPkgCreate(A);
#endif
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   norm_type = fabs(option);


/* Set up the new matrix AN */


#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_AN = hypre_CTAlloc(int, 2);
   for (i=0; i < 2; i++)
   {
      row_starts_AN[i] = row_starts[i];
   }
#else
   row_starts_AN = hypre_CTAlloc(int, num_procs+1);
  for (i=0; i < num_procs+1; i++)
   {
      row_starts_AN[i] = row_starts[i];
   }
#endif

   global_num_nodes = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_nodes = hypre_CSRBlockMatrixNumRows(A_diag);

   /* the diag part */

   num_nonzeros_diag = A_diag_i[num_nodes];
   AN_diag_i = hypre_CTAlloc(int, num_nodes+1);

   for (i=0; i <= num_nodes; i++)
   {
      AN_diag_i[i] = A_diag_i[i];
   }

   AN_diag_j = hypre_CTAlloc(int, num_nonzeros_diag);	
   AN_diag_data = hypre_CTAlloc(double, num_nonzeros_diag);	


   AN_diag = hypre_CSRMatrixCreate(num_nodes, num_nodes, num_nonzeros_diag);
   hypre_CSRMatrixI(AN_diag) = AN_diag_i;
   hypre_CSRMatrixJ(AN_diag) = AN_diag_j;
   hypre_CSRMatrixData(AN_diag) = AN_diag_data;

   for (i=0; i< num_nonzeros_diag; i++)
   {
      AN_diag_j[i]  = A_diag_j[i];
      hypre_CSRBlockMatrixBlockNorm(norm_type, &A_diag_data[i*bnnz], 
                                    &tmp, block_size);
      AN_diag_data[i] = tmp;
   }
   
#if 0
/* to compare with serial - make diag entries negative*/

   for (i=0; i < num_nodes; i++)
   {
      AN_diag_data[AN_diag_i[i]] = - AN_diag_data[AN_diag_i[i]];
   }
   
#endif

   /* copy the commpkg */
   if (comm_pkg)
   {
      comm_pkg_AN = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(comm_pkg_AN) = comm;

      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      hypre_ParCSRCommPkgNumSends(comm_pkg_AN) = num_sends;

      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      hypre_ParCSRCommPkgNumRecvs(comm_pkg_AN) = num_recvs;

      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      if (num_sends) 
      {
         send_procs_AN = hypre_CTAlloc(int, num_sends);
         send_map_elmts_AN = hypre_CTAlloc(int, send_map_starts[num_sends]);
      }
      send_map_starts_AN = hypre_CTAlloc(int, num_sends+1);
      send_map_starts_AN[0] = 0;
      for (i=0; i < num_sends; i++)
      {
         send_procs_AN[i] = send_procs[i];
         send_map_starts_AN[i+1] = send_map_starts[i+1];
      }
      cnt = send_map_starts_AN[num_sends];
      for (i=0; i< cnt; i++)
      {
         send_map_elmts_AN[i] = send_map_elmts[i];
      }
      hypre_ParCSRCommPkgSendProcs(comm_pkg_AN) = send_procs_AN;
      hypre_ParCSRCommPkgSendMapStarts(comm_pkg_AN) = send_map_starts_AN;
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg_AN) = send_map_elmts_AN;

      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      recv_vec_starts_AN = hypre_CTAlloc(int, num_recvs+1);
      if (num_recvs) recv_procs_AN = hypre_CTAlloc(int, num_recvs);

      recv_vec_starts_AN[0] = recv_vec_starts[0];
      for (i=0; i < num_recvs; i++)
      {
         recv_procs_AN[i] = recv_procs[i];
         recv_vec_starts_AN[i+1] = recv_vec_starts[i+1];
         
      }
      hypre_ParCSRCommPkgRecvProcs(comm_pkg_AN) = recv_procs_AN;
      hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_AN) = recv_vec_starts_AN;

   }

 /* the off-diag part */

  
   num_cols_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   col_map_offd_AN = hypre_CTAlloc(int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
   {
      col_map_offd_AN[i] = col_map_offd[i];
   }

   num_nonzeros_offd = A_offd_i[num_nodes];
   AN_offd_i = hypre_CTAlloc(int, num_nodes+1);
   for (i=0; i <= num_nodes; i++)
   {
      AN_offd_i[i] = A_offd_i[i];
   }
      
   AN_offd_j = hypre_CTAlloc(int, num_nonzeros_offd);	
   AN_offd_data = hypre_CTAlloc(double, num_nonzeros_offd);

   for (i=0; i< num_nonzeros_offd; i++)
   {
      AN_offd_j[i]  = A_offd_j[i];
      hypre_CSRBlockMatrixBlockNorm(norm_type, &A_offd_data[i*bnnz], 
                                    &tmp, block_size);
      AN_offd_data[i] = tmp;
   }
   
   AN_offd = hypre_CSRMatrixCreate(num_nodes, num_cols_offd, num_nonzeros_offd);
  
   hypre_CSRMatrixI(AN_offd) = AN_offd_i;
   hypre_CSRMatrixJ(AN_offd) = AN_offd_j;
   hypre_CSRMatrixData(AN_offd) = AN_offd_data;
   
   /* now create AN */   
    
   AN = hypre_ParCSRMatrixCreate(comm, global_num_nodes, global_num_nodes,
		row_starts_AN, row_starts_AN, num_cols_offd,
		num_nonzeros_diag, num_nonzeros_offd);

   /* we already created the diag and offd matrices - so we don't need the ones
      created above */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(AN));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(AN));
   hypre_ParCSRMatrixDiag(AN) = AN_diag;
   hypre_ParCSRMatrixOffd(AN) = AN_offd;


   hypre_ParCSRMatrixColMapOffd(AN) = col_map_offd_AN;
   hypre_ParCSRMatrixCommPkg(AN) = comm_pkg_AN;

   *AN_ptr        = AN;

    return (ierr);
}
