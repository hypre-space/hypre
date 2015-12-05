/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.9 $
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
   6 = sum of all elements
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGBlockCreateNodalA(hypre_ParCSRBlockMatrix    *A,
                                 HYPRE_Int                    option, HYPRE_Int diag_option,
                                 hypre_ParCSRMatrix   **AN_ptr)
{
   MPI_Comm 	            comm         = hypre_ParCSRBlockMatrixComm(A);
   hypre_CSRBlockMatrix    *A_diag       = hypre_ParCSRBlockMatrixDiag(A);
   HYPRE_Int                     *A_diag_i     = hypre_CSRBlockMatrixI(A_diag);
   double                  *A_diag_data  = hypre_CSRBlockMatrixData(A_diag);

   HYPRE_Int                  block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   HYPRE_Int                  bnnz = block_size*block_size;

   hypre_CSRBlockMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                     *A_offd_i        = hypre_CSRBlockMatrixI(A_offd);
   double                  *A_offd_data     = hypre_CSRBlockMatrixData(A_offd);
   HYPRE_Int                     *A_diag_j        = hypre_CSRBlockMatrixJ(A_diag);
   HYPRE_Int                     *A_offd_j        = hypre_CSRBlockMatrixJ(A_offd);

   HYPRE_Int 		      *row_starts      = hypre_ParCSRBlockMatrixRowStarts(A);
   HYPRE_Int 		      *col_map_offd    = hypre_ParCSRBlockMatrixColMapOffd(A);
   HYPRE_Int 		       num_nonzeros_diag;
   HYPRE_Int 		       num_nonzeros_offd = 0;
   HYPRE_Int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *AN;
   hypre_CSRMatrix    *AN_diag;
   HYPRE_Int                *AN_diag_i;
   HYPRE_Int                *AN_diag_j=NULL;
   double             *AN_diag_data = NULL; 
   hypre_CSRMatrix    *AN_offd;
   HYPRE_Int                *AN_offd_i;
   HYPRE_Int                *AN_offd_j = NULL;
   double             *AN_offd_data = NULL; 
   HYPRE_Int		      *col_map_offd_AN = NULL;
   HYPRE_Int		      *row_starts_AN;

                 
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   HYPRE_Int		       num_sends;
   HYPRE_Int		       num_recvs;
   HYPRE_Int		      *send_procs;
   HYPRE_Int		      *send_map_starts;
   HYPRE_Int		      *send_map_elmts;
   HYPRE_Int		      *recv_procs;
   HYPRE_Int		      *recv_vec_starts;

   hypre_ParCSRCommPkg *comm_pkg_AN = NULL;
   HYPRE_Int		      *send_procs_AN = NULL;
   HYPRE_Int		      *send_map_starts_AN = NULL;
   HYPRE_Int		      *send_map_elmts_AN = NULL;
   HYPRE_Int		      *recv_procs_AN = NULL;
   HYPRE_Int		      *recv_vec_starts_AN = NULL;

   HYPRE_Int                 i;
                      
   HYPRE_Int                 ierr = 0;

   HYPRE_Int		       num_procs;
   HYPRE_Int		       cnt;
   HYPRE_Int		       norm_type;

   HYPRE_Int		       global_num_nodes;
   HYPRE_Int		       num_nodes;

   HYPRE_Int                 index, k;
   
   double             tmp;
   double             sum;
   


   hypre_MPI_Comm_size(comm,&num_procs);

   if (!comm_pkg)
   {
      hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   }

   norm_type = fabs(option);


/* Set up the new matrix AN */


#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_AN = hypre_CTAlloc(HYPRE_Int, 2);
   for (i=0; i < 2; i++)
   {
      row_starts_AN[i] = row_starts[i];
   }
#else
   row_starts_AN = hypre_CTAlloc(HYPRE_Int, num_procs+1);
  for (i=0; i < num_procs+1; i++)
   {
      row_starts_AN[i] = row_starts[i];
   }
#endif

   global_num_nodes = hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_nodes = hypre_CSRBlockMatrixNumRows(A_diag);

   /* the diag part */

   num_nonzeros_diag = A_diag_i[num_nodes];
   AN_diag_i = hypre_CTAlloc(HYPRE_Int, num_nodes+1);

   for (i=0; i <= num_nodes; i++)
   {
      AN_diag_i[i] = A_diag_i[i];
   }

   AN_diag_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag);	
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
   

   if (diag_option ==1 )
   {
      /* make the diag entry the negative of the sum of off-diag entries (NEED to get more below!)*/
      /* the diagonal is the first element listed in each row - */
      for (i=0; i < num_nodes; i++)
      {
         index = AN_diag_i[i]; 
         sum = 0.0;
         for (k = AN_diag_i[i]+1; k < AN_diag_i[i+1]; k++)
         {
            sum += AN_diag_data[k];
            
         }

         AN_diag_data[index] = -sum;
      }
      
   }
   else if (diag_option == 2)
   {
      
      /*  make all diagonal entries negative */
      /* the diagonal is the first element listed in each row - */
      
      for (i=0; i < num_nodes; i++)
      {
         index = AN_diag_i[i];
         AN_diag_data[index] = -AN_diag_data[index];
      }
   }




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
         send_procs_AN = hypre_CTAlloc(HYPRE_Int, num_sends);
         send_map_elmts_AN = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends]);
      }
      send_map_starts_AN = hypre_CTAlloc(HYPRE_Int, num_sends+1);
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
      recv_vec_starts_AN = hypre_CTAlloc(HYPRE_Int, num_recvs+1);
      if (num_recvs) recv_procs_AN = hypre_CTAlloc(HYPRE_Int, num_recvs);

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
   col_map_offd_AN = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
   {
      col_map_offd_AN[i] = col_map_offd[i];
   }

   num_nonzeros_offd = A_offd_i[num_nodes];
   AN_offd_i = hypre_CTAlloc(HYPRE_Int, num_nodes+1);
   for (i=0; i <= num_nodes; i++)
   {
      AN_offd_i[i] = A_offd_i[i];
   }
      
   AN_offd_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros_offd);	
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
   

   if (diag_option ==1 )
   {
      /* make the diag entry the negative of the sum of off-diag entries (here we are adding the 
         off_diag contribution)*/
      /* the diagonal is the first element listed in each row of AN_diag_data - */
       for (i=0; i < num_nodes; i++)
      {
         sum = 0.0;
         for (k = AN_offd_i[i]; k < AN_offd_i[i+1]; k++)
         {
            sum += AN_offd_data[k];
            
         }
         index = AN_diag_i[i];/* location of diag entry in data */ 
         AN_diag_data[index] -= sum; /* subtract from current value */
      }
      
   }




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
