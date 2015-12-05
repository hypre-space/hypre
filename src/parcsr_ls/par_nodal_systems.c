/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.20 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"



/*==========================================================================*/
/*==========================================================================*/
/**
  Generates nodal norm matrix for use with nodal systems version

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param A [IN]
  coefficient matrix
  @param AN_ptr [OUT]
  nodal norm matrix
  
  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateNodalA(hypre_ParCSRMatrix    *A,
                            HYPRE_Int                    num_functions,
                            HYPRE_Int                   *dof_func,
                            HYPRE_Int                    option,
                            HYPRE_Int                    diag_option,     
                            hypre_ParCSRMatrix   **AN_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int 		      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int 		      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int 		       num_nonzeros_offd = 0;
   HYPRE_Int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *AN;
   hypre_CSRMatrix    *AN_diag;
   HYPRE_Int                *AN_diag_i;
   HYPRE_Int                *AN_diag_j;
   double             *AN_diag_data; 
   hypre_CSRMatrix    *AN_offd;
   HYPRE_Int                *AN_offd_i;
   HYPRE_Int                *AN_offd_j;
   double             *AN_offd_data; 
   HYPRE_Int		      *col_map_offd_AN;
   HYPRE_Int		      *new_col_map_offd;
   HYPRE_Int		      *row_starts_AN;
   HYPRE_Int		       AN_num_nonzeros_diag = 0;
   HYPRE_Int		       AN_num_nonzeros_offd = 0;
   HYPRE_Int		       num_cols_offd_AN;
   HYPRE_Int		       new_num_cols_offd;
                 
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int		       num_sends;
   HYPRE_Int		       num_recvs;
   HYPRE_Int		      *send_procs;
   HYPRE_Int		      *send_map_starts;
   HYPRE_Int		      *send_map_elmts;
   HYPRE_Int		      *new_send_map_elmts;
   HYPRE_Int		      *recv_procs;
   HYPRE_Int		      *recv_vec_starts;

   hypre_ParCSRCommPkg *comm_pkg_AN;
   HYPRE_Int		      *send_procs_AN;
   HYPRE_Int		      *send_map_starts_AN;
   HYPRE_Int		      *send_map_elmts_AN;
   HYPRE_Int		      *recv_procs_AN;
   HYPRE_Int		      *recv_vec_starts_AN;

   HYPRE_Int                 i, j, k, k_map;
                      
   HYPRE_Int                 ierr = 0;

   HYPRE_Int		       index, row;
   HYPRE_Int		       start_index;
   HYPRE_Int		       num_procs;
   HYPRE_Int		       node, cnt;
   HYPRE_Int		       mode;
   HYPRE_Int		       new_send_elmts_size;

   HYPRE_Int		       global_num_nodes;
   HYPRE_Int		       num_nodes;
   HYPRE_Int		       num_fun2;
   HYPRE_Int		      *map_to_node;
   HYPRE_Int		      *map_to_map;
   HYPRE_Int		      *counter;

   double sum;
   double *data;
   

   hypre_MPI_Comm_size(comm,&num_procs);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   mode = fabs(option);

   comm_pkg_AN = NULL;
   col_map_offd_AN = NULL;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_AN = hypre_CTAlloc(HYPRE_Int, 2);

   for (i=0; i < 2; i++)
   {
      row_starts_AN[i] = row_starts[i]/num_functions;
      if (row_starts_AN[i]*num_functions < row_starts[i])
      {
	  hypre_printf("nodes not properly aligned or incomplete info!\n");
	  return (87);
      }
   }
   
   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(A)/num_functions;


#else
   row_starts_AN = hypre_CTAlloc(HYPRE_Int, num_procs+1);

  for (i=0; i < num_procs+1; i++)
   {
      row_starts_AN[i] = row_starts[i]/num_functions;
      if (row_starts_AN[i]*num_functions < row_starts[i])
      {
	  hypre_printf("nodes not properly aligned or incomplete info!\n");
	  return (87);
      }
   }
   
   global_num_nodes = row_starts_AN[num_procs];

#endif

 
   num_nodes =  num_variables/num_functions;
   num_fun2 = num_functions*num_functions;

   map_to_node = hypre_CTAlloc(HYPRE_Int, num_variables);
   AN_diag_i = hypre_CTAlloc(HYPRE_Int, num_nodes+1);
   counter = hypre_CTAlloc(HYPRE_Int, num_nodes);
   for (i=0; i < num_variables; i++)
      map_to_node[i] = i/num_functions;
   for (i=0; i < num_nodes; i++)
      counter[i] = -1;

   AN_num_nonzeros_diag = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      AN_diag_i[i] = AN_num_nonzeros_diag;
      for (j=0; j < num_functions; j++)
      {
	 for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	 {
	    k_map = map_to_node[A_diag_j[k]];
	    if (counter[k_map] < i)
	    {
	       counter[k_map] = i;
	       AN_num_nonzeros_diag++;
	    }
	 }
	 row++;
      }
   }
   AN_diag_i[num_nodes] = AN_num_nonzeros_diag;

   AN_diag_j = hypre_CTAlloc(HYPRE_Int, AN_num_nonzeros_diag);	
   AN_diag_data = hypre_CTAlloc(double, AN_num_nonzeros_diag);	

   AN_diag = hypre_CSRMatrixCreate(num_nodes,num_nodes,AN_num_nonzeros_diag);
   hypre_CSRMatrixI(AN_diag) = AN_diag_i;
   hypre_CSRMatrixJ(AN_diag) = AN_diag_j;
   hypre_CSRMatrixData(AN_diag) = AN_diag_data;
       
   for (i=0; i < num_nodes; i++)
      counter[i] = -1;
   index = 0;
   start_index = 0;
   row = 0;

   switch (mode)
   {
      case 1:  /* frobenius norm */
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = A_diag_data[k]*A_diag_data[k];
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += 
				A_diag_data[k]*A_diag_data[k];
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
            AN_diag_data[i] = sqrt(AN_diag_data[i]);

      }
      break;
      
      case 2:  /* sum of abs. value of all elements in each block */
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = fabs(A_diag_data[k]);
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += fabs(A_diag_data[k]);
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
            AN_diag_data[i] /= num_fun2;
      }
      break;

      case 3:  /* largest element of each block (sets true value - not abs. value) */
      {

         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
      	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
      	       {
      	          k_map = map_to_node[A_diag_j[k]];
      	          if (counter[k_map] < start_index)
      	          {
      	             counter[k_map] = index;
      	             AN_diag_j[index] = k_map;
      	             AN_diag_data[index] = A_diag_data[k];
      	             index++;
      	          }
      	          else
      	          {
      	             if (fabs(A_diag_data[k]) > 
				fabs(AN_diag_data[counter[k_map]]))
      	                AN_diag_data[counter[k_map]] = A_diag_data[k];
      	          }
      	       }
      	       row++;
            }
            start_index = index;
         }
      }
      break;

      case 4:  /* inf. norm (row-sum)  */
      {

         data = hypre_CTAlloc(double, AN_num_nonzeros_diag*num_functions);

         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             data[index*num_functions + j] = fabs(A_diag_data[k]);
	             index++;
	          }
	          else
	          {
	             data[(counter[k_map])*num_functions + j] += fabs(A_diag_data[k]);
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
         {
            AN_diag_data[i]  = data[i*num_functions];
            
            for (j=1; j< num_functions; j++)
            {
               AN_diag_data[i]  = hypre_max( AN_diag_data[i],data[i*num_functions+j]);
            }
         }
         hypre_TFree(data);
      
      }
      break;

      case 6:  /* sum of all elements in each block */
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = (A_diag_data[k]);
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += (A_diag_data[k]);
	          }
	       }
	       row++;
            }
            start_index = index;
         }
      }
      break;

   }

   if (diag_option ==1 )
   {
      /* make the diag entry the negative of the sum of off-diag entries (DO MORE BELOW) */
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
         AN_diag_data[index] = - AN_diag_data[index];
      }
   }






   num_nonzeros_offd = A_offd_i[num_variables];
   AN_offd_i = hypre_CTAlloc(HYPRE_Int, num_nodes+1);

   num_cols_offd_AN = 0;

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
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_procs_AN = NULL;
      send_map_elmts_AN = NULL;
      if (num_sends) 
      {
         send_procs_AN = hypre_CTAlloc(HYPRE_Int,num_sends);
         send_map_elmts_AN = hypre_CTAlloc(HYPRE_Int,send_map_starts[num_sends]);
      }
      send_map_starts_AN = hypre_CTAlloc(HYPRE_Int,num_sends+1);
      recv_vec_starts_AN = hypre_CTAlloc(HYPRE_Int,num_recvs+1);
      recv_procs_AN = NULL;
      if (num_recvs) recv_procs_AN = hypre_CTAlloc(HYPRE_Int,num_recvs);
      for (i=0; i < num_sends; i++)
         send_procs_AN[i] = send_procs[i];
      for (i=0; i < num_recvs; i++)
         recv_procs_AN[i] = recv_procs[i];

      send_map_starts_AN[0] = 0;
      cnt = 0;
      for (i=0; i < num_sends; i++)
      {
	 k_map = send_map_starts[i];
	 if (send_map_starts[i+1]-k_map)
            send_map_elmts_AN[cnt++] = send_map_elmts[k_map]/num_functions;
         for (j=send_map_starts[i]+1; j < send_map_starts[i+1]; j++)
         {
            node = send_map_elmts[j]/num_functions;
            if (node > send_map_elmts_AN[cnt-1])
	       send_map_elmts_AN[cnt++] = node; 
         }
         send_map_starts_AN[i+1] = cnt;
      }
      hypre_ParCSRCommPkgSendProcs(comm_pkg_AN) = send_procs_AN;
      hypre_ParCSRCommPkgSendMapStarts(comm_pkg_AN) = send_map_starts_AN;
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg_AN) = send_map_elmts_AN;
      hypre_ParCSRCommPkgRecvProcs(comm_pkg_AN) = recv_procs_AN;
      hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_AN) = recv_vec_starts_AN;
   }

   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   if (num_cols_offd)
   {
      if (num_cols_offd > num_variables)
      {
         hypre_TFree(map_to_node);
         map_to_node = hypre_CTAlloc(HYPRE_Int,num_cols_offd);
      }

      num_cols_offd_AN = 1;
      map_to_node[0] = col_map_offd[0]/num_functions;
      for (i=1; i < num_cols_offd; i++)
      {
         map_to_node[i] = col_map_offd[i]/num_functions;
         if (map_to_node[i] > map_to_node[i-1]) num_cols_offd_AN++;
      }
      
      if (num_cols_offd_AN > num_nodes)
      {
         hypre_TFree(counter);
         counter = hypre_CTAlloc(HYPRE_Int,num_cols_offd_AN);
      }

      map_to_map = NULL;
      col_map_offd_AN = NULL;
      map_to_map = hypre_CTAlloc(HYPRE_Int, num_cols_offd);
      col_map_offd_AN = hypre_CTAlloc(HYPRE_Int,num_cols_offd_AN);
      col_map_offd_AN[0] = map_to_node[0];
      recv_vec_starts_AN[0] = 0;
      cnt = 1;
      for (i=0; i < num_recvs; i++)
      {
         for (j=recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
         {
            node = map_to_node[j];
	    if (node > col_map_offd_AN[cnt-1])
	    {
	       col_map_offd_AN[cnt++] = node; 
	    }
	    map_to_map[j] = cnt-1;
         }
         recv_vec_starts_AN[i+1] = cnt;
      }

      for (i=0; i < num_cols_offd_AN; i++)
         counter[i] = -1;

      AN_num_nonzeros_offd = 0;
      row = 0;
      for (i=0; i < num_nodes; i++)
      {
         AN_offd_i[i] = AN_num_nonzeros_offd;
         for (j=0; j < num_functions; j++)
         {
	    for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	    {
	       k_map = map_to_map[A_offd_j[k]];
	       if (counter[k_map] < i)
	       {
	          counter[k_map] = i;
	          AN_num_nonzeros_offd++;
	       }
	    }
	    row++;
         }
      }
      AN_offd_i[num_nodes] = AN_num_nonzeros_offd;
   }

       
   AN_offd = hypre_CSRMatrixCreate(num_nodes,num_cols_offd_AN,	
		AN_num_nonzeros_offd);
   hypre_CSRMatrixI(AN_offd) = AN_offd_i;
   if (AN_num_nonzeros_offd)
   {
      AN_offd_j = hypre_CTAlloc(HYPRE_Int, AN_num_nonzeros_offd);	
      AN_offd_data = hypre_CTAlloc(double, AN_num_nonzeros_offd);	
      hypre_CSRMatrixJ(AN_offd) = AN_offd_j;
      hypre_CSRMatrixData(AN_offd) = AN_offd_data;
   
      for (i=0; i < num_cols_offd_AN; i++)
         counter[i] = -1;
      index = 0;
      row = 0;
      AN_offd_i[0] = 0;
      start_index = 0;
      switch (mode)
      {
         case 1: /* frobenius norm */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	          {
	             k_map = map_to_map[A_offd_j[k]];
	             if (counter[k_map] < start_index)
	             {
	                counter[k_map] = index;
	                AN_offd_j[index] = k_map;
	                AN_offd_data[index] = A_offd_data[k]*A_offd_data[k];
	                index++;
	             }
	             else
	             {
	                AN_offd_data[counter[k_map]] += 
				A_offd_data[k]*A_offd_data[k];
	             }
	          }
	          row++;
               }
               start_index = index;
            }
            for (i=0; i < AN_num_nonzeros_offd; i++)
	       AN_offd_data[i] = sqrt(AN_offd_data[i]);
         }
         break;
      
         case 2:  /* sum of abs. value of all elements in block */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	          {
	             k_map = map_to_map[A_offd_j[k]];
	             if (counter[k_map] < start_index)
	             {
	                counter[k_map] = index;
	                AN_offd_j[index] = k_map;
	                AN_offd_data[index] = fabs(A_offd_data[k]);
	                index++;
	             }
	             else
	             {
	                AN_offd_data[counter[k_map]] += fabs(A_offd_data[k]);
	             }
	          }
	          row++;
               }
               start_index = index;
            }
            for (i=0; i < AN_num_nonzeros_offd; i++)
               AN_offd_data[i] /= num_fun2;
         }
         break;

         case 3: /* largest element in each block (not abs. value ) */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
      	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
      	          {
      	             k_map = map_to_map[A_offd_j[k]];
      	             if (counter[k_map] < start_index)
      	             {
      	                counter[k_map] = index;
      	                AN_offd_j[index] = k_map;
      	                AN_offd_data[index] = A_offd_data[k];
      	                index++;
      	             }
      	             else
      	             {
      	                if (fabs(A_offd_data[k]) > 
				fabs(AN_offd_data[counter[k_map]]))
      	                   AN_offd_data[counter[k_map]] = A_offd_data[k];
      	             }
      	          }
      	          row++;
               }
               start_index = index;
            }
         }
         break;
         
         case 4:  /* inf. norm (row-sum)  */
         {
            
            data = hypre_CTAlloc(double, AN_num_nonzeros_offd*num_functions);
            
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
                  for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
                  {
                     k_map = map_to_map[A_offd_j[k]];
                     if (counter[k_map] < start_index)
                     {
                        counter[k_map] = index;
                        AN_offd_j[index] = k_map;
                        data[index*num_functions + j] = fabs(A_offd_data[k]);
                        index++;
                     }
                     else
                     {
                        data[(counter[k_map])*num_functions + j] += fabs(A_offd_data[k]);
                     }
                  }
                  row++;
               }
               start_index = index;
            }
            for (i=0; i < AN_num_nonzeros_offd; i++)
            {
               AN_offd_data[i]  = data[i*num_functions];
               
               for (j=1; j< num_functions; j++)
               {
                  AN_offd_data[i]  = hypre_max( AN_offd_data[i],data[i*num_functions+j]);
               }
            }
            hypre_TFree(data);
            
         }
         break;
         
         case 6:  /* sum of value of all elements in block */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
                  for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
                  {
                     k_map = map_to_map[A_offd_j[k]];
                     if (counter[k_map] < start_index)
                     {
                        counter[k_map] = index;
                        AN_offd_j[index] = k_map;
                        AN_offd_data[index] = (A_offd_data[k]);
                        index++;
                     }
                     else
                     {
                        AN_offd_data[counter[k_map]] += (A_offd_data[k]);
                     }
                  }
                  row++;
               }
               start_index = index;
            }
            
         }
         break;
      }
   
      hypre_TFree(map_to_map);
   }

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

    
   AN = hypre_ParCSRMatrixCreate(comm, global_num_nodes, global_num_nodes,
		row_starts_AN, row_starts_AN, num_cols_offd_AN,
		AN_num_nonzeros_diag, AN_num_nonzeros_offd);

   /* we already created the diag and offd matrices - so we don't need the ones
      created above */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(AN));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(AN));
   hypre_ParCSRMatrixDiag(AN) = AN_diag;
   hypre_ParCSRMatrixOffd(AN) = AN_offd;


   hypre_ParCSRMatrixColMapOffd(AN) = col_map_offd_AN;
   hypre_ParCSRMatrixCommPkg(AN) = comm_pkg_AN;

   new_num_cols_offd = num_functions*num_cols_offd_AN;

   if (new_num_cols_offd > num_cols_offd)
   {
      new_col_map_offd = hypre_CTAlloc(HYPRE_Int, new_num_cols_offd);
      cnt = 0;
      for (i=0; i < num_cols_offd_AN; i++)
      {
	 for (j=0; j < num_functions; j++)
         {
 	    new_col_map_offd[cnt++] = num_functions*col_map_offd_AN[i]+j;
         }
      }
      cnt = 0;
      for (i=0; i < num_cols_offd; i++)
      {
         while (col_map_offd[i] >  new_col_map_offd[cnt])
            cnt++;
         col_map_offd[i] = cnt++;
      }
      for (i=0; i < num_recvs+1; i++)
      {
         recv_vec_starts[i] = num_functions*recv_vec_starts_AN[i];
      }

      for (i=0; i < num_nonzeros_offd; i++)
      {
         j = A_offd_j[i];
	 A_offd_j[i] = col_map_offd[j];
      }
      hypre_ParCSRMatrixColMapOffd(A) = new_col_map_offd;
      hypre_CSRMatrixNumCols(A_offd) = new_num_cols_offd;
      hypre_TFree(col_map_offd);
   }
 
   hypre_TFree(map_to_node);
   new_send_elmts_size = send_map_starts_AN[num_sends]*num_functions;

   if (new_send_elmts_size > send_map_starts[num_sends])
   {
      new_send_map_elmts = hypre_CTAlloc(HYPRE_Int,new_send_elmts_size);
      cnt = 0;
      send_map_starts[0] = 0;
      for (i=0; i < num_sends; i++)
      {
         send_map_starts[i+1] = send_map_starts_AN[i+1]*num_functions;
         for (j=send_map_starts_AN[i]; j < send_map_starts_AN[i+1]; j++)
	 {
            for (k=0; k < num_functions; k++)
	       new_send_map_elmts[cnt++] = send_map_elmts_AN[j]*num_functions+k;
	 }
      }
      hypre_TFree(send_map_elmts);
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = new_send_map_elmts;
   }
 
   *AN_ptr        = AN;

   hypre_TFree(counter);

   return (ierr);
}


/* This creates a scalar version of the CF_marker, dof_array and strength matrix (SN) */

HYPRE_Int
hypre_BoomerAMGCreateScalarCFS(hypre_ParCSRMatrix    *SN,
                       HYPRE_Int                   *CFN_marker,
                       HYPRE_Int                   *col_offd_SN_to_AN,
                       HYPRE_Int                    num_functions,
                       HYPRE_Int                    nodal,
                       HYPRE_Int                    data,
                       HYPRE_Int                  **dof_func_ptr,
                       HYPRE_Int                  **CF_marker_ptr,
                       HYPRE_Int                  **col_offd_S_to_A_ptr,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm	       comm = hypre_ParCSRMatrixComm(SN);
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int		      *S_diag_i;
   HYPRE_Int		      *S_diag_j;
   double	      *S_diag_data;
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int		      *S_offd_i;
   HYPRE_Int		      *S_offd_j;
   double	      *S_offd_data;
   HYPRE_Int		      *row_starts_S;
   HYPRE_Int		      *col_starts_S;
   HYPRE_Int		      *row_starts_SN = hypre_ParCSRMatrixRowStarts(SN);
   HYPRE_Int		      *col_starts_SN = hypre_ParCSRMatrixColStarts(SN);
   hypre_CSRMatrix    *SN_diag = hypre_ParCSRMatrixDiag(SN);
   HYPRE_Int		      *SN_diag_i = hypre_CSRMatrixI(SN_diag);
   HYPRE_Int		      *SN_diag_j = hypre_CSRMatrixJ(SN_diag);
   double	      *SN_diag_data;
   hypre_CSRMatrix    *SN_offd = hypre_ParCSRMatrixOffd(SN);
   HYPRE_Int		      *SN_offd_i = hypre_CSRMatrixI(SN_offd);
   HYPRE_Int		      *SN_offd_j = hypre_CSRMatrixJ(SN_offd);
   double	      *SN_offd_data;
   HYPRE_Int		      *CF_marker;
   HYPRE_Int		      *col_map_offd_SN = hypre_ParCSRMatrixColMapOffd(SN);
   HYPRE_Int		      *col_map_offd_S;
   HYPRE_Int		      *dof_func;
   HYPRE_Int		       num_nodes = hypre_CSRMatrixNumRows(SN_diag);
   HYPRE_Int		       num_variables;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(SN);
   HYPRE_Int		       num_sends;
   HYPRE_Int		       num_recvs;
   HYPRE_Int		      *send_procs;
   HYPRE_Int		      *send_map_starts;
   HYPRE_Int		      *send_map_elmts;
   HYPRE_Int		      *recv_procs;
   HYPRE_Int		      *recv_vec_starts;
   hypre_ParCSRCommPkg *comm_pkg_S;
   HYPRE_Int		      *send_procs_S;
   HYPRE_Int		      *send_map_starts_S;
   HYPRE_Int		      *send_map_elmts_S;
   HYPRE_Int		      *recv_procs_S;
   HYPRE_Int		      *recv_vec_starts_S;
   HYPRE_Int		      *col_offd_S_to_A = NULL;
   
   HYPRE_Int		       num_coarse_nodes;
   HYPRE_Int		       i,j,k,k1,jj,cnt;
   HYPRE_Int		       row, start, end;
   HYPRE_Int		       num_procs;
   HYPRE_Int		       num_cols_offd_SN = hypre_CSRMatrixNumCols(SN_offd);
   HYPRE_Int		       num_cols_offd_S;
   HYPRE_Int		       SN_num_nonzeros_diag;
   HYPRE_Int		       SN_num_nonzeros_offd;
   HYPRE_Int		       S_num_nonzeros_diag;
   HYPRE_Int		       S_num_nonzeros_offd;
   HYPRE_Int		       global_num_vars;
   HYPRE_Int		       global_num_cols;
   HYPRE_Int		       global_num_nodes;
   HYPRE_Int		       ierr = 0;
 
   hypre_MPI_Comm_size(comm, &num_procs);
 
   num_variables = num_functions*num_nodes;
   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);

   if (nodal < 0)
   {
      cnt = 0;
      num_coarse_nodes = 0;
      for (i=0; i < num_nodes; i++)
      {
	 if (CFN_marker[i] == 1) num_coarse_nodes++;
         for (j=0; j < num_functions; j++)
	    CF_marker[cnt++] = CFN_marker[i];
      }

      dof_func = hypre_CTAlloc(HYPRE_Int,num_coarse_nodes*num_functions);
      cnt = 0;
      for (i=0; i < num_nodes; i++)
      {
	 if (CFN_marker[i] == 1)
	 {
	    for (k=0; k < num_functions; k++)
	       dof_func[cnt++] = k;
	 }
      }
      *dof_func_ptr = dof_func;
   }
   else
   {
      cnt = 0;
      for (i=0; i < num_nodes; i++)
         for (j=0; j < num_functions; j++)
	    CF_marker[cnt++] = CFN_marker[i];
   }

   *CF_marker_ptr = CF_marker;


#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_S = hypre_CTAlloc(HYPRE_Int,2);
   for (i=0; i < 2; i++)
      row_starts_S[i] = num_functions*row_starts_SN[i];

   if (row_starts_SN != col_starts_SN)
   {
      col_starts_S = hypre_CTAlloc(HYPRE_Int,2);
      for (i=0; i < 2; i++)
         col_starts_S[i] = num_functions*col_starts_SN[i];
   }
   else
   {
      col_starts_S = row_starts_S;
   }
#else
   row_starts_S = hypre_CTAlloc(HYPRE_Int,num_procs+1);
   for (i=0; i < num_procs+1; i++)
      row_starts_S[i] = num_functions*row_starts_SN[i];

   if (row_starts_SN != col_starts_SN)
   {
      col_starts_S = hypre_CTAlloc(HYPRE_Int,num_procs+1);
      for (i=0; i < num_procs+1; i++)
         col_starts_S[i] = num_functions*col_starts_SN[i];
   }
   else
   {
      col_starts_S = row_starts_S;
   }
#endif


   SN_num_nonzeros_diag = SN_diag_i[num_nodes];
   SN_num_nonzeros_offd = SN_offd_i[num_nodes];
 
   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(SN);
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(SN)*num_functions;
 
   global_num_vars = global_num_nodes*num_functions;
   S_num_nonzeros_diag = num_functions*SN_num_nonzeros_diag;
   S_num_nonzeros_offd = num_functions*SN_num_nonzeros_offd;
   num_cols_offd_S = num_functions*num_cols_offd_SN;
   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_cols,
		row_starts_S, col_starts_S, num_cols_offd_S,
		S_num_nonzeros_diag, S_num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);
   S_diag_i = hypre_CTAlloc(HYPRE_Int, num_variables+1);
   S_offd_i = hypre_CTAlloc(HYPRE_Int, num_variables+1);
   S_diag_j = hypre_CTAlloc(HYPRE_Int, S_num_nonzeros_diag);
   hypre_CSRMatrixI(S_diag) = S_diag_i;
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   if (data) 
   {
      SN_diag_data = hypre_CSRMatrixData(SN_diag);
      S_diag_data = hypre_CTAlloc(double, S_num_nonzeros_diag);
      hypre_CSRMatrixData(S_diag) = S_diag_data;
      if (num_cols_offd_S)
      {
         SN_offd_data = hypre_CSRMatrixData(SN_offd);
         S_offd_data = hypre_CTAlloc(double, S_num_nonzeros_offd);
         hypre_CSRMatrixData(S_offd) = S_offd_data;
      }

   }
   hypre_CSRMatrixI(S_offd) = S_offd_i;

   if (comm_pkg)
   {
      comm_pkg_S = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(comm_pkg_S) = comm;
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      hypre_ParCSRCommPkgNumSends(comm_pkg_S) = num_sends;
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      hypre_ParCSRCommPkgNumRecvs(comm_pkg_S) = num_recvs;
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_procs_S = NULL;
      send_map_elmts_S = NULL;
      if (num_sends) 
      {
         send_procs_S = hypre_CTAlloc(HYPRE_Int,num_sends);
         send_map_elmts_S = hypre_CTAlloc(HYPRE_Int,
		num_functions*send_map_starts[num_sends]);
      }
      send_map_starts_S = hypre_CTAlloc(HYPRE_Int,num_sends+1);
      recv_vec_starts_S = hypre_CTAlloc(HYPRE_Int,num_recvs+1);
      recv_procs_S = NULL;
      if (num_recvs) recv_procs_S = hypre_CTAlloc(HYPRE_Int,num_recvs);
      send_map_starts_S[0] = 0;
      for (i=0; i < num_sends; i++)
      {
         send_procs_S[i] = send_procs[i];
         send_map_starts_S[i+1] = num_functions*send_map_starts[i+1];
      }
      recv_vec_starts_S[0] = 0;
      for (i=0; i < num_recvs; i++)
      {
         recv_procs_S[i] = recv_procs[i];
         recv_vec_starts_S[i+1] = num_functions*recv_vec_starts[i+1];
      }

      cnt = 0;
      for (i=0; i < send_map_starts[num_sends]; i++)
      {
	 k1 = num_functions*send_map_elmts[i];
         for (j=0; j < num_functions; j++)
         {
	    send_map_elmts_S[cnt++] = k1+j;
         }
      }
      hypre_ParCSRCommPkgSendProcs(comm_pkg_S) = send_procs_S;
      hypre_ParCSRCommPkgSendMapStarts(comm_pkg_S) = send_map_starts_S;
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg_S) = send_map_elmts_S;
      hypre_ParCSRCommPkgRecvProcs(comm_pkg_S) = recv_procs_S;
      hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_S) = recv_vec_starts_S;
      hypre_ParCSRMatrixCommPkg(S) = comm_pkg_S;
   }

   if (num_cols_offd_S)
   {
      S_offd_j = hypre_CTAlloc(HYPRE_Int, S_num_nonzeros_offd);
      hypre_CSRMatrixJ(S_offd) = S_offd_j;

      col_map_offd_S = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S);

      cnt = 0;
      for (i=0; i < num_cols_offd_SN; i++)
      {
         k1 = col_map_offd_SN[i]*num_functions;
         for (j=0; j < num_functions; j++)
            col_map_offd_S[cnt++] = k1+j;
      }
      hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
   }
   
   if (col_offd_SN_to_AN)
   {
      col_offd_S_to_A = hypre_CTAlloc(HYPRE_Int, num_cols_offd_S);

      cnt = 0;
      for (i=0; i < num_cols_offd_SN; i++)
      {
         k1 = col_offd_SN_to_AN[i]*num_functions;
         for (j=0; j < num_functions; j++)
            col_offd_S_to_A[cnt++] = k1+j;
      }
      *col_offd_S_to_A_ptr = col_offd_S_to_A;
   }
   


   cnt = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      row++;
      start = cnt;
      for (j=SN_diag_i[i]; j < SN_diag_i[i+1]; j++)
      {
         jj = SN_diag_j[j];
	 if (data) S_diag_data[cnt] = SN_diag_data[j];
	 S_diag_j[cnt++] = jj*num_functions;
      }
      end = cnt;
      S_diag_i[row] = cnt;
      for (k1=1; k1 < num_functions; k1++)
      {
         row++;
	 for (k=start; k < end; k++)
	 {
	    if (data) S_diag_data[cnt] = S_diag_data[k];
	    S_diag_j[cnt++] = S_diag_j[k]+k1;
	 }
         S_diag_i[row] = cnt;
      }
   } 

   cnt = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      row++;
      start = cnt;
      for (j=SN_offd_i[i]; j < SN_offd_i[i+1]; j++)
      {
         jj = SN_offd_j[j];
	 if (data) S_offd_data[cnt] = SN_offd_data[j];
	 S_offd_j[cnt++] = jj*num_functions;
      }
      end = cnt;
      S_offd_i[row] = cnt;
      for (k1=1; k1 < num_functions; k1++)
      {
         row++;
	 for (k=start; k < end; k++)
	 {
	    if (data) S_offd_data[cnt] = S_offd_data[k];
	    S_offd_j[cnt++] = S_offd_j[k]+k1;
	 }
         S_offd_i[row] = cnt;
      }
   } 

   *S_ptr = S; 

   return (ierr);
}


/* This function just finds the scalaer CF_marker and dof_func */

HYPRE_Int
hypre_BoomerAMGCreateScalarCF(HYPRE_Int                   *CFN_marker,
                              HYPRE_Int                    num_functions,
                              HYPRE_Int                    num_nodes,
                              HYPRE_Int                  **dof_func_ptr,
                              HYPRE_Int                  **CF_marker_ptr)

{
   HYPRE_Int		      *CF_marker;
   HYPRE_Int		      *dof_func;
   HYPRE_Int		       num_variables;
   HYPRE_Int		       num_coarse_nodes;
   HYPRE_Int		       i,j,k,cnt;
   HYPRE_Int		       ierr = 0;
 
 
   num_variables = num_functions*num_nodes;
   CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);

   cnt = 0;
   num_coarse_nodes = 0;
   for (i=0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1) num_coarse_nodes++;
      for (j=0; j < num_functions; j++)
         CF_marker[cnt++] = CFN_marker[i];
   }

   
   dof_func = hypre_CTAlloc(HYPRE_Int,num_coarse_nodes*num_functions);
   cnt = 0;
   for (i=0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1)
      {
         for (k=0; k < num_functions; k++)
            dof_func[cnt++] = k;
      }
   }
   

   *dof_func_ptr = dof_func;
   *CF_marker_ptr = CF_marker;


   return (ierr);
}
