/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "_hypre_parcsr_ls.h"



/*==========================================================================*/
/*==========================================================================*/
/**
  Generates nodal norm matrix for use with nodal systems version

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param A [IN]
  coefficient matrix
  @param AN_ptr [OUT]
  nodal norm matrix

  TODO: RL GPU version
  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateNodalA(hypre_ParCSRMatrix    *A,
                            HYPRE_Int              num_functions,
                            HYPRE_Int             *dof_func,
                            HYPRE_Int              option,
                            HYPRE_Int              diag_option,
                            hypre_ParCSRMatrix   **AN_ptr)
{
   HYPRE_UNUSED_VAR(dof_func);

   MPI_Comm            comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Real         *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Real         *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int          *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt       *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *AN;
   hypre_CSRMatrix    *AN_diag;
   HYPRE_Int          *AN_diag_i;
   HYPRE_Int          *AN_diag_j;
   HYPRE_Real         *AN_diag_data;
   hypre_CSRMatrix    *AN_offd;
   HYPRE_Int          *AN_offd_i;
   HYPRE_Int          *AN_offd_j;
   HYPRE_Real         *AN_offd_data = NULL;
   HYPRE_BigInt       *col_map_offd_AN;
   HYPRE_BigInt       *new_col_map_offd;
   HYPRE_BigInt        row_starts_AN[2];
   HYPRE_Int           AN_num_nonzeros_diag = 0;
   HYPRE_Int           AN_num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd_AN;
   HYPRE_Int           new_num_cols_offd;

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int            num_sends = 0;
   HYPRE_Int            num_recvs = 0;
   HYPRE_Int           *send_procs;
   HYPRE_Int           *send_map_starts = NULL;
   HYPRE_Int           *send_map_elmts  = NULL;
   HYPRE_Int           *new_send_map_elmts;
   HYPRE_Int           *recv_procs;
   HYPRE_Int           *recv_vec_starts = NULL;

   hypre_ParCSRCommPkg *comm_pkg_AN;
   HYPRE_Int           *send_procs_AN;
   HYPRE_Int           *send_map_starts_AN = NULL;
   HYPRE_Int           *send_map_elmts_AN = NULL;
   HYPRE_Int           *recv_procs_AN;
   HYPRE_Int           *recv_vec_starts_AN = NULL;

   HYPRE_Int           i, j, k, k_map;

   HYPRE_Int           index, row;
   HYPRE_Int           start_index;
   HYPRE_Int           num_procs;
   HYPRE_Int           node, cnt;
   HYPRE_Int           mode;
   HYPRE_BigInt        big_node;
   HYPRE_Int           new_send_elmts_size;

   HYPRE_BigInt        global_num_nodes;
   HYPRE_Int           num_nodes;
   HYPRE_Int           num_fun2;
   HYPRE_BigInt       *big_map_to_node = NULL;
   HYPRE_Int          *map_to_node;
   HYPRE_Int          *map_to_map = NULL;
   HYPRE_Int          *counter;

   HYPRE_Real sum;
   HYPRE_Real *data;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_MPI_Comm_size(comm, &num_procs);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   mode = hypre_abs(option);

   comm_pkg_AN = NULL;
   col_map_offd_AN = NULL;

   for (i = 0; i < 2; i++)
   {
      row_starts_AN[i] = row_starts[i] / (HYPRE_BigInt)num_functions;
      if (row_starts_AN[i] * (HYPRE_BigInt)num_functions < row_starts[i])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "nodes not properly aligned or incomplete info!\n");
         return hypre_error_flag;
      }
   }

   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(A) / (HYPRE_BigInt)num_functions;

   num_nodes =  num_variables / num_functions;
   num_fun2 = num_functions * num_functions;

   map_to_node = hypre_CTAlloc(HYPRE_Int, num_variables, HYPRE_MEMORY_HOST);
   AN_diag_i = hypre_CTAlloc(HYPRE_Int, num_nodes + 1, memory_location);
   counter = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_variables; i++)
   {
      map_to_node[i] = i / num_functions;
   }
   for (i = 0; i < num_nodes; i++)
   {
      counter[i] = -1;
   }

   AN_num_nonzeros_diag = 0;
   row = 0;
   for (i = 0; i < num_nodes; i++)
   {
      AN_diag_i[i] = AN_num_nonzeros_diag;
      for (j = 0; j < num_functions; j++)
      {
         for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
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

   AN_diag_j = hypre_CTAlloc(HYPRE_Int, AN_num_nonzeros_diag, memory_location);
   AN_diag_data = hypre_CTAlloc(HYPRE_Real, AN_num_nonzeros_diag, memory_location);

   AN_diag = hypre_CSRMatrixCreate(num_nodes, num_nodes, AN_num_nonzeros_diag);
   hypre_CSRMatrixI(AN_diag) = AN_diag_i;
   hypre_CSRMatrixJ(AN_diag) = AN_diag_j;
   hypre_CSRMatrixData(AN_diag) = AN_diag_data;

   for (i = 0; i < num_nodes; i++)
   {
      counter[i] = -1;
   }
   index = 0;
   start_index = 0;
   row = 0;

   switch (mode)
   {
      case 1:  /* frobenius norm */
      {
         for (i = 0; i < num_nodes; i++)
         {
            for (j = 0; j < num_functions; j++)
            {
               for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
               {
                  k_map = map_to_node[A_diag_j[k]];
                  if (counter[k_map] < start_index)
                  {
                     counter[k_map] = index;
                     AN_diag_j[index] = k_map;
                     AN_diag_data[index] = A_diag_data[k] * A_diag_data[k];
                     index++;
                  }
                  else
                  {
                     AN_diag_data[counter[k_map]] +=
                        A_diag_data[k] * A_diag_data[k];
                  }
               }
               row++;
            }
            start_index = index;
         }
         for (i = 0; i < AN_num_nonzeros_diag; i++)
         {
            AN_diag_data[i] = hypre_sqrt(AN_diag_data[i]);
         }

      }
      break;

      case 2:  /* sum of abs. value of all elements in each block */
      {
         for (i = 0; i < num_nodes; i++)
         {
            for (j = 0; j < num_functions; j++)
            {
               for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
               {
                  k_map = map_to_node[A_diag_j[k]];
                  if (counter[k_map] < start_index)
                  {
                     counter[k_map] = index;
                     AN_diag_j[index] = k_map;
                     AN_diag_data[index] = hypre_abs(A_diag_data[k]);
                     index++;
                  }
                  else
                  {
                     AN_diag_data[counter[k_map]] += hypre_abs(A_diag_data[k]);
                  }
               }
               row++;
            }
            start_index = index;
         }
         for (i = 0; i < AN_num_nonzeros_diag; i++)
         {
            AN_diag_data[i] /= num_fun2;
         }
      }
      break;

      case 3:  /* largest element of each block (sets true value - not abs. value) */
      {

         for (i = 0; i < num_nodes; i++)
         {
            for (j = 0; j < num_functions; j++)
            {
               for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
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
                     if (hypre_abs(A_diag_data[k]) >
                         hypre_abs(AN_diag_data[counter[k_map]]))
                     {
                        AN_diag_data[counter[k_map]] = A_diag_data[k];
                     }
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

         data = hypre_CTAlloc(HYPRE_Real,  AN_num_nonzeros_diag * num_functions, HYPRE_MEMORY_HOST);

         for (i = 0; i < num_nodes; i++)
         {
            for (j = 0; j < num_functions; j++)
            {
               for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
               {
                  k_map = map_to_node[A_diag_j[k]];
                  if (counter[k_map] < start_index)
                  {
                     counter[k_map] = index;
                     AN_diag_j[index] = k_map;
                     data[index * num_functions + j] = hypre_abs(A_diag_data[k]);
                     index++;
                  }
                  else
                  {
                     data[(counter[k_map])*num_functions + j] += hypre_abs(A_diag_data[k]);
                  }
               }
               row++;
            }
            start_index = index;
         }
         for (i = 0; i < AN_num_nonzeros_diag; i++)
         {
            AN_diag_data[i]  = data[i * num_functions];

            for (j = 1; j < num_functions; j++)
            {
               AN_diag_data[i]  = hypre_max( AN_diag_data[i], data[i * num_functions + j]);
            }
         }
         hypre_TFree(data, HYPRE_MEMORY_HOST);

      }
      break;

      case 6:  /* sum of all elements in each block */
      {
         for (i = 0; i < num_nodes; i++)
         {
            for (j = 0; j < num_functions; j++)
            {
               for (k = A_diag_i[row]; k < A_diag_i[row + 1]; k++)
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

   if (diag_option == 1 )
   {
      /* make the diag entry the negative of the sum of off-diag entries (DO MORE BELOW) */
      for (i = 0; i < num_nodes; i++)
      {
         index = AN_diag_i[i];
         sum = 0.0;
         for (k = AN_diag_i[i] + 1; k < AN_diag_i[i + 1]; k++)
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

      for (i = 0; i < num_nodes; i++)
      {
         index = AN_diag_i[i];
         AN_diag_data[index] = - AN_diag_data[index];
      }
   }

   num_nonzeros_offd = A_offd_i[num_variables];
   AN_offd_i = hypre_CTAlloc(HYPRE_Int, num_nodes + 1, memory_location);

   num_cols_offd_AN = 0;

   if (comm_pkg)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

      send_procs_AN = NULL;
      send_map_elmts_AN = NULL;
      if (num_sends)
      {
         send_procs_AN = hypre_CTAlloc(HYPRE_Int, num_sends, HYPRE_MEMORY_HOST);
         send_map_elmts_AN = hypre_CTAlloc(HYPRE_Int, send_map_starts[num_sends], HYPRE_MEMORY_HOST);
      }
      send_map_starts_AN = hypre_CTAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
      recv_vec_starts_AN = hypre_CTAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);
      recv_procs_AN = NULL;
      if (num_recvs) { recv_procs_AN = hypre_CTAlloc(HYPRE_Int, num_recvs, HYPRE_MEMORY_HOST); }
      for (i = 0; i < num_sends; i++)
      {
         send_procs_AN[i] = send_procs[i];
      }
      for (i = 0; i < num_recvs; i++)
      {
         recv_procs_AN[i] = recv_procs[i];
      }

      send_map_starts_AN[0] = 0;
      cnt = 0;
      for (i = 0; i < num_sends; i++)
      {
         k_map = send_map_starts[i];
         if (send_map_starts[i + 1] - k_map)
         {
            send_map_elmts_AN[cnt++] = send_map_elmts[k_map] / num_functions;
         }
         for (j = send_map_starts[i] + 1; j < send_map_starts[i + 1]; j++)
         {
            node = send_map_elmts[j] / num_functions;
            if (node > send_map_elmts_AN[cnt - 1])
            {
               send_map_elmts_AN[cnt++] = node;
            }
         }
         send_map_starts_AN[i + 1] = cnt;
      }

      /* Create communication package */
      hypre_ParCSRCommPkgCreateAndFill(comm,
                                       num_recvs, recv_procs_AN, recv_vec_starts_AN,
                                       num_sends, send_procs_AN, send_map_starts_AN,
                                       send_map_elmts_AN,
                                       &comm_pkg_AN);
   }
   hypre_TFree(map_to_node, HYPRE_MEMORY_HOST);

   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   if (num_cols_offd)
   {
      big_map_to_node = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

      num_cols_offd_AN = 1;
      big_map_to_node[0] = col_map_offd[0] / (HYPRE_BigInt)num_functions;
      for (i = 1; i < num_cols_offd; i++)
      {
         big_map_to_node[i] = col_map_offd[i] / (HYPRE_BigInt)num_functions;
         if (big_map_to_node[i] > big_map_to_node[i - 1]) { num_cols_offd_AN++; }
      }

      if (num_cols_offd_AN > num_nodes)
      {
         hypre_TFree(counter, HYPRE_MEMORY_HOST);
         counter = hypre_CTAlloc(HYPRE_Int, num_cols_offd_AN, HYPRE_MEMORY_HOST);
      }

      map_to_map = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      col_map_offd_AN = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_AN, HYPRE_MEMORY_HOST);
      col_map_offd_AN[0] = big_map_to_node[0];
      recv_vec_starts_AN[0] = 0;
      cnt = 1;
      for (i = 0; i < num_recvs; i++)
      {
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
         {
            big_node = big_map_to_node[j];
            if (big_node > col_map_offd_AN[cnt - 1])
            {
               col_map_offd_AN[cnt++] = big_node;
            }
            map_to_map[j] = cnt - 1;
         }
         recv_vec_starts_AN[i + 1] = cnt;
      }

      for (i = 0; i < num_cols_offd_AN; i++)
      {
         counter[i] = -1;
      }

      AN_num_nonzeros_offd = 0;
      row = 0;
      for (i = 0; i < num_nodes; i++)
      {
         AN_offd_i[i] = AN_num_nonzeros_offd;
         for (j = 0; j < num_functions; j++)
         {
            for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
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


   AN_offd = hypre_CSRMatrixCreate(num_nodes, num_cols_offd_AN,
                                   AN_num_nonzeros_offd);
   hypre_CSRMatrixI(AN_offd) = AN_offd_i;
   if (AN_num_nonzeros_offd)
   {
      AN_offd_j = hypre_CTAlloc(HYPRE_Int,  AN_num_nonzeros_offd, memory_location);
      AN_offd_data = hypre_CTAlloc(HYPRE_Real,  AN_num_nonzeros_offd, memory_location);
      hypre_CSRMatrixJ(AN_offd) = AN_offd_j;
      hypre_CSRMatrixData(AN_offd) = AN_offd_data;

      for (i = 0; i < num_cols_offd_AN; i++)
      {
         counter[i] = -1;
      }
      index = 0;
      row = 0;
      AN_offd_i[0] = 0;
      start_index = 0;
      switch (mode)
      {
         case 1: /* frobenius norm */
         {
            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
                  {
                     k_map = map_to_map[A_offd_j[k]];
                     if (counter[k_map] < start_index)
                     {
                        counter[k_map] = index;
                        AN_offd_j[index] = k_map;
                        AN_offd_data[index] = A_offd_data[k] * A_offd_data[k];
                        index++;
                     }
                     else
                     {
                        AN_offd_data[counter[k_map]] +=
                           A_offd_data[k] * A_offd_data[k];
                     }
                  }
                  row++;
               }
               start_index = index;
            }
            for (i = 0; i < AN_num_nonzeros_offd; i++)
            {
               AN_offd_data[i] = hypre_sqrt(AN_offd_data[i]);
            }
         }
         break;

         case 2:  /* sum of abs. value of all elements in block */
         {
            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
                  {
                     k_map = map_to_map[A_offd_j[k]];
                     if (counter[k_map] < start_index)
                     {
                        counter[k_map] = index;
                        AN_offd_j[index] = k_map;
                        AN_offd_data[index] = hypre_abs(A_offd_data[k]);
                        index++;
                     }
                     else
                     {
                        AN_offd_data[counter[k_map]] += hypre_abs(A_offd_data[k]);
                     }
                  }
                  row++;
               }
               start_index = index;
            }
            for (i = 0; i < AN_num_nonzeros_offd; i++)
            {
               AN_offd_data[i] /= num_fun2;
            }
         }
         break;

         case 3: /* largest element in each block (not abs. value ) */
         {
            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
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
                        if (hypre_abs(A_offd_data[k]) >
                            hypre_abs(AN_offd_data[counter[k_map]]))
                        {
                           AN_offd_data[counter[k_map]] = A_offd_data[k];
                        }
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

            data = hypre_CTAlloc(HYPRE_Real,  AN_num_nonzeros_offd * num_functions, HYPRE_MEMORY_HOST);

            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
                  {
                     k_map = map_to_map[A_offd_j[k]];
                     if (counter[k_map] < start_index)
                     {
                        counter[k_map] = index;
                        AN_offd_j[index] = k_map;
                        data[index * num_functions + j] = hypre_abs(A_offd_data[k]);
                        index++;
                     }
                     else
                     {
                        data[(counter[k_map])*num_functions + j] += hypre_abs(A_offd_data[k]);
                     }
                  }
                  row++;
               }
               start_index = index;
            }
            for (i = 0; i < AN_num_nonzeros_offd; i++)
            {
               AN_offd_data[i]  = data[i * num_functions];

               for (j = 1; j < num_functions; j++)
               {
                  AN_offd_data[i]  = hypre_max( AN_offd_data[i], data[i * num_functions + j]);
               }
            }
            hypre_TFree(data, HYPRE_MEMORY_HOST);

         }
         break;

         case 6:  /* sum of value of all elements in block */
         {
            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  for (k = A_offd_i[row]; k < A_offd_i[row + 1]; k++)
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
   }

   if (diag_option == 1 )
   {
      /* make the diag entry the negative of the sum of off-diag entries (here we are adding the
         off_diag contribution)*/
      /* the diagonal is the first element listed in each row of AN_diag_data - */
      for (i = 0; i < num_nodes; i++)
      {
         sum = 0.0;
         for (k = AN_offd_i[i]; k < AN_offd_i[i + 1]; k++)
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

   hypre_CSRMatrixMemoryLocation(AN_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(AN_offd) = memory_location;

   hypre_ParCSRMatrixColMapOffd(AN) = col_map_offd_AN;
   hypre_ParCSRMatrixCommPkg(AN) = comm_pkg_AN;

   new_num_cols_offd = num_functions * num_cols_offd_AN;

   if (new_num_cols_offd > num_cols_offd)
   {
      new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, new_num_cols_offd, HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < num_cols_offd_AN; i++)
      {
         for (j = 0; j < num_functions; j++)
         {
            new_col_map_offd[cnt++] = (HYPRE_BigInt)num_functions * col_map_offd_AN[i] + (HYPRE_BigInt)j;
         }
      }
      cnt = 0;
      for (i = 0; i < num_cols_offd; i++)
      {
         while (col_map_offd[i] >  new_col_map_offd[cnt])
         {
            cnt++;
         }
         col_map_offd[i] = (HYPRE_BigInt)cnt++;
      }
      for (i = 0; i < num_recvs + 1; i++)
      {
         recv_vec_starts[i] = num_functions * recv_vec_starts_AN[i];
      }

      for (i = 0; i < num_nonzeros_offd; i++)
      {
         j = A_offd_j[i];
         A_offd_j[i] = (HYPRE_Int)col_map_offd[j];
      }
      hypre_ParCSRMatrixColMapOffd(A) = new_col_map_offd;
      hypre_CSRMatrixNumCols(A_offd) = new_num_cols_offd;
      hypre_TFree(col_map_offd, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(big_map_to_node, HYPRE_MEMORY_HOST);
   new_send_elmts_size = send_map_starts_AN[num_sends] * num_functions;

   if (new_send_elmts_size > send_map_starts[num_sends])
   {
      new_send_map_elmts = hypre_CTAlloc(HYPRE_Int, new_send_elmts_size, HYPRE_MEMORY_HOST);
      cnt = 0;
      send_map_starts[0] = 0;
      for (i = 0; i < num_sends; i++)
      {
         send_map_starts[i + 1] = send_map_starts_AN[i + 1] * num_functions;
         for (j = send_map_starts_AN[i]; j < send_map_starts_AN[i + 1]; j++)
         {
            for (k = 0; k < num_functions; k++)
            {
               new_send_map_elmts[cnt++] = send_map_elmts_AN[j] * num_functions + k;
            }
         }
      }
      hypre_TFree(send_map_elmts, HYPRE_MEMORY_HOST);
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = new_send_map_elmts;
   }

   *AN_ptr = AN;

   hypre_TFree(counter, HYPRE_MEMORY_HOST);
   hypre_TFree(map_to_map, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}


/* This creates a scalar version of the CF_marker, dof_array and strength matrix (SN)
 * RL: TODO GPU version */

HYPRE_Int
hypre_BoomerAMGCreateScalarCFS(hypre_ParCSRMatrix  *SN,
                               hypre_ParCSRMatrix  *A,
                               HYPRE_Int           *CFN_marker,
                               HYPRE_Int            num_functions,
                               HYPRE_Int            nodal,
                               HYPRE_Int            keep_same_sign,
                               hypre_IntArray      **dof_func_ptr,
                               hypre_IntArray     **CF_marker_ptr,
                               hypre_ParCSRMatrix **S_ptr)
{
   MPI_Comm            comm = hypre_ParCSRMatrixComm(SN);
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   HYPRE_Int          *S_diag_i;
   HYPRE_Int          *S_diag_j;
   hypre_CSRMatrix    *S_offd;
   HYPRE_Int          *S_offd_i;
   HYPRE_Int          *S_offd_j;
   HYPRE_BigInt        row_starts_S[2];
   HYPRE_BigInt        col_starts_S[2];
   HYPRE_BigInt       *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt       *col_starts_A = hypre_ParCSRMatrixColStarts(A);
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real         *A_diag_data = hypre_CSRMatrixData(A_diag);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real         *A_offd_data = hypre_CSRMatrixData(A_offd);
   hypre_CSRMatrix    *SN_diag = hypre_ParCSRMatrixDiag(SN);
   HYPRE_Int          *SN_diag_i = hypre_CSRMatrixI(SN_diag);
   HYPRE_Int          *SN_diag_j = hypre_CSRMatrixJ(SN_diag);
   hypre_CSRMatrix    *SN_offd = hypre_ParCSRMatrixOffd(SN);
   HYPRE_Int          *SN_offd_i = hypre_CSRMatrixI(SN_offd);
   HYPRE_Int          *SN_offd_j = hypre_CSRMatrixJ(SN_offd);
   HYPRE_Int          *CF_marker;
   HYPRE_BigInt       *col_map_offd_SN = hypre_ParCSRMatrixColMapOffd(SN);
   HYPRE_BigInt       *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt       *col_map_offd_S = NULL;
   HYPRE_Int          *dof_func;
   HYPRE_Int           num_nodes = hypre_CSRMatrixNumRows(SN_diag);
   HYPRE_Int           num_variables;
   HYPRE_Int          *S_marker;
   HYPRE_Int          *S_marker_offd = NULL;
   HYPRE_Int          *S_tmp_j;

   HYPRE_Int           num_coarse_nodes;
   HYPRE_Int           i, j, k, cnt;
   HYPRE_Int           num_procs;
   HYPRE_Int           num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int           A_num_nonzeros_diag;
   HYPRE_Int           A_num_nonzeros_offd;
   HYPRE_Int           S_num_nonzeros_diag;
   HYPRE_Int           S_num_nonzeros_offd;
   HYPRE_BigInt        global_num_vars;
   HYPRE_BigInt        global_num_cols;
   HYPRE_BigInt        global_num_nodes;
   HYPRE_Int           nnz, S_cnt, in;

   HYPRE_MemoryLocation memory_locationS = hypre_ParCSRMatrixMemoryLocation(SN);

   hypre_MPI_Comm_size(comm, &num_procs);

   num_variables = num_functions * num_nodes;

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   if (nodal < 0)
   {
      cnt = 0;
      num_coarse_nodes = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (CFN_marker[i] == 1)
         {
            num_coarse_nodes++;
         }
         for (j = 0; j < num_functions; j++)
         {
            CF_marker[cnt++] = CFN_marker[i];
         }
      }

      *dof_func_ptr = hypre_IntArrayCreate(num_coarse_nodes * num_functions);
      hypre_IntArrayInitialize(*dof_func_ptr);
      dof_func = hypre_IntArrayData(*dof_func_ptr);
      cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         if (CFN_marker[i] == 1)
         {
            for (k = 0; k < num_functions; k++)
            {
               dof_func[cnt++] = k;
            }
         }
      }
   }
   else
   {
      cnt = 0;
      for (i = 0; i < num_nodes; i++)
      {
         for (j = 0; j < num_functions; j++)
         {
            CF_marker[cnt++] = CFN_marker[i];
         }
      }
   }

   for (i = 0; i < 2; i++)
   {
      row_starts_S[i] = row_starts_A[i];
      col_starts_S[i] = col_starts_A[i];
   }

   /*SN_num_nonzeros_diag = SN_diag_i[num_nodes];
   SN_num_nonzeros_offd = SN_offd_i[num_nodes];*/
   A_num_nonzeros_diag = A_diag_i[num_variables];
   A_num_nonzeros_offd = A_offd_i[num_variables];

   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(SN);
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(SN) * num_functions;

   global_num_vars = global_num_nodes * (HYPRE_BigInt)num_functions;

   S_marker = hypre_TAlloc(HYPRE_Int, num_variables, HYPRE_MEMORY_HOST);
   nnz = A_num_nonzeros_diag;
   if (nnz < A_num_nonzeros_offd) { nnz = A_num_nonzeros_offd; }
   S_tmp_j = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_HOST);
   S_diag_i = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_locationS);
   S_offd_i = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_locationS);

   //Generate S_diag_i and S_diag_j
   for (i = 0; i < A_num_nonzeros_diag; i++)
   {
      S_tmp_j[i] = -1;
   }
   for (i = 0; i < num_variables; i++)
   {
      S_marker[i] = -1;
   }

   S_diag_i[0] = 0;
   S_cnt = 0;
   for (in = 0; in < num_nodes; in++)
   {
      HYPRE_Int index, index_A, kn, position;
      for (kn = 0; kn < num_functions; kn++)
      {
         i = in * num_functions + kn;
         position = A_diag_i[i] - 1;
         if (!keep_same_sign)
         {
            if (A_diag_data[A_diag_i[i]] > 0.0)
            {
               for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
               {
                  if (A_diag_data[j] < 0.0)
                  {
                     S_marker[A_diag_j[j]] = j;
                  }
               }
            }
            else
            {
               for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
               {
                  if (A_diag_data[j] > 0.0)
                  {
                     S_marker[A_diag_j[j]] = j;
                  }
               }
            }
         }
         else
         {
            for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
            {
               S_marker[A_diag_j[j]] = j;
            }
         }
         for (j = SN_diag_i[in]; j < SN_diag_i[in + 1]; j++)
         {
            // only include diagonal elements of block, assuming unknown-based
            // approach for interpolation, i.e. ignore connections between different variables
            index = SN_diag_j[j] * num_functions + kn;
            index_A = S_marker[index];
            if (index_A > position)
            {
               S_tmp_j[index_A] = A_diag_j[index_A];
               S_cnt++;
            }
         }
         S_diag_i[i + 1] = S_cnt;
      }
   }

   S_diag_j = hypre_CTAlloc(HYPRE_Int, S_cnt, memory_locationS);
   S_cnt = 0;
   for (i = 0; i < A_num_nonzeros_diag; i++)
   {
      if (S_tmp_j[i] > -1)
      {
         S_diag_j[S_cnt++] = S_tmp_j[i];
      }
   }

   S_num_nonzeros_diag = S_cnt;

   for (i = 0; i < A_num_nonzeros_offd; i++)
   {
      S_tmp_j[i] = -1;
   }

   S_marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd_A, HYPRE_MEMORY_HOST);
   col_map_offd_S = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_A, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_offd_A; i++)
   {
      S_marker_offd[i] = -1;
      col_map_offd_S[i] = col_map_offd_A[i];
   }

   S_offd_i[0] = 0;
   S_cnt = 0;
   for (in = 0; in < num_nodes; in++)
   {
      HYPRE_Int index, index_A, kn, position;
      HYPRE_BigInt big_index;
      for (kn = 0; kn < num_functions; kn++)
      {
         i = in * num_functions + kn;
         position = A_offd_i[i] - 1;
         if (!keep_same_sign)
         {
            if (A_diag_data[A_diag_i[i]] > 0.0)
            {
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  if (A_offd_data[j] < 0.0)
                  {
                     S_marker_offd[A_offd_j[j]] = j;
                  }
               }
            }
            else
            {
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  if (A_offd_data[j] > 0.0)
                  {
                     S_marker_offd[A_offd_j[j]] = j;
                  }
               }
            }
         }
         else
         {
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               S_marker_offd[A_offd_j[j]] = j;
            }
         }
         for (j = SN_offd_i[in]; j < SN_offd_i[in + 1]; j++)
         {
            big_index = col_map_offd_SN[SN_offd_j[j]] * num_functions + kn;
            index = hypre_BigBinarySearch(col_map_offd_A, big_index, num_cols_offd_A);
            if (index > -1)
            {
               index_A = S_marker_offd[index];
               if (index_A > position)
               {
                  S_tmp_j[index_A] = A_offd_j[index_A];
                  S_cnt++;
               }
            }
         }
         S_offd_i[i + 1] = S_cnt;
      }
   }

   S_num_nonzeros_offd = S_cnt;
   S_offd_j = hypre_CTAlloc(HYPRE_Int, S_cnt, memory_locationS);
   S_cnt = 0;
   for (i = 0; i < A_num_nonzeros_offd; i++)
   {
      if (S_tmp_j[i] > -1)
      {
         S_offd_j[S_cnt++] = S_tmp_j[i];
      }
   }

   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_cols,
                                row_starts_S, col_starts_S, num_cols_offd_A,
                                S_num_nonzeros_diag, S_num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);

   hypre_CSRMatrixMemoryLocation(S_diag) = memory_locationS;
   hypre_CSRMatrixMemoryLocation(S_offd) = memory_locationS;

   hypre_CSRMatrixI(S_diag) = S_diag_i;
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   hypre_CSRMatrixI(S_offd) = S_offd_i;
   hypre_CSRMatrixJ(S_offd) = S_offd_j;
   hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;

   hypre_TFree(S_tmp_j, HYPRE_MEMORY_HOST);
   hypre_TFree(S_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(S_marker_offd, HYPRE_MEMORY_HOST);

   *S_ptr = S;

   return hypre_error_flag;
}


/* This function just finds the scalar CF_marker and dof_func */

HYPRE_Int
hypre_BoomerAMGCreateScalarCF(HYPRE_Int                   *CFN_marker,
                              HYPRE_Int                    num_functions,
                              HYPRE_Int                    num_nodes,
                              hypre_IntArray             **dof_func_ptr,
                              hypre_IntArray             **CF_marker_ptr)

{
   HYPRE_Int      *CF_marker;
   HYPRE_Int      *dof_func;
   HYPRE_Int       num_variables;
   HYPRE_Int       num_coarse_nodes;
   HYPRE_Int       i, j, k, cnt;


   num_variables = num_functions * num_nodes;

   /* Allocate CF_marker if not done before */
   if (*CF_marker_ptr == NULL)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   cnt = 0;
   num_coarse_nodes = 0;
   for (i = 0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1) { num_coarse_nodes++; }
      for (j = 0; j < num_functions; j++)
      {
         CF_marker[cnt++] = CFN_marker[i];
      }
   }

   *dof_func_ptr = hypre_IntArrayCreate(num_coarse_nodes * num_functions);
   hypre_IntArrayInitialize(*dof_func_ptr);
   dof_func = hypre_IntArrayData(*dof_func_ptr);
   cnt = 0;
   for (i = 0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1)
      {
         for (k = 0; k < num_functions; k++)
         {
            dof_func[cnt++] = k;
         }
      }
   }

   return hypre_error_flag;
}
