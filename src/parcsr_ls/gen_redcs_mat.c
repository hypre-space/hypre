/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"


#define USE_ALLTOALL 0

/* here we have the sequential setup and solve - called from the
 * parallel one - for the coarser levels */

HYPRE_Int
hypre_seqAMGSetup( hypre_ParAMGData *amg_data,
                   HYPRE_Int         p_level,
                   HYPRE_Int         coarse_threshold)
{
   HYPRE_UNUSED_VAR(coarse_threshold);

   /* Par Data Structure variables */
   hypre_ParCSRMatrix **Par_A_array = hypre_ParAMGDataAArray(amg_data);

   MPI_Comm      comm = hypre_ParCSRMatrixComm(Par_A_array[0]);
   MPI_Comm      new_comm, seq_comm;

   hypre_ParCSRMatrix   *A_seq = NULL;
   hypre_CSRMatrix  *A_seq_diag;
   hypre_CSRMatrix  *A_seq_offd;
   hypre_ParVector   *F_seq = NULL;
   hypre_ParVector   *U_seq = NULL;

   hypre_ParCSRMatrix *A;

   hypre_IntArray         **dof_func_array;
   HYPRE_Int                num_procs, my_id;

   HYPRE_Int                level;
   HYPRE_Int                redundant;
   HYPRE_Int                num_functions;

   HYPRE_Solver  coarse_solver;

   /* misc */
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   num_functions = hypre_ParAMGDataNumFunctions(amg_data);
   redundant = hypre_ParAMGDataRedundant(amg_data);

   /*MPI Stuff */
   hypre_MPI_Comm_size(comm, &num_procs);

   /*initial */
   level = p_level;

   /* convert A at this level to sequential */
   A = Par_A_array[level];

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   {
      HYPRE_Real *A_seq_data = NULL;
      HYPRE_Int *A_seq_i = NULL;
      HYPRE_Int *A_seq_offd_i = NULL;
      HYPRE_Int *A_seq_j = NULL;
      HYPRE_Int *seq_dof_func = NULL;

      HYPRE_Real *A_tmp_data = NULL;
      HYPRE_Int *A_tmp_i = NULL;
      HYPRE_Int *A_tmp_j = NULL;

      HYPRE_Int *info = NULL;
      HYPRE_Int *displs = NULL;
      HYPRE_Int *displs2 = NULL;
      HYPRE_Int i, j, size, num_nonzeros, total_nnz = 0, cnt;

      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
      HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
      HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
      HYPRE_BigInt first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
      HYPRE_Int new_num_procs;
      HYPRE_BigInt  row_starts[2];

      hypre_GenerateSubComm(comm, num_rows, &new_comm);


      /*hypre_MPI_Group orig_group, new_group;
      HYPRE_Int *ranks, new_num_procs, *row_starts;

      info = hypre_CTAlloc(HYPRE_Int,  num_procs, HYPRE_MEMORY_HOST);

      hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, comm);

      ranks = hypre_CTAlloc(HYPRE_Int,  num_procs, HYPRE_MEMORY_HOST);

      new_num_procs = 0;
      for (i=0; i < num_procs; i++)
         if (info[i])
         {
            ranks[new_num_procs] = i;
            info[new_num_procs++] = info[i];
         }

      hypre_MPI_Comm_group(comm, &orig_group);
      hypre_MPI_Group_incl(orig_group, new_num_procs, ranks, &new_group);
      hypre_MPI_Comm_create(comm, new_group, &new_comm);
      hypre_MPI_Group_free(&new_group);
      hypre_MPI_Group_free(&orig_group); */

      if (num_rows)
      {
         hypre_ParAMGDataParticipate(amg_data) = 1;
         hypre_MPI_Comm_size(new_comm, &new_num_procs);
         hypre_MPI_Comm_rank(new_comm, &my_id);
         info = hypre_CTAlloc(HYPRE_Int,  new_num_procs, HYPRE_MEMORY_HOST);

         if (redundant)
         {
            hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);
         }
         else
         {
            hypre_MPI_Gather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, 0, new_comm);
         }

         /* alloc space in seq data structure only for participating procs*/
         if (redundant || my_id == 0)
         {
            HYPRE_BoomerAMGCreate(&coarse_solver);
            HYPRE_BoomerAMGSetMaxRowSum(coarse_solver,
                                        hypre_ParAMGDataMaxRowSum(amg_data));
            HYPRE_BoomerAMGSetStrongThreshold(coarse_solver,
                                              hypre_ParAMGDataStrongThreshold(amg_data));
            HYPRE_BoomerAMGSetCoarsenType(coarse_solver,
                                          hypre_ParAMGDataCoarsenType(amg_data));
            HYPRE_BoomerAMGSetInterpType(coarse_solver,
                                         hypre_ParAMGDataInterpType(amg_data));
            HYPRE_BoomerAMGSetTruncFactor(coarse_solver,
                                          hypre_ParAMGDataTruncFactor(amg_data));
            HYPRE_BoomerAMGSetPMaxElmts(coarse_solver,
                                        hypre_ParAMGDataPMaxElmts(amg_data));
            if (hypre_ParAMGDataUserRelaxType(amg_data) > -1)
               HYPRE_BoomerAMGSetRelaxType(coarse_solver,
                                           hypre_ParAMGDataUserRelaxType(amg_data));
            HYPRE_BoomerAMGSetRelaxOrder(coarse_solver,
                                         hypre_ParAMGDataRelaxOrder(amg_data));
            HYPRE_BoomerAMGSetRelaxWt(coarse_solver,
                                      hypre_ParAMGDataUserRelaxWeight(amg_data));
            if (hypre_ParAMGDataUserNumSweeps(amg_data) > -1)
               HYPRE_BoomerAMGSetNumSweeps(coarse_solver,
                                           hypre_ParAMGDataUserNumSweeps(amg_data));
            HYPRE_BoomerAMGSetNumFunctions(coarse_solver,
                                           num_functions);
            HYPRE_BoomerAMGSetMaxIter(coarse_solver, 1);
            HYPRE_BoomerAMGSetTol(coarse_solver, 0);
         }

         /* Create CSR Matrix, will be Diag part of new matrix */
         A_tmp_i = hypre_CTAlloc(HYPRE_Int,  num_rows + 1, HYPRE_MEMORY_HOST);

         A_tmp_i[0] = 0;
         for (i = 1; i < num_rows + 1; i++)
         {
            A_tmp_i[i] = A_diag_i[i] - A_diag_i[i - 1] + A_offd_i[i] - A_offd_i[i - 1];
         }

         num_nonzeros = A_offd_i[num_rows] + A_diag_i[num_rows];

         A_tmp_j = hypre_CTAlloc(HYPRE_Int,  num_nonzeros, HYPRE_MEMORY_HOST);
         A_tmp_data = hypre_CTAlloc(HYPRE_Real,  num_nonzeros, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_rows; i++)
         {
            for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
            {
               A_tmp_j[cnt] = A_diag_j[j] + (HYPRE_Int)first_row_index;
               A_tmp_data[cnt++] = A_diag_data[j];
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               A_tmp_j[cnt] = (HYPRE_Int)col_map_offd[A_offd_j[j]];
               A_tmp_data[cnt++] = A_offd_data[j];
            }
         }

         displs = hypre_CTAlloc(HYPRE_Int,  new_num_procs + 1, HYPRE_MEMORY_HOST);
         displs[0] = 0;
         for (i = 1; i < new_num_procs + 1; i++)
         {
            displs[i] = displs[i - 1] + info[i - 1];
         }
         size = displs[new_num_procs];

         if (redundant || my_id == 0)
         {
            A_seq_i = hypre_CTAlloc(HYPRE_Int,  size + 1, memory_location);
            A_seq_offd_i = hypre_CTAlloc(HYPRE_Int,  size + 1, memory_location);
            if (num_functions > 1) { seq_dof_func = hypre_CTAlloc(HYPRE_Int,  size, memory_location); }
         }

         if (redundant)
         {
            hypre_MPI_Allgatherv ( &A_tmp_i[1], num_rows, HYPRE_MPI_INT, &A_seq_i[1], info,
                                   displs, HYPRE_MPI_INT, new_comm );
            if (num_functions > 1)
            {
               hypre_MPI_Allgatherv ( hypre_IntArrayData(dof_func_array[level]), num_rows, HYPRE_MPI_INT,
                                      seq_dof_func, info, displs, HYPRE_MPI_INT, new_comm );
               HYPRE_BoomerAMGSetDofFunc(coarse_solver, seq_dof_func);
            }
         }
         else
         {
            if (A_seq_i)
               hypre_MPI_Gatherv ( &A_tmp_i[1], num_rows, HYPRE_MPI_INT, &A_seq_i[1], info,
                                   displs, HYPRE_MPI_INT, 0, new_comm );
            else
               hypre_MPI_Gatherv ( &A_tmp_i[1], num_rows, HYPRE_MPI_INT, A_seq_i, info,
                                   displs, HYPRE_MPI_INT, 0, new_comm );
            if (num_functions > 1)
            {
               hypre_MPI_Gatherv ( hypre_IntArrayData(dof_func_array[level]), num_rows, HYPRE_MPI_INT,
                                   seq_dof_func, info, displs, HYPRE_MPI_INT, 0, new_comm );
               if (my_id == 0) { HYPRE_BoomerAMGSetDofFunc(coarse_solver, seq_dof_func); }
            }
         }

         if (redundant || my_id == 0)
         {
            displs2 = hypre_CTAlloc(HYPRE_Int,  new_num_procs + 1, HYPRE_MEMORY_HOST);

            A_seq_i[0] = 0;
            displs2[0] = 0;
            for (j = 1; j < displs[1]; j++)
            {
               A_seq_i[j] = A_seq_i[j] + A_seq_i[j - 1];
            }
            for (i = 1; i < new_num_procs; i++)
            {
               for (j = displs[i]; j < displs[i + 1]; j++)
               {
                  A_seq_i[j] = A_seq_i[j] + A_seq_i[j - 1];
               }
            }
            A_seq_i[size] = A_seq_i[size] + A_seq_i[size - 1];
            displs2[new_num_procs] = A_seq_i[size];
            for (i = 1; i < new_num_procs + 1; i++)
            {
               displs2[i] = A_seq_i[displs[i]];
               info[i - 1] = displs2[i] - displs2[i - 1];
            }

            total_nnz = displs2[new_num_procs];
            A_seq_j = hypre_CTAlloc(HYPRE_Int,  total_nnz, memory_location);
            A_seq_data = hypre_CTAlloc(HYPRE_Real,  total_nnz, memory_location);
         }
         if (redundant)
         {
            hypre_MPI_Allgatherv ( A_tmp_j, num_nonzeros, HYPRE_MPI_INT,
                                   A_seq_j, info, displs2,
                                   HYPRE_MPI_INT, new_comm );

            hypre_MPI_Allgatherv ( A_tmp_data, num_nonzeros, HYPRE_MPI_REAL,
                                   A_seq_data, info, displs2,
                                   HYPRE_MPI_REAL, new_comm );
         }
         else
         {
            hypre_MPI_Gatherv ( A_tmp_j, num_nonzeros, HYPRE_MPI_INT,
                                A_seq_j, info, displs2,
                                HYPRE_MPI_INT, 0, new_comm );

            hypre_MPI_Gatherv ( A_tmp_data, num_nonzeros, HYPRE_MPI_REAL,
                                A_seq_data, info, displs2,
                                HYPRE_MPI_REAL, 0, new_comm );
         }

         hypre_TFree(info, HYPRE_MEMORY_HOST);
         hypre_TFree(displs, HYPRE_MEMORY_HOST);
         hypre_TFree(A_tmp_i, HYPRE_MEMORY_HOST);
         hypre_TFree(A_tmp_j, HYPRE_MEMORY_HOST);
         hypre_TFree(A_tmp_data, HYPRE_MEMORY_HOST);

         if (redundant || my_id == 0)
         {
            hypre_TFree(displs2, HYPRE_MEMORY_HOST);

            row_starts[0] = 0;
            row_starts[1] = size;

            /* Create 1 proc communicator */
            seq_comm = hypre_MPI_COMM_SELF;

            A_seq = hypre_ParCSRMatrixCreate(seq_comm, size, size,
                                             row_starts, row_starts,
                                             0, total_nnz, 0);

            A_seq_diag = hypre_ParCSRMatrixDiag(A_seq);
            A_seq_offd = hypre_ParCSRMatrixOffd(A_seq);

            hypre_CSRMatrixData(A_seq_diag) = A_seq_data;
            hypre_CSRMatrixI(A_seq_diag) = A_seq_i;
            hypre_CSRMatrixJ(A_seq_diag) = A_seq_j;
            hypre_CSRMatrixI(A_seq_offd) = A_seq_offd_i;

            F_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
            U_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
            hypre_ParVectorInitialize(F_seq);
            hypre_ParVectorInitialize(U_seq);

            hypre_BoomerAMGSetup(coarse_solver, A_seq, F_seq, U_seq);

            hypre_ParAMGDataCoarseSolver(amg_data) = coarse_solver;
            hypre_ParAMGDataACoarse(amg_data) = A_seq;
            hypre_ParAMGDataFCoarse(amg_data) = F_seq;
            hypre_ParAMGDataUCoarse(amg_data) = U_seq;
         }
         hypre_ParAMGDataNewComm(amg_data) = new_comm;
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_seqAMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_seqAMGCycle( hypre_ParAMGData *amg_data,
                   HYPRE_Int p_level,
                   hypre_ParVector  **Par_F_array,
                   hypre_ParVector  **Par_U_array   )
{

   hypre_ParVector    *Aux_U;
   hypre_ParVector    *Aux_F;

   /* Local variables  */

   HYPRE_Int       Solve_err_flag = 0;

   HYPRE_Int n;
   HYPRE_Int i;

   hypre_Vector   *u_local;
   HYPRE_Real     *u_data;

   HYPRE_Int       first_index;

   /* Acquire seq data */
   MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
   HYPRE_Solver coarse_solver = hypre_ParAMGDataCoarseSolver(amg_data);
   hypre_ParCSRMatrix *A_coarse = hypre_ParAMGDataACoarse(amg_data);
   hypre_ParVector *F_coarse = hypre_ParAMGDataFCoarse(amg_data);
   hypre_ParVector *U_coarse = hypre_ParAMGDataUCoarse(amg_data);
   HYPRE_Int redundant = hypre_ParAMGDataRedundant(amg_data);

   Aux_U = Par_U_array[p_level];
   Aux_F = Par_F_array[p_level];

   first_index = (HYPRE_Int)hypre_ParVectorFirstIndex(Aux_U);
   u_local = hypre_ParVectorLocalVector(Aux_U);
   u_data  = hypre_VectorData(u_local);
   n =  hypre_VectorSize(u_local);


   /*if (A_coarse)*/
   if (hypre_ParAMGDataParticipate(amg_data))
   {
      HYPRE_Real     *f_data;
      hypre_Vector   *f_local;
      hypre_Vector   *tmp_vec;

      HYPRE_Int nf;
      HYPRE_Int local_info;
      HYPRE_Real *recv_buf = NULL;
      HYPRE_Int *displs = NULL;
      HYPRE_Int *info = NULL;
      HYPRE_Int new_num_procs, my_id;

      hypre_MPI_Comm_size(new_comm, &new_num_procs);
      hypre_MPI_Comm_rank(new_comm, &my_id);

      f_local = hypre_ParVectorLocalVector(Aux_F);
      f_data = hypre_VectorData(f_local);
      nf =  hypre_VectorSize(f_local);

      /* first f */
      info = hypre_CTAlloc(HYPRE_Int,  new_num_procs, HYPRE_MEMORY_HOST);
      local_info = nf;
      if (redundant)
      {
         hypre_MPI_Allgather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);
      }
      else
      {
         hypre_MPI_Gather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, 0, new_comm);
      }

      if (redundant || my_id == 0)
      {
         displs = hypre_CTAlloc(HYPRE_Int,  new_num_procs + 1, HYPRE_MEMORY_HOST);
         displs[0] = 0;
         for (i = 1; i < new_num_procs + 1; i++)
         {
            displs[i] = displs[i - 1] + info[i - 1];
         }

         if (F_coarse)
         {
            tmp_vec =  hypre_ParVectorLocalVector(F_coarse);
            recv_buf = hypre_VectorData(tmp_vec);
         }
      }

      if (redundant)
         hypre_MPI_Allgatherv ( f_data, nf, HYPRE_MPI_REAL,
                                recv_buf, info, displs,
                                HYPRE_MPI_REAL, new_comm );
      else
         hypre_MPI_Gatherv ( f_data, nf, HYPRE_MPI_REAL,
                             recv_buf, info, displs,
                             HYPRE_MPI_REAL, 0, new_comm );

      if (redundant || my_id == 0)
      {
         tmp_vec =  hypre_ParVectorLocalVector(U_coarse);
         recv_buf = hypre_VectorData(tmp_vec);
      }

      /*then u */
      if (redundant)
      {
         hypre_MPI_Allgatherv ( u_data, n, HYPRE_MPI_REAL,
                                recv_buf, info, displs,
                                HYPRE_MPI_REAL, new_comm );
         hypre_TFree(displs, HYPRE_MEMORY_HOST);
         hypre_TFree(info, HYPRE_MEMORY_HOST);
      }
      else
         hypre_MPI_Gatherv ( u_data, n, HYPRE_MPI_REAL,
                             recv_buf, info, displs,
                             HYPRE_MPI_REAL, 0, new_comm );

      /* clean up */
      if (redundant || my_id == 0)
      {
         hypre_BoomerAMGSolve(coarse_solver, A_coarse, F_coarse, U_coarse);
      }

      /*copy my part of U to parallel vector */
      if (redundant)
      {
         HYPRE_Real *local_data;

         local_data =  hypre_VectorData(hypre_ParVectorLocalVector(U_coarse));

         for (i = 0; i < n; i++)
         {
            u_data[i] = local_data[first_index + i];
         }
      }
      else
      {
         HYPRE_Real *local_data = NULL;

         if (my_id == 0)
         {
            local_data =  hypre_VectorData(hypre_ParVectorLocalVector(U_coarse));
         }

         hypre_MPI_Scatterv ( local_data, info, displs, HYPRE_MPI_REAL,
                              u_data, n, HYPRE_MPI_REAL, 0, new_comm );
         /*if (my_id == 0)
            local_data =  hypre_VectorData(hypre_ParVectorLocalVector(F_coarse));
            hypre_MPI_Scatterv ( local_data, info, displs, HYPRE_MPI_REAL,
                       f_data, n, HYPRE_MPI_REAL, 0, new_comm );*/
         if (my_id == 0) { hypre_TFree(displs, HYPRE_MEMORY_HOST); }
         hypre_TFree(info, HYPRE_MEMORY_HOST);
      }
   }

   return (Solve_err_flag);
}

/*--------------------------------------------------------------------------
 * hypre_GenerateSubComm
 *
 * generate sub communicator, which contains no idle processors
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GenerateSubComm(MPI_Comm   comm,
                      HYPRE_Int  participate,
                      MPI_Comm  *new_comm_ptr)
{
   MPI_Comm          new_comm;
   hypre_MPI_Group   orig_group, new_group;
   hypre_MPI_Op      hypre_MPI_MERGE;
   HYPRE_Int        *info, *ranks, new_num_procs, my_info, my_id, num_procs;
   HYPRE_Int        *list_len;

   hypre_MPI_Comm_rank(comm, &my_id);

   if (participate)
   {
      my_info = 1;
   }
   else
   {
      my_info = 0;
   }

   hypre_MPI_Allreduce(&my_info, &new_num_procs, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);

   if (new_num_procs == 0)
   {
      new_comm = hypre_MPI_COMM_NULL;
      *new_comm_ptr = new_comm;

      return hypre_error_flag;
   }

   ranks = hypre_CTAlloc(HYPRE_Int, new_num_procs + 2, HYPRE_MEMORY_HOST);

   if (new_num_procs == 1)
   {
      if (participate)
      {
         my_info = my_id;
      }
      hypre_MPI_Allreduce(&my_info, &ranks[2], 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   }
   else
   {
      info = hypre_CTAlloc(HYPRE_Int, new_num_procs + 2, HYPRE_MEMORY_HOST);
      list_len = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST);

      if (participate)
      {
         info[0] = 1;
         info[1] = 1;
         info[2] = my_id;
      }
      else
      {
         info[0] = 0;
      }

      list_len[0] = new_num_procs + 2;

      hypre_MPI_Op_create((hypre_MPI_User_function *)hypre_merge_lists, 0, &hypre_MPI_MERGE);

      hypre_MPI_Allreduce(info, ranks, list_len[0], HYPRE_MPI_INT, hypre_MPI_MERGE, comm);

      hypre_MPI_Op_free (&hypre_MPI_MERGE);

      hypre_TFree(list_len, HYPRE_MEMORY_HOST);
      hypre_TFree(info, HYPRE_MEMORY_HOST);
   }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_group(comm, &orig_group);
   hypre_MPI_Group_incl(orig_group, new_num_procs, &ranks[2], &new_group);
   hypre_MPI_Comm_create(comm, new_group, &new_comm);
   hypre_MPI_Group_free(&new_group);
   hypre_MPI_Group_free(&orig_group);

   hypre_TFree(ranks, HYPRE_MEMORY_HOST);

   *new_comm_ptr = new_comm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_merge_lists
 *--------------------------------------------------------------------------*/

void
hypre_merge_lists(HYPRE_Int          *list1,
                  HYPRE_Int          *list2,
                  hypre_int          *np1,
                  hypre_MPI_Datatype *dptr)
{
   HYPRE_UNUSED_VAR(dptr);

   HYPRE_Int i, len1, len2, indx1, indx2;

   if (list1[0] == 0)
   {
      return;
   }
   else
   {
      list2[0] = 1;
      len1 = list1[1];
      len2 = list2[1];
      list2[1] = len1 + len2;
      if ((hypre_int)(list2[1]) > *np1 + 2) // RL:???
      {
         printf("segfault in MPI User function merge_list\n");
      }
      indx1 = len1 + 1;
      indx2 = len2 + 1;
      for (i = len1 + len2 + 1; i > 1; i--)
      {
         if (indx2 > 1 && indx1 > 1 && list1[indx1] > list2[indx2])
         {
            list2[i] = list1[indx1];
            indx1--;
         }
         else if (indx2 > 1)
         {
            list2[i] = list2[indx2];
            indx2--;
         }
         else if (indx1 > 1)
         {
            list2[i] = list1[indx1];
            indx1--;
         }
      }
   }
}
