#include "_hypre_parcsr_ls.h"
#include "par_amg.h"


#define USE_ALLTOALL 0

/* here we have the sequential setup and solve - called from the
 * parallel one - for the coarser levels */

HYPRE_Int hypre_seqAMGSetup( hypre_ParAMGData *amg_data,
                      HYPRE_Int p_level,
                      HYPRE_Int coarse_threshold)


{

   /* Par Data Structure variables */
   hypre_ParCSRMatrix **Par_A_array = hypre_ParAMGDataAArray(amg_data);

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(Par_A_array[0]); 
   MPI_Comm 	      new_comm, seq_comm;

   hypre_ParCSRMatrix   *A_seq = NULL;
   hypre_CSRMatrix  *A_seq_diag;
   hypre_CSRMatrix  *A_seq_offd;
   hypre_ParVector   *F_seq = NULL;
   hypre_ParVector   *U_seq = NULL;

   hypre_ParCSRMatrix *A;

   HYPRE_Int               **dof_func_array;   
   HYPRE_Int                num_procs, my_id;

   HYPRE_Int                not_finished_coarsening;
   HYPRE_Int                level;

   HYPRE_Solver  coarse_solver;

   /* misc */
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);

   /*MPI Stuff */
   hypre_MPI_Comm_size(comm, &num_procs);   
   hypre_MPI_Comm_rank(comm,&my_id);
  
   /*initial */
   level = p_level;
   
   not_finished_coarsening = 1;
  
   /* convert A at this level to sequential */
   A = Par_A_array[level];

   {
      double *A_seq_data = NULL;
      HYPRE_Int *A_seq_i = NULL;
      HYPRE_Int *A_seq_offd_i = NULL;
      HYPRE_Int *A_seq_j = NULL;

      double *A_tmp_data = NULL;
      HYPRE_Int *A_tmp_i = NULL;
      HYPRE_Int *A_tmp_j = NULL;

      HYPRE_Int *info, *displs, *displs2;
      HYPRE_Int i, j, size, num_nonzeros, total_nnz, cnt;
  
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
      HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
      double *A_diag_data = hypre_CSRMatrixData(A_diag);
      double *A_offd_data = hypre_CSRMatrixData(A_offd);
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
      HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);

      hypre_MPI_Group orig_group, new_group; 
      HYPRE_Int *ranks, new_num_procs, *row_starts;

      info = hypre_CTAlloc(HYPRE_Int, num_procs);

      hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, comm);

      ranks = hypre_CTAlloc(HYPRE_Int, num_procs);

      new_num_procs = 0;
      for (i=0; i < num_procs; i++)
         if (info[i]) 
         {
            ranks[new_num_procs] = i;
            info[new_num_procs++] = info[i];
         }

      MPI_Comm_group(comm, &orig_group);
      hypre_MPI_Group_incl(orig_group, new_num_procs, ranks, &new_group);
      MPI_Comm_create(comm, new_group, &new_comm);
      hypre_MPI_Group_free(&new_group);
      hypre_MPI_Group_free(&orig_group);

      if (num_rows)
      {
         /* alloc space in seq data structure only for participating procs*/
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
		hypre_ParAMGDataNumFunctions(amg_data)); 
         HYPRE_BoomerAMGSetMaxIter(coarse_solver, 1); 
         HYPRE_BoomerAMGSetTol(coarse_solver, 0); 

         /* Create CSR Matrix, will be Diag part of new matrix */
         A_tmp_i = hypre_CTAlloc(HYPRE_Int, num_rows+1);

         A_tmp_i[0] = 0;
         for (i=1; i < num_rows+1; i++)
            A_tmp_i[i] = A_diag_i[i]-A_diag_i[i-1]+A_offd_i[i]-A_offd_i[i-1];

         num_nonzeros = A_offd_i[num_rows]+A_diag_i[num_rows];

         A_tmp_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros);
         A_tmp_data = hypre_CTAlloc(double, num_nonzeros);

         cnt = 0;
         for (i=0; i < num_rows; i++)
         {
            for (j=A_diag_i[i]; j < A_diag_i[i+1]; j++)
	    {
	       A_tmp_j[cnt] = A_diag_j[j]+first_row_index;
	       A_tmp_data[cnt++] = A_diag_data[j];
	    }
            for (j=A_offd_i[i]; j < A_offd_i[i+1]; j++)
	    {
	       A_tmp_j[cnt] = col_map_offd[A_offd_j[j]];
	       A_tmp_data[cnt++] = A_offd_data[j];
	    }
         }

         displs = hypre_CTAlloc(HYPRE_Int, new_num_procs+1);
         displs[0] = 0;
         for (i=1; i < new_num_procs+1; i++)
            displs[i] = displs[i-1]+info[i-1];
         size = displs[new_num_procs];
  
         A_seq_i = hypre_CTAlloc(HYPRE_Int, size+1);
         A_seq_offd_i = hypre_CTAlloc(HYPRE_Int, size+1);

         hypre_MPI_Allgatherv ( &A_tmp_i[1], num_rows, HYPRE_MPI_INT, &A_seq_i[1], info, 
			displs, HYPRE_MPI_INT, new_comm );

         displs2 = hypre_CTAlloc(HYPRE_Int, new_num_procs+1);

         A_seq_i[0] = 0;
         displs2[0] = 0;
         for (j=1; j < displs[1]; j++)
            A_seq_i[j] = A_seq_i[j]+A_seq_i[j-1];
         for (i=1; i < new_num_procs; i++)
         {
            for (j=displs[i]; j < displs[i+1]; j++)
            {
               A_seq_i[j] = A_seq_i[j]+A_seq_i[j-1];
            }
         }
         A_seq_i[size] = A_seq_i[size]+A_seq_i[size-1];
         displs2[new_num_procs] = A_seq_i[size];
         for (i=1; i < new_num_procs+1; i++)
         {
            displs2[i] = A_seq_i[displs[i]];
            info[i-1] = displs2[i] - displs2[i-1];
         }

         total_nnz = displs2[new_num_procs];
         A_seq_j = hypre_CTAlloc(HYPRE_Int, total_nnz);
         A_seq_data = hypre_CTAlloc(double, total_nnz);

         hypre_MPI_Allgatherv ( A_tmp_j, num_nonzeros, HYPRE_MPI_INT,
                       A_seq_j, info, displs2,
                       HYPRE_MPI_INT, new_comm );

         hypre_MPI_Allgatherv ( A_tmp_data, num_nonzeros, hypre_MPI_DOUBLE,
                       A_seq_data, info, displs2,
                       hypre_MPI_DOUBLE, new_comm );

         hypre_TFree(displs);
         hypre_TFree(displs2);
         hypre_TFree(A_tmp_i);
         hypre_TFree(A_tmp_j);
         hypre_TFree(A_tmp_data);
   
         row_starts = hypre_CTAlloc(HYPRE_Int,2);
         row_starts[0] = 0; 
         row_starts[1] = size;
 
         /* Create 1 proc communicator */
         seq_comm = hypre_MPI_COMM_SELF;

         A_seq = hypre_ParCSRMatrixCreate(seq_comm,size,size,
					  row_starts, row_starts,
						0,total_nnz,0); 

         A_seq_diag = hypre_ParCSRMatrixDiag(A_seq);
         A_seq_offd = hypre_ParCSRMatrixOffd(A_seq);

         hypre_CSRMatrixData(A_seq_diag) = A_seq_data;
         hypre_CSRMatrixI(A_seq_diag) = A_seq_i;
         hypre_CSRMatrixJ(A_seq_diag) = A_seq_j;
         hypre_CSRMatrixI(A_seq_offd) = A_seq_offd_i;

         F_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
         U_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
         hypre_ParVectorOwnsPartitioning(F_seq) = 0;
         hypre_ParVectorOwnsPartitioning(U_seq) = 0;
         hypre_ParVectorInitialize(F_seq);
         hypre_ParVectorInitialize(U_seq);

         hypre_BoomerAMGSetup(coarse_solver,A_seq,F_seq,U_seq);

         hypre_ParAMGDataCoarseSolver(amg_data) = coarse_solver;
         hypre_ParAMGDataACoarse(amg_data) = A_seq;
         hypre_ParAMGDataFCoarse(amg_data) = F_seq;
         hypre_ParAMGDataUCoarse(amg_data) = U_seq;
         hypre_ParAMGDataNewComm(amg_data) = new_comm;
      }
      hypre_TFree(info);
      hypre_TFree(ranks);
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
   double         *u_data;
   
   HYPRE_Int	   first_index;
   
   /* Acquire seq data */
   MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
   HYPRE_Solver coarse_solver = hypre_ParAMGDataCoarseSolver(amg_data);
   hypre_ParCSRMatrix *A_coarse = hypre_ParAMGDataACoarse(amg_data);
   hypre_ParVector *F_coarse = hypre_ParAMGDataFCoarse(amg_data);
   hypre_ParVector *U_coarse = hypre_ParAMGDataUCoarse(amg_data);

   Aux_U = Par_U_array[p_level];
   Aux_F = Par_F_array[p_level];

   first_index = hypre_ParVectorFirstIndex(Aux_U);
   u_local = hypre_ParVectorLocalVector(Aux_U);
   u_data  = hypre_VectorData(u_local);
   n =  hypre_VectorSize(u_local);


   if (A_coarse)
   {
      double         *f_data;
      hypre_Vector   *f_local;
      hypre_Vector   *tmp_vec;
      
      HYPRE_Int nf;
      HYPRE_Int local_info;
      double *recv_buf;
      HYPRE_Int *displs, *info;
      HYPRE_Int size;
      HYPRE_Int new_num_procs;
      
      hypre_MPI_Comm_size(new_comm, &new_num_procs);

      f_local = hypre_ParVectorLocalVector(Aux_F);
      f_data = hypre_VectorData(f_local);
      nf =  hypre_VectorSize(f_local);

      /* first f */
      info = hypre_CTAlloc(HYPRE_Int, new_num_procs);
      local_info = nf;
      hypre_MPI_Allgather(&local_info, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);

      displs = hypre_CTAlloc(HYPRE_Int, new_num_procs+1);
      displs[0] = 0;
      for (i=1; i < new_num_procs+1; i++)
         displs[i] = displs[i-1]+info[i-1]; 
      size = displs[new_num_procs];
      
      tmp_vec =  hypre_ParVectorLocalVector(F_coarse);
      recv_buf = hypre_VectorData(tmp_vec);

      hypre_MPI_Allgatherv ( f_data, nf, hypre_MPI_DOUBLE, 
                          recv_buf, info, displs, 
                          hypre_MPI_DOUBLE, new_comm );

      tmp_vec =  hypre_ParVectorLocalVector(U_coarse);
      recv_buf = hypre_VectorData(tmp_vec);
      
      /*then u */
      hypre_MPI_Allgatherv ( u_data, n, hypre_MPI_DOUBLE, 
                       recv_buf, info, displs, 
                       hypre_MPI_DOUBLE, new_comm );
         
      /* clean up */
      hypre_TFree(displs);
      hypre_TFree(info);

      hypre_BoomerAMGSolve(coarse_solver, A_coarse, F_coarse, U_coarse);      

      /*copy my part of U to parallel vector */
      {
         double *local_data;

         local_data =  hypre_VectorData(hypre_ParVectorLocalVector(U_coarse));

         for (i = 0; i < n; i++)
         {
            u_data[i] = local_data[first_index+i];
         }
      }
   }

   return(Solve_err_flag);
}

