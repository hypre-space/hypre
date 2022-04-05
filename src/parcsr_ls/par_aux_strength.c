/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*==========================================================================*/
/*==========================================================================*/
/**
  Generates an auxiilliary matrix, S_aux, with  M-matrix properties such that S_aux = A - B, 
  where B is obtained by distributing positive off-diagonal contributions to 'strong' neighbors.
  
  NOTE:: Current implementation assumes A is a square matrix with a regular partitioning of 
              row and columns.

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param A [IN]
  coefficient matrix
  @param S_ptr [OUT]
  strength matrix

  @see */
/*--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCreateAuxMMatrix(hypre_ParCSRMatrix    *A,
                           hypre_ParCSRMatrix   **S_aux_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /* diag part of S */
   hypre_CSRMatrix    *S_diag   = NULL;
   HYPRE_Int          *S_diag_i = NULL;
   HYPRE_Int          *S_diag_j = NULL;
//   HYPRE_Int           skip_diag = S ? 0 : 1;
   /* off-diag part of S */
   hypre_CSRMatrix    *S_offd   = NULL;
   HYPRE_Int          *S_offd_i = NULL;
   HYPRE_Int          *S_offd_j = NULL;

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt        global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int           num_cols_A_diag = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_BigInt   *A_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt   *A_offd_colmap = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_Int                A_diag_nnz = A_diag_i[num_variables];
   HYPRE_Int                A_offd_nnz = A_offd_i[num_variables];
   
   HYPRE_Int           num_nonzeros_diag;
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *S_aux;
   hypre_CSRMatrix    *S_aux_diag;
   HYPRE_Int          *S_aux_diag_i = NULL;
   HYPRE_Int          *S_aux_diag_j = NULL;
   HYPRE_Complex      *S_aux_diag_data = NULL;      
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_aux_offd;
   HYPRE_Int          *S_aux_offd_i = NULL;
   HYPRE_Int          *S_aux_offd_j = NULL;
   HYPRE_Complex      *S_aux_offd_data = NULL;   
   HYPRE_BigInt   *S_aux_offd_colmap = NULL;      
   /* HYPRE_Real         *S_offd_data; */
   /* off processor portions of S */
   hypre_CSRMatrix    *A_ext                 = NULL;
   HYPRE_Int          *A_ext_i               = NULL;
   HYPRE_Real         *A_ext_data            = NULL;
   HYPRE_BigInt       *A_ext_j               = NULL;
   HYPRE_Int                A_ext_nnz = 0;  
   
   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int           i,j,k,jj, jA, jS, kp, kn, m, n, P_n_max, N_n_max, cnt_n, cnt_p;

   HYPRE_Int           ierr = 0;

   HYPRE_Int          *dof_func_offd;
   HYPRE_Int           num_sends;
   HYPRE_Int          *int_buf_data;
   HYPRE_Int           index, start;

   HYPRE_Int          *P_i = NULL;
   HYPRE_Int          *P_n = NULL;   
   HYPRE_BigInt          *N_i = NULL;
   HYPRE_Int          *N_n = NULL;   
   HYPRE_BigInt          *B_i = NULL;   
   HYPRE_Int          *B_n = NULL;   
   
   HYPRE_Complex pval, bval;
   
   HYPRE_Int *prefix_sum_workspace;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   
   HYPRE_Int num_procs, my_id;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   
   // Begin:
   /* Allocate memory for auxilliary strength matrix */
   S_aux = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_A_offd, A_diag_nnz, A_offd_nnz);

   S_aux_diag = hypre_ParCSRMatrixDiag(S_aux);
   hypre_CSRMatrixI(S_aux_diag) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);
   hypre_CSRMatrixJ(S_aux_diag) = hypre_CTAlloc(HYPRE_Int, A_diag_nnz, memory_location);
   hypre_CSRMatrixData(S_aux_diag) = hypre_CTAlloc(HYPRE_Complex, A_diag_nnz, memory_location);

   S_aux_diag_i = hypre_CSRMatrixI(S_aux_diag);
   S_aux_diag_j = hypre_CSRMatrixJ(S_aux_diag);   
   S_aux_diag_data = hypre_CSRMatrixData(S_aux_diag); 

   S_aux_offd = hypre_ParCSRMatrixOffd(S_aux);
   hypre_CSRMatrixI(S_aux_offd) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);

   S_aux_offd_i = hypre_CSRMatrixI(S_aux_offd);


   if (num_cols_A_offd)
   {
      hypre_CSRMatrixJ(S_aux_offd) = hypre_CTAlloc(HYPRE_Int, A_offd_nnz, memory_location);
      hypre_CSRMatrixData(S_aux_offd) = hypre_CTAlloc(HYPRE_Complex, A_offd_nnz, memory_location);

      S_aux_offd_j = hypre_CSRMatrixJ(S_aux_offd);   
      S_aux_offd_data = hypre_CSRMatrixData(S_aux_offd); 

      S_aux_offd_colmap  = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, memory_location);
      hypre_ParCSRMatrixColMapOffd(S_aux) = S_aux_offd_colmap;

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         S_aux_offd_colmap[i] = A_offd_colmap[i];
      }
   }

   // extract off-processor portion of A
   if (num_procs > 1)
   {
      A_ext      = hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_data = hypre_CSRMatrixData(A_ext);
      A_ext_i    = hypre_CSRMatrixI(A_ext);
      A_ext_j    = hypre_CSRMatrixBigJ(A_ext);
      A_ext_nnz = A_ext_i[num_cols_A_offd]; 
      
   }
   
   /* allocate work arrays */
   HYPRE_Int P_nnz = A_diag_nnz + A_offd_nnz;
   HYPRE_Int N_nnz = A_diag_nnz + A_offd_nnz + A_ext_nnz;
   P_i    = hypre_CTAlloc(HYPRE_Int, P_nnz, memory_location);
   P_n    = hypre_CTAlloc(HYPRE_Int, (num_variables + 1), memory_location);
   N_i    = hypre_CTAlloc(HYPRE_BigInt, N_nnz, memory_location);
   N_n    = hypre_CTAlloc(HYPRE_Int, (num_variables + num_cols_A_offd + 1), memory_location);
      
   /* 1. Loop over matrix rows to extract positive and negative neighbors */
   P_n_max=0;
   N_n_max=0;
   kp = 0;
   kn = 0;
   for(i = 0; i<num_variables; i++)
   {
      cnt_p = kp;
      cnt_n = kn;
      // diagonal part
      // Check for positive off-diagonal entry
      for(j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
      {
         if( (A_diag_data[j] > 0) && (A_diag_j[j] != i))
         {
            // position of positive neighbor. Store position for quick and easy access
            P_i[kp++] = j; //A_diag_j[j];
         }
         else if (A_diag_j[j] != i)
         {
            N_i[kn++] = A_diag_j[j] + A_col_starts[0];
         }
      }
      // Off-diagonal part
      for(j=A_offd_i[i]; j<A_offd_i[i+1]; j++)
      {
         if( A_offd_data[j] > 0)
         {
            // position of positive neighbor. Offset to distinguish from diag part columns
            P_i[kp++] = j + A_diag_nnz;
         }
         else
         {
            N_i[kn++] = A_offd_colmap[A_offd_j[j]]; 
         }
      }
      cnt_p = kp - cnt_p;
      cnt_n = kn - cnt_n;
      P_n_max = P_n_max > cnt_p ? P_n_max : cnt_p;      
      N_n_max = N_n_max > cnt_n ? N_n_max : cnt_n;
      // sort neighbor arrays
      hypre_BigQsort0(N_i, N_n[i], (kn-1));
      // update index pointer arrays
      P_n[i + 1] = kp;
      N_n[i + 1] = kn;
   }
   // add negative neighbors list of external rows
   kn = N_n[num_variables];
   for(i=0; i<num_cols_A_offd; i++)
   {
      for(j=A_ext_i[i]; j<A_ext_i[i+1]; j++)
      {
         if( A_ext_data[j] < 0)
         {
            // get negative neighbor list
            N_i[kn++] = A_ext_j[j]; 
         }
      }
      jj = num_variables + i;      
      // sort neighbor arrays
      hypre_BigQsort0(N_i, N_n[jj], (kn-1));
      // update index pointer arrays
      N_n[jj + 1] = kn;            
   }
   hypre_CSRMatrixDestroy(A_ext);   
   /* 2. Loop to compute intersections of strong (negative) neighbors of common weak (positive) entry */
   B_i    = hypre_CTAlloc(HYPRE_BigInt, (N_n_max * P_n[num_variables]), memory_location);
   B_n    = hypre_CTAlloc(HYPRE_Int, (P_n_max * num_variables + 1), memory_location);   
   kn = 0;
   kp = 0;
   for(i = 0; i<num_variables; i++)
   {
      // loop over positive neighbor list 
      for(j = P_n[i]; j< P_n[i+1]; j++)
      {
         jA = P_i[j] - A_diag_nnz;
         // Local row (assumes square matrix)
         if( jA < 0 )
         {
            k = A_diag_j[P_i[j]] ;
         }
         else
         {
            // external row. Access external variable range of N.
            k = A_offd_j[jA] + num_variables;           
         }
         // get intersection of N_i [i] and N_i[k]
         m = N_n[i + 1] - N_n[i];
         n = N_n[k + 1] - N_n[k];
         hypre_IntersectTwoBigIntegerArrays(&N_i[N_n[i]], m, &N_i[N_n[k]], n, &B_i[kn], &kp); 
         // update position of intersection of strong neighbors
         kn += kp;         
         B_n[j + 1] = kn;            
      }
   }
   hypre_TFree(N_i, memory_location);
   hypre_TFree(N_n, memory_location);
   /* 3. Fill in data for S_aux */
   kp = 0;
   kn = 0;
   for(i = 0; i<num_variables; i++)
   {    
      // diagonal part
      for(j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
      {
         // Skip positive off-diagonal entry
         if( (A_diag_data[j] > 0) && (A_diag_j[j] != i)) continue;
         
         S_aux_diag_data[kp] = A_diag_data[j];
         S_aux_diag_j[kp++] = A_diag_j[j];
      }
      // Off-diagonal part
      for(j=A_offd_i[i]; j<A_offd_i[i+1]; j++)
      {
         // Skip positive off-diagonal entry
         if( A_offd_data[j] > 0) continue;
         
         S_aux_offd_data[kn] = A_offd_data[j];
         S_aux_offd_j[kn++] = A_offd_j[j]; 
      }
             
      // update index pointer arrays
      S_aux_diag_i[i + 1] = kp;
      S_aux_offd_i[i + 1] = kn;
   }     
   /* Distribute positive contributions to strong neighbors */
   for(i = 0; i<num_variables; i++)
   {   
      // loop over positive neighbors and compute contributions to (strong) negative neighbors
      for(j = P_n[i]; j< P_n[i+1]; j++)
      {
        jA = P_i[j] - A_diag_nnz;
         // positive entry is in diag part
         if( jA < 0 )
         {
            pval = A_diag_data[P_i[j]] ;
         }
         else
         {
            // positive entry is in offd part
            pval = A_offd_data[jA];           
         }

         // compute distribution to negative neighbors 
         bval = -2.0 * pval / (HYPRE_Complex) (B_n[ j+1 ] - B_n[ j ]);

         // Loop over negative connections to positive neighbor and distribute 
         for(k = B_n[ j ]; k<B_n[ j+1 ]; k++)
         {         
           HYPRE_BigInt big_col = B_i[k];
            if( big_col >= A_col_starts[0] && big_col < A_col_starts[1])
            {
               // neighbor is in diag part
               jS = S_aux_diag_i[i] + 1;
               jj = (HYPRE_Int) (big_col - A_col_starts[0]);
               while (S_aux_diag_j[jS] != jj) { jS++; }
               S_aux_diag_data[jS] -= bval;
            }
            else
            {
               // neighbor is in offd part
               jj = hypre_BigBinarySearch( A_offd_colmap, big_col, num_cols_A_offd);
               jS = S_aux_offd_i[i];               
               while (S_aux_offd_j[jS] != jj) { jS++; }
               S_aux_offd_data[jS] -= bval;               
            }
        }
         // update diagonal entry
        S_aux_diag_data[S_aux_diag_i[i]] -= pval;          
      }
   }

   hypre_CSRMatrixNumNonzeros(S_aux_diag) = S_aux_diag_i[num_variables];
   hypre_CSRMatrixNumNonzeros(S_aux_offd) = S_aux_offd_i[num_variables];
   hypre_CSRMatrixJ(S_aux_diag) = S_aux_diag_j;
   hypre_CSRMatrixJ(S_aux_offd) = S_aux_offd_j;

   hypre_CSRMatrixMemoryLocation(S_aux_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_aux_offd) = memory_location;

//   hypre_ParCSRMatrixCommPkg(S_aux) = NULL;

   *S_aux_ptr = S_aux;

   hypre_TFree(P_i, memory_location);
   hypre_TFree(P_n, memory_location);
   hypre_TFree(B_i, memory_location);
   hypre_TFree(B_n, memory_location);
   
   return (ierr);
}

/*==========================================================================*/
/*==========================================================================*/
/**
  Generates an auxiilliary matrix, S_aux, with  M-matrix properties such that S_aux = A - B, 
  where B is obtained by distributing positive off-diagonal contributions to 'strong' neighbors.
  Here, strong neighbors are defined by the pattern of a strength matrix S.
  
  NOTE:: Current implementation assumes A is a square matrix with a regular partitioning of 
              row and columns.

  {\bf Input files:}
  _hypre_parcsr_ls.h

  @return Error code.

  @param A [IN]
  coefficient matrix
  @param S_ptr [OUT]
  strength matrix

  @see */
/*--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGCreateAuxMMatrixFromS(hypre_ParCSRMatrix    *A,
                           hypre_ParCSRMatrix    *S,
                           hypre_ParCSRMatrix   **S_aux_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_CREATES] -= hypre_MPI_Wtime();
#endif

   MPI_Comm                 comm     = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   hypre_CSRMatrix    *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j = hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   hypre_CSRMatrix    *A_offd   = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j = hypre_CSRMatrixJ(A_offd);

   /* diag part of S */
   hypre_CSRMatrix    *S_diag   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int          *S_diag_i = hypre_CSRMatrixI(S_diag);
   HYPRE_Int          *S_diag_j = hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   hypre_CSRMatrix    *S_offd   = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int          *S_offd_i = hypre_CSRMatrixI(S_offd);
   HYPRE_Int          *S_offd_j = hypre_CSRMatrixJ(S_offd);

   HYPRE_BigInt       *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int           num_variables   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_BigInt        global_num_vars = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int           num_cols_A_diag = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int           num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_Int           num_cols_S_diag = hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int           num_cols_S_offd = hypre_CSRMatrixNumCols(S_offd);

   HYPRE_BigInt   *A_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt   *A_offd_colmap = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_BigInt   *S_col_starts = hypre_ParCSRMatrixColStarts(S);
   HYPRE_BigInt   *S_offd_colmap = hypre_ParCSRMatrixColMapOffd(S);

   HYPRE_Int                A_diag_nnz = A_diag_i[num_variables];
   HYPRE_Int                A_offd_nnz = A_offd_i[num_variables];

   HYPRE_Int                S_diag_nnz = S_diag_i[num_variables];
   HYPRE_Int                S_offd_nnz = S_offd_i[num_variables];
   
   HYPRE_Int           num_nonzeros_diag;
   HYPRE_Int           num_nonzeros_offd = 0;
   HYPRE_Int           num_cols_offd = 0;

   hypre_ParCSRMatrix *S_aux;
   hypre_CSRMatrix    *S_aux_diag;
   HYPRE_Int          *S_aux_diag_i = NULL;
   HYPRE_Int          *S_aux_diag_j = NULL;
   HYPRE_Complex      *S_aux_diag_data = NULL;      
   /* HYPRE_Real         *S_diag_data; */
   hypre_CSRMatrix    *S_aux_offd;
   HYPRE_Int          *S_aux_offd_i = NULL;
   HYPRE_Int          *S_aux_offd_j = NULL;
   HYPRE_Complex      *S_aux_offd_data = NULL;   
   HYPRE_BigInt   *S_aux_offd_colmap = NULL;      
   /* HYPRE_Real         *S_offd_data; */
   /* off processor portions of S */
   hypre_CSRMatrix    *S_ext                 = NULL;
   HYPRE_Int          *S_ext_i               = NULL;
   HYPRE_Real         *S_ext_data            = NULL;
   HYPRE_BigInt       *S_ext_j               = NULL;
   HYPRE_Int                S_ext_nnz = 0;  
   
   HYPRE_Real          diag, row_scale, row_sum;
   HYPRE_Int           i,j,k,jj, jA, jS, kp, kn, m, n, P_n_max, N_n_max, cnt_n, cnt_p;

   HYPRE_Int           ierr = 0;

   HYPRE_Int          *dof_func_offd;
   HYPRE_Int           num_sends;
   HYPRE_Int          *int_buf_data;
   HYPRE_Int           index, start;

   HYPRE_Int          *P_i = NULL;
   HYPRE_Int          *P_n = NULL;   
   HYPRE_BigInt          *N_i = NULL;
   HYPRE_Int          *N_n = NULL;   
   HYPRE_BigInt          *B_i = NULL;   
   HYPRE_Int          *B_n = NULL;   
   
   HYPRE_Complex pval, bval;
   
   HYPRE_Int *prefix_sum_workspace;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);
   
   HYPRE_Int num_procs, my_id;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   
   // Begin:
   /* Allocate memory for auxilliary strength matrix */
   S_aux = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                row_starts, row_starts,
                                num_cols_A_offd, A_diag_nnz, A_offd_nnz);

   S_aux_diag = hypre_ParCSRMatrixDiag(S_aux);
   hypre_CSRMatrixI(S_aux_diag) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);
   hypre_CSRMatrixJ(S_aux_diag) = hypre_CTAlloc(HYPRE_Int, A_diag_nnz, memory_location);
   hypre_CSRMatrixData(S_aux_diag) = hypre_CTAlloc(HYPRE_Complex, A_diag_nnz, memory_location);

   S_aux_diag_i = hypre_CSRMatrixI(S_aux_diag);
   S_aux_diag_j = hypre_CSRMatrixJ(S_aux_diag);   
   S_aux_diag_data = hypre_CSRMatrixData(S_aux_diag); 

   S_aux_offd = hypre_ParCSRMatrixOffd(S_aux);
   hypre_CSRMatrixI(S_aux_offd) = hypre_CTAlloc(HYPRE_Int, num_variables + 1, memory_location);

   S_aux_offd_i = hypre_CSRMatrixI(S_aux_offd);


   if (num_cols_A_offd)
   {
      hypre_CSRMatrixJ(S_aux_offd) = hypre_CTAlloc(HYPRE_Int, A_offd_nnz, memory_location);
      hypre_CSRMatrixData(S_aux_offd) = hypre_CTAlloc(HYPRE_Complex, A_offd_nnz, memory_location);

      S_aux_offd_j = hypre_CSRMatrixJ(S_aux_offd);   
      S_aux_offd_data = hypre_CSRMatrixData(S_aux_offd); 

      S_aux_offd_colmap  = hypre_TAlloc(HYPRE_BigInt, num_cols_A_offd, memory_location);
      hypre_ParCSRMatrixColMapOffd(S_aux) = S_aux_offd_colmap;

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         S_aux_offd_colmap[i] = A_offd_colmap[i];
      }
   }
   // extract off-processor portion of A
   if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S, A, 0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixBigJ(S_ext);
      S_ext_nnz = S_ext_i[num_cols_A_offd]; 
      
   }
   /* allocate work arrays */
   HYPRE_Int P_nnz = S_diag_nnz + S_offd_nnz + num_variables;
   HYPRE_Int N_nnz = S_diag_nnz + S_offd_nnz + S_ext_nnz + num_cols_S_offd + num_variables;
   P_i    = hypre_CTAlloc(HYPRE_Int, P_nnz, memory_location);
   P_n    = hypre_CTAlloc(HYPRE_Int, (num_variables + 1), memory_location);
   N_i    = hypre_CTAlloc(HYPRE_BigInt, N_nnz, memory_location);
   N_n    = hypre_CTAlloc(HYPRE_Int, (num_variables + num_cols_A_offd + 1), memory_location);
      
   /* 1. Loop over matrix rows to extract positive and negative neighbors */
   P_n_max=0;
   N_n_max=0;
   kp = 0;
   kn = 0;
   for(i = 0; i<num_variables; i++)
   {
      cnt_p = kp;
      cnt_n = kn;
      // diagonal part
      // Check for positive off-diagonal entry
      for(j=A_diag_i[i]; j<A_diag_i[i+1]; j++)
      {
         if( (A_diag_data[j] > 0) && (A_diag_j[j] != i))
         {
            // position of positive neighbor. Store position for quick and easy access
            P_i[kp++] = j; //A_diag_j[j];
         }
      }
      // Off-diagonal part
      for(j=A_offd_i[i]; j<A_offd_i[i+1]; j++)
      {
         if( A_offd_data[j] > 0)
         {
            // position of positive neighbor. Offset to distinguish from diag part columns
            P_i[kp++] = j + A_diag_nnz;
         }
      }
      // Populate negative neighbor list from S. We need this so we can sort it.
      // diagonal part
      for(j=S_diag_i[i]; j<S_diag_i[i+1]; j++)
      {
         N_i[kn++] = S_diag_j[j] + S_col_starts[0];
      }
       // Off-diagonal part
      for(j=S_offd_i[i]; j<S_offd_i[i+1]; j++)
      {
         N_i[kn++] = S_offd_colmap[S_offd_j[j]]; 
      }     
            
      cnt_p = kp - cnt_p;
      cnt_n = kn - cnt_n;
      P_n_max = P_n_max > cnt_p ? P_n_max : cnt_p;      
      N_n_max = N_n_max > cnt_n ? N_n_max : cnt_n;
      // sort neighbor arrays
      hypre_BigQsort0(N_i, N_n[i], (kn-1));
      // update index pointer arrays
      P_n[i + 1] = kp;
      N_n[i + 1] = kn;
   }
   // add negative neighbors list of external rows
   kn = N_n[num_variables];
   for(i=0; i<num_cols_A_offd; i++)
   {
      for(j=S_ext_i[i]; j<S_ext_i[i+1]; j++)
      {
         N_i[kn++] = S_ext_j[j]; 
      }
      jj = num_variables + i;      
      // sort neighbor arrays
      hypre_BigQsort0(N_i, N_n[jj], (kn-1));
      // update index pointer arrays
      N_n[jj + 1] = kn;            
   }
   hypre_CSRMatrixDestroy(S_ext);   
   /* 2. Loop to compute intersections of strong (negative) neighbors of common weak (positive) entry */
   B_i    = hypre_CTAlloc(HYPRE_BigInt, (N_n_max * P_n[num_variables]), memory_location);
   B_n    = hypre_CTAlloc(HYPRE_Int, (P_n_max * num_variables + 1), memory_location);   
   kn = 0;
   kp = 0;
   for(i = 0; i<num_variables; i++)
   {
      // loop over positive neighbor list 
      for(j = P_n[i]; j< P_n[i+1]; j++)
      {
         jA = P_i[j] - A_diag_nnz;
         // Local row (assumes square matrix)
         if( jA < 0 )
         {
            k = A_diag_j[P_i[j]] ;
         }
         else
         {
            // external row. Access external variable range of N.
            k = A_offd_j[jA] + num_variables;           
         }
         // get intersection of N_i [i] and N_i[k]
         m = N_n[i + 1] - N_n[i];
         n = N_n[k + 1] - N_n[k];
         hypre_IntersectTwoBigIntegerArrays(&N_i[N_n[i]], m, &N_i[N_n[k]], n, &B_i[kn], &kp); 
         // update position of intersection of strong neighbors
         kn += kp;         
         B_n[j + 1] = kn;            
      }
   }
   hypre_TFree(N_i, memory_location);
   hypre_TFree(N_n, memory_location);
   /* 3. Fill in data for S_aux */
   kp = 0;
   kn = 0;
   for(i = 0; i<num_variables; i++)
   {
      // first insert diagonal entry 
      S_aux_diag_data[kp] = A_diag_data[A_diag_i[i] ];
      S_aux_diag_j[kp++] = A_diag_j[A_diag_i[i] ];    
      // diagonal part
      S_diag = hypre_ParCSRMatrixDiag(S);
      S_diag_i = hypre_CSRMatrixI(S_diag);
      S_diag_j = hypre_CSRMatrixJ(S_diag);
        
      for(j=S_diag_i[i]; j<S_diag_i[i+1]; j++)
      {
         jA = A_diag_i[i] + 1;
         jS = S_diag_j[j];
         while (A_diag_j[jA] != jS) { jA++; }     
         S_aux_diag_data[kp] = A_diag_data[jA];
         S_aux_diag_j[kp++] = A_diag_j[jA];                   
      }         
       // Off-diagonal part
      S_offd   = hypre_ParCSRMatrixOffd(S) ;
      S_offd_i = hypre_CSRMatrixI(S_offd);
      S_offd_j = hypre_CSRMatrixJ(S_offd);
     
      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         jA = A_offd_i[i];
         jS = S_offd_j[j];
         while (jS != A_offd_j[jA]) { jA++; }
         S_aux_offd_data[kn] = A_offd_data[jA];
         S_aux_offd_j[kn++] = A_offd_j[jA];
      }       
      // update index pointer arrays
      S_aux_diag_i[i + 1] = kp;
      S_aux_offd_i[i + 1] = kn;
   }     
   /* Distribute positive contributions to strong neighbors */
   for(i = 0; i<num_variables; i++)
   {   
      // loop over positive neighbors and compute contributions to (strong) negative neighbors
      for(j = P_n[i]; j< P_n[i+1]; j++)
      {
        jA = P_i[j] - A_diag_nnz;
         // positive entry is in diag part
         if( jA < 0 )
         {
            pval = A_diag_data[P_i[j]] ;
         }
         else
         {
            // positive entry is in offd part
            pval = A_offd_data[jA];           
         }

         // compute distribution to negative neighbors 
         bval = -2.0 * pval / (HYPRE_Complex) (B_n[ j+1 ] - B_n[ j ]);

         // Loop over negative connections to positive neighbor and distribute 
         for(k = B_n[ j ]; k<B_n[ j+1 ]; k++)
         {         
           HYPRE_BigInt big_col = B_i[k];
            if( big_col >= A_col_starts[0] && big_col < A_col_starts[1])
            {
               // neighbor is in diag part
               jS = S_aux_diag_i[i] + 1;
               jj = (HYPRE_Int) (big_col - A_col_starts[0]);
               while (S_aux_diag_j[jS] != jj) { jS++; }
               S_aux_diag_data[jS] -= bval;
            }
            else
            {
               // neighbor is in offd part
               jj = hypre_BigBinarySearch( A_offd_colmap, big_col, num_cols_A_offd);
               jS = S_aux_offd_i[i];               
               while (S_aux_offd_j[jS] != jj) { jS++; }
               S_aux_offd_data[jS] -= bval;               
            }
        }
         // update diagonal entry
        S_aux_diag_data[S_aux_diag_i[i]] -= pval;          
      }
   }

   hypre_CSRMatrixNumNonzeros(S_aux_diag) = S_aux_diag_i[num_variables];
   hypre_CSRMatrixNumNonzeros(S_aux_offd) = S_aux_offd_i[num_variables];
   hypre_CSRMatrixJ(S_aux_diag) = S_aux_diag_j;
   hypre_CSRMatrixJ(S_aux_offd) = S_aux_offd_j;

   hypre_CSRMatrixMemoryLocation(S_aux_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(S_aux_offd) = memory_location;

//   hypre_ParCSRMatrixCommPkg(S_aux) = NULL;

   *S_aux_ptr = S_aux;

   hypre_TFree(P_i, memory_location);
   hypre_TFree(P_n, memory_location);
   hypre_TFree(B_i, memory_location);
   hypre_TFree(B_n, memory_location);
   
   return (ierr);
}

HYPRE_Int hypre_BoomerAMGCreateAuxS(hypre_ParCSRMatrix    *A, hypre_ParCSRMatrix    *S, hypre_ParCSRMatrix   **S_aux_ptr, HYPRE_Int method)
{
   int ierr;
   if(method == 0)
   {
       ierr = hypre_BoomerAMGCreateAuxMMatrix(A, S_aux_ptr);      
   }
   else if (method == 1)
   {
      ierr =  hypre_BoomerAMGCreateAuxMMatrixFromS(A, S, S_aux_ptr);
   }
   else // default
   {
       ierr = hypre_BoomerAMGCreateAuxMMatrix(A, S_aux_ptr);         
   }
   
   return (ierr);
}


/* Compute the intersection of x and y, placing
 * the intersection in z.  Additionally, the array
 * x_data is associated with x, i.e., the entries
 * that we grab from x, we also grab from x_data.
 * If x[k] is placed in z[m], then x_data[k] goes to
 * output_x_data[m].
 *
 * Assumptions:
 *      z is of length min(x_length, y_length)
 *      x and y are sorted
 *      x_length and y_length are similar in size, otherwise,
 *          looping over the smaller array and doing binary search
 *          in the longer array is faster.
 * */
HYPRE_Int
hypre_IntersectTwoIntegerArrays(HYPRE_Int *x,
                         HYPRE_Int  x_length,
                         HYPRE_Int *y,
                         HYPRE_Int  y_length,
                         HYPRE_Int *z,
                         HYPRE_Int  *intersect_length)
{
   HYPRE_Int x_index = 0;
   HYPRE_Int y_index = 0;
   *intersect_length = 0;

   /* Compute Intersection, looping over each array */
   while ( (x_index < x_length) && (y_index < y_length) )
   {
      if (x[x_index] > y[y_index])
      {
         y_index = y_index + 1;
      }
      else if (x[x_index] < y[y_index])
      {
         x_index = x_index + 1;
      }
      else
      {
         z[*intersect_length] = x[x_index];
         x_index = x_index + 1;
         y_index = y_index + 1;
         *intersect_length = *intersect_length + 1;
      }
   }

   return 1;
}

HYPRE_Int
hypre_IntersectTwoBigIntegerArrays(HYPRE_BigInt *x,
                            HYPRE_Int  x_length,
                            HYPRE_BigInt *y,
                            HYPRE_Int  y_length,
                            HYPRE_BigInt *z,
                            HYPRE_Int  *intersect_length)
{
   HYPRE_Int x_index = 0;
   HYPRE_Int y_index = 0;
   *intersect_length = 0;

   /* Compute Intersection, looping over each array */
   while ( (x_index < x_length) && (y_index < y_length) )
   {
      if (x[x_index] > y[y_index])
      {
         y_index = y_index + 1;
      }
      else if (x[x_index] < y[y_index])
      {
         x_index = x_index + 1;
      }
      else
      {
         z[*intersect_length] = x[x_index];
         x_index = x_index + 1;
         y_index = y_index + 1;
         *intersect_length = *intersect_length + 1;
      }
   }

   return 1;
}

