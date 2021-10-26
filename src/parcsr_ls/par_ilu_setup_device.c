/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_ilu.h"
#include "seq_mv.hpp"


/*********************************************************************************/
/*                       hypre_ILUSetupILU0Device                                */
/*********************************************************************************/

/* ILU(0) (GPU)
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *    Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors. Note that with CUDA, Dptr and Uptr are unused
 * xtempp, ytempp = helper vector used in 2-level solve.
 * A_fake_diagp = fake diagonal for matvec
 * will form global Schur Matrix if nLU < n
 */

#if defined(HYPRE_USING_GPU)

HYPRE_Int
hypre_ILUSetupILU0Device(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *qperm,
                         HYPRE_Int n, HYPRE_Int nLU, hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                         hypre_CsrsvData **matBLU_csrsvdata_ptr, hypre_CsrsvData **matSLU_csrsvdata_ptr,
                         hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr,
                         hypre_CSRMatrix **Fptr, HYPRE_Int **A_fake_diag_ip, HYPRE_Int tri_solve)
{
   HYPRE_Int               i, j, k1, k2, k3, col;

   /* communication stuffs for S */
   MPI_Comm                comm                 = hypre_ParCSRMatrixComm(A);

   HYPRE_Int               my_id, num_procs;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_BigInt            *send_buf            = NULL;
   HYPRE_Int               *rperm               = NULL;
   HYPRE_Int               *rqperm              = NULL;
   hypre_ParCSRMatrix      *matS                = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   HYPRE_Int               *A_fake_diag_i       = NULL;
   hypre_CSRMatrix         *A_offd              = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;
   /* opaque pointers to vendor library data structs */
   hypre_CsrsvData         *matBLU_csrsvdata    = NULL;
   hypre_CsrsvData         *matSLU_csrsvdata    = NULL;

   /* variables for matS */
   HYPRE_Int               m                    = n - nLU;
   HYPRE_Int               nI                   = nLU;//use default
   HYPRE_Int               e                    = 0;
   HYPRE_Int               m_e                  = m;
   HYPRE_BigInt            total_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;

#ifdef HYPRE_USING_CUDA
   cudaEvent_t start, stop;
   float time;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
#endif

   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int A_offd_nnz                         = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int A_offd_n                           = hypre_CSRMatrixNumRows(A_offd);

   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (n > 0)
   {
      HYPRE_THRUST_CALL( sequence,
                         thrust::make_permutation_iterator(rperm, perm),
                         thrust::make_permutation_iterator(rperm+n, perm+n),
                         0 );
      HYPRE_THRUST_CALL( sequence,
                         thrust::make_permutation_iterator(rqperm, qperm),
                         thrust::make_permutation_iterator(rqperm+n, qperm+n),
                         0 );
   }
#else
   // not sure if this works
   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }
#endif

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_CSRMatrixApplyRowColPermutation(hypre_ParCSRMatrixDiag(A), perm, rqperm, &A_diag);

      /* Apply ILU factorization to the entile A_diag */
      HYPRE_ILUSetupDeviceCSRILU0(A_diag);

      /* | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */
      hypre_ParILUDeviceILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);
   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* create B */
   /* only analyse when nacessary */
   if ( nLU > 0 )
   {
      /* Solve Analysis of BILU */
      if (tri_solve)
      {
         HYPRE_ILUSetupDeviceCSRILU0SetupSolve(*BLUptr, matL_des, matU_des, &matBLU_csrsvdata);
      }
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

    /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }
      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      if (SLU)
      {
         /* Solve Analysis of SILU */
         if (tri_solve)
         {
            HYPRE_ILUSetupDeviceCSRILU0SetupSolve(SLU, matL_des, matU_des, &matSLU_csrsvdata);
         }
      }
      else
      {
         SLU = hypre_CSRMatrixCreate(0, 0, 0);
         hypre_CSRMatrixInitialize(SLU);
      }
      S_diag_nnz = hypre_CSRMatrixNumNonzeros(SLU);
      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
                                       total_rows,
                                       total_rows,
                                       col_starts,
                                       col_starts,
                                       S_offd_ncols,
                                       S_diag_nnz,
                                       S_offd_nnz);

      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);

      S_offd_i = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);

      HYPRE_Int * S_offd_i_host = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_HOST);
      HYPRE_Int * S_offd_j_host = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_HOST);
      HYPRE_Real * S_offd_data_host = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_HOST);

      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      HYPRE_Int * A_offd_i_host = hypre_TAlloc(HYPRE_Int, A_offd_n+1, HYPRE_MEMORY_HOST);
      HYPRE_Int * A_offd_j_host = hypre_TAlloc(HYPRE_Int, A_offd_nnz, HYPRE_MEMORY_HOST);
      HYPRE_Real * A_offd_data_host = hypre_TAlloc(HYPRE_Real, A_offd_nnz, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(A_offd_i_host, A_offd_i, HYPRE_Int, A_offd_n+1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_offd_j_host, A_offd_j, HYPRE_Int, A_offd_nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_offd_data_host, A_offd_data, HYPRE_Real, A_offd_nnz, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      HYPRE_Int * perm_host = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(perm_host, perm, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      /* simply use a loop to copy data from A_offd */
      S_offd_i_host[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i_host[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm_host[i + nI];
         k1 = A_offd_i_host[col];
         k2 = A_offd_i_host[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j_host[k3] = A_offd_j_host[j];
            S_offd_data_host[k3++] = A_offd_data_host[j];
         }
         S_offd_i_host[i + 1 + e] = k3;
      }
      hypre_TFree(perm_host, HYPRE_MEMORY_HOST);
      hypre_TFree(A_offd_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(A_offd_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(A_offd_data_host, HYPRE_MEMORY_HOST);

      /* give I, J, DATA to S_offd */
      hypre_TMemcpy(S_offd_i,     S_offd_i_host, HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(S_offd_j,     S_offd_j_host, HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(S_offd_data,  S_offd_data_host, HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(S_offd_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(S_offd_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(S_offd_data_host, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);

      /* copy new index into send_buf */
      HYPRE_Int * rperm_host = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(rperm_host, rperm, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm_host[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
      }
      hypre_TFree(rperm_host, HYPRE_MEMORY_HOST);

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }
   else {
       /** need to clean this up here potentially if its empty */
       if (hypre_CSRMatrixNumRows(SLU)==0 && hypre_CSRMatrixNumNonzeros(SLU)==0) {
           hypre_CSRMatrixDestroy( SLU );
           SLU = NULL;
       }
   }

   /* end of forming S */
   *matSptr       = matS;
   *matBLU_csrsvdata_ptr = matBLU_csrsvdata;
   *matSLU_csrsvdata_ptr = matSLU_csrsvdata;
   *A_fake_diag_ip = A_fake_diag_i;

   /* Destroy the bridge after acrossing the river */
   hypre_CSRMatrixDestroy(A_diag);
   hypre_TFree(rperm, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rqperm, HYPRE_MEMORY_DEVICE);


#ifdef HYPRE_USING_CUDA
   cudaEventRecord( stop, 0 );
   cudaEventSynchronize( stop );
   cudaEventElapsedTime( &time, start, stop );
   printf("%s %s %d : time=%1.5g\n",__FILE__,__FUNCTION__,__LINE__,time/1000.);
   cudaEventDestroy( start );
   cudaEventDestroy( stop );
#endif
   return hypre_error_flag;
}


/*********************************************************************************/
/*                   HYPRE_ILUSetupCusparseCSRILU0                               */
/*********************************************************************************/

HYPRE_Int HYPRE_ILUSetupDeviceCSRILU0(hypre_CSRMatrix *A)
{
#if defined(HYPRE_USING_CUSPARSE)
    HYPRE_ILUSetupCusparseCSRILU0(A, CUSPARSE_SOLVE_POLICY_USE_LEVEL);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
    HYPRE_ILUSetupRocsparseCSRILU0(A, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto);
#endif
   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)

/* Wrapper for ILU0 with cusparse on a matrix, csr sort was done in this function */
HYPRE_Int
HYPRE_ILUSetupCusparseCSRILU0(hypre_CSRMatrix *A,
                              cusparseSolvePolicy_t ilu_solve_policy)
{

   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A);

   /* pointers to cusparse data */
   csrilu02Info_t          matA_info            = NULL;

   /* variables and working arrays used during the ilu */
   HYPRE_Int               zero_pivot;
   HYPRE_Int               matA_buffersize;
   void                    *matA_buffer         = NULL;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);

   /* 1. Sort columns inside each row first, we can't assume that's sorted */
   hypre_SortCSRCusparse(n, m, nnz_A, descr, A_i, A_j, A_data);

   /* 2. Create info for ilu setup and solve */
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));

   /* 3. Get working array size */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_bufferSize(handle, n, nnz_A, descr,
														  A_data, A_i, A_j,
														  matA_info, &matA_buffersize));

   /* 4. Create working array, since they won't be visited by host, allocate on device */
   matA_buffer                                  = hypre_MAlloc(matA_buffersize, HYPRE_MEMORY_DEVICE);

   /* 5. Now perform the analysis */
   /* 5-1. Analysis */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_analysis(handle, n, nnz_A, descr,
														A_data, A_i, A_j,
														matA_info, ilu_solve_policy, matA_buffer));

   /* 5-2. Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* 6. Apply the factorization */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02(handle, n, nnz_A, descr,
											   A_data, A_i, A_j,
											   matA_info, ilu_solve_policy, matA_buffer));

   /* Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* Done with factorization, finishing up */
   hypre_TFree(matA_buffer, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrilu02Info(matA_info));

   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
HYPRE_ILUSetupRocsparseCSRILU0(hypre_CSRMatrix *A, rocsparse_analysis_policy analysis_policy, rocsparse_solve_policy solve_policy)
{
   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A);

   /* pointers to cusparse data */
   rocsparse_mat_info info;

   /* variables and working arrays used during the ilu */
   HYPRE_Int               zero_pivot;
   size_t                  buffer_size;
   void                    *buffer         = NULL;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(A);

   /* 1. Sort columns inside each row first, we can't assume that's sorted */
   hypre_SortCSRRocsparse(n, m, nnz_A, descr, A_i, A_j, A_data);

   /* 2. Create info for ilu setup and solve */
   HYPRE_ROCSPARSE_CALL(rocsparse_create_mat_info(&info));

   /* 3. Get working array size */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_buffer_size(handle, n, nnz_A, descr, A_data, A_i, A_j,
															info, &buffer_size));

   /* 4. Create working array, since they won't be visited by host, allocate on device */
   buffer = hypre_MAlloc(buffer_size, HYPRE_MEMORY_DEVICE);

   /* 5. Now perform the analysis */
   /* 5-1. Analysis */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_analysis(handle,  n, nnz_A, descr, A_data, A_i, A_j,
														 info, analysis_policy, solve_policy, buffer));

   /* 5-2. Check for zero pivot */
   HYPRE_ROCSPARSE_CALL(rocsparse_csrsv_zero_pivot(handle, descr, info, &zero_pivot));

   /* 6. Apply the factorization */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0(handle, n, nnz_A, descr, A_data, A_i, A_j,
												info, solve_policy, buffer));

   /* Check for zero pivot */
   HYPRE_ROCSPARSE_CALL(rocsparse_csrsv_zero_pivot(handle, descr, info, &zero_pivot));

   /* Done with factorization, finishing up */
   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);
   HYPRE_ROCSPARSE_CALL(rocsparse_destroy_mat_info(info));

   return hypre_error_flag;

}

#endif




/*********************************************************************************/
/*                   HYPRE_ILUSetupCusparseCSRILU0SetupSolve                     */
/*********************************************************************************/

HYPRE_Int HYPRE_ILUSetupDeviceCSRILU0SetupSolve(hypre_CSRMatrix *A, hypre_GpuMatData * matL_des,
                                                hypre_GpuMatData * matU_des, hypre_CsrsvData ** matLU_csrsvdata_ptr)
{
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_ILUSetupCusparseCSRILU0SetupSolve(A, matL_des, matU_des, matLU_csrsvdata_ptr);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ILUSetupRocsparseCSRILU0SetupSolve(A, matL_des, matU_des, matLU_csrsvdata_ptr);
#endif
   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_CUSPARSE)

/* Wrapper for ILU0 solve analysis phase with cusparse on a matrix */
HYPRE_Int
HYPRE_ILUSetupCusparseCSRILU0SetupSolve(hypre_CSRMatrix *A, hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                                        hypre_CsrsvData ** matLU_csrsvdata_ptr)
{

    if (!A)
   {
      /* return if A is NULL */
      *matLU_csrsvdata_ptr    = NULL;
      return hypre_error_flag;
   }

   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   if (n == 0)
   {
      /* return if A is 0 by 0 */
      *matLU_csrsvdata_ptr    = NULL;
      return hypre_error_flag;
   }

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A);

   /* variables and working arrays used during the ilu */
   HYPRE_Int               matL_buffersize;
   HYPRE_Int               matU_buffersize;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /************************************/
   /* Create these data structures now */
   if (*matLU_csrsvdata_ptr) {
       hypre_CsrsvDataDestroy(*matLU_csrsvdata_ptr);
   }
   hypre_CsrsvData * matLU_csrsvdata           = hypre_CsrsvDataCreate();
   cusparseSolvePolicy_t ilu_solve_policy      = hypre_CsrsvDataSolvePolicy(matLU_csrsvdata);
   csrsv2Info_t matL_info = hypre_CsrsvDataInfoL(matLU_csrsvdata);
   csrsv2Info_t matU_info = hypre_CsrsvDataInfoU(matLU_csrsvdata);
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&matL_info));
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&matU_info));
   size_t solve_buffersize;
   char * solve_buffer;

   /* 2. Get working array size */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
														hypre_GpuMatDataMatDescr(matL_des), A_data, A_i, A_j,
														matL_info, &matL_buffersize));

   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
														hypre_GpuMatDataMatDescr(matU_des), A_data, A_i, A_j,
														matU_info, &matU_buffersize));

   /* 3. Create working array, since they won't be visited by host, allocate on device */
   solve_buffersize = hypre_max( matL_buffersize, matU_buffersize );
   solve_buffer     = (char *)hypre_MAlloc(solve_buffersize, HYPRE_MEMORY_DEVICE);

   /* 4. Now perform the analysis */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													  n, nnz_A, hypre_GpuMatDataMatDescr(matL_des),
													  A_data, A_i, A_j,
													  matL_info, ilu_solve_policy, solve_buffer));

   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													  n, nnz_A, hypre_GpuMatDataMatDescr(matU_des),
													  A_data, A_i, A_j,
													  matU_info, ilu_solve_policy, solve_buffer));

   /* Done with analysis, finishing up */
   /* Set return value */
   hypre_CsrsvDataInfoL(matLU_csrsvdata) =  matL_info;
   hypre_CsrsvDataInfoU(matLU_csrsvdata) =  matU_info;
   hypre_CsrsvDataBufferSize(matLU_csrsvdata) = solve_buffersize;
   hypre_CsrsvDataBuffer(matLU_csrsvdata) = solve_buffer;
   *matLU_csrsvdata_ptr    = matLU_csrsvdata;
   return hypre_error_flag;
}

#endif


#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
HYPRE_ILUSetupRocsparseCSRILU0SetupSolve(hypre_CSRMatrix *A, hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                                         hypre_CsrsvData ** matLU_csrsvdata_ptr)
{
   if (!A)
   {
      /* return if A is NULL */
      *matLU_csrsvdata_ptr    = NULL;
      return hypre_error_flag;
   }

   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   if (n == 0)
   {
      /* return if A is 0 by 0 */
      *matLU_csrsvdata_ptr    = NULL;
      return hypre_error_flag;
   }

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A);

   /* variables and working arrays used during the ilu */
   size_t               matL_buffersize;
   size_t               matU_buffersize;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
   /************************************/
   /* Create these data structures now */
   if (*matLU_csrsvdata_ptr) {
       hypre_CsrsvDataDestroy(*matLU_csrsvdata_ptr);
   }
   hypre_CsrsvData * matLU_csrsvdata             = hypre_CsrsvDataCreate();
   rocsparse_analysis_policy ilu_analysis_policy = hypre_CsrsvDataAnalysisPolicy(matLU_csrsvdata);
   rocsparse_solve_policy ilu_solve_policy       = hypre_CsrsvDataSolvePolicy(matLU_csrsvdata);
   rocsparse_mat_info matL_info = hypre_CsrsvDataInfoL(matLU_csrsvdata);
   rocsparse_mat_info matU_info = hypre_CsrsvDataInfoU(matLU_csrsvdata);
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&matL_info) );
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&matU_info) );
   size_t solve_buffersize;
   char * solve_buffer;

   /* 2. Get working array size */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrsv_buffer_size(handle, rocsparse_operation_none, n, nnz_A,
														  hypre_GpuMatDataMatDescr(matL_des), A_data, A_i, A_j,
														  matL_info, &matL_buffersize));

   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrsv_buffer_size(handle, rocsparse_operation_none, n, nnz_A,
														  hypre_GpuMatDataMatDescr(matU_des), A_data, A_i, A_j,
														  matU_info, &matU_buffersize));

   /* 3. Create working array, since they won't be visited by host, allocate on device */
   solve_buffersize = hypre_max( matL_buffersize, matU_buffersize );
   solve_buffer     = (char *)hypre_MAlloc(solve_buffersize, HYPRE_MEMORY_DEVICE);

   /* 4. Now perform the analysis */
   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
														n, nnz_A, hypre_GpuMatDataMatDescr(matL_des),
														A_data, A_i, A_j, matL_info,
														ilu_analysis_policy, ilu_solve_policy, solve_buffer));

   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
														n, nnz_A, hypre_GpuMatDataMatDescr(matU_des),
														A_data, A_i, A_j, matU_info,
														ilu_analysis_policy, ilu_solve_policy, solve_buffer));

   /* Done with analysis, finishing up */
   /* Set return value */
   hypre_CsrsvDataInfoL(matLU_csrsvdata) =  matL_info;
   hypre_CsrsvDataInfoU(matLU_csrsvdata) =  matU_info;
   hypre_CsrsvDataBufferSize(matLU_csrsvdata) = solve_buffersize;
   hypre_CsrsvDataBuffer(matLU_csrsvdata) = solve_buffer;
   *matLU_csrsvdata_ptr    = matLU_csrsvdata;

   return hypre_error_flag;
}

#endif






/* Extract submatrix from diagonal part of A into a
 * | B F |
 * | E C |
 * Struct in order to do ILU with cusparse.
 * WARNING: Cusparse requires each row been sorted by column
 *          This function only works when rows are sorted!.
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * Bp = pointer to the output B matrix.
 * Cp = pointer to the output C matrix.
 * Ep = pointer to the output E matrix.
 * Fp = pointer to the output F matrix.
 */
/*********************************************************************************/
/*                   hypre_ParILUCusparseILUExtractEBFC                          */
/*********************************************************************************/

HYPRE_Int hypre_ParILUDeviceILUExtractEBFC(hypre_CSRMatrix *A_diag, HYPRE_Int nLU,
                                           hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp, hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp)
{
   /* Get necessary slots */
   HYPRE_Int            n              = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            nnz_A_diag     = hypre_CSRMatrixNumNonzeros(A_diag);

#if (defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)) || !defined(HYPRE_USING_GPU)
   HYPRE_Int           *A_diag_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real          *A_diag_data    = hypre_CSRMatrixData(A_diag);
#elif defined(HYPRE_USING_GPU)
   /* move the data back to the host */
   HYPRE_Int            *A_diag_i = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   HYPRE_Int            *A_diag_j = hypre_CTAlloc(HYPRE_Int, nnz_A_diag, HYPRE_MEMORY_HOST);
   HYPRE_Real            *A_diag_data = hypre_CTAlloc(HYPRE_Real, nnz_A_diag, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(A_diag_i,     hypre_CSRMatrixI(A_diag), HYPRE_Int, n+1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(A_diag_j,     hypre_CSRMatrixJ(A_diag), HYPRE_Int, nnz_A_diag, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(A_diag_data,  hypre_CSRMatrixData(A_diag), HYPRE_Real, nnz_A_diag, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
#else
#error "This can't happen!"
#endif

   HYPRE_Int            i, j, row, col;

   hypre_assert(nLU >= 0 && nLU <= n);
   if (nLU == n)
   {
      /* No schur complement makes everything easy :) */
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      B                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(B);
      hypre_CSRMatrixCopy(A_diag, B, 1);

      /* What is the point of this ? PJM 4/8/2022 
       Commenting out to avoid memory leaks. */
      C                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(C);
      E                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(F);
      *Bp = B;
      *Cp = C;
      *Ep = E;
      *Fp = F;
   }
   else if (nLU == 0)
   {
      /* All schur complement also makes everything easy :) */
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      C                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(C);
      hypre_CSRMatrixCopy(A_diag, C, 1);
      /* What is the point of this ? PJM 4/8/2022 
       Commenting out to avoid memory leaks. */
      B                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(B);
      E                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(F);
      *Bp = B;
      *Cp = C;
      *Ep = E;
      *Fp = F;
   }
   else
   {
      /* Has schur complement :( */
      HYPRE_Int         m              = n - nLU;
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      HYPRE_Int         capacity_B;
      HYPRE_Int         capacity_E;
      HYPRE_Int         capacity_F;
      HYPRE_Int         capacity_C;
      HYPRE_Int         ctrB;
      HYPRE_Int         ctrC;
      HYPRE_Int         ctrE;
      HYPRE_Int         ctrF;

      HYPRE_Int        *B_i            = NULL;
      HYPRE_Int        *C_i            = NULL;
      HYPRE_Int        *E_i            = NULL;
      HYPRE_Int        *F_i            = NULL;
      HYPRE_Int        *B_j            = NULL;
      HYPRE_Int        *C_j            = NULL;
      HYPRE_Int        *E_j            = NULL;
      HYPRE_Int        *F_j            = NULL;
      HYPRE_Real       *B_data         = NULL;
      HYPRE_Real       *C_data         = NULL;
      HYPRE_Real       *E_data         = NULL;
      HYPRE_Real       *F_data         = NULL;

      /* Create CSRMatrices */
      B                                = hypre_CSRMatrixCreate(nLU, nLU, 0);
      hypre_CSRMatrixInitialize(B);
      C                                = hypre_CSRMatrixCreate(m, m, 0);
      hypre_CSRMatrixInitialize(C);
      E                                = hypre_CSRMatrixCreate(m, nLU, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(nLU, m, 0);
      hypre_CSRMatrixInitialize(F);

      /* Estimate # of nonzeros */
      capacity_B                       = nLU + ceil(nnz_A_diag * 1.0 * nLU / n * nLU / n);
      capacity_C                       = m + ceil(nnz_A_diag * 1.0 * m / n * m / n);
      capacity_E                       = hypre_min(m, nLU) + ceil(nnz_A_diag * 1.0 * nLU / n * m / n);
      capacity_F                       = capacity_E;

      /* Allocate memory */
      ctrB                             = 0;
      ctrC                             = 0;
      ctrE                             = 0;
      ctrF                             = 0;

      /* Host arrays */
      HYPRE_Int * B_i_host                         = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(B)+1, HYPRE_MEMORY_HOST);
      HYPRE_Int * B_j_host                         = hypre_CTAlloc(HYPRE_Int, capacity_B, HYPRE_MEMORY_HOST);
      HYPRE_Real *B_data_host                      = hypre_CTAlloc(HYPRE_Real, capacity_B, HYPRE_MEMORY_HOST);
      HYPRE_Int * C_i_host                         = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(C)+1, HYPRE_MEMORY_HOST);
      HYPRE_Int * C_j_host                         = hypre_CTAlloc(HYPRE_Int, capacity_C, HYPRE_MEMORY_HOST);
      HYPRE_Real *C_data_host                      = hypre_CTAlloc(HYPRE_Real, capacity_C, HYPRE_MEMORY_HOST);
      HYPRE_Int * E_i_host                         = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(E)+1, HYPRE_MEMORY_HOST);
      HYPRE_Int * E_j_host                         = hypre_CTAlloc(HYPRE_Int, capacity_E, HYPRE_MEMORY_HOST);
      HYPRE_Real *E_data_host                      = hypre_CTAlloc(HYPRE_Real, capacity_E, HYPRE_MEMORY_HOST);
      HYPRE_Int * F_i_host                         = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(F)+1, HYPRE_MEMORY_HOST);
      HYPRE_Int * F_j_host                         = hypre_CTAlloc(HYPRE_Int, capacity_F, HYPRE_MEMORY_HOST);
      HYPRE_Real * F_data_host                     = hypre_CTAlloc(HYPRE_Real, capacity_F, HYPRE_MEMORY_HOST);

      /* Loop to copy data */
      /* B and F first */
      for (i = 0; i < nLU; i++)
      {
         B_i_host[i]   = ctrB;
         F_i_host[i]   = ctrF;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            B_j_host[ctrB] = col;
            B_data_host[ctrB++] = A_diag_data[j];
            /* check capacity */
            if (ctrB >= capacity_B)
            {
               HYPRE_Int tmp;
               tmp = capacity_B;
               capacity_B = capacity_B * EXPAND_FACT + 1;
               B_j_host = hypre_TReAlloc_v2(B_j_host, HYPRE_Int, tmp, HYPRE_Int, capacity_B, HYPRE_MEMORY_HOST);
               B_data_host = hypre_TReAlloc_v2(B_data_host, HYPRE_Real, tmp, HYPRE_Real, capacity_B, HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            F_j_host[ctrF] = col;
            F_data_host[ctrF++] = A_diag_data[j];
            if (ctrF >= capacity_F)
            {
               HYPRE_Int tmp;
               tmp = capacity_F;
               capacity_F = capacity_F * EXPAND_FACT + 1;
               F_j_host = hypre_TReAlloc_v2(F_j_host, HYPRE_Int, tmp, HYPRE_Int, capacity_F, HYPRE_MEMORY_HOST);
               F_data_host = hypre_TReAlloc_v2(F_data_host, HYPRE_Real, tmp, HYPRE_Real, capacity_F, HYPRE_MEMORY_HOST);
            }
         }
      }
      B_i_host[nLU] = ctrB;
      F_i_host[nLU] = ctrF;

      /* E and C afterward */
      for (i = nLU; i < n; i++)
      {
         row = i - nLU;
         E_i_host[row] = ctrE;
         C_i_host[row] = ctrC;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            E_j_host[ctrE] = col;
            E_data_host[ctrE++] = A_diag_data[j];
            /* check capacity */
            if (ctrE >= capacity_E)
            {
               HYPRE_Int tmp;
               tmp = capacity_E;
               capacity_E = capacity_E * EXPAND_FACT + 1;
               E_j_host = hypre_TReAlloc_v2(E_j_host, HYPRE_Int, tmp, HYPRE_Int, capacity_E, HYPRE_MEMORY_HOST);
               E_data_host = hypre_TReAlloc_v2(E_data_host, HYPRE_Real, tmp, HYPRE_Real, capacity_E, HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            C_j_host[ctrC] = col;
            C_data_host[ctrC++] = A_diag_data[j];
            if (ctrC >= capacity_C)
            {
               HYPRE_Int tmp;
               tmp = capacity_C;
               capacity_C = capacity_C * EXPAND_FACT + 1;
               C_j_host = hypre_TReAlloc_v2(C_j_host, HYPRE_Int, tmp, HYPRE_Int, capacity_C, HYPRE_MEMORY_HOST);
               C_data_host = hypre_TReAlloc_v2(C_data_host, HYPRE_Real, tmp, HYPRE_Real, capacity_C, HYPRE_MEMORY_HOST);
            }
         }
      }
      E_i_host[m] = ctrE;
      C_i_host[m] = ctrC;

      hypre_assert((ctrB + ctrC + ctrE + ctrF) == nnz_A_diag);

      B_i                              = hypre_CSRMatrixI(B);
      B_j                              = hypre_CTAlloc(HYPRE_Int, ctrB, HYPRE_MEMORY_DEVICE);
      B_data                           = hypre_CTAlloc(HYPRE_Real, ctrB, HYPRE_MEMORY_DEVICE);
      C_i                              = hypre_CSRMatrixI(C);
      C_j                              = hypre_CTAlloc(HYPRE_Int, ctrC, HYPRE_MEMORY_DEVICE);
      C_data                           = hypre_CTAlloc(HYPRE_Real, ctrC, HYPRE_MEMORY_DEVICE);
      E_i                              = hypre_CSRMatrixI(E);
      E_j                              = hypre_CTAlloc(HYPRE_Int, ctrE, HYPRE_MEMORY_DEVICE);
      E_data                           = hypre_CTAlloc(HYPRE_Real, ctrE, HYPRE_MEMORY_DEVICE);
      F_i                              = hypre_CSRMatrixI(F);
      F_j                              = hypre_CTAlloc(HYPRE_Int, ctrF, HYPRE_MEMORY_DEVICE);
      F_data                           = hypre_CTAlloc(HYPRE_Real, ctrF, HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(B_i, B_i_host, HYPRE_Int, hypre_CSRMatrixNumRows(B)+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(B_j, B_j_host, HYPRE_Int, ctrB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(B_data, B_data_host, HYPRE_Real, ctrB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(C_i, C_i_host, HYPRE_Int, hypre_CSRMatrixNumRows(C)+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(C_j, C_j_host, HYPRE_Int, ctrC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(C_data, C_data_host, HYPRE_Real, ctrC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(E_i, E_i_host, HYPRE_Int, hypre_CSRMatrixNumRows(E)+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(E_j, E_j_host, HYPRE_Int, ctrE, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(E_data, E_data_host, HYPRE_Real, ctrE, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(F_i, F_i_host, HYPRE_Int, hypre_CSRMatrixNumRows(F)+1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(F_j, F_j_host, HYPRE_Int, ctrF, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(F_data, F_data_host, HYPRE_Real, ctrF, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

      /* Create CSRMatrices */
      hypre_CSRMatrixJ(B)              = B_j;
      hypre_CSRMatrixData(B)           = B_data;
      hypre_CSRMatrixNumNonzeros(B)    = ctrB;
      hypre_CSRMatrixSetDataOwner(B, 1);
      *Bp                              = B;

      hypre_CSRMatrixJ(C)              = C_j;
      hypre_CSRMatrixData(C)           = C_data;
      hypre_CSRMatrixNumNonzeros(C)    = ctrC;
      hypre_CSRMatrixSetDataOwner(C, 1);
      *Cp                              = C;

      hypre_CSRMatrixJ(E)              = E_j;
      hypre_CSRMatrixData(E)           = E_data;
      hypre_CSRMatrixNumNonzeros(E)    = ctrE;
      hypre_CSRMatrixSetDataOwner(E, 1);
      *Ep                              = E;

      hypre_CSRMatrixJ(F)              = F_j;
      hypre_CSRMatrixData(F)           = F_data;
      hypre_CSRMatrixNumNonzeros(F)    = ctrF;
      hypre_CSRMatrixSetDataOwner(F, 1);
      *Fp                              = F;

      hypre_TFree(B_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(B_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(B_data_host, HYPRE_MEMORY_HOST);
      hypre_TFree(C_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(C_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(C_data_host, HYPRE_MEMORY_HOST);
      hypre_TFree(E_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(E_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(E_data_host, HYPRE_MEMORY_HOST);
      hypre_TFree(F_i_host, HYPRE_MEMORY_HOST);
      hypre_TFree(F_j_host, HYPRE_MEMORY_HOST);
      hypre_TFree(F_data_host, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_TFree(A_diag_i, HYPRE_MEMORY_HOST);
   hypre_TFree(A_diag_j, HYPRE_MEMORY_HOST);
   hypre_TFree(A_diag_data, HYPRE_MEMORY_HOST);
#endif
   return hypre_error_flag;
}
