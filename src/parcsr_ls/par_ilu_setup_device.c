/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_ilu.h"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILU0Device
 *
 * ILU(0) setup on the device
 *
 * Arguments:
 *    A = input matrix
 *    perm_data  = permutation array indicating ordering of rows.
 *                 Could come from a CF_marker array or a reordering routine.
 *    qperm_data = permutation array indicating ordering of columns
 *    nI  = number of internal unknowns
 *    nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *          Schur complement is formed if nLU < n
 *    Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 *                             Note that with CUDA, Dptr and Uptr are unused
 *    A_fake_diagp = fake diagonal for matvec
 *
 * This function will form the global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILU0Device(hypre_ParCSRMatrix *A,
                         HYPRE_Int *perm_data,
                         HYPRE_Int *qperm_data,
                         HYPRE_Int n,
                         HYPRE_Int nLU,
                         hypre_GpuMatData  *matL_des,
                         hypre_GpuMatData  *matU_des,
                         hypre_CsrsvData  **matBLU_csrsvdata_ptr,
                         hypre_CsrsvData  **matSLU_csrsvdata_ptr,
                         hypre_CSRMatrix  **BLUptr,
                         hypre_ParCSRMatrix **matSptr,
                         hypre_CSRMatrix **Eptr,
                         hypre_CSRMatrix **Fptr,
                         HYPRE_Int **A_fake_diag_ip,
                         HYPRE_Int tri_solve)
{
   /* Input matrix data */
   MPI_Comm                 comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                num_sends, begin, end;
   HYPRE_BigInt            *send_buf      = NULL;

   hypre_ParCSRMatrix      *matS          = NULL;
   hypre_CSRMatrix         *A_diag        = NULL;
   HYPRE_Int               *A_fake_diag_i = NULL;
   hypre_CSRMatrix         *A_offd        = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix         *h_A_offd      = NULL;
   HYPRE_Int               *A_offd_i      = NULL;
   HYPRE_Int               *A_offd_j      = NULL;
   HYPRE_Real              *A_offd_data   = NULL;
   hypre_CSRMatrix         *SLU           = NULL;

   /* Pointers to vendor library data structures  */
   hypre_CsrsvData         *matBLU_csrsvdata    = NULL;
   hypre_CsrsvData         *matSLU_csrsvdata    = NULL;

   /* Permutation arrays */
   HYPRE_Int               *rperm_data    = NULL;
   HYPRE_Int               *rqperm_data   = NULL;
   hypre_IntArray          *perm          = NULL;
   hypre_IntArray          *rperm         = NULL;
   hypre_IntArray          *qperm         = NULL;
   hypre_IntArray          *rqperm        = NULL;
   hypre_IntArray          *h_perm        = NULL;
   hypre_IntArray          *h_rperm       = NULL;

   /* Variables for matS */
   HYPRE_Int                m             = n - nLU;
   HYPRE_Int                nI            = nLU; //use default
   HYPRE_Int                e             = 0;
   HYPRE_Int                m_e           = m;
   HYPRE_Int               *S_diag_i      = NULL;
   hypre_CSRMatrix         *S_offd        = NULL;
   HYPRE_Int               *S_offd_i      = NULL;
   HYPRE_Int               *S_offd_j      = NULL;
   HYPRE_Real              *S_offd_data   = NULL;
   HYPRE_BigInt            *S_offd_colmap = NULL;
   HYPRE_Int                S_offd_nnz;
   HYPRE_Int                S_offd_ncols;
   HYPRE_Int                S_diag_nnz;

   /* Local variables */
   HYPRE_BigInt             total_rows, col_starts[2];
   HYPRE_Int                i, j, k1, k2, k3, col;
   HYPRE_Int                my_id, num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Build the inverse permutation arrays */
   if (perm_data && qperm_data)
   {
      /* Create arrays */
      perm   = hypre_IntArrayCreate(n);
      qperm  = hypre_IntArrayCreate(n);

      /* Set existing data */
      hypre_IntArrayData(perm)  = perm_data;
      hypre_IntArrayData(qperm) = qperm_data;

      /* Initialize arrays */
      hypre_IntArrayInitialize_v2(perm, memory_location);
      hypre_IntArrayInitialize_v2(qperm, memory_location);

      /* Compute inverse permutation arrays */
      hypre_IntArrayInverseMapping(perm, &rperm);
      hypre_IntArrayInverseMapping(qperm, &rqperm);

      rqperm_data = hypre_IntArrayData(rqperm);
   }

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_CSRMatrixPermute(hypre_ParCSRMatrixDiag(A), perm_data, rqperm_data, &A_diag);

      /* Apply ILU factorization to the entire A_diag */
      hypre_ILUSetupILU0LocalDevice(A_diag);

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

   /* Create B - only analyze when necessary */
   if ((nLU > 0) && tri_solve)
   {
      /* Solve Analysis of BILU */
      HYPRE_ILUSetupDeviceCSRILU0SetupSolve(*BLUptr, matL_des, matU_des, &matBLU_csrsvdata);
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt) m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* Only form S when total_rows > 0 */
   if (total_rows > 0)
   {
      /* Create S - need to get new column start */
      {
         HYPRE_BigInt global_start;

         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - big_m;
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

      S_diag_i = hypre_CSRMatrixI(SLU);
      hypre_TMemcpy(&S_diag_nnz, S_diag_i + m, HYPRE_Int, 1,
                    HYPRE_MEMORY_HOST, hypre_CSRMatrixMemoryLocation(SLU));

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

      matS = hypre_ParCSRMatrixCreate(comm,
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
      hypre_CSRMatrixInitialize_v2(S_offd, 0, HYPRE_MEMORY_HOST);
      S_offd_i = hypre_CSRMatrixI(S_offd);
      S_offd_j = hypre_CSRMatrixJ(S_offd);
      S_offd_data = hypre_CSRMatrixData(S_offd);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* Set/Move A_offd to host */
      h_A_offd = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
                 hypre_CSRMatrixClone_v2(A_offd, 1, HYPRE_MEMORY_HOST) : A_offd;
      A_offd_i    = hypre_CSRMatrixI(h_A_offd);
      A_offd_j    = hypre_CSRMatrixJ(h_A_offd);
      A_offd_data = hypre_CSRMatrixData(h_A_offd);

      /* Clone permutation arrays on the host */
      if (rperm && perm)
      {
         h_perm  = hypre_IntArrayCloneDeep_v2(perm, HYPRE_MEMORY_HOST);
         h_rperm = hypre_IntArrayCloneDeep_v2(rperm, HYPRE_MEMORY_HOST);

         perm_data  = hypre_IntArrayData(h_perm);
         rperm_data = hypre_IntArrayData(h_rperm);
      }

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = (perm_data) ? perm_data[i + nI] : i + nI;
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + 1 + e] = k3;
      }

      /* give I, J, DATA to S_offd */
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
      for (i = 0; i < (end - begin); i++)
      {
         send_buf[i] = (rperm_data) ?
                       rperm_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i + begin)] -
                       nLU + col_starts[0] :
                       hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i + begin) -
                       nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* Move S_offd to final memory location */
      hypre_CSRMatrixMigrate(S_offd, memory_location);

      /* Free memory */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
      if (h_A_offd != A_offd)
      {
         hypre_CSRMatrixDestroy(h_A_offd);
      }
   } /* end of forming S */
   else
   {
      hypre_CSRMatrixDestroy(SLU);
   }

   *matBLU_csrsvdata_ptr = matBLU_csrsvdata;
   *matSLU_csrsvdata_ptr = matSLU_csrsvdata;
   *A_fake_diag_ip = A_fake_diag_i;

   /* Do not free perm_data/qperm_data */
   if (perm)
   {
      hypre_IntArrayData(perm)  = NULL;
   }
   if (qperm)
   {
      hypre_IntArrayData(qperm) = NULL;
   }

   /* Free memory */
   hypre_CSRMatrixDestroy(A_diag);
   hypre_IntArrayDestroy(perm);
   hypre_IntArrayDestroy(qperm);
   hypre_IntArrayDestroy(rperm);
   hypre_IntArrayDestroy(rqperm);
   hypre_IntArrayDestroy(h_perm);
   hypre_IntArrayDestroy(h_rperm);

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILU0LocalDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILU0LocalDevice(hypre_CSRMatrix *A)
{

   /* Input matrix data */
   HYPRE_Int               num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               num_cols     = hypre_CSRMatrixNumCols(A);
   HYPRE_Int               num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex          *A_data       = hypre_CSRMatrixData(A);
   HYPRE_Int              *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int              *A_j          = hypre_CSRMatrixJ(A);

   /* pointers to vendor math libraries data */
#if defined(HYPRE_USING_CUSPARSE)
   csrilu02Info_t          matA_info    = NULL;
   cusparseHandle_t        handle       = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t      descr        = hypre_CSRMatrixGPUMatDescr(A);

#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info      matA_info    = NULL;
   rocsparse_handle        handle       = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_mat_descr     descr        = hypre_CSRMatrixGPUMatDescr(A);
#endif

   /* variables and working arrays used during the ilu */
   HYPRE_Int               zero_pivot;
   HYPRE_Int               matA_buffersize;
   void                   *matA_buffer  = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SetupILU0LocalDevice");

   /* Sanity check */
   hypre_assert(num_rows == num_cols);

   /* 1. Sort columns inside each row first, we can't assume that's sorted */
   hypre_CSRMatrixSortRow(A);

   /* 2. Create info for ilu setup and solve */
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(rocsparse_create_mat_info(&matA_info));
#endif

   /* 3. Get working array size */
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_bufferSize(handle, num_rows, num_nonzeros,
                                                          descr, A_data, A_i, A_j,
                                                          matA_info, &matA_buffersize));
#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_buffer_size(handle, num_rows, num_nonzeros,
                                                            descr, A_data, A_i, A_j,
                                                            matA_info, &matA_buffersize));
#endif

   /* 4. Create working array, since they won't be visited by host, allocate on device */
   matA_buffer = hypre_MAlloc(matA_buffersize, HYPRE_MEMORY_DEVICE);

   /* 5. Now perform the analysis */
#if defined(HYPRE_USING_CUSPARSE)
   /* 5-1. Analysis */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_analysis(handle, num_rows, num_nonzeros,
                                                        descr, A_data, A_i, A_j,
                                                        matA_info,
                                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                        matA_buffer));

   /* 5-2. Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* 6. Apply the factorization */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02(handle, num_rows, num_nonzeros,
                                               descr, A_data, A_i, A_j,
                                               matA_info,
                                               CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                               matA_buffer));

   /* Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* Free info */
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrilu02Info(matA_info));

#elif defined(HYPRE_USING_ROCSPARSE)
   /* 5-1. Analysis */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_analysis(handle, num_rows, num_nonzeros,
                                                         descr, A_data, A_i, A_j,
                                                         matA_info,
                                                         rocsparse_analysis_policy_reuse,
                                                         rocsparse_solve_policy_auto,
                                                         matA_buffer));

   /* 5-2. Check for zero pivot */
   HYPRE_ROCSPARSE_CALL(rocsparse_csrsv_zero_pivot(handle, descr, info, &zero_pivot));

   /* 6. Apply the factorization */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0(handle, num_rows, num_nonzeros,
                                                descr, A_data, A_i, A_j,
                                                matA_info,
                                                rocsparse_solve_policy_auto,
                                                matA_buffer));

   /* Check for zero pivot */
   HYPRE_ROCSPARSE_CALL(rocsparse_csrsv_zero_pivot(handle, descr, info, &zero_pivot));

   /* Free info */
   HYPRE_ROCSPARSE_CALL(rocsparse_destroy_mat_info(matA_info));
#endif

   /* Done with factorization, finishing up */
   hypre_TFree(matA_buffer, HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#endif




/*********************************************************************************/
/*                   HYPRE_ILUSetupCusparseCSRILU0SetupSolve                     */
/*********************************************************************************/

HYPRE_Int HYPRE_ILUSetupDeviceCSRILU0SetupSolve(hypre_CSRMatrix *A, hypre_GpuMatData * matL_des,
                                                hypre_GpuMatData * matU_des, hypre_CsrsvData ** matLU_csrsvdata_ptr)
{
   hypre_GpuProfilingPushRange("ILU0SolveAnalysis");

    /* TODO (VPM): Refactor the functions below */
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_ILUSetupCusparseCSRILU0SetupSolve(A, matL_des, matU_des, matLU_csrsvdata_ptr);

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ILUSetupRocsparseCSRILU0SetupSolve(A, matL_des, matU_des, matLU_csrsvdata_ptr);
#endif

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */

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
