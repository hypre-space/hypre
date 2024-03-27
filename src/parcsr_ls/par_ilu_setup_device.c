/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_ILUSetupDevice
 *
 * ILU(0), ILUK, ILUT setup on the device
 *
 * Arguments:
 *    A = input matrix
 *    perm_data  = permutation array indicating ordering of rows.
 *                 Could come from a CF_marker array or a reordering routine.
 *    qperm_data = permutation array indicating ordering of columns
 *    nI  = number of internal unknowns
 *    nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *          Schur complement is formed if nLU < n
 *
 * This function will form the global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupDevice(hypre_ParILUData       *ilu_data,
                     hypre_ParCSRMatrix     *A,
                     HYPRE_Int              *perm_data,
                     HYPRE_Int              *qperm_data,
                     HYPRE_Int               n,
                     HYPRE_Int               nLU,
                     hypre_CSRMatrix       **BLUptr,
                     hypre_ParCSRMatrix    **matSptr,
                     hypre_CSRMatrix       **Eptr,
                     hypre_CSRMatrix       **Fptr)
{
   /* Input ILU data */
   HYPRE_Int                ilu_type            = hypre_ParILUDataIluType(ilu_data);
   HYPRE_Int                fill_level          = hypre_ParILUDataLfil(ilu_data);
   HYPRE_Int                max_row_nnz         = hypre_ParILUDataMaxRowNnz(ilu_data);
   HYPRE_Real              *droptol             = hypre_ParILUDataDroptol(ilu_data);
   HYPRE_Int                iter_setup_type     = hypre_ParILUDataIterativeSetupType(ilu_data);
   HYPRE_Int                iter_setup_option   = hypre_ParILUDataIterativeSetupOption(ilu_data);
   HYPRE_Int                iter_setup_max_iter = hypre_ParILUDataIterativeSetupMaxIter(ilu_data);
   HYPRE_Complex            iter_setup_tol      = hypre_ParILUDataIterativeSetupTolerance(ilu_data);

   /* Input matrix data */
   MPI_Comm                 comm                = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation     memory_location     = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParCSRMatrix      *matS                = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   hypre_CSRMatrix         *A_offd              = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix         *h_A_offd            = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;

   /* Permutation arrays */
   HYPRE_Int               *rperm_data          = NULL;
   HYPRE_Int               *rqperm_data         = NULL;
   hypre_IntArray          *perm                = NULL;
   hypre_IntArray          *rperm               = NULL;
   hypre_IntArray          *qperm               = NULL;
   hypre_IntArray          *rqperm              = NULL;
   hypre_IntArray          *h_perm              = NULL;
   hypre_IntArray          *h_rperm             = NULL;

   /* Variables for matS */
   HYPRE_Int                m                   = n - nLU;
   HYPRE_Int                nI                  = nLU; //use default
   HYPRE_Int                e                   = 0;
   HYPRE_Int                m_e                 = m;
   HYPRE_Int               *S_diag_i            = NULL;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int                S_offd_nnz;
   HYPRE_Int                S_offd_ncols;
   HYPRE_Int                S_diag_nnz;

   hypre_ParCSRMatrix      *Apq                 = NULL;
   hypre_ParCSRMatrix      *ALU                 = NULL;
   hypre_ParCSRMatrix      *parL                = NULL;
   hypre_ParCSRMatrix      *parU                = NULL;
   hypre_ParCSRMatrix      *parS                = NULL;
   HYPRE_Real              *parD                = NULL;
   HYPRE_Int               *uend                = NULL;

   /* Local variables */
   HYPRE_BigInt            *send_buf            = NULL;
   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int                num_sends, begin, end;
   HYPRE_BigInt             total_rows, col_starts[2];
   HYPRE_Int                i, j, k1, k2, k3, col;
   HYPRE_Int                my_id, num_procs;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Sanity checks */
#if !defined(HYPRE_USING_UNIFIED_MEMORY)
   if (ilu_type == 0 && fill_level)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "ILUK setup on device runs requires unified memory!");
      return hypre_error_flag;
   }
   else if (ilu_type == 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "ILUT setup on device runs requires unified memory!");
      return hypre_error_flag;
   }
   else if (ilu_type == 10 && fill_level)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "GMRES+ILUK setup on device runs requires unified memory!");
      return hypre_error_flag;
   }
   else if (ilu_type == 11)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "GMRES+ILUT setup on device runs requires unified memory!");
      return hypre_error_flag;
   }
#endif

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
      /*
       * Apply ILU factorization to the entire A_diag
       *
       * | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       *
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */

#if !defined(HYPRE_USING_SYCL)
      if ((fill_level == 0) && !(ilu_type % 10))
      {
         /* Copy diagonal matrix into a new place with permutation
          * That is, A_diag = A_diag(perm,qperm); */
         hypre_CSRMatrixPermute(hypre_ParCSRMatrixDiag(A), perm_data, rqperm_data, &A_diag);

         /* Compute ILU0 on the device */
         if (iter_setup_type)
         {
            hypre_ILUSetupIterativeILU0Device(A_diag, iter_setup_type, iter_setup_option,
                                              iter_setup_max_iter, iter_setup_tol,
                                              &hypre_ParILUDataIterativeSetupNumIter(ilu_data),
                                              &hypre_ParILUDataIterativeSetupHistory(ilu_data));
         }
         else
         {
            hypre_CSRMatrixILU0(A_diag);
         }

         hypre_ParILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);
         hypre_CSRMatrixDestroy(A_diag);
      }
      else
#endif
      {
         hypre_ParILURAPReorder(A, perm_data, rqperm_data, &Apq);
#if defined(HYPRE_USING_SYCL)
         /* WM: note - ILU0 is not yet available in oneMKL sparse */
         if (fill_level == 0 && !(ilu_type % 10))
         {
            hypre_ILUSetupILU0(Apq, NULL, NULL, n, n, &parL, &parD, &parU, &parS, &uend);
         }
#endif
         if (fill_level != 0 && !(ilu_type % 10))
         {
            hypre_ILUSetupILUK(Apq, fill_level, NULL, NULL, n, n, &parL, &parD, &parU, &parS, &uend);
         }
         else if ((ilu_type % 10) == 1)
         {
            hypre_ILUSetupILUT(Apq, max_row_nnz, droptol, NULL, NULL, n, n,
                               &parL, &parD, &parU, &parS, &uend);
         }

         hypre_ParCSRMatrixDestroy(Apq);
         hypre_TFree(uend, HYPRE_MEMORY_HOST);
         hypre_ParCSRMatrixDestroy(parS);

         hypre_ILUSetupLDUtoCusparse(parL, parD, parU, &ALU);

         hypre_ParCSRMatrixDestroy(parL);
         hypre_ParCSRMatrixDestroy(parU);
         hypre_TFree(parD, HYPRE_MEMORY_DEVICE);

         hypre_ParILUExtractEBFC(hypre_ParCSRMatrixDiag(ALU), nLU,
                                 BLUptr, &SLU, Eptr, Fptr);

         hypre_ParCSRMatrixDestroy(ALU);
      }
   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* Compute total rows in Schur block */
   HYPRE_BigInt big_m = (HYPRE_BigInt) m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* only form when total_rows > 0 */
   if (total_rows > 0)
   {
      /* now create S - need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      if (!SLU)
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
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
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

   /* Set output pointer */
   *matSptr = matS;

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
   hypre_IntArrayDestroy(perm);
   hypre_IntArrayDestroy(qperm);
   hypre_IntArrayDestroy(rperm);
   hypre_IntArrayDestroy(rqperm);
   hypre_IntArrayDestroy(h_perm);
   hypre_IntArrayDestroy(h_rperm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupIterativeILU0Device
 *
 * This function computes an ILU0 iteratively with rocSPARSE.
 *
 * Input arguments:
 *   A - input matrix
 *   type - algorithm for computing iterative ILU0
 *   option - internal flags used by rocSPARSE
 *   max_iter - max. number of iterations
 *   tolerance - stopping criteria in iterative algorithm
 *
 * Output arguments:
 *   num_iter_ptr - number of iterations
 *   history_ptr - list of corrections and residual values for each iteration
 *                 (computed only when the 5th bit of option is active)
 *
 * Note: This function requires rocSPARSE 2.4.0 at least.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupIterativeILU0Device(hypre_CSRMatrix  *A,
                                  HYPRE_Int         type,
                                  HYPRE_Int         option,
                                  HYPRE_Int         max_iter,
                                  HYPRE_Real        tolerance,
                                  HYPRE_Int        *num_iter_ptr,
                                  HYPRE_Real      **history_ptr)
{
#if defined (HYPRE_USING_ROCSPARSE) && (ROCSPARSE_VERSION >= 200400)
   /* Input matrix data */
   HYPRE_Int                 num_rows      = hypre_CSRMatrixNumRows(A);
   HYPRE_Int                 num_cols      = hypre_CSRMatrixNumCols(A);
   HYPRE_Int                 num_nonzeros  = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int                *A_i           = hypre_CSRMatrixI(A);
   HYPRE_Int                *A_j           = hypre_CSRMatrixJ(A);
   HYPRE_Complex            *A_data        = hypre_CSRMatrixData(A);
   HYPRE_Complex            *A_data_new;

   /* Vendor math sparse libraries data */
   void                     *buffer        = NULL;
   rocsparse_index_base      idx_base      = rocsparse_index_base_zero;
   rocsparse_handle          handle        = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_datatype        data_type;
   size_t                    buffer_size;
   HYPRE_Int                 history_size;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CSRMatrixITILU0");

   /* Set default output */
   *num_iter_ptr = 0;

   /*-------------------------------------------------------------------------------------
    * 0. Sanity checks
    *-------------------------------------------------------------------------------------*/

   if (num_rows != num_cols)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not a square matrix!");
      return hypre_error_flag;
   }

#if defined(HYPRE_COMPLEX)
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Complex data type is not supported!");
   return hypre_error_flag;

#elif defined(HYPRE_LONG_DOUBLE)
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Long-double type is not supported!");
   return hypre_error_flag;

#elif defined(HYPRE_SINGLE)
   data_type = rocsparse_datatype_f32_r;

#else
   data_type = rocsparse_datatype_f64_r;
#endif

   /*-------------------------------------------------------------------------------------
    * 1. Sort columns belonging to each row, then copy result to new matrix
    *-------------------------------------------------------------------------------------*/

   hypre_CSRMatrixSortRow(A);

   /* TODO (VPM): make use of a non-zero initial guess if available. */
   A_data_new = hypre_CTAlloc(HYPRE_Complex, num_nonzeros, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------------------------
    * 2. Get work array size
    *-------------------------------------------------------------------------------------*/

   HYPRE_ROCSPARSE_CALL(rocsparse_csritilu0_buffer_size(handle,
                                                        (rocsparse_itilu0_alg) type,
                                                        (rocsparse_int) option,
                                                        (rocsparse_int) max_iter,
                                                        (rocsparse_int) num_rows,
                                                        (rocsparse_int) num_nonzeros,
                                                        (const rocsparse_int*) A_i,
                                                        (const rocsparse_int*) A_j,
                                                        idx_base,
                                                        data_type,
                                                        &buffer_size));

   /*-------------------------------------------------------------------------------------
    * 3. Create work array on the device
    *-------------------------------------------------------------------------------------*/

   buffer = hypre_TAlloc(char, buffer_size, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------------------------
    * 4. Perform the analysis (pre-processing)
    *-------------------------------------------------------------------------------------*/

   hypre_GpuProfilingPushRange("Analysis");
   HYPRE_ROCSPARSE_CALL(rocsparse_csritilu0_preprocess(handle,
                                                       (rocsparse_itilu0_alg) type,
                                                       (rocsparse_int) option,
                                                       (rocsparse_int) max_iter,
                                                       (rocsparse_int) num_rows,
                                                       (rocsparse_int) num_nonzeros,
                                                       (const rocsparse_int*) A_i,
                                                       (const rocsparse_int*) A_j,
                                                       idx_base,
                                                       data_type,
                                                       buffer_size,
                                                       buffer));
   hypre_GpuProfilingPopRange();

   /*-------------------------------------------------------------------------------------
    * 5. Compute the numerical factorization iteratively
    *-------------------------------------------------------------------------------------*/

   *num_iter_ptr = max_iter;
   hypre_GpuProfilingPushRange("Factorization");
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csritilu0_compute(handle,
                                                          (rocsparse_itilu0_alg) type,
                                                          (rocsparse_int) option,
                                                          (rocsparse_int*) num_iter_ptr,
                                                          tolerance,
                                                          (rocsparse_int) num_rows,
                                                          (rocsparse_int) num_nonzeros,
                                                          (const rocsparse_int*) A_i,
                                                          (const rocsparse_int*) A_j,
                                                          (const HYPRE_Complex*) A_data,
                                                          A_data_new,
                                                          idx_base,
                                                          buffer_size,
                                                          buffer));
   hypre_GpuProfilingPopRange();
   hypre_CSRMatrixData(A) = A_data_new;
   hypre_TFree(A_data, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------------------------
    * 6. Compute history if requested
    *-------------------------------------------------------------------------------------*/

   if (option & rocsparse_itilu0_option_convergence_history)
   {
      history_size = (*num_iter_ptr) * 2;
      *history_ptr = hypre_TAlloc(HYPRE_Complex, history_size, HYPRE_MEMORY_HOST);

      HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csritilu0_history(handle,
                                                             (rocsparse_itilu0_alg) type,
                                                             (rocsparse_int*) num_iter_ptr,
                                                             *history_ptr,
                                                             buffer_size,
                                                             buffer));
   }

   /*-------------------------------------------------------------------------------------
    * 7. Free memory
    *-------------------------------------------------------------------------------------*/

   /* Free buffer */
   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;
#else
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(type);
   HYPRE_UNUSED_VAR(option);
   HYPRE_UNUSED_VAR(max_iter);
   HYPRE_UNUSED_VAR(tolerance);
   HYPRE_UNUSED_VAR(num_iter_ptr);
   HYPRE_UNUSED_VAR(history_ptr);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Iterative ILU0 requires rocSPARSE 2.4.0 at least!");
#endif

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
