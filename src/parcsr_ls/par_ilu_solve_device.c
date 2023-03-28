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
 * hypre_ILUApplyLowerUpperDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUApplyLowerUpperDevice(hypre_GpuMatData  *matL_des,
                               hypre_GpuMatData  *matU_des,
                               hypre_CsrsvData   *matLU_csrsvdata,
                               hypre_CSRMatrix   *LU,
                               HYPRE_Complex     *ftemp_data,
                               HYPRE_Complex     *utemp_data)
{
   /* Lower/Upper matrix data */
   HYPRE_Int            num_rows      = hypre_CSRMatrixNumRows(LU);
   HYPRE_Int            num_nonzeros  = hypre_CSRMatrixNumNonzeros(LU);
   HYPRE_Int           *LU_i          = hypre_CSRMatrixI(LU);
   HYPRE_Int           *LU_j          = hypre_CSRMatrixJ(LU);
   HYPRE_Complex       *LU_data       = hypre_CSRMatrixData(LU);

   /* Local variables */
   HYPRE_Complex        one = 1.0;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ILUApplyLowerUpper");

#if defined(HYPRE_USING_CUSPARSE)
   cusparseHandle_t  handle = hypre_HandleCusparseHandle(hypre_handle());

   /* L solve - Forward solve */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   num_rows, num_nonzeros, &one,
                                                   hypre_GpuMatDataMatDescr(matL_des),
                                                   LU_data, LU_i, LU_j,
                                                   hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                   utemp_data, ftemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
                                                   hypre_CsrsvDataBuffer(matLU_csrsvdata)));

   /* U solve - Backward substitution */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   num_rows, num_nonzeros, &one,
                                                   hypre_GpuMatDataMatDescr(matU_des),
                                                   LU_data, LU_i, LU_j,
                                                   hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                   ftemp_data, utemp_data,
                                                   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
                                                   hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_handle  handle = hypre_HandleCusparseHandle(hypre_handle());

   /* L solve - Forward solve */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                    num_rows, num_nonzeros, &one,
                                                    hypre_GpuMatDataMatDescr(matL_des),
                                                    LU_data, LU_i, LU_j,
                                                    hypre_CsrsvDataInfoL(matLU_csrsvdata),
                                                    utemp_data, ftemp_data,
                                                    hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
                                                    hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

   /* U solve - Backward substitution */
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                    num_rows, num_nonzeros, &one,
                                                    hypre_GpuMatDataMatDescr(matU_des),
                                                    LU_data, LU_i, LU_j,
                                                    hypre_CsrsvDataInfoU(matLU_csrsvdata),
                                                    ftemp_data, utemp_data,
                                                    hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
                                                    hypre_CsrsvDataBuffer(matLU_csrsvdata) ));
#endif

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSolveLUDevice
 *
 * Incomplete LU solve (GPU)
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
 *
 * TODO (VPM): Merge this function with hypre_ILUSolveLUIterDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLUDevice(hypre_ParCSRMatrix *A,
                       hypre_GpuMatData   *matL_des,
                       hypre_GpuMatData   *matU_des,
                       hypre_CsrsvData    *matLU_csrsvdata,
                       hypre_CSRMatrix    *matLU,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u,
                       HYPRE_Int          *perm,
                       hypre_ParVector    *ftemp,
                       hypre_ParVector    *utemp)
{
   HYPRE_Int            num_rows      = hypre_ParCSRMatrixNumRows(A);

   hypre_Vector        *utemp_local   = hypre_ParVectorLocalVector(utemp);
   HYPRE_Complex       *utemp_data    = hypre_VectorData(utemp_local);
   hypre_Vector        *ftemp_local   = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Complex       *ftemp_data    = hypre_VectorData(ftemp_local);

   HYPRE_Complex        alpha = -1.0;
   HYPRE_Complex        beta  = 1.0;

   /* Sanity check */
   if (num_rows == 0)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ILUSolve");

   /* Compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* Apply permutation */
   if (perm)
   {
      HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Apply preconditioner (u = U^{-1} * L^{-1} * f) */
   hypre_ILUApplyLowerUpperDevice(matL_des, matU_des, matLU_csrsvdata,
                                  matLU, ftemp_data, utemp_data);

   /* Apply reverse permutation */
   if (perm)
   {
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
   }
   else
   {
      hypre_TMemcpy(ftemp_data, utemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUApplyLowerJacIterDevice
 *
 * Incomplete L solve (Forward) of u^{k+1} = L^{-1}u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUApplyLowerJacIterDevice(hypre_CSRMatrix *A,
                                 hypre_Vector    *input,
                                 hypre_Vector    *work,
                                 hypre_Vector    *output,
                                 HYPRE_Int        lower_jacobi_iters)
{
   HYPRE_Complex   *input_data  = hypre_VectorData(input);
   HYPRE_Complex   *work_data   = hypre_VectorData(work);
   HYPRE_Complex   *output_data = hypre_VectorData(output);
   HYPRE_Int        num_rows    = hypre_CSRMatrixNumRows(A);

   HYPRE_Int        kk = 0;

   /* Since the initial guess to the jacobi iteration is 0, the result of
      the first L SpMV is 0, so no need to compute.
      However, we still need to compute the transformation */
   hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, 0.0);

   /* Do the remaining iterations */
   for (kk = 1; kk < lower_jacobi_iters; kk++)
   {
      /* apply SpMV */
      hypre_CSRMatrixSpMVDevice(0, 1.0, A, output, 0.0, work, -2);

      /* transform */
      hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, -1.0);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUApplyUpperJacIterDevice
 *
 * Incomplete U solve (Backward) of u^{k+1} = U^{-1}u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUApplyUpperJacIterDevice(hypre_CSRMatrix *A,
                                 hypre_Vector    *input,
                                 hypre_Vector    *work,
                                 hypre_Vector    *output,
                                 hypre_Vector    *diag,
                                 HYPRE_Int        upper_jacobi_iters)
{
   HYPRE_Complex   *output_data    = hypre_VectorData(output);
   HYPRE_Complex   *work_data      = hypre_VectorData(work);
   HYPRE_Complex   *input_data     = hypre_VectorData(input);
   HYPRE_Complex   *diag_data      = hypre_VectorData(diag);
   HYPRE_Int        num_rows       = hypre_CSRMatrixNumRows(A);

   HYPRE_Int        kk = 0;

   /* Since the initial guess to the jacobi iteration is 0,
      the result of the first U SpMV is 0, so no need to compute.
      However, we still need to compute the transformation */
   hypreDevice_zeqxmydd(num_rows, input_data, 0.0, work_data, output_data, diag_data);

   /* Do the remaining iterations */
   for (kk = 1; kk < upper_jacobi_iters; kk++)
   {
      /* apply SpMV */
      hypre_CSRMatrixSpMVDevice(0, 1.0, A, output, 0.0, work, 2);

      /* transform */
      hypreDevice_zeqxmydd(num_rows, input_data, -1.0, work_data, output_data, diag_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUApplyLowerUpperJacIterDevice
 *
 * Incomplete LU solve of u^{k+1} = U^{-1} L^{-1} u^k on the GPU using the
 * Jacobi iterative approach.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUApplyLowerUpperJacIterDevice(hypre_CSRMatrix *A,
                                      hypre_Vector    *work1,
                                      hypre_Vector    *work2,
                                      hypre_Vector    *inout,
                                      hypre_Vector    *diag,
                                      HYPRE_Int        lower_jacobi_iters,
                                      HYPRE_Int        upper_jacobi_iters)
{
   /* apply the iterative solve to L */
   hypre_ILUApplyLowerJacIterDevice(A, inout, work1, work2, lower_jacobi_iters);

   /* apply the iterative solve to U */
   hypre_ILUApplyUpperJacIterDevice(A, work2, work1, inout, diag, upper_jacobi_iters);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSolveLUIterDevice
 *
 * Incomplete LU solve using jacobi iterations on GPU.
 * L, D and U factors only have local scope (no off-diagonal processor terms).
 *
 * TODO (VPM): Merge this function with hypre_ILUSolveLUDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLUIterDevice(hypre_ParCSRMatrix *A,
                           hypre_CSRMatrix    *matLU,
                           hypre_ParVector    *f,
                           hypre_ParVector    *u,
                           HYPRE_Int          *perm,
                           hypre_ParVector    *ftemp,
                           hypre_ParVector    *utemp,
                           hypre_ParVector    *xtemp,
                           hypre_Vector      **diag_ptr,
                           HYPRE_Int           lower_jacobi_iters,
                           HYPRE_Int           upper_jacobi_iters)
{
   HYPRE_Int        num_rows    = hypre_ParCSRMatrixNumRows(A);

   hypre_Vector    *diag        = *diag_ptr;
   hypre_Vector    *xtemp_local = hypre_ParVectorLocalVector(xtemp);
   hypre_Vector    *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Complex   *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector    *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Complex   *ftemp_data  = hypre_VectorData(ftemp_local);

   HYPRE_Complex    alpha = -1.0;
   HYPRE_Complex    beta  = 1.0;

   /* Sanity check */
   if (num_rows == 0)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ILUSolveLUIter");

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!diag)
   {
      /* Storage for the diagonal */
      diag = hypre_SeqVectorCreate(num_rows);
      hypre_SeqVectorInitialize(diag);

      /* extract with device kernel */
      hypre_CSRMatrixExtractDiagonalDevice(matLU, hypre_VectorData(diag), 2);

      /* Save output pointer */
      *diag_ptr = diag;
   }

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   if (perm)
   {
      HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* apply the iterative solve to L and U */
   hypre_ILUApplyLowerUpperJacIterDevice(matLU, ftemp_local, xtemp_local, utemp_local,
                                         diag, lower_jacobi_iters, upper_jacobi_iters);

   /* apply reverse permutation */
   if (perm)
   {
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
   }
   else
   {
      hypre_TMemcpy(ftemp_data, utemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
