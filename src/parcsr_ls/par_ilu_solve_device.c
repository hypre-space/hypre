/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_ILUSolveLUDevice
 *
 * Incomplete LU solve (GPU)
 *
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
 *
 * TODO (VPM): Merge this function with hypre_ILUSolveLUIterDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLUDevice(hypre_ParCSRMatrix  *A,
                       hypre_CSRMatrix     *matLU_d,
                       hypre_ParVector     *f,
                       hypre_ParVector     *u,
                       HYPRE_Int           *perm,
                       hypre_ParVector     *ftemp,
                       hypre_ParVector     *utemp)
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
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(perm, perm + num_rows, ftemp_data, utemp_data);
#else
      HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* L solve - Forward solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matLU_d, NULL, utemp_local, ftemp_local);

   /* U solve - Backward substitution */
   hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matLU_d, NULL, ftemp_local, utemp_local);

   /* Apply reverse permutation */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_scatter(utemp_data, utemp_data + num_rows, perm, ftemp_data);
#else
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
#endif
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
      /* Apply SpMV */
      hypre_CSRMatrixSpMVDevice(0, 1.0, A, output, 0.0, work, -2);

      /* Transform */
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
   /* Apply the iterative solve to L */
   hypre_ILUApplyLowerJacIterDevice(A, inout, work1, work2, lower_jacobi_iters);

   /* Apply the iterative solve to U */
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

   /* Compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* Apply permutation */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(perm, perm + num_rows, ftemp_data, utemp_data);
#else
      HYPRE_THRUST_CALL(gather, perm, perm + num_rows, ftemp_data, utemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, num_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Apply the iterative solve to L and U */
   hypre_ILUApplyLowerUpperJacIterDevice(matLU, ftemp_local, xtemp_local, utemp_local,
                                         diag, lower_jacobi_iters, upper_jacobi_iters);

   /* Apply reverse permutation */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_scatter(utemp_data, utemp_data + num_rows, perm, ftemp_data);
#else
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + num_rows, perm, ftemp_data);
#endif
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
 * hypre_ParILUSchurGMRESMatvecDevice
 *
 * Slightly different, for this new matvec, the diagonal of the original
 * matrix is the LU factorization. Thus, the matvec is done in an different way
 *
 * |IS_1 E_12 E_13|
 * |E_21 IS_2 E_23| = S
 * |E_31 E_32 IS_3|
 *
 * |IS_1          |
 * |     IS_2     | = M
 * |          IS_3|
 *
 * Solve Sy = g is just M^{-1}S = M^{-1}g
 *
 * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
 * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
 * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUSchurGMRESMatvecDevice(void          *matvec_data,
                                   HYPRE_Complex  alpha,
                                   void          *ilu_vdata,
                                   void          *x,
                                   HYPRE_Complex  beta,
                                   void          *y)
{
   /* Get matrix information first */
   hypre_ParILUData    *ilu_data       = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix  *S              = hypre_ParILUDataMatS(ilu_data);
   hypre_CSRMatrix     *S_diag         = hypre_ParCSRMatrixDiag(S);

   /* Fist step, apply matvec on empty diagonal slot */
   HYPRE_Int            num_rows       = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int            num_nonzeros   = hypre_CSRMatrixNumNonzeros(S_diag);

   hypre_ParVector     *xtemp          = hypre_ParILUDataXTemp(ilu_data);
   hypre_ParVector     *ytemp          = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector        *xtemp_local    = hypre_ParVectorLocalVector(xtemp);
   hypre_Vector        *ytemp_local    = hypre_ParVectorLocalVector(ytemp);

   /* Local variables */
   HYPRE_Complex        zero           = 0.0;
   HYPRE_Complex        one            = 1.0;

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */

   /* RL: temp. set S_diag's nnz = 0 to skip the matvec
      (based on the assumption in seq_mv/csr_matvec impl.) */
   hypre_CSRMatrixNumRows(S_diag)     = 0;
   hypre_CSRMatrixNumNonzeros(S_diag) = 0;
   hypre_ParCSRMatrixMatvec(alpha, (hypre_ParCSRMatrix *) S, (hypre_ParVector *) x, zero, xtemp);
   hypre_CSRMatrixNumRows(S_diag)     = num_rows;
   hypre_CSRMatrixNumNonzeros(S_diag) = num_nonzeros;

   /* Compute U^{-1}*L^{-1}*(S_offd * x)
    * Or in other words, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */

   /* L solve - Forward solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, S_diag, NULL, xtemp_local, ytemp_local);

   /* U solve - Backward substitution */
   hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, S_diag, NULL, ytemp_local, xtemp_local);

   /* xtemp = xtemp + alpha*x */
   hypre_ParVectorAxpy(alpha, (hypre_ParVector *) x, xtemp);

   /* y = xtemp + beta*y */
   hypre_ParVectorAxpyz(one, xtemp, beta, (hypre_ParVector *) y, (hypre_ParVector *) y);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSolveSchurGMRESDevice
 *
 * Schur Complement solve with GMRES on schur complement
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly use
 *  S, L, D and U factors only have local scope (no off-diag terms) so apart
 *  from the residual calculation (which uses A), the solves with the L and U
 *  factors are local.
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vectors for solving the Schur system
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveSchurGMRESDevice(hypre_ParCSRMatrix  *A,
                               hypre_ParVector     *f,
                               hypre_ParVector     *u,
                               HYPRE_Int           *perm,
                               HYPRE_Int            nLU,
                               hypre_ParCSRMatrix  *S,
                               hypre_ParVector     *ftemp,
                               hypre_ParVector     *utemp,
                               HYPRE_Solver         schur_solver,
                               HYPRE_Solver         schur_precond,
                               hypre_ParVector     *rhs,
                               hypre_ParVector     *x,
                               HYPRE_Int           *u_end,
                               hypre_CSRMatrix     *matBLU_d,
                               hypre_CSRMatrix     *matE_d,
                               hypre_CSRMatrix     *matF_d)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      return hypre_ILUSolveLUDevice(A, matBLU_d, f, u, perm, ftemp, utemp);
   }

   /* Data objects for temp vector */
   hypre_Vector      *utemp_local      = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data       = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local      = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data       = hypre_VectorData(ftemp_local);
   hypre_Vector      *rhs_local        = hypre_ParVectorLocalVector(rhs);
   hypre_Vector      *x_local          = hypre_ParVectorLocalVector(x);
   HYPRE_Real        *x_data           = hypre_VectorData(x_local);

   /* Problem size */
   hypre_CSRMatrix   *matSLU_d         = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int          m                = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int          n                = nLU + m;

   /* Local variables */
   HYPRE_Real         alpha            = -1.0;
   HYPRE_Real         beta             = 1.0;
   hypre_Vector      *ftemp_upper;
   hypre_Vector      *utemp_lower;

   /* Temporary vectors */
   ftemp_upper = hypre_SeqVectorCreate(nLU);
   utemp_lower = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(ftemp_upper) = 0;
   hypre_VectorOwnsData(utemp_lower) = 0;
   hypre_VectorData(ftemp_upper) = ftemp_data;
   hypre_VectorData(utemp_lower) = utemp_data + nLU;
   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);

   /* Compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* Apply permutation before we can start our solve */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(perm, perm + n, ftemp_data, utemp_data);
#else
      HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* This solve won't touch data in utemp, thus, gi is still in utemp_lower */
   /* L solve - Forward solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL, utemp_local, ftemp_local);

   /* 2nd need to compute g'i = gi - Ei*UBi^{-1}*xi
    * Ei*UBi^{-1} is exactly the matE_d here
    * Now:  LBi^{-1}f_i is in ftemp_upper
    *       gi' is in utemp_lower
    */
   hypre_CSRMatrixMatvec(alpha, matE_d, ftemp_upper, beta, utemp_lower);

   /* 3rd need to solve global Schur Complement M^{-1}Sy = M^{-1}g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */

   /* Setup vectors for solve
    * rhs = M^{-1}g'
    */

   /* L solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice_core('L', 1, matSLU_d, NULL, utemp_local,
                                                nLU, ftemp_local, nLU);

   /* U solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice_core('U', 0, matSLU_d, NULL, ftemp_local,
                                                nLU, rhs_local, 0);

   /* Solve with tricky initial guess */
   HYPRE_GMRESSolve(schur_solver,
                    (HYPRE_Matrix) schur_precond,
                    (HYPRE_Vector) rhs,
                    (HYPRE_Vector) x);

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   hypre_CSRMatrixMatvec(alpha, matF_d, x_local, beta, ftemp_upper);

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   /* U solve - Forward solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL, ftemp_local, utemp_local);

   /* Copy lower part solution into u_temp as well */
   hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Perm back */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_scatter(utemp_data, utemp_data + n, perm, ftemp_data);
#else
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(ftemp_data, utemp_data, HYPRE_Complex, n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   /* Free memory */
   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSolveSchurGMRESJacIterDevice
 *
 * Schur Complement solve with GMRES.
 *
 * ParCSRMatrix S is already built in the ilu data structure. S, L, D and U
 *  factors only have local scope (no off-diag terms). So apart from the
 *  residual calculation (which uses A), the solves with the L and U factors
 *  are local.
 * S: the global Schur complement
 * schur_solver: GMRES solver
 * schur_precond: ILU preconditioner for GMRES
 * rhs and x are helper vectors for solving the Schur system
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveSchurGMRESJacIterDevice(hypre_ParCSRMatrix *A,
                                      hypre_ParVector    *f,
                                      hypre_ParVector    *u,
                                      HYPRE_Int          *perm,
                                      HYPRE_Int           nLU,
                                      hypre_ParCSRMatrix *S,
                                      hypre_ParVector    *ftemp,
                                      hypre_ParVector    *utemp,
                                      HYPRE_Solver        schur_solver,
                                      HYPRE_Solver        schur_precond,
                                      hypre_ParVector    *rhs,
                                      hypre_ParVector    *x,
                                      HYPRE_Int          *u_end,
                                      hypre_CSRMatrix    *matBLU_d,
                                      hypre_CSRMatrix    *matE_d,
                                      hypre_CSRMatrix    *matF_d,
                                      hypre_ParVector    *ztemp,
                                      hypre_Vector      **Adiag_diag,
                                      hypre_Vector      **Sdiag_diag,
                                      HYPRE_Int           lower_jacobi_iters,
                                      HYPRE_Int           upper_jacobi_iters)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      return hypre_ILUSolveLUIterDevice(A, matBLU_d, f, u, perm,
                                        ftemp, utemp, ztemp, Adiag_diag,
                                        lower_jacobi_iters, upper_jacobi_iters);
   }

   /* Data objects for work vectors */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   hypre_Vector      *ztemp_local = hypre_ParVectorLocalVector(ztemp);
   hypre_Vector      *rhs_local   = hypre_ParVectorLocalVector(rhs);
   hypre_Vector      *x_local     = hypre_ParVectorLocalVector(x);

   HYPRE_Complex     *utemp_data  = hypre_VectorData(utemp_local);
   HYPRE_Complex     *ftemp_data  = hypre_VectorData(ftemp_local);
   HYPRE_Complex     *x_data      = hypre_VectorData(x_local);

   /* Problem size */
   hypre_CSRMatrix   *matSLU_d    = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int          m           = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int          n           = nLU + m;

   /* Local variables */
   HYPRE_Complex      alpha = -1.0;
   HYPRE_Complex      beta  = 1.0;
   hypre_Vector      *ftemp_upper;
   hypre_Vector      *utemp_lower;
   hypre_Vector      *ftemp_shift;
   hypre_Vector      *utemp_shift;

   /* Set work vectors */
   ftemp_upper = hypre_SeqVectorCreate(nLU);
   utemp_lower = hypre_SeqVectorCreate(m);
   ftemp_shift = hypre_SeqVectorCreate(m);
   utemp_shift = hypre_SeqVectorCreate(m);

   hypre_VectorOwnsData(ftemp_upper) = 0;
   hypre_VectorOwnsData(utemp_lower) = 0;
   hypre_VectorOwnsData(ftemp_shift) = 0;
   hypre_VectorOwnsData(utemp_shift) = 0;

   hypre_VectorData(ftemp_upper) = ftemp_data;
   hypre_VectorData(utemp_lower) = utemp_data + nLU;
   hypre_VectorData(ftemp_shift) = ftemp_data + nLU;
   hypre_VectorData(utemp_shift) = utemp_data + nLU;

   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);
   hypre_SeqVectorInitialize(ftemp_shift);
   hypre_SeqVectorInitialize(utemp_shift);

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!(*Adiag_diag))
   {
      /* Storage for the diagonal */
      *Adiag_diag = hypre_SeqVectorCreate(n);
      hypre_SeqVectorInitialize(*Adiag_diag);

      /* Extract with device kernel */
      hypre_CSRMatrixExtractDiagonalDevice(matBLU_d, hypre_VectorData(*Adiag_diag), 2);
   }

   /* Compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* Apply permutation before we can start our solve */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(perm, perm + n, ftemp_data, utemp_data);
#else
      HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   if (nLU > 0)
   {
      /* Apply the iterative solve to L */
      hypre_ILUApplyLowerJacIterDevice(matBLU_d, utemp_local, ztemp_local,
                                       ftemp_local, lower_jacobi_iters);

      /* 2nd need to compute g'i = gi - Ei*UBi^{-1}*xi
       * Ei*UBi^{-1} is exactly the matE_d here
       * Now:  LBi^{-1}f_i is in ftemp_upper
       *       gi' is in utemp_lower
       */
      hypre_CSRMatrixMatvec(alpha, matE_d, ftemp_upper, beta, utemp_lower);
   }

   /* 3rd need to solve global Schur Complement M^{-1}Sy = M^{-1}g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */

   /* Setup vectors for solve
    * rhs = M^{-1}g'
    */
   if (m > 0)
   {
      /* Grab the main diagonal from the diagonal block. Only do this once */
      if (!(*Sdiag_diag))
      {
         /* Storage for the diagonal */
         *Sdiag_diag = hypre_SeqVectorCreate(m);
         hypre_SeqVectorInitialize(*Sdiag_diag);

         /* Extract with device kernel */
         hypre_CSRMatrixExtractDiagonalDevice(matSLU_d, hypre_VectorData(*Sdiag_diag), 2);
      }

      /* Apply the iterative solve to L */
      hypre_ILUApplyLowerJacIterDevice(matSLU_d, utemp_shift, rhs_local,
                                       ftemp_shift, lower_jacobi_iters);

      /* Apply the iterative solve to U */
      hypre_ILUApplyUpperJacIterDevice(matSLU_d, ftemp_shift, utemp_shift,
                                       rhs_local, *Sdiag_diag, upper_jacobi_iters);
   }

   /* Solve with tricky initial guess */
   HYPRE_GMRESSolve(schur_solver,
                    (HYPRE_Matrix) schur_precond,
                    (HYPRE_Vector) rhs,
                    (HYPRE_Vector) x);

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU > 0)
   {
      hypre_CSRMatrixMatvec(alpha, matF_d, x_local, beta, ftemp_upper);

      /* 5th need to solve UBi*ui = zi */
      /* put result in u_temp upper */

      /* Apply the iterative solve to U */
      hypre_ILUApplyUpperJacIterDevice(matBLU_d, ftemp_local, ztemp_local,
                                       utemp_local, *Adiag_diag, upper_jacobi_iters);
   }

   /* Copy lower part solution into u_temp as well */
   hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Perm back */
   if (perm)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_scatter(utemp_data, utemp_data + n, perm, ftemp_data);
#else
      HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
#endif
   }
   else
   {
      hypre_TMemcpy(ftemp_data, utemp_data, HYPRE_Complex, n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   /* Free memory */
   hypre_SeqVectorDestroy(ftemp_shift);
   hypre_SeqVectorDestroy(utemp_shift);
   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUSchurGMRESMatvecJacIterDevice
 *
 * Slightly different, for this new matvec, the diagonal of the original matrix
 * is the LU factorization. Thus, the matvec is done in an different way
 *
 * |IS_1 E_12 E_13|
 * |E_21 IS_2 E_23| = S
 * |E_31 E_32 IS_3|
 *
 * |IS_1          |
 * |     IS_2     | = M
 * |          IS_3|
 *
 * Solve Sy = g is just M^{-1}S = M^{-1}g
 *
 * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
 * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
 * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUSchurGMRESMatvecJacIterDevice(void          *matvec_data,
                                          HYPRE_Complex  alpha,
                                          void          *ilu_vdata,
                                          void          *x,
                                          HYPRE_Complex  beta,
                                          void          *y)
{
   /* get matrix information first */
   hypre_ParILUData    *ilu_data           = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix  *S                  = hypre_ParILUDataMatS(ilu_data);
   hypre_Vector        *Sdiag_diag         = hypre_ParILUDataSDiagDiag(ilu_data);
   HYPRE_Int            lower_jacobi_iters = hypre_ParILUDataLowerJacobiIters(ilu_data);
   HYPRE_Int            upper_jacobi_iters = hypre_ParILUDataUpperJacobiIters(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   hypre_CSRMatrix     *S_diag            = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int            S_diag_n          = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int            S_diag_nnz        = hypre_CSRMatrixNumNonzeros(S_diag);

   hypre_ParVector     *xtemp             = hypre_ParILUDataXTemp(ilu_data);
   hypre_Vector        *xtemp_local       = hypre_ParVectorLocalVector(xtemp);
   hypre_ParVector     *ytemp             = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector        *ytemp_local       = hypre_ParVectorLocalVector(ytemp);
   hypre_ParVector     *ztemp             = hypre_ParILUDataZTemp(ilu_data);
   hypre_Vector        *ztemp_local       = hypre_ParVectorLocalVector(ztemp);
   HYPRE_Real           zero              = 0.0;
   HYPRE_Real           one               = 1.0;

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */

   /* RL: temp. set S_diag's nnz = 0 to skip the matvec
      (based on the assumption in seq_mv/csr_matvec impl.) */
   hypre_CSRMatrixNumRows(S_diag)     = 0;
   hypre_CSRMatrixNumNonzeros(S_diag) = 0;
   hypre_ParCSRMatrixMatvec(alpha, (hypre_ParCSRMatrix *) S, (hypre_ParVector *) x, zero, xtemp);
   hypre_CSRMatrixNumRows(S_diag)     = S_diag_n;
   hypre_CSRMatrixNumNonzeros(S_diag) = S_diag_nnz;

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!Sdiag_diag)
   {
      /* Storage for the diagonal */
      Sdiag_diag = hypre_SeqVectorCreate(S_diag_n);
      hypre_SeqVectorInitialize(Sdiag_diag);

      /* Extract with device kernel */
      hypre_CSRMatrixExtractDiagonalDevice(S_diag, hypre_VectorData(Sdiag_diag), 2);

      /* Save Schur diagonal */
      hypre_ParILUDataSDiagDiag(ilu_data) = Sdiag_diag;
   }

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another words, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if (S_diag_n)
   {
      /* apply the iterative solve to L and U */
      hypre_ILUApplyLowerUpperJacIterDevice(S_diag, ytemp_local, ztemp_local,
                                            xtemp_local, Sdiag_diag,
                                            lower_jacobi_iters, upper_jacobi_iters);
   }

   /* now add the original x onto it */
   hypre_ParVectorAxpy(alpha, (hypre_ParVector *) x, xtemp);

   /* y = xtemp + beta*y */
   hypre_ParVectorAxpyz(one, xtemp, beta, (hypre_ParVector *) y, (hypre_ParVector *) y);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveRAPGMRESDevice
 *
 * Device solve with GMRES on schur complement, RAP style.
 *
 * See hypre_ILUSolveRAPGMRESHost for more comments
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveRAPGMRESDevice(hypre_ParCSRMatrix   *A,
                             hypre_ParVector      *f,
                             hypre_ParVector      *u,
                             HYPRE_Int            *perm,
                             HYPRE_Int             nLU,
                             hypre_ParCSRMatrix   *S,
                             hypre_ParVector      *ftemp,
                             hypre_ParVector      *utemp,
                             hypre_ParVector      *xtemp,
                             hypre_ParVector      *ytemp,
                             HYPRE_Solver          schur_solver,
                             HYPRE_Solver          schur_precond,
                             hypre_ParVector      *rhs,
                             hypre_ParVector      *x,
                             HYPRE_Int            *u_end,
                             hypre_ParCSRMatrix   *Aperm,
                             hypre_CSRMatrix      *matALU_d,
                             hypre_CSRMatrix      *matBLU_d,
                             hypre_CSRMatrix      *matE_d,
                             hypre_CSRMatrix      *matF_d,
                             HYPRE_Int             test_opt)
{
   /* If we don't have S block, just do one L/U solve */
   if (!S)
   {
      return hypre_ILUSolveLUDevice(A, matBLU_d, f, u, perm, ftemp, utemp);
   }

   /* data objects for vectors */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   hypre_Vector      *xtemp_local = hypre_ParVectorLocalVector(xtemp);
   hypre_Vector      *rhs_local   = hypre_ParVectorLocalVector(rhs);
   hypre_Vector      *x_local     = hypre_ParVectorLocalVector(x);

   HYPRE_Complex     *utemp_data  = hypre_VectorData(utemp_local);
   HYPRE_Complex     *ftemp_data  = hypre_VectorData(ftemp_local);
   HYPRE_Complex     *xtemp_data  = hypre_VectorData(xtemp_local);
   HYPRE_Complex     *rhs_data    = hypre_VectorData(rhs_local);
   HYPRE_Complex     *x_data      = hypre_VectorData(x_local);

   hypre_CSRMatrix   *matSLU_d    = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int          m           = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int          n           = nLU + m;
   HYPRE_Real         one         = 1.0;
   HYPRE_Real         mone        = -1.0;
   HYPRE_Real         zero        = 0.0;

   /* Temporary vectors */
   hypre_Vector      *ftemp_upper;
   hypre_Vector      *utemp_lower;

   /* Create temporary vectors */
   ftemp_upper = hypre_SeqVectorCreate(nLU);
   utemp_lower = hypre_SeqVectorCreate(m);

   hypre_VectorOwnsData(ftemp_upper) = 0;
   hypre_VectorOwnsData(utemp_lower) = 0;
   hypre_VectorData(ftemp_upper)     = ftemp_data;
   hypre_VectorData(utemp_lower)     = utemp_data + nLU;

   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);

   switch (test_opt)
   {
      case 1: case 3:
      {
         /* E and F */
         /* compute residual */
         hypre_ParCSRMatrixMatvecOutOfPlace(mone, A, u, one, f, utemp);

         /* apply permutation before we can start our solve
          * Au=f -> (PAQ)Q'u=Pf
          */
         if (perm)
         {
#if defined(HYPRE_USING_SYCL)
            hypreSycl_gather(perm, perm + n, utemp_data, ftemp_data);
#else
            HYPRE_THRUST_CALL(gather, perm, perm + n, utemp_data, ftemp_data);
#endif
         }
         else
         {
            hypre_TMemcpy(ftemp_data, utemp_data, HYPRE_Complex, n,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }

         /* A-smoothing
          * x = [UA\(LA\(P*f_u))] fill to xtemp
          */

         /* L solve - Forward solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matALU_d, NULL,
                                                 ftemp_local, utemp_local);

         /* U solve - Backward solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matALU_d, NULL,
                                                 utemp_local, xtemp_local);

         /* residual, we should not touch xtemp for now
          * r = R*(f-PAQx)
          */
         hypre_ParCSRMatrixMatvec(mone, Aperm, xtemp, one, ftemp);

         /* with R is complex */
         /* copy partial data in */
         hypre_TMemcpy(rhs_data, ftemp_data + nLU, HYPRE_Real, m,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                 ftemp_local, utemp_local);

         /* -U^{-1}L^{-1} */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(mone, matE_d, ftemp_upper, one, rhs_local);

         /* Solve S */
         if (S)
         {
            /* if we have a schur complement */
            hypre_ParVectorSetConstantValues(x, 0.0);
            HYPRE_GMRESSolve(schur_solver,
                             (HYPRE_Matrix) schur_precond,
                             (HYPRE_Vector) rhs,
                             (HYPRE_Vector) x);

            /* u = xtemp + P*x */
            /* -Fx */
            hypre_CSRMatrixMatvec(mone, matF_d, x_local, zero, ftemp_upper);

            /* -L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* -U{-1}L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    utemp_local, ftemp_local);

            /* now copy data to y_lower */
            hypre_TMemcpy(ftemp_data + nLU, x_data, HYPRE_Real, m,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }
         else
         {
            /* otherwise just apply triangular solves */
            /* L solve - Forward solve */
            hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matSLU_d, NULL, rhs_local, x_local);

            /* U solve - Backward solve */
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matSLU_d, NULL, x_local, rhs_local);

            /* u = xtemp + P*x */
            /* -Fx */
            hypre_CSRMatrixMatvec(mone, matF_d, rhs_local, zero, ftemp_upper);

            /* -L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* -U{-1}L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    utemp_local, ftemp_local);

            /* now copy data to y_lower */
            hypre_TMemcpy(ftemp_data + nLU, rhs_data, HYPRE_Real, m,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }

         /* correction to the residual */
         hypre_ParVectorAxpy(one, ftemp, xtemp);

         /* perm back */
         if (perm)
         {
#if defined(HYPRE_USING_SYCL)
            hypreSycl_scatter(xtemp_data, xtemp_data + n, perm, ftemp_data);
#else
            HYPRE_THRUST_CALL(scatter, xtemp_data, xtemp_data + n, perm, ftemp_data);
#endif
         }
         else
         {
            hypre_TMemcpy(ftemp_data, xtemp_data, HYPRE_Complex, n,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }
      }
      break;

   case 0: case 2: default:
      {
         /* EU^{-1} and L^{-1}F */
         /* compute residual */
         hypre_ParCSRMatrixMatvecOutOfPlace(mone, A, u, one, f, ftemp);

         /* apply permutation before we can start our solve
          * Au=f -> (PAQ)Q'u=Pf
          */
         if (perm)
         {
#if defined(HYPRE_USING_SYCL)
            hypreSycl_gather(perm, perm + n, ftemp_data, utemp_data);
#else
            HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);
#endif
         }
         else
         {
            hypre_TMemcpy(utemp_data, ftemp_data, HYPRE_Complex, n,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }

         /* A-smoothing
          * x = [UA\(LA\(P*f_u))] fill to xtemp
          */

         /* L solve - Forward solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matALU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* U solve - Backward solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matALU_d, NULL,
                                                 ftemp_local, xtemp_local);

         /* residual, we should not touch xtemp for now
          * r = R*(f-PAQx)
          */
         hypre_ParCSRMatrixMatvec(mone, Aperm, xtemp, one, utemp);

         /* with R is complex */
         /* copy partial data in */
         hypre_TMemcpy(rhs_data, utemp_data + nLU, HYPRE_Real, m,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matBLU_d, NULL,
                                                 utemp_local, ftemp_local);

         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(mone, matE_d, ftemp_upper, one, rhs_local);

         /* Solve S */
         if (S)
         {
            /* if we have a schur complement */
            hypre_ParVectorSetConstantValues(x, 0.0);
            HYPRE_GMRESSolve(schur_solver,
                             (HYPRE_Matrix) schur_precond,
                             (HYPRE_Vector) rhs,
                             (HYPRE_Vector) x);

            /* u = xtemp + P*x */
            /* -L^{-1}Fx */
            hypre_CSRMatrixMatvec(mone, matF_d, x_local, zero, ftemp_upper);

            /* -U{-1}L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* now copy data to y_lower */
            hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }
         else
         {
            /* otherwise just apply triangular solves */
            hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, matSLU_d, NULL, rhs_local, x_local);
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matSLU_d, NULL, x_local, rhs_local);

            /* u = xtemp + P*x */
            /* -L^{-1}Fx */
            hypre_CSRMatrixMatvec(mone, matF_d, rhs_local, zero, ftemp_upper);

            /* -U{-1}L^{-1}Fx */
            hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, matBLU_d, NULL,
                                                    ftemp_local, utemp_local);

            /* now copy data to y_lower */
            hypre_TMemcpy(utemp_data + nLU, rhs_data, HYPRE_Real, m,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }

         /* Update xtemp */
         hypre_ParVectorAxpy(one, utemp, xtemp);

         /* perm back */
         if (perm)
         {
#if defined(HYPRE_USING_SYCL)
            hypreSycl_scatter(xtemp_data, xtemp_data + n, perm, ftemp_data);
#else
            HYPRE_THRUST_CALL(scatter, xtemp_data, xtemp_data + n, perm, ftemp_data);
#endif
         }
         else
         {
            hypre_TMemcpy(ftemp_data, xtemp_data, HYPRE_Complex, n,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         }
      }
      break;
   }

   /* Done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(one, ftemp, u);

   /* Destroy temporary vectors */
   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
