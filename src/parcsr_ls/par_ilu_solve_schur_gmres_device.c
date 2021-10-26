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

/*--------------------------------------------------------------------------
 * hypre_ParILUDeviceSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

   /* Slightly different, for this new matvec, the diagonal of the original matrix
    * is the LU factorization. Thus, the matvec is done in an different way
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
    *
    * */

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ParILUDeviceSchurGMRESMatvec( void   *matvec_data,
                                      HYPRE_Complex  alpha,
                                      void   *ilu_vdata,
                                      void   *x,
                                      HYPRE_Complex  beta,
                                      void   *y           )
{
#if defined(HYPRE_USING_CUSPARSE)
   hypre_ParILUCusparseSchurGMRESMatvec(matvec_data, alpha, ilu_vdata, x, beta, y);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   hypre_ParILURocsparseSchurGMRESMatvec(matvec_data, alpha, ilu_vdata, x, beta, y);
#endif

   return hypre_error_flag;
}

#endif


#if defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypre_ParILUCusparseSchurGMRESMatvec( void   *matvec_data,
                                      HYPRE_Complex  alpha,
                                      void   *ilu_vdata,
                                      void   *x,
                                      HYPRE_Complex  beta,
                                      void   *y           )
{
   /* get matrix information first */
   hypre_ParILUData *ilu_data                   = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A                        = hypre_ParILUDataMatS(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   hypre_CSRMatrix   *A_diag                    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int         *A_diag_i                  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         *A_diag_j                  = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real        *A_diag_data               = hypre_CSRMatrixData(A_diag);
   HYPRE_Int         A_diag_n                   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int         A_diag_nnz                 = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int         *A_diag_fake_i             = hypre_ParILUDataMatAFakeDiagonal(ilu_data);

   hypre_ParVector         *xtemp               = hypre_ParILUDataXTemp(ilu_data);
   hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
   HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
   hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);
   HYPRE_Real              zero                 = 0.0;
   HYPRE_Real              one                  = 1.0;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */
   /* more recent versions of cusparse require zeroing of the matrix meta data nrow/nnz, in order to fake a zero diagonal
      PJM 4/8/2022 */
   hypre_CSRMatrixI(A_diag)                     = A_diag_fake_i;
   HYPRE_Int t1 = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int t2 = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrixNumRows(A_diag) = 0;
   hypre_CSRMatrixNumNonzeros(A_diag) = 0;

   hypre_ParCSRMatrixMatvec( alpha, (hypre_ParCSRMatrix *) A, (hypre_ParVector *) x, zero, xtemp );

   hypre_CSRMatrixNumRows(A_diag) = t1;
   hypre_CSRMatrixNumNonzeros(A_diag) = t2;
   hypre_CSRMatrixI(A_diag)                     = A_diag_i;

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another word, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if ( A_diag_n > 0 )
   {
	   /* L solve - Forward solve */
	   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													   A_diag_n, A_diag_nnz, &one,
													   hypre_GpuMatDataMatDescr(hypre_ParILUDataMatLMatData(ilu_data)),
													   A_diag_data, A_diag_i, A_diag_j,
													   hypre_CsrsvDataInfoL(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
													   xtemp_data, ytemp_data,
													   hypre_CsrsvDataSolvePolicy(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
													   hypre_CsrsvDataBuffer(hypre_ParILUDataMatSLUCsrsvData(ilu_data))));

	   /* U solve - Backward substitution */
	   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													   A_diag_n, A_diag_nnz, &one,
													   hypre_GpuMatDataMatDescr(hypre_ParILUDataMatUMatData(ilu_data)),
													   A_diag_data, A_diag_i, A_diag_j,
													   hypre_CsrsvDataInfoU(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
													   ytemp_data, xtemp_data,
													   hypre_CsrsvDataSolvePolicy(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
													   hypre_CsrsvDataBuffer(hypre_ParILUDataMatSLUCsrsvData(ilu_data))));
   }

   /* now add the original x onto it */
   hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x, (hypre_ParVector *) xtemp);

   /* finall, add that into y and get final result */
   hypre_ParVectorScale( beta, (hypre_ParVector *) y );
   hypre_ParVectorAxpy( one, xtemp, (hypre_ParVector *) y);

   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ParILURocsparseSchurGMRESMatvec( void   *matvec_data,
                                      HYPRE_Complex  alpha,
                                      void   *ilu_vdata,
                                      void   *x,
                                      HYPRE_Complex  beta,
                                      void   *y           )
{
   /* get matrix information first */
   hypre_ParILUData *ilu_data                   = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A                        = hypre_ParILUDataMatS(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   hypre_CSRMatrix   *A_diag                    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int         *A_diag_i                  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         *A_diag_j                  = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real        *A_diag_data               = hypre_CSRMatrixData(A_diag);
   HYPRE_Int         A_diag_n                   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int         A_diag_nnz                 = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int         *A_diag_fake_i             = hypre_ParILUDataMatAFakeDiagonal(ilu_data);

   hypre_ParVector         *xtemp               = hypre_ParILUDataXTemp(ilu_data);
   hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
   HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
   hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);
   HYPRE_Real              zero                 = 0.0;
   HYPRE_Real              one                  = 1.0;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */
   /* more recent versions of cusparse require zeroing of the matrix meta data nrow/nnz, in order to fake a zero diagonal
      PJM 4/8/2022 */
   hypre_CSRMatrixI(A_diag)                     = A_diag_fake_i;
   HYPRE_Int t1 = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int t2 = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrixNumRows(A_diag) = 0;
   hypre_CSRMatrixNumNonzeros(A_diag) = 0;

   hypre_ParCSRMatrixMatvec( alpha, (hypre_ParCSRMatrix *) A, (hypre_ParVector *) x, zero, xtemp );

   hypre_CSRMatrixNumRows(A_diag) = t1;
   hypre_CSRMatrixNumNonzeros(A_diag) = t2;
   hypre_CSRMatrixI(A_diag)                     = A_diag_i;

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another word, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if ( A_diag_n > 0 )
   {
	   /* L solve - Forward solve */
	   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														 A_diag_n, A_diag_nnz, &one,
														 hypre_GpuMatDataMatDescr(hypre_ParILUDataMatLMatData(ilu_data)),
														 A_diag_data, A_diag_i, A_diag_j,
														 hypre_CsrsvDataInfoL(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
														 xtemp_data, ytemp_data,
														 hypre_CsrsvDataSolvePolicy(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
														 hypre_CsrsvDataBuffer(hypre_ParILUDataMatSLUCsrsvData(ilu_data))));

	   /* U solve - Backward substitution */
	   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														 A_diag_n, A_diag_nnz, &one,
														 hypre_GpuMatDataMatDescr(hypre_ParILUDataMatUMatData(ilu_data)),
														 A_diag_data, A_diag_i, A_diag_j,
														 hypre_CsrsvDataInfoU(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
														 ytemp_data, xtemp_data,
														 hypre_CsrsvDataSolvePolicy(hypre_ParILUDataMatSLUCsrsvData(ilu_data)),
														 hypre_CsrsvDataBuffer(hypre_ParILUDataMatSLUCsrsvData(ilu_data))));
   }

   /* now add the original x onto it */
   hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x, (hypre_ParVector *) xtemp);

   /* finall, add that into y and get final result */
   hypre_ParVectorScale( beta, (hypre_ParVector *) y );
   hypre_ParVectorAxpy( one, xtemp, (hypre_ParVector *) y);

   return hypre_error_flag;
}

#endif


/*********************************************************************************/
/*                   hypre_ILUSolveDeviceSchurGMRES                              */
/*********************************************************************************/

/* Schur Complement solve with GMRES on schur complement
 * ParCSRMatrix S is already built in ilu data sturcture, here directly use S
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
*/
#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ILUSolveDeviceSchurGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u, HYPRE_Int *perm,
                               HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                               hypre_ParVector *ftemp, hypre_ParVector *utemp,
                               HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                               hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end,
                               hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                               hypre_CsrsvData *matBLU_csrsvdata, hypre_CsrsvData *matSLU_csrsvdata,
                               hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d)
{
#if defined(HYPRE_USING_CUSPARSE)
   hypre_ILUSolveCusparseSchurGMRES(A, f, u, perm, nLU, S, ftemp, utemp,
                                    schur_solver, schur_precond, rhs, x, u_end,
                                    matL_des, matU_des, matBLU_csrsvdata, matSLU_csrsvdata,
                                    matBLU_d, matE_d, matF_d);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   hypre_ILUSolveRocsparseSchurGMRES(A, f, u, perm, nLU, S, ftemp, utemp,
                                    schur_solver, schur_precond, rhs, x, u_end,
                                    matL_des, matU_des, matBLU_csrsvdata, matSLU_csrsvdata,
                                    matBLU_d, matE_d, matF_d);
#endif
   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypre_ILUSolveCusparseSchurGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u, HYPRE_Int *perm,
                               HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                               hypre_ParVector *ftemp, hypre_ParVector *utemp,
                               HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                               hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end,
                               hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                               hypre_CsrsvData *matBLU_csrsvdata, hypre_CsrsvData *matSLU_csrsvdata,
                               hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      /* Just call BJ cusparse and return */
      return hypre_ILUSolveCusparseLU(A, matL_des, matU_des, matBLU_csrsvdata, matBLU_d,
                                      f, u, perm, nLU, ftemp, utemp);
   }

   /* data objects for communication */
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* data objects for temp vector */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);
   hypre_Vector      *rhs_local   = hypre_ParVectorLocalVector(rhs);
   HYPRE_Real        *rhs_data    = hypre_VectorData(rhs_local);
   hypre_Vector      *x_local     = hypre_ParVectorLocalVector(x);
   HYPRE_Real        *x_data      = hypre_VectorData(x_local);

   HYPRE_Real        alpha;
   HYPRE_Real        beta;
   //HYPRE_Real        gamma;
   //HYPRE_Int         i, j, k1, k2, col;

   /* problem size */
   HYPRE_Int         *BLU_i      = NULL;
   HYPRE_Int         *BLU_j      = NULL;
   HYPRE_Real        *BLU_data   = NULL;
   HYPRE_Int         BLU_nnz     = 0;
   hypre_CSRMatrix   *matSLU_d   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int         *SLU_i      = hypre_CSRMatrixI(matSLU_d);
   HYPRE_Int         *SLU_j      = hypre_CSRMatrixJ(matSLU_d);
   HYPRE_Real        *SLU_data   = hypre_CSRMatrixData(matSLU_d);
   HYPRE_Int         m           = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int         n           = nLU + m;
   HYPRE_Int         SLU_nnz     = hypre_CSRMatrixNumNonzeros(matSLU_d);

   hypre_Vector *ftemp_upper           = hypre_SeqVectorCreate(nLU);
   hypre_Vector *utemp_lower           = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(ftemp_upper)   = 0;
   hypre_VectorOwnsData(utemp_lower)   = 0;
   hypre_VectorData(ftemp_upper)       = ftemp_data;
   hypre_VectorData(utemp_lower)       = utemp_data + nLU;
   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);

   //printf("%s %s %d : nLU=%d,  m=%d,  ftemp size-(nLU+m)=%d\n",__FILE__,__FUNCTION__,__LINE__,nLU,m,hypre_VectorSize(ftemp_local)-nLU-m);

   if ( nLU > 0)
   {
      BLU_i                      = hypre_CSRMatrixI(matBLU_d);
      BLU_j                      = hypre_CSRMatrixJ(matBLU_d);
      BLU_data                   = hypre_CSRMatrixData(matBLU_d);
      BLU_nnz                    = hypre_CSRMatrixNumNonzeros(matBLU_d);
   }

   /* begin */
   beta = 1.0;
   alpha = -1.0;
   //gamma = 0.0;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* apply permutation before we can start our solve */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   if (nLU > 0)
   {
      /* This solve won't touch data in utemp, thus, gi is still in utemp_lower */
	   /* L solve - Forward solve */
	   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													   nLU, BLU_nnz, &beta, hypre_GpuMatDataMatDescr(matL_des),
													   BLU_data, BLU_i, BLU_j, hypre_CsrsvDataInfoL(matBLU_csrsvdata),
													   utemp_data, ftemp_data,
													   hypre_CsrsvDataSolvePolicy(matBLU_csrsvdata),
													   hypre_CsrsvDataBuffer(matBLU_csrsvdata) ));
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

   /* setup vectors for solve
    * rhs = M^{-1}g'
    */

   if (m > 0)
   {
         /* L solve */
	   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													   m, SLU_nnz, &beta, hypre_GpuMatDataMatDescr(matL_des),
													   SLU_data, SLU_i, SLU_j, hypre_CsrsvDataInfoL(matSLU_csrsvdata),
													   utemp_data + nLU, ftemp_data + nLU,
													   hypre_CsrsvDataSolvePolicy(matSLU_csrsvdata),
													   hypre_CsrsvDataBuffer(matSLU_csrsvdata) ));

	   /* U solve */
	   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													   m, SLU_nnz, &beta, hypre_GpuMatDataMatDescr(matU_des),
													   SLU_data, SLU_i, SLU_j, hypre_CsrsvDataInfoU(matSLU_csrsvdata),
													   ftemp_data + nLU, rhs_data,
													   hypre_CsrsvDataSolvePolicy(matSLU_csrsvdata),
													   hypre_CsrsvDataBuffer(matSLU_csrsvdata) ));
   }


   /* solve */
   /* with tricky initial guess */
   //hypre_Vector *tv = hypre_ParVectorLocalVector(x);
   //HYPRE_Real *tz = hypre_VectorData(tv);
   HYPRE_GMRESSolve(schur_solver, (HYPRE_Matrix)schur_precond, (HYPRE_Vector)rhs, (HYPRE_Vector)x);
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

	  /* U solve - Forward solve */
	  HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													  nLU, BLU_nnz, &beta, hypre_GpuMatDataMatDescr(matU_des),
													  BLU_data, BLU_i, BLU_j, hypre_CsrsvDataInfoU(matBLU_csrsvdata),
													  ftemp_data, utemp_data,
													  hypre_CsrsvDataSolvePolicy(matBLU_csrsvdata),
													  hypre_CsrsvDataBuffer(matBLU_csrsvdata) ));
   }

   /* copy lower part solution into u_temp as well */
   hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* perm back */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);

   /* done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);

   return hypre_error_flag;
}

#endif


#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ILUSolveRocsparseSchurGMRES(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u, HYPRE_Int *perm,
                                  HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                                  hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                  HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                  hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end,
                                  hypre_GpuMatData * matL_des, hypre_GpuMatData * matU_des,
                                  hypre_CsrsvData *matBLU_csrsvdata, hypre_CsrsvData *matSLU_csrsvdata,
                                  hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      /* Just call BJ cusparse and return */
      return hypre_ILUSolveRocsparseLU(A, matL_des, matU_des, matBLU_csrsvdata, matBLU_d,
                                       f, u, perm, nLU, ftemp, utemp);
   }

   /* data objects for communication */
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* data objects for temp vector */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);
   hypre_Vector      *rhs_local   = hypre_ParVectorLocalVector(rhs);
   HYPRE_Real        *rhs_data    = hypre_VectorData(rhs_local);
   hypre_Vector      *x_local     = hypre_ParVectorLocalVector(x);
   HYPRE_Real        *x_data      = hypre_VectorData(x_local);

   HYPRE_Real        alpha;
   HYPRE_Real        beta;
   //HYPRE_Real        gamma;
   //HYPRE_Int         i, j, k1, k2, col;

   /* problem size */
   HYPRE_Int         *BLU_i      = NULL;
   HYPRE_Int         *BLU_j      = NULL;
   HYPRE_Real        *BLU_data   = NULL;
   HYPRE_Int         BLU_nnz     = 0;
   hypre_CSRMatrix   *matSLU_d   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int         *SLU_i      = hypre_CSRMatrixI(matSLU_d);
   HYPRE_Int         *SLU_j      = hypre_CSRMatrixJ(matSLU_d);
   HYPRE_Real        *SLU_data   = hypre_CSRMatrixData(matSLU_d);
   HYPRE_Int         m           = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int         n           = nLU + m;
   HYPRE_Int         SLU_nnz     = hypre_CSRMatrixNumNonzeros(matSLU_d);

   hypre_Vector *ftemp_upper           = hypre_SeqVectorCreate(nLU);
   hypre_Vector *utemp_lower           = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(ftemp_upper)   = 0;
   hypre_VectorOwnsData(utemp_lower)   = 0;
   hypre_VectorData(ftemp_upper)       = ftemp_data;
   hypre_VectorData(utemp_lower)       = utemp_data + nLU;
   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);

   //printf("%s %s %d : nLU=%d,  m=%d,  ftemp size-(nLU+m)=%d\n",__FILE__,__FUNCTION__,__LINE__,nLU,m,hypre_VectorSize(ftemp_local)-nLU-m);

   if ( nLU > 0)
   {
      BLU_i                      = hypre_CSRMatrixI(matBLU_d);
      BLU_j                      = hypre_CSRMatrixJ(matBLU_d);
      BLU_data                   = hypre_CSRMatrixData(matBLU_d);
      BLU_nnz                    = hypre_CSRMatrixNumNonzeros(matBLU_d);
   }

   /* begin */
   beta = 1.0;
   alpha = -1.0;
   //gamma = 0.0;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* apply permutation before we can start our solve */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   if (nLU > 0)
   {
      /* This solve won't touch data in utemp, thus, gi is still in utemp_lower */
	   /* L solve - Forward solve */
	   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														 nLU, BLU_nnz, &beta, hypre_GpuMatDataMatDescr(matL_des),
														 BLU_data, BLU_i, BLU_j, hypre_CsrsvDataInfoL(matBLU_csrsvdata),
														 utemp_data, ftemp_data,
														 hypre_CsrsvDataSolvePolicy(matBLU_csrsvdata),
														 hypre_CsrsvDataBuffer(matBLU_csrsvdata) ));
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

   /* setup vectors for solve
    * rhs = M^{-1}g'
    */

   if (m > 0)
   {
	   /* L solve */
	   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														 m, SLU_nnz, &beta, hypre_GpuMatDataMatDescr(matL_des),
														 SLU_data, SLU_i, SLU_j, hypre_CsrsvDataInfoL(matSLU_csrsvdata),
														 utemp_data + nLU, ftemp_data + nLU,
														 hypre_CsrsvDataSolvePolicy(matSLU_csrsvdata),
														 hypre_CsrsvDataBuffer(matSLU_csrsvdata) ));

	   /* U solve */
	   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														 m, SLU_nnz, &beta, hypre_GpuMatDataMatDescr(matU_des),
														 SLU_data, SLU_i, SLU_j, hypre_CsrsvDataInfoU(matSLU_csrsvdata),
														 ftemp_data + nLU, rhs_data,
														 hypre_CsrsvDataSolvePolicy(matSLU_csrsvdata),
														 hypre_CsrsvDataBuffer(matSLU_csrsvdata) ));
   }


   /* solve */
   /* with tricky initial guess */
   //hypre_Vector *tv = hypre_ParVectorLocalVector(x);
   //HYPRE_Real *tz = hypre_VectorData(tv);
   HYPRE_GMRESSolve(schur_solver, (HYPRE_Matrix)schur_precond, (HYPRE_Vector)rhs, (HYPRE_Vector)x);
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
	  /* U solve - Forward solve */
	  HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
														nLU, BLU_nnz, &beta, hypre_GpuMatDataMatDescr(matU_des),
														BLU_data, BLU_i, BLU_j, hypre_CsrsvDataInfoU(matBLU_csrsvdata),
														ftemp_data, utemp_data,
														hypre_CsrsvDataSolvePolicy(matBLU_csrsvdata),
														hypre_CsrsvDataBuffer(matBLU_csrsvdata) ));
   }

   /* copy lower part solution into u_temp as well */
   hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* perm back */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);

   /* done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);

   return hypre_error_flag;
}

#endif




#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*********************************************************************************/
/*                   hypre_ILUSolveDeviceSchurGMRESIter                          */
/*********************************************************************************/

/* Schur Complement solve with GMRES on schur complement
 * ParCSRMatrix S is already built in ilu data sturcture, here directly use S
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
*/

HYPRE_Int
hypre_ILUSolveDeviceSchurGMRESIter(hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                   hypre_ParVector *u, HYPRE_Int *perm,
                                   HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                                   hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                   HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                   hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end,
                                   hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d,
                                   hypre_Vector *ztemp_local, hypre_Vector **Adiag_diag, hypre_Vector **Sdiag_diag,
                                   HYPRE_Int lower_jacobi_iters, HYPRE_Int upper_jacobi_iters)
{
   /* If we don't have S block, just do one L solve and one U solve */
   if (!S)
   {
      /* Just call BJ cusparse and return */
      return hypre_ILUSolveDeviceLUIter(A, matBLU_d,
                                        f, u, perm, nLU, ftemp, utemp, ztemp_local, Adiag_diag,
                                        lower_jacobi_iters, upper_jacobi_iters);
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* data objects for communication */
   //   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   /* data objects for temp vector */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);
   hypre_Vector      *rhs_local   = hypre_ParVectorLocalVector(rhs);
   hypre_Vector      *x_local     = hypre_ParVectorLocalVector(x);
   HYPRE_Real        *x_data      = hypre_VectorData(x_local);

   HYPRE_Real        alpha;
   HYPRE_Real        beta;
   //HYPRE_Real        gamma;
   //HYPRE_Int         i, j, k1, k2, col;

   /* problem size */
   hypre_CSRMatrix   *matSLU_d   = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int         m           = hypre_CSRMatrixNumRows(matSLU_d);
   HYPRE_Int         n           = nLU + m;

   hypre_Vector *ftemp_upper           = hypre_SeqVectorCreate(nLU);
   hypre_Vector *utemp_lower           = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(ftemp_upper)   = 0;
   hypre_VectorOwnsData(utemp_lower)   = 0;
   hypre_VectorData(ftemp_upper)       = ftemp_data;
   hypre_VectorData(utemp_lower)       = utemp_data + nLU;
   hypre_SeqVectorInitialize(ftemp_upper);
   hypre_SeqVectorInitialize(utemp_lower);


   hypre_Vector *ftemp_shift           = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(ftemp_shift)   = 0;
   hypre_VectorData(ftemp_shift)       = ftemp_data+nLU;
   hypre_SeqVectorInitialize(ftemp_shift);

   hypre_Vector *utemp_shift           = hypre_SeqVectorCreate(m);
   hypre_VectorOwnsData(utemp_shift)   = 0;
   hypre_VectorData(utemp_shift)       = utemp_data+nLU;
   hypre_SeqVectorInitialize(utemp_shift);

   //printf("%s %s %d : nLU=%d,  m=%d,  ftemp size-(nLU+m)=%d\n",__FILE__,__FUNCTION__,__LINE__,nLU,m,hypre_VectorSize(ftemp_local)-nLU-m);

   /* begin */
   beta = 1.0;
   alpha = -1.0;
   //gamma = 0.0;

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!(*Adiag_diag)) {
      /* storage for the diagonal */
      *Adiag_diag = hypre_SeqVectorCreate(n);
      hypre_SeqVectorInitialize(*Adiag_diag);
      /* extract with device kernel */
      hypre_CSRMatrixExtractDiagonalDevice(matBLU_d, hypre_VectorData(*Adiag_diag), 2);
      //hypre_CSRMatrixGetMainDiag(matLU_d, *Adiag);
   }

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */

   /* apply permutation before we can start our solve */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   if (nLU > 0)
   {

      /* apply the iterative solve to L */
      hypre_ILUSolveLJacobiIter(matBLU_d, utemp_local, ztemp_local, ftemp_local, lower_jacobi_iters);

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

   /* setup vectors for solve
    * rhs = M^{-1}g'
    */


   if (m > 0)
   {
      /* Grab the main diagonal from the diagonal block. Only do this once */
      if (!(*Sdiag_diag)) {
         /* storage for the diagonal */
         *Sdiag_diag = hypre_SeqVectorCreate(m);
         hypre_SeqVectorInitialize(*Sdiag_diag);
         /* extract with device kernel */
         hypre_CSRMatrixExtractDiagonalDevice(matSLU_d, hypre_VectorData(*Sdiag_diag), 2);
         //hypre_CSRMatrixGetMainDiag(matLU_d, *Adiag);
      }

      /* apply the iterative solve to L */
      hypre_ILUSolveLJacobiIter(matSLU_d, utemp_shift, rhs_local, ftemp_shift, lower_jacobi_iters);

      hypre_ILUSolveUJacobiIter(matSLU_d, ftemp_shift, utemp_shift, rhs_local, *Sdiag_diag, upper_jacobi_iters);

      /* apply the iterative solve to L and U */
      //hypre_ILUSolveLUJacobiIter(matSLU_d, utemp_shift, ftemp_shift, rhs_local, *Sdiag_diag,
      //				 lower_jacobi_iters, upper_jacobi_iters, my_id);

   }

   /* solve */
   /* with tricky initial guess */
   //hypre_Vector *tv = hypre_ParVectorLocalVector(x);
   //HYPRE_Real *tz = hypre_VectorData(tv);
   HYPRE_GMRESSolve(schur_solver, (HYPRE_Matrix)schur_precond, (HYPRE_Vector)rhs, (HYPRE_Vector)x);
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

      /* apply the iterative solve to U */
      hypre_ILUSolveUJacobiIter(matBLU_d, ftemp_local, ztemp_local, utemp_local, *Adiag_diag, upper_jacobi_iters);

   }

   /* copy lower part solution into u_temp as well */
   hypre_TMemcpy(utemp_data + nLU, x_data, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* perm back */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);

   /* done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   hypre_SeqVectorDestroy(ftemp_shift);
   hypre_SeqVectorDestroy(utemp_shift);
   hypre_SeqVectorDestroy(ftemp_upper);
   hypre_SeqVectorDestroy(utemp_lower);
   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_ParILUDeviceSchurGMRESMatvecJacobiIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUDeviceSchurGMRESMatvecJacobiIter( void   *matvec_data,
                                                HYPRE_Complex  alpha,
                                                void   *ilu_vdata,
                                                void   *x,
                                                HYPRE_Complex  beta,
                                                void   *y           )
{
   /* Slightly different, for this new matvec, the diagonal of the original matrix
    * is the LU factorization. Thus, the matvec is done in an different way
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
    *
    * */

   /* get matrix information first */
   hypre_ParILUData *ilu_data                   = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A                        = hypre_ParILUDataMatS(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   hypre_CSRMatrix   *A_diag                    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int         *A_diag_i                  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         A_diag_n                   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int         *A_diag_fake_i             = hypre_ParILUDataMatAFakeDiagonal(ilu_data);

   hypre_ParVector         *xtemp               = hypre_ParILUDataXTemp(ilu_data);
   hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
   hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real              zero                 = 0.0;
   HYPRE_Real              one                  = 1.0;

   hypre_Vector            *ztemp_local         = hypre_ParILUDataZTemp(ilu_data);
   hypre_Vector            *SchurMatVec_diag    = hypre_ParILUDataSchurMatVecDiag(ilu_data);

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */

   /* more recent versions of cusparse require zeroing of the matrix meta data nrow/nnz, in order to fake a zero diagonal
      PJM 4/8/2022 */
   hypre_CSRMatrixI(A_diag)                     = A_diag_fake_i;
   HYPRE_Int t1 = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int t2 = hypre_CSRMatrixNumNonzeros(A_diag);
   hypre_CSRMatrixNumRows(A_diag) = 0;
   hypre_CSRMatrixNumNonzeros(A_diag) = 0;

   hypre_ParCSRMatrixMatvec( alpha, (hypre_ParCSRMatrix *) A, (hypre_ParVector *) x, zero, xtemp );

   hypre_CSRMatrixNumRows(A_diag) = t1;
   hypre_CSRMatrixNumNonzeros(A_diag) = t2;
   hypre_CSRMatrixI(A_diag)                     = A_diag_i;


   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another word, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if ( A_diag_n > 0 )
   {
      /* Grab the main diagonal from the diagonal block. Only do this once */
      if (!SchurMatVec_diag) {
         /* storage for the diagonal */
         SchurMatVec_diag = hypre_SeqVectorCreate(A_diag_n);
         hypre_SeqVectorInitialize(SchurMatVec_diag);
         hypre_ParILUDataSchurMatVecDiag(ilu_data) = SchurMatVec_diag;
         /* extract with device kernel */
         hypre_CSRMatrixExtractDiagonalDevice(A_diag, hypre_VectorData(SchurMatVec_diag), 2);
      }

      /* apply the iterative solve to L and U */
      hypre_ILUSolveLUJacobiIter(A_diag, ytemp_local, ztemp_local, xtemp_local, SchurMatVec_diag,
                                 hypre_ParILUDataLowerJacobiIters(ilu_data), hypre_ParILUDataUpperJacobiIters(ilu_data), 0);
   }

   /* now add the original x onto it */
   hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x, (hypre_ParVector *) xtemp);

   /* finall, add that into y and get final result */
   hypre_ParVectorScale( beta, (hypre_ParVector *) y );
   hypre_ParVectorAxpy( one, xtemp, (hypre_ParVector *) y);

   return hypre_error_flag;
}


#endif
