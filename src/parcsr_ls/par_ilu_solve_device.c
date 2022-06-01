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
/*                   hypre_ILUSolveDeviceLU                                      */
/*********************************************************************************/
/* Incomplete LU solve (GPU)
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/

HYPRE_Int
hypre_ILUSolveDeviceLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                       hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                       hypre_CSRMatrix *matLU_d, hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                       HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
#if defined(HYPRE_USING_CUSPARSE)
   hypre_ILUSolveCusparseLU(A, matL_des, matU_des, matLU_csrsvdata,
                            matLU_d, f,  u, perm, n, ftemp, utemp);
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   hypre_ILUSolveRocsparseLU(A, matL_des, matU_des, matLU_csrsvdata,
                             matLU_d, f,  u, perm, n, ftemp, utemp);
#endif
   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypre_ILUSolveCusparseLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                         hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                         hypre_CSRMatrix *matLU_d,
                         hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                         HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return hypre_error_flag;
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* ILU data */
   HYPRE_Real              *LU_data             = hypre_CSRMatrixData(matLU_d);
   HYPRE_Int               *LU_i                = hypre_CSRMatrixI(matLU_d);
   HYPRE_Int               *LU_j                = hypre_CSRMatrixJ(matLU_d);
   HYPRE_Int               nnz                  = hypre_CSRMatrixNumNonzeros(matLU_d);

   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   HYPRE_Real              alpha;
   HYPRE_Real              beta;
   //HYPRE_Int               i, j, k1, k2;

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   /* L solve - Forward solve */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
												   n, nnz, &beta, hypre_GpuMatDataMatDescr(matL_des),
												   LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
												   utemp_data, ftemp_data,
												   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
												   hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

   /* U solve - Backward substitution */
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
												   n, nnz, &beta, hypre_GpuMatDataMatDescr(matU_des),
												   LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
												   ftemp_data, utemp_data,
												   hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
												   hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

   /* apply reverse permutation */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);


   return hypre_error_flag;
}

#endif

#if defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ILUSolveRocsparseLU(hypre_ParCSRMatrix *A, hypre_GpuMatData * matL_des,
                          hypre_GpuMatData * matU_des, hypre_CsrsvData * matLU_csrsvdata,
                          hypre_CSRMatrix *matLU_d,
                          hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                          HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return hypre_error_flag;
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   /* ILU data */
   HYPRE_Real              *LU_data             = hypre_CSRMatrixData(matLU_d);
   HYPRE_Int               *LU_i                = hypre_CSRMatrixI(matLU_d);
   HYPRE_Int               *LU_j                = hypre_CSRMatrixJ(matLU_d);
   HYPRE_Int               nnz                  = hypre_CSRMatrixNumNonzeros(matLU_d);

   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   HYPRE_Real              alpha;
   HYPRE_Real              beta;
   //HYPRE_Int               i, j, k1, k2;

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   /* L solve - Forward solve */
   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none, n, nnz,
													 &beta, hypre_GpuMatDataMatDescr(matL_des),
													 LU_data, LU_i, LU_j, hypre_CsrsvDataInfoL(matLU_csrsvdata),
													 utemp_data, ftemp_data,
													 hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
													 hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

   /* U solve - Backward substitution */
   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none, n, nnz,
													 &beta, hypre_GpuMatDataMatDescr(matU_des),
													 LU_data, LU_i, LU_j, hypre_CsrsvDataInfoU(matLU_csrsvdata),
													 ftemp_data, utemp_data,
													 hypre_CsrsvDataSolvePolicy(matLU_csrsvdata),
													 hypre_CsrsvDataBuffer(matLU_csrsvdata) ));

   /* apply reverse permutation */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);
   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);


   return hypre_error_flag;
}

#endif


/*********************************************************************************/
/*                   hypre_ILUSolveDeviceLUIter                                  */
/*********************************************************************************/
/* Incomplete LU solve (GPU) using Jacobi iterative approach
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypre_ILUSolveLJacobiIter(hypre_CSRMatrix *A, hypre_Vector *input_local, hypre_Vector *work_local,
                          hypre_Vector *output_local, HYPRE_Int lower_jacobi_iters)
{
   HYPRE_Real              *input_data          = hypre_VectorData(input_local);
   HYPRE_Real              *work_data           = hypre_VectorData(work_local);
   HYPRE_Real              *output_data         = hypre_VectorData(output_local);
   HYPRE_Int               num_rows             = hypre_CSRMatrixNumRows(A);
   HYPRE_Int kk=0;

   /* L solve - Forward solve ; u^{k+1} = f - Lu^k*/
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first L SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_zeqxmy(num_rows, input_data, 0.0, work_data, output_data);

   /* Do the remaining iterations */
   for( kk = 1; kk < lower_jacobi_iters; ++kk ) {

       /* apply SpMV */
       hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, NULL, -2);

       /* transform */
       hypreDevice_zeqxmy(num_rows, input_data, -1.0, work_data, output_data);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUSolveUJacobiIter(hypre_CSRMatrix *A, hypre_Vector *input_local, hypre_Vector *work_local,
                          hypre_Vector *output_local, hypre_Vector *diag_diag, HYPRE_Int upper_jacobi_iters)
{
   HYPRE_Real              *output_data         = hypre_VectorData(output_local);
   HYPRE_Real              *work_data           = hypre_VectorData(work_local);
   HYPRE_Real              *input_data          = hypre_VectorData(input_local);
   HYPRE_Real              *diag_diag_data      = hypre_VectorData(diag_diag);
   HYPRE_Int               num_rows             = hypre_CSRMatrixNumRows(A);
   HYPRE_Int kk=0;

   /* U solve - Backward solve :  u^{k+1} = f - Uu^k */
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first U SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_zeqxmydd(num_rows, input_data, 0.0, work_data, output_data, diag_diag_data);

   /* Do the remaining iterations */
   for( kk = 1; kk < upper_jacobi_iters; ++kk ) {

       /* apply SpMV */
       hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, NULL, 2);

       /* transform */
       hypreDevice_zeqxmydd(num_rows, input_data, -1.0, work_data, output_data, diag_diag_data);
   }

   return hypre_error_flag;
}


HYPRE_Int
hypre_ILUSolveLUJacobiIter(hypre_CSRMatrix *A, hypre_Vector *work1_local,
                           hypre_Vector *work2_local, hypre_Vector *inout_local, hypre_Vector *diag_diag,
                           HYPRE_Int lower_jacobi_iters, HYPRE_Int upper_jacobi_iters, HYPRE_Int my_id)
{
#if 1
   /* apply the iterative solve to L */
   hypre_ILUSolveLJacobiIter(A, inout_local, work1_local, work2_local, lower_jacobi_iters);

   /* apply the iterative solve to U */
   hypre_ILUSolveUJacobiIter(A, work2_local, work1_local, inout_local, diag_diag, upper_jacobi_iters);

#else
   HYPRE_Real              *inout_data          = hypre_VectorData(inout_local);
   HYPRE_Real              *work1_data          = hypre_VectorData(work1_local);
   HYPRE_Real              *work2_data          = hypre_VectorData(work2_local);
   HYPRE_Real              *diag_diag_data      = hypre_VectorData(diag_diag);
   HYPRE_Int               num_rows             = hypre_CSRMatrixNumRows(A);
   HYPRE_Int kk=0;

   /* L solve - Forward solve ; u^{k+1} = f - Lu^k*/
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first L SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_zeqxmy(num_rows, inout_data, 0.0, work1_data, work2_data);

   /* Do the remaining iterations */
   for( kk = 1; kk < lower_jacobi_iters-1; ++kk ) {

       /* apply SpMV */
       hypre_CSRMatrixSpMVDevice(0, 1.0, A, work2_local, 0.0, work1_local, NULL, -2);

       /* transform */
       hypreDevice_zeqxmy(num_rows, inout_data, -1.0, work1_data, work2_data);
   }

   /* apply SpMV */
   hypre_CSRMatrixSpMVDevice(0, 1.0, A, work2_local, 0.0, work1_local, NULL, -2);

   /* transform */
   //hypreDevice_zeqxmy(num_rows, inout_data, -1.0, ytemp_data, work2_data);

   /* U solve - Backward solve :  u^{k+1} = f - Uu^k */
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first U SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   //hypreDevice_zeqxmydd(num_rows, work2_data, 0.0, ytemp_data, inout_data, diag_diag_data);

   // This operation fuses the 2 vector operations above.
   hypreDevice_fused_vecop(num_rows, -1.0, work2_data, 0.0, work1_data, inout_data, diag_diag_data);

   /* Do the remaining iterations */
   for( kk = 1; kk < upper_jacobi_iters; ++kk ) {

       /* apply SpMV */
       hypre_CSRMatrixSpMVDevice(1.0, A, inout_local, 0.0, work1_local, 2);

       /* transform */
       hypreDevice_zeqxmydd(num_rows, work2_data, -1.0, work1_data, inout_data, diag_diag_data);
   }

#endif

   return hypre_error_flag;
}


/* Incomplete LU solve using jacobi iterations on GPU
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/
HYPRE_Int
hypre_ILUSolveDeviceLUIter(hypre_ParCSRMatrix *A, hypre_CSRMatrix *matLU_d,
                             hypre_ParVector *f,  hypre_ParVector *u, HYPRE_Int *perm,
                             HYPRE_Int n, hypre_ParVector *ftemp, hypre_ParVector *utemp,
                             hypre_Vector *xtemp_local, hypre_Vector **Adiag_diag,
                             HYPRE_Int lower_jacobi_iters, HYPRE_Int upper_jacobi_iters)
{
   /* Only solve when we have stuffs to be solved */
   if (n == 0)
   {
      return hypre_error_flag;
   }

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int my_id;
   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   HYPRE_Real              alpha;
   HYPRE_Real              beta;

   /* begin */
   alpha = -1.0;
   beta = 1.0;

   /* Grab the main diagonal from the diagonal block. Only do this once */
   if (!(*Adiag_diag)) {
      /* storage for the diagonal */
      *Adiag_diag = hypre_SeqVectorCreate(n);
      hypre_SeqVectorInitialize(*Adiag_diag);
      /* extract with device kernel */
      hypre_CSRMatrixExtractDiagonalDevice(matLU_d, hypre_VectorData(*Adiag_diag), 2);
      //hypre_CSRMatrixGetMainDiag(matLU_d, *Adiag);
   }

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
   */

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* apply permutation */
   HYPRE_THRUST_CALL(gather, perm, perm + n, ftemp_data, utemp_data);

   /* apply the iterative solve to L and U */
   hypre_ILUSolveLUJacobiIter(matLU_d, ftemp_local, xtemp_local, utemp_local, *Adiag_diag,
                              lower_jacobi_iters, upper_jacobi_iters, my_id);

   /* apply reverse permutation */
   HYPRE_THRUST_CALL(scatter, utemp_data, utemp_data + n, perm, ftemp_data);

   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   return hypre_error_flag;
}


#endif
