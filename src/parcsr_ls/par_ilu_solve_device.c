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
/*                   hypre_ILUSolveDeviceLUIter                                  */
/*********************************************************************************/
/* Incomplete LU solve (GPU) using Jacobi iterative approach
 * L, D and U factors only have local scope (no off-diagonal processor terms)
 * so apart from the residual calculation (which uses A), the solves with the
 * L and U factors are local.
*/

#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypre_ILUSolveLJacobiIter(hypre_CSRMatrix *A, hypre_Vector *input_local, hypre_Vector *work_local,
                          hypre_Vector *output_local, HYPRE_Int lower_jacobi_iters)
{
   HYPRE_Real              *input_data          = hypre_VectorData(input_local);
   HYPRE_Real              *work_data           = hypre_VectorData(work_local);
   HYPRE_Real              *output_data         = hypre_VectorData(output_local);
   HYPRE_Int               num_rows             = hypre_CSRMatrixNumRows(A);
   HYPRE_Int kk = 0;

   /* L solve - Forward solve ; u^{k+1} = f - Lu^k*/
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first L SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, 0.0);

   /* Do the remaining iterations */
   for ( kk = 1; kk < lower_jacobi_iters; ++kk )
   {

      /* apply SpMV */
      hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, -2);

      /* transform */
      hypreDevice_ComplexAxpyn(work_data, num_rows, input_data, output_data, -1.0);
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
   HYPRE_Int kk = 0;

   /* U solve - Backward solve :  u^{k+1} = f - Uu^k */
   /* Jacobi iteration loop */

   /* Since the initial guess to the jacobi iteration is 0, the result of the first U SpMV is 0, so no need to compute
      However, we still need to compute the transformation */
   hypreDevice_zeqxmydd(num_rows, input_data, 0.0, work_data, output_data, diag_diag_data);

   /* Do the remaining iterations */
   for ( kk = 1; kk < upper_jacobi_iters; ++kk )
   {

      /* apply SpMV */
      hypre_CSRMatrixSpMVDevice(0, 1.0, A, output_local, 0.0, work_local, 2);

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
   /* apply the iterative solve to L */
   hypre_ILUSolveLJacobiIter(A, inout_local, work1_local, work2_local, lower_jacobi_iters);

   /* apply the iterative solve to U */
   hypre_ILUSolveUJacobiIter(A, work2_local, work1_local, inout_local, diag_diag, upper_jacobi_iters);

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
   if (!(*Adiag_diag))
   {
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
