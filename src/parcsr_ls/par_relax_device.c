/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

HYPRE_Int
hypre_BoomerAMGRelaxHybridGaussSeidelDevice( hypre_ParCSRMatrix *A,
                                             hypre_ParVector    *f,
                                             HYPRE_Int          *cf_marker,
                                             HYPRE_Int           relax_points,
                                             HYPRE_Real          relax_weight,
                                             HYPRE_Real          omega,
                                             HYPRE_Real         *l1_norms,
                                             hypre_ParVector    *u,
                                             hypre_ParVector    *Vtemp,
                                             hypre_ParVector    *Ztemp,
                                             HYPRE_Int           GS_order,
                                             HYPRE_Int           Symm )
{
   /* Vtemp, Ztemp have the fine-grid size. Create two shell vectors that have the correct size */
   hypre_ParVector *w1 = hypre_ParVectorCloneShallow(f);
   hypre_ParVector *w2 = hypre_ParVectorCloneShallow(u);
   hypre_VectorData(hypre_ParVectorLocalVector(w1)) = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
   hypre_VectorData(hypre_ParVectorLocalVector(w2)) = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));

   if (Symm)
   {
      /* V = f - A*u */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, w1);

      /* Z = L^{-1}*V */
      hypre_CSRMatrixTriLowerUpperSolveCusparse('L', hypre_ParCSRMatrixDiag(A),
                                                hypre_ParVectorLocalVector(w1), hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      hypre_ParVectorAxpy(relax_weight, w2, u);

      /* Note: only update V from local change of u, i.e., V = V - w*A_diag*Z_local */
      hypre_CSRMatrixMatvec(-relax_weight, hypre_ParCSRMatrixDiag(A), hypre_ParVectorLocalVector(w2),
                            1.0, hypre_ParVectorLocalVector(w1));

      /* Z = U^{-1}*V */
      hypre_CSRMatrixTriLowerUpperSolveCusparse('U', hypre_ParCSRMatrixDiag(A),
                                                hypre_ParVectorLocalVector(w1), hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      hypre_ParVectorAxpy(relax_weight, w2, u);
   }
   else
   {
      const char uplo = GS_order > 0 ? 'L' : 'U';
      /* V = f - A*u */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, w1);

      /* Z = L^{-1}*V or Z = U^{-1}*V */
      hypre_CSRMatrixTriLowerUpperSolveCusparse(uplo, hypre_ParCSRMatrixDiag(A),
                                                hypre_ParVectorLocalVector(w1), hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      hypre_ParVectorAxpy(relax_weight, w2, u);
   }

   hypre_ParVectorDestroy(w1);
   hypre_ParVectorDestroy(w2);

   return hypre_error_flag;
}

/* option 0: inout = inout + D^{-1}*[(1+w)*r - w*A*x]
 * option 1: inout = inout + D^{-1}*[r - tril(A,-1)*x]
 * Note: r is modified */
void
hypre_TwoStageGaussSeidelMatvec(hypre_CSRMatrix *A,
                                hypre_Vector    *x,
                                HYPRE_Complex   *invdiag,
                                hypre_Vector    *r,
                                HYPRE_Complex    omega,
                                hypre_Vector    *inout,
                                HYPRE_Int        option)
{
   if (option == 0)
   {
      hypre_CSRMatrixMatvecDevice(0.0, -omega, A, x, 1.0 + omega, r, r, 0.0);
   }
   else
   {
      hypre_CSRMatrixSpMVDevice(-1.0, A, x, 1.0, r, -2);
   }
   /*
   HYPRE_THRUST_CALL( transform,
                      invdiag,
                      invdiag + hypre_CSRMatrixNumRows(A),
                      hypre_VectorData(r),
                      hypre_VectorData(r),
                      thrust::multiplies<HYPRE_Complex>() );
   */
   hypreDevice_DiagScaleVector(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixI(A), hypre_CSRMatrixData(A),
                               hypre_VectorData(r), 1.0, hypre_VectorData(inout));
}

HYPRE_Int
hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice ( hypre_ParCSRMatrix *A,
                                                hypre_ParVector    *f,
                                                HYPRE_Real          relax_weight,
                                                HYPRE_Real          omega,
                                                hypre_ParVector    *u,
                                                hypre_ParVector    *r,
                                                hypre_ParVector    *z,
                                                HYPRE_Int           choice)
{
   hypre_NvtxPushRange("BoomerAMGRelax11");

   /*
   if (hypre_ParVectorNumVectors(z) < 2)
   {
      hypre_Vector *old_z = hypre_ParVectorLocalVector(z);
      hypre_Vector *new_z = hypre_SeqMultiVectorCreate(hypre_VectorSize(old_z), 2);
      hypre_VectorMemoryLocation(new_z) = hypre_VectorMemoryLocation(old_z);
      hypre_SeqVectorDestroy(old_z);
      hypre_SeqVectorInitialize(new_z);
      hypre_ParVectorLocalVector(z) = new_z;
   }
   */
   hypre_CSRMatrix *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int        num_rows     = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex   *A_diag_data  = hypre_CSRMatrixData(A_diag);
   hypre_Vector    *u_local      = hypre_ParVectorLocalVector(u);
   hypre_Vector    *r_local      = hypre_ParVectorLocalVector(r);
   hypre_Vector    *z_local      = hypre_ParVectorLocalVector(z);
   HYPRE_Complex   *r_data       = hypre_VectorData(r_local);
   HYPRE_Complex   *z_data       = hypre_VectorData(z_local);

   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A_diag);

   HYPRE_Complex *workvector = z_data;
   hypre_Vector *w1 = hypre_SeqVectorCreate(num_rows);
   hypre_VectorData(w1) = workvector;
   hypre_VectorMemoryLocation(w1) = memory_location;
   hypre_VectorOwnsData(w1) = 0;

   HYPRE_Complex *workvector2 = NULL;

   hypre_ParVectorCopy(f, r);
   hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, r);

   /* Need to subtract out the diagonal matrix diagonal times u_data
    * because L/U have the diagonal built in for the solves */
   /*
   HYPRE_THRUST_CALL( transform,
                      r_data,
                      r_data + num_rows_diag,
                      workvector2,
                      workvector,
                      thrust::multiplies<HYPRE_Complex>() );
   */
   hypreDevice_DiagScaleVector(num_rows, A_diag_i, A_diag_data, r_data, 0.0, workvector);

   if (choice == 0)
   {
      /* spmv with the full matrix */
      hypre_TwoStageGaussSeidelMatvec(A_diag, w1, workvector2, r_local, omega, u_local, 0);
   }
   else if (choice == 1)
   {
      /* spmv with L */
      hypre_TwoStageGaussSeidelMatvec(A_diag, w1, workvector2, r_local, omega, u_local, 1);

      //spmvL(num_rows, nnz_diag, hypre_CSRMatrixData(A_diag), hypre_CSRMatrixI(A_diag),
      //      hypre_CSRMatrixJ(A_diag), workvector, workvector2, r_data, omega, u_data);
   }

   hypre_NvtxPopRange();

   hypre_SeqVectorDestroy(w1);

   return 0;
}

#endif /* #if defined(HYPRE_USING_CUDA) */
