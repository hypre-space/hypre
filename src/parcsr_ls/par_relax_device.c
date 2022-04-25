/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

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
   hypre_VectorData(hypre_ParVectorLocalVector(w1)) = hypre_VectorData(hypre_ParVectorLocalVector(
                                                                          Vtemp));
   hypre_VectorData(hypre_ParVectorLocalVector(w2)) = hypre_VectorData(hypre_ParVectorLocalVector(
                                                                          Ztemp));

   if (Symm)
   {
      /* V = f - A*u */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, w1);

      /* Z = L^{-1}*V */
      hypre_CSRMatrixTriLowerUpperSolveDevice('L', hypre_ParCSRMatrixDiag(A), l1_norms,
                                              hypre_ParVectorLocalVector(w1), hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      hypre_ParVectorAxpy(relax_weight, w2, u);

      /* Note: only update V from local change of u, i.e., V = V - w*A_diag*Z_local */
      hypre_CSRMatrixMatvec(-relax_weight, hypre_ParCSRMatrixDiag(A), hypre_ParVectorLocalVector(w2),
                            1.0, hypre_ParVectorLocalVector(w1));

      /* Z = U^{-1}*V */
      hypre_CSRMatrixTriLowerUpperSolveDevice('U', hypre_ParCSRMatrixDiag(A), l1_norms,
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
      hypre_CSRMatrixTriLowerUpperSolveDevice(uplo, hypre_ParCSRMatrixDiag(A), l1_norms,
                                              hypre_ParVectorLocalVector(w1), hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      hypre_ParVectorAxpy(relax_weight, w2, u);
   }

   hypre_ParVectorDestroy(w1);
   hypre_ParVectorDestroy(w2);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice ( hypre_ParCSRMatrix *A,
                                                hypre_ParVector    *f,
                                                HYPRE_Real          relax_weight,
                                                HYPRE_Real          omega,
                                                hypre_ParVector    *u,
                                                hypre_ParVector    *r,
                                                hypre_ParVector    *z,
                                                HYPRE_Int           num_inner_iters)
{
   hypre_GpuProfilingPushRange("BoomerAMGRelaxTwoStageGaussSeidelDevice");

   hypre_CSRMatrix *A_diag       = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int        num_rows     = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex   *A_diag_data  = hypre_CSRMatrixData(A_diag);
   hypre_Vector    *u_local      = hypre_ParVectorLocalVector(u);
   hypre_Vector    *r_local      = hypre_ParVectorLocalVector(r);
   hypre_Vector    *z_local      = hypre_ParVectorLocalVector(z);
   HYPRE_Complex   *u_data       = hypre_VectorData(u_local);
   HYPRE_Complex   *r_data       = hypre_VectorData(r_local);
   HYPRE_Complex   *z_data       = hypre_VectorData(z_local);
   HYPRE_Int        zsize        = hypre_VectorSize(z_local);
   HYPRE_Int        rsize        = hypre_VectorSize(r_local);
   HYPRE_Complex    multiplier   = 1.0;
   HYPRE_Int        i;

   hypre_ParCSRMatrixMatvecOutOfPlace(-relax_weight, A, u, relax_weight, f, r);

   hypreDevice_DiagScaleVector(num_rows, A_diag_i, A_diag_data, r_data, 0.0, z_data);

   // set this so that axpy works out properly. Reset later.
   hypre_VectorSize(z_local) = rsize;

   // 1) u = u + z
   hypre_SeqVectorAxpy(multiplier, z_local, u_local);
   multiplier *= -1.0;

   for (i = 0; i < num_inner_iters; ++i)
   {
      // 2) r = Lz
      hypre_CSRMatrixSpMVDevice(0, 1.0, A_diag, z_local, 0.0, r_local, -2);
      // 3) z = r/D, u = u + m*z
      hypreDevice_DiagScaleVector2(num_rows, A_diag_i, A_diag_data, r_data, multiplier, z_data, u_data);
      multiplier *= -1.0;
   }

   // reset this
   hypre_VectorSize(z_local) = zsize;

   hypre_GpuProfilingPopRange();

   return 0;
}

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */

