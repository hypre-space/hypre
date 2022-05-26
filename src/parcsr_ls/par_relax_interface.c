/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGRelaxIF( hypre_ParCSRMatrix *A,
                        hypre_ParVector    *f,
                        HYPRE_Int          *cf_marker,
                        HYPRE_Int           relax_type,
                        HYPRE_Int           relax_order,
                        HYPRE_Int           cycle_param,
                        HYPRE_Real          relax_weight,
                        HYPRE_Real          omega,
                        HYPRE_Real         *l1_norms,
                        hypre_ParVector    *u,
                        hypre_ParVector    *Vtemp,
                        hypre_ParVector    *Ztemp )
{
   HYPRE_Int i, Solve_err_flag = 0;
   HYPRE_Int relax_points[2];

   if (relax_order == 1 && cycle_param < 3)
   {
      if (cycle_param < 2)
      {
         /* CF down cycle */
         relax_points[0] =  1;
         relax_points[1] = -1;
      }
      else
      {
         /* FC up cycle */
         relax_points[0] = -1;
         relax_points[1] =  1;
      }

      for (i = 0; i < 2; i++)
      {
         Solve_err_flag = hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, relax_points[i],
                                               relax_weight, omega, l1_norms, u, Vtemp, Ztemp);
      }
   }
   else
   {
      Solve_err_flag = hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, 0, relax_weight, omega,
                                            l1_norms, u, Vtemp, Ztemp);
   }

   return Solve_err_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRRelax_L1_Jacobi (same as the one in AMS, but this allows CF)
 * u_new = u_old + w D^{-1}(f - A u), where D_ii = ||A(i,:)||_1
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_ParCSRRelax_L1_Jacobi( hypre_ParCSRMatrix *A,
                             hypre_ParVector    *f,
                             HYPRE_Int          *cf_marker,
                             HYPRE_Int           relax_points,
                             HYPRE_Real          relax_weight,
                             HYPRE_Real         *l1_norms,
                             hypre_ParVector    *u,
                             hypre_ParVector    *Vtemp )

{
   return hypre_BoomerAMGRelax(A, f, cf_marker, 18, relax_points, relax_weight, 0.0, l1_norms, u,
                               Vtemp, NULL);
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax_FCFJacobi
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BoomerAMGRelax_FCFJacobi( hypre_ParCSRMatrix *A,
                                hypre_ParVector    *f,
                                HYPRE_Int          *cf_marker,
                                HYPRE_Real          relax_weight,
                                hypre_ParVector    *u,
                                hypre_ParVector    *Vtemp)
{
   HYPRE_Int i;
   HYPRE_Int relax_points[3];
   HYPRE_Int relax_type = 0;

   relax_points[0] = -1; /*F */
   relax_points[1] =  1; /*C */
   relax_points[2] = -1; /*F */

   /* cf == NULL --> size == 0 */
   if (cf_marker == NULL)
   {
      hypre_assert(hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)) == 0);
   }

   for (i = 0; i < 3; i++)
   {
      hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, relax_points[i],
                           relax_weight, 0.0, NULL, u, Vtemp, NULL);
   }

   return hypre_error_flag;
}



