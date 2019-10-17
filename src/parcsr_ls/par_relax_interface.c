/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

HYPRE_Int  hypre_BoomerAMGRelaxIF( hypre_ParCSRMatrix *A,
                                   hypre_ParVector    *f,
                                   HYPRE_Int          *cf_marker,
                                   HYPRE_Int           relax_type,
                                   HYPRE_Int           relax_order,
                                   HYPRE_Int           cycle_type,
                                   HYPRE_Real          relax_weight,
                                   HYPRE_Real          omega,
                                   HYPRE_Real         *l1_norms,
                                   hypre_ParVector    *u,
                                   hypre_ParVector    *Vtemp,
                                   hypre_ParVector    *Ztemp )
{
   HYPRE_Int i, Solve_err_flag = 0;
   HYPRE_Int relax_points[2];
   if (relax_order == 1 && cycle_type < 3)
   {
      if (cycle_type < 2)
      {
         relax_points[0] = 1;
         relax_points[1] = -1;
      }
      else
      {
         relax_points[0] = -1;
         relax_points[1] = 1;
      }

      for (i=0; i < 2; i++)
         Solve_err_flag = hypre_BoomerAMGRelax(A,
                                               f,
                                               cf_marker,
                                               relax_type,
                                               relax_points[i],
                                               relax_weight,
                                               omega,
                                               l1_norms,
                                               u,
                                               Vtemp,
                                               Ztemp);

   }
   else
   {
      Solve_err_flag = hypre_BoomerAMGRelax(A,
                                            f,
                                            cf_marker,
                                            relax_type,
                                            0,
                                            relax_weight,
                                            omega,
                                            l1_norms,
                                            u,
                                            Vtemp,
                                            Ztemp);
   }

   return Solve_err_flag;
}


