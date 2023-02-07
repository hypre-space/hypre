/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "pfmg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_PFMGSolve
 *
 * NOTE regarding hypre_StructVectorClearAllValues:
 *
 * Since r_l and e_l point to the same temporary data, the boundary ghost values
 * are not guaranteed to stay clear as needed in the constant coefficient case.
 * In addition, for the Galerkin case, the interpolation operator is set to be a
 * variable coefficient operator.  However, interpolation values that reach
 * outside of the boundary are currently not always computed to be zero in this
 * case, so we can't rewrite SemiRestrict and SemiInterp to faithfully zero out
 * boundary ghost values only when needed because there isn't enough context.
 * So, below we clear the values of r_l and e_l before computing the residual
 * and calling interpolation.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSolve( void               *pfmg_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x         )
{
   hypre_PFMGData       *pfmg_data = (hypre_PFMGData       *)pfmg_vdata;

   HYPRE_Real            tol             = (pfmg_data -> tol);
   HYPRE_Int             max_iter        = (pfmg_data -> max_iter);
   HYPRE_Int             rel_change      = (pfmg_data -> rel_change);
   HYPRE_Int             zero_guess      = (pfmg_data -> zero_guess);
   HYPRE_Int             num_pre_relax   = (pfmg_data -> num_pre_relax);
   HYPRE_Int             num_post_relax  = (pfmg_data -> num_post_relax);
   HYPRE_Int             num_levels      = (pfmg_data -> num_levels);
   hypre_StructMatrix  **A_l             = (pfmg_data -> A_l);
   hypre_StructMatrix  **P_l             = (pfmg_data -> P_l);
   hypre_StructMatrix  **RT_l            = (pfmg_data -> RT_l);
   hypre_StructVector  **b_l             = (pfmg_data -> b_l);
   hypre_StructVector  **x_l             = (pfmg_data -> x_l);
   hypre_StructVector  **r_l             = (pfmg_data -> r_l);
   hypre_StructVector  **e_l             = (pfmg_data -> e_l);
   void                **relax_data_l    = (pfmg_data -> relax_data_l);
   void                **matvec_data_l   = (pfmg_data -> matvec_data_l);
   void                **restrict_data_l = (pfmg_data -> restrict_data_l);
   void                **interp_data_l   = (pfmg_data -> interp_data_l);
   HYPRE_Int             logging         = (pfmg_data -> logging);
   HYPRE_Real           *norms           = (pfmg_data -> norms);
   HYPRE_Real           *rel_norms       = (pfmg_data -> rel_norms);
   HYPRE_Int            *active_l        = (pfmg_data -> active_l);

   HYPRE_Real            b_dot_b = 0, r_dot_r, eps = 0;
   HYPRE_Real            e_dot_e = 0.0, x_dot_x = 1.0;

   HYPRE_Int             i, l;
   HYPRE_Int             constant_coefficient;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_BeginTiming(pfmg_data -> time_index);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   hypre_StructMatrixDestroy(A_l[0]);
   hypre_StructVectorDestroy(b_l[0]);
   hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   (pfmg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(pfmg_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = hypre_StructInnerProd(b_l[0], b_l[0]);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(pfmg_data -> time_index);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
   }

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle
       *--------------------------------------------------*/

      HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);

      if (constant_coefficient)
      {
         hypre_StructVectorClearAllValues(r_l[0]);
      }

      /* fine grid pre-relaxation */
      hypre_PFMGRelaxSetPreRelax(relax_data_l[0]);
      hypre_PFMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_PFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      hypre_StructCopy(b_l[0], r_l[0]);
      hypre_StructMatvecCompute(matvec_data_l[0],
                                -1.0, A_l[0], x_l[0], 1.0, r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = hypre_StructInnerProd(r_l[0], r_l[0]);

         if (logging > 0)
         {
            norms[i] = hypre_sqrt(r_dot_r);
            if (b_dot_b > 0)
            {
               rel_norms[i] = hypre_sqrt(r_dot_r / b_dot_b);
            }
            else
            {
               rel_norms[i] = 0.0;
            }
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r / b_dot_b < eps) && (i > 0))
         {
            if ( ((rel_change) && (e_dot_e / x_dot_x) < eps) || (!rel_change) )
            {
               HYPRE_ANNOTATE_MGLEVEL_END(0);
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         hypre_SemiRestrict(restrict_data_l[0], RT_l[0], r_l[0], b_l[1]);
#if DEBUG
         hypre_sprintf(filename, "zout_xdown.%02d", 0);
         hypre_StructVectorPrint(filename, x_l[0], 0);
         hypre_sprintf(filename, "zout_rdown.%02d", 0);
         hypre_StructVectorPrint(filename, r_l[0], 0);
         hypre_sprintf(filename, "zout_b.%02d", 1);
         hypre_StructVectorPrint(filename, b_l[1], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_END(0);

         for (l = 1; l <= (num_levels - 2); l++)
         {
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            if (hypre_StructGridDataLocation(hypre_StructVectorGrid(r_l[l])) == HYPRE_MEMORY_HOST)
            {
               hypre_SetDeviceOff();
            }
#endif
            if (constant_coefficient)
            {
               hypre_StructVectorClearAllValues(r_l[l]);
            }

            if (active_l[l])
            {
               /* pre-relaxation */
               hypre_PFMGRelaxSetPreRelax(relax_data_l[l]);
               hypre_PFMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
               hypre_PFMGRelaxSetZeroGuess(relax_data_l[l], 1);
               hypre_PFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

               /* compute residual (b - Ax) */
               hypre_StructCopy(b_l[l], r_l[l]);
               hypre_StructMatvecCompute(matvec_data_l[l],
                                         -1.0, A_l[l], x_l[l], 1.0, r_l[l]);
            }
            else
            {
               /* inactive level, set x=0, so r=(b-Ax)=b */
               hypre_StructVectorSetConstantValues(x_l[l], 0.0);
               hypre_StructCopy(b_l[l], r_l[l]);
            }

            /* restrict residual */
            hypre_SemiRestrict(restrict_data_l[l], RT_l[l], r_l[l], b_l[l + 1]);
#if DEBUG
            hypre_printf("Level %d: b_l = %.30e\n", l + 1, hypre_StructInnerProd(b_l[l + 1], b_l[l + 1]));
            hypre_sprintf(filename, "zout_xdown.%02d", l);
            hypre_StructVectorPrint(filename, x_l[l], 0);
            hypre_sprintf(filename, "zout_rdown.%02d", l);
            hypre_StructVectorPrint(filename, r_l[l], 0);
            hypre_sprintf(filename, "zout_b.%02d", l + 1);
            hypre_StructVectorPrint(filename, b_l[l + 1], 0);
#endif

            HYPRE_ANNOTATE_MGLEVEL_END(l);
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);

         if (active_l[l])
         {
            hypre_PFMGRelaxSetZeroGuess(relax_data_l[l], 1);
            hypre_PFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }
         else
         {
            hypre_StructVectorSetConstantValues(x_l[l], 0.0);
         }
#if DEBUG
         hypre_sprintf(filename, "zout_xbottom.%02d", l);
         hypre_StructVectorPrint(filename, x_l[l], 0);
         hypre_printf("Level %d: x_l = %.30e\n", l, hypre_StructInnerProd(x_l[l], x_l[l]));
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            if (hypre_StructGridDataLocation(hypre_StructVectorGrid(e_l[l])) == HYPRE_MEMORY_DEVICE)
            {
               hypre_SetDeviceOn();
            }
#endif
            if (constant_coefficient)
            {
               hypre_StructVectorClearAllValues(e_l[l]);
            }
            /* interpolate error and correct (x = x + Pe_c) */
            hypre_SemiInterp(interp_data_l[l], P_l[l], x_l[l + 1], e_l[l]);
            hypre_StructAxpy(1.0, e_l[l], x_l[l]);
            HYPRE_ANNOTATE_MGLEVEL_END(l + 1);
#if DEBUG
            hypre_sprintf(filename, "zout_eup.%02d", l);
            hypre_StructVectorPrint(filename, e_l[l], 0);
            hypre_sprintf(filename, "zout_xup.%02d", l);
            hypre_StructVectorPrint(filename, x_l[l], 0);
            hypre_printf("Level %d: x_l = %.15e\n", l, hypre_StructInnerProd(x_l[l], x_l[l]));
#endif
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            if (active_l[l])
            {
               /* post-relaxation */
               hypre_PFMGRelaxSetPostRelax(relax_data_l[l]);
               hypre_PFMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
               hypre_PFMGRelaxSetZeroGuess(relax_data_l[l], 0);
               hypre_PFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
         }
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         if (hypre_StructGridDataLocation(hypre_StructVectorGrid(e_l[0])) == HYPRE_MEMORY_DEVICE)
         {
            hypre_SetDeviceOn();
         }
#endif
         if (constant_coefficient)
         {
            hypre_StructVectorClearAllValues(e_l[0]);
         }
         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         hypre_SemiInterp(interp_data_l[0], P_l[0], x_l[1], e_l[0]);
         hypre_StructAxpy(1.0, e_l[0], x_l[0]);
         HYPRE_ANNOTATE_MGLEVEL_END(1);
#if DEBUG
         hypre_printf("Level 0: x_l = %.15e\n", hypre_StructInnerProd(x_l[0], x_l[0]));
         hypre_sprintf(filename, "zout_eup.%02d", 0);
         hypre_StructVectorPrint(filename, e_l[0], 0);
         hypre_sprintf(filename, "zout_xup.%02d", 0);
         hypre_StructVectorPrint(filename, x_l[0], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            e_dot_e = hypre_StructInnerProd(e_l[0], e_l[0]);
            x_dot_x = hypre_StructInnerProd(x_l[0], x_l[0]);
         }
         else
         {
            e_dot_e = 0.0;
            x_dot_x = 1.0;
         }
      }

      hypre_PFMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_PFMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_PFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      (pfmg_data -> num_iterations) = (i + 1);

      HYPRE_ANNOTATE_MGLEVEL_END(0);
   }

   hypre_EndTiming(pfmg_data -> time_index);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
