/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSolve( void                 *sys_pfmg_vdata,
                    hypre_SStructMatrix  *A_in,
                    hypre_SStructVector  *b_in,
                    hypre_SStructVector  *x_in         )
{
   hypre_SysPFMGData       *sys_pfmg_data = (hypre_SysPFMGData*)sys_pfmg_vdata;

   hypre_SStructPMatrix    *A;
   hypre_SStructPVector    *b;
   hypre_SStructPVector    *x;

   HYPRE_Real              tol             = (sys_pfmg_data -> tol);
   HYPRE_Int               max_iter        = (sys_pfmg_data -> max_iter);
   HYPRE_Int               rel_change      = (sys_pfmg_data -> rel_change);
   HYPRE_Int               zero_guess      = (sys_pfmg_data -> zero_guess);
   HYPRE_Int               num_pre_relax   = (sys_pfmg_data -> num_pre_relax);
   HYPRE_Int               num_post_relax  = (sys_pfmg_data -> num_post_relax);
   HYPRE_Int               num_levels      = (sys_pfmg_data -> num_levels);
   hypre_SStructPMatrix  **A_l             = (sys_pfmg_data -> A_l);
   hypre_SStructPMatrix  **P_l             = (sys_pfmg_data -> P_l);
   hypre_SStructPMatrix  **RT_l            = (sys_pfmg_data -> RT_l);
   hypre_SStructPVector  **b_l             = (sys_pfmg_data -> b_l);
   hypre_SStructPVector  **x_l             = (sys_pfmg_data -> x_l);
   hypre_SStructPVector  **r_l             = (sys_pfmg_data -> r_l);
   hypre_SStructPVector  **e_l             = (sys_pfmg_data -> e_l);
   void                  **relax_data_l    = (sys_pfmg_data -> relax_data_l);
   void                  **matvec_data_l   = (sys_pfmg_data -> matvec_data_l);
   void                  **restrict_data_l = (sys_pfmg_data -> restrict_data_l);
   void                  **interp_data_l   = (sys_pfmg_data -> interp_data_l);
   HYPRE_Real             *norms           = (sys_pfmg_data -> norms);
   HYPRE_Real             *rel_norms       = (sys_pfmg_data -> rel_norms);
   HYPRE_Int              *active_l        = (sys_pfmg_data -> active_l);
   HYPRE_Int               logging         = (sys_pfmg_data -> logging);
   HYPRE_Int               print_level     = (sys_pfmg_data -> print_level);
   HYPRE_Int               print_freq      = (sys_pfmg_data -> print_freq);

   HYPRE_Real            b_dot_b, r_dot_r, eps = 0;
   HYPRE_Real            e_dot_e = 0, x_dot_x = 1;

   HYPRE_Int             i, l;
   char                  filename[255];

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_BeginTiming(sys_pfmg_data -> time_index);

   /*-----------------------------------------------------
    * Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/

   /* RDF: Why do we need the '_in' objects? */
   hypre_SStructPMatrixRef(hypre_SStructMatrixPMatrix(A_in, 0), &A);
   hypre_SStructPVectorRef(hypre_SStructVectorPVector(b_in, 0), &b);
   hypre_SStructPVectorRef(hypre_SStructVectorPVector(x_in, 0), &x);

   hypre_SStructPMatrixDestroy(A_l[0]);
   hypre_SStructPVectorDestroy(b_l[0]);
   hypre_SStructPVectorDestroy(x_l[0]);
   hypre_SStructPMatrixRef(A, &A_l[0]);
   hypre_SStructPVectorRef(b, &b_l[0]);
   hypre_SStructPVectorRef(x, &x_l[0]);

   (sys_pfmg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   /* RDF: Why should we support max_iter = 0? I think we should throw an error. */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructPVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(sys_pfmg_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

#if 0
   /* RDF: Not needed if max_iter > 0 */
   if (((tol > 0.) && (logging > 0)) || (print_level > 1))
   {
      /* Compute fine grid residual (r = b - Ax) */
      hypre_SStructPMatvecCompute(matvec_data_l[0], -1.0,
                                  A_l[0], x_l[0], 1.0, b_l[0], r_l[0]);
   }
#endif

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      hypre_SStructPInnerProd(b_l[0], b_l[0], &b_dot_b);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (!(b_dot_b > 0.0))
      {
         hypre_SStructPVectorSetConstantValues(x_l[0], 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(ssamg_data -> time_index);
         HYPRE_ANNOTATE_FUNC_END;
         return hypre_error_flag;
      }

   }

   /* Print initial solution and residual */
#if DEBUG
   hypre_sprintf(filename, "sys_pfmg_x.i%02d", 0);
   hypre_SStructPVectorPrint(filename, x_l[0], 0);

   hypre_sprintf(filename, "sys_pfmg_r.i%02d", 0);
   hypre_SStructPVectorPrint(filename, r_l[0], 0);
#endif

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

      /* fine grid pre-relaxation */
      hypre_SysPFMGRelaxSetPreRelax(relax_data_l[0]);
      hypre_SysPFMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_SysPFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (r = b - Ax) */
      /* RDF: I don't think we should have a different interface for matvec as elsewhere */
      hypre_SStructPMatvecCompute(matvec_data_l[0], -1.0, A_l[0], x_l[0], 1.0, b_l[0], r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         hypre_SStructPInnerProd(r_l[0], r_l[0], &r_dot_r);

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
//         hypre_SysSemiRestrict(restrict_data_l[0], RT_l[0], r_l[0], b_l[1]);
         hypre_SStructPMatvecCompute(restrict_data_l[0], 1.0, RT_l[0], r_l[l], 0.0,
                                     b_l[1], b_l[1]);
#if DEBUG
         hypre_sprintf(filename, "zout_xdown.%02d", 0);
         hypre_SStructPVectorPrint(filename, x_l[0], 0);
         hypre_sprintf(filename, "zout_rdown.%02d", 0);
         hypre_SStructPVectorPrint(filename, r_l[0], 0);
         hypre_sprintf(filename, "zout_b.%02d", 1);
         hypre_SStructPVectorPrint(filename, b_l[1], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_END(0);

         for (l = 1; l <= (num_levels - 2); l++)
         {
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            if (active_l[l])
            {
               /* pre-relaxation */
               hypre_SysPFMGRelaxSetPreRelax(relax_data_l[l]);
               hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
               hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 1);
               hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

               /* compute residual (b - Ax) */
               hypre_SStructPMatvecCompute(matvec_data_l[l], -1.0, A_l[l],
                                           x_l[l], 1.0, b_l[l], r_l[l]);
            }
            else
            {
               /* inactive level, set x=0, so r=(b-Ax)=b */
               hypre_SStructPVectorSetConstantValues(x_l[l], 0.0);
               hypre_SStructPCopy(b_l[l], r_l[l]);
            }

            /* restrict residual */
//            hypre_SysSemiRestrict(restrict_data_l[l], RT_l[l], r_l[l], b_l[l + 1]);
            hypre_SStructPMatvecCompute(restrict_data_l[l], 1.0, RT_l[l], r_l[l], 0.0,
                                        b_l[l + 1], b_l[l + 1]);
#if DEBUG
            hypre_sprintf(filename, "zout_xdown.%02d", l);
            hypre_SStructPVectorPrint(filename, x_l[l], 0);
            hypre_sprintf(filename, "zout_rdown.%02d", l);
            hypre_SStructPVectorPrint(filename, r_l[l], 0);
            hypre_sprintf(filename, "zout_b.%02d", l + 1);
            hypre_SStructPVectorPrint(filename, b_l[l + 1], 0);
#endif
            HYPRE_ANNOTATE_MGLEVEL_END(l);
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/

         HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);

         hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 1);
         hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
#if DEBUG
         hypre_sprintf(filename, "zout_xbottom.%02d", l);
         hypre_SStructPVectorPrint(filename, x_l[l], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
//            hypre_SysSemiInterp(interp_data_l[l], P_l[l], x_l[l + 1], e_l[l]);
            hypre_SStructPMatvecCompute(interp_data_l[l], 1.0, P_l[l], x_l[l + 1], 0.0,
                                        e_l[l], e_l[l]);
            hypre_SStructPAxpy(1.0, e_l[l], x_l[l]);
            HYPRE_ANNOTATE_MGLEVEL_END(l + 1);
#if DEBUG
            hypre_sprintf(filename, "zout_eup.%02d", l);
            hypre_SStructPVectorPrint(filename, e_l[l], 0);
            hypre_sprintf(filename, "zout_xup.%02d", l);
            hypre_SStructPVectorPrint(filename, x_l[l], 0);
#endif
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);
            if (active_l[l])
            {
               /* post-relaxation */
               hypre_SysPFMGRelaxSetPostRelax(relax_data_l[l]);
               hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
               hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 0);
               hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
//         hypre_SysSemiInterp(interp_data_l[0], P_l[0], x_l[1], e_l[0]);
         hypre_SStructPMatvecCompute(interp_data_l[0], 1.0, P_l[0], x_l[1], 0.0,
                                     e_l[0], e_l[0]);
         hypre_SStructPAxpy(1.0, e_l[0], x_l[0]);
         HYPRE_ANNOTATE_MGLEVEL_END(1);
#if DEBUG
         hypre_sprintf(filename, "zout_eup.%02d", 0);
         hypre_SStructPVectorPrint(filename, e_l[0], 0);
         hypre_sprintf(filename, "zout_xup.%02d", 0);
         hypre_SStructPVectorPrint(filename, x_l[0], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            hypre_SStructPInnerProd(e_l[0], e_l[0], &e_dot_e);
            hypre_SStructPInnerProd(x_l[0], x_l[0], &x_dot_x);
         }
         else
         {
            e_dot_e = 0.0;
            x_dot_x = 1.0;
         }
      }

      /* fine grid post-relaxation */
      hypre_SysPFMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_SysPFMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_SysPFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

      (sys_pfmg_data -> num_iterations) = (i + 1);
      HYPRE_ANNOTATE_MGLEVEL_END(0);

      /* RDF: In PFMG, we don't do any of this */
      if ((logging > 0) || (print_level > 1))
      {
         /* Recompute fine grid residual (r = b - Ax) after post-smoothing */
         hypre_SStructPMatvecCompute(matvec_data_l[0], -1.0, A_l[0], x_l[0], 1.0, b_l[0], r_l[0]);

         if (logging > 0)
         {
            hypre_SStructPInnerProd(r_l[0], r_l[0], &r_dot_r);

            norms[i + 1] = sqrt(r_dot_r);
            rel_norms[i + 1] = sqrt(r_dot_r / b_dot_b);
         }

         if (print_level > 1 && !((i + 1) % print_freq))
         {
            /* Print solution */
            hypre_sprintf(filename, "sys_pfmg_x.i%02d", (i + 1));
            hypre_SStructPVectorPrint(filename, x_l[0], 0);

            /* Print residual */
            hypre_sprintf(filename, "sys_pfmg_r.i%02d", (i + 1));
            hypre_SStructPVectorPrint(filename, r_l[0], 0);
         }
      }
      else
      {
         /* r_l[0] is the fine grid residual computed after pre-smoothing */
         /* RDF: This is redundant unless the next convergence check passes,
          * which will never happen because it's already been checked above */
         hypre_SStructPInnerProd(r_l[0], r_l[0], &r_dot_r);
      }

      /* convergence check */
      if (tol > 0.0)
      {
         if (r_dot_r / b_dot_b < eps)
         {
            if (((rel_change) && (e_dot_e / x_dot_x) < eps) || (!rel_change))
            {
               break;
            }
         }
      }
   }

   /*-----------------------------------------------------
    * Destroy Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors).
    *-----------------------------------------------------*/

   hypre_SStructPMatrixDestroy(A);
   hypre_SStructPVectorDestroy(x);
   hypre_SStructPVectorDestroy(b);

   hypre_EndTiming(sys_pfmg_data -> time_index);
   hypre_SysPFMGPrintLogging((void *) sys_pfmg_data);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
