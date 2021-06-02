/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSolve( void                 *ssamg_vdata,
                  hypre_SStructMatrix  *A,
                  hypre_SStructVector  *b,
                  hypre_SStructVector  *x )
{
   hypre_SSAMGData       *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   /* Solver parameters */
   HYPRE_Real             tol             =  hypre_SSAMGDataTol(ssamg_data);
   HYPRE_Int              max_iter        =  hypre_SSAMGDataMaxIter(ssamg_data);
   HYPRE_Int              logging         =  hypre_SSAMGDataLogging(ssamg_data);
   HYPRE_Int              rel_change      =  hypre_SSAMGDataRelChange(ssamg_data);
   HYPRE_Int              zero_guess      =  hypre_SSAMGDataZeroGuess(ssamg_data);
   HYPRE_Int              num_pre_relax   =  hypre_SSAMGDataNumPreRelax(ssamg_data);
   HYPRE_Int              num_post_relax  =  hypre_SSAMGDataNumPosRelax(ssamg_data);
   HYPRE_Int              num_levels      =  hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Real            *norms           =  hypre_SSAMGDataNorms(ssamg_data);
   HYPRE_Real            *rel_norms       =  hypre_SSAMGDataRelNorms(ssamg_data);
   HYPRE_Int            **active_l        =  hypre_SSAMGDataActivel(ssamg_data);

   /* Work data structures */
   hypre_SStructMatrix  **A_l             = (ssamg_data -> A_l);
   hypre_SStructMatrix  **P_l             = (ssamg_data -> P_l);
   hypre_SStructMatrix  **RT_l            = (ssamg_data -> RT_l);
   hypre_SStructVector  **b_l             = (ssamg_data -> b_l);
   hypre_SStructVector  **x_l             = (ssamg_data -> x_l);
   hypre_SStructVector  **r_l             = (ssamg_data -> r_l);
   hypre_SStructVector  **e_l             = (ssamg_data -> e_l);
   void                 **relax_data_l    = (ssamg_data -> relax_data_l);
   void                 **matvec_data_l   = (ssamg_data -> matvec_data_l);
   void                 **restrict_data_l = (ssamg_data -> restrict_data_l);
   void                 **interp_data_l   = (ssamg_data -> interp_data_l);

   /* Local Variables */
   HYPRE_Real             b_dot_b = 1.0, r_dot_r, eps = 0;
   HYPRE_Real             e_dot_e = 0.0, x_dot_x = 1.0;
   HYPRE_Int              i, l;
#ifdef DEBUG_SOLVE
   char                   filename[255];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   hypre_BeginTiming(ssamg_data -> time_index);

   /*-----------------------------------------------------
    * Refs to A,x,b (the SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/
   HYPRE_SStructMatrixDestroy(A_l[0]);
   HYPRE_SStructVectorDestroy(b_l[0]);
   HYPRE_SStructVectorDestroy(x_l[0]);
   hypre_SStructMatrixRef(A, &A_l[0]);
   hypre_SStructVectorRef(b, &b_l[0]);
   hypre_SStructVectorRef(x, &x_l[0]);

   (ssamg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructVectorSetConstantValues(x_l[0], 0.0);
      }
      hypre_EndTiming(ssamg_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.)
   {
      /* eps = (tol^2) */
      hypre_SStructInnerProd(b_l[0], b_l[0], &b_dot_b);
      eps = tol*tol;

      if (logging > 0)
      {
         norms[0]     = 0.0;
         rel_norms[0] = 0.0;
      }

      if (!(b_dot_b > 0.0))
      {
#if 0
         /* if rhs is zero, return a zero solution */
         hypre_SStructVectorSetConstantValues(x_l[0], 0.0);
         hypre_EndTiming(ssamg_data -> time_index);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
#else
         b_dot_b = 1.0;
#endif
      }
   }

#ifdef DEBUG_SOLVE
   HYPRE_Real x_dot_x;

   hypre_SStructInnerProd(x, x, &x_dot_x);
   hypre_printf("<x0, x0> = %20.15e\n", x_dot_x);
   hypre_printf("<b, b> = %20.15e\n", b_dot_b);

   /* Print initial solution and residual */
   hypre_sprintf(filename, "ssamg_x.i%02d", 0);
   HYPRE_SStructVectorPrint(filename, x_l[0], 0);
   hypre_sprintf(filename, "ssamg_r.i%02d", 0);
   HYPRE_SStructVectorPrint(filename, r_l[0], 0);
#endif

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   HYPRE_Int start_relax = num_levels - 6;
   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle
       *--------------------------------------------------*/
      HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);

      /* fine grid pre-relaxation */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
      hypre_SSAMGRelaxSetPreRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;
      HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");

      /* compute fine grid residual (r = b - Ax) */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Residual");
      hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                 A_l[0], x_l[0], 1.0, b_l[0], r_l[0]);
      HYPRE_ANNOTATE_REGION_END("%s", "Residual");

      /* convergence check */
      if (tol > 0.0)
      {
         hypre_SStructInnerProd(r_l[0], r_l[0], &r_dot_r);

         if (logging > 0)
         {
            norms[i]     = sqrt(r_dot_r);
            rel_norms[i] = sqrt(r_dot_r/b_dot_b);
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r/b_dot_b < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e/x_dot_x) < eps)
               {
                  HYPRE_ANNOTATE_MGLEVEL_END(0);
                  break;
               }
            }
            else
            {
               HYPRE_ANNOTATE_MGLEVEL_END(0);
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Restriction");
         hypre_SStructMatvecCompute(restrict_data_l[0], 1.0,
                                    RT_l[0], r_l[0], 0.0, b_l[1], b_l[1]);
         HYPRE_ANNOTATE_REGION_END("%s", "Restriction");
#if DEBUG_SOLVE
         hypre_sprintf(filename, "ssamg_xdown.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
         hypre_sprintf(filename, "ssamg_rdown.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, r_l[0], 0);
         hypre_sprintf(filename, "ssamg_b.i%02d.l%02d", i, 1);
         HYPRE_SStructVectorPrint(filename, b_l[1], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_END(0);

         for (l = 1; l <= (num_levels - 2); l++)
         {
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            /* Set active parts */
            hypre_SStructMatvecSetActiveParts(matvec_data_l[l], active_l[l]);

            /* pre-relaxation */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
            hypre_SSAMGRelaxSetPreRelax(relax_data_l[l]);
            hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
            hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
            if (l == start_relax)
            {
               hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
            else
            {
               hypre_SStructVectorSetConstantValues(x_l[l], 0.0);
            }
            HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");

            /* compute residual (r = b - Ax) */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Residual");
            hypre_SStructMatvecCompute(matvec_data_l[l], -1.0,
                                       A_l[l], x_l[l], 1.0, b_l[l], r_l[l]);
            HYPRE_ANNOTATE_REGION_END("%s", "Residual");

            /* Set all parts to active */
            hypre_SStructMatvecSetAllPartsActive(matvec_data_l[l]);

            /* restrict residual */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Restriction");
            hypre_SStructMatvecCompute(restrict_data_l[l], 1.0,
                                       RT_l[l], r_l[l], 0.0, b_l[l+1], b_l[l+1]);
            HYPRE_ANNOTATE_REGION_END("%s", "Restriction");
#if DEBUG_SOLVE
            hypre_sprintf(filename, "ssamg_xdown.i%02d.l%02d", i, l);
            HYPRE_SStructVectorPrint(filename, x_l[l], 0);
            hypre_sprintf(filename, "ssamg_rdown.i%02d.l%02d", i, l);
            HYPRE_SStructVectorPrint(filename, r_l[l], 0);
            hypre_sprintf(filename, "ssamg_b.i%02d.l%02d", i, l+1);
            HYPRE_SStructVectorPrint(filename, b_l[l+1], 0);
#endif
            HYPRE_ANNOTATE_MGLEVEL_END(l);
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);

         /* Run coarse solver */
         hypre_SSAMGCoarseSolve(ssamg_vdata);

#if DEBUG_SOLVE
         hypre_sprintf(filename, "ssamg_xbottom.i%02d.l%02d", i, l);
         HYPRE_SStructVectorPrint(filename, x_l[l], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            /* TODO: Can we simplify the next two calls? */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
            hypre_SStructMatvecCompute(interp_data_l[l], 1.0,
                                       P_l[l], x_l[l+1], 0.0, e_l[l], e_l[l]);
            hypre_SStructAxpy(1.0, e_l[l], x_l[l]);
            HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
            HYPRE_ANNOTATE_MGLEVEL_END(l + 1);
#if DEBUG_SOLVE
            hypre_sprintf(filename, "ssamg_eup.i%02d.l%02d", i, l);
            HYPRE_SStructVectorPrint(filename, e_l[l], 0);
            hypre_sprintf(filename, "ssamg_xup.i%02d.l%02d", i, l);
            HYPRE_SStructVectorPrint(filename, x_l[l], 0);
#endif
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            /* Set active parts */
            hypre_SStructMatvecSetActiveParts(matvec_data_l[l], active_l[l]);

            /* post-relaxation */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
            hypre_SSAMGRelaxSetPostRelax(relax_data_l[l]);
            hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
            hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 0);
            if (l == start_relax)
            {
               hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
            HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");

            /* Set all parts to active */
            hypre_SStructMatvecSetAllPartsActive(matvec_data_l[l]);
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         /* TODO: Can we simplify the next two calls? */
         HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
         hypre_SStructMatvecCompute(interp_data_l[0], 1.0,
                                    P_l[0], x_l[1], 0.0, e_l[0], e_l[0]);
         hypre_SStructAxpy(1.0, e_l[0], x_l[0]);
         HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
         HYPRE_ANNOTATE_MGLEVEL_END(1);
#if DEBUG_SOLVE
         hypre_sprintf(filename, "ssamg_eup.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, e_l[0], 0);
         hypre_sprintf(filename, "ssamg_xup.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
#endif
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            hypre_SStructInnerProd(e_l[0], e_l[0], &e_dot_e);
            hypre_SStructInnerProd(x_l[0], x_l[0], &x_dot_x);
         }
         else
         {
            e_dot_e = 0.0;
            x_dot_x = 1.0;
         }
      }

      /* fine grid post-relaxation */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
      hypre_SSAMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");

#if DEBUG_SOLVE
      hypre_sprintf(filename, "ssamg_xpostf.i%02d.l%02d", i, 0);
      HYPRE_SStructVectorPrint(filename, x_l[0], 0);
#endif

      (ssamg_data -> num_iterations) = (i + 1);
      HYPRE_ANNOTATE_MGLEVEL_END(0);
   }

#if DEBUG_SOLVE
   HYPRE_Real b_dot_x;

   hypre_SStructInnerProd(b, x, &b_dot_x);
   hypre_SStructInnerProd(b, b, &b_dot_b);

   if (b_dot_x < 0)
   {
      hypre_printf("b_dot_x: %e\n", b_dot_x);
      hypre_printf("b_dot_b: %e\n", b_dot_b);
      hypre_printf("b_dot_x/b_dot_b: %e\n", b_dot_x/b_dot_b);

      hypre_sprintf(filename, "ssamg_b.negdot");
      HYPRE_SStructVectorPrint(filename, b, 0);

      hypre_sprintf(filename, "ssamg_x.negdot");
      HYPRE_SStructVectorPrint(filename, x, 0);
   }
#endif

   /*-----------------------------------------------------
    * Destroy Refs to A_in, x_in, b_in
    *-----------------------------------------------------*/
   /* hypre_SStructMatrixDestroy(A_l[0]); */
   /* hypre_SStructVectorDestroy(b_l[0]); */
   /* hypre_SStructVectorDestroy(x_l[0]); */

   hypre_EndTiming(ssamg_data -> time_index);
   hypre_SSAMGPrintLogging((void *) ssamg_data);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
