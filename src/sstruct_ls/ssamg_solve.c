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
   HYPRE_Int              print_level     =  hypre_SSAMGDataPrintLevel(ssamg_data);
   HYPRE_Int              print_freq      =  hypre_SSAMGDataPrintFreq(ssamg_data);
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
   HYPRE_Int              part;
   HYPRE_Int              nparts = hypre_SStructMatrixNParts(A);
   hypre_SStructPVector  *px_l;
   char                   filename[255];

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

   if (((tol > 0.) && (logging > 0)) || (print_level > 1))
   {
      /* Compute fine grid residual (b - Ax) */
      hypre_SStructCopy(b_l[0], r_l[0]);
      hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                 A_l[0], x_l[0], 1.0, r_l[0]);
   }

   /* part of convergence check */
   if (tol > 0.)
   {
      /* eps = (tol^2) */
      hypre_SStructInnerProd(b_l[0], b_l[0], &b_dot_b);
      eps = tol*tol;

      /* if rhs is zero, return a zero solution */
      if (!(b_dot_b > 0.0))
      {
#if 0
         hypre_SStructVectorSetConstantValues(x_l[0], 0.0);
         hypre_EndTiming(ssamg_data -> time_index);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
#else
         b_dot_b = 1.0;
#endif
      }

      if (logging > 0)
      {
         hypre_SStructInnerProd(r_l[0], r_l[0], &r_dot_r);

         norms[0] = sqrt(r_dot_r);
         rel_norms[0] = sqrt(r_dot_r/b_dot_b);
      }
   }

   /* Print initial solution and residual */
   if (print_level > 1)
   {
      /* Print solution */
      hypre_sprintf(filename, "ssamg_x.i%02d", 0);
      HYPRE_SStructVectorPrint(filename, x_l[0], 0);

      /* Print residual */
      hypre_sprintf(filename, "ssamg_r.i%02d", 0);
      HYPRE_SStructVectorPrint(filename, r_l[0], 0);
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

#if HYPRE_DEBUG
      /* compute fine grid residual (b - Ax) */
      hypre_SStructCopy(b_l[0], r_l[0]);
      hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                 A_l[0], x_l[0], 1.0, r_l[0]);
      if (print_level > 1 && !(i%print_freq))
      {
         hypre_sprintf(filename, "ssamg_rpre.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, r_l[0], 0);
      }
#endif

      /* fine grid pre-relaxation */
      hypre_SSAMGRelaxSetPreRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

#if HYPRE_DEBUG
      if (print_level > 1 && !(i%print_freq))
      {
         hypre_sprintf(filename, "ssamg_xpref.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
      }
#endif

      /* compute fine grid residual (b - Ax) */
      hypre_SStructCopy(b_l[0], r_l[0]);
      hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                 A_l[0], x_l[0], 1.0, r_l[0]);

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         hypre_SStructMatvecCompute(restrict_data_l[0], 1.0,
                                    RT_l[0], r_l[0], 0.0, b_l[1]);
#if HYPRE_DEBUG
         if (print_level > 1 && !(i%print_freq))
         {
            hypre_sprintf(filename, "ssamg_xdown.i%02d.l%02d", i, 0);
            HYPRE_SStructVectorPrint(filename, x_l[0], 0);
            hypre_sprintf(filename, "ssamg_rdown.i%02d.l%02d", i, 0);
            HYPRE_SStructVectorPrint(filename, r_l[0], 0);
            hypre_sprintf(filename, "ssamg_b.i%02d.l%02d", i, 1);
            HYPRE_SStructVectorPrint(filename, b_l[1], 0);
         }
#endif
         HYPRE_ANNOTATE_MGLEVEL_END(0);

         for (l = 1; l <= (num_levels - 2); l++)
         {
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            /* pre-relaxation */
            hypre_SSAMGRelaxSetPreRelax(relax_data_l[l]);
            hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
            hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
            hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

            /* Set x=0, so r=(b-Ax)=b on inactive parts */
            for (part = 0; part < nparts; part++)
            {
               if (!active_l[l][part])
               {
                  px_l = hypre_SStructVectorPVector(x_l[l], part);
                  hypre_SStructPVectorSetConstantValues(px_l, 0.0);
               }
            }

            /* compute residual (b - Ax) */
            hypre_SStructMatvecSetActiveParts(matvec_data_l[l], active_l[l]);
            hypre_SStructCopy(b_l[l], r_l[l]);
            hypre_SStructMatvecCompute(matvec_data_l[l], -1.0,
                                       A_l[l], x_l[l], 1.0, r_l[l]);
            hypre_SStructMatvecSetAllPartsActive(matvec_data_l[l]);

            /* restrict residual */
            hypre_SStructMatvecCompute(restrict_data_l[l], 1.0,
                                       RT_l[l], r_l[l], 0.0, b_l[l+1]);
#if HYPRE_DEBUG
            if (print_level > 1 && !(i%print_freq))
            {
               hypre_sprintf(filename, "ssamg_xdown.i%02d.l%02d", i, l);
               HYPRE_SStructVectorPrint(filename, x_l[l], 0);
               hypre_sprintf(filename, "ssamg_rdown.i%02d.l%02d", i, l);
               HYPRE_SStructVectorPrint(filename, r_l[l], 0);
               hypre_sprintf(filename, "ssamg_b.i%02d.l%02d", i, l+1);
               HYPRE_SStructVectorPrint(filename, b_l[l+1], 0);
            }
#endif
            HYPRE_ANNOTATE_MGLEVEL_END(l);
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
         HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);

         hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
         hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

#if HYPRE_DEBUG
         if (print_level > 1 && !(i%print_freq))
         {
            hypre_sprintf(filename, "ssamg_xbottom.i%02d.l%02d", i, l);
            HYPRE_SStructVectorPrint(filename, x_l[l], 0);
         }
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            hypre_SStructMatvecCompute(interp_data_l[l], 1.0,
                                       P_l[l], x_l[l+1], 0.0, e_l[l]);
            hypre_SStructAxpy(1.0, e_l[l], x_l[l]);
            HYPRE_ANNOTATE_MGLEVEL_END(l + 1);
#if HYPRE_DEBUG
            if (print_level > 1 && !(i%print_freq))
            {
                hypre_sprintf(filename, "ssamg_eup.i%02d.l%02d", i, l);
                HYPRE_SStructVectorPrint(filename, e_l[l], 0);
                hypre_sprintf(filename, "ssamg_xup.i%02d.l%02d", i, l);
                HYPRE_SStructVectorPrint(filename, x_l[l], 0);
            }
#endif
            HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            /* post-relaxation */
            hypre_SSAMGRelaxSetPostRelax(relax_data_l[l]);
            hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
            hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 0);
            hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         hypre_SStructMatvecCompute(interp_data_l[0], 1.0,
                                    P_l[0], x_l[1], 0.0, e_l[0]);
         hypre_SStructAxpy(1.0, e_l[0], x_l[0]);
         HYPRE_ANNOTATE_MGLEVEL_END(1);
#if HYPRE_DEBUG
         if (print_level > 1 && !(i%print_freq))
         {
            hypre_sprintf(filename, "ssamg_eup.i%02d.l%02d", i, 0);
            HYPRE_SStructVectorPrint(filename, e_l[0], 0);
            hypre_sprintf(filename, "ssamg_xup.i%02d.l%02d", i, 0);
            HYPRE_SStructVectorPrint(filename, x_l[0], 0);
         }
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
      }

      hypre_SSAMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

#if HYPRE_DEBUG
      if (print_level > 1 && !(i%print_freq))
      {
         hypre_sprintf(filename, "ssamg_xpostf.i%02d.l%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
      }
#endif

      if ((logging > 0) || (print_level > 1))
      {
         /* Recompute fine grid residual r_l[0] to account post-smoothing */
         hypre_SStructCopy(b_l[0], r_l[0]);
         hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                    A_l[0], x_l[0], 1.0, r_l[0]);

         if (logging > 0)
         {
            hypre_SStructInnerProd(r_l[0], r_l[0], &r_dot_r);

            norms[i+1] = sqrt(r_dot_r);
            rel_norms[i+1] = sqrt(r_dot_r/b_dot_b);
         }

         if (print_level > 1 && !((i + 1)%print_freq))
         {
            /* Print solution */
            hypre_sprintf(filename, "ssamg_x.i%02d", (i + 1));
            HYPRE_SStructVectorPrint(filename, x_l[0], 0);

            /* Print residual */
            hypre_sprintf(filename, "ssamg_r.i%02d", (i + 1));
            HYPRE_SStructVectorPrint(filename, r_l[0], 0);
         }
      }
      else
      {
         /* r_l[0] is the fine grid residual computed after pre-smoothing */
         hypre_SStructInnerProd(r_l[0], r_l[0], &r_dot_r);
      }

      HYPRE_ANNOTATE_MGLEVEL_END(0);

      /* convergence check */
      if (tol > 0.0)
      {
         if (r_dot_r/b_dot_b < eps)
         {
            if (((rel_change) && (e_dot_e/x_dot_x) < eps) || (!rel_change))
            {
               i++; break;
            }
         }
      }
   }
   (ssamg_data -> num_iterations) = i;

   /*-----------------------------------------------------
    * Destroy Refs to A_in, x_in, b_in
    *-----------------------------------------------------*/
   /* hypre_SStructMatrixDestroy(A_l[0]); */
   /* hypre_SStructVectorDestroy(b_l[0]); */
   /* hypre_SStructVectorDestroy(x_l[0]); */

   hypre_EndTiming(ssamg_data -> time_index);
   hypre_SSAMGPrintLogging(ssamg_data);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
