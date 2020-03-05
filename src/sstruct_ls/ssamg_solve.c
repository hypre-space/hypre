/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

#define DEBUG 0

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
   HYPRE_Int             *num_iterations  = &hypre_SSAMGDataNumIterations(ssamg_data);
   HYPRE_Real            *norms           =  hypre_SSAMGDataNorms(ssamg_data);
   HYPRE_Real            *rel_norms       =  hypre_SSAMGDataRelNorms(ssamg_data);

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
   HYPRE_Real            b_dot_b, r_dot_r, eps = 0;
   HYPRE_Real            e_dot_e = 0, x_dot_x = 1;
   HYPRE_Int             i, l;

#if DEBUG
   char                   filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   hypre_BeginTiming(*time_index);

   /*-----------------------------------------------------
    * Refs to A,x,b (the SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/
   HYPRE_SStructMatrixDestroy(A_l[0]);
   HYPRE_SStructVectorDestroy(b_l[0]);
   HYPRE_SStructVectorDestroy(x_l[0]);
   hypre_SStructMatrixRef(A, &A_l[0]);
   hypre_SStructVectorRef(b, &b_l[0]);
   hypre_SStructVectorRef(x, &x_l[0]);

   *num_iterations = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructVectorSetConstantValues(x_l[0], 0.0);
      }

      hypre_EndTiming(*time_index);
      return hypre_error_flag;
   }

#if DEBUG
   hypre_sprintf(filename, "zout_A");
   HYPRE_SStructMatrixPrint(filename, A_l[0], 0);
   hypre_sprintf(filename, "zout_x");
   HYPRE_SStructVectorPrint(filename, x_l[0], 0);
   hypre_sprintf(filename, "zout_b");
   HYPRE_SStructVectorPrint(filename, b_l[0], 0);
#endif

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      hypre_SStructInnerProd(b_l[0], b_l[0], &b_dot_b);
      eps = tol*tol;

      /* if rhs is zero, return a zero solution */
      if (!(b_dot_b > 0.0))
      {
         hypre_SStructVectorSetConstantValues(x_l[0], 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(*time_index);
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

      /* fine grid pre-relaxation */
      hypre_SSAMGRelaxSetPreRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      hypre_SStructCopy(b_l[0], r_l[0]);
      hypre_SStructMatvecCompute(matvec_data_l[0], -1.0,
                                 A_l[0], x_l[0], 1.0, r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         hypre_SStructInnerProd(r_l[0], r_l[0], &r_dot_r);

         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
            {
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            }
            else
            {
               rel_norms[i] = 0.0;
            }
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r/b_dot_b < eps) && (i > 0))
         {
            if ( ((rel_change) && (e_dot_e/x_dot_x) < eps) || (!rel_change) )
            {
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         hypre_SStructMatvecCompute(restrict_data_l[0], 1.0,
                                    RT_l[0], r_l[0], 0.0, b_l[1]);
#if DEBUG
         hypre_sprintf(filename, "zout_xdown.%02d.%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
         hypre_sprintf(filename, "zout_rdown.%02d.%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, r_l[0], 0);
         hypre_sprintf(filename, "zout_b.%02d.%02d", i, 1);
         HYPRE_SStructVectorPrint(filename, b_l[1], 0);
#endif
         for (l = 1; l <= (num_levels - 2); l++)
         {
            //if (active_l[l])
            if (l > 0) // True always
            {
               /* pre-relaxation */
               hypre_SSAMGRelaxSetPreRelax(relax_data_l[l]);
               hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
               hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
               hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

               /* compute residual (b - Ax) */
               hypre_SStructCopy(b_l[l], r_l[l]);
               hypre_SStructMatvecCompute(matvec_data_l[l], -1.0,
                                          A_l[l], x_l[l], 1.0, r_l[l]);
            }
            else
            {
               /* inactive level, set x=0, so r=(b-Ax)=b */
               hypre_SStructVectorSetConstantValues(x_l[l], 0.0);
               hypre_SStructCopy(b_l[l], r_l[l]);
            }

            /* restrict residual */
            hypre_SStructMatvecCompute(restrict_data_l[l], 1.0,
                                       RT_l[l], r_l[l], 0.0, b_l[l+1]);
#if DEBUG
            hypre_sprintf(filename, "zout_xdown.%02d.%02d", i, l);
            HYPRE_SStructVectorPrint(filename, x_l[l], 0);
            hypre_sprintf(filename, "zout_rdown.%02d.%02d", i, l);
            HYPRE_SStructVectorPrint(filename, r_l[l], 0);
            hypre_sprintf(filename, "zout_b.%02d.%02d", i, l+1);
            HYPRE_SStructVectorPrint(filename, b_l[l+1], 0);
#endif
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/

         //if (active_l[l])
         if (l > 0) // True always
         {
            hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 1);
            hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }
         else
         {
            /* inactive level, set x=0, so r=(b-Ax)=b */
            hypre_SStructVectorSetConstantValues(x_l[l], 0.0);
         }
#if DEBUG
         hypre_sprintf(filename, "zout_xbottom.%02d.%02d", i, l);
         HYPRE_SStructVectorPrint(filename, x_l[l], 0);
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
#if DEBUG
            hypre_sprintf(filename, "zout_eup.%02d.%02d", i, l);
            HYPRE_SStructVectorPrint(filename, e_l[l], 0);
            hypre_sprintf(filename, "zout_xup.%02d.%02d", i, l);
            HYPRE_SStructVectorPrint(filename, x_l[l], 0);
#endif
            //if (active_l[l])
            if (l > 0) // True always
            {
               /* post-relaxation */
               hypre_SSAMGRelaxSetPostRelax(relax_data_l[l]);
               hypre_SSAMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
               hypre_SSAMGRelaxSetZeroGuess(relax_data_l[l], 0);
               hypre_SSAMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         hypre_SStructMatvecCompute(interp_data_l[0], 1.0,
                                    P_l[0], x_l[1], 0.0, e_l[0]);
         hypre_SStructAxpy(1.0, e_l[0], x_l[0]);
#if DEBUG
         hypre_sprintf(filename, "zout_eup.%02d.%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, e_l[0], 0);
         hypre_sprintf(filename, "zout_xup.%02d.%02d", i, 0);
         HYPRE_SStructVectorPrint(filename, x_l[0], 0);
#endif
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

      /* fine grid post-relaxation */
      hypre_SSAMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_SSAMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_SSAMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_SSAMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

      *num_iterations = i + 1;
   }

   /*-----------------------------------------------------
    * Destroy Refs to A_in, x_in, b_in
    *-----------------------------------------------------*/
   /* hypre_SStructMatrixDestroy(A_l[0]); */
   /* hypre_SStructVectorDestroy(b_l[0]); */
   /* hypre_SStructVectorDestroy(x_l[0]); */

   hypre_EndTiming(*time_index);

   return hypre_error_flag;
}
