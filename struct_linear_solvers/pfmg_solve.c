/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_PFMGSolve
 *--------------------------------------------------------------------------*/

int
hypre_PFMGSolve( void               *pfmg_vdata,
                 hypre_StructMatrix *A,
                 hypre_StructVector *b,
                 hypre_StructVector *x         )
{
   hypre_PFMGData       *pfmg_data = pfmg_vdata;

   double                tol             = (pfmg_data -> tol);
   int                   max_iter        = (pfmg_data -> max_iter);
   int                   rel_change      = (pfmg_data -> rel_change);
   int                   zero_guess      = (pfmg_data -> zero_guess);
   int                   num_pre_relax   = (pfmg_data -> num_pre_relax);
   int                   num_post_relax  = (pfmg_data -> num_post_relax);
   int                   num_levels      = (pfmg_data -> num_levels);
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
   int                   logging         = (pfmg_data -> logging);
   double               *norms           = (pfmg_data -> norms);
   double               *rel_norms       = (pfmg_data -> rel_norms);
   int                  *active_l        = (pfmg_data -> active_l);

   double                b_dot_b, r_dot_r, eps;
   double                e_dot_e, x_dot_x;
                    
   int                   i, l;
                    
   int                   ierr = 0;
#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   hypre_BeginTiming(pfmg_data -> time_index);

   hypre_DestroyStructMatrix(A_l[0]);
   hypre_DestroyStructVector(b_l[0]);
   hypre_DestroyStructVector(x_l[0]);
   A_l[0] = hypre_RefStructMatrix(A);
   b_l[0] = hypre_RefStructVector(b);
   x_l[0] = hypre_RefStructVector(x);

   (pfmg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SetStructVectorConstantValues(x, 0.0);
      }

      hypre_EndTiming(pfmg_data -> time_index);
      return ierr;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2)*<b,b> */
      b_dot_b = hypre_StructInnerProd(b_l[0], b_l[0]);
      eps = (tol*tol)*b_dot_b;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_SetStructVectorConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(pfmg_data -> time_index);
         return ierr;
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
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            else
               rel_norms[i] = 0.0;
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e/x_dot_x) < (eps/b_dot_b))
                  break;
            }
            else
            {
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         hypre_PFMGRestrict(restrict_data_l[0], RT_l[0], r_l[0], b_l[1]);
#if DEBUG
         sprintf(filename, "zout_xdown.%02d", 0);
         hypre_PrintStructVector(filename, x_l[0], 0);
         sprintf(filename, "zout_rdown.%02d", 0);
         hypre_PrintStructVector(filename, r_l[0], 0);
         sprintf(filename, "zout_b.%02d", 1);
         hypre_PrintStructVector(filename, b_l[1], 0);
#endif
         for (l = 1; l <= (num_levels - 2); l++)
         {
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
               hypre_SetStructVectorConstantValues(x_l[l], 0.0);
               hypre_StructCopy(b_l[l], r_l[l]);
            }

            /* restrict residual */
            hypre_PFMGRestrict(restrict_data_l[l], RT_l[l], r_l[l], b_l[l+1]);
#if DEBUG
            sprintf(filename, "zout_xdown.%02d", l);
            hypre_PrintStructVector(filename, x_l[l], 0);
            sprintf(filename, "zout_rdown.%02d", l);
            hypre_PrintStructVector(filename, r_l[l], 0);
            sprintf(filename, "zout_b.%02d", l+1);
            hypre_PrintStructVector(filename, b_l[l+1], 0);
#endif
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/

         hypre_PFMGRelaxSetZeroGuess(relax_data_l[l], 1);
         hypre_PFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
#if DEBUG
         sprintf(filename, "zout_xbottom.%02d", l);
         hypre_PrintStructVector(filename, x_l[l], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            hypre_PFMGInterp(interp_data_l[l], P_l[l], x_l[l+1], e_l[l]);
            hypre_StructAxpy(1.0, e_l[l], x_l[l]);
#if DEBUG
            sprintf(filename, "zout_eup.%02d", l);
            hypre_PrintStructVector(filename, e_l[l], 0);
            sprintf(filename, "zout_xup.%02d", l);
            hypre_PrintStructVector(filename, x_l[l], 0);
#endif
            if (active_l[l])
            {
               /* post-relaxation */
               hypre_PFMGRelaxSetPostRelax(relax_data_l[l]);
               hypre_PFMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
               hypre_PFMGRelaxSetZeroGuess(relax_data_l[l], 0);
               hypre_PFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         hypre_PFMGInterp(interp_data_l[0], P_l[0], x_l[1], e_l[0]);
         hypre_StructAxpy(1.0, e_l[0], x_l[0]);
#if DEBUG
         sprintf(filename, "zout_eup.%02d", 0);
         hypre_PrintStructVector(filename, e_l[0], 0);
         sprintf(filename, "zout_xup.%02d", 0);
         hypre_PrintStructVector(filename, x_l[0], 0);
#endif
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

      /* fine grid post-relaxation */
      hypre_PFMGRelaxSetPostRelax(relax_data_l[0]);
      hypre_PFMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_PFMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_PFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

      (pfmg_data -> num_iterations) = (i + 1);
   }

   hypre_EndTiming(pfmg_data -> time_index);

   return ierr;
}

