/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
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
#include "smg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * hypre_SMGSolve
 *    This is the main solve routine for the Schaffer multigrid method.
 *    This solver works for 1D, 2D, or 3D linear systems.  The dimension
 *    is determined by the hypre_StructStencilDim argument of the matrix
 *    stencil.  The hypre_StructGridDim argument of the matrix grid is
 *    allowed to be larger than the dimension of the solver, and in fact,
 *    this feature is used in the smaller-dimensional solves required
 *    in the relaxation method for both the 2D and 3D algorithms.  This
 *    allows one to do multiple 2D or 1D solves in parallel (e.g., multiple
 *    2D solves, where the 2D problems are "stacked" planes in 3D).
 *    The only additional requirement is that the linear system(s) data
 *    be contiguous in memory.
 *
 * Notes:
 * - Iterations are counted as follows: 1 iteration consists of a
 *   V-cycle plus an extra pre-relaxation.  If the number of MG levels
 *   is equal to 1, then only the extra pre-relaxation step is done at
 *   each iteration.  When the solver exits because the maximum number
 *   of iterations is reached, the last extra pre-relaxation is not done.
 *   This allows one to use the solver as a preconditioner for conjugate
 *   gradient and insure symmetry.
 * - hypre_SMGRelax is the relaxation routine.  There are different "data"
 *   structures for each call to reflect different arguments and parameters.
 *   One important parameter sets whether or not an initial guess of zero
 *   is to be used in the relaxation.
 * - hypre_SMGResidual computes the residual, b - Ax.
 * - hypre_SemiRestrict restricts the residual to the coarse grid.
 * - hypre_SemiInterp interpolates the coarse error and adds it to the
 *   fine grid solution.
 *
 *--------------------------------------------------------------------------*/

int
hypre_SMGSolve( void               *smg_vdata,
                hypre_StructMatrix *A,
                hypre_StructVector *b,
                hypre_StructVector *x         )
{

   hypre_SMGData        *smg_data = smg_vdata;

   double                tol             = (smg_data -> tol);
   int                   max_iter        = (smg_data -> max_iter);
   int                   rel_change      = (smg_data -> rel_change);
   int                   zero_guess      = (smg_data -> zero_guess);
   int                   num_levels      = (smg_data -> num_levels);
   int                   num_pre_relax   = (smg_data -> num_pre_relax);
   int                   num_post_relax  = (smg_data -> num_post_relax);
   hypre_IndexRef        base_index      = (smg_data -> base_index);
   hypre_IndexRef        base_stride     = (smg_data -> base_stride);
   hypre_StructMatrix  **A_l             = (smg_data -> A_l);
   hypre_StructMatrix  **PT_l            = (smg_data -> PT_l);
   hypre_StructMatrix  **R_l             = (smg_data -> R_l);
   hypre_StructVector  **b_l             = (smg_data -> b_l);
   hypre_StructVector  **x_l             = (smg_data -> x_l);
   hypre_StructVector  **r_l             = (smg_data -> r_l);
   hypre_StructVector  **e_l             = (smg_data -> e_l);
   void                **relax_data_l    = (smg_data -> relax_data_l);
   void                **residual_data_l = (smg_data -> residual_data_l);
   void                **restrict_data_l = (smg_data -> restrict_data_l);
   void                **interp_data_l   = (smg_data -> interp_data_l);
   int                   logging         = (smg_data -> logging);
   double               *norms           = (smg_data -> norms);
   double               *rel_norms       = (smg_data -> rel_norms);

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

   hypre_BeginTiming(smg_data -> time_index);

   hypre_StructMatrixDestroy(A_l[0]);
   hypre_StructVectorDestroy(b_l[0]);
   hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   (smg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(smg_data -> time_index);
      return ierr;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = hypre_StructInnerProd(b_l[0], b_l[0]);
      eps = tol*tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_StructVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(smg_data -> time_index);
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
      if (num_levels > 1)
      {
         hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 0, 0);
         hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 1, 1);
      }
      hypre_SMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      hypre_SMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      hypre_SMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      hypre_SMGResidual(residual_data_l[0], A_l[0], x_l[0], b_l[0], r_l[0]);

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
         if ((r_dot_r/b_dot_b < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e/x_dot_x) < eps)
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
         hypre_SemiRestrict(restrict_data_l[0], R_l[0], r_l[0], b_l[1]);
#if DEBUG
         if(hypre_StructStencilDim(hypre_StructMatrixStencil(A)) == 3)
         {
            sprintf(filename, "zout_xdown.%02d", 0);
            hypre_StructVectorPrint(filename, x_l[0], 0);
            sprintf(filename, "zout_rdown.%02d", 0);
            hypre_StructVectorPrint(filename, r_l[0], 0);
            sprintf(filename, "zout_b.%02d", 1);
            hypre_StructVectorPrint(filename, b_l[1], 0);
         }
#endif
         for (l = 1; l <= (num_levels - 2); l++)
         {
            /* pre-relaxation */
            hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 0);
            hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 1);
            hypre_SMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
            hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 1);
            hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

            /* compute residual (b - Ax) */
            hypre_SMGResidual(residual_data_l[l],
                              A_l[l], x_l[l], b_l[l], r_l[l]);

            /* restrict residual */
            hypre_SemiRestrict(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1]);
#if DEBUG
            if(hypre_StructStencilDim(hypre_StructMatrixStencil(A)) == 3)
            {
               sprintf(filename, "zout_xdown.%02d", l);
               hypre_StructVectorPrint(filename, x_l[l], 0);
               sprintf(filename, "zout_rdown.%02d", l);
               hypre_StructVectorPrint(filename, r_l[l], 0);
               sprintf(filename, "zout_b.%02d", l+1);
               hypre_StructVectorPrint(filename, b_l[l+1], 0);
            }
#endif
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/

         hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 1);
         hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
#if DEBUG
         if(hypre_StructStencilDim(hypre_StructMatrixStencil(A)) == 3)
         {
            sprintf(filename, "zout_xbottom.%02d", l);
            hypre_StructVectorPrint(filename, x_l[l], 0);
         }
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            hypre_SemiInterp(interp_data_l[l], PT_l[l], x_l[l+1], e_l[l]);
            hypre_StructAxpy(1.0, e_l[l], x_l[l]);
#if DEBUG
            if(hypre_StructStencilDim(hypre_StructMatrixStencil(A)) == 3)
            {
               sprintf(filename, "zout_eup.%02d", l);
               hypre_StructVectorPrint(filename, e_l[l], 0);
               sprintf(filename, "zout_xup.%02d", l);
               hypre_StructVectorPrint(filename, x_l[l], 0);
            }
#endif
            /* post-relaxation */
            hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 1);
            hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 0);
            hypre_SMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
            hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 0);
            hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         hypre_SemiInterp(interp_data_l[0], PT_l[0], x_l[1], e_l[0]);
         hypre_SMGAxpy(1.0, e_l[0], x_l[0], base_index, base_stride);
#if DEBUG
         if(hypre_StructStencilDim(hypre_StructMatrixStencil(A)) == 3)
         {
            sprintf(filename, "zout_eup.%02d", 0);
            hypre_StructVectorPrint(filename, e_l[0], 0);
            sprintf(filename, "zout_xup.%02d", 0);
            hypre_StructVectorPrint(filename, x_l[0], 0);
         }
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
      if (num_levels > 1)
      {
         hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 0, 1);
         hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 1, 0);
      }
      hypre_SMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      hypre_SMGRelaxSetZeroGuess(relax_data_l[0], 0);
      hypre_SMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

      (smg_data -> num_iterations) = (i + 1);
   }

   hypre_EndTiming(smg_data -> time_index);

   return ierr;
}
