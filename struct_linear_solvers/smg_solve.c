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

/*--------------------------------------------------------------------------
 * zzz_SMGSolve
 *    This is the main solve routine for the Schaffer multigrid method.
 *    This solver works for 1D, 2D, or 3D linear systems.  The dimension
 *    is determined by the zzz_StructStencilDim argument of the matrix
 *    stencil.  The zzz_StructGridDim argument of the matrix grid is
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
 * - zzz_SMGRelax is the relaxation routine.  There are different "data"
 *   structures for each call to reflect different arguments and parameters.
 *   One important parameter sets whether or not an initial guess of zero
 *   is to be used in the relaxation.
 * - zzz_SMGResidual computes the residual, b - Ax.
 * - zzz_SMGRestrict restricts the residual to the coarse grid.
 * - zzz_SMGIntAdd interpolates the coarse error and adds it to the
 *   fine grid solution.
 *
 *--------------------------------------------------------------------------*/

int
zzz_SMGSolve( void             *smg_vdata,
              zzz_StructMatrix *A,
              zzz_StructVector *b,
              zzz_StructVector *x         )
{
   zzz_SMGData        *smg_data = smg_vdata;

   double              tol             = (smg_data -> tol);
   int                 max_iter        = (smg_data -> max_iter);
   int                 num_levels      = (smg_data -> num_levels);
   zzz_StructMatrix  **A_l             = (smg_data -> A_l);
   zzz_StructMatrix  **PT_l            = (smg_data -> PT_l);
   zzz_StructMatrix  **R_l             = (smg_data -> R_l);
   zzz_StructVector  **b_l             = (smg_data -> b_l);
   zzz_StructVector  **x_l             = (smg_data -> x_l);
   zzz_StructVector  **r_l             = (smg_data -> r_l);
   zzz_StructVector  **e_l             = (smg_data -> e_l);
   void              **relax_data_l    = (smg_data -> relax_data_l);
   void              **residual_data_l = (smg_data -> residual_data_l);
   void              **restrict_data_l = (smg_data -> restrict_data_l);
   void              **intadd_data_l   = (smg_data -> intadd_data_l);
   int                 logging         = (smg_data -> logging);
   double             *norms           = (smg_data -> norms);
   double             *rel_norms       = (smg_data -> rel_norms);

   double              b_dot_b, r_dot_r, eps;

   int                 i, l;

   int                 ierr;

   zzz_BeginTiming(smg_data -> time_index);

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   if (tol > 0.0)
   {
      /* eps = (tol^2)*<b,b> */
      b_dot_b = zzz_StructInnerProd(b_l[0], b_l[0]);
      eps = (tol*tol)*b_dot_b;
   }

   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle
       *--------------------------------------------------*/

      zzz_SMGRelaxSetMaxIter(relax_data_l[0], 1);
      zzz_SMGRelaxSetRegSpaceRank(relax_data_l[0], 0, 0);
      zzz_SMGRelaxSetRegSpaceRank(relax_data_l[0], 1, 1);
      if (i == 0)
      {
         if (smg_data -> zero_guess)
            zzz_SMGRelaxSetZeroGuess(relax_data_l[0]);
         else
            zzz_SMGRelaxSetNonZeroGuess(relax_data_l[0]);
      }
      else
      {
         zzz_SMGRelaxSetNonZeroGuess(relax_data_l[0]);
      }
      zzz_SMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zzz_SMGResidual(residual_data_l[0], A_l[0], x_l[0], b_l[0], r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = zzz_StructInnerProd(r_l[0], r_l[0]);

         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            else
               rel_norms[i] = 0.0;
         }

         if (r_dot_r < eps)
            break;
      }

      for (l = 0; l <= (num_levels - 2); l++)
      {
         if (l > 0)
         {
            zzz_SMGRelaxSetMaxIter(relax_data_l[l], 1);
            zzz_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 0);
            zzz_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 1);
            zzz_SMGRelaxSetZeroGuess(relax_data_l[l]);
            zzz_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            zzz_SMGResidual(residual_data_l[l],
                            A_l[l], x_l[l], b_l[l], r_l[l]);
         }
         zzz_SMGRestrict(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1]);
#if 0
         /* for debugging purposes */
         {
            char  filename[255];

            sprintf(filename, "zout_xbefore.%02d", l);
            zzz_PrintStructVector(filename, x_l[l], 0);
            sprintf(filename, "zout_b.%02d", l+1);
            zzz_PrintStructVector(filename, b_l[l+1], 0);
         }
#endif
      }

      /*--------------------------------------------------
       * Bottom
       *--------------------------------------------------*/

      if (num_levels > 1)
      {
         zzz_SMGRelaxSetZeroGuess(relax_data_l[l]);
         zzz_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }

      /*--------------------------------------------------
       * Up cycle
       *--------------------------------------------------*/

      for (l = (num_levels - 2); l >= 0; l--)
      {
         zzz_SMGIntAdd(intadd_data_l[l], PT_l[l], x_l[l+1], e_l[l], x_l[l]);
#if 0
         /* for debugging purposes */
         {
            char  filename[255];

            sprintf(filename, "zout_xafter.%02d", l);
            zzz_PrintStructVector(filename, x_l[l], 0);
         }
#endif
         zzz_SMGRelaxSetMaxIter(relax_data_l[l], 1);
         zzz_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 1);
         zzz_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 0);
         zzz_SMGRelaxSetNonZeroGuess(relax_data_l[l]);
         zzz_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
      }

      (smg_data -> num_iterations) = (i + 1);
   }

   zzz_EndTiming(smg_data -> time_index);

   return ierr;
}
