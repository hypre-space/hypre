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
#include "smg3.h"

/*--------------------------------------------------------------------------
 * zzz_SMG3Solve
 *--------------------------------------------------------------------------*/

int
zzz_SMG3Solve( zzz_SMG3Data     *smg3_data,
               zzz_StructVector *b,
               zzz_StructVector *x         )
{
   int ierr;

   /*-----------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------*/

   b_l[0] = b;
   x_l[0] = x;

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

      /* relax (tol = 0.0) */
      if (i == 0)
         zzz_SMG3Relax(relax_data_initial, x_l[0], b_l[0]); /* (zero = zero) */
      else
         zzz_SMG3Relax(relax_data_l[0], x_l[0], b_l[0]);    /* (zero = 0) */

      /* compute residual (b - Ax) (use r_l[0] = temp_vec_l[0]) */
      zzz_SMGResidual(A_l[0], x_l[0], b_l[0], residual_compute_pkg[0], r_l[0]);

      /* do convergence check */
      if (tol > 0.0)
      {
         r_dot_r = zzz_StructInnerProd(r_l[0], r_l[0]);
         if (r_dot_r < eps)
            break;

         if (logging > 0)
         {
            norm_log[i]     = sqrt(r_dot_r);
            rel_norm_log[i] = b_dot_b ? sqrt(r_dot_r/b_dot_b) : 0.0;
         }
#if 0
   if(!amps_Rank(amps_CommWorld))
      amps_Printf("Iteration (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
		  i, sqrt(r_dot_r), (b_dot_b ? sqrt(r_dot_r/b_dot_b) : 0.0));
#endif
      }

      /* restrict residual */
      zzz_SMGRestrict(restrict_data_l[0], R_l[0], r_l[0], b_l[1]);

#if 0
      /* for debugging purposes */
      PrintVector("b.01", b_l[1]);
#endif

      for (l = 1; l <= (num_levels - 2); l++)
      {
         /* relax (tol = 0.0; zero = 1) */
         zzz_SMG3Relax(relax_data_l[l], x_l[l], b_l[l]);

	 /* compute residual (b - Ax) (use r_l[l] = temp_vec_l[l]) */
         zzz_SMGResidual(A_l[l], x_l[l], b_l[l],
                         residual_compute_pkg[l], r_l[l]);

	 /* restrict residual */
         zzz_SMGRestrict(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1]);

#if 0
	 /* for debugging purposes */
	 {
	    char  filename[255];

	    sprintf(filename, "b.%02d", l+1);
	    PrintVector(filename, b_l[l+1]);
	 }
#endif
      }

      /*--------------------------------------------------
       * Bottom
       *--------------------------------------------------*/

      /* solve the coarsest system (tol = 0.0; zero = 1) */
      zzz_SMG3Relax(relax_data_coarsest, x_l[l], b_l[l]);

      /*--------------------------------------------------
       * Up cycle
       *--------------------------------------------------*/

      for (l = (num_levels - 2); l >= 1; l--)
      {
	 /* interpolate error and update solution */
	 zzz_SMGIntAdd(intadd_data_l[l], PT_l[l], x_l[l+1], x_l[l]);
#if 0
	 /* for debugging purposes */
	 {
	    char  filename[255];

	    sprintf(filename, "e.%02d", l);
	    PrintVector(filename, e_l[l]);
	 }
#endif

         /* relax (tol = 0.0; zero = 0) */
         zzz_SMG3Relax(relax_data_l[l], x_l[l], b_l[l]);
      }

      /* interpolate error and update solution */
      zzz_SMGIntAdd(intadd_data_l[0], PT_l[0], x_l[1], x_l[0]);
#if 0
      /* for debugging purposes */
      PrintVector("e.00", temp_vec_l[0]);
#endif

      /* relax (tol = 0.0; zero = 0) */
      zzz_SMG3Relax(relax_data_l[0], x_l[0], b_l[0]);
   }

#if 1
   if (tol > 0.0)
   {
      if(!amps_Rank(amps_CommWorld))
         amps_Printf("Iterations = %d, ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                     i, sqrt(r_dot_r),
                     (b_dot_b ? sqrt(r_dot_r/b_dot_b) : 0.0));
   }
#endif

   return ierr;
}
