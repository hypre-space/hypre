/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *  FAC cycle. Refinement patches are solved using relaxation.
 *  Note that the level solves compute corrections to the composite solution.
 ******************************************************************************/

#include "headers.h"
#include "fac.h"

#define DEBUG 0

HYPRE_Int
hypre_FACSolve3( void                 *fac_vdata,
                hypre_SStructMatrix  *A_user,
                hypre_SStructVector  *b_in,
                hypre_SStructVector  *x_in         )
{
   hypre_SStructGrid       *grid; 
   hypre_FACData           *fac_data           = fac_vdata;

   hypre_SStructMatrix     *A_in               =(fac_data-> A_rap);
   hypre_SStructMatrix    **A_level            =(fac_data-> A_level);
   hypre_SStructVector    **b_level            =(fac_data-> b_level);
   hypre_SStructVector    **x_level            =(fac_data-> x_level);
   hypre_SStructVector    **e_level            =(fac_data-> e_level);
   hypre_SStructPVector   **tx_level           =(fac_data-> tx_level);
   hypre_SStructVector     *tx                 =(fac_data-> tx);
   void                   **relax_data_level   =(fac_data-> relax_data_level);
   void                   **matvec_data_level  =(fac_data-> matvec_data_level);
   void                   **pmatvec_data_level =(fac_data-> pmatvec_data_level);
   void                   **restrict_data_level=(fac_data-> restrict_data_level);
   void                   **interp_data_level  =(fac_data-> interp_data_level);
   void                    *matvec_data        =(fac_data-> matvec_data);
   HYPRE_SStructSolver      csolver            =(fac_data-> csolver);

   HYPRE_Int                max_level          =(fac_data-> max_levels);
   HYPRE_Int               *levels             =(fac_data-> level_to_part);
   HYPRE_Int                max_cycles         =(fac_data-> max_cycles);
   HYPRE_Int                rel_change         =(fac_data-> rel_change);
   HYPRE_Int                zero_guess         =(fac_data-> zero_guess);
   HYPRE_Int                num_pre_smooth     =(fac_data-> num_pre_smooth);
   HYPRE_Int                num_post_smooth    =(fac_data-> num_post_smooth);
   HYPRE_Int                csolver_type       =(fac_data-> csolver_type);
   HYPRE_Int                logging            =(fac_data-> logging);
   double                  *norms              =(fac_data-> norms);
   double                  *rel_norms          =(fac_data-> rel_norms);
   double                   tol                =(fac_data-> tol);

   HYPRE_Int                part_crse= 0;
   HYPRE_Int                part_fine= 1;

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_ParCSRMatrix      *parcsrA;
   hypre_ParVector         *parx;
   hypre_ParVector         *pary;

   double                   b_dot_b, r_dot_r, eps;
   double                   e_dot_e, e_dot_e_l, x_dot_x;
                    
   HYPRE_Int                level, i;
   HYPRE_Int                ierr = 0;
  
   grid= hypre_SStructGraphGrid( hypre_SStructMatrixGraph(A_in) );

   /*--------------------------------------------------------------
    * Special cases
    *--------------------------------------------------------------*/

   hypre_BeginTiming(fac_data -> time_index);

   (fac_data -> num_iterations) = 0;

   /* if max_cycles is zero, return */
   if (max_cycles == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructVectorSetConstantValues(x_in, 0.0);
      }

      hypre_EndTiming(fac_data -> time_index);
      return ierr;
   }

   /*--------------------------------------------------------------
    * Convergence check- we need to compute the norm of the 
    * composite rhs.
    *--------------------------------------------------------------*/

   if (tol > 0.0)
   {
      /* eps = (tol^2) */

      hypre_SStructInnerProd(b_in, b_in, &b_dot_b);
      if (b_dot_b < 0.000000001)
      {
          hypre_SStructInnerProd(x_in, x_in, &b_dot_b);
      }

      eps = tol*tol;

      /* if rhs is zero, return a zero solution */

      if (b_dot_b == 0.0)
      {
         hypre_SStructVectorSetConstantValues(x_in, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         hypre_EndTiming(fac_data -> time_index);
         return ierr; 
      }
   }

   /*--------------------------------------------------------------
    * FAC-cycles:
    *--------------------------------------------------------------*/
   for (i = 0; i < max_cycles; i++)
   {
      hypre_SStructCopy(b_in, tx);
      hypre_SStructMatvecCompute(matvec_data, -1.0, A_in, x_in, 1.0, tx);

      /*-----------------------------------------------------------
       * convergence check 
       *-----------------------------------------------------------*/
      if (tol > 0.0)
      {
         /*-----------------------------------------------------------
          * Compute the inner product of the composite residual.
          *-----------------------------------------------------------*/
         hypre_SStructInnerProd(tx, tx, &r_dot_r);

         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
               rel_norms[i] = sqrt(r_dot_r/b_dot_b);
            else
               rel_norms[i] = 0.0;
         }

         /* always do at least 1 FAC V-cycle */
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

      /*-----------------------------------------------------------
       * Extract the level composite rhs's. Since we are using a
       * correction scheme fac cycle, the rhs's is the composite
       * residuals of A_in, x_in, and b_in.
       *-----------------------------------------------------------*/
      hypre_SStructPCopy(hypre_SStructVectorPVector(tx, levels[max_level]),
                         hypre_SStructVectorPVector(b_level[max_level], part_fine));

      for (level= 1; level<= max_level; level++)
      {
         hypre_SStructPCopy(hypre_SStructVectorPVector(tx, levels[level-1]),
                            hypre_SStructVectorPVector(b_level[level], part_crse));
      }

      /*--------------------------------------------------------------
       * Down cycle:
       *--------------------------------------------------------------*/
      hypre_SStructVectorSetConstantValues(x_level[max_level], 0.0);
      for (level= max_level; level> 0; level--)
      {
         /*-----------------------------------------------------------
          * local fine solve: the rhs has already been updated with 
          * the "unstructured" interface coupling. That is, since the
          * composite corrections are initialized to zero, the patch
          * fine-to-coarse boundary couplings (conditions) do not
          * contribute to the rhs of the patch equations.
          *-----------------------------------------------------------*/
          pA = hypre_SStructMatrixPMatrix(A_level[level], part_fine);
          px = hypre_SStructVectorPVector(x_level[level], part_fine);
          py = hypre_SStructVectorPVector(b_level[level], part_fine);
          
          hypre_FacLocalRelax(relax_data_level[level], pA, px, py,
                              num_pre_smooth, &zero_guess);
          
         /*-----------------------------------------------------------
          * set up the coarse part problem: update two-level composite
          * residual, restrict, and zero coarse approximation.
          *
          * The residual is updated using the patch solution. This
          * involves coarse-to-fine matvec contributions. Since 
          * part_crse of x_level is zero, only zero  fine-to-coarse 
          * contributions are involved.
          *-----------------------------------------------------------*/

         /* structured contribution */
          hypre_SStructPMatvecCompute(pmatvec_data_level[level],
                                     -1.0, pA, px, 1.0, py);
      
         /* unstructured contribution */
          parcsrA = hypre_SStructMatrixParCSRMatrix(A_level[level]);
          hypre_SStructVectorConvert(x_level[level], &parx);
          hypre_SStructVectorConvert(b_level[level], &pary);
          hypre_ParCSRMatrixMatvec(-1.0, parcsrA, parx, 1.0, pary);
          hypre_SStructVectorRestore(x_level[level], parx);
          hypre_SStructVectorRestore(b_level[level], pary);

         /*-----------------------------------------------------------
          *  restrict the two-level composite residual. 
          *  
          *  This involves restricting the two-level composite residual 
          *  of the current level to the part_fine rhs of the next
          *  descending level, or part_crse if the next descending 
          *  level is the coarsest. Part_fine of the two-level composite
          *  residual is resricted, part_crse is injected.
          *-----------------------------------------------------------*/
          if (level > 1)
          {
             hypre_FACRestrict2(restrict_data_level[level], 
                                b_level[level], 
                                hypre_SStructVectorPVector(b_level[level-1],part_fine));
          }
          else
          {
             hypre_FACRestrict2(restrict_data_level[level], 
                                b_level[level], 
                                hypre_SStructVectorPVector(b_level[level-1],part_crse));
          }

          hypre_SStructVectorSetConstantValues(x_level[level-1], 0.0);
      }

      /*-----------------------------------------------------------
       * coarsest solve:
       * The coarsest level is solved using the part_crse data of
       * A_level[0], b_level[0], x_level[0]. Therefore, copy the
       * solution to the part_fine.
       *-----------------------------------------------------------*/
       level= 0;
       if (csolver_type==1)
       {
           HYPRE_PCGSolve((HYPRE_Solver) csolver, 
                          (HYPRE_Matrix) A_level[0],
                          (HYPRE_Vector) b_level[0],
                          (HYPRE_Vector) x_level[0]);
       }
       else if (csolver_type==2)
       {
           HYPRE_SStructSysPFMGSolve(csolver, A_level[0], b_level[0], x_level[0]);
       }
       hypre_SStructPCopy(hypre_SStructVectorPVector(x_level[0], part_crse),
                          hypre_SStructVectorPVector(x_level[0], part_fine));

#if DEBUG
#endif

      /*-----------------------------------------------------------
       * Up cycle
       *-----------------------------------------------------------*/
       for (level= 1; level<= max_level; level++)
       {

         /*-----------------------------------------------------------
          * Interpolate error, update the residual, and correct 
          * (x = x + Pe_c). Interpolation is done in several stages:
          *   1)interpolate only the coarse unknowns away from the
          *     refinement patch: identity interpolation, interpolated
          *     to part_crse of the finer composite level.
          *   2) interpolate the coarse unknowns under the fine grid
          *      patch
          *-----------------------------------------------------------*/
          hypre_SStructVectorSetConstantValues(e_level[level], 0.0);
/*
hypre_SStructVectorSetConstantValues(x_level[max_level-1], 1.0);
*/

         /*-----------------------------------------------------------
          *  interpolation of unknowns away from the underlying 
          *  fine grid patch. Identity interpolation.
          *-----------------------------------------------------------*/
          hypre_FAC_IdentityInterp2(interp_data_level[level-1],
                             hypre_SStructVectorPVector(x_level[level-1], part_fine),
                             e_level[level]);

         /*-----------------------------------------------------------
          *  complete the interpolation- unknowns under the fine
          *  patch. Weighted interpolation.
          *-----------------------------------------------------------*/
          hypre_FAC_WeightedInterp2(interp_data_level[level-1],
                             hypre_SStructVectorPVector(x_level[level-1], part_fine),
                             e_level[level]);

         /*-----------------------------------------------------------
          *  add the correction to x_level
          *-----------------------------------------------------------*/
          hypre_SStructAxpy(1.0, e_level[level], x_level[level]);

         /*-----------------------------------------------------------
          *  update residual due to the interpolated correction
          *-----------------------------------------------------------*/
          if (num_post_smooth)
          {
             hypre_SStructMatvecCompute(matvec_data_level[level], -1.0,
                                        A_level[level], e_level[level], 
                                        1.0, b_level[level]);
          }

         /*-----------------------------------------------------------
          *  post-smooth on the refinement patch
          *-----------------------------------------------------------*/
          if (num_post_smooth)
          {
             hypre_SStructPVectorSetConstantValues(tx_level[level], 0.0);
             pA = hypre_SStructMatrixPMatrix(A_level[level], part_fine);
             py = hypre_SStructVectorPVector(b_level[level], part_fine);
          
             hypre_FacLocalRelax(relax_data_level[level], pA, tx_level[level], py,
                                 num_post_smooth, &zero_guess);

         /*-----------------------------------------------------------
          *  add the post-smooth solution to x_level and to the error
          *  vector e_level if level= max_level. The e_levels should
          *  contain only the correction to x_in. 
          *-----------------------------------------------------------*/
             hypre_SStructPAxpy(1.0, tx_level[level], 
                                hypre_SStructVectorPVector(x_level[level], part_fine));

             if (level == max_level)
             {
                hypre_SStructPAxpy(1.0, tx_level[level], 
                                hypre_SStructVectorPVector(e_level[level], part_fine));
             }
          }

      }

      /*--------------------------------------------------------------
       * Add two-level corrections x_level to the composite solution
       * x_in. 
       *
       * Notice that except for the finest two-level sstruct_vector, 
       * only the part_crse of each two-level sstruct_vector has
       * a correction to x_in. For max_level, both part_crse and
       * part_fine has a correction to x_in.
       *--------------------------------------------------------------*/
 
      hypre_SStructPAxpy(1.0, 
                         hypre_SStructVectorPVector(x_level[max_level], part_fine),
                         hypre_SStructVectorPVector(x_in, levels[max_level]));

      for (level= 1; level<= max_level; level++)
      {
          hypre_SStructPAxpy(1.0, 
                             hypre_SStructVectorPVector(x_level[level], part_crse),
                             hypre_SStructVectorPVector(x_in, levels[level-1]) );
      }

      /*-----------------------------------------------
       * convergence check 
       *-----------------------------------------------*/
      if ((tol > 0.0) && (rel_change))
      {
          hypre_SStructInnerProd(x_in, x_in, &x_dot_x);

          hypre_SStructInnerProd(e_level[max_level], e_level[max_level], &e_dot_e);
          for (level= 1; level< max_level; level++)
          {
             hypre_SStructPInnerProd( 
                         hypre_SStructVectorPVector(e_level[level], part_crse),
                         hypre_SStructVectorPVector(e_level[level], part_crse),
                         &e_dot_e_l);
             e_dot_e += e_dot_e_l;
          }
      }

      (fac_data -> num_iterations) = (i + 1);

   }
#if DEBUG
#endif

   hypre_EndTiming(fac_data -> time_index);

   return ierr;
}

