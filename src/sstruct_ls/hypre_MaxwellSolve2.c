/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_MaxwellSolve- note that there is no input operator Aee. We assume
 * that maxwell_vdata has the exact operators. This prevents the need to
 * to recompute Ann in the solve phase. However, we do allow the f_edge &
 * u_edge to change per call.
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MaxwellSolve2( void                * maxwell_vdata,
                     hypre_SStructMatrix * A_in,
                     hypre_SStructVector * f,
                     hypre_SStructVector * u )
{
   hypre_MaxwellData     *maxwell_data = maxwell_vdata;

   hypre_ParVector       *f_edge;
   hypre_ParVector       *u_edge;

   HYPRE_Int              max_iter     = maxwell_data-> max_iter;
   double                 tol          = maxwell_data-> tol;
   HYPRE_Int              rel_change   = maxwell_data-> rel_change;
   HYPRE_Int              zero_guess   = maxwell_data-> zero_guess;
   HYPRE_Int              npre_relax   = maxwell_data-> num_pre_relax;
   HYPRE_Int              npost_relax  = maxwell_data-> num_post_relax;

   hypre_ParCSRMatrix   **Ann_l        = maxwell_data-> Ann_l;
   hypre_ParCSRMatrix   **Pn_l         = maxwell_data-> Pn_l;
   hypre_ParCSRMatrix   **RnT_l        = maxwell_data-> RnT_l;
   hypre_ParVector      **bn_l         = maxwell_data-> bn_l;
   hypre_ParVector      **xn_l         = maxwell_data-> xn_l;
   hypre_ParVector      **resn_l       = maxwell_data-> resn_l;
   hypre_ParVector      **en_l         = maxwell_data-> en_l;
   hypre_ParVector      **nVtemp2_l    = maxwell_data-> nVtemp2_l;
   HYPRE_Int            **nCF_marker_l = maxwell_data-> nCF_marker_l;
   double                *nrelax_weight= maxwell_data-> nrelax_weight;
   double                *nomega       = maxwell_data-> nomega;
   HYPRE_Int              nrelax_type  = maxwell_data-> nrelax_type;
   HYPRE_Int              node_numlevs = maxwell_data-> node_numlevels;

   hypre_ParCSRMatrix    *Tgrad        = maxwell_data-> Tgrad;
   hypre_ParCSRMatrix    *T_transpose  = maxwell_data-> T_transpose;

   hypre_ParCSRMatrix   **Aee_l        = maxwell_data-> Aee_l;
   hypre_IJMatrix       **Pe_l         = maxwell_data-> Pe_l;
   hypre_IJMatrix       **ReT_l        = maxwell_data-> ReT_l;
   hypre_ParVector      **be_l         = maxwell_data-> be_l;
   hypre_ParVector      **xe_l         = maxwell_data-> xe_l;
   hypre_ParVector      **rese_l       = maxwell_data-> rese_l;
   hypre_ParVector      **ee_l         = maxwell_data-> ee_l;
   hypre_ParVector      **eVtemp2_l    = maxwell_data-> eVtemp2_l;
   HYPRE_Int            **eCF_marker_l = maxwell_data-> eCF_marker_l;
   double                *erelax_weight= maxwell_data-> erelax_weight;
   double                *eomega       = maxwell_data-> eomega;
   HYPRE_Int              erelax_type  = maxwell_data-> erelax_type;
   HYPRE_Int              edge_numlevs = maxwell_data-> edge_numlevels;

   HYPRE_Int            **BdryRanks_l  = maxwell_data-> BdryRanks_l;
   HYPRE_Int             *BdryRanksCnts_l= maxwell_data-> BdryRanksCnts_l;

   HYPRE_Int              logging      = maxwell_data-> logging;
   double                *norms        = maxwell_data-> norms;
   double                *rel_norms    = maxwell_data-> rel_norms;

   HYPRE_Int              Solve_err_flag;
   HYPRE_Int              relax_local, cycle_param;
                                                                                                            
   double                 b_dot_b, r_dot_r, eps;
   double                 e_dot_e, x_dot_x;

   HYPRE_Int              i, j;
   HYPRE_Int              level;

   HYPRE_Int              ierr= 0;

   
   /* added for the relaxation routines */
   hypre_ParVector *ze = NULL;

   if (hypre_NumThreads() > 1)
   {
      /* Aee is always bigger than Ann */

      ze = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(Aee_l[0]),
                                hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                hypre_ParCSRMatrixRowStarts(Aee_l[0]));
      hypre_ParVectorInitialize(ze);
      hypre_ParVectorSetPartitioningOwner(ze,0);

   }

   hypre_BeginTiming(maxwell_data-> time_index);

   hypre_SStructVectorConvert(f, &f_edge);
   hypre_SStructVectorConvert(u, &u_edge);
   hypre_ParVectorZeroBCValues(f_edge, BdryRanks_l[0], BdryRanksCnts_l[0]);
   hypre_ParVectorZeroBCValues(u_edge, BdryRanks_l[0], BdryRanksCnts_l[0]);
   be_l[0]= f_edge;
   xe_l[0]= u_edge;

  /* the nodal fine vectors: xn= 0. bn= T'*(be- Aee*xe) is updated in the cycle. */
   hypre_ParVectorSetConstantValues(xn_l[0], 0.0);

   relax_local= 0;
   cycle_param= 0;

  (maxwell_data-> num_iterations) = 0;
  /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_ParVectorSetConstantValues(xe_l[0], 0.0);
      }
                                                                                                            
      hypre_EndTiming(maxwell_data -> time_index);
      return ierr;
   }
                                                                                                            
   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b= hypre_ParVectorInnerProd(be_l[0], be_l[0]);
      eps = tol*tol;
                                                                                                            
      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_ParVectorSetConstantValues(xe_l[0], 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }
                                                                                                            
         hypre_EndTiming(maxwell_data -> time_index);
         return ierr;
      }
   }

   /*-----------------------------------------------------
    * Do V-cycles:
    * For each index l, "fine" = l, "coarse" = (l-1)
    *   
    *   solution update:
    *      edge_sol= edge_sol + T*node_sol
    *-----------------------------------------------------*/
   for (i = 0; i < max_iter; i++)
   {
     /* compute fine grid residual & nodal rhs. */
      hypre_ParVectorCopy(be_l[0], rese_l[0]);
      hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);
      hypre_ParVectorZeroBCValues(rese_l[0], BdryRanks_l[0], BdryRanksCnts_l[0]);
      hypre_ParCSRMatrixMatvec(1.0, T_transpose, rese_l[0], 0.0, bn_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r= hypre_ParVectorInnerProd(rese_l[0], rese_l[0]);

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

      hypre_ParVectorCopy(bn_l[0], resn_l[0]);
      hypre_ParCSRMatrixMatvec(-1.0, Ann_l[0], xn_l[0], 1.0, resn_l[0]);
      r_dot_r= hypre_ParVectorInnerProd(resn_l[0], resn_l[0]);

      for (level= 0; level<= node_numlevs-2; level++)
      {
         /*-----------------------------------------------
          * Down cycle
          *-----------------------------------------------*/
          for (j= 0; j< npre_relax; j++)
          {
             Solve_err_flag = hypre_BoomerAMGRelaxIF(Ann_l[level],
                                                     bn_l[level],
                                                     nCF_marker_l[level],
                                                     nrelax_type,
                                                     relax_local,
                                                     cycle_param,
                                                     nrelax_weight[level],
                                                     nomega[level],
                                                     NULL,
                                                     xn_l[level],
                                                     nVtemp2_l[level],
                                                     ze);
          }  /*for (j= 0; j< npre_relax; j++) */

         /* compute residuals */
          hypre_ParVectorCopy(bn_l[level], resn_l[level]);
          hypre_ParCSRMatrixMatvec(-1.0, Ann_l[level], xn_l[level], 
                                    1.0, resn_l[level]);

         /* restrict residuals */
          hypre_ParCSRMatrixMatvecT(1.0, RnT_l[level], resn_l[level],
                                    0.0, bn_l[level+1]);

         /* zero off initial guess for the next level */
          hypre_ParVectorSetConstantValues(xn_l[level+1], 0.0);

      }  /* for (level= 0; level<= node_numlevs-2; level++) */
 
      /* coarsest node solve */
      level= node_numlevs-1;
      Solve_err_flag = hypre_BoomerAMGRelaxIF(Ann_l[level],
                                              bn_l[level],
                                              nCF_marker_l[level],
                                              nrelax_type,
                                              relax_local,
                                              cycle_param,
                                              nrelax_weight[level],
                                              nomega[level],
                                              NULL,
                                              xn_l[level],
                                              nVtemp2_l[level],
                                              ze);

     /*---------------------------------------------------------------------
      *  Cycle up the levels.
      *---------------------------------------------------------------------*/
      for (level= (node_numlevs - 2); level>= 1; level--)
      {
          hypre_ParCSRMatrixMatvec(1.0, Pn_l[level], xn_l[level+1], 0.0,
                                   en_l[level]);
          hypre_ParVectorAxpy(1.0, en_l[level], xn_l[level]);

         /* post smooth */
          for (j= 0; j< npost_relax; j++)
          {
             Solve_err_flag = hypre_BoomerAMGRelaxIF(Ann_l[level],
                                                     bn_l[level],
                                                     nCF_marker_l[level],
                                                     nrelax_type,
                                                     relax_local,
                                                     cycle_param,
                                                     nrelax_weight[level],
                                                     nomega[level],
                                                     NULL,
                                                     xn_l[level],
                                                     nVtemp2_l[level],
                                                     ze);
          }
      }   /* for (level= (en_numlevs - 2); level>= 1; level--) */

      /* interpolate error and correct on finest grids */
      hypre_ParCSRMatrixMatvec(1.0, Pn_l[0], xn_l[1], 0.0, en_l[0]);
      hypre_ParVectorAxpy(1.0, en_l[0], xn_l[0]);
                                                                                                              
      for (j= 0; j< npost_relax; j++)
      {
         Solve_err_flag = hypre_BoomerAMGRelaxIF(Ann_l[0],
                                                 bn_l[0],
                                                 nCF_marker_l[0],
                                                 nrelax_type,
                                                 relax_local,
                                                 cycle_param,
                                                 nrelax_weight[0],
                                                 nomega[0],
                                                 NULL,
                                                 xn_l[0],
                                                 nVtemp2_l[0],
                                                 ze);
      }  /* for (j= 0; j< npost_relax; j++) */
      hypre_ParVectorCopy(bn_l[0], resn_l[0]);
      hypre_ParCSRMatrixMatvec(-1.0, Ann_l[0], xn_l[0], 1.0, resn_l[0]);

      /* add the gradient solution component to xe_l[0] */
      hypre_ParCSRMatrixMatvec(1.0, Tgrad, xn_l[0], 1.0, xe_l[0]);

      hypre_ParVectorCopy(be_l[0], rese_l[0]);
      hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);
      r_dot_r= hypre_ParVectorInnerProd(rese_l[0], rese_l[0]);

      for (level= 0; level<= edge_numlevs-2; level++)
      {
         /*-----------------------------------------------
          * Down cycle
          *-----------------------------------------------*/
          for (j= 0; j< npre_relax; j++)
          {
             Solve_err_flag = hypre_BoomerAMGRelaxIF(Aee_l[level],
                                                     be_l[level],
                                                     eCF_marker_l[level],
                                                     erelax_type,
                                                     relax_local,
                                                     cycle_param,
                                                     erelax_weight[level],
                                                     eomega[level],
                                                     NULL,
                                                     xe_l[level],
                                                     eVtemp2_l[level], 
                                                     ze);
          }  /*for (j= 0; j< npre_relax; j++) */
                                                                                                              
         /* compute residuals */
          hypre_ParVectorCopy(be_l[level], rese_l[level]);
          hypre_ParCSRMatrixMatvec(-1.0, Aee_l[level], xe_l[level],
                                    1.0, rese_l[level]);

         /* restrict residuals */
          hypre_ParCSRMatrixMatvecT(1.0,
             (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ReT_l[level]),
                                    rese_l[level], 0.0, be_l[level+1]);
          hypre_ParVectorZeroBCValues(be_l[level+1], BdryRanks_l[level+1],
                                      BdryRanksCnts_l[level+1]);

         /* zero off initial guess for the next level */
          hypre_ParVectorSetConstantValues(xe_l[level+1], 0.0);
                                                                                                              
      }  /* for (level= 1; level<= edge_numlevels-2; level++) */
                                                                                                              
      /* coarsest edge solve */
      level= edge_numlevs-1;
      for (j= 0; j< npre_relax; j++)
      {
         Solve_err_flag = hypre_BoomerAMGRelaxIF(Aee_l[level],
                                                 be_l[level],
                                                 eCF_marker_l[level],
                                                 erelax_type,
                                                 relax_local,
                                                 cycle_param,
                                                 erelax_weight[level],
                                                 eomega[level],
                                                 NULL,
                                                 xe_l[level],
                                                 eVtemp2_l[level], 
                                                 ze);
      }

     /*---------------------------------------------------------------------
      *  Up cycle. 
      *---------------------------------------------------------------------*/
      for (level= (edge_numlevs - 2); level>= 1; level--)
      {
         hypre_ParCSRMatrixMatvec(1.0, 
           (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[level]), 
                                  xe_l[level+1], 0.0, ee_l[level]);
         hypre_ParVectorZeroBCValues(ee_l[level], BdryRanks_l[level],
                                     BdryRanksCnts_l[level]);
         hypre_ParVectorAxpy(1.0, ee_l[level], xe_l[level]);

         /* post smooth */
         for (j= 0; j< npost_relax; j++)
         {
            Solve_err_flag = hypre_BoomerAMGRelaxIF(Aee_l[level],
                                                    be_l[level],
                                                    eCF_marker_l[level],
                                                    erelax_type,
                                                    relax_local,
                                                    cycle_param,
                                                    erelax_weight[level],
                                                    eomega[level],
                                                    NULL,
                                                    xe_l[level],
                                                    eVtemp2_l[level], 
                                                    ze);
         }

      }  /* for (level= (edge_numlevs - 2); level>= 1; level--) */

      /* interpolate error and correct on finest grids */
      hypre_ParCSRMatrixMatvec(1.0, 
        (hypre_ParCSRMatrix *) hypre_IJMatrixObject(Pe_l[0]), 
                               xe_l[1], 0.0, ee_l[0]);
      hypre_ParVectorZeroBCValues(ee_l[0], BdryRanks_l[0],
                                  BdryRanksCnts_l[0]);
      hypre_ParVectorAxpy(1.0, ee_l[0], xe_l[0]);

      for (j= 0; j< npost_relax; j++)
      {
         Solve_err_flag = hypre_BoomerAMGRelaxIF(Aee_l[0],
                                                 be_l[0],
                                                 eCF_marker_l[0],
                                                 erelax_type,
                                                 relax_local,
                                                 cycle_param,
                                                 erelax_weight[0],
                                                 eomega[0],
                                                 NULL,
                                                 xe_l[0],
                                                 eVtemp2_l[0],
                                                 ze);
      }  /* for (j= 0; j< npost_relax; j++) */

      e_dot_e= hypre_ParVectorInnerProd(ee_l[0], ee_l[0]);
      x_dot_x= hypre_ParVectorInnerProd(xe_l[0], xe_l[0]);

      hypre_ParVectorCopy(be_l[0], rese_l[0]);
      hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);

      (maxwell_data -> num_iterations) = (i + 1);
   }

   hypre_EndTiming(maxwell_data -> time_index);


   if (ze)
      hypre_ParVectorDestroy(ze);

   return ierr;
}

