/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.35 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "headers.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGCycle( void              *amg_vdata, 
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = amg_vdata;

   MPI_Comm comm;
   HYPRE_Solver *smoother;
   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   hypre_ParVector    *Utemp;
   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Aux_U;
   hypre_ParVector    *Aux_F;

   hypre_ParCSRBlockMatrix    **A_block_array;
   hypre_ParCSRBlockMatrix    **P_block_array;
   hypre_ParCSRBlockMatrix    **R_block_array;

   double   *Ztemp_data;
   double   *Ptemp_data;
   HYPRE_Int     **CF_marker_array;
   /* HYPRE_Int     **unknown_map_array;
   HYPRE_Int     **point_map_array;
   HYPRE_Int     **v_at_point_array; */

   double    cycle_op_count;   
   HYPRE_Int       cycle_type;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;

   double   *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;   
   HYPRE_Int      *grid_relax_type;   
   HYPRE_Int     **grid_relax_points;  

   HYPRE_Int     block_mode;
   
   double  *max_eig_est;
   double  *min_eig_est;
   HYPRE_Int      cheby_order;
   double   cheby_fraction;

 /* Local variables  */ 
   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       i, j, jj;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       cg_num_sweep = 1;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points;
   HYPRE_Int       relax_order;
   HYPRE_Int       relax_local;
   HYPRE_Int       old_version = 0;
   double   *relax_weight;
   double   *omega;
   double    alfa, beta, gammaold;
   double    gamma = 1.0;
   HYPRE_Int       local_size;
/*   HYPRE_Int      *smooth_option; */
   HYPRE_Int       smooth_type;
   HYPRE_Int       smooth_num_levels;
   HYPRE_Int       num_threads;

   double    alpha;
   double  **l1_norms = NULL;
   double   *l1_norms_level;

   HYPRE_Int seq_cg = 0;

#if 0
   double   *D_mat;
   double   *S_vec;
#endif
   
   /* Acquire data and allocate storage */

   num_threads = hypre_NumThreads();

   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);

   A_block_array     = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array     = hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array     = hypre_ParAMGDataRBlockArray(amg_data);
   block_mode        = hypre_ParAMGDataBlockMode(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data); 
   omega               = hypre_ParAMGDataOmega(amg_data); 
   smooth_type         = hypre_ParAMGDataSmoothType(amg_data); 
   smooth_num_levels   = hypre_ParAMGDataSmoothNumLevels(amg_data); 
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data); 
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */

   max_eig_est = hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = hypre_ParAMGDataMinEigEst(amg_data);
   cheby_order = hypre_ParAMGDataChebyOrder(amg_data);
   cheby_fraction = hypre_ParAMGDataChebyFraction(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int, num_levels);

   if (hypre_ParAMGDataACoarse(amg_data)) seq_cg = 1;

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) old_version = 1;

   num_coeffs = hypre_CTAlloc(double, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);

   if (block_mode)
   {
      for (j = 1; j < num_levels; j++)
         num_coeffs[j] = hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
      
   }
   else 
   {
       for (j = 1; j < num_levels; j++)
         num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
   }
   
   
   /*---------------------------------------------------------------------
    *    Initialize cycling control counter
    *
    *     Cycling is controlled using a level counter: lev_counter[k]
    *     
    *     Each time relaxation is performed on level k, the
    *     counter is decremented by 1. If the counter is then
    *     negative, we go to the next finer level. If non-
    *     negative, we go to the next coarser level. The
    *     following actions control cycling:
    *     
    *     a. lev_counter[0] is initialized to 1.
    *     b. lev_counter[k] is initialized to cycle_type for k>0.
    *     
    *     c. During cycling, when going down to level k, lev_counter[k]
    *        is set to the max of (lev_counter[k],cycle_type)
    *---------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k) 
   {
      lev_counter[k] = cycle_type;
   }

   level = 0;
   cycle_param = 1;

   smoother = hypre_ParAMGDataSmoother(amg_data);

   if (smooth_num_levels > 0)
   {
      if (smooth_type == 7 || smooth_type == 8
          || smooth_type == 17 || smooth_type == 18
          || smooth_type == 9 || smooth_type == 19)
      {
         Utemp = hypre_ParVectorCreate(comm,hypre_ParVectorGlobalSize(Vtemp),
                        hypre_ParVectorPartitioning(Vtemp));
         hypre_ParVectorOwnsPartitioning(Utemp) = 0;
         hypre_ParVectorInitialize(Utemp);
      }
   }
   
  
   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/
  
   while (Not_Finished)
   {
      if (num_levels > 1) 
      {
        local_size 
            = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
        hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)) = local_size;
        if (smooth_num_levels <= level)
	{
           cg_num_sweep = 1;
           num_sweep = num_grid_sweeps[cycle_param];
           Aux_U = U_array[level];
           Aux_F = F_array[level];
	}
	else if (smooth_type > 9)
	{
           hypre_VectorSize(hypre_ParVectorLocalVector(Ztemp)) = local_size;
           hypre_VectorSize(hypre_ParVectorLocalVector(Rtemp)) = local_size;
           hypre_VectorSize(hypre_ParVectorLocalVector(Ptemp)) = local_size;
           Ztemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));
           Ptemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ptemp));
           hypre_ParVectorSetConstantValues(Ztemp,0);
           hypre_ParVectorCopy(F_array[level],Rtemp);
           alpha = -1.0;
           beta = 1.0;
           hypre_ParCSRMatrixMatvec(alpha, A_array[level], 
                                U_array[level], beta, Rtemp);
	   cg_num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
           num_sweep = num_grid_sweeps[cycle_param];
           Aux_U = Ztemp;
           Aux_F = Rtemp;
	}
	else 
	{
           cg_num_sweep = 1;
	   num_sweep = hypre_ParAMGDataSmoothNumSweeps(amg_data);
           Aux_U = U_array[level];
           Aux_F = F_array[level];
	}
        relax_type = grid_relax_type[cycle_param];
      }
      else /* AB: 4/08: removed the max_levels > 1 check - should do this when max-levels = 1 also */
      {
        /* If no coarsening occurred, apply a simple smoother once */
        Aux_U = U_array[level];
        Aux_F = F_array[level];
        num_sweep = 1;
        /* TK: Use the user relax type (instead of 0) to allow for setting a
           convergent smoother (e.g. in the solution of singular problems). */
        relax_type = hypre_ParAMGDataUserRelaxType(amg_data);
      }

      if (l1_norms != NULL)
         l1_norms_level = l1_norms[level];
      else
         l1_norms_level = NULL;

      if (cycle_param == 3 && seq_cg)
      {
         hypre_seqAMGCycle(amg_data, level, F_array, U_array);
      }
      else
      {
         
        /*------------------------------------------------------------------
         * Do the relaxation num_sweep times
         *-----------------------------------------------------------------*/
         for (jj = 0; jj < cg_num_sweep; jj++)
         {
	   if (smooth_num_levels > level && smooth_type > 9)
              hypre_ParVectorSetConstantValues(Aux_U,0);

           for (j = 0; j < num_sweep; j++)
           {
              if (num_levels == 1 && max_levels > 1)
              {
                 relax_points = 0;
                 relax_local = 0;
              }
              else
              {
                 if (old_version)
		    relax_points = grid_relax_points[cycle_param][j];
                 relax_local = relax_order;
              }

              /*-----------------------------------------------
               * VERY sloppy approximation to cycle complexity
               *-----------------------------------------------*/
              if (old_version && level < num_levels -1)
              {
                 switch (relax_points)
                 {
                    case 1:
                    cycle_op_count += num_coeffs[level+1];
                    break;
  
                    case -1: 
                    cycle_op_count += (num_coeffs[level]-num_coeffs[level+1]); 
                    break;
                 }
              }
	      else
              {
                 cycle_op_count += num_coeffs[level]; 
              }
              /*-----------------------------------------------
                Choose Smoother
                -----------------------------------------------*/

              if (smooth_num_levels > level && 
			(smooth_type == 7 || smooth_type == 8 ||
			smooth_type == 9 || smooth_type == 19 ||
			smooth_type == 17 || smooth_type == 18))
              {
                 hypre_VectorSize(hypre_ParVectorLocalVector(Utemp)) = local_size;
                 hypre_ParVectorCopy(Aux_F,Vtemp);
                 alpha = -1.0;
                 beta = 1.0;
                 hypre_ParCSRMatrixMatvec(alpha, A_array[level], 
                                U_array[level], beta, Vtemp);
                 if (smooth_type == 8 || smooth_type == 18)
                    HYPRE_ParCSRParaSailsSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Vtemp,
                                 (HYPRE_ParVector) Utemp);
                 else if (smooth_type == 7 || smooth_type == 17)
                    HYPRE_ParCSRPilutSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Vtemp,
                                 (HYPRE_ParVector) Utemp);
                 else if (smooth_type == 9 || smooth_type == 19)
                    HYPRE_EuclidSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Vtemp,
                                 (HYPRE_ParVector) Utemp);
                 hypre_ParVectorAxpy(relax_weight[level],Utemp,Aux_U);
	      }
              else if (smooth_num_levels > level &&
			(smooth_type == 6 || smooth_type == 16))
              {
                 HYPRE_SchwarzSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Aux_F,
                                  (HYPRE_ParVector) Aux_U);
              }
              else if (relax_type == 18)
              {   /* L1 - Jacobi*/
                 if (relax_order == 1 && cycle_type < 3)
                 {
                    /* need to do CF - so can't use the AMS one */
                    HYPRE_Int i;
                    HYPRE_Int loc_relax_points[2];
                    if (cycle_type < 2)
                    {
                       loc_relax_points[0] = 1;
                       loc_relax_points[1] = -1;
                    }
                    else
                    {
                       loc_relax_points[0] = -1;
                       loc_relax_points[1] = 1;
                    }
                    for (i=0; i < 2; i++)
                       hypre_ParCSRRelax_L1_Jacobi(A_array[level],
                                                 Aux_F,
                                                 CF_marker_array[level],
                                                 loc_relax_points[i],
                                                 relax_weight[level],
                                                 l1_norms[level],
                                                 Aux_U,
                                                 Vtemp);
                 }
                 else /* not CF - so use through AMS */
                 {
                    if (num_threads == 1)
                       hypre_ParCSRRelax(A_array[level], 
                                       Aux_F,
                                       1,
                                       1,
                                       l1_norms_level,
                                       relax_weight[level],
                                       omega[level],0,0,0,0,
                                       Aux_U,
                                       Vtemp, 
                                       Ztemp);

                    else
                       hypre_ParCSRRelaxThreads(A_array[level], 
                                              Aux_F,
                                              1,
                                              1,
                                              l1_norms_level,
                                              relax_weight[level],
                                              omega[level],
                                              Aux_U,
                                              Vtemp,
                                              Ztemp);
                 }
              }
              else if (relax_type == 15)
              {  /* CG */
                 if (j ==0) /* do num sweep iterations of CG */
                    hypre_ParCSRRelax_CG( smoother[level],
                                        A_array[level], 
                                        Aux_F,      
                                        Aux_U,
                                        num_sweep);
              }
              else if (relax_type == 16)
              { /* scaled Chebyshev */
                 HYPRE_Int scale = 1;
                 HYPRE_Int variant = 0;
                 hypre_ParCSRRelax_Cheby(A_array[level], 
                                       Aux_F,
                                       max_eig_est[level],     
                                       min_eig_est[level],     
                                       cheby_fraction, cheby_order, scale,
                                       variant, Aux_U, Vtemp, Ztemp );
              }
              else if (relax_type ==17)
              {
                 hypre_BoomerAMGRelax_FCFJacobi(A_array[level], 
                                              Aux_F,
                                              CF_marker_array[level],
                                              relax_weight[level],
                                              Aux_U,
                                              Vtemp);
              }
	      else if (old_version)
	      {
                 Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], 
                                                     Aux_F,
                                                     CF_marker_array[level],
                                                     relax_type, relax_points,
                                                     relax_weight[level],
                                                     omega[level],
                                                     l1_norms_level,
                                                     Aux_U,
                                                     Vtemp, 
                                                     Ztemp);
	      }
	      else 
	      {
                 /* smoother than can have CF ordering */
                 if (block_mode)
                 {
                     Solve_err_flag = hypre_BoomerAMGBlockRelaxIF(A_block_array[level], 
                                                                  Aux_F,
                                                                  CF_marker_array[level],
                                                                  relax_type,
                                                                  relax_local,
                                                                  cycle_param,
                                                                  relax_weight[level],
                                                                  omega[level],
                                                                  Aux_U,
                                                                  Vtemp);
                 }
                 else
                 {
                    Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level], 
                                                          Aux_F,
                                                          CF_marker_array[level],
                                                          relax_type,
                                                          relax_local,
                                                          cycle_param,
                                                          relax_weight[level],
                                                          omega[level],
                                                          l1_norms_level,
                                                          Aux_U,
                                                          Vtemp, 
                                                          Ztemp);
                 }
	      }
 
              if (Solve_err_flag != 0)
                 return(Solve_err_flag);
           }
           if  (smooth_num_levels > level && smooth_type > 9)
           {
              gammaold = gamma;
              gamma = hypre_ParVectorInnerProd(Rtemp,Ztemp);
              if (jj == 0)
                 hypre_ParVectorCopy(Ztemp,Ptemp);
              else
              {
                 beta = gamma/gammaold;
                 for (i=0; i < local_size; i++)
		    Ptemp_data[i] = Ztemp_data[i] + beta*Ptemp_data[i];
              }
              hypre_ParCSRMatrixMatvec(1.0,A_array[level],Ptemp,0.0,Vtemp);
              alfa = gamma /hypre_ParVectorInnerProd(Ptemp,Vtemp);
              hypre_ParVectorAxpy(alfa,Ptemp,U_array[level]);
              hypre_ParVectorAxpy(-alfa,Vtemp,Rtemp);
           }
        }
      }

      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {
                               
         /*---------------------------------------------------------------
          * Visit coarser level next.  
 	  * Compute residual using hypre_ParCSRMatrixMatvec.
          * Perform restriction using hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0); 
          
         hypre_ParVectorCopy(F_array[fine_grid],Vtemp);
         alpha = -1.0;
         beta = 1.0;

         if (block_mode)
         {
            hypre_ParCSRBlockMatrixMatvec(alpha, A_block_array[fine_grid], U_array[fine_grid],
                                          beta, Vtemp);
         }
         else 
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                                     beta, Vtemp);
         }

         alpha = 1.0;
         beta = 0.0;

         if (block_mode)
         {
            hypre_ParCSRBlockMatrixMatvecT(alpha,R_block_array[fine_grid],Vtemp,
                                      beta,F_array[coarse_grid]);
         }
         else
         {
            hypre_ParCSRMatrixMatvecT(alpha,R_array[fine_grid],Vtemp,
                                      beta,F_array[coarse_grid]);
         }

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level],cycle_type);
         cycle_param = 1;
         if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;
         if (block_mode)
         {
            hypre_ParCSRBlockMatrixMatvec(alpha, P_block_array[fine_grid], 
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);   
         }
         else 
         {
            hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid], 
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);            
         }
         
         --level;
         cycle_param = 2;
      }
      else
      {
         Not_Finished = 0;
      }
   }

   hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter);
   hypre_TFree(num_coeffs);
   if (smooth_num_levels > 0)
   {
     if (smooth_type == 7 || smooth_type == 8 || smooth_type == 9 || 
	smooth_type == 17 || smooth_type == 18 || smooth_type == 19 )
        hypre_ParVectorDestroy(Utemp);
   }
   return(Solve_err_flag);
}
