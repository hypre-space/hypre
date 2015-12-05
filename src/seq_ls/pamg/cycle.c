/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * AMG cycling routine
 *
 *****************************************************************************/

#include "headers.h"
#include "amg.h"

/*--------------------------------------------------------------------------
 * hypre_AMGCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGCycle( void           *amg_vdata, 
                hypre_Vector  **F_array,
                hypre_Vector  **U_array   )
{
   hypre_AMGData *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_CSRMatrix    **A_array;
   hypre_BCSRMatrix    **B_array;
   hypre_CSRMatrix    **P_array;
   hypre_Vector    *Vtemp;

   HYPRE_Int     **CF_marker_array;
/* HYPRE_Int     **dof_func_array;
   HYPRE_Int     **dof_point_array;
   HYPRE_Int     **point_dof_map_array; */

   HYPRE_Int       cycle_op_count;   
   HYPRE_Int       cycle_type;
   HYPRE_Int       num_levels;
/* HYPRE_Int       num_functions; */

   HYPRE_Int      *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;   
   HYPRE_Int      *grid_relax_type;   
   HYPRE_Int     **grid_relax_points;  
 
   /* Local variables  */

   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       j;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points;
   double   *relax_weight;
   HYPRE_Int use_block_flag;

   double    alpha;
   double    beta;
#if 0
   double   *D_mat;
   double   *S_vec;
#endif
   
   /* Acquire data and allocate storage */

   A_array           = hypre_AMGDataAArray(amg_data);
   B_array           = hypre_AMGDataBArray(amg_data);
   P_array           = hypre_AMGDataPArray(amg_data);
   CF_marker_array   = hypre_AMGDataCFMarkerArray(amg_data);
   /* dof_func_array    = hypre_AMGDataDofFuncArray(amg_data); */
   /* dof_point_array   = hypre_AMGDataDofPointArray(amg_data); */
   /* point_dof_map_array = hypre_AMGDataPointDofMapArray(amg_data); */
   Vtemp             = hypre_AMGDataVtemp(amg_data);
   num_levels        = hypre_AMGDataNumLevels(amg_data);
   cycle_type        = hypre_AMGDataCycleType(amg_data);
   /*  num_functions     = hypre_CSRMatrixFunctions(A_array[0]);  ??? */

   num_grid_sweeps     = hypre_AMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_AMGDataGridRelaxType(amg_data);
   grid_relax_points   =  hypre_AMGDataGridRelaxPoints(amg_data); 
   relax_weight        =  hypre_AMGDataRelaxWeight(amg_data); 
   use_block_flag = hypre_AMGDataUseBlockFlag(amg_data);

   cycle_op_count = hypre_AMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int, num_levels);

   /* Initialize */

   Solve_err_flag = 0;

   num_coeffs = hypre_CTAlloc(HYPRE_Int, num_levels);
   num_coeffs[0]    = hypre_CSRMatrixNumNonzeros(A_array[0]);

   for (j = 1; j < num_levels; j++)
      num_coeffs[j] = hypre_CSRMatrixNumNonzeros(A_array[j]);

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
   cycle_param = 0;

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/
  
   while (Not_Finished)
   {
      num_sweep = num_grid_sweeps[cycle_param];
      relax_type = grid_relax_type[cycle_param];

      /*------------------------------------------------------------------
       * Do the relaxation num_sweep times
       *-----------------------------------------------------------------*/

      if (hypre_AMGDataSchwarzOption(amg_data)[level] > -1) 
	 /* num_sweep = hypre_AMGDataSchwarzOption(amg_data)[level];*/
	 num_sweep = 1;
      for (j = 0; j < num_sweep; j++)
      {
         relax_points =   grid_relax_points[cycle_param][j];

         /*-----------------------------------------------
          * VERY sloppy approximation to cycle complexity
          *-----------------------------------------------*/

         if (level < num_levels -1)
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

	 if (hypre_AMGDataSchwarzOption(amg_data)[level] > -1)
	 {
            Solve_err_flag = hypre_SchwarzSolve(A_array[level],
                              F_array[level],
                              hypre_AMGDataNumDomains(amg_data)[level],
                              hypre_AMGDataIDomainDof(amg_data)[level],
                              hypre_AMGDataJDomainDof(amg_data)[level],
                              hypre_AMGDataDomainMatrixInverse(amg_data)[level],
                              U_array[level],
                              Vtemp);
	 }
	 else if (use_block_flag && relax_type != 9) {
	   Solve_err_flag = hypre_BCSRMatrixRelax(B_array[level],
						  F_array[level],
						  CF_marker_array[level],
						  relax_points,
						  U_array[level]);
	 }
	 else
	 {
            Solve_err_flag = hypre_AMGRelax(A_array[level], 
                                         F_array[level],
                                         CF_marker_array[level],
                                         relax_type,
                                         relax_points,
                                         relax_weight[level],
                                         U_array[level],
                                         Vtemp);
	 }
 
         if (Solve_err_flag != 0)
            return(Solve_err_flag);
      }


      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {
                               
         /*---------------------------------------------------------------
          * Visit coarser level next.  Compute residual using hypre_CSRMatrixMatvec.
          * Perform restriction using hypre_CSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_SeqVectorSetConstantValues(U_array[coarse_grid], 0.0);
          
         hypre_SeqVectorCopy(F_array[fine_grid],Vtemp);
         alpha = -1.0;
         beta = 1.0;
         hypre_CSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                      beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_CSRMatrixMatvecT(alpha,P_array[fine_grid],Vtemp,
                       beta,F_array[coarse_grid]);

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level],cycle_type);
         cycle_param = 1;
         if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
                            
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_CSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         hypre_CSRMatrixMatvec(alpha, P_array[fine_grid], U_array[coarse_grid],
                      beta, U_array[fine_grid]);            
 
         --level;
         cycle_param = 2;
         if (level == 0) cycle_param = 0;
      }
      else
      {
         Not_Finished = 0;
      }
   }

   hypre_AMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter);
   hypre_TFree(num_coeffs);

   return(Solve_err_flag);
}
