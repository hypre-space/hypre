/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "headers.h"
#include "par_amg.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGCycle
 *--------------------------------------------------------------------------*/

int
hypre_ParAMGCycle( void              *amg_vdata, 
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   hypre_ParVector    *Vtemp;

   int     **CF_marker_array;
   int     **unknown_map_array;
   int     **point_map_array;
   int     **v_at_point_array;

   int       cycle_op_count;   
   int       cycle_type;
   int       num_levels;
   int       num_unknowns;

   int      *num_coeffs;
   int      *num_grid_sweeps;   
   int      *grid_relax_type;   
   int     **grid_relax_points;  
 
   /* Local variables  */

   int      *lev_counter;
   int       Solve_err_flag;
   int       k;
   int       j;
   int       level;
   int       cycle_param;
   int       coarse_grid;
   int       fine_grid;
   int       Not_Finished;
   int       num_sweep;
   int       relax_type;
   int       relax_points;
   double    relax_weight;

   double    alpha;
   double    beta;
#if 0
   double   *D_mat;
   double   *S_vec;
#endif
   
   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   unknown_map_array = hypre_ParAMGDataUnknownMapArray(amg_data);
   point_map_array   = hypre_ParAMGDataPointMapArray(amg_data);
   v_at_point_array  = hypre_ParAMGDataVatPointArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);
   num_unknowns      =  hypre_ParCSRMatrixNumRows(A_array[0]);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data); 

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(int, num_levels);

   /* Initialize */

   Solve_err_flag = 0;

   num_coeffs = hypre_CTAlloc(int, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixNumNonzeros(A_array[0]);

   for (j = 1; j < num_levels; j++)
      num_coeffs[j] = hypre_ParCSRMatrixNumNonzeros(A_array[j]);

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


         Solve_err_flag = hypre_ParAMGRelax(A_array[level], 
                                            F_array[level],
                                            CF_marker_array[level],
                                            relax_type,
                                            relax_points,
                                            relax_weight,
                                            U_array[level],
                                            Vtemp);

 
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
          * Visit coarser level next.  Compute residual using hypre_ParMatvec.
          * Perform restriction using hypre_ParMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_SetParVectorConstantValues(U_array[coarse_grid], 0.0);
          
         hypre_CopyParVector(F_array[fine_grid],Vtemp);
         alpha = -1.0;
         beta = 1.0;
         hypre_ParMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                         beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_ParMatvecT(alpha,R_array[fine_grid],Vtemp,
                          beta,F_array[coarse_grid]);

         ++level;
         lev_counter[level] = max(lev_counter[level],cycle_type);
         cycle_param = 1;
         if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
                            
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_ParMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         hypre_ParMatvec(alpha, P_array[fine_grid], U_array[coarse_grid],
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

   hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter);
   hypre_TFree(num_coeffs);

   return(Solve_err_flag);
}
