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
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

int
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

   double   *Ztemp_data;
   double   *Ptemp_data;
   int     **CF_marker_array;
   /* int     **unknown_map_array;
   int     **point_map_array;
   int     **v_at_point_array; */

   int       cycle_op_count;   
   int       cycle_type;
   int       num_levels;
   int       max_levels;

   int      *num_coeffs;
   int      *num_grid_sweeps;   
   int      *grid_relax_type;   
   int     **grid_relax_points;  
 /* Local variables  */ 
   int      *lev_counter;
   int       Solve_err_flag;
   int       k;
   int       i, j, jj;
   int       level;
   int       cycle_param;
   int       coarse_grid;
   int       fine_grid;
   int       Not_Finished;
   int       num_sweep;
   int       cg_num_sweep = 1;
   int       relax_type;
   int       relax_points;
   int       relax_order;
   int       relax_local;
   int       old_version = 0;
   double   *relax_weight;
   double   *omega;
   double    alfa, beta, gammaold;
   double    gamma = 1.0;
   int       local_size;
/*   int      *smooth_option; */
   int       smooth_type;
   int       smooth_num_levels;

   double    alpha;

#if 0
   double   *D_mat;
   double   *S_vec;
#endif
   
   /* Acquire data and allocate storage */

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

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data); 
   omega               = hypre_ParAMGDataOmega(amg_data); 
   smooth_type         = hypre_ParAMGDataSmoothType(amg_data); 
   smooth_num_levels   = hypre_ParAMGDataSmoothNumLevels(amg_data); 
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(int, num_levels);

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) old_version = 1;

   num_coeffs = hypre_CTAlloc(int, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);

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

   if (smooth_num_levels > 0)
   {
      smoother = hypre_ParAMGDataSmoother(amg_data);
      if (smooth_type == 7 || smooth_type == 8
          || smooth_type == 17 || smooth_type == 18)
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
      else if (max_levels > 1)
      {
        /* If no coarsening occurred, apply a simple smoother once */
        Aux_U = U_array[level];
        Aux_F = F_array[level];
        num_sweep = 1;
        relax_type = 0;
      }

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

            if (smooth_num_levels > level && 
			(smooth_type == 7 || smooth_type == 8 ||
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
               hypre_ParVectorAxpy(relax_weight[level],Utemp,Aux_U);
	    }
            else if (smooth_num_levels > level &&
			(smooth_type == 6 || smooth_type == 16))
               HYPRE_SchwarzSolve(smoother[level],
                                 (HYPRE_ParCSRMatrix) A_array[level],
                                 (HYPRE_ParVector) Aux_F,
                                 (HYPRE_ParVector) Aux_U);
	    else if (old_version)
	    {
               Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], 
                                            Aux_F,
                                            CF_marker_array[level],
                                            relax_type,
                                            relax_points,
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
                                            Aux_U,
                                            Vtemp);
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
         hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                         beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_ParCSRMatrixMatvecT(alpha,R_array[fine_grid],Vtemp,
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
          * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid], 
			 U_array[coarse_grid],
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
   if (smooth_type == 7 || smooth_type == 8 || smooth_type == 17
		|| smooth_type == 18)
        hypre_ParVectorDestroy(Utemp);

   return(Solve_err_flag);
}
