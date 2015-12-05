/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.30 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGHybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{

   double                tol;
   double                a_tol;
   double                cf_tol;
   HYPRE_Int                   dscg_max_its;
   HYPRE_Int                   pcg_max_its;
   HYPRE_Int                   two_norm;
   HYPRE_Int                   stop_crit;
   HYPRE_Int                   rel_change;
   HYPRE_Int                   solver_type;
   HYPRE_Int                   k_dim;

   HYPRE_Int                   pcg_default;              /* boolean */
   HYPRE_Int                 (*pcg_precond_solve)();
   HYPRE_Int                 (*pcg_precond_setup)();
   void                 *pcg_precond;
   void                 *pcg_solver;

   /* log info (always logged) */
   HYPRE_Int                   dscg_num_its;
   HYPRE_Int                   pcg_num_its;
   double                final_rel_res_norm;
   HYPRE_Int                   time_index;

   /* additional information (place-holder currently used to print norms) */
   HYPRE_Int                   logging;
   HYPRE_Int                   print_level; 

   /* info for BoomerAMG */
   double		strong_threshold;
   double		max_row_sum;
   double		trunc_factor;
   HYPRE_Int                  pmax;
   HYPRE_Int                  setup_type;
   HYPRE_Int			max_levels;
   HYPRE_Int			measure_type;
   HYPRE_Int			coarsen_type;
   HYPRE_Int			interp_type;
   HYPRE_Int			cycle_type;
   HYPRE_Int		        relax_order;
   HYPRE_Int		        max_coarse_size;
   HYPRE_Int		        seq_threshold;
   HYPRE_Int		       *num_grid_sweeps;
   HYPRE_Int		       *grid_relax_type;
   HYPRE_Int		      **grid_relax_points;
   double	       *relax_weight;
   double	       *omega;
   HYPRE_Int		        num_paths;
   HYPRE_Int		        agg_num_levels;
   HYPRE_Int		        num_functions;
   HYPRE_Int		        nodal;
   HYPRE_Int		       *dof_func;

} hypre_AMGHybridData;

/*--------------------------------------------------------------------------
 * hypre_AMGHybridCreate
 *--------------------------------------------------------------------------*/

void *
hypre_AMGHybridCreate( )
{
   hypre_AMGHybridData *AMGhybrid_data;

   AMGhybrid_data = hypre_CTAlloc(hypre_AMGHybridData, 1);

   (AMGhybrid_data -> time_index)  = hypre_InitializeTiming("AMGHybrid");

   /* set defaults */
   (AMGhybrid_data -> tol)               = 1.0e-06;
   (AMGhybrid_data -> a_tol)             = 0.0;
   (AMGhybrid_data -> cf_tol)            = 0.90;
   (AMGhybrid_data -> dscg_max_its)      = 1000;
   (AMGhybrid_data -> pcg_max_its)       = 200;
   (AMGhybrid_data -> two_norm)          = 0;
   (AMGhybrid_data -> stop_crit)         = 0;
   (AMGhybrid_data -> rel_change)        = 0;
   (AMGhybrid_data -> pcg_default)       = 1;
   (AMGhybrid_data -> solver_type)       = 1;
   (AMGhybrid_data -> pcg_precond_solve) = NULL;
   (AMGhybrid_data -> pcg_precond_setup) = NULL;
   (AMGhybrid_data -> pcg_precond)       = NULL;
   (AMGhybrid_data -> pcg_solver)       = NULL;
   
   /* initialize */ 
   (AMGhybrid_data -> dscg_num_its)      = 0; 
   (AMGhybrid_data -> pcg_num_its)       = 0; 
   (AMGhybrid_data -> logging)           = 0; 
   (AMGhybrid_data -> print_level)       = 0; 
   (AMGhybrid_data -> k_dim)             = 5; 

   /* BoomerAMG info */
   (AMGhybrid_data -> setup_type)       = 1;
   (AMGhybrid_data -> strong_threshold)  = 0.25;
   (AMGhybrid_data -> max_row_sum)  = 0.9;
   (AMGhybrid_data -> trunc_factor)  = 0.0;
   (AMGhybrid_data -> pmax)  = 0;
   (AMGhybrid_data -> max_levels)  = 25;
   (AMGhybrid_data -> measure_type)  = 0;
   (AMGhybrid_data -> coarsen_type)  = 6;
   (AMGhybrid_data -> interp_type)  = 0;
   (AMGhybrid_data -> cycle_type)  = 1;
   (AMGhybrid_data -> relax_order)  = 1;
   (AMGhybrid_data -> max_coarse_size)  = 9;
   (AMGhybrid_data -> seq_threshold)  = 0;
   (AMGhybrid_data -> num_grid_sweeps)  = NULL;
   (AMGhybrid_data -> grid_relax_type)  = NULL;
   (AMGhybrid_data -> grid_relax_points)  = NULL;
   (AMGhybrid_data -> relax_weight)  = NULL;
   (AMGhybrid_data -> omega)  = NULL;
   (AMGhybrid_data -> agg_num_levels)  = 0;
   (AMGhybrid_data -> num_paths)  = 1;
   (AMGhybrid_data -> num_functions)  = 1;
   (AMGhybrid_data -> nodal)  = 0;
   (AMGhybrid_data -> dof_func)  = NULL;

   return (void *) AMGhybrid_data; 
}

/*-------------------------------------------------------------------------- *
  hypre_AMGHybridDestroy 
*--------------------------------------------------------------------------*/ 

HYPRE_Int
hypre_AMGHybridDestroy( void  *AMGhybrid_vdata )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int i;
   HYPRE_Int solver_type = (AMGhybrid_data -> solver_type);
   /*HYPRE_Int pcg_default = (AMGhybrid_data -> pcg_default);*/
   void *pcg_solver = (AMGhybrid_data -> pcg_solver);
   void *pcg_precond = (AMGhybrid_data -> pcg_precond);

   if (pcg_precond) hypre_BoomerAMGDestroy(pcg_precond);
   if (solver_type == 1) hypre_PCGDestroy(pcg_solver);
   if (solver_type == 2) hypre_GMRESDestroy(pcg_solver);
   if (solver_type == 3) hypre_BiCGSTABDestroy(pcg_solver);

   if (AMGhybrid_data -> num_grid_sweeps)  
   {
      hypre_TFree( (AMGhybrid_data -> num_grid_sweeps) );
      (AMGhybrid_data -> num_grid_sweeps) = NULL;
   }
   if (AMGhybrid_data -> grid_relax_type)  
   {
      hypre_TFree( (AMGhybrid_data -> grid_relax_type) );
      (AMGhybrid_data -> grid_relax_type) = NULL;
   }
   if (AMGhybrid_data -> grid_relax_points)  
   {
      for (i=0; i < 4; i++)
         hypre_TFree( (AMGhybrid_data -> grid_relax_points)[i] );
      hypre_TFree( (AMGhybrid_data -> grid_relax_points) );
      (AMGhybrid_data -> grid_relax_points) = NULL;
   }
   if (AMGhybrid_data -> relax_weight)  
   {
      hypre_TFree( (AMGhybrid_data -> relax_weight) );
      (AMGhybrid_data -> relax_weight) = NULL;
   }
   if (AMGhybrid_data -> omega)  
   {
      hypre_TFree( (AMGhybrid_data -> omega) );
      (AMGhybrid_data -> omega) = NULL;
   }
   if (AMGhybrid_data -> dof_func)  
   {
      hypre_TFree( (AMGhybrid_data -> dof_func) );
      (AMGhybrid_data -> dof_func) = NULL;
   }
   if (AMGhybrid_data)
   {
      hypre_TFree(AMGhybrid_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetTol( void   *AMGhybrid_vdata,
                    double  tol       )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;

   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (tol < 0 || tol > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   (AMGhybrid_data -> tol) = tol;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetAbsoluteTol( void   *AMGhybrid_vdata,
                    double  a_tol       )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;

   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (a_tol < 0 || a_tol > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   (AMGhybrid_data -> a_tol) = a_tol;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetConvergenceTol( void   *AMGhybrid_vdata,
                               double  cf_tol       )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (cf_tol < 0 || cf_tol > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> cf_tol) = cf_tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetDSCGMaxIter( void   *AMGhybrid_vdata,
                            HYPRE_Int     dscg_max_its )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (dscg_max_its < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> dscg_max_its) = dscg_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetPCGMaxIter( void   *AMGhybrid_vdata,
                           HYPRE_Int     pcg_max_its  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (pcg_max_its < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> pcg_max_its) = pcg_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetSetupType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetSetupType( void   *AMGhybrid_vdata,
                           HYPRE_Int     setup_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> setup_type) = setup_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetSolverType( void   *AMGhybrid_vdata,
                           HYPRE_Int     solver_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> solver_type) = solver_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetKDim( void   *AMGhybrid_vdata,
                           HYPRE_Int     k_dim  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (k_dim < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> k_dim) = k_dim;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetStopCrit( void *AMGhybrid_vdata,
                        HYPRE_Int   stop_crit  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> stop_crit) = stop_crit;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetTwoNorm( void *AMGhybrid_vdata,
                        HYPRE_Int   two_norm  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> two_norm) = two_norm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetRelChange( void *AMGhybrid_vdata,
                          HYPRE_Int   rel_change  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetPrecond( void  *pcg_vdata,
                        HYPRE_Int  (*pcg_precond_solve)(),
                        HYPRE_Int  (*pcg_precond_setup)(),
                        void  *pcg_precond          )
{
   hypre_AMGHybridData *pcg_data = pcg_vdata;
   if (!pcg_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
 
   (pcg_data -> pcg_default)       = 0;
   (pcg_data -> pcg_precond_solve) = pcg_precond_solve;
   (pcg_data -> pcg_precond_setup) = pcg_precond_setup;
   (pcg_data -> pcg_precond)       = pcg_precond;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetLogging( void *AMGhybrid_vdata,
                        HYPRE_Int   logging  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> logging) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetPrintLevel( void *AMGhybrid_vdata,
                        HYPRE_Int   print_level  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> print_level) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetStrongThreshold( void *AMGhybrid_vdata,
                        double strong_threshold)
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (strong_threshold < 0 || strong_threshold > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> strong_threshold) = strong_threshold;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetMaxRowSum( void *AMGhybrid_vdata,
                        double   max_row_sum  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (max_row_sum < 0 || max_row_sum > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> max_row_sum) = max_row_sum;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetTruncFactor( void *AMGhybrid_vdata,
                        double   trunc_factor  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (trunc_factor < 0 || trunc_factor > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> trunc_factor) = trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetPMaxElmts( void   *AMGhybrid_vdata,
                             HYPRE_Int    P_max_elmts )
{

   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

    (AMGhybrid_data -> pmax) = P_max_elmts;

   return hypre_error_flag;
}




/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetMaxLevels( void *AMGhybrid_vdata,
                        HYPRE_Int   max_levels  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (max_levels < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> max_levels) = max_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMeasureType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetMeasureType( void *AMGhybrid_vdata,
                        HYPRE_Int   measure_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> measure_type) = measure_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetCoarsenType( void *AMGhybrid_vdata,
                        HYPRE_Int   coarsen_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> coarsen_type) = coarsen_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetInterpType( void *AMGhybrid_vdata,
                        HYPRE_Int   interp_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (interp_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> interp_type) = interp_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetCycleType( void *AMGhybrid_vdata,
                        HYPRE_Int   cycle_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (cycle_type < 1 || cycle_type > 2)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> cycle_type) = cycle_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetNumSweeps( void *AMGhybrid_vdata,
                        HYPRE_Int   num_sweeps  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                 *num_grid_sweeps;
   HYPRE_Int               i;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> num_grid_sweeps) == NULL)
      (AMGhybrid_data -> num_grid_sweeps) = hypre_CTAlloc(HYPRE_Int,4);
   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   for (i=0; i < 3; i++)
   {
      num_grid_sweeps[i] = num_sweeps;
   }
   num_grid_sweeps[3] = 1;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetCycleNumSweeps( void *AMGhybrid_vdata,
                                  HYPRE_Int   num_sweeps,
                                  HYPRE_Int   k)
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                 *num_grid_sweeps;
   HYPRE_Int               i;

   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   if (num_grid_sweeps == NULL)
   {
      (AMGhybrid_data -> num_grid_sweeps) = hypre_CTAlloc(HYPRE_Int,4);
      num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
      for (i=0; i < 4; i++)
      {
          num_grid_sweeps[i] = 1;
      }
   }
   num_grid_sweeps[k] = num_sweeps;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetRelaxType( void *AMGhybrid_vdata,
                        HYPRE_Int  relax_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int               *grid_relax_type;
   HYPRE_Int               i;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_type) == NULL )
      (AMGhybrid_data -> grid_relax_type) = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   for (i=0; i < 3; i++)
      grid_relax_type[i] = relax_type;
   grid_relax_type[3] = 9;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetCycleRelaxType( void *AMGhybrid_vdata,
                                  HYPRE_Int   relax_type,
                                  HYPRE_Int   k  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                 *grid_relax_type;
   HYPRE_Int                 i;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (k<1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! Relax type not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   if (grid_relax_type == NULL )
   {
      (AMGhybrid_data -> grid_relax_type) = hypre_CTAlloc(HYPRE_Int,4);
      grid_relax_type = (AMGhybrid_data -> grid_relax_type);
      for (i=0; i < 3; i++)
         grid_relax_type[i] = 3;
      grid_relax_type[3] = 9;
   }
   grid_relax_type[k] = relax_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetRelaxOrder( void *AMGhybrid_vdata,
                              HYPRE_Int   relax_order  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> relax_order) = relax_order;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMaxCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetMaxCoarseSize( void *AMGhybrid_vdata,
                        HYPRE_Int   max_coarse_size  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (max_coarse_size < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> max_coarse_size) = max_coarse_size;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetSeqThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetSeqThreshold( void *AMGhybrid_vdata,
                        HYPRE_Int   seq_threshold  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (seq_threshold < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> seq_threshold) = seq_threshold;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetNumGridSweeps( void *AMGhybrid_vdata,
                        HYPRE_Int  *num_grid_sweeps  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!num_grid_sweeps)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> num_grid_sweeps) != NULL)
      hypre_TFree((AMGhybrid_data -> num_grid_sweeps));
   (AMGhybrid_data -> num_grid_sweeps) = num_grid_sweeps;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetGridRelaxType( void *AMGhybrid_vdata,
                        HYPRE_Int  *grid_relax_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_type)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_type) != NULL )
      hypre_TFree((AMGhybrid_data -> grid_relax_type));
   (AMGhybrid_data -> grid_relax_type) = grid_relax_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetGridRelaxPoints( void *AMGhybrid_vdata,
                        HYPRE_Int  **grid_relax_points  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!grid_relax_points)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> grid_relax_points) != NULL )
      hypre_TFree((AMGhybrid_data -> grid_relax_points));
   (AMGhybrid_data -> grid_relax_points) = grid_relax_points;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetRelaxWeight( void *AMGhybrid_vdata,
                        double *relax_weight  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!relax_weight)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> relax_weight) != NULL )
      hypre_TFree((AMGhybrid_data -> relax_weight));
   (AMGhybrid_data -> relax_weight) = relax_weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetOmega( void *AMGhybrid_vdata,
                        double *omega  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!omega)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> omega) != NULL )
      hypre_TFree((AMGhybrid_data -> omega));
   (AMGhybrid_data -> omega) = omega;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetRelaxWt( void *AMGhybrid_vdata,
                        double  relax_wt  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int               i , num_levels;
   double	       *relax_wt_array;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   relax_wt_array = (AMGhybrid_data -> relax_weight);
   if (relax_wt_array == NULL)
   {
      relax_wt_array = hypre_CTAlloc(double,num_levels);
      (AMGhybrid_data -> relax_weight) = relax_wt_array;
   }
   for (i=0; i < num_levels; i++)
      relax_wt_array[i] = relax_wt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetLevelRelaxWt( void   *AMGhybrid_vdata,
                                double  relax_wt,
                                HYPRE_Int     level  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                i , num_levels;
   double	       *relax_wt_array;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   if (level > num_levels-1) 
   {
      hypre_printf (" Warning! Invalid level! Relax weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   relax_wt_array = (AMGhybrid_data -> relax_weight);
   if (relax_wt_array == NULL)
   {
      relax_wt_array = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         relax_wt_array[i] = 1.0;
      (AMGhybrid_data -> relax_weight) = relax_wt_array;
   }
   relax_wt_array[level] = relax_wt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetOuterWt( void *AMGhybrid_vdata,
                        double  outer_wt  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                i , num_levels;
   double	       *outer_wt_array;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   outer_wt_array = (AMGhybrid_data -> omega);
   if (outer_wt_array == NULL)
   {
      outer_wt_array = hypre_CTAlloc(double,num_levels);
      (AMGhybrid_data -> omega) = outer_wt_array;
   }
   for (i=0; i < num_levels; i++)
      outer_wt_array[i] = outer_wt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetLevelOuterWt( void   *AMGhybrid_vdata,
                                double  outer_wt,
                                HYPRE_Int     level  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   HYPRE_Int                i , num_levels;
   double	       *outer_wt_array;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   num_levels = (AMGhybrid_data -> max_levels);
   if (level > num_levels-1) 
   {
      hypre_printf (" Warning! Invalid level! Outer weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   outer_wt_array = (AMGhybrid_data -> omega);
   if (outer_wt_array == NULL)
   {
      outer_wt_array = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         outer_wt_array[i] = 1.0;
      (AMGhybrid_data -> omega) = outer_wt_array;
   }
   outer_wt_array[level] = outer_wt;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNumPaths
 *--------------------------------------------------------------------------*/
                                                                                                                  
HYPRE_Int
hypre_AMGHybridSetNumPaths( void   *AMGhybrid_vdata,
                              HYPRE_Int    num_paths      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_paths < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   (AMGhybrid_data -> num_paths) = num_paths;

   return hypre_error_flag;
}
                                                                                                                  
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetDofFunc
 *--------------------------------------------------------------------------*/
                                                                                                                  
HYPRE_Int
hypre_AMGHybridSetDofFunc( void *AMGhybrid_vdata,
                        HYPRE_Int *dof_func  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!dof_func)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if ((AMGhybrid_data -> dof_func) != NULL )
      hypre_TFree((AMGhybrid_data -> dof_func));
   (AMGhybrid_data -> dof_func) = dof_func;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/
                                                                                                                  
HYPRE_Int
hypre_AMGHybridSetAggNumLevels( void   *AMGhybrid_vdata,
                              HYPRE_Int    agg_num_levels      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (agg_num_levels < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
                                                                                                                  
   (AMGhybrid_data -> agg_num_levels) = agg_num_levels;
                                                                                                                  
   return hypre_error_flag;
}
                                                                                                                  
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNumFunctions
 *--------------------------------------------------------------------------*/
                                                                                                                  
HYPRE_Int
hypre_AMGHybridSetNumFunctions( void   *AMGhybrid_vdata,
                              HYPRE_Int    num_functions      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_functions < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
                                                                                                                  
   (AMGhybrid_data -> num_functions) = num_functions;
                                                                                                                  
   return hypre_error_flag;
}
                                                                                                                  
/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNodal
 *--------------------------------------------------------------------------*/
                                                                                                                  
HYPRE_Int
hypre_AMGHybridSetNodal( void   *AMGhybrid_vdata,
                              HYPRE_Int    nodal      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
                                                                                                                  
   (AMGhybrid_data -> nodal) = nodal;
                                                                                                                  
   return hypre_error_flag;
}
                                                                                                                  
/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridGetNumIterations( void   *AMGhybrid_vdata,
                              HYPRE_Int    *num_its      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_its = (AMGhybrid_data -> dscg_num_its) + (AMGhybrid_data -> pcg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridGetDSCGNumIterations( void   *AMGhybrid_vdata,
                                  HYPRE_Int    *dscg_num_its )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *dscg_num_its = (AMGhybrid_data -> dscg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridGetPCGNumIterations( void   *AMGhybrid_vdata,
                                 HYPRE_Int    *pcg_num_its  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *pcg_num_its = (AMGhybrid_data -> pcg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridGetFinalRelativeResidualNorm( void   *AMGhybrid_vdata,
                                          double *final_rel_res_norm )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *final_rel_res_norm = (AMGhybrid_data -> final_rel_res_norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSetup( void               *AMGhybrid_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x            )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a AMGhybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If sufficient
 * progress is not made, the algorithm switches to preconditioned
 * conjugate gradients with user-specified preconditioner.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGHybridSolve( void               *AMGhybrid_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x            )
{
   hypre_AMGHybridData  *AMGhybrid_data    = AMGhybrid_vdata;

   double             tol;
   double             a_tol;
   double             cf_tol;
   HYPRE_Int                dscg_max_its;
   HYPRE_Int                pcg_max_its;
   HYPRE_Int                two_norm;
   HYPRE_Int                stop_crit;
   HYPRE_Int                rel_change;
   HYPRE_Int                logging;
   HYPRE_Int                print_level;
   HYPRE_Int                setup_type;
   HYPRE_Int                solver_type;
   HYPRE_Int                k_dim;
   /* BoomerAMG info */
   double 	strong_threshold;
   double      	max_row_sum;
   double	trunc_factor;
   HYPRE_Int          pmax;
   HYPRE_Int		max_levels;
   HYPRE_Int		measure_type;
   HYPRE_Int		coarsen_type;
   HYPRE_Int		interp_type;
   HYPRE_Int		cycle_type;
   HYPRE_Int		num_paths;
   HYPRE_Int		agg_num_levels;
   HYPRE_Int		num_functions;
   HYPRE_Int		nodal;
   HYPRE_Int	       *num_grid_sweeps;
   HYPRE_Int	       *grid_relax_type;
   HYPRE_Int	      **grid_relax_points;
   double      *relax_weight;
   double      *omega;
   HYPRE_Int         *dof_func;

   HYPRE_Int	       *boom_ngs;
   HYPRE_Int	       *boom_grt;
   HYPRE_Int         *boom_dof_func;
   HYPRE_Int	      **boom_grp;
   double      *boom_rlxw;
   double      *boom_omega;

   HYPRE_Int                pcg_default;
   HYPRE_Int              (*pcg_precond_solve)();
   HYPRE_Int              (*pcg_precond_setup)();
   void              *pcg_precond;

   void              *pcg_solver;
   hypre_PCGFunctions *pcg_functions;
   hypre_GMRESFunctions *gmres_functions;
   hypre_BiCGSTABFunctions *bicgstab_functions;
                                                                                                                                        
   HYPRE_Int                dscg_num_its=0;
   HYPRE_Int                pcg_num_its=0;
   HYPRE_Int                converged=0;
   HYPRE_Int                num_variables = hypre_VectorSize(hypre_ParVectorLocalVector(b));
   double             res_norm;

   HYPRE_Int                i, j;
   HYPRE_Int		      sol_print_level; /* print_level for solver */
   HYPRE_Int		      pre_print_level; /* print_level for preconditioner */
   HYPRE_Int		      max_coarse_size, seq_threshold; 

   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   /*-----------------------------------------------------------------------
    * Setup diagonal scaled solver
    *-----------------------------------------------------------------------*/

   if (!AMGhybrid_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   tol            = (AMGhybrid_data -> tol);
   a_tol          = (AMGhybrid_data -> a_tol);
   cf_tol         = (AMGhybrid_data -> cf_tol);
   dscg_max_its   = (AMGhybrid_data -> dscg_max_its);
   pcg_max_its    = (AMGhybrid_data -> pcg_max_its);
   two_norm       = (AMGhybrid_data -> two_norm);
   stop_crit      = (AMGhybrid_data -> stop_crit);
   rel_change     = (AMGhybrid_data -> rel_change);
   logging        = (AMGhybrid_data -> logging);
   print_level    = (AMGhybrid_data -> print_level);
   setup_type     = (AMGhybrid_data -> setup_type);
   solver_type    = (AMGhybrid_data -> solver_type);
   k_dim          = (AMGhybrid_data -> k_dim);
   strong_threshold = (AMGhybrid_data -> strong_threshold);
   max_row_sum = (AMGhybrid_data -> max_row_sum);
   trunc_factor = (AMGhybrid_data -> trunc_factor);
   pmax = (AMGhybrid_data -> pmax);
   max_levels = (AMGhybrid_data -> max_levels);
   measure_type = (AMGhybrid_data -> measure_type);
   coarsen_type = (AMGhybrid_data -> coarsen_type);
   interp_type = (AMGhybrid_data -> interp_type);
   cycle_type = (AMGhybrid_data -> cycle_type);
   num_paths = (AMGhybrid_data -> num_paths);
   agg_num_levels = (AMGhybrid_data -> agg_num_levels);
   num_functions = (AMGhybrid_data -> num_functions);
   nodal = (AMGhybrid_data -> nodal);
   num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   grid_relax_points = (AMGhybrid_data -> grid_relax_points);
   relax_weight = (AMGhybrid_data -> relax_weight);
   omega = (AMGhybrid_data -> omega);
   max_coarse_size = (AMGhybrid_data -> max_coarse_size);
   seq_threshold = (AMGhybrid_data -> seq_threshold);
   dof_func = (AMGhybrid_data -> dof_func);
   pcg_default    = (AMGhybrid_data -> pcg_default);
   if (!b)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   num_variables = hypre_VectorSize(hypre_ParVectorLocalVector(b));
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (!x)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }
  
/*  print_level definitions: xy,  sol_print_level = y, pre_print_level = x */
   pre_print_level = print_level/10;
   sol_print_level = print_level - pre_print_level*10;
  
   pcg_solver = (AMGhybrid_data -> pcg_solver);
   pcg_precond = (AMGhybrid_data -> pcg_precond);
   (AMGhybrid_data -> dscg_num_its) = 0;
   (AMGhybrid_data -> pcg_num_its) = 0;
                                                                                                                                        
   if (setup_type || pcg_precond == NULL)
   {
    if (pcg_precond)
    {
       hypre_BoomerAMGDestroy(pcg_precond);
       pcg_precond = NULL;
       (AMGhybrid_data -> pcg_precond) = NULL;
    }
    if (solver_type == 1)
    {
     if (pcg_solver == NULL)
     {
      pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
      pcg_solver = hypre_PCGCreate( pcg_functions );
                                                                                                                                        
      hypre_PCGSetMaxIter(pcg_solver, dscg_max_its);
      hypre_PCGSetTol(pcg_solver, tol);
      hypre_PCGSetAbsoluteTol(pcg_solver, a_tol);
      hypre_PCGSetTwoNorm(pcg_solver, two_norm);
      hypre_PCGSetStopCrit(pcg_solver, stop_crit);
      hypre_PCGSetRelChange(pcg_solver, rel_change);
      hypre_PCGSetLogging(pcg_solver, logging);
      hypre_PCGSetPrintLevel(pcg_solver, sol_print_level);
                                                                                                                                        
      pcg_precond = NULL;
     }
                                                                                                                                        
     hypre_PCGSetConvergenceFactorTol(pcg_solver, cf_tol);
     hypre_PCGSetPrecond(pcg_solver,
                       HYPRE_ParCSRDiagScale,
                       HYPRE_ParCSRDiagScaleSetup,
                       pcg_precond);
     hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
     (AMGhybrid_data -> pcg_solver) = pcg_solver;

     /*---------------------------------------------------------------------
      * Solve with DSCG.
      *---------------------------------------------------------------------*/
     hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

     /*---------------------------------------------------------------------
      * Get information for DSCG.
      *---------------------------------------------------------------------*/
     hypre_PCGGetNumIterations(pcg_solver, &dscg_num_its);
     (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
     hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

     hypre_PCGGetConverged(pcg_solver, &converged);

    }
    else if (solver_type == 2)
    {
     if (pcg_solver == NULL)
     {
      gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
      pcg_solver = hypre_GMRESCreate( gmres_functions );
                                                                                                                                        
      hypre_GMRESSetMaxIter(pcg_solver, dscg_max_its);
      hypre_GMRESSetTol(pcg_solver, tol);
      hypre_GMRESSetAbsoluteTol(pcg_solver, a_tol);
      hypre_GMRESSetKDim(pcg_solver, k_dim);
      hypre_GMRESSetStopCrit(pcg_solver, stop_crit);
      hypre_GMRESSetRelChange(pcg_solver, rel_change);
      hypre_GMRESSetLogging(pcg_solver, logging);
      hypre_GMRESSetPrintLevel(pcg_solver, sol_print_level);
                                                                                                                                        
      pcg_precond = NULL;
     }
                                                                                                                                        
     hypre_GMRESSetConvergenceFactorTol(pcg_solver, cf_tol);
     hypre_GMRESSetPrecond(pcg_solver,
                       HYPRE_ParCSRDiagScale,
                       HYPRE_ParCSRDiagScaleSetup,
                       pcg_precond);
     hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
     (AMGhybrid_data -> pcg_solver) = pcg_solver;

     /*---------------------------------------------------------------------
      * Solve with diagonal scaled GMRES
      *---------------------------------------------------------------------*/
      hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

     /*---------------------------------------------------------------------
      * Get information for GMRES
      *---------------------------------------------------------------------*/
      hypre_GMRESGetNumIterations(pcg_solver, &dscg_num_its);
      (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

      hypre_GMRESGetConverged(pcg_solver, &converged);

    }
    else if (solver_type == 3)
    {
     if (pcg_solver == NULL)
     {
      bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
      pcg_solver = hypre_BiCGSTABCreate( bicgstab_functions );
                                                                                                                                        
      hypre_BiCGSTABSetMaxIter(pcg_solver, dscg_max_its);
      hypre_BiCGSTABSetTol(pcg_solver, tol);
      hypre_BiCGSTABSetAbsoluteTol(pcg_solver, a_tol);
      hypre_BiCGSTABSetStopCrit(pcg_solver, stop_crit);
      hypre_BiCGSTABSetLogging(pcg_solver, logging);
      hypre_BiCGSTABSetPrintLevel(pcg_solver, sol_print_level);
                                                                                                                                        
      pcg_precond = NULL;
     }
                                                                                                                                        
     hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, cf_tol);
     hypre_BiCGSTABSetPrecond(pcg_solver,
                       HYPRE_ParCSRDiagScale,
                       HYPRE_ParCSRDiagScaleSetup,
                       pcg_precond);
     hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);
     (AMGhybrid_data -> pcg_solver) = pcg_solver;
                                                                                                                                        
     /*---------------------------------------------------------------------
      * Solve with diagonal scaled BiCGSTAB
      *---------------------------------------------------------------------*/
      hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

     /*---------------------------------------------------------------------
      * Get information for BiCGSTAB
      *---------------------------------------------------------------------*/
     hypre_BiCGSTABGetNumIterations(pcg_solver, &dscg_num_its);
     (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
     hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

     hypre_BiCGSTABGetConverged(pcg_solver, &converged);

    }
   }

   /*---------------------------------------------------------------------
    * If converged, done... 
    *---------------------------------------------------------------------*/
   if (converged)
   {
      if (logging)
         (AMGhybrid_data -> final_rel_res_norm) = res_norm;
   }
   /*-----------------------------------------------------------------------
    * ... otherwise, use AMG+solver
    *-----------------------------------------------------------------------*/
   else
   {
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      if (solver_type == 1)
      {
         hypre_PCGSetMaxIter(pcg_solver, pcg_max_its);
         hypre_PCGSetConvergenceFactorTol(pcg_solver, 0.0);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESSetMaxIter(pcg_solver, pcg_max_its);
         hypre_GMRESSetConvergenceFactorTol(pcg_solver, 0.0);
      }
      else if (solver_type == 3)
      {
         hypre_BiCGSTABSetMaxIter(pcg_solver, pcg_max_its);
         hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, 0.0);
      }

      /* Setup preconditioner */
      if (setup_type && pcg_default)
      {
         pcg_precond = hypre_BoomerAMGCreate();
         hypre_BoomerAMGSetMaxIter(pcg_precond, 1);
         hypre_BoomerAMGSetTol(pcg_precond, 0.0);
         hypre_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         hypre_BoomerAMGSetInterpType(pcg_precond, interp_type);
         hypre_BoomerAMGSetSetupType(pcg_precond, setup_type);
         hypre_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         hypre_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         hypre_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         hypre_BoomerAMGSetPMaxElmts(pcg_precond, pmax);
         hypre_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         hypre_BoomerAMGSetPrintLevel(pcg_precond, pre_print_level);
         hypre_BoomerAMGSetMaxLevels(pcg_precond,  max_levels);
         hypre_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         hypre_BoomerAMGSetMaxCoarseSize(pcg_precond, max_coarse_size);
         hypre_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         hypre_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         hypre_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         hypre_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         hypre_BoomerAMGSetNodal(pcg_precond, nodal);
   	 if (num_grid_sweeps)
         {	
	    boom_ngs = hypre_CTAlloc(HYPRE_Int,4);
	    for (i=0; i < 4; i++)
	       boom_ngs[i] = num_grid_sweeps[i];
            hypre_BoomerAMGSetNumGridSweeps(pcg_precond, boom_ngs);
         }
   	 if (grid_relax_type)
         {
	    boom_grt = hypre_CTAlloc(HYPRE_Int,4);
	    for (i=0; i < 4; i++)
	       boom_grt[i] = grid_relax_type[i];
	    if (solver_type == 1 && grid_relax_type[1] == 3 &&
		grid_relax_type[2] == 3)
	       boom_grt[2] = 4;
   	    hypre_BoomerAMGSetGridRelaxType(pcg_precond, boom_grt);
         }
   	 if (relax_weight)
         {
	    boom_rlxw = hypre_CTAlloc(double,max_levels);
	    for (i=0; i < max_levels; i++)
	       boom_rlxw[i] = relax_weight[i];
            hypre_BoomerAMGSetRelaxWeight(pcg_precond, boom_rlxw);
         }
   	 if (omega)
         {
	    boom_omega = hypre_CTAlloc(double,max_levels);
	    for (i=0; i < max_levels; i++)
	       boom_omega[i] = omega[i];
            hypre_BoomerAMGSetOmega(pcg_precond, boom_omega);
         }
   	 if (grid_relax_points)
         {
	    boom_grp = hypre_CTAlloc(HYPRE_Int*,4);
	    for (i=0; i < 4; i++)
 	    {
	       boom_grp[i] = hypre_CTAlloc(HYPRE_Int, num_grid_sweeps[i]);
	       for (j=0; j < num_grid_sweeps[i]; j++)
		  boom_grp[i][j] = grid_relax_points[i][j];
    	    }
            hypre_BoomerAMGSetGridRelaxPoints(pcg_precond, boom_grp);
         }
   	 if (dof_func)
         {
	    boom_dof_func = hypre_CTAlloc(HYPRE_Int,num_variables);
	    for (i=0; i < num_variables; i++)
	       boom_dof_func[i] = dof_func[i];
            hypre_BoomerAMGSetDofFunc(pcg_precond, boom_dof_func);
         }
         pcg_precond_solve = hypre_BoomerAMGSolve;
         pcg_precond_setup = hypre_BoomerAMGSetup;
         (AMGhybrid_data -> pcg_precond_setup) = pcg_precond_setup;
         (AMGhybrid_data -> pcg_precond_solve) = pcg_precond_solve;
         (AMGhybrid_data -> pcg_precond) = pcg_precond;
         /*(AMGhybrid_data -> pcg_default) = 0;*/
         /*(AMGhybrid_data -> setup_type) = 0;*/
      }
      else
      {
         pcg_precond       = (AMGhybrid_data -> pcg_precond);
         pcg_precond_solve = (AMGhybrid_data -> pcg_precond_solve);
         pcg_precond_setup = (AMGhybrid_data -> pcg_precond_setup);
         hypre_BoomerAMGSetSetupType(pcg_precond, setup_type);
      }

      /* Complete setup of solver+AMG */
      if (solver_type == 1)
      {
         hypre_PCGSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
         hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from PCG that is always logged in AMGhybrid solver*/
         hypre_PCGGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
            hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }
      }
      else if (solver_type == 2)
      {
         hypre_GMRESSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
         hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from GMRES that is always logged in AMGhybrid solver*/
         hypre_GMRESGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
            hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }
      }
      else if (solver_type == 3)
      {
         hypre_BiCGSTABSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
         hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from BiCGSTAB that is always logged in AMGhybrid solver*/
         hypre_BiCGSTABGetNumIterations(pcg_solver, &pcg_num_its);
         (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
         if (logging)
         {
	    hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
            (AMGhybrid_data -> final_rel_res_norm) = res_norm;
         }
      }
   }

   return hypre_error_flag;
}

