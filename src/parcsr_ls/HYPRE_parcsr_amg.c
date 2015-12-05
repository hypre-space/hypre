/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.40 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParAMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGCreate( HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_BoomerAMGCreate( ) ;
   if (!solver)
      hypre_error_in_arg(1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BoomerAMGDestroy( HYPRE_Solver solver )
{
   return( hypre_BoomerAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BoomerAMGSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   return( hypre_BoomerAMGSetup( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BoomerAMGSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{


   return( hypre_BoomerAMGSolve( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BoomerAMGSolveT( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{


   return( hypre_BoomerAMGSolveT( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetRestriction( HYPRE_Solver solver,
                          HYPRE_Int          restr_par  )
{
   return( hypre_BoomerAMGSetRestriction( (void *) solver, restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxLevels, HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMaxLevels( HYPRE_Solver solver,
                          HYPRE_Int          max_levels  )
{
   return( hypre_BoomerAMGSetMaxLevels( (void *) solver, max_levels ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetMaxLevels( HYPRE_Solver solver,
                          HYPRE_Int        * max_levels  )
{
   return( hypre_BoomerAMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxCoarseSize, HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMaxCoarseSize( HYPRE_Solver solver,
                          HYPRE_Int          max_coarse_size  )
{
   return( hypre_BoomerAMGSetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetMaxCoarseSize( HYPRE_Solver solver,
                          HYPRE_Int        * max_coarse_size  )
{
   return( hypre_BoomerAMGGetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSeqThreshold, HYPRE_BoomerAMGGetSeqThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSeqThreshold( HYPRE_Solver solver,
                          HYPRE_Int          seq_threshold  )
{
   return( hypre_BoomerAMGSetSeqThreshold( (void *) solver, seq_threshold ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetSeqThreshold( HYPRE_Solver solver,
                          HYPRE_Int        * seq_threshold  )
{
   return( hypre_BoomerAMGGetSeqThreshold( (void *) solver, seq_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetStrongThreshold( HYPRE_Solver solver,
                                double       strong_threshold  )
{
   return( hypre_BoomerAMGSetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetStrongThreshold( HYPRE_Solver solver,
                                double     * strong_threshold  )
{
   return( hypre_BoomerAMGGetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMaxRowSum( HYPRE_Solver solver,
                          double       max_row_sum  )
{
   return( hypre_BoomerAMGSetMaxRowSum( (void *) solver,
                                     max_row_sum ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetMaxRowSum( HYPRE_Solver solver,
                          double     * max_row_sum  )
{
   return( hypre_BoomerAMGGetMaxRowSum( (void *) solver,
                                     max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetTruncFactor( HYPRE_Solver solver,
                            double       trunc_factor  )
{
   return( hypre_BoomerAMGSetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetTruncFactor( HYPRE_Solver solver,
                            double     * trunc_factor  )
{
   return( hypre_BoomerAMGGetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPMaxElmts, HYPRE_BoomerAMGGetPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetPMaxElmts( HYPRE_Solver solver,
                            HYPRE_Int   P_max_elmts  )
{
   return( hypre_BoomerAMGSetPMaxElmts( (void *) solver,
                                           P_max_elmts ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetPMaxElmts( HYPRE_Solver solver,
                            HYPRE_Int   * P_max_elmts  )
{
   return( hypre_BoomerAMGGetPMaxElmts( (void *) solver,
                                           P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetJacobiTruncThreshold, HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetJacobiTruncThreshold( HYPRE_Solver solver,
                            double       jacobi_trunc_threshold  )
{
   return( hypre_BoomerAMGSetJacobiTruncThreshold( (void *) solver,
                                           jacobi_trunc_threshold ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetJacobiTruncThreshold( HYPRE_Solver solver,
                            double     * jacobi_trunc_threshold  )
{
   return( hypre_BoomerAMGGetJacobiTruncThreshold( (void *) solver,
                                           jacobi_trunc_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPostInterpType, HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetPostInterpType( HYPRE_Solver solver,
                                  HYPRE_Int       post_interp_type  )
{
   return( hypre_BoomerAMGSetPostInterpType( (void *) solver,
                                             post_interp_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetPostInterpType( HYPRE_Solver solver,
                                  HYPRE_Int     * post_interp_type  )
{
   return( hypre_BoomerAMGGetPostInterpType( (void *) solver,
                                             post_interp_type ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSCommPkgSwitch
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSCommPkgSwitch( HYPRE_Solver solver,
                            double       S_commpkg_switch  )
{
   return( hypre_BoomerAMGSetSCommPkgSwitch( (void *) solver,
                                           S_commpkg_switch ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetInterpType( HYPRE_Solver solver,
                           HYPRE_Int          interp_type  )
{
   return( hypre_BoomerAMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSepWeight( HYPRE_Solver solver,
                           HYPRE_Int          sep_weight  )
{
   return( hypre_BoomerAMGSetSepWeight( (void *) solver, sep_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMinIter( HYPRE_Solver solver,
                        HYPRE_Int          min_iter  )
{
   return( hypre_BoomerAMGSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMaxIter( HYPRE_Solver solver,
                        HYPRE_Int          max_iter  )
{
   return( hypre_BoomerAMGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetMaxIter( HYPRE_Solver solver,
                        HYPRE_Int        * max_iter  )
{
   return( hypre_BoomerAMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType, HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCoarsenType( HYPRE_Solver solver,
                            HYPRE_Int          coarsen_type  )
{
   return( hypre_BoomerAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetCoarsenType( HYPRE_Solver solver,
                            HYPRE_Int        * coarsen_type  )
{
   return( hypre_BoomerAMGGetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType, HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMeasureType( HYPRE_Solver solver,
                               HYPRE_Int          measure_type  )
{
   return( hypre_BoomerAMGSetMeasureType( (void *) solver, measure_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetMeasureType( HYPRE_Solver solver,
                               HYPRE_Int        * measure_type  )
{
   return( hypre_BoomerAMGGetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSetupType( HYPRE_Solver solver,
                             HYPRE_Int          setup_type  )
{
   return( hypre_BoomerAMGSetSetupType( (void *) solver, setup_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleType, HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCycleType( HYPRE_Solver solver,
                          HYPRE_Int          cycle_type  )
{
   return( hypre_BoomerAMGSetCycleType( (void *) solver, cycle_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetCycleType( HYPRE_Solver solver,
                          HYPRE_Int        * cycle_type  )
{
   return( hypre_BoomerAMGGetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol, HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetTol( HYPRE_Solver solver,
                    double       tol    )
{
   return( hypre_BoomerAMGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetTol( HYPRE_Solver solver,
                    double     * tol    )
{
   return( hypre_BoomerAMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumGridSweeps
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumGridSweeps( HYPRE_Solver  solver,
                              HYPRE_Int          *num_grid_sweeps  )
{
   return( hypre_BoomerAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 * There is no corresponding Get function.  Use GetCycleNumSweeps.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumSweeps( HYPRE_Solver  solver,
                              HYPRE_Int          num_sweeps  )
{
   return( hypre_BoomerAMGSetNumSweeps( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleNumSweeps, HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCycleNumSweeps( HYPRE_Solver  solver,
                              HYPRE_Int          num_sweeps, HYPRE_Int k  )
{
   return( hypre_BoomerAMGSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetCycleNumSweeps( HYPRE_Solver  solver,
                              HYPRE_Int        * num_sweeps, HYPRE_Int k  )
{
   return( hypre_BoomerAMGGetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGInitGridRelaxation( HYPRE_Int     **num_grid_sweeps_ptr,
                                HYPRE_Int     **grid_relax_type_ptr,
                                HYPRE_Int    ***grid_relax_points_ptr,
                                HYPRE_Int       coarsen_type,
                                double  **relax_weights_ptr,
                                HYPRE_Int       max_levels         )
{  HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   HYPRE_Int *grid_relax_type;
   HYPRE_Int **grid_relax_points;
   double *relax_weights;

   *num_grid_sweeps_ptr   = hypre_CTAlloc(HYPRE_Int, 4);
   *grid_relax_type_ptr   = hypre_CTAlloc(HYPRE_Int, 4);
   *grid_relax_points_ptr = hypre_CTAlloc(HYPRE_Int*, 4);
   *relax_weights_ptr     = hypre_CTAlloc(double, max_levels);

   num_grid_sweeps   = *num_grid_sweeps_ptr;
   grid_relax_type   = *grid_relax_type_ptr;
   grid_relax_points = *grid_relax_points_ptr;
   relax_weights     = *relax_weights_ptr;

   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = 3;
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, 4);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, 4);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, 4);
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {  
      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = 3;
      grid_relax_points[0] = hypre_CTAlloc(HYPRE_Int, 2);
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;
 
      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(HYPRE_Int, 2);
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;
  
      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(HYPRE_Int, 2);
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 3;
   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int, 1);
   grid_relax_points[3][0] = 0;

   for (i = 0; i < max_levels; i++)
      relax_weights[i] = 1.;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetGridRelaxType( HYPRE_Solver  solver,
                              HYPRE_Int          *grid_relax_type  )
{
   return( hypre_BoomerAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetRelaxType( HYPRE_Solver  solver,
                              HYPRE_Int          relax_type  )
{
   return( hypre_BoomerAMGSetRelaxType( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleRelaxType, HYPRE_BoomerAMGetCycleRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCycleRelaxType( HYPRE_Solver  solver,
                              HYPRE_Int          relax_type, HYPRE_Int k  )
{
   return( hypre_BoomerAMGSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetCycleRelaxType( HYPRE_Solver  solver,
                              HYPRE_Int        * relax_type, HYPRE_Int k  )
{
   return( hypre_BoomerAMGGetCycleRelaxType( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetRelaxOrder( HYPRE_Solver  solver,
                              HYPRE_Int           relax_order)
{
   return( hypre_BoomerAMGSetRelaxOrder( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Ulrike Yang suspects that nobody uses this function.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetGridRelaxPoints( HYPRE_Solver   solver,
                                HYPRE_Int          **grid_relax_points  )
{
   return( hypre_BoomerAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetRelaxWeight( HYPRE_Solver  solver,
                               double       *relax_weight  )
{
   return( hypre_BoomerAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetRelaxWt( HYPRE_Solver  solver,
                           double        relax_wt  )
{
   return( hypre_BoomerAMGSetRelaxWt( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetLevelRelaxWt( HYPRE_Solver  solver,
                                double        relax_wt, 
				HYPRE_Int 	      level  )
{
   return( hypre_BoomerAMGSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetOmega( HYPRE_Solver  solver,
                         double       *omega  )
{
   return( hypre_BoomerAMGSetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetOuterWt( HYPRE_Solver  solver,
                           double        outer_wt  )
{
   return( hypre_BoomerAMGSetOuterWt( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetLevelOuterWt( HYPRE_Solver  solver,
                                double        outer_wt, 
				HYPRE_Int 	      level  )
{
   return( hypre_BoomerAMGSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothType, HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSmoothType( HYPRE_Solver  solver,
                              HYPRE_Int       smooth_type )
{
   return( hypre_BoomerAMGSetSmoothType( (void *) solver, smooth_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetSmoothType( HYPRE_Solver  solver,
                              HYPRE_Int     * smooth_type )
{
   return( hypre_BoomerAMGGetSmoothType( (void *) solver, smooth_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumLevels, HYPRE_BoomerAMGGetSmoothNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSmoothNumLevels( HYPRE_Solver  solver,
                            HYPRE_Int       smooth_num_levels  )
{
   return( hypre_BoomerAMGSetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

HYPRE_Int
HYPRE_BoomerAMGGetSmoothNumLevels( HYPRE_Solver  solver,
                            HYPRE_Int     * smooth_num_levels  )
{
   return( hypre_BoomerAMGGetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSweeps, HYPRE_BoomerAMGGetSmoothNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSmoothNumSweeps( HYPRE_Solver  solver,
                            HYPRE_Int       smooth_num_sweeps  )
{
   return( hypre_BoomerAMGSetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

HYPRE_Int
HYPRE_BoomerAMGGetSmoothNumSweeps( HYPRE_Solver  solver,
                            HYPRE_Int     * smooth_num_sweeps  )
{
   return( hypre_BoomerAMGGetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetLogging( HYPRE_Solver solver,
                            HYPRE_Int          logging  )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   return( hypre_BoomerAMGSetLogging( (void *) solver, logging ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetLogging( HYPRE_Solver solver,
                            HYPRE_Int        * logging  )
{
   return( hypre_BoomerAMGGetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintLevel, HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int        print_level  )
{
   return( hypre_BoomerAMGSetPrintLevel( (void *) solver, print_level ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int      * print_level  )
{
   return( hypre_BoomerAMGGetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetPrintFileName( HYPRE_Solver  solver,
                               const char   *print_file_name  )
{
   return( hypre_BoomerAMGSetPrintFileName( (void *) solver, print_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag, HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetDebugFlag( HYPRE_Solver solver,
                          HYPRE_Int          debug_flag  )
{
   return( hypre_BoomerAMGSetDebugFlag( (void *) solver, debug_flag ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetDebugFlag( HYPRE_Solver solver,
                          HYPRE_Int        * debug_flag  )
{
   return( hypre_BoomerAMGGetDebugFlag( (void *) solver, debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGGetNumIterations( HYPRE_Solver  solver,
                              HYPRE_Int          *num_iterations  )
{
   return( hypre_BoomerAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGGetCumNumIterations( HYPRE_Solver  solver,
                                    HYPRE_Int          *cum_num_iterations  )
{
   return( hypre_BoomerAMGGetCumNumIterations( (void *) solver, cum_num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGGetResidual( HYPRE_Solver solver, HYPRE_ParVector * residual )
{
   return hypre_BoomerAMGGetResidual( (void *) solver,
                                      (hypre_ParVector **) residual );
}
                            

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                          double       *rel_resid_norm  )
{
   return( hypre_BoomerAMGGetRelResidualNorm( (void *) solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant, HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetVariant( HYPRE_Solver  solver,
                              HYPRE_Int          variant  )
{
   return( hypre_BoomerAMGSetVariant( (void *) solver, variant ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetVariant( HYPRE_Solver  solver,
                              HYPRE_Int        * variant  )
{
   return( hypre_BoomerAMGGetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetOverlap( HYPRE_Solver  solver,
                              HYPRE_Int          overlap  )
{
   return( hypre_BoomerAMGSetOverlap( (void *) solver, overlap ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetOverlap( HYPRE_Solver  solver,
                              HYPRE_Int        * overlap  )
{
   return( hypre_BoomerAMGGetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType, HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetDomainType( HYPRE_Solver  solver,
                              HYPRE_Int          domain_type  )
{
   return( hypre_BoomerAMGSetDomainType( (void *) solver, domain_type ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetDomainType( HYPRE_Solver  solver,
                              HYPRE_Int        * domain_type  )
{
   return( hypre_BoomerAMGGetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWeight, HYPRE_BoomerAMGGetSchwarzRlxWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSchwarzRlxWeight( HYPRE_Solver  solver,
                                double schwarz_rlx_weight)
{
   return( hypre_BoomerAMGSetSchwarzRlxWeight( (void *) solver, 
			schwarz_rlx_weight ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetSchwarzRlxWeight( HYPRE_Solver  solver,
                                double * schwarz_rlx_weight)
{
   return( hypre_BoomerAMGGetSchwarzRlxWeight( (void *) solver, 
			schwarz_rlx_weight ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzUseNonSymm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSchwarzUseNonSymm( HYPRE_Solver  solver,
                                     HYPRE_Int use_nonsymm)
{
   return( hypre_BoomerAMGSetSchwarzUseNonSymm( (void *) solver, 
			use_nonsymm ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSym
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetSym( HYPRE_Solver  solver,
                       HYPRE_Int           sym)
{
   return( hypre_BoomerAMGSetSym( (void *) solver, sym ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetLevel( HYPRE_Solver  solver,
                         HYPRE_Int           level)
{
   return( hypre_BoomerAMGSetLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetThreshold( HYPRE_Solver  solver,
                             double        threshold  )
{
   return( hypre_BoomerAMGSetThreshold( (void *) solver, threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetFilter( HYPRE_Solver  solver,
                          double        filter  )
{
   return( hypre_BoomerAMGSetFilter( (void *) solver, filter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetDropTol( HYPRE_Solver  solver,
                           double        drop_tol  )
{
   return( hypre_BoomerAMGSetDropTol( (void *) solver, drop_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetMaxNzPerRow( HYPRE_Solver  solver,
                              HYPRE_Int          max_nz_per_row  )
{
   return( hypre_BoomerAMGSetMaxNzPerRow( (void *) solver, max_nz_per_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetEuclidFile( HYPRE_Solver  solver,
                              char         *euclidfile)
{
   return( hypre_BoomerAMGSetEuclidFile( (void *) solver, euclidfile ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetEuLevel( HYPRE_Solver  solver,
                           HYPRE_Int           eu_level)
{
   return( hypre_BoomerAMGSetEuLevel( (void *) solver, eu_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetEuSparseA( HYPRE_Solver  solver,
                             double        eu_sparse_A  )
{
   return( hypre_BoomerAMGSetEuSparseA( (void *) solver, eu_sparse_A ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetEuBJ( HYPRE_Solver  solver,
                        HYPRE_Int	      eu_bj)
{
   return( hypre_BoomerAMGSetEuBJ( (void *) solver, eu_bj ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions, HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumFunctions( HYPRE_Solver  solver,
                              HYPRE_Int          num_functions  )
{
   return( hypre_BoomerAMGSetNumFunctions( (void *) solver, num_functions ) );
}

HYPRE_Int
HYPRE_BoomerAMGGetNumFunctions( HYPRE_Solver  solver,
                              HYPRE_Int        * num_functions  )
{
   return( hypre_BoomerAMGGetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNodal( HYPRE_Solver  solver,
                         HYPRE_Int          nodal  )
{
   return( hypre_BoomerAMGSetNodal( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodalLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNodalLevels( HYPRE_Solver  solver,
                         HYPRE_Int          nodal_levels  )
{
   return( hypre_BoomerAMGSetNodalLevels( (void *) solver, nodal_levels ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNodalDiag( HYPRE_Solver  solver,
                         HYPRE_Int          nodal  )
{
   return( hypre_BoomerAMGSetNodalDiag( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetDofFunc( HYPRE_Solver  solver,
                              HYPRE_Int          *dof_func  )
/* Warning about a possible memory problem: When the BoomerAMG object is destroyed
   in hypre_BoomerAMGDestroy, dof_func aka DofFunc will be destroyed (currently
   line 246 of par_amg.c).  Normally this is what we want.  But if the user provided
   dof_func by calling HYPRE_BoomerAMGSetDofFunc, this could be an unwanted surprise.
   As hypre is currently commonly used, this situation is likely to be rare. */
{
   return( hypre_BoomerAMGSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumPaths( HYPRE_Solver  solver,
                              HYPRE_Int          num_paths  )
{
   return( hypre_BoomerAMGSetNumPaths( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggNumLevels( HYPRE_Solver  solver,
                              HYPRE_Int          agg_num_levels  )
{
   return( hypre_BoomerAMGSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggInterpType( HYPRE_Solver  solver,
                              HYPRE_Int          agg_interp_type  )
{
   return( hypre_BoomerAMGSetAggInterpType( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggTruncFactor( HYPRE_Solver  solver,
                              double        agg_trunc_factor  )
{
   return( hypre_BoomerAMGSetAggTruncFactor( (void *) solver, agg_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggP12TruncFactor( HYPRE_Solver  solver,
                              double        agg_P12_trunc_factor  )
{
   return( hypre_BoomerAMGSetAggP12TruncFactor( (void *) solver, agg_P12_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggPMaxElmts( HYPRE_Solver  solver,
                              HYPRE_Int          agg_P_max_elmts  )
{
   return( hypre_BoomerAMGSetAggPMaxElmts( (void *) solver, agg_P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetAggP12MaxElmts( HYPRE_Solver  solver,
                              HYPRE_Int          agg_P12_max_elmts  )
{
   return( hypre_BoomerAMGSetAggP12MaxElmts( (void *) solver, agg_P12_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumCRRelaxSteps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumCRRelaxSteps( HYPRE_Solver  solver,
                              HYPRE_Int          num_CR_relax_steps  )
{
   return( hypre_BoomerAMGSetNumCRRelaxSteps( (void *) solver, num_CR_relax_steps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRRate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCRRate( HYPRE_Solver  solver,
                          double        CR_rate  )
{
   return( hypre_BoomerAMGSetCRRate( (void *) solver, CR_rate ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRStrongTh
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCRStrongTh( HYPRE_Solver  solver,
                          double        CR_strong_th  )
{
   return( hypre_BoomerAMGSetCRStrongTh( (void *) solver, CR_strong_th ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetISType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetISType( HYPRE_Solver  solver,
                              HYPRE_Int          IS_type  )
{
   return( hypre_BoomerAMGSetISType( (void *) solver, IS_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRUseCG
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetCRUseCG( HYPRE_Solver  solver,
                          HYPRE_Int    CR_use_CG  )
{
   return( hypre_BoomerAMGSetCRUseCG( (void *) solver, CR_use_CG ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetGSMG( HYPRE_Solver  solver,
                              HYPRE_Int        gsmg  )
{
   return( hypre_BoomerAMGSetGSMG( (void *) solver, gsmg ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetNumSamples( HYPRE_Solver  solver,
                              HYPRE_Int        gsmg  )
{
   return( hypre_BoomerAMGSetNumSamples( (void *) solver, gsmg ) );
}
/* BM Aug 25, 2006 */
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetCGCIts (HYPRE_Solver solver,
                          HYPRE_Int its)
{
  return (hypre_BoomerAMGSetCGCIts ( (void *) solver, its ) );
}
                                                                                                       
/* BM Oct 23, 2006 */
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPlotGrids
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetPlotGrids (HYPRE_Solver solver,
                             HYPRE_Int plotgrids)
{
  return (hypre_BoomerAMGSetPlotGrids ( (void *) solver, plotgrids ) );
}
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPlotFileName
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetPlotFileName (HYPRE_Solver solver,
                                const char *plotfilename)
{
  return (hypre_BoomerAMGSetPlotFileName ( (void *) solver, plotfilename ) );
}
                                                                                                       
/* BM Oct 17, 2006 */
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoordDim
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetCoordDim (HYPRE_Solver solver,
                            HYPRE_Int coorddim)
{
  return (hypre_BoomerAMGSetCoordDim ( (void *) solver, coorddim ) );
}
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoordinates
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetCoordinates (HYPRE_Solver solver,
                               float *coordinates)
{
  return (hypre_BoomerAMGSetCoordinates ( (void *) solver, coordinates ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetChebyOrder( HYPRE_Solver  solver,
                              HYPRE_Int        order )
{
   return( hypre_BoomerAMGSetChebyOrder( (void *) solver, order ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyEigRatio
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetChebyFraction( HYPRE_Solver  solver,
                                 double         ratio )
{
   return( hypre_BoomerAMGSetChebyFraction( (void *) solver, ratio ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVectors
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetInterpVectors (HYPRE_Solver solver, HYPRE_Int num_vectors,
                                 HYPRE_ParVector *vectors)
{
   return (hypre_BoomerAMGSetInterpVectors ( (void *) solver, 
                                             num_vectors, 
                                             (hypre_ParVector **) vectors ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecVariant
 *--------------------------------------------------------------------------*/
                                                                                                       
HYPRE_Int
HYPRE_BoomerAMGSetInterpVecVariant(HYPRE_Solver solver, HYPRE_Int num)
                            
{
   return (hypre_BoomerAMGSetInterpVecVariant ( (void *) solver, num ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecQMax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BoomerAMGSetInterpVecQMax( HYPRE_Solver solver,
                                 HYPRE_Int       q_max  )
{
   return( hypre_BoomerAMGSetInterpVecQMax( (void *) solver,
                                            q_max ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecAbsQTrunc
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_BoomerAMGSetInterpVecAbsQTrunc( HYPRE_Solver solver,
                                      double       q_trunc  )
{
   return( hypre_BoomerAMGSetInterpVecAbsQTrunc( (void *) solver,
                                                 q_trunc ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothInterpVectors
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_BoomerAMGSetSmoothInterpVectors( HYPRE_Solver solver,
                                       HYPRE_Int       smooth_interp_vectors  )
{
   return( hypre_BoomerAMGSetSmoothInterpVectors( (void *) solver,
                                                  smooth_interp_vectors) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpRefine
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_BoomerAMGSetInterpRefine( HYPRE_Solver solver,
                                HYPRE_Int          num_refine  )
{
   return( hypre_BoomerAMGSetInterpRefine( (void *) solver,
                                           num_refine ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecFirstLevel(
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_BoomerAMGSetInterpVecFirstLevel( HYPRE_Solver solver,
                                       HYPRE_Int       level  )
{
   return( hypre_BoomerAMGSetInterpVecFirstLevel( (void *) solver,
                                                  level ) );
}
