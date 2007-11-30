/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

int
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

int 
HYPRE_BoomerAMGDestroy( HYPRE_Solver solver )
{
   return( hypre_BoomerAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

int 
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

int 
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

int 
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

int
HYPRE_BoomerAMGSetRestriction( HYPRE_Solver solver,
                          int          restr_par  )
{
   return( hypre_BoomerAMGSetRestriction( (void *) solver, restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxLevels, HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxLevels( HYPRE_Solver solver,
                          int          max_levels  )
{
   return( hypre_BoomerAMGSetMaxLevels( (void *) solver, max_levels ) );
}

int
HYPRE_BoomerAMGGetMaxLevels( HYPRE_Solver solver,
                          int        * max_levels  )
{
   return( hypre_BoomerAMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetStrongThreshold( HYPRE_Solver solver,
                                double       strong_threshold  )
{
   return( hypre_BoomerAMGSetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

int
HYPRE_BoomerAMGGetStrongThreshold( HYPRE_Solver solver,
                                double     * strong_threshold  )
{
   return( hypre_BoomerAMGGetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxRowSum( HYPRE_Solver solver,
                          double       max_row_sum  )
{
   return( hypre_BoomerAMGSetMaxRowSum( (void *) solver,
                                     max_row_sum ) );
}

int
HYPRE_BoomerAMGGetMaxRowSum( HYPRE_Solver solver,
                          double     * max_row_sum  )
{
   return( hypre_BoomerAMGGetMaxRowSum( (void *) solver,
                                     max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetTruncFactor( HYPRE_Solver solver,
                            double       trunc_factor  )
{
   return( hypre_BoomerAMGSetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

int
HYPRE_BoomerAMGGetTruncFactor( HYPRE_Solver solver,
                            double     * trunc_factor  )
{
   return( hypre_BoomerAMGGetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPMaxElmts, HYPRE_BoomerAMGGetPMaxElmts
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetPMaxElmts( HYPRE_Solver solver,
                            int   P_max_elmts  )
{
   return( hypre_BoomerAMGSetPMaxElmts( (void *) solver,
                                           P_max_elmts ) );
}

int
HYPRE_BoomerAMGGetPMaxElmts( HYPRE_Solver solver,
                            int   * P_max_elmts  )
{
   return( hypre_BoomerAMGGetPMaxElmts( (void *) solver,
                                           P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetJacobiTruncThreshold, HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetJacobiTruncThreshold( HYPRE_Solver solver,
                            double       jacobi_trunc_threshold  )
{
   return( hypre_BoomerAMGSetJacobiTruncThreshold( (void *) solver,
                                           jacobi_trunc_threshold ) );
}

int
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

int
HYPRE_BoomerAMGSetPostInterpType( HYPRE_Solver solver,
                                  int       post_interp_type  )
{
   return( hypre_BoomerAMGSetPostInterpType( (void *) solver,
                                             post_interp_type ) );
}

int
HYPRE_BoomerAMGGetPostInterpType( HYPRE_Solver solver,
                                  int     * post_interp_type  )
{
   return( hypre_BoomerAMGGetPostInterpType( (void *) solver,
                                             post_interp_type ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSCommPkgSwitch
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSCommPkgSwitch( HYPRE_Solver solver,
                            double       S_commpkg_switch  )
{
   return( hypre_BoomerAMGSetSCommPkgSwitch( (void *) solver,
                                           S_commpkg_switch ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetInterpType( HYPRE_Solver solver,
                           int          interp_type  )
{
   return( hypre_BoomerAMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMinIter( HYPRE_Solver solver,
                        int          min_iter  )
{
   return( hypre_BoomerAMGSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxIter( HYPRE_Solver solver,
                        int          max_iter  )
{
   return( hypre_BoomerAMGSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_BoomerAMGGetMaxIter( HYPRE_Solver solver,
                        int        * max_iter  )
{
   return( hypre_BoomerAMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType, HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCoarsenType( HYPRE_Solver solver,
                            int          coarsen_type  )
{
   return( hypre_BoomerAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

int
HYPRE_BoomerAMGGetCoarsenType( HYPRE_Solver solver,
                            int        * coarsen_type  )
{
   return( hypre_BoomerAMGGetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType, HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMeasureType( HYPRE_Solver solver,
                               int          measure_type  )
{
   return( hypre_BoomerAMGSetMeasureType( (void *) solver, measure_type ) );
}

int
HYPRE_BoomerAMGGetMeasureType( HYPRE_Solver solver,
                               int        * measure_type  )
{
   return( hypre_BoomerAMGGetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSetupType( HYPRE_Solver solver,
                             int          setup_type  )
{
   return( hypre_BoomerAMGSetSetupType( (void *) solver, setup_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleType, HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCycleType( HYPRE_Solver solver,
                          int          cycle_type  )
{
   return( hypre_BoomerAMGSetCycleType( (void *) solver, cycle_type ) );
}

int
HYPRE_BoomerAMGGetCycleType( HYPRE_Solver solver,
                          int        * cycle_type  )
{
   return( hypre_BoomerAMGGetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol, HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetTol( HYPRE_Solver solver,
                    double       tol    )
{
   return( hypre_BoomerAMGSetTol( (void *) solver, tol ) );
}

int
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

int
HYPRE_BoomerAMGSetNumGridSweeps( HYPRE_Solver  solver,
                              int          *num_grid_sweeps  )
{
   return( hypre_BoomerAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 * There is no corresponding Get function.  Use GetCycleNumSweeps.
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumSweeps( HYPRE_Solver  solver,
                              int          num_sweeps  )
{
   return( hypre_BoomerAMGSetNumSweeps( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleNumSweeps, HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCycleNumSweeps( HYPRE_Solver  solver,
                              int          num_sweeps, int k  )
{
   return( hypre_BoomerAMGSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

int
HYPRE_BoomerAMGGetCycleNumSweeps( HYPRE_Solver  solver,
                              int        * num_sweeps, int k  )
{
   return( hypre_BoomerAMGGetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGInitGridRelaxation( int     **num_grid_sweeps_ptr,
                                int     **grid_relax_type_ptr,
                                int    ***grid_relax_points_ptr,
                                int       coarsen_type,
                                double  **relax_weights_ptr,
                                int       max_levels         )
{  int i;
   int *num_grid_sweeps;
   int *grid_relax_type;
   int **grid_relax_points;
   double *relax_weights;

   *num_grid_sweeps_ptr   = hypre_CTAlloc(int, 4);
   *grid_relax_type_ptr   = hypre_CTAlloc(int, 4);
   *grid_relax_points_ptr = hypre_CTAlloc(int*, 4);
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
      grid_relax_points[0] = hypre_CTAlloc(int, 4);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(int, 4);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(int, 4);
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
      grid_relax_points[0] = hypre_CTAlloc(int, 2);
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;
 
      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(int, 2);
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;
  
      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(int, 2);
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 3;
   grid_relax_points[3] = hypre_CTAlloc(int, 1);
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

int
HYPRE_BoomerAMGSetGridRelaxType( HYPRE_Solver  solver,
                              int          *grid_relax_type  )
{
   return( hypre_BoomerAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetRelaxType( HYPRE_Solver  solver,
                              int          relax_type  )
{
   return( hypre_BoomerAMGSetRelaxType( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleRelaxType, HYPRE_BoomerAMGetCycleRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCycleRelaxType( HYPRE_Solver  solver,
                              int          relax_type, int k  )
{
   return( hypre_BoomerAMGSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

int
HYPRE_BoomerAMGGetCycleRelaxType( HYPRE_Solver  solver,
                              int        * relax_type, int k  )
{
   return( hypre_BoomerAMGGetCycleRelaxType( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetRelaxOrder( HYPRE_Solver  solver,
                              int           relax_order)
{
   return( hypre_BoomerAMGSetRelaxOrder( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Ulrike Yang suspects that nobody uses this function.
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetGridRelaxPoints( HYPRE_Solver   solver,
                                int          **grid_relax_points  )
{
   return( hypre_BoomerAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetRelaxWeight( HYPRE_Solver  solver,
                               double       *relax_weight  )
{
   return( hypre_BoomerAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetRelaxWt( HYPRE_Solver  solver,
                           double        relax_wt  )
{
   return( hypre_BoomerAMGSetRelaxWt( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLevelRelaxWt( HYPRE_Solver  solver,
                                double        relax_wt, 
				int 	      level  )
{
   return( hypre_BoomerAMGSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOmega
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetOmega( HYPRE_Solver  solver,
                         double       *omega  )
{
   return( hypre_BoomerAMGSetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetOuterWt( HYPRE_Solver  solver,
                           double        outer_wt  )
{
   return( hypre_BoomerAMGSetOuterWt( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLevelOuterWt( HYPRE_Solver  solver,
                                double        outer_wt, 
				int 	      level  )
{
   return( hypre_BoomerAMGSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothType, HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSmoothType( HYPRE_Solver  solver,
                              int       smooth_type )
{
   return( hypre_BoomerAMGSetSmoothType( (void *) solver, smooth_type ) );
}

int
HYPRE_BoomerAMGGetSmoothType( HYPRE_Solver  solver,
                              int     * smooth_type )
{
   return( hypre_BoomerAMGGetSmoothType( (void *) solver, smooth_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumLevels, HYPRE_BoomerAMGGetSmoothNumLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSmoothNumLevels( HYPRE_Solver  solver,
                            int       smooth_num_levels  )
{
   return( hypre_BoomerAMGSetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

int
HYPRE_BoomerAMGGetSmoothNumLevels( HYPRE_Solver  solver,
                            int     * smooth_num_levels  )
{
   return( hypre_BoomerAMGGetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSweeps, HYPRE_BoomerAMGGetSmoothNumSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSmoothNumSweeps( HYPRE_Solver  solver,
                            int       smooth_num_sweeps  )
{
   return( hypre_BoomerAMGSetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

int
HYPRE_BoomerAMGGetSmoothNumSweeps( HYPRE_Solver  solver,
                            int     * smooth_num_sweeps  )
{
   return( hypre_BoomerAMGGetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLogging( HYPRE_Solver solver,
                            int          logging  )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   return( hypre_BoomerAMGSetLogging( (void *) solver, logging ) );
}

int
HYPRE_BoomerAMGGetLogging( HYPRE_Solver solver,
                            int        * logging  )
{
   return( hypre_BoomerAMGGetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintLevel, HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetPrintLevel( HYPRE_Solver solver,
                              int        print_level  )
{
   return( hypre_BoomerAMGSetPrintLevel( (void *) solver, print_level ) );
}

int
HYPRE_BoomerAMGGetPrintLevel( HYPRE_Solver solver,
                              int      * print_level  )
{
   return( hypre_BoomerAMGGetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetPrintFileName( HYPRE_Solver  solver,
                               const char   *print_file_name  )
{
   return( hypre_BoomerAMGSetPrintFileName( (void *) solver, print_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag, HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDebugFlag( HYPRE_Solver solver,
                          int          debug_flag  )
{
   return( hypre_BoomerAMGSetDebugFlag( (void *) solver, debug_flag ) );
}

int
HYPRE_BoomerAMGGetDebugFlag( HYPRE_Solver solver,
                          int        * debug_flag  )
{
   return( hypre_BoomerAMGGetDebugFlag( (void *) solver, debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGGetNumIterations( HYPRE_Solver  solver,
                              int          *num_iterations  )
{
   return( hypre_BoomerAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGGetCumNumIterations( HYPRE_Solver  solver,
                                    int          *cum_num_iterations  )
{
   return( hypre_BoomerAMGGetCumNumIterations( (void *) solver, cum_num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGGetResidual( HYPRE_Solver solver, HYPRE_ParVector * residual )
{
   return hypre_BoomerAMGGetResidual( (void *) solver,
                                      (hypre_ParVector **) residual );
}
                            

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                          double       *rel_resid_norm  )
{
   return( hypre_BoomerAMGGetRelResidualNorm( (void *) solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant, HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetVariant( HYPRE_Solver  solver,
                              int          variant  )
{
   return( hypre_BoomerAMGSetVariant( (void *) solver, variant ) );
}

int
HYPRE_BoomerAMGGetVariant( HYPRE_Solver  solver,
                              int        * variant  )
{
   return( hypre_BoomerAMGGetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetOverlap( HYPRE_Solver  solver,
                              int          overlap  )
{
   return( hypre_BoomerAMGSetOverlap( (void *) solver, overlap ) );
}

int
HYPRE_BoomerAMGGetOverlap( HYPRE_Solver  solver,
                              int        * overlap  )
{
   return( hypre_BoomerAMGGetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType, HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDomainType( HYPRE_Solver  solver,
                              int          domain_type  )
{
   return( hypre_BoomerAMGSetDomainType( (void *) solver, domain_type ) );
}

int
HYPRE_BoomerAMGGetDomainType( HYPRE_Solver  solver,
                              int        * domain_type  )
{
   return( hypre_BoomerAMGGetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWeight, HYPRE_BoomerAMGGetSchwarzRlxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSchwarzRlxWeight( HYPRE_Solver  solver,
                                double schwarz_rlx_weight)
{
   return( hypre_BoomerAMGSetSchwarzRlxWeight( (void *) solver, 
			schwarz_rlx_weight ) );
}

int
HYPRE_BoomerAMGGetSchwarzRlxWeight( HYPRE_Solver  solver,
                                double * schwarz_rlx_weight)
{
   return( hypre_BoomerAMGGetSchwarzRlxWeight( (void *) solver, 
			schwarz_rlx_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSym
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSym( HYPRE_Solver  solver,
                       int           sym)
{
   return( hypre_BoomerAMGSetSym( (void *) solver, sym ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLevel( HYPRE_Solver  solver,
                         int           level)
{
   return( hypre_BoomerAMGSetLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetThreshold( HYPRE_Solver  solver,
                             double        threshold  )
{
   return( hypre_BoomerAMGSetThreshold( (void *) solver, threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetFilter( HYPRE_Solver  solver,
                          double        filter  )
{
   return( hypre_BoomerAMGSetFilter( (void *) solver, filter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDropTol( HYPRE_Solver  solver,
                           double        drop_tol  )
{
   return( hypre_BoomerAMGSetDropTol( (void *) solver, drop_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxNzPerRow( HYPRE_Solver  solver,
                              int          max_nz_per_row  )
{
   return( hypre_BoomerAMGSetMaxNzPerRow( (void *) solver, max_nz_per_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetEuclidFile( HYPRE_Solver  solver,
                              char         *euclidfile)
{
   return( hypre_BoomerAMGSetEuclidFile( (void *) solver, euclidfile ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetEuLevel( HYPRE_Solver  solver,
                           int           eu_level)
{
   return( hypre_BoomerAMGSetEuLevel( (void *) solver, eu_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetEuSparseA( HYPRE_Solver  solver,
                             double        eu_sparse_A  )
{
   return( hypre_BoomerAMGSetEuSparseA( (void *) solver, eu_sparse_A ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetEuBJ( HYPRE_Solver  solver,
                        int	      eu_bj)
{
   return( hypre_BoomerAMGSetEuBJ( (void *) solver, eu_bj ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions, HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumFunctions( HYPRE_Solver  solver,
                              int          num_functions  )
{
   return( hypre_BoomerAMGSetNumFunctions( (void *) solver, num_functions ) );
}

int
HYPRE_BoomerAMGGetNumFunctions( HYPRE_Solver  solver,
                              int        * num_functions  )
{
   return( hypre_BoomerAMGGetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNodal( HYPRE_Solver  solver,
                         int          nodal  )
{
   return( hypre_BoomerAMGSetNodal( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNodalDiag( HYPRE_Solver  solver,
                         int          nodal  )
{
   return( hypre_BoomerAMGSetNodalDiag( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDofFunc( HYPRE_Solver  solver,
                              int          *dof_func  )
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

int
HYPRE_BoomerAMGSetNumPaths( HYPRE_Solver  solver,
                              int          num_paths  )
{
   return( hypre_BoomerAMGSetNumPaths( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetAggNumLevels( HYPRE_Solver  solver,
                              int          agg_num_levels  )
{
   return( hypre_BoomerAMGSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumCRRelaxSteps
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumCRRelaxSteps( HYPRE_Solver  solver,
                              int          num_CR_relax_steps  )
{
   return( hypre_BoomerAMGSetNumCRRelaxSteps( (void *) solver, num_CR_relax_steps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRRate
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCRRate( HYPRE_Solver  solver,
                          double        CR_rate  )
{
   return( hypre_BoomerAMGSetCRRate( (void *) solver, CR_rate ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRStrongTh
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCRStrongTh( HYPRE_Solver  solver,
                          double        CR_strong_th  )
{
   return( hypre_BoomerAMGSetCRStrongTh( (void *) solver, CR_strong_th ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetISType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetISType( HYPRE_Solver  solver,
                              int          IS_type  )
{
   return( hypre_BoomerAMGSetISType( (void *) solver, IS_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCRUseCG
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCRUseCG( HYPRE_Solver  solver,
                          int    CR_use_CG  )
{
   return( hypre_BoomerAMGSetCRUseCG( (void *) solver, CR_use_CG ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetGSMG( HYPRE_Solver  solver,
                              int        gsmg  )
{
   return( hypre_BoomerAMGSetGSMG( (void *) solver, gsmg ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumSamples( HYPRE_Solver  solver,
                              int        gsmg  )
{
   return( hypre_BoomerAMGSetNumSamples( (void *) solver, gsmg ) );
}
/* BM Aug 25, 2006 */
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/
                                                                                                       
int
HYPRE_BoomerAMGSetCGCIts (HYPRE_Solver solver,
                          int its)
{
  return (hypre_BoomerAMGSetCGCIts ( (void *) solver, its ) );
}
                                                                                                       
/* BM Oct 23, 2006 */
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPlotGrids
 *--------------------------------------------------------------------------*/
                                                                                                       
int
HYPRE_BoomerAMGSetPlotGrids (HYPRE_Solver solver,
                             int plotgrids)
{
  return (hypre_BoomerAMGSetPlotGrids ( (void *) solver, plotgrids ) );
}
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPlotFileName
 *--------------------------------------------------------------------------*/
                                                                                                       
int
HYPRE_BoomerAMGSetPlotFileName (HYPRE_Solver solver,
                                const char *plotfilename)
{
  return (hypre_BoomerAMGSetPlotFileName ( (void *) solver, plotfilename ) );
}
                                                                                                       
/* BM Oct 17, 2006 */
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoordDim
 *--------------------------------------------------------------------------*/
                                                                                                       
int
HYPRE_BoomerAMGSetCoordDim (HYPRE_Solver solver,
                            int coorddim)
{
  return (hypre_BoomerAMGSetCoordDim ( (void *) solver, coorddim ) );
}
                                                                                                       
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoordinates
 *--------------------------------------------------------------------------*/
                                                                                                       
int
HYPRE_BoomerAMGSetCoordinates (HYPRE_Solver solver,
                               float *coordinates)
{
  return (hypre_BoomerAMGSetCoordinates ( (void *) solver, coordinates ) );
}
