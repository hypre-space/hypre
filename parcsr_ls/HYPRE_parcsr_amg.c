/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
   return 0;
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
 * HYPRE_BoomerAMGSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxLevels( HYPRE_Solver solver,
                          int          max_levels  )
{
   return( hypre_BoomerAMGSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetStrongThreshold( HYPRE_Solver solver,
                                double       strong_threshold  )
{
   return( hypre_BoomerAMGSetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMaxRowSum( HYPRE_Solver solver,
                          double       max_row_sum  )
{
   return( hypre_BoomerAMGSetMaxRowSum( (void *) solver,
                                     max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetTruncFactor( HYPRE_Solver solver,
                            double       trunc_factor  )
{
   return( hypre_BoomerAMGSetTruncFactor( (void *) solver,
                                           trunc_factor ) );
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

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCoarsenType( HYPRE_Solver solver,
                            int          coarsen_type  )
{
   return( hypre_BoomerAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetMeasureType( HYPRE_Solver solver,
                               int          measure_type  )
{
   return( hypre_BoomerAMGSetMeasureType( (void *) solver, measure_type ) );
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
 * HYPRE_BoomerAMGSetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetCycleType( HYPRE_Solver solver,
                          int          cycle_type  )
{
   return( hypre_BoomerAMGSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetTol( HYPRE_Solver solver,
                    double       tol    )
{
   return( hypre_BoomerAMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumGridSweeps( HYPRE_Solver  solver,
                              int          *num_grid_sweeps  )
{
   return( hypre_BoomerAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
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

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetGridRelaxType( HYPRE_Solver  solver,
                              int          *grid_relax_type  )
{
   return( hypre_BoomerAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetGridRelaxPoints( HYPRE_Solver   solver,
                                int          **grid_relax_points  )
{
   return( hypre_BoomerAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetRelaxWeight( HYPRE_Solver  solver,
                            double       *relax_weight  )
{
   return( hypre_BoomerAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothOption
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSmoothOption( HYPRE_Solver  solver,
                            int       *smooth_option  )
{
   return( hypre_BoomerAMGSetSmoothOption( (void *) solver, smooth_option ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSweep
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSmoothNumSweep( HYPRE_Solver  solver,
                            int       smooth_num_sweep  )
{
   return( hypre_BoomerAMGSetSmoothNumSweep((void *)solver,smooth_num_sweep ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetIOutDat
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetIOutDat( HYPRE_Solver solver,
                        int          ioutdat  )
{
   return( hypre_BoomerAMGSetIOutDat( (void *) solver, ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogFileName
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLogFileName( HYPRE_Solver  solver,
                               const char   *log_file_name  )
{
   return( hypre_BoomerAMGSetLogFileName( (void *) solver, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetLogging( HYPRE_Solver  solver,
                           int           ioutdat,
                           const char   *log_file_name  )
{
   return( hypre_BoomerAMGSetLogging( (void *) solver, ioutdat, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDebugFlag( HYPRE_Solver solver,
                          int          debug_flag  )
{
   return( hypre_BoomerAMGSetDebugFlag( (void *) solver, debug_flag ) );
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
 * HYPRE_BoomerAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                          double       *rel_resid_norm  )
{
   return( hypre_BoomerAMGGetRelResidualNorm( (void *) solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetVariant( HYPRE_Solver  solver,
                              int          variant  )
{
   return( hypre_BoomerAMGSetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetOverlap( HYPRE_Solver  solver,
                              int          overlap  )
{
   return( hypre_BoomerAMGSetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDomainType( HYPRE_Solver  solver,
                              int          domain_type  )
{
   return( hypre_BoomerAMGSetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetSchwarzRlxWeight( HYPRE_Solver  solver,
                                double schwarz_rlx_weight)
{
   return( hypre_BoomerAMGSetSchwarzRlxWeight( (void *) solver, 
			schwarz_rlx_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetNumFunctions( HYPRE_Solver  solver,
                              int          num_functions  )
{
   return( hypre_BoomerAMGSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

int
HYPRE_BoomerAMGSetDofFunc( HYPRE_Solver  solver,
                              int          *dof_func  )
{
   return( hypre_BoomerAMGSetDofFunc( (void *) solver, dof_func ) );
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
