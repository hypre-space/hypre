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
 * HYPRE_ParAMGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGCreate( HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_ParAMGCreate( ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParAMGDestroy( HYPRE_Solver solver )
{
   return( hypre_ParAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParAMGSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   return( hypre_ParAMGSetup( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParAMGSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{


   return( hypre_ParAMGSolve( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSolveT
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParAMGSolveT( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{


   return( hypre_ParAMGSolveT( (void *) solver,
                              (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b,
                              (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetRestriction
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetRestriction( HYPRE_Solver solver,
                          int          restr_par  )
{
   return( hypre_ParAMGSetRestriction( (void *) solver, restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetMaxLevels( HYPRE_Solver solver,
                          int          max_levels  )
{
   return( hypre_ParAMGSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetStrongThreshold( HYPRE_Solver solver,
                                double       strong_threshold  )
{
   return( hypre_ParAMGSetStrongThreshold( (void *) solver,
                                           strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetTruncFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetTruncFactor( HYPRE_Solver solver,
                            double       trunc_factor  )
{
   return( hypre_ParAMGSetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetInterpType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetInterpType( HYPRE_Solver solver,
                           int          interp_type  )
{
   return( hypre_ParAMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetMaxIter( HYPRE_Solver solver,
                        int          max_iter  )
{
   return( hypre_ParAMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetCoarsenType( HYPRE_Solver solver,
                            int          coarsen_type  )
{
   return( hypre_ParAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMeasureType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetMeasureType( HYPRE_Solver solver,
                            int          measure_type  )
{
   return( hypre_ParAMGSetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetCycleType( HYPRE_Solver solver,
                          int          cycle_type  )
{
   return( hypre_ParAMGSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetTol( HYPRE_Solver solver,
                    double       tol    )
{
   return( hypre_ParAMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetNumGridSweeps( HYPRE_Solver  solver,
                              int          *num_grid_sweeps  )
{
   return( hypre_ParAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGInitGridRelaxation( int     **num_grid_sweeps_ptr,
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
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int, 1);
   grid_relax_points[3][0] = 0;

   for (i = 0; i < max_levels; i++)
      relax_weights[i] = 0.;

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetGridRelaxType( HYPRE_Solver  solver,
                              int          *grid_relax_type  )
{
   return( hypre_ParAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetGridRelaxPoints( HYPRE_Solver   solver,
                                int          **grid_relax_points  )
{
   return( hypre_ParAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetRelaxWeight( HYPRE_Solver  solver,
                            double       *relax_weight  )
{
   return( hypre_ParAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetIOutDat
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetIOutDat( HYPRE_Solver solver,
                        int          ioutdat  )
{
   return( hypre_ParAMGSetIOutDat( (void *) solver, ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetLogFileName
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetLogFileName( HYPRE_Solver  solver,
                            char         *log_file_name  )
{
   return( hypre_ParAMGSetLogFileName( (void *) solver, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetLogging( HYPRE_Solver  solver,
                        int           ioutdat,
                        char         *log_file_name  )
{
   return( hypre_ParAMGSetLogging( (void *) solver, ioutdat, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetDebugFlag
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetDebugFlag( HYPRE_Solver solver,
                          int          debug_flag  )
{
   return( hypre_ParAMGSetDebugFlag( (void *) solver, debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGGetNumIterations( HYPRE_Solver  solver,
                              int          *num_iterations  )
{
   return( hypre_ParAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                          double       *rel_resid_norm  )
{
   return( hypre_ParAMGGetRelativeResidualNorm( (void *) solver, rel_resid_norm ) );
}

