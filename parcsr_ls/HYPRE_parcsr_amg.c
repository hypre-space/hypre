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
 * HYPRE_ParAMGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Solver
HYPRE_ParAMGInitialize( )
{
   return ( (HYPRE_Solver) hypre_ParAMGInitialize( ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParAMGFinalize( HYPRE_Solver solver )
{
   return( hypre_ParAMGFinalize( (void *) solver ) );
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
   return( hypre_ParAMGSetGridRelaxPoints( (void *) solver,
                                           grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_ParAMGSetRelaxWeight( HYPRE_Solver solver,
                            double       relax_weight  )
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

