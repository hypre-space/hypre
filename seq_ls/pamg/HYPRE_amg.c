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
 * HYPRE_AMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Solver
HYPRE_AMGInitialize( )
{
   return ( (HYPRE_Solver) hypre_AMGInitialize( ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGFinalize( HYPRE_Solver solver )
{
   return( hypre_AMGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSetup( HYPRE_Solver solver,
                HYPRE_CSRMatrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_AMGSetup( (void *) solver,
                           (hypre_CSRMatrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_AMGSolve( HYPRE_Solver solver,
                HYPRE_CSRMatrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{


   return( hypre_AMGSolve( (void *) solver,
                           (hypre_CSRMatrix *) A,
                           (hypre_Vector *) b,
                           (hypre_Vector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMaxLevels( HYPRE_Solver solver,
                       int          max_levels  )
{
   return( hypre_AMGSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver,
                             double       strong_threshold  )
{
   return( hypre_AMGSetStrongThreshold( (void *) solver, strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetCoarsenType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetCoarsenType( HYPRE_Solver solver,
                         int          coarsen_type  )
{
   return( hypre_AMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetInterpType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetInterpType( HYPRE_Solver solver,
                        int          interp_type  )
{
   return( hypre_AMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetMaxIter( HYPRE_Solver solver,
                     int          max_iter  )
{
   return( hypre_AMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetCycleType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetCycleType( HYPRE_Solver solver,
                       int          cycle_type  )
{
   return( hypre_AMGSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetTol( HYPRE_Solver solver,
                 double       tol    )
{
   return( hypre_AMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetNumGridSweeps( HYPRE_Solver  solver,
                           int          *num_grid_sweeps  )
{
   return( hypre_AMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetGridRelaxType( HYPRE_Solver  solver,
                           int          *grid_relax_type  )
{
   return( hypre_AMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver   solver,
                             int          **grid_relax_points  )
{
   return( hypre_AMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetRelaxWeight( HYPRE_Solver   solver,
                         double         relax_weight  )
{
   return( hypre_AMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetIOutDat
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetIOutDat( HYPRE_Solver solver,
                     int          ioutdat  )
{
   return( hypre_AMGSetIOutDat( (void *) solver, ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetLogFileName
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetLogFileName( HYPRE_Solver  solver,
                         char         *log_file_name  )
{
   return( hypre_AMGSetLogFileName( (void *) solver, log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_AMGSetLogging( HYPRE_Solver  solver,
                     int           ioutdat,
                     char         *log_file_name  )
{
   return( hypre_AMGSetLogging( (void *) solver, ioutdat, log_file_name ) );
}

