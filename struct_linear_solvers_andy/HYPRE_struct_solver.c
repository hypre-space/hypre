/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructSolver interface
 *
 *****************************************************************************/

#include "./headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructSolver
 *--------------------------------------------------------------------------*/

HYPRE_StructSolver 
HYPRE_NewStructSolver( MPI_Comm     context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil )
{
   return ( (HYPRE_StructSolver)
	    hypre_NewStructSolver( context,
				  grid,
				  stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructSolver
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeStructSolver( HYPRE_StructSolver struct_solver )
{
   return( hypre_FreeStructSolver( (hypre_StructSolver *) struct_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverSetType
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverSetType( HYPRE_StructSolver solver, int type )
{
   return( hypre_StructSolverSetType( (hypre_StructSolver *) solver, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverInitialize( HYPRE_StructSolver solver )
{
   return( hypre_StructSolverInitialize( (hypre_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverSetup( HYPRE_StructSolver solver, HYPRE_StructMatrix matrix,
                        HYPRE_StructVector soln, HYPRE_StructVector rhs )
{
   return( hypre_StructSolverSetup( (hypre_StructSolver *) solver,
                                  matrix,
                                  soln,
                                  rhs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverSolve( HYPRE_StructSolver solver )
{
   return( hypre_StructSolverSolve( (hypre_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverSetDropTolerance( HYPRE_StructSolver solver, double tol )
{
   return( hypre_StructSolverSetDropTolerance( (hypre_StructSolver *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSolverSetFactorRowSize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSolverSetFactorRowSize( HYPRE_StructSolver solver, int size )
{
   return( hypre_StructSolverSetFactorRowSize( (hypre_StructSolver *) solver, size) );
}

