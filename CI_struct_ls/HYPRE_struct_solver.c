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
 * HYPRE_StructInterfaceSolver interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverCreate( MPI_Comm     context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil,
                      HYPRE_StructInterfaceSolver *solver )
{
   *solver = ( (HYPRE_StructInterfaceSolver)
	    hypre_NewStructInterfaceSolver( context,
				  grid,
				  stencil ) );
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverDestroy( HYPRE_StructInterfaceSolver solver )
{
   return( hypre_FreeStructInterfaceSolver( (hypre_StructInterfaceSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverSetType
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverSetType( HYPRE_StructInterfaceSolver solver, int type )
{
   return( hypre_StructInterfaceSolverSetType( (hypre_StructInterfaceSolver *) solver, type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverInitialize( HYPRE_StructInterfaceSolver solver )
{
   return( hypre_StructInterfaceSolverInitialize( (hypre_StructInterfaceSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverSetup( HYPRE_StructInterfaceSolver solver, HYPRE_StructInterfaceMatrix matrix,
                        HYPRE_StructInterfaceVector soln, HYPRE_StructInterfaceVector rhs )
{
   return( hypre_StructInterfaceSolverSetup( (hypre_StructInterfaceSolver *) solver,
                                  matrix,
                                  soln,
                                  rhs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverSolve( HYPRE_StructInterfaceSolver solver )
{
   return( hypre_StructInterfaceSolverSolve( (hypre_StructInterfaceSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverSetDropTolerance( HYPRE_StructInterfaceSolver solver, double tol )
{
   return( hypre_StructInterfaceSolverSetDropTolerance( (hypre_StructInterfaceSolver *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructInterfaceSolverSetFactorRowSize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructInterfaceSolverSetFactorRowSize( HYPRE_StructInterfaceSolver solver, int size )
{
   return( hypre_StructInterfaceSolverSetFactorRowSize( (hypre_StructInterfaceSolver *) solver, size) );
}

