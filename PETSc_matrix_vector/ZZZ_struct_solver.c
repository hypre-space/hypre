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
 * ZZZ_StructSolver interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructSolver
 *--------------------------------------------------------------------------*/

ZZZ_StructSolver *
ZZZ_NewStructSolver( MPI_Comm     context,
		      ZZZ_StructGrid    *grid,
		      ZZZ_StructStencil *stencil )
{
   return ( (ZZZ_StructSolver *)
	    zzz_NewStructSolver( context,
				  (zzz_StructGrid *) grid,
				  (zzz_StructStencil *) stencil ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructSolver
 *--------------------------------------------------------------------------*/

int 
ZZZ_FreeStructSolver( ZZZ_StructSolver *struct_solver )
{
   return( zzz_FreeStructSolver( (zzz_StructSolver *) struct_solver ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSolverSetType
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSolverSetType( ZZZ_StructSolver *solver, int type )
{
   return( zzz_StructSolverSetType( (zzz_StructSolver *) solver, type ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSolverSetup
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSolverSetup( ZZZ_StructSolver *solver, ZZZ_StructMatrix *matrix,
                        ZZZ_StructVector *soln, ZZZ_StructVector *rhs )
{
   return( zzz_StructSolverSetup( (zzz_StructSolver *) solver,
                                   (zzz_StructMatrix *) matrix,
                                   (zzz_StructVector *) soln,
                                   (zzz_StructVector *) rhs ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_StructSolverSolve
 *--------------------------------------------------------------------------*/

int 
ZZZ_StructSolverSolve( ZZZ_StructSolver *solver )
{
   return( zzz_StructSolverSolve( (zzz_StructSolver *) solver ) );
}

