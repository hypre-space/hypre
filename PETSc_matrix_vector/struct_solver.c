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
 * Member functions for zzz_StructSolver class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_NewStructSolver
 *--------------------------------------------------------------------------*/

zzz_StructSolver *
zzz_NewStructSolver( MPI_Comm     context,
		      zzz_StructGrid    *grid,
		      zzz_StructStencil *stencil )
{
   zzz_StructSolver    *struct_solver;


   struct_solver = talloc(zzz_StructSolver, 1);

   zzz_StructSolverContext(struct_solver) = context;
   zzz_StructSolverStructGrid(struct_solver)    = grid;
   zzz_StructSolverStructStencil(struct_solver) = stencil;

   zzz_StructSolverMatrix(struct_solver) = NULL;
   zzz_StructSolverSoln(struct_solver) = NULL;
   zzz_StructSolverRhs(struct_solver) = NULL;

   zzz_StructSolverData(struct_solver) = NULL;

   /* set defaults */
   zzz_StructSolverSolverType(struct_solver) = ZZZ_PETSC_SOLVER;

   return struct_solver;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructSolver
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructSolver( zzz_StructSolver *struct_solver )
{
   if ( zzz_StructSolverSolverType(struct_solver) == ZZZ_PETSC_SOLVER )
      zzz_FreeStructSolverPETSc( struct_solver );
   else
      return(-1);

   tfree(struct_solver);

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_StructSolverSetType
 *--------------------------------------------------------------------------*/

int 
zzz_StructSolverSetType( zzz_StructSolver *solver, int type )
{
   if( type == ZZZ_PETSC_SOLVER )
   {
      zzz_StructSolverSolverType(solver) = type;
      return(0);
   }
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_StructSolverSetup
 *   Internal routine for setting up solver data like factorizations etc.
 *--------------------------------------------------------------------------*/

int 
zzz_StructSolverSetup( zzz_StructSolver *solver, zzz_StructMatrix *matrix,
                        zzz_StructVector *soln, zzz_StructVector *rhs )
{
  zzz_StructSolverMatrix( solver ) = matrix;
  zzz_StructSolverSoln( solver ) = soln;
  zzz_StructSolverRhs( solver ) = rhs;

  if ( zzz_StructSolverSolverType(solver) == ZZZ_PETSC_SOLVER )
     return( zzz_StructSolverSetupPETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * zzz_StructSolverSolve
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
zzz_StructSolverSolve( zzz_StructSolver *solver )
{
  if ( zzz_StructSolverSolverType(solver) == ZZZ_PETSC_SOLVER )
     return( zzz_StructSolverSolvePETSc( solver ) );
  else
     return(-1);
}

