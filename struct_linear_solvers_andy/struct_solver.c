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
 * Member functions for hypre_StructSolver class.
 *
 *****************************************************************************/

#include "./headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructSolver
 *--------------------------------------------------------------------------*/

hypre_StructSolver *
hypre_NewStructSolver( MPI_Comm     context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil )
{
   hypre_StructSolver    *struct_solver;


   struct_solver = (hypre_StructSolver *) hypre_CTAlloc(hypre_StructSolver,1);

   hypre_StructSolverContext(struct_solver) = context;
   hypre_StructSolverStructGrid(struct_solver)    = grid;
   hypre_StructSolverStructStencil(struct_solver) = stencil;

   hypre_StructSolverMatrix(struct_solver) = NULL;
   hypre_StructSolverSoln(struct_solver) = NULL;
   hypre_StructSolverRhs(struct_solver) = NULL;

   hypre_StructSolverData(struct_solver) = NULL;

   /* set defaults */
   hypre_StructSolverSolverType(struct_solver) = HYPRE_PETSC_MAT_PARILUT_SOLVER;

   return struct_solver;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructSolver
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructSolver( hypre_StructSolver *struct_solver )
{
   if ( hypre_StructSolverSolverType(struct_solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
      hypre_FreeStructSolverPETSc( struct_solver );
   else
      return(-1);

   hypre_TFree(struct_solver);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSetType
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSetType( hypre_StructSolver *solver, int type )
{
   if( type == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
      hypre_StructSolverSolverType(solver) = type;
      return(0);
   }
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverInitialize( hypre_StructSolver *solver )
{

  if ( hypre_StructSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructSolverInitializePETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSetup
 *   Internal routine for setting up solver data like factorizations etc.
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSetup( hypre_StructSolver *solver, HYPRE_StructMatrix matrix,
                        HYPRE_StructVector soln, HYPRE_StructVector rhs )
{
  hypre_StructSolverMatrix( solver ) = matrix;
  hypre_StructSolverSoln( solver ) = soln;
  hypre_StructSolverRhs( solver ) = rhs;

  if ( hypre_StructSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructSolverSetupPETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSolve
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSolve( hypre_StructSolver *solver )
{
  if ( hypre_StructSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructSolverSolvePETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSetDropTolerance( hypre_StructSolver *solver, double tol )
{
   if( hypre_StructSolverSolverType( solver ) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
     return( hypre_StructSolverPETScSetDropTolerance( solver, tol ) );
   }
   else
   {
#ifdef HYPRE_DEBUG
      printf("Warning: attempt to set drop tolerance for solver that does not use it.\n");
#endif
      return(1);
   }
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSetFactorRowSize
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSetFactorRowSize( hypre_StructSolver *solver, int size )
{
   if( hypre_StructSolverSolverType( solver ) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
     return( hypre_StructSolverPETScSetFactorRowSize( solver, size ) );
   }
   else
   {
#ifdef HYPRE_DEBUG
      printf("Warning: attempt to set factor' row size for solver that does not use it.\n");
#endif
      return(1);
   }
}

