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
 * Member functions for hypre_StructInterfaceSolver class.
 *
 *****************************************************************************/

#include "./headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructInterfaceSolver
 *--------------------------------------------------------------------------*/

hypre_StructInterfaceSolver *
hypre_NewStructInterfaceSolver( MPI_Comm     context,
		      HYPRE_StructGrid    grid,
		      HYPRE_StructStencil stencil )
{
   hypre_StructInterfaceSolver    *struct_solver;


   struct_solver = (hypre_StructInterfaceSolver *) hypre_CTAlloc(hypre_StructInterfaceSolver,1);

   hypre_StructInterfaceSolverContext(struct_solver) = context;
   hypre_StructInterfaceSolverStructGrid(struct_solver)    = grid;
   hypre_StructInterfaceSolverStructStencil(struct_solver) = stencil;

   hypre_StructInterfaceSolverMatrix(struct_solver) = NULL;
   hypre_StructInterfaceSolverSoln(struct_solver) = NULL;
   hypre_StructInterfaceSolverRhs(struct_solver) = NULL;

   hypre_StructInterfaceSolverData(struct_solver) = NULL;

   /* set defaults */
   hypre_StructInterfaceSolverSolverType(struct_solver) = HYPRE_PETSC_MAT_PARILUT_SOLVER;

   return struct_solver;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceSolver
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceSolver( hypre_StructInterfaceSolver *struct_solver )
{
   if ( hypre_StructInterfaceSolverSolverType(struct_solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
      hypre_FreeStructInterfaceSolverPETSc( struct_solver );
   else
      return(-1);

   hypre_TFree(struct_solver);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSetType
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSetType( hypre_StructInterfaceSolver *solver, int type )
{
   if( type == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
      hypre_StructInterfaceSolverSolverType(solver) = type;
      return(0);
   }
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverInitialize( hypre_StructInterfaceSolver *solver )
{

  if ( hypre_StructInterfaceSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructInterfaceSolverInitializePETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSetup
 *   Internal routine for setting up solver data like factorizations etc.
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSetup( hypre_StructInterfaceSolver *solver, HYPRE_StructInterfaceMatrix matrix,
                        HYPRE_StructInterfaceVector soln, HYPRE_StructInterfaceVector rhs )
{
  hypre_StructInterfaceSolverMatrix( solver ) = matrix;
  hypre_StructInterfaceSolverSoln( solver ) = soln;
  hypre_StructInterfaceSolverRhs( solver ) = rhs;

  if ( hypre_StructInterfaceSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructInterfaceSolverSetupPETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSolve
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSolve( hypre_StructInterfaceSolver *solver )
{
  if ( hypre_StructInterfaceSolverSolverType(solver) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
     return( hypre_StructInterfaceSolverSolvePETSc( solver ) );
  else
     return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSetDropTolerance( hypre_StructInterfaceSolver *solver, double tol )
{
   if( hypre_StructInterfaceSolverSolverType( solver ) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
     return( hypre_StructInterfaceSolverPETScSetDropTolerance( solver, tol ) );
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
 * hypre_StructInterfaceSolverSetFactorRowSize
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSetFactorRowSize( hypre_StructInterfaceSolver *solver, int size )
{
   if( hypre_StructInterfaceSolverSolverType( solver ) == HYPRE_PETSC_MAT_PARILUT_SOLVER )
   {
     return( hypre_StructInterfaceSolverPETScSetFactorRowSize( solver, size ) );
   }
   else
   {
#ifdef HYPRE_DEBUG
      printf("Warning: attempt to set factor' row size for solver that does not use it.\n");
#endif
      return(1);
   }
}

