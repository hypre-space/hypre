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
 * Member functions for hypre_StructSolver class for PETSc storage scheme.
 *
 *****************************************************************************/
#include "./headers.h"

/* include PETSc linear solver headers */
#include "sles.h"

/*--------------------------------------------------------------------------
 * hypre_FreeStructSolverPETSc
 *   Internal routine for freeing a solver stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructSolverPETSc( hypre_StructSolver *struct_solver )
{

   HYPRE_FreePETScSolverParILUT
      ( (HYPRE_PETScSolverParILUT) hypre_StructSolverData( struct_solver ) );

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverInitializePETSc
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverInitializePETSc( hypre_StructSolver *struct_solver )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSetupPETSc
 *   Internal routine for setting up a solver for a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSetupPETSc( hypre_StructSolver *struct_solver )
{
   HYPRE_StructMatrix matrix=hypre_StructSolverMatrix(struct_solver);
   HYPRE_StructVector soln=hypre_StructSolverSoln(struct_solver);
   HYPRE_StructVector rhs=hypre_StructSolverRhs(struct_solver);
   HYPRE_PETScSolverParILUT solver_data;
   Mat         Petsc_matrix;
   Vec         Petsc_soln, Petsc_rhs;

   int  ierr;


   solver_data = HYPRE_NewPETScSolverParILUT
      ( hypre_StructSolverContext( struct_solver ));
   hypre_StructSolverData( struct_solver ) = solver_data;

   Petsc_matrix = (Mat) HYPRE_StructMatrixGetData( matrix );
   HYPRE_PETScSolverParILUTSetSystemMatrix( solver_data, Petsc_matrix );

   Petsc_soln = (Vec) HYPRE_StructVectorGetData( soln );
   Petsc_rhs = (Vec) HYPRE_StructVectorGetData( rhs );
   
   ierr = HYPRE_PETScSolverParILUTSetup( 
      solver_data, Petsc_soln, Petsc_rhs );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverSolvePETSc
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverSolvePETSc( hypre_StructSolver *struct_solver )
{
   HYPRE_StructVector  soln=hypre_StructSolverSoln(struct_solver);
   HYPRE_StructVector  rhs=hypre_StructSolverRhs(struct_solver);
   HYPRE_PETScSolverParILUT solver_data=hypre_StructSolverData( struct_solver );
   Vec        Petsc_soln, Petsc_rhs;

   int  ierr;


   Petsc_soln = (Vec) HYPRE_StructVectorGetData( soln );
   Petsc_rhs = (Vec) HYPRE_StructVectorGetData( rhs );
   
   ierr = HYPRE_PETScSolverParILUTSolve( 
      solver_data, Petsc_soln, Petsc_rhs );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_StructSolverPETScSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
hypre_StructSolverPETScSetDropTolerance( hypre_StructSolver *struct_solver, 
                                       double tol )
{
   int ierr=0;

   ierr = HYPRE_PETScSolverParILUTSetDropTolerance
      ( (HYPRE_PETScSolverParILUT) hypre_StructSolverData( struct_solver ), tol );

   return(ierr);
}


