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
 * Member functions for hypre_StructInterfaceSolver class for PETSc storage scheme.
 *
 *****************************************************************************/
#include "./headers.h"

#ifdef PETSC_AVAILABLE
/* include PETSc linear solver headers */
#include "sles.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_FreeStructInterfaceSolverPETSc
 *   Internal routine for freeing a solver stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_FreeStructInterfaceSolverPETSc( hypre_StructInterfaceSolver *struct_solver )
{

#ifdef PETSC_AVAILABLE
   HYPRE_FreePETScSolverParILUT
      ( (HYPRE_PETScSolverParILUT) hypre_StructInterfaceSolverData( struct_solver ) );

   return(0);

#else
   return(-1);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverInitializePETSc
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverInitializePETSc( hypre_StructInterfaceSolver *struct_solver )
{
#ifdef PETSC_AVAILABLE
   int ierr = 0;

   hypre_StructInterfaceSolverData( struct_solver ) = HYPRE_NewPETScSolverParILUT
      ( hypre_StructInterfaceSolverContext( struct_solver ));

   return(ierr);

#else
   return(-1);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSetupPETSc
 *   Internal routine for setting up a solver for a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSetupPETSc( hypre_StructInterfaceSolver *struct_solver )
{
#ifdef PETSC_AVAILABLE
   HYPRE_StructInterfaceMatrix matrix=hypre_StructInterfaceSolverMatrix(struct_solver);
   HYPRE_StructInterfaceVector soln=hypre_StructInterfaceSolverSoln(struct_solver);
   HYPRE_StructInterfaceVector rhs=hypre_StructInterfaceSolverRhs(struct_solver);
   HYPRE_PETScSolverParILUT solver_data;
   Mat         Petsc_matrix;
   Vec         Petsc_soln, Petsc_rhs;

   int  ierr;


   solver_data = hypre_StructInterfaceSolverData( struct_solver );

   Petsc_matrix = (Mat) HYPRE_StructInterfaceMatrixGetData( matrix );
   HYPRE_PETScSolverParILUTSetSystemMatrix( solver_data, Petsc_matrix );

   Petsc_soln = (Vec) HYPRE_StructInterfaceVectorGetData( soln );
   Petsc_rhs = (Vec) HYPRE_StructInterfaceVectorGetData( rhs );
   
   ierr = HYPRE_PETScSolverParILUTSetup( 
      solver_data, Petsc_soln, Petsc_rhs );

   return(ierr);

#else
   return(-1);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverSolvePETSc
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverSolvePETSc( hypre_StructInterfaceSolver *struct_solver )
{
#ifdef PETSC_AVAILABLE
   HYPRE_StructInterfaceVector  soln=hypre_StructInterfaceSolverSoln(struct_solver);
   HYPRE_StructInterfaceVector  rhs=hypre_StructInterfaceSolverRhs(struct_solver);
   HYPRE_PETScSolverParILUT solver_data=hypre_StructInterfaceSolverData( struct_solver );
   Vec        Petsc_soln, Petsc_rhs;

   int  ierr;


   Petsc_soln = (Vec) HYPRE_StructInterfaceVectorGetData( soln );
   Petsc_rhs = (Vec) HYPRE_StructInterfaceVectorGetData( rhs );
   
   ierr = HYPRE_PETScSolverParILUTSolve( 
      solver_data, Petsc_soln, Petsc_rhs );

   return(ierr);

#else
   return(-1);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverPETScSetDropTolerance
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverPETScSetDropTolerance( hypre_StructInterfaceSolver *struct_solver, 
                                       double tol )
{
#ifdef PETSC_AVAILABLE
   int ierr=0;

   ierr = HYPRE_PETScSolverParILUTSetDropTolerance
      ( (HYPRE_PETScSolverParILUT) hypre_StructInterfaceSolverData( struct_solver ), tol );

   return(ierr);

#else
   return(-1);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_StructInterfaceSolverPETScSetFactorRowSize
 *--------------------------------------------------------------------------*/

int 
hypre_StructInterfaceSolverPETScSetFactorRowSize( hypre_StructInterfaceSolver *struct_solver, 
                                       int size )
{
#ifdef PETSC_AVAILABLE
   int ierr=0;

   ierr = HYPRE_PETScSolverParILUTSetFactorRowSize
      ( (HYPRE_PETScSolverParILUT) hypre_StructInterfaceSolverData( struct_solver ), size );

   return(ierr);

#else
   return(-1);
#endif
}


