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
 * Member functions for zzz_StructSolver class for PETSc storage scheme.
 *
 *****************************************************************************/
#include "headers.h"

/* include PETSc linear solver headers */
#include "sles.h"

/*--------------------------------------------------------------------------
 * zzz_FreeStructSolverPETSc
 *   Internal routine for freeing a solver stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_FreeStructSolverPETSc( zzz_StructSolver *struct_solver )
{

   BlockJacobiILUPcKspFinalize( zzz_StructSolverData( struct_solver ) );

   return(0);
}

/*--------------------------------------------------------------------------
 * zzz_StructSolverSetupPETSc
 *   Internal routine for setting up a solver stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
zzz_StructSolverSetupPETSc( zzz_StructSolver *struct_solver )
{
   zzz_StructMatrix *matrix=zzz_StructSolverMatrix(struct_solver);
   zzz_StructVector *soln=zzz_StructSolverSoln(struct_solver);
   zzz_StructVector *rhs=zzz_StructSolverRhs(struct_solver);
   void        *bj_data;
   Mat         *Petsc_matrix;
   Vec         *Petsc_soln, *Petsc_rhs;

   int  ierr;


   bj_data = BlockJacobiILUPcKspInitialize( (void *) NULL );
   zzz_StructSolverData( struct_solver ) = bj_data;

   Petsc_matrix = (Mat *) zzz_StructMatrixData( matrix );
   Petsc_soln = (Vec *) zzz_StructVectorData( soln );
   Petsc_rhs = (Vec *) zzz_StructVectorData( rhs );
   
   ierr = BlockJacobiILUPcKspSetup( 
      bj_data, *Petsc_matrix, *Petsc_soln, *Petsc_rhs );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * zzz_StructSolverSolvePETSc
 *   Internal routine for solving
 *--------------------------------------------------------------------------*/

int 
zzz_StructSolverSolvePETSc( zzz_StructSolver *struct_solver )
{
   zzz_StructMatrix *matrix=zzz_StructSolverMatrix(struct_solver);
   zzz_StructVector *soln=zzz_StructSolverSoln(struct_solver);
   zzz_StructVector *rhs=zzz_StructSolverRhs(struct_solver);
   void       *bj_data=zzz_StructSolverData( struct_solver );
   Mat        *Petsc_matrix;
   Vec        *Petsc_soln, *Petsc_rhs;

   int  ierr;


   Petsc_matrix = (Mat *) zzz_StructMatrixData( matrix );
   Petsc_soln = (Vec *) zzz_StructVectorData( soln );
   Petsc_rhs = (Vec *) zzz_StructVectorData( rhs );
   
   ierr = BlockJacobiILUPcKspSolve( 
      bj_data, *Petsc_matrix, *Petsc_soln, *Petsc_rhs );

   return(ierr);
}

