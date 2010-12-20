/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/* Include headers for problem and solver data structure */
#include "./PETScMatPilutSolver.h"


/*--------------------------------------------------------------------------
 * HYPRE_NewPETScMatPilutSolver
 *--------------------------------------------------------------------------*/

HYPRE_PETScMatPilutSolver  HYPRE_NewPETScMatPilutSolver( 
                                  MPI_Comm comm,
                                  Mat matrix )
     /* Allocates and Initializes solver structure */
{

   hypre_PETScMatPilutSolver     *solver;
   HYPRE_Int            ierr;

   /* Allocate structure for holding solver data */
   solver = (hypre_PETScMatPilutSolver *) 
            hypre_CTAlloc( hypre_PETScMatPilutSolver, 1);

   /* Initialize components of solver */
   hypre_PETScMatPilutSolverComm(solver) = comm;

   hypre_PETScMatPilutSolverMatrix(solver) = matrix;

   /* DistributedMatrixPilutSolver */
   ierr =
      HYPRE_NewDistributedMatrixPilutSolver( comm, NULL,
        &( hypre_PETScMatPilutSolverDistributedSolver( solver )  ) );

   /* Return created structure to calling routine */
   return( (HYPRE_PETScMatPilutSolver) solver );

}

/*--------------------------------------------------------------------------
 * HYPRE_FreePETScMatPilutSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_FreePETScMatPilutSolver ( 
                  HYPRE_PETScMatPilutSolver in_ptr )
{
  HYPRE_Int ierr=0;

   hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;

   /* Distributed Matrix was created not by user but by 
      HYPRE_ConvertPETScMatrixToDistributedMatrix, so we must free it */

  ierr = HYPRE_DistributedMatrixLimitedDestroy( HYPRE_DistributedMatrixPilutSolverGetMatrix( 
     hypre_PETScMatPilutSolverDistributedSolver ( solver ) ) );

  ierr = HYPRE_FreeDistributedMatrixPilutSolver( 
     hypre_PETScMatPilutSolverDistributedSolver ( solver ) );

  hypre_TFree(solver);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverInitialize ( 
                  HYPRE_PETScMatPilutSolver in_ptr )
{
   hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;
   HYPRE_Int ierr = 0;

   HYPRE_DistributedMatrixPilutSolverInitialize( 
     hypre_PETScMatPilutSolverDistributedSolver ( solver ) );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSetMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSetMatrix( 
                  HYPRE_PETScMatPilutSolver in_ptr,
                  Mat matrix )
{
  HYPRE_Int ierr=0;
  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;

  hypre_PETScMatPilutSolverMatrix( solver ) = matrix;
  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverGetMatrix
 *--------------------------------------------------------------------------*/

Mat
   HYPRE_PETScMatPilutSolverGetMatrix( 
                  HYPRE_PETScMatPilutSolver in_ptr )
{
  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;

  return( hypre_PETScMatPilutSolverMatrix( solver ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSetFactorRowSize
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSetFactorRowSize( 
                  HYPRE_PETScMatPilutSolver in_ptr,
                  HYPRE_Int size )
{
  HYPRE_Int ierr=0;
  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;
  HYPRE_DistributedMatrixPilutSolver distributed_solver =
      hypre_PETScMatPilutSolverDistributedSolver(solver);

  ierr = HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(distributed_solver, size);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSetDropTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSetDropTolerance( 
                  HYPRE_PETScMatPilutSolver in_ptr,
                  double tol )
{
  HYPRE_Int ierr=0;
  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;
  HYPRE_DistributedMatrixPilutSolver distributed_solver =
      hypre_PETScMatPilutSolverDistributedSolver(solver);

  ierr = HYPRE_DistributedMatrixPilutSolverSetDropTolerance(distributed_solver, tol);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSetMaxIts
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSetMaxIts( 
                  HYPRE_PETScMatPilutSolver in_ptr,
                  HYPRE_Int its )
{
  HYPRE_Int ierr=0;
  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;
  HYPRE_DistributedMatrixPilutSolver distributed_solver =
      hypre_PETScMatPilutSolverDistributedSolver(solver);

  ierr = HYPRE_DistributedMatrixPilutSolverSetMaxIts(distributed_solver, its );

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSetup( HYPRE_PETScMatPilutSolver in_ptr,
                            Vec x, Vec b )
{
  HYPRE_Int ierr=0;

  hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;
  HYPRE_DistributedMatrixPilutSolver distributed_solver =
      hypre_PETScMatPilutSolverDistributedSolver(solver);
  HYPRE_DistributedMatrix DistributedPreconditionerMatrix;
  Mat PETScMat = hypre_PETScMatPilutSolverMatrix( solver );


  /**** Executable code ****/

  if( !PETScMat ) return(-1);

  /* Convert the matrix into format suitable for
        the HYPRE_DistributedMatrixPilutSolver preconditioner */

  ierr = HYPRE_ConvertPETScMatrixToDistributedMatrix( PETScMat,
						    &DistributedPreconditionerMatrix );
  if (ierr) return(ierr);

  distributed_solver = 
        hypre_PETScMatPilutSolverDistributedSolver( solver );

  ierr = HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver,
                DistributedPreconditionerMatrix );

  /* Complete setup of distributed_solver structure with computation of
         preconditioner, etc. */
  ierr = HYPRE_DistributedMatrixPilutSolverSetup ( distributed_solver );

  return(ierr);
  
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverApply
 *   Exists solely because PETSc's PCApply function takes b and x in opposite
 *   order than HYPRE_PETScMatPilutSolverSolve. Just a wrapper to the latter.
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverApply( HYPRE_PETScMatPilutSolver in_ptr,
                                           Vec b, Vec x )
{
   HYPRE_Int ierr=0;
   ierr = HYPRE_PETScMatPilutSolverSolve(
       in_ptr, x, b );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScMatPilutSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PETScMatPilutSolverSolve( HYPRE_PETScMatPilutSolver in_ptr,
                                           Vec x, Vec b )
{
   HYPRE_Int ierr=0, size;
   double    *x_vals, *b_vals;

   hypre_PETScMatPilutSolver *solver = 
      (hypre_PETScMatPilutSolver *) in_ptr;

   ierr = VecGetArray( x, &x_vals ); CHKERRA(ierr);
   ierr = VecGetLocalSize( x, &size ); CHKERRA(ierr);

   ierr = VecGetArray( b, &b_vals ); CHKERRA(ierr);
   ierr = VecGetLocalSize( b, &size ); CHKERRA(ierr);

   ierr = HYPRE_DistributedMatrixPilutSolverSolve(
       hypre_PETScMatPilutSolverDistributedSolver( solver ), x_vals, b_vals );

   return(ierr);
}

