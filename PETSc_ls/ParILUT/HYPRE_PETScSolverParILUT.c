/* Include headers for problem and solver data structure */
#include "./PETScSolverParILUT.h"


/*--------------------------------------------------------------------------
 * HYPRE_NewPETScSolverParILUT
 *--------------------------------------------------------------------------*/

HYPRE_PETScSolverParILUT  HYPRE_NewPETScSolverParILUT( 
                                  MPI_Comm comm )
     /* Allocates and Initializes solver structure */
{

   hypre_PETScSolverParILUT     *solver;
   int            ierr;

   /* Allocate structure for holding solver data */
   solver = (hypre_PETScSolverParILUT *) 
            hypre_CTAlloc( hypre_PETScSolverParILUT, 1);

   /* Initialize components of solver */
   hypre_PETScSolverParILUTComm(solver) = comm;

   hypre_PETScSolverParILUTSles(solver) = PETSC_NULL;
   hypre_PETScSolverParILUTSlesOwner(solver) = ParILUTUser;

   hypre_PETScSolverParILUTSystemMatrix(solver) = NULL;
   hypre_PETScSolverParILUTPreconditionerMatrix(solver) = NULL;

   /* PETScMatPilutSolver */
   hypre_PETScSolverParILUTPETScMatPilutSolver( solver ) =
      HYPRE_NewPETScMatPilutSolver( comm, PETSC_NULL );

   /* Return created structure to calling routine */
   return( (HYPRE_PETScSolverParILUT) solver );

}

/*--------------------------------------------------------------------------
 * HYPRE_FreePETScSolverParILUT
 *--------------------------------------------------------------------------*/

int HYPRE_FreePETScSolverParILUT ( 
                  HYPRE_PETScSolverParILUT in_ptr )
{
  int ierr=0;

   hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  if( hypre_PETScSolverParILUTSlesOwner( solver ) == ParILUTLibrary )
  {  
     SLESDestroy(hypre_PETScSolverParILUTSles( solver ));
  }

  ierr = HYPRE_FreePETScMatPilutSolver( 
     hypre_PETScSolverParILUTPETScMatPilutSolver ( solver ) );

  hypre_TFree(solver);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTInitialize
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTInitialize ( 
                  HYPRE_PETScSolverParILUT in_ptr )
{
   int ierr = 0;
   hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

   HYPRE_PETScMatPilutSolverInitialize( 
     hypre_PETScSolverParILUTPETScMatPilutSolver ( solver ) );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetSles
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetSystemSles( 
                  HYPRE_PETScSolverParILUT in_ptr,
                  SLES Sles)
{
  int ierr=0;
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  hypre_PETScSolverParILUTSles( solver ) = Sles;
  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetSystemMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetSystemMatrix( 
                  HYPRE_PETScSolverParILUT in_ptr,
                  Mat matrix )
{
  int ierr=0;
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  hypre_PETScSolverParILUTSystemMatrix( solver ) = matrix;
  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetPreconditionerMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetPreconditionerMatrix( 
                  HYPRE_PETScSolverParILUT in_ptr,
                  Mat matrix )
{
  int ierr=0;
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  hypre_PETScSolverParILUTPreconditionerMatrix( solver ) = matrix;
  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTGetSystemMatrix
 *--------------------------------------------------------------------------*/

Mat
   HYPRE_PETScSolverParILUTGetSystemMatrix( 
                  HYPRE_PETScSolverParILUT in_ptr )
{
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  return( hypre_PETScSolverParILUTSystemMatrix( solver ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTGetPreconditionerMatrix
 *--------------------------------------------------------------------------*/

Mat
   HYPRE_PETScSolverParILUTGetPreconditionerMatrix( 
                  HYPRE_PETScSolverParILUT in_ptr )
{
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;

  return( hypre_PETScSolverParILUTPreconditionerMatrix( solver ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetFactorRowSize
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetFactorRowSize( 
                  HYPRE_PETScSolverParILUT in_ptr,
                  int size )
{
  int ierr=0;
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;
  HYPRE_PETScMatPilutSolver distributed_solver =
      hypre_PETScSolverParILUTPETScMatPilutSolver(solver);

  HYPRE_PETScMatPilutSolverSetFactorRowSize(distributed_solver, size);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetDropTolerance
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSetDropTolerance( 
                  HYPRE_PETScSolverParILUT in_ptr,
                  double tol )
{
  int ierr=0;
  hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;
  HYPRE_PETScMatPilutSolver distributed_solver =
      hypre_PETScSolverParILUTPETScMatPilutSolver(solver);

  HYPRE_PETScMatPilutSolverSetDropTolerance(distributed_solver, tol);

  return(ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSetup
 *--------------------------------------------------------------------------*/

/* In separate file */

/*--------------------------------------------------------------------------
 * HYPRE_PETScSolverParILUTSolve
 *--------------------------------------------------------------------------*/

int HYPRE_PETScSolverParILUTSolve( HYPRE_PETScSolverParILUT in_ptr,
                                           Vec x, Vec b )
{
   int ierr = 0;

   hypre_PETScSolverParILUT *solver = 
      (hypre_PETScSolverParILUT *) in_ptr;


   ierr = SLESSolve(
       hypre_PETScSolverParILUTSles( solver ), b, x, 
       &(hypre_PETScSolverParILUTNumIts(solver)) );

   return(ierr);
}

