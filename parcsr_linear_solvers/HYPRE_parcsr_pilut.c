/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRPilut interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./HYPRE_parcsr_ls.h"

#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"

#include "../distributed_linear_solvers/pilut/HYPRE_DistributedMatrixPilutSolver_types.h"
#include "../distributed_linear_solvers/pilut/HYPRE_DistributedMatrixPilutSolver_protos.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_matrix_vector/vector.h"
#include "../parcsr_matrix_vector/par_vector.h"


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutInitialize( MPI_Comm comm, HYPRE_Solver *solver )
{
   int ierr = 0;
   
   *solver = (HYPRE_Solver) HYPRE_NewDistributedMatrixPilutSolver( comm, NULL);

   ierr = HYPRE_DistributedMatrixPilutSolverInitialize( 
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutFinalize( HYPRE_Solver solver )
{
   int ierr = 0;

   ierr = HYPRE_LimitedFreeDistributedMatrix( 
      HYPRE_DistributedMatrixPilutSolverGetMatrix(
         (HYPRE_DistributedMatrixPilutSolver) solver ) );
   if (ierr) return(ierr);

   ierr = HYPRE_FreeDistributedMatrixPilutSolver(
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   int ierr = 0;
   HYPRE_DistributedMatrix *matrix;
   HYPRE_DistributedMatrixPilutSolver distributed_solver = 
      (HYPRE_DistributedMatrixPilutSolver) solver;

   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix(
             A, matrix );
   if (ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver, *matrix );
   if (ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixPilutSolverSetup( distributed_solver );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   int ierr = 0;
   double *rhs, *soln;

   rhs = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)b ) );
   soln = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)x ) );

   ierr = HYPRE_DistributedMatrixPilutSolverSolve(
      (HYPRE_DistributedMatrixPilutSolver) solver,
      soln, rhs );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver,
                        int          max_iter  )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetDropTolerance( HYPRE_Solver solver,
                    double       tol    )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
      (HYPRE_DistributedMatrixPilutSolver) solver, tol );


   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFactorRowSize
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver,
                    int       size    )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (HYPRE_DistributedMatrixPilutSolver) solver, size );


   return( ierr );
}

