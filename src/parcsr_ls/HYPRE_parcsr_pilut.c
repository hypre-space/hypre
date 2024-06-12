/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

#include "../distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_types.h"
#include "../distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_protos.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"

/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(comm);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_NewDistributedMatrixPilutSolver( comm, NULL,
                                          (HYPRE_DistributedMatrixPilutSolver *) solver);

   HYPRE_DistributedMatrixPilutSolverInitialize(
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutDestroy( HYPRE_Solver solver )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrix mat = HYPRE_DistributedMatrixPilutSolverGetMatrix(
                                    (HYPRE_DistributedMatrixPilutSolver) solver );
   if ( mat ) { HYPRE_DistributedMatrixDestroy( mat ); }

   HYPRE_FreeDistributedMatrixPilutSolver(
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(A);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrix matrix;
   HYPRE_DistributedMatrixPilutSolver distributed_solver =
      (HYPRE_DistributedMatrixPilutSolver) solver;

   HYPRE_ConvertParCSRMatrixToDistributedMatrix(
      A, &matrix );

   HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver, matrix );

   HYPRE_DistributedMatrixPilutSolverSetup( distributed_solver );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   HYPRE_UNUSED_VAR(A);

#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(solver);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_Real *rhs, *soln;

   rhs = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)b ) );
   soln = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)x ) );

   HYPRE_DistributedMatrixPilutSolverSolve(
      (HYPRE_DistributedMatrixPilutSolver) solver,
      soln, rhs );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter  )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(max_iter);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetDropTolerance( HYPRE_Solver solver,
                                   HYPRE_Real   tol    )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(tol);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
      (HYPRE_DistributedMatrixPilutSolver) solver, tol );

   return hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFactorRowSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver,
                                   HYPRE_Int       size    )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(size);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (HYPRE_DistributedMatrixPilutSolver) solver, size );

   return hypre_error_flag;
#endif
}

HYPRE_Int
HYPRE_ParCSRPilutSetLogging( HYPRE_Solver solver,
                             HYPRE_Int    logging    )
{
#ifdef HYPRE_MIXEDINT
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(logging);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return hypre_error_flag;
#else

   HYPRE_DistributedMatrixPilutSolverSetLogging(
      (HYPRE_DistributedMatrixPilutSolver) solver, logging );

   return hypre_error_flag;
#endif
}
