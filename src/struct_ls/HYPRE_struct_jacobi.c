/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiCreate( MPI_Comm            comm,
                          HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_JacobiCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_JacobiDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return ( hypre_JacobiSetup( (void *) solver,
                               (hypre_StructMatrix *) A,
                               (hypre_StructVector *) b,
                               (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return ( hypre_JacobiSolve( (void *) solver,
                               (hypre_StructMatrix *) A,
                               (hypre_StructVector *) b,
                               (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver,
                          HYPRE_Real         tol    )
{
   return ( hypre_JacobiSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetTol( HYPRE_StructSolver solver,
                          HYPRE_Real       * tol    )
{
   return ( hypre_JacobiGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver,
                              HYPRE_Int          max_iter  )
{
   return ( hypre_JacobiSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetMaxIter( HYPRE_StructSolver solver,
                              HYPRE_Int        * max_iter  )
{
   return ( hypre_JacobiGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_JacobiSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetZeroGuess( HYPRE_StructSolver solver,
                                HYPRE_Int * zeroguess )
{
   return ( hypre_JacobiGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_JacobiSetZeroGuess( (void *) solver, 0 ) );
}


/* NOT YET IMPLEMENTED */

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver  solver,
                                    HYPRE_Int          *num_iterations )
{
   return ( hypre_JacobiGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                HYPRE_Real         *norm   )
{
   return ( hypre_JacobiGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
