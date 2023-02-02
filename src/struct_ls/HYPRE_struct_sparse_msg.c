/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructSparseMSG interface
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_SparseMSGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x      )
{
   return ( hypre_SparseMSGSetup( (void *) solver,
                                  (hypre_StructMatrix *) A,
                                  (hypre_StructVector *) b,
                                  (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSolve( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x      )
{
   return ( hypre_SparseMSGSolve( (void *) solver,
                                  (hypre_StructMatrix *) A,
                                  (hypre_StructVector *) b,
                                  (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetTol( HYPRE_StructSolver solver,
                             HYPRE_Real         tol    )
{
   return ( hypre_SparseMSGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetMaxIter( HYPRE_StructSolver solver,
                                 HYPRE_Int          max_iter  )
{
   return ( hypre_SparseMSGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetJump( HYPRE_StructSolver solver,
                              HYPRE_Int              jump )
{
   return ( hypre_SparseMSGSetJump( (void *) solver, jump ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetRelChange( HYPRE_StructSolver solver,
                                   HYPRE_Int          rel_change  )
{
   return ( hypre_SparseMSGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetRelaxType( HYPRE_StructSolver solver,
                                   HYPRE_Int          relax_type )
{
   return ( hypre_SparseMSGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_StructSparseMSGSetJacobiWeight(HYPRE_StructSolver solver,
                                     HYPRE_Real         weight)
{
   return ( hypre_SparseMSGSetJacobiWeight( (void *) solver, weight) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetNumPreRelax( HYPRE_StructSolver solver,
                                     HYPRE_Int          num_pre_relax )
{
   return ( hypre_SparseMSGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetNumPostRelax( HYPRE_StructSolver solver,
                                      HYPRE_Int          num_post_relax )
{
   return ( hypre_SparseMSGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetNumFineRelax( HYPRE_StructSolver solver,
                                      HYPRE_Int          num_fine_relax )
{
   return ( hypre_SparseMSGSetNumFineRelax( (void *) solver, num_fine_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetLogging( HYPRE_StructSolver solver,
                                 HYPRE_Int          logging )
{
   return ( hypre_SparseMSGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGSetPrintLevel( HYPRE_StructSolver solver,
                                    HYPRE_Int        print_level )
{
   return ( hypre_SparseMSGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGGetNumIterations( HYPRE_StructSolver  solver,
                                       HYPRE_Int          *num_iterations )
{
   return ( hypre_SparseMSGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSparseMSGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                   HYPRE_Real         *norm   )
{
   return ( hypre_SparseMSGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

