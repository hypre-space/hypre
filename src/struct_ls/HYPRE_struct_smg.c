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
HYPRE_StructSMGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_SMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_SMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return ( hypre_SMGSetup( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return ( hypre_SMGSolve( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetMemoryUse( HYPRE_StructSolver solver,
                             HYPRE_Int          memory_use )
{
   return ( hypre_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

HYPRE_Int
HYPRE_StructSMGGetMemoryUse( HYPRE_StructSolver solver,
                             HYPRE_Int        * memory_use )
{
   return ( hypre_SMGGetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetTol( HYPRE_StructSolver solver,
                       HYPRE_Real         tol    )
{
   return ( hypre_SMGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_StructSMGGetTol( HYPRE_StructSolver solver,
                       HYPRE_Real       * tol    )
{
   return ( hypre_SMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int          max_iter  )
{
   return ( hypre_SMGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_StructSMGGetMaxIter( HYPRE_StructSolver solver,
                           HYPRE_Int        * max_iter  )
{
   return ( hypre_SMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetRelChange( HYPRE_StructSolver solver,
                             HYPRE_Int          rel_change  )
{
   return ( hypre_SMGSetRelChange( (void *) solver, rel_change ) );
}

HYPRE_Int
HYPRE_StructSMGGetRelChange( HYPRE_StructSolver solver,
                             HYPRE_Int        * rel_change  )
{
   return ( hypre_SMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_SMGSetZeroGuess( (void *) solver, 1 ) );
}

HYPRE_Int
HYPRE_StructSMGGetZeroGuess( HYPRE_StructSolver solver,
                             HYPRE_Int * zeroguess )
{
   return ( hypre_SMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_SMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetNumPreRelax( HYPRE_StructSolver solver,
                               HYPRE_Int          num_pre_relax )
{
   return ( hypre_SMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

HYPRE_Int
HYPRE_StructSMGGetNumPreRelax( HYPRE_StructSolver solver,
                               HYPRE_Int        * num_pre_relax )
{
   return ( hypre_SMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetNumPostRelax( HYPRE_StructSolver solver,
                                HYPRE_Int          num_post_relax )
{
   return ( hypre_SMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

HYPRE_Int
HYPRE_StructSMGGetNumPostRelax( HYPRE_StructSolver solver,
                                HYPRE_Int        * num_post_relax )
{
   return ( hypre_SMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int          logging )
{
   return ( hypre_SMGSetLogging( (void *) solver, logging) );
}

HYPRE_Int
HYPRE_StructSMGGetLogging( HYPRE_StructSolver solver,
                           HYPRE_Int        * logging )
{
   return ( hypre_SMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGSetPrintLevel( HYPRE_StructSolver solver,
                              HYPRE_Int  print_level )
{
   return ( hypre_SMGSetPrintLevel( (void *) solver, print_level) );
}

HYPRE_Int
HYPRE_StructSMGGetPrintLevel( HYPRE_StructSolver solver,
                              HYPRE_Int      * print_level )
{
   return ( hypre_SMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGGetNumIterations( HYPRE_StructSolver  solver,
                                 HYPRE_Int          *num_iterations )
{
   return ( hypre_SMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructSMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             HYPRE_Real         *norm   )
{
   return ( hypre_SMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int
HYPRE_StructSMGSetDeviceLevel( HYPRE_StructSolver  solver,
                               HYPRE_Int   device_level  )
{
   return (hypre_StructSMGSetDeviceLevel( (void *) solver, device_level ));
}
#endif
