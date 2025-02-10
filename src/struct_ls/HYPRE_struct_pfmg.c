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
HYPRE_StructPFMGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_PFMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_PFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetup( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return ( hypre_PFMGSetup( (void *) solver,
                             (hypre_StructMatrix *) A,
                             (hypre_StructVector *) b,
                             (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSolve( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return ( hypre_PFMGSolve( (void *) solver,
                             (hypre_StructMatrix *) A,
                             (hypre_StructVector *) b,
                             (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetTol( HYPRE_StructSolver solver,
                        HYPRE_Real         tol    )
{
   return ( hypre_PFMGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_StructPFMGGetTol( HYPRE_StructSolver solver,
                        HYPRE_Real       * tol    )
{
   return ( hypre_PFMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetMaxIter( HYPRE_StructSolver solver,
                            HYPRE_Int          max_iter  )
{
   return ( hypre_PFMGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_StructPFMGGetMaxIter( HYPRE_StructSolver solver,
                            HYPRE_Int        * max_iter  )
{
   return ( hypre_PFMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetMaxLevels( HYPRE_StructSolver solver,
                              HYPRE_Int          max_levels  )
{
   return ( hypre_PFMGSetMaxLevels( (void *) solver, max_levels ) );
}

HYPRE_Int
HYPRE_StructPFMGGetMaxLevels( HYPRE_StructSolver solver,
                              HYPRE_Int        * max_levels  )
{
   return ( hypre_PFMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetRelChange( HYPRE_StructSolver solver,
                              HYPRE_Int          rel_change  )
{
   return ( hypre_PFMGSetRelChange( (void *) solver, rel_change ) );
}

HYPRE_Int
HYPRE_StructPFMGGetRelChange( HYPRE_StructSolver solver,
                              HYPRE_Int        * rel_change  )
{
   return ( hypre_PFMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_PFMGSetZeroGuess( (void *) solver, 1 ) );
}

HYPRE_Int
HYPRE_StructPFMGGetZeroGuess( HYPRE_StructSolver solver,
                              HYPRE_Int * zeroguess )
{
   return ( hypre_PFMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return ( hypre_PFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * GetJacobiWeight will not return the actual weight
 * if SetJacobiWeight has not been called.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetRelaxType( HYPRE_StructSolver solver,
                              HYPRE_Int          relax_type )
{
   return ( hypre_PFMGSetRelaxType( (void *) solver, relax_type) );
}

HYPRE_Int
HYPRE_StructPFMGGetRelaxType( HYPRE_StructSolver solver,
                              HYPRE_Int        * relax_type )
{
   return ( hypre_PFMGGetRelaxType( (void *) solver, relax_type) );
}

HYPRE_Int
HYPRE_StructPFMGSetJacobiWeight(HYPRE_StructSolver solver,
                                HYPRE_Real         weight)
{
   return ( hypre_PFMGSetJacobiWeight( (void *) solver, weight) );
}
HYPRE_Int
HYPRE_StructPFMGGetJacobiWeight(HYPRE_StructSolver solver,
                                HYPRE_Real        *weight)
{
   return ( hypre_PFMGGetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetRAPType( HYPRE_StructSolver solver,
                            HYPRE_Int          rap_type )
{
   return ( hypre_PFMGSetRAPType( (void *) solver, rap_type) );
}

HYPRE_Int
HYPRE_StructPFMGGetRAPType( HYPRE_StructSolver solver,
                            HYPRE_Int        * rap_type )
{
   return ( hypre_PFMGGetRAPType( (void *) solver, rap_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetNumPreRelax( HYPRE_StructSolver solver,
                                HYPRE_Int          num_pre_relax )
{
   return ( hypre_PFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

HYPRE_Int
HYPRE_StructPFMGGetNumPreRelax( HYPRE_StructSolver solver,
                                HYPRE_Int        * num_pre_relax )
{
   return ( hypre_PFMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetNumPostRelax( HYPRE_StructSolver solver,
                                 HYPRE_Int          num_post_relax )
{
   return ( hypre_PFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

HYPRE_Int
HYPRE_StructPFMGGetNumPostRelax( HYPRE_StructSolver solver,
                                 HYPRE_Int        * num_post_relax )
{
   return ( hypre_PFMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetSkipRelax( HYPRE_StructSolver solver,
                              HYPRE_Int          skip_relax )
{
   return ( hypre_PFMGSetSkipRelax( (void *) solver, skip_relax) );
}

HYPRE_Int
HYPRE_StructPFMGGetSkipRelax( HYPRE_StructSolver solver,
                              HYPRE_Int        * skip_relax )
{
   return ( hypre_PFMGGetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetDxyz( HYPRE_StructSolver  solver,
                         HYPRE_Real         *dxyz   )
{
   return ( hypre_PFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetLogging( HYPRE_StructSolver solver,
                            HYPRE_Int          logging )
{
   return ( hypre_PFMGSetLogging( (void *) solver, logging) );
}

HYPRE_Int
HYPRE_StructPFMGGetLogging( HYPRE_StructSolver solver,
                            HYPRE_Int        * logging )
{
   return ( hypre_PFMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGSetPrintLevel( HYPRE_StructSolver solver,
                               HYPRE_Int            print_level )
{
   return ( hypre_PFMGSetPrintLevel( (void *) solver, print_level) );
}

HYPRE_Int
HYPRE_StructPFMGGetPrintLevel( HYPRE_StructSolver solver,
                               HYPRE_Int          * print_level )
{
   return ( hypre_PFMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGGetNumIterations( HYPRE_StructSolver  solver,
                                  HYPRE_Int          *num_iterations )
{
   return ( hypre_PFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructPFMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                              HYPRE_Real         *norm   )
{
   return ( hypre_PFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

#if 0 //defined(HYPRE_USING_GPU)
HYPRE_Int
HYPRE_StructPFMGSetDeviceLevel( HYPRE_StructSolver  solver,
                                HYPRE_Int   device_level  )
{
   return ( hypre_PFMGSetDeviceLevel( (void *) solver, device_level ) );
}
#endif
