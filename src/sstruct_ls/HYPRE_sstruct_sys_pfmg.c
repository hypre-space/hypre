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

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_SysPFMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysPFMGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysPFMGSetup( HYPRE_SStructSolver  solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysPFMGSetup( (void *) solver,
                               (hypre_SStructMatrix *) A,
                               (hypre_SStructVector *) b,
                               (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysPFMGSolve( HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysPFMGSolve( (void *) solver,
                            (hypre_SStructMatrix *) A,
                            (hypre_SStructVector *) b,
                            (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetTol( HYPRE_SStructSolver solver,
                            HYPRE_Real         tol    )
{
   return( hypre_SysPFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetMaxIter( HYPRE_SStructSolver solver,
                                HYPRE_Int          max_iter  )
{
   return( hypre_SysPFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetRelChange( HYPRE_SStructSolver solver,
                                  HYPRE_Int          rel_change  )
{
   return( hypre_SysPFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysPFMGSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysPFMGSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetRelaxType( HYPRE_SStructSolver solver,
                                  HYPRE_Int          relax_type )
{
   return( hypre_SysPFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
HYPRE_Int
HYPRE_SStructSysPFMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                    HYPRE_Real          weight)
{
   return( hypre_SysPFMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetNumPreRelax( HYPRE_SStructSolver solver,
                                    HYPRE_Int          num_pre_relax )
{
   return( hypre_SysPFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetNumPostRelax( HYPRE_SStructSolver solver,
                                     HYPRE_Int          num_post_relax )
{
   return( hypre_SysPFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetSkipRelax( HYPRE_SStructSolver solver,
                                  HYPRE_Int          skip_relax )
{
   return( hypre_SysPFMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetDxyz( HYPRE_SStructSolver  solver,
                         HYPRE_Real         *dxyz   )
{
   return( hypre_SysPFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetLogging( HYPRE_SStructSolver solver,
                                HYPRE_Int          logging )
{
   return( hypre_SysPFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetPrintLevel( HYPRE_SStructSolver solver,
                                HYPRE_Int         print_level )
{
   return( hypre_SysPFMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGGetNumIterations( HYPRE_SStructSolver  solver,
                                      HYPRE_Int          *num_iterations )
{
   return( hypre_SysPFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  HYPRE_Real         *norm   )
{
   return( hypre_SysPFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

