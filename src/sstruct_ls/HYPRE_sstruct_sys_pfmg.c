/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_SStructSysPFMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_SysPFMGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysPFMGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetup
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
 * HYPRE_SStructSysPFMGSolve
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
 * HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetTol( HYPRE_SStructSolver solver,
                            double             tol    )
{
   return( hypre_SysPFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetMaxIter( HYPRE_SStructSolver solver,
                                HYPRE_Int          max_iter  )
{
   return( hypre_SysPFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetRelChange( HYPRE_SStructSolver solver,
                                  HYPRE_Int          rel_change  )
{
   return( hypre_SysPFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysPFMGSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysPFMGSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetRelaxType( HYPRE_SStructSolver solver,
                                  HYPRE_Int          relax_type )
{
   return( hypre_SysPFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
HYPRE_Int
HYPRE_SStructSysPFMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                    double              weight)
{
   return( hypre_SysPFMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetNumPreRelax( HYPRE_SStructSolver solver,
                                    HYPRE_Int          num_pre_relax )
{
   return( hypre_SysPFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetNumPostRelax( HYPRE_SStructSolver solver,
                                     HYPRE_Int          num_post_relax )
{
   return( hypre_SysPFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetSkipRelax( HYPRE_SStructSolver solver,
                                  HYPRE_Int          skip_relax )
{
   return( hypre_SysPFMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetDxyz( HYPRE_SStructSolver  solver,
                         double             *dxyz   )
{
   return( hypre_SysPFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetLogging( HYPRE_SStructSolver solver,
                                HYPRE_Int          logging )
{
   return( hypre_SysPFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
HYPRE_SStructSysPFMGSetPrintLevel
*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGSetPrintLevel( HYPRE_SStructSolver solver,
                                HYPRE_Int         print_level )
{
   return( hypre_SysPFMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGGetNumIterations( HYPRE_SStructSolver  solver,
                                      HYPRE_Int          *num_iterations )
{
   return( hypre_SysPFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  double             *norm   )
{
   return( hypre_SysPFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

