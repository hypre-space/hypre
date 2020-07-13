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
HYPRE_SStructSSAMGCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_SSAMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_SSAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( hypre_SSAMGSetup( (void *) solver,
                             (hypre_SStructMatrix *) A,
                             (hypre_SStructVector *) b,
                             (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x)
{
   return( hypre_SSAMGSolve( (void *) solver,
                             (hypre_SStructMatrix *) A,
                             (hypre_SStructVector *) b,
                             (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetTol( HYPRE_SStructSolver solver,
                          HYPRE_Real          tol )
{
   return( hypre_SSAMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetMaxIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter )
{
   return( hypre_SSAMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetMaxLevels( HYPRE_SStructSolver solver,
                                HYPRE_Int           max_levels )
{
   return( hypre_SSAMGSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetRelChange( HYPRE_SStructSolver solver,
                                HYPRE_Int           rel_change )
{
   return( hypre_SSAMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SSAMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SSAMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetRelaxType( HYPRE_SStructSolver solver,
                                HYPRE_Int           relax_type )
{
   return( hypre_SSAMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetRelaxWeight( HYPRE_SStructSolver solver,
                                  HYPRE_Real          relax_weight )
{
   return( hypre_SSAMGSetRelaxWeight( (void *) solver, relax_weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetNumPreRelax( HYPRE_SStructSolver solver,
                                  HYPRE_Int           num_pre_relax )
{
   return( hypre_SSAMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetNumPostRelax( HYPRE_SStructSolver solver,
                                   HYPRE_Int           num_post_relax )
{
   return( hypre_SSAMGSetNumPosRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetNumCoarseRelax( HYPRE_SStructSolver solver,
                                     HYPRE_Int           num_coarse_relax )
{
   return( hypre_SSAMGSetNumCoarseRelax( (void *) solver, num_coarse_relax ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetDxyz( HYPRE_SStructSolver  solver,
                           HYPRE_Int            nparts,
                           HYPRE_Real         **dxyz   )
{
   return( hypre_SSAMGSetDxyz( (void *) solver, nparts, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetLogging( HYPRE_SStructSolver solver,
                              HYPRE_Int           logging )
{
   return( hypre_SSAMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGSetPrintLevel( HYPRE_SStructSolver solver,
                                 HYPRE_Int           print_level )
{
   return( hypre_SSAMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGGetNumIterations( HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations )
{
   return( hypre_SSAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSSAMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                HYPRE_Real          *norm   )
{
   return( hypre_SSAMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
