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

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_BAMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBAMGDestroy( HYPRE_StructSolver solver )
{
   return( hypre_BAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBAMGSetup( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_BAMGSetup( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructBAMGSolve( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_BAMGSolve( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetTol( HYPRE_StructSolver solver,
                        HYPRE_Real         tol    )
{
   return( hypre_BAMGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_StructBAMGGetTol( HYPRE_StructSolver solver,
                        HYPRE_Real       * tol    )
{
   return( hypre_BAMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetMaxIter( HYPRE_StructSolver solver,
                            HYPRE_Int          max_iter  )
{
   return( hypre_BAMGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_StructBAMGGetMaxIter( HYPRE_StructSolver solver,
                            HYPRE_Int        * max_iter  )
{
   return( hypre_BAMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetMaxLevels( HYPRE_StructSolver solver,
                              HYPRE_Int          max_levels  )
{
   return( hypre_BAMGSetMaxLevels( (void *) solver, max_levels ) );
}

HYPRE_Int
HYPRE_StructBAMGGetMaxLevels( HYPRE_StructSolver solver,
                              HYPRE_Int        * max_levels  )
{
   return( hypre_BAMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetRelChange( HYPRE_StructSolver solver,
                              HYPRE_Int          rel_change  )
{
   return( hypre_BAMGSetRelChange( (void *) solver, rel_change ) );
}

HYPRE_Int
HYPRE_StructBAMGGetRelChange( HYPRE_StructSolver solver,
                              HYPRE_Int        * rel_change  )
{
   return( hypre_BAMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructBAMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_BAMGSetZeroGuess( (void *) solver, 1 ) );
}

HYPRE_Int
HYPRE_StructBAMGGetZeroGuess( HYPRE_StructSolver solver,
                              HYPRE_Int * zeroguess )
{
   return( hypre_BAMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructBAMGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_BAMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * GetJacobiWeight will not return the actual weight
 * if SetJacobiWeight has not been called.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetRelaxType( HYPRE_StructSolver solver,
                              HYPRE_Int          relax_type )
{
   return( hypre_BAMGSetRelaxType( (void *) solver, relax_type) );
}

HYPRE_Int
HYPRE_StructBAMGGetRelaxType( HYPRE_StructSolver solver,
                              HYPRE_Int        * relax_type )
{
   return( hypre_BAMGGetRelaxType( (void *) solver, relax_type) );
}

HYPRE_Int
HYPRE_StructBAMGSetJacobiWeight(HYPRE_StructSolver solver,
                                HYPRE_Real         weight)
{
   return( hypre_BAMGSetJacobiWeight( (void *) solver, weight) );
}
HYPRE_Int
HYPRE_StructBAMGGetJacobiWeight(HYPRE_StructSolver solver,
                                HYPRE_Real        *weight)
{
   return( hypre_BAMGGetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetRAPType( HYPRE_StructSolver solver,
                            HYPRE_Int          rap_type )
{
   return( hypre_BAMGSetRAPType( (void *) solver, rap_type) );
}

HYPRE_Int
HYPRE_StructBAMGGetRAPType( HYPRE_StructSolver solver,
                            HYPRE_Int        * rap_type )
{
   return( hypre_BAMGGetRAPType( (void *) solver, rap_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetNumPreRelax( HYPRE_StructSolver solver,
                                HYPRE_Int          num_pre_relax )
{
   return( hypre_BAMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

HYPRE_Int
HYPRE_StructBAMGGetNumPreRelax( HYPRE_StructSolver solver,
                                HYPRE_Int        * num_pre_relax )
{
   return( hypre_BAMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetNumPostRelax( HYPRE_StructSolver solver,
                                 HYPRE_Int          num_post_relax )
{
   return( hypre_BAMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

HYPRE_Int
HYPRE_StructBAMGGetNumPostRelax( HYPRE_StructSolver solver,
                                 HYPRE_Int        * num_post_relax )
{
   return( hypre_BAMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetSkipRelax( HYPRE_StructSolver solver,
                              HYPRE_Int          skip_relax )
{
   return( hypre_BAMGSetSkipRelax( (void *) solver, skip_relax) );
}

HYPRE_Int
HYPRE_StructBAMGGetSkipRelax( HYPRE_StructSolver solver,
                              HYPRE_Int        * skip_relax )
{
   return( hypre_BAMGGetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetDxyz( HYPRE_StructSolver  solver,
                         HYPRE_Real         *dxyz   )
{
   return( hypre_BAMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetLogging( HYPRE_StructSolver solver,
                            HYPRE_Int          logging )
{
   return( hypre_BAMGSetLogging( (void *) solver, logging) );
}

HYPRE_Int
HYPRE_StructBAMGGetLogging( HYPRE_StructSolver solver,
                            HYPRE_Int        * logging )
{
   return( hypre_BAMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGSetPrintLevel( HYPRE_StructSolver solver,
                               HYPRE_Int            print_level )
{
   return( hypre_BAMGSetPrintLevel( (void *) solver, print_level) );
}

HYPRE_Int
HYPRE_StructBAMGGetPrintLevel( HYPRE_StructSolver solver,
                               HYPRE_Int          * print_level )
{
   return( hypre_BAMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGGetNumIterations( HYPRE_StructSolver  solver,
                                  HYPRE_Int          *num_iterations )
{
   return( hypre_BAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructBAMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                              HYPRE_Real         *norm   )
{
   return( hypre_BAMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

