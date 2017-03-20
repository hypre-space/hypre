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
HYPRE_SStructSysBAMGCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_SysBAMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysBAMGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_SysBAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysBAMGSetup( HYPRE_SStructSolver  solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysBAMGSetup( (void *) solver,
                               (hypre_SStructMatrix *) A,
                               (hypre_SStructVector *) b,
                               (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructSysBAMGSolve( HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysBAMGSolve( (void *) solver,
                            (hypre_SStructMatrix *) A,
                            (hypre_SStructVector *) b,
                            (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetTol( HYPRE_SStructSolver solver,
                            HYPRE_Real         tol    )
{
   return( hypre_SysBAMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetMaxIter( HYPRE_SStructSolver solver,
                                HYPRE_Int          max_iter  )
{
   return( hypre_SysBAMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetRelChange( HYPRE_SStructSolver solver,
                                  HYPRE_Int          rel_change  )
{
   return( hypre_SysBAMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysBAMGSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysBAMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_SStructSysBAMGSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysBAMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetRelaxType( HYPRE_SStructSolver solver,
                                  HYPRE_Int          relax_type )
{
   return( hypre_SysBAMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
HYPRE_Int
HYPRE_SStructSysBAMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                    HYPRE_Real          weight)
{
   return( hypre_SysBAMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetNumPreRelax( HYPRE_SStructSolver solver,
                                    HYPRE_Int          num_pre_relax )
{
   return( hypre_SysBAMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetNumPostRelax( HYPRE_SStructSolver solver,
                                     HYPRE_Int          num_post_relax )
{
   return( hypre_SysBAMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetSkipRelax( HYPRE_SStructSolver solver,
                                  HYPRE_Int          skip_relax )
{
   return( hypre_SysBAMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetDxyz( HYPRE_SStructSolver  solver,
                         HYPRE_Real         *dxyz   )
{
   return( hypre_SysBAMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetLogging( HYPRE_SStructSolver solver,
                                HYPRE_Int          logging )
{
   return( hypre_SysBAMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
*--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGSetPrintLevel( HYPRE_SStructSolver solver,
                                HYPRE_Int         print_level )
{
   return( hypre_SysBAMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGGetNumIterations( HYPRE_SStructSolver  solver,
                                      HYPRE_Int          *num_iterations )
{
   return( hypre_SysBAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSysBAMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  HYPRE_Real         *norm   )
{
   return( hypre_SysBAMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

