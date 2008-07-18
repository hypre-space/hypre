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




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_HybridCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridDestroy( HYPRE_StructSolver solver )
{
   return( hypre_HybridDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_HybridSetup( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructHybridSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_HybridSolve( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetTol( HYPRE_StructSolver solver,
                          double             tol    )
{
   return( hypre_HybridSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetConvergenceTol( HYPRE_StructSolver solver,
                                     double             cf_tol    )
{
   return( hypre_HybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetDSCGMaxIter( HYPRE_StructSolver solver,
                                  int                dscg_max_its )
{
   return( hypre_HybridSetDSCGMaxIter( (void *) solver, dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetPCGMaxIter( HYPRE_StructSolver solver,
                                 int                pcg_max_its )
{
   return( hypre_HybridSetPCGMaxIter( (void *) solver, pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetPCGAbsoluteTolFactor( HYPRE_StructSolver solver,
                                           double      pcg_atolf )
{
   return( hypre_HybridSetPCGAbsoluteTolFactor( (void *) solver, pcg_atolf ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetTwoNorm( HYPRE_StructSolver solver,
                              int                two_norm    )
{
   return( hypre_HybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetStopCrit( HYPRE_StructSolver solver,
                              int                stop_crit    )
{
   return( hypre_HybridSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetRelChange( HYPRE_StructSolver solver,
                                int                rel_change    )
{
   return( hypre_HybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetSolverType( HYPRE_StructSolver solver,
                                int                solver_type    )
{
   return( hypre_HybridSetSolverType( (void *) solver, solver_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetKDim( HYPRE_StructSolver solver,
                                int                k_dim    )
{
   return( hypre_HybridSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetPrecond( HYPRE_StructSolver         solver,
                              HYPRE_PtrToStructSolverFcn precond,
                              HYPRE_PtrToStructSolverFcn precond_setup,
                              HYPRE_StructSolver         precond_solver )
{
   return( hypre_HybridSetPrecond( (void *) solver,
                                   precond, precond_setup,
                                   (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetLogging( HYPRE_StructSolver solver,
                              int                logging    )
{
   return( hypre_HybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridSetPrintLevel( HYPRE_StructSolver solver,
                              int                print_level    )
{
   return( hypre_HybridSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetNumIterations( HYPRE_StructSolver solver,
                                    int               *num_its    )
{
   return( hypre_HybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetDSCGNumIterations( HYPRE_StructSolver solver,
                                        int               *dscg_num_its )
{
   return( hypre_HybridGetDSCGNumIterations( (void *) solver, dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetPCGNumIterations( HYPRE_StructSolver solver,
                                       int               *pcg_num_its )
{
   return( hypre_HybridGetPCGNumIterations( (void *) solver, pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructHybridGetFinalRelativeResidualNorm( HYPRE_StructSolver solver,
                                                double            *norm    )
{
   return( hypre_HybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

