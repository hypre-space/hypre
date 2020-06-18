/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ILUCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (HYPRE_Solver) hypre_ILUCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ILUDestroy( HYPRE_Solver solver )
{
   return( hypre_ILUDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ILUSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_ILUSetup( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ILUSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x      )
{
   return( hypre_ILUSolve( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level )
{
   return hypre_ILUSetPrintLevel( solver, print_level );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetLogging( HYPRE_Solver solver, HYPRE_Int logging )
{
   return hypre_ILUSetLogging(solver, logging );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return hypre_ILUSetMaxIter( solver, max_iter );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetTol( HYPRE_Solver solver, HYPRE_Real tol )
{
   return hypre_ILUSetTol( solver, tol );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetDropThreshold( HYPRE_Solver solver, HYPRE_Real threshold )
{
   return hypre_ILUSetDropThreshold( solver, threshold );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetDropThresholdArray( HYPRE_Solver solver, HYPRE_Real *threshold )
{
   return hypre_ILUSetDropThresholdArray( solver, threshold );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetNSHDropThreshold( HYPRE_Solver solver, HYPRE_Real threshold )
{
   return hypre_ILUSetSchurNSHDropThreshold( solver, threshold );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetNSHDropThresholdArray
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetNSHDropThresholdArray( HYPRE_Solver solver, HYPRE_Real *threshold )
{
   return hypre_ILUSetSchurNSHDropThresholdArray( solver, threshold );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetSchurMaxIter( HYPRE_Solver solver, HYPRE_Int ss_max_iter )
{
   return hypre_ILUSetSchurSolverMaxIter( solver, ss_max_iter );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetMaxNnzPerRow( HYPRE_Solver solver, HYPRE_Int nzmax )
{
   return hypre_ILUSetMaxNnzPerRow( solver, nzmax );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetFillLevel
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetLevelOfFill( HYPRE_Solver solver, HYPRE_Int lfil )
{
   return hypre_ILUSetLevelOfFill( solver, lfil );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetType( HYPRE_Solver solver, HYPRE_Int ilu_type )
{
   return hypre_ILUSetType( solver, ilu_type );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations )
{
   return hypre_ILUGetNumIterations( solver, num_iterations );
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUGetFinalRelativeResidualNorm(  HYPRE_Solver solver, HYPRE_Real *res_norm )
{
   return hypre_ILUGetFinalRelativeResidualNorm(solver, res_norm);
}
/*--------------------------------------------------------------------------
 * HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_ILUSetLocalReordering(  HYPRE_Solver solver, HYPRE_Int ordering_type )
{
   return hypre_ILUSetLocalReordering(solver, ordering_type);
}
