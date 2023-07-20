/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_FSAICreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAICreate( HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_FSAICreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIDestroy( HYPRE_Solver solver )
{
   return ( hypre_FSAIDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetup( HYPRE_Solver       solver,
                 HYPRE_ParCSRMatrix A,
                 HYPRE_ParVector    b,
                 HYPRE_ParVector    x )
{
   return ( hypre_FSAISetup( (void *) solver,
                             (hypre_ParCSRMatrix *) A,
                             (hypre_ParVector *) b,
                             (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISolve( HYPRE_Solver       solver,
                 HYPRE_ParCSRMatrix A,
                 HYPRE_ParVector    b,
                 HYPRE_ParVector    x )
{
   return ( hypre_FSAISolve( (void *) solver,
                             (hypre_ParCSRMatrix *) A,
                             (hypre_ParVector *) b,
                             (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetAlgoType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetAlgoType( HYPRE_Solver solver,
                       HYPRE_Int    algo_type  )
{
   return ( hypre_FSAISetAlgoType( (void *) solver, algo_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetAlgoType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetAlgoType( HYPRE_Solver  solver,
                       HYPRE_Int    *algo_type  )
{
   return ( hypre_FSAIGetAlgoType( (void *) solver, algo_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetLocalSolveType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetLocalSolveType( HYPRE_Solver solver,
                             HYPRE_Int    local_solve_type  )
{
   return ( hypre_FSAISetLocalSolveType( (void *) solver, local_solve_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetLocalSolveType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetLocalSolveType( HYPRE_Solver  solver,
                             HYPRE_Int    *local_solve_type  )
{
   return ( hypre_FSAIGetLocalSolveType( (void *) solver, local_solve_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetMaxSteps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetMaxSteps( HYPRE_Solver solver,
                       HYPRE_Int    max_steps  )
{
   return ( hypre_FSAISetMaxSteps( (void *) solver, max_steps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetMaxSteps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetMaxSteps( HYPRE_Solver  solver,
                       HYPRE_Int    *max_steps  )
{
   return ( hypre_FSAIGetMaxSteps( (void *) solver, max_steps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetMaxNnzRow
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetMaxNnzRow( HYPRE_Solver solver,
                        HYPRE_Int    max_nnz_row  )
{
   return ( hypre_FSAISetMaxNnzRow( (void *) solver, max_nnz_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetMaxNnzRow
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetMaxNnzRow( HYPRE_Solver  solver,
                        HYPRE_Int    *max_nnz_row  )
{
   return ( hypre_FSAIGetMaxNnzRow( (void *) solver, max_nnz_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetMaxStepSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetMaxStepSize( HYPRE_Solver solver,
                          HYPRE_Int    max_step_size )
{
   return ( hypre_FSAISetMaxStepSize( (void *) solver, max_step_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetMaxStepSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetMaxStepSize( HYPRE_Solver  solver,
                          HYPRE_Int    *max_step_size )
{
   return ( hypre_FSAIGetMaxStepSize( (void *) solver, max_step_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetNumLevels( HYPRE_Solver solver,
                        HYPRE_Int    num_levels )
{
   return ( hypre_FSAISetNumLevels( (void *) solver, num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetNumLevels( HYPRE_Solver  solver,
                        HYPRE_Int    *num_levels )
{
   return ( hypre_FSAIGetNumLevels( (void *) solver, num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetThreshold( HYPRE_Solver solver,
                        HYPRE_Real   threshold )
{
   return ( hypre_FSAISetThreshold( (void *) solver, threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetThreshold( HYPRE_Solver  solver,
                        HYPRE_Real   *threshold )
{
   return ( hypre_FSAIGetThreshold( (void *) solver, threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetZeroGuess( HYPRE_Solver solver,
                        HYPRE_Int    zero_guess )
{
   return ( hypre_FSAISetZeroGuess( (void *) solver, zero_guess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetZeroGuess( HYPRE_Solver  solver,
                        HYPRE_Int    *zero_guess )
{
   return ( hypre_FSAIGetZeroGuess( (void *) solver, zero_guess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetKapTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetKapTolerance( HYPRE_Solver solver,
                           HYPRE_Real   kap_tolerance )
{
   return ( hypre_FSAISetKapTolerance( (void *) solver, kap_tolerance ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetKapTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetKapTolerance( HYPRE_Solver  solver,
                           HYPRE_Real   *kap_tolerance )
{
   return ( hypre_FSAIGetKapTolerance( (void *) solver, kap_tolerance ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetTolerance( HYPRE_Solver solver,
                        HYPRE_Real   tolerance )
{
   return ( hypre_FSAISetTolerance( (void *) solver, tolerance ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetTolerance( HYPRE_Solver  solver,
                        HYPRE_Real   *tolerance )
{
   return ( hypre_FSAIGetTolerance( (void *) solver, tolerance ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetOmega( HYPRE_Solver solver,
                    HYPRE_Real   omega )
{
   return ( hypre_FSAISetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetOmega( HYPRE_Solver  solver,
                    HYPRE_Real   *omega )
{
   return ( hypre_FSAIGetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetMaxIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetMaxIterations( HYPRE_Solver solver,
                            HYPRE_Int    max_iterations )
{
   return ( hypre_FSAISetMaxIterations( (void *) solver, max_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetMaxIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetMaxIterations( HYPRE_Solver  solver,
                            HYPRE_Int    *max_iterations )
{
   return ( hypre_FSAIGetMaxIterations( (void *) solver, max_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetEigMaxIters
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetEigMaxIters( HYPRE_Solver solver,
                          HYPRE_Int    eig_max_iters )
{
   return ( hypre_FSAISetEigMaxIters( (void *) solver, eig_max_iters ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetEigMaxIters
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetEigMaxIters( HYPRE_Solver  solver,
                          HYPRE_Int    *eig_max_iters )
{
   return ( hypre_FSAIGetEigMaxIters( (void *) solver, eig_max_iters ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAISetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAISetPrintLevel( HYPRE_Solver solver,
                         HYPRE_Int    print_level )
{
   return ( hypre_FSAISetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FSAIGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FSAIGetPrintLevel( HYPRE_Solver  solver,
                         HYPRE_Int    *print_level )
{
   return ( hypre_FSAIGetPrintLevel( (void *) solver, print_level ) );
}
