/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebyCreate( HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *solver = (HYPRE_Solver) hypre_ParChebyCreate();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebyDestroy( HYPRE_Solver solver )
{
   return ( hypre_ParChebyDestroy( (hypre_ParChebyData *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetup( HYPRE_Solver       solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector    b,
                     HYPRE_ParVector    x )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_ParChebySetup( (hypre_ParChebyData *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *)    b,
                                 (hypre_ParVector *)    x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySolve( HYPRE_Solver       solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector    b,
                     HYPRE_ParVector    x )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!b)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!x)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   return ( hypre_ParChebySolve( (hypre_ParChebyData *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *)    b,
                                 (hypre_ParVector *)    x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetMaxIterations( HYPRE_Solver solver,
                                HYPRE_Int    max_iterations )
{
   return ( hypre_ParChebySetMaxIterations( (hypre_ParChebyData *) solver, max_iterations ) );
}

HYPRE_Int
HYPRE_ParChebyGetMaxIterations( HYPRE_Solver  solver,
                                HYPRE_Int    *max_iterations )
{
   return ( hypre_ParChebyGetMaxIterations( (hypre_ParChebyData *) solver, max_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetTolerance( HYPRE_Solver solver,
                            HYPRE_Real   tol )
{
   return ( hypre_ParChebySetTolerance( (hypre_ParChebyData *) solver, tol ) );
}

HYPRE_Int
HYPRE_ParChebyGetTolerance( HYPRE_Solver  solver,
                            HYPRE_Real   *tol )
{
   return ( hypre_ParChebyGetTolerance( (hypre_ParChebyData *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetZeroGuess( HYPRE_Solver solver,
                            HYPRE_Int    zero_guess )
{
   return ( hypre_ParChebySetZeroGuess( (hypre_ParChebyData *) solver, zero_guess ) );
}

HYPRE_Int
HYPRE_ParChebyGetZeroGuess( HYPRE_Solver  solver,
                            HYPRE_Int    *zero_guess )
{
   return ( hypre_ParChebyGetZeroGuess( (hypre_ParChebyData *) solver, zero_guess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetPrintLevel( HYPRE_Solver solver,
                             HYPRE_Int    print_level )
{
   return ( hypre_ParChebySetPrintLevel( (hypre_ParChebyData *) solver, print_level ) );
}

HYPRE_Int
HYPRE_ParChebyGetPrintLevel( HYPRE_Solver  solver,
                             HYPRE_Int    *print_level )
{
   return ( hypre_ParChebyGetPrintLevel( (hypre_ParChebyData *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetLogging( HYPRE_Solver solver,
                          HYPRE_Int    logging )
{
   return ( hypre_ParChebySetLogging( (hypre_ParChebyData *) solver, logging ) );
}

HYPRE_Int
HYPRE_ParChebyGetLogging( HYPRE_Solver  solver,
                          HYPRE_Int    *logging )
{
   return ( hypre_ParChebyGetLogging( (hypre_ParChebyData *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetOrder( HYPRE_Solver solver,
                        HYPRE_Int    order )
{
   return ( hypre_ParChebySetOrder( (hypre_ParChebyData *) solver, order ) );
}

HYPRE_Int
HYPRE_ParChebyGetOrder( HYPRE_Solver  solver,
                        HYPRE_Int    *order )
{
   return ( hypre_ParChebyGetOrder( (hypre_ParChebyData *) solver, order ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetVariant( HYPRE_Solver solver,
                          HYPRE_Int    variant )
{
   return ( hypre_ParChebySetVariant( (hypre_ParChebyData *) solver, variant ) );
}

HYPRE_Int
HYPRE_ParChebyGetVariant( HYPRE_Solver  solver,
                          HYPRE_Int    *variant )
{
   return ( hypre_ParChebyGetVariant( (hypre_ParChebyData *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetScale( HYPRE_Solver solver,
                        HYPRE_Int    scale )
{
   return ( hypre_ParChebySetScale( (hypre_ParChebyData *) solver, scale ) );
}

HYPRE_Int
HYPRE_ParChebyGetScale( HYPRE_Solver  solver,
                        HYPRE_Int    *scale )
{
   return ( hypre_ParChebyGetScale( (hypre_ParChebyData *) solver, scale ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetEigRatio( HYPRE_Solver solver,
                           HYPRE_Real   eig_ratio )
{
   return ( hypre_ParChebySetEigRatio( (hypre_ParChebyData *) solver, eig_ratio ) );
}

HYPRE_Int
HYPRE_ParChebyGetEigRatio( HYPRE_Solver  solver,
                           HYPRE_Real   *eig_ratio )
{
   return ( hypre_ParChebyGetEigRatio( (hypre_ParChebyData *) solver, eig_ratio ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetEigEst( HYPRE_Solver solver,
                         HYPRE_Int    eig_est )
{
   return ( hypre_ParChebySetEigEst( (hypre_ParChebyData *) solver, eig_est ) );
}

HYPRE_Int
HYPRE_ParChebyGetEigEst( HYPRE_Solver  solver,
                         HYPRE_Int    *eig_est )
{
   return ( hypre_ParChebyGetEigEst( (hypre_ParChebyData *) solver, eig_est ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParChebySetMinMaxEigEst( HYPRE_Solver  solver,
                               HYPRE_Real    eig_min_est,
                               HYPRE_Real    eig_max_est )
{
   return ( hypre_ParChebySetMinMaxEigEst( (hypre_ParChebyData *) solver, eig_min_est, eig_max_est ) );
}

HYPRE_Int
HYPRE_ParChebyGetMinMaxEigEst( HYPRE_Solver  solver,
                               HYPRE_Real   *eig_min_est,
                               HYPRE_Real   *eig_max_est )
{
   return ( hypre_ParChebyGetMinMaxEigEst( (hypre_ParChebyData *) solver, eig_min_est, eig_max_est ) );
}
