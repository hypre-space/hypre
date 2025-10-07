/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev solver/relaxation
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Create the data structure for Chebyshev solver/relaxation
 *
 * Can specify order 1-4 (this is the order of the resid polynomial)- here we
 * explicitly code the coefficients (instead of iteratively determining)
 *
 * variant 0: standard chebyshev
 * variant 1: modified cheby: T(t)* f(t) where f(t) = (1-b/t)
 *
 * ratio indicates the percentage of the whole spectrum to use
 * (so .5 means half, and .1 means 10%)
 *--------------------------------------------------------------------------*/

hypre_ParChebyData *
hypre_ParChebyCreate( void )
{
   hypre_ParChebyData  *cheby_data;
   hypre_Solver        *base;

   cheby_data = hypre_CTAlloc(hypre_ParChebyData, 1, HYPRE_MEMORY_HOST);
   base       = (hypre_Solver*) cheby_data;

   /* Set base solver function pointers */
   hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  hypre_ParChebySetup;
   hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  hypre_ParChebySolve;
   hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) hypre_ParChebyDestroy;

   hypre_ParChebyDataPrintLevel(cheby_data)    = 0;
   hypre_ParChebyDataZeroGuess(cheby_data)     = 0;
   hypre_ParChebyDataMaxIterations(cheby_data) = 100;
   hypre_ParChebyDataOrder(cheby_data)         = 2;
   hypre_ParChebyDataVariant(cheby_data)       = 0;
   hypre_ParChebyDataScale(cheby_data)         = 1;
   hypre_ParChebyDataEigEst(cheby_data)        = 10;
   hypre_ParChebyDataEigRatio(cheby_data)      = 0.3;
   hypre_ParChebyDataTol(cheby_data)           = 1.0e-6;

   hypre_ParChebyDataLogging(cheby_data)       = 1;
   hypre_ParChebyDataNumIterations(cheby_data) = 0;
   hypre_ParChebyDataRelResidNorm(cheby_data)  = 0.0;
   hypre_ParChebyDataResidual(cheby_data)      = NULL;

   hypre_ParChebyDataEigProvided(cheby_data)   = 0;
   hypre_ParChebyDataMinEigEst(cheby_data)     = 0.0;
   hypre_ParChebyDataMaxEigEst(cheby_data)     = 0.0;
   hypre_ParChebyDataScaling(cheby_data)       = NULL;
   hypre_ParChebyDataCoefs(cheby_data)         = NULL;
   hypre_ParChebyDataPtemp(cheby_data)         = NULL;
   hypre_ParChebyDataRtemp(cheby_data)         = NULL;
   hypre_ParChebyDataVtemp(cheby_data)         = NULL;
   hypre_ParChebyDataZtemp(cheby_data)         = NULL;
   hypre_ParChebyDataOwnsTemp(cheby_data)      = 0;

   return cheby_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyDestroy( hypre_ParChebyData *cheby_data )
{
   if (cheby_data)
   {
      hypre_TFree(hypre_ParChebyDataCoefs(cheby_data), HYPRE_MEMORY_HOST);
      hypre_ParVectorDestroy(hypre_ParChebyDataScaling(cheby_data));
      hypre_ParVectorDestroy(hypre_ParChebyDataResidual(cheby_data));
      if (hypre_ParChebyDataOwnsTemp(cheby_data))
      {
         hypre_ParVectorDestroy(hypre_ParChebyDataPtemp(cheby_data));
         hypre_ParVectorDestroy(hypre_ParChebyDataRtemp(cheby_data));
         hypre_ParVectorDestroy(hypre_ParChebyDataVtemp(cheby_data));
         hypre_ParVectorDestroy(hypre_ParChebyDataZtemp(cheby_data));
      }

      hypre_TFree(cheby_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetMaxIterations( hypre_ParChebyData  *cheby_data,
                                HYPRE_Int            max_iterations )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_iterations < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParChebyDataMaxIterations(cheby_data) = max_iterations;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetMaxIterations( hypre_ParChebyData  *cheby_data,
                                HYPRE_Int           *max_iterations )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_iterations = hypre_ParChebyDataMaxIterations(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetZeroGuess( hypre_ParChebyData  *cheby_data,
                            HYPRE_Int            zero_guess )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataZeroGuess(cheby_data) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetZeroGuess( hypre_ParChebyData  *cheby_data,
                            HYPRE_Int           *zero_guess )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *zero_guess = hypre_ParChebyDataZeroGuess(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetTolerance( hypre_ParChebyData  *cheby_data,
                            HYPRE_Real           tol )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (tol < 0.0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParChebyDataTol(cheby_data) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetTolerance( hypre_ParChebyData  *cheby_data,
                            HYPRE_Real          *tol )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *tol = hypre_ParChebyDataTol(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetOwsTemp( hypre_ParChebyData  *cheby_data,
                          HYPRE_Int            owns_temp )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataOwnsTemp(cheby_data) = owns_temp;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetPrintLevel(hypre_ParChebyData  *cheby_data,
                            HYPRE_Int            print_level)
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataPrintLevel(cheby_data) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetPrintLevel(hypre_ParChebyData  *cheby_data,
                            HYPRE_Int           *print_level)
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *print_level = hypre_ParChebyDataPrintLevel(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetLogging( hypre_ParChebyData  *cheby_data,
                          HYPRE_Int            logging )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataLogging(cheby_data) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetLogging( hypre_ParChebyData  *cheby_data,
                          HYPRE_Int           *logging )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *logging = hypre_ParChebyDataLogging(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetOrder( hypre_ParChebyData  *cheby_data,
                        HYPRE_Int            order )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (order < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParChebyDataOrder(cheby_data) = order;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetOrder( hypre_ParChebyData  *cheby_data,
                        HYPRE_Int           *order )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *order = hypre_ParChebyDataOrder(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetEigRatio( hypre_ParChebyData  *cheby_data,
                           HYPRE_Real           eig_ratio )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (eig_ratio <= 0.0 || eig_ratio > 1.0 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParChebyDataEigRatio(cheby_data) = eig_ratio;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetEigRatio( hypre_ParChebyData  *cheby_data,
                           HYPRE_Real          *eig_ratio )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *eig_ratio = hypre_ParChebyDataEigRatio(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetEigEst( hypre_ParChebyData  *cheby_data,
                         HYPRE_Int            eig_est )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataEigEst(cheby_data) = eig_est;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetEigEst( hypre_ParChebyData  *cheby_data,
                         HYPRE_Int           *eig_est )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *eig_est = hypre_ParChebyDataEigEst(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetMinMaxEigEst( hypre_ParChebyData  *cheby_data,
                               HYPRE_Real           eig_min_est,
                               HYPRE_Real           eig_max_est )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataMinEigEst(cheby_data)   = eig_min_est;
   hypre_ParChebyDataMaxEigEst(cheby_data)   = eig_max_est;
   hypre_ParChebyDataEigProvided(cheby_data) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetMinMaxEigEst( hypre_ParChebyData  *cheby_data,
                               HYPRE_Real          *eig_min_est,
                               HYPRE_Real          *eig_max_est )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *eig_min_est = hypre_ParChebyDataMinEigEst(cheby_data);
   *eig_max_est = hypre_ParChebyDataMaxEigEst(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetVariant( hypre_ParChebyData  *cheby_data,
                          HYPRE_Int            variant )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataVariant(cheby_data) = variant;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetVariant( hypre_ParChebyData  *cheby_data,
                          HYPRE_Int           *variant )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *variant = hypre_ParChebyDataVariant(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetScale( hypre_ParChebyData  *cheby_data,
                        HYPRE_Int            scale )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParChebyDataScale(cheby_data) = scale;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebyGetScale( hypre_ParChebyData  *cheby_data,
                        HYPRE_Int           *scale )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *scale = hypre_ParChebyDataScale(cheby_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParChebySetTempVectors( hypre_ParChebyData  *cheby_data,
                              hypre_ParVector     *Ptemp,
                              hypre_ParVector     *Rtemp,
                              hypre_ParVector     *Vtemp,
                              hypre_ParVector     *Ztemp )
{
   if (!cheby_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Free current work vectors */
   if (hypre_ParChebyDataOwnsTemp(cheby_data))
   {
      hypre_ParVectorDestroy(hypre_ParChebyDataPtemp(cheby_data));
      hypre_ParVectorDestroy(hypre_ParChebyDataRtemp(cheby_data));
      hypre_ParVectorDestroy(hypre_ParChebyDataVtemp(cheby_data));
      hypre_ParVectorDestroy(hypre_ParChebyDataZtemp(cheby_data));

      /* By default, the Chebyshev solver does not own
         the work vectors passed through this function  */
      hypre_ParChebyDataOwnsTemp(cheby_data) = 0;
   }

   /* Set pointers */
   hypre_ParChebyDataPtemp(cheby_data) = Ptemp;
   hypre_ParChebyDataRtemp(cheby_data) = Rtemp;
   hypre_ParChebyDataVtemp(cheby_data) = Vtemp;
   hypre_ParChebyDataZtemp(cheby_data) = Ztemp;

   return hypre_error_flag;
}
