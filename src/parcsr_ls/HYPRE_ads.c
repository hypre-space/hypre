/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ADSCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSCreate(HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_ADSCreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSDestroy(HYPRE_Solver solver)
{
   return hypre_ADSDestroy((void *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetup (HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return hypre_ADSSetup((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSolve (HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return hypre_ADSSolve((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}


/*--------------------------------------------------------------------------
 * HYPRE_ADSSetDiscreteCurl
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetDiscreteCurl(HYPRE_Solver solver,
                                   HYPRE_ParCSRMatrix C)
{
   return hypre_ADSSetDiscreteCurl((void *) solver,
                                   (hypre_ParCSRMatrix *) C);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetDiscreteGradient(HYPRE_Solver solver,
                                       HYPRE_ParCSRMatrix G)
{
   return hypre_ADSSetDiscreteGradient((void *) solver,
                                       (hypre_ParCSRMatrix *) G);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetCoordinateVectors(HYPRE_Solver solver,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector y,
                                        HYPRE_ParVector z)
{
   return hypre_ADSSetCoordinateVectors((void *) solver,
                                        (hypre_ParVector *) x,
                                        (hypre_ParVector *) y,
                                        (hypre_ParVector *) z);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetInterpolations
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetInterpolations(HYPRE_Solver solver,
                                     HYPRE_ParCSRMatrix RT_Pi,
                                     HYPRE_ParCSRMatrix RT_Pix,
                                     HYPRE_ParCSRMatrix RT_Piy,
                                     HYPRE_ParCSRMatrix RT_Piz,
                                     HYPRE_ParCSRMatrix ND_Pi,
                                     HYPRE_ParCSRMatrix ND_Pix,
                                     HYPRE_ParCSRMatrix ND_Piy,
                                     HYPRE_ParCSRMatrix ND_Piz)
{
   return hypre_ADSSetInterpolations((void *) solver,
                                     (hypre_ParCSRMatrix *) RT_Pi,
                                     (hypre_ParCSRMatrix *) RT_Pix,
                                     (hypre_ParCSRMatrix *) RT_Piy,
                                     (hypre_ParCSRMatrix *) RT_Piz,
                                     (hypre_ParCSRMatrix *) ND_Pi,
                                     (hypre_ParCSRMatrix *) ND_Pix,
                                     (hypre_ParCSRMatrix *) ND_Piy,
                                     (hypre_ParCSRMatrix *) ND_Piz);

}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int maxit)
{
   return hypre_ADSSetMaxIter((void *) solver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetTol(HYPRE_Solver solver,
                          HYPRE_Real tol)
{
   return hypre_ADSSetTol((void *) solver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int cycle_type)
{
   return hypre_ADSSetCycleType((void *) solver, cycle_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int print_level)
{
   return hypre_ADSSetPrintLevel((void *) solver, print_level);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetSmoothingOptions(HYPRE_Solver solver,
                                       HYPRE_Int relax_type,
                                       HYPRE_Int relax_times,
                                       HYPRE_Real relax_weight,
                                       HYPRE_Real omega)
{
   return hypre_ADSSetSmoothingOptions((void *) solver,
                                       relax_type,
                                       relax_times,
                                       relax_weight,
                                       omega);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetChebyOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetChebySmoothingOptions(HYPRE_Solver solver,
                                            HYPRE_Int cheby_order,
                                            HYPRE_Real cheby_fraction)
{
   return hypre_ADSSetChebySmoothingOptions((void *) solver,
                                            cheby_order,
                                            cheby_fraction);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetAMSOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetAMSOptions(HYPRE_Solver solver,
                                 HYPRE_Int cycle_type,
                                 HYPRE_Int coarsen_type,
                                 HYPRE_Int agg_levels,
                                 HYPRE_Int relax_type,
                                 HYPRE_Real strength_threshold,
                                 HYPRE_Int interp_type,
                                 HYPRE_Int Pmax)
{
   return hypre_ADSSetAMSOptions((void *) solver,
                                 cycle_type,
                                 coarsen_type,
                                 agg_levels,
                                 relax_type,
                                 strength_threshold,
                                 interp_type,
                                 Pmax);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSSetAMGOptions(HYPRE_Solver solver,
                                 HYPRE_Int coarsen_type,
                                 HYPRE_Int agg_levels,
                                 HYPRE_Int relax_type,
                                 HYPRE_Real strength_threshold,
                                 HYPRE_Int interp_type,
                                 HYPRE_Int Pmax)
{
   return hypre_ADSSetAMGOptions((void *) solver,
                                 coarsen_type,
                                 agg_levels,
                                 relax_type,
                                 strength_threshold,
                                 interp_type,
                                 Pmax);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSGetNumIterations(HYPRE_Solver solver,
                                    HYPRE_Int *num_iterations)
{
   return hypre_ADSGetNumIterations((void *) solver,
                                    num_iterations);
}

/*--------------------------------------------------------------------------
 * HYPRE_ADSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                HYPRE_Real *rel_resid_norm)
{
   return hypre_ADSGetFinalRelativeResidualNorm((void *) solver,
                                                rel_resid_norm);
}
