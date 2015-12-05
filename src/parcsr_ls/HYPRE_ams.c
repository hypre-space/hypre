/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.16 $
 ***********************************************************************EHEADER*/





#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSCreate(HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_AMSCreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSDestroy(HYPRE_Solver solver)
{
   return hypre_AMSDestroy((void *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetup (HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return hypre_AMSSetup((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSolve (HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return hypre_AMSSolve((void *) solver,
                         (hypre_ParCSRMatrix *) A,
                         (hypre_ParVector *) b,
                         (hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetDimension(HYPRE_Solver solver,
                                HYPRE_Int dim)
{
   return hypre_AMSSetDimension((void *) solver, dim);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetDiscreteGradient(HYPRE_Solver solver,
                                       HYPRE_ParCSRMatrix G)
{
   return hypre_AMSSetDiscreteGradient((void *) solver,
                                       (hypre_ParCSRMatrix *) G);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetCoordinateVectors(HYPRE_Solver solver,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector y,
                                        HYPRE_ParVector z)
{
   return hypre_AMSSetCoordinateVectors((void *) solver,
                                        (hypre_ParVector *) x,
                                        (hypre_ParVector *) y,
                                        (hypre_ParVector *) z);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetEdgeConstantVectors(HYPRE_Solver solver,
                                          HYPRE_ParVector Gx,
                                          HYPRE_ParVector Gy,
                                          HYPRE_ParVector Gz)
{
   return hypre_AMSSetEdgeConstantVectors((void *) solver,
                                          (hypre_ParVector *) Gx,
                                          (hypre_ParVector *) Gy,
                                          (hypre_ParVector *) Gz);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetInterpolations
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetInterpolations(HYPRE_Solver solver,
                                     HYPRE_ParCSRMatrix Pi,
                                     HYPRE_ParCSRMatrix Pix,
                                     HYPRE_ParCSRMatrix Piy,
                                     HYPRE_ParCSRMatrix Piz)
{
   return hypre_AMSSetInterpolations((void *) solver,
                                     (hypre_ParCSRMatrix *) Pi,
                                     (hypre_ParCSRMatrix *) Pix,
                                     (hypre_ParCSRMatrix *) Piy,
                                     (hypre_ParCSRMatrix *) Piz);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix(HYPRE_Solver solver,
                                         HYPRE_ParCSRMatrix A_alpha)
{
   return hypre_AMSSetAlphaPoissonMatrix((void *) solver,
                                         (hypre_ParCSRMatrix *) A_alpha);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix(HYPRE_Solver solver,
                                        HYPRE_ParCSRMatrix A_beta)
{
   return hypre_AMSSetBetaPoissonMatrix((void *) solver,
                                        (hypre_ParCSRMatrix *) A_beta);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSetInteriorNodes
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetInteriorNodes(HYPRE_Solver solver,
                                    HYPRE_ParVector interior_nodes)
{
   return hypre_AMSSetInteriorNodes((void *) solver,
                                    (hypre_ParVector *) interior_nodes);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSetProjectionFrequency
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetProjectionFrequency(HYPRE_Solver solver,
                                          HYPRE_Int projection_frequency)
{
   return hypre_AMSSetProjectionFrequency((void *) solver,
                                          projection_frequency);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int maxit)
{
   return hypre_AMSSetMaxIter((void *) solver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetTol(HYPRE_Solver solver,
                          double tol)
{
   return hypre_AMSSetTol((void *) solver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int cycle_type)
{
   return hypre_AMSSetCycleType((void *) solver, cycle_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int print_level)
{
   return hypre_AMSSetPrintLevel((void *) solver, print_level);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetSmoothingOptions(HYPRE_Solver solver,
                                       HYPRE_Int relax_type,
                                       HYPRE_Int relax_times,
                                       double relax_weight,
                                       double omega)
{
   return hypre_AMSSetSmoothingOptions((void *) solver,
                                       relax_type,
                                       relax_times,
                                       relax_weight,
                                       omega);
}

/*--------------------------------------------------------------------------
 * HYPRE_ARTSSetChebyOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetChebySmoothingOptions(HYPRE_Solver solver,
                                            HYPRE_Int cheby_order,
                                            HYPRE_Int cheby_fraction)
{
   return hypre_AMSSetChebySmoothingOptions((void *) solver,
                                            cheby_order,
                                            cheby_fraction);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetAlphaAMGOptions(HYPRE_Solver solver,
                                      HYPRE_Int alpha_coarsen_type,
                                      HYPRE_Int alpha_agg_levels,
                                      HYPRE_Int alpha_relax_type,
                                      double alpha_strength_threshold,
                                      HYPRE_Int alpha_interp_type,
                                      HYPRE_Int alpha_Pmax)
{
   return hypre_AMSSetAlphaAMGOptions((void *) solver,
                                      alpha_coarsen_type,
                                      alpha_agg_levels,
                                      alpha_relax_type,
                                      alpha_strength_threshold,
                                      alpha_interp_type,
                                      alpha_Pmax);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSSetBetaAMGOptions(HYPRE_Solver solver,
                                     HYPRE_Int beta_coarsen_type,
                                     HYPRE_Int beta_agg_levels,
                                     HYPRE_Int beta_relax_type,
                                     double beta_strength_threshold,
                                     HYPRE_Int beta_interp_type,
                                     HYPRE_Int beta_Pmax)
{
   return hypre_AMSSetBetaAMGOptions((void *) solver,
                                     beta_coarsen_type,
                                     beta_agg_levels,
                                     beta_relax_type,
                                     beta_strength_threshold,
                                     beta_interp_type,
                                     beta_Pmax);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSGetNumIterations(HYPRE_Solver solver,
                                    HYPRE_Int *num_iterations)
{
   return hypre_AMSGetNumIterations((void *) solver,
                                    num_iterations);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                double *rel_resid_norm)
{
   return hypre_AMSGetFinalRelativeResidualNorm((void *) solver,
                                                rel_resid_norm);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSProjectOutGradients
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSProjectOutGradients(HYPRE_Solver solver,
                                       HYPRE_ParVector x)
{
   return hypre_AMSProjectOutGradients((void *) solver,
                                       (hypre_ParVector *) x);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSConstructDiscreteGradient(HYPRE_ParCSRMatrix A,
                                             HYPRE_ParVector x_coord,
                                             HYPRE_Int *edge_vertex,
                                             HYPRE_Int edge_orientation,
                                             HYPRE_ParCSRMatrix *G)
{
   return hypre_AMSConstructDiscreteGradient((hypre_ParCSRMatrix *) A,
                                             (hypre_ParVector *) x_coord,
                                             edge_vertex,
                                             edge_orientation,
                                             (hypre_ParCSRMatrix **) G);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSFEISetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSFEISetup(HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x,
                            HYPRE_Int *EdgeNodeList_,
                            HYPRE_Int *NodeNumbers_,
                            HYPRE_Int numEdges_,
                            HYPRE_Int numLocalNodes_,
                            HYPRE_Int numNodes_,
                            double *NodalCoord_)
{
   return hypre_AMSFEISetup((void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x,
                            numNodes_,
                            numLocalNodes_,
                            NodeNumbers_,
                            NodalCoord_,
                            numEdges_,
                            EdgeNodeList_);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSFEIDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_AMSFEIDestroy(HYPRE_Solver solver)
{
   return hypre_AMSFEIDestroy((void *) solver);
}
