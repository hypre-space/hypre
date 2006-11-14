/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

int HYPRE_AMSCreate(HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_AMSCreate();
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_AMSDestroy(HYPRE_Solver solver)
{
   return hypre_AMSDestroy((void *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetup (HYPRE_Solver solver,
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

int HYPRE_AMSSolve (HYPRE_Solver solver,
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

int HYPRE_AMSSetDimension(HYPRE_Solver solver,
                          int dim)
{
   return hypre_AMSSetDimension((void *) solver, dim);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetDiscreteGradient(HYPRE_Solver solver,
                                 HYPRE_ParCSRMatrix G)
{
   return hypre_AMSSetDiscreteGradient((void *) solver,
                                       (hypre_ParCSRMatrix *) G);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetCoordinateVectors(HYPRE_Solver solver,
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

int HYPRE_AMSSetEdgeConstantVectors(HYPRE_Solver solver,
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
 * HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetAlphaPoissonMatrix(HYPRE_Solver solver,
                                   HYPRE_ParCSRMatrix A_alpha)
{
   return hypre_AMSSetAlphaPoissonMatrix((void *) solver,
                                         (hypre_ParCSRMatrix *) A_alpha);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetBetaPoissonMatrix(HYPRE_Solver solver,
                                  HYPRE_ParCSRMatrix A_beta)
{
   return hypre_AMSSetBetaPoissonMatrix((void *) solver,
                                        (hypre_ParCSRMatrix *) A_beta);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetMaxIter(HYPRE_Solver solver,
                        int maxit)
{
   return hypre_AMSSetMaxIter((void *) solver, maxit);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetTol(HYPRE_Solver solver,
                    double tol)
{
   return hypre_AMSSetTol((void *) solver, tol);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetCycleType(HYPRE_Solver solver,
                          int cycle_type)
{
   return hypre_AMSSetCycleType((void *) solver, cycle_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetPrintLevel(HYPRE_Solver solver,
                           int print_level)
{
   return hypre_AMSSetPrintLevel((void *) solver, print_level);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetSmoothingOptions(HYPRE_Solver solver,
                                 int relax_type,
                                 int relax_times,
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
 * HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetAlphaAMGOptions(HYPRE_Solver solver,
                                int alpha_coarsen_type,
                                int alpha_agg_levels,
                                int alpha_relax_type,
                                double alpha_strength_threshold)
{
   return hypre_AMSSetAlphaAMGOptions((void *) solver,
                                      alpha_coarsen_type,
                                      alpha_agg_levels,
                                      alpha_relax_type,
                                      alpha_strength_threshold);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

int HYPRE_AMSSetBetaAMGOptions(HYPRE_Solver solver,
                               int beta_coarsen_type,
                               int beta_agg_levels,
                               int beta_relax_type,
                               double beta_strength_threshold)
{
   return hypre_AMSSetBetaAMGOptions((void *) solver,
                                     beta_coarsen_type,
                                     beta_agg_levels,
                                     beta_relax_type,
                                     beta_strength_threshold);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_AMSGetNumIterations(HYPRE_Solver solver,
                              int *num_iterations)
{
   return hypre_AMSGetNumIterations((void *) solver,
                                    num_iterations);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_AMSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                          double *rel_resid_norm)
{
   return hypre_AMSGetFinalRelativeResidualNorm((void *) solver,
                                                rel_resid_norm);
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

int HYPRE_AMSConstructDiscreteGradient(HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector x_coord,
				       int *edge_vertex,
                                       HYPRE_ParCSRMatrix *G)
{
   return hypre_AMSConstructDiscreteGradient((hypre_ParCSRMatrix *) A,
                                             (hypre_ParVector *) x_coord,
                                             edge_vertex,
                                             (hypre_ParCSRMatrix **) G);
}
