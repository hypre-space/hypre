/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.1 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * HYPRE_AMS Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amscreate, HYPRE_AMSCREATE)(
   long int *solver,
   int      *ierr)

{
   *ierr = (int) ( HYPRE_AMSCreate( (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsdestroy, HYPRE_AMSDESTROY)(
   long int *solver,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetup, HYPRE_AMSSETUP)(
   long int *solver,
   long int *A,
   long int *b,
   long int *x,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetup( (HYPRE_Solver)       *solver,
                                   (HYPRE_ParCSRMatrix) *A,
                                   (HYPRE_ParVector)    *b,
                                   (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssolve, HYPRE_AMSSOLVE)(
   long int *solver,
   long int *A,
   long int *b,
   long int *x,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSolve( (HYPRE_Solver)       *solver,
                                   (HYPRE_ParCSRMatrix) *A,
                                   (HYPRE_ParVector)    *b,
                                   (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdimension, HYPRE_AMSSETDIMENSION)(
   long int *solver,
   int      *dim,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetDimension( (HYPRE_Solver) *solver,
                                          (int)          *dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdiscretegradient, HYPRE_AMSSETDISCRETEGRADIENT)(
   long int *solver,
   long int *G,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetDiscreteGradient( (HYPRE_Solver)       *solver,
                                                 (HYPRE_ParCSRMatrix) *G ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcoordinatevectors, HYPRE_AMSSETCOORDINATEVECTORS)(
   long int *solver,
   long int *x,
   long int *y,
   long int *z,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetCoordinateVectors( (HYPRE_Solver)    *solver,
                                                  (HYPRE_ParVector) *x,
                                                  (HYPRE_ParVector) *y,
                                                  (HYPRE_ParVector) *z ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetedgeconstantvectors, HYPRE_AMSSETEDGECONSTANTVECTORS)(
   long int *solver,
   long int *Gx,
   long int *Gy,
   long int *Gz,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetEdgeConstantVectors( (HYPRE_Solver)    *solver,
                                                    (HYPRE_ParVector) *Gx,
                                                    (HYPRE_ParVector) *Gy,
                                                    (HYPRE_ParVector) *Gz ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphapoissonmatrix, HYPRE_AMSSETALPHAPOISSONMATRIX)(
   long int *solver,
   long int *A_alpha,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetAlphaPoissonMatrix( (HYPRE_Solver)       *solver,
                                                   (HYPRE_ParCSRMatrix) *A_alpha ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetapoissonmatrix, HYPRE_AMSSETBETAPOISSONMATRIX)(
   long int *solver,
   long int *A_beta,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetBetaPoissonMatrix( (HYPRE_Solver)       *solver,
                                                  (HYPRE_ParCSRMatrix) *A_beta ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetmaxiter, HYPRE_AMSSETMAXITER)(
   long int *solver,
   int      *maxiter,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetMaxIter( (HYPRE_Solver) *solver,
                                        (int)          *maxiter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssettol, HYPRE_AMSSETTOL)(
   long int *solver,
   double   *tol,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetTol( (HYPRE_Solver) *solver,
                                    (double)       *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcycletype, HYPRE_AMSSETCYCLETYPE)(
   long int *solver,
   int      *cycle_type,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetCycleType( (HYPRE_Solver) *solver,
                                          (int)          *cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetprintlevel, HYPRE_AMSSETPRINTLEVEL)(
   long int *solver,
   int      *print_level,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetPrintLevel( (HYPRE_Solver) *solver,
                                           (int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetsmoothingoptions, HYPRE_AMSSETSMOOTHINGOPTIONS)(
   long int *solver,
   int      *relax_type,
   int      *relax_times,
   double   *relax_weight,
   double   *omega,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSSetSmoothingOptions( (HYPRE_Solver) *solver,
                                                 (int)          *relax_type,
                                                 (int)          *relax_times,
                                                 (double)       *relax_weight,
                                                 (double)       *omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphaamgoptions, HYPRE_AMSSETALPHAAMGOPTIONS)(
   long int *solver,
   int      *alpha_coarsen_type,
   int      *alpha_agg_levels,
   int      *alpha_relax_type,
   double   *alpha_strength_threshold,
   int      *alpha_interp_type,
   int      *alpha_Pmax,
   int      *ierr)

{
   *ierr = (int) ( HYPRE_AMSSetAlphaAMGOptions( (HYPRE_Solver) *solver,
                                                (int)          *alpha_coarsen_type,
                                                (int)          *alpha_agg_levels,
                                                (int)          *alpha_relax_type,
                                                (double)       *alpha_strength_threshold,
                                                (int)          *alpha_interp_type,
                                                (int)          *alpha_Pmax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetaamgoptions, HYPRE_AMSSETBETAAMGOPTIONS)(
   long int *solver,
   int      *beta_coarsen_type,
   int      *beta_agg_levels,
   int      *beta_relax_type,
   double   *beta_strength_threshold,
   int      *beta_interp_type,
   int      *beta_Pmax,
   int      *ierr)

{
   *ierr = (int) ( HYPRE_AMSSetBetaAMGOptions( (HYPRE_Solver) *solver,
                                               (int)          *beta_coarsen_type,
                                               (int)          *beta_agg_levels,
                                               (int)          *beta_relax_type,
                                               (double)       *beta_strength_threshold,
                                               (int)          *beta_interp_type,
                                               (int)          *beta_Pmax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetnumiterations, HYPRE_AMSGETNUMITERATIONS)(
   long int *solver,
   int      *num_iterations,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSGetNumIterations( (HYPRE_Solver) *solver,
                                              (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetfinalrelativeresidualnorm, HYPRE_AMSGETFINALRELATIVERESIDUALNORM)(
   long int *solver,
   double   *rel_resid_norm,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                          (double *)      rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsconstructdiscretegradient, HYPRE_AMSCONSTRUCTDISCRETEGRADIENT)(
   long int *A,
   long int *x_coord,
   int      *edge_vertex,
   int      *edge_orientation,
   long int *G,
   int      *ierr)
{
   *ierr = (int) ( HYPRE_AMSConstructDiscreteGradient( (HYPRE_ParCSRMatrix)   *A,
                                                       (HYPRE_ParVector)      *x_coord,
                                                       (int *)                 edge_vertex,
                                                       (int)                  *edge_orientation,
                                                       (HYPRE_ParCSRMatrix *)  G ) );
}
