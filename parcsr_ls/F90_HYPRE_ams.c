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
   hypre_F90_Obj *solver,
   HYPRE_Int      *ierr)

{
   *ierr = (HYPRE_Int) ( HYPRE_AMSCreate( (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsdestroy, HYPRE_AMSDESTROY)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetup, HYPRE_AMSSETUP)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A,
   hypre_F90_Obj *b,
   hypre_F90_Obj *x,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetup( (HYPRE_Solver)       *solver,
                                   (HYPRE_ParCSRMatrix) *A,
                                   (HYPRE_ParVector)    *b,
                                   (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssolve, HYPRE_AMSSOLVE)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A,
   hypre_F90_Obj *b,
   hypre_F90_Obj *x,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSolve( (HYPRE_Solver)       *solver,
                                   (HYPRE_ParCSRMatrix) *A,
                                   (HYPRE_ParVector)    *b,
                                   (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdimension, HYPRE_AMSSETDIMENSION)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *dim,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetDimension( (HYPRE_Solver) *solver,
                                          (HYPRE_Int)          *dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdiscretegradient, HYPRE_AMSSETDISCRETEGRADIENT)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *G,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetDiscreteGradient( (HYPRE_Solver)       *solver,
                                                 (HYPRE_ParCSRMatrix) *G ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcoordinatevectors, HYPRE_AMSSETCOORDINATEVECTORS)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *x,
   hypre_F90_Obj *y,
   hypre_F90_Obj *z,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetCoordinateVectors( (HYPRE_Solver)    *solver,
                                                  (HYPRE_ParVector) *x,
                                                  (HYPRE_ParVector) *y,
                                                  (HYPRE_ParVector) *z ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetedgeconstantvectors, HYPRE_AMSSETEDGECONSTANTVECTORS)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *Gx,
   hypre_F90_Obj *Gy,
   hypre_F90_Obj *Gz,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetEdgeConstantVectors( (HYPRE_Solver)    *solver,
                                                    (HYPRE_ParVector) *Gx,
                                                    (HYPRE_ParVector) *Gy,
                                                    (HYPRE_ParVector) *Gz ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphapoissonmatrix, HYPRE_AMSSETALPHAPOISSONMATRIX)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A_alpha,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetAlphaPoissonMatrix( (HYPRE_Solver)       *solver,
                                                   (HYPRE_ParCSRMatrix) *A_alpha ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetapoissonmatrix, HYPRE_AMSSETBETAPOISSONMATRIX)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A_beta,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetBetaPoissonMatrix( (HYPRE_Solver)       *solver,
                                                  (HYPRE_ParCSRMatrix) *A_beta ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetmaxiter, HYPRE_AMSSETMAXITER)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *maxiter,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetMaxIter( (HYPRE_Solver) *solver,
                                        (HYPRE_Int)          *maxiter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssettol, HYPRE_AMSSETTOL)(
   hypre_F90_Obj *solver,
   double   *tol,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetTol( (HYPRE_Solver) *solver,
                                    (double)       *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcycletype, HYPRE_AMSSETCYCLETYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *cycle_type,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetCycleType( (HYPRE_Solver) *solver,
                                          (HYPRE_Int)          *cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetprintlevel, HYPRE_AMSSETPRINTLEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *print_level,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetPrintLevel( (HYPRE_Solver) *solver,
                                           (HYPRE_Int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetsmoothingoptions, HYPRE_AMSSETSMOOTHINGOPTIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_type,
   HYPRE_Int      *relax_times,
   double   *relax_weight,
   double   *omega,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetSmoothingOptions( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *relax_type,
                                                 (HYPRE_Int)          *relax_times,
                                                 (double)       *relax_weight,
                                                 (double)       *omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphaamgoptions, HYPRE_AMSSETALPHAAMGOPTIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *alpha_coarsen_type,
   HYPRE_Int      *alpha_agg_levels,
   HYPRE_Int      *alpha_relax_type,
   double   *alpha_strength_threshold,
   HYPRE_Int      *alpha_interp_type,
   HYPRE_Int      *alpha_Pmax,
   HYPRE_Int      *ierr)

{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetAlphaAMGOptions( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *alpha_coarsen_type,
                                                (HYPRE_Int)          *alpha_agg_levels,
                                                (HYPRE_Int)          *alpha_relax_type,
                                                (double)       *alpha_strength_threshold,
                                                (HYPRE_Int)          *alpha_interp_type,
                                                (HYPRE_Int)          *alpha_Pmax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetaamgoptions, HYPRE_AMSSETBETAAMGOPTIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *beta_coarsen_type,
   HYPRE_Int      *beta_agg_levels,
   HYPRE_Int      *beta_relax_type,
   double   *beta_strength_threshold,
   HYPRE_Int      *beta_interp_type,
   HYPRE_Int      *beta_Pmax,
   HYPRE_Int      *ierr)

{
   *ierr = (HYPRE_Int) ( HYPRE_AMSSetBetaAMGOptions( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *beta_coarsen_type,
                                               (HYPRE_Int)          *beta_agg_levels,
                                               (HYPRE_Int)          *beta_relax_type,
                                               (double)       *beta_strength_threshold,
                                               (HYPRE_Int)          *beta_interp_type,
                                               (HYPRE_Int)          *beta_Pmax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetnumiterations, HYPRE_AMSGETNUMITERATIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_iterations,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSGetNumIterations( (HYPRE_Solver) *solver,
                                              (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetfinalrelativeresidualnorm, HYPRE_AMSGETFINALRELATIVERESIDUALNORM)(
   hypre_F90_Obj *solver,
   double   *rel_resid_norm,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                          (double *)      rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsconstructdiscretegradient, HYPRE_AMSCONSTRUCTDISCRETEGRADIENT)(
   hypre_F90_Obj *A,
   hypre_F90_Obj *x_coord,
   HYPRE_Int      *edge_vertex,
   HYPRE_Int      *edge_orientation,
   hypre_F90_Obj *G,
   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_AMSConstructDiscreteGradient( (HYPRE_ParCSRMatrix)   *A,
                                                       (HYPRE_ParVector)      *x_coord,
                                                       (HYPRE_Int *)                 edge_vertex,
                                                       (HYPRE_Int)                  *edge_orientation,
                                                       (HYPRE_ParCSRMatrix *)  G ) );
}
