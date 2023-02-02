/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_AMS Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_AMSCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amscreate, HYPRE_AMSCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsdestroy, HYPRE_AMSDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetup, HYPRE_AMSSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssolve, HYPRE_AMSSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDimension
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdimension, HYPRE_AMSSETDIMENSION)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dim,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetDimension(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (dim) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetdiscretegradient, HYPRE_AMSSETDISCRETEGRADIENT)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *G,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetDiscreteGradient(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, G) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCoordinateVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcoordinatevectors, HYPRE_AMSSETCOORDINATEVECTORS)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *x,
  hypre_F90_Obj *y,
  hypre_F90_Obj *z,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetCoordinateVectors(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParVector, x),
                hypre_F90_PassObj (HYPRE_ParVector, y),
                hypre_F90_PassObj (HYPRE_ParVector, z) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetedgeconstantvectors, HYPRE_AMSSETEDGECONSTANTVECTORS)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *Gx,
  hypre_F90_Obj *Gy,
  hypre_F90_Obj *Gz,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetEdgeConstantVectors(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParVector, Gx),
                hypre_F90_PassObj (HYPRE_ParVector, Gy),
                hypre_F90_PassObj (HYPRE_ParVector, Gz) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphapoissonmatrix, HYPRE_AMSSETALPHAPOISSONMATRIX)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A_alpha,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetAlphaPoissonMatrix(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A_alpha) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaPoissonMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetapoissonmatrix, HYPRE_AMSSETBETAPOISSONMATRIX)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A_beta,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetBetaPoissonMatrix(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A_beta) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetmaxiter, HYPRE_AMSSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *maxiter,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (maxiter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssettol, HYPRE_AMSSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetcycletype, HYPRE_AMSSETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetCycleType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetprintlevel, HYPRE_AMSSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetSmoothingOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetsmoothingoptions, HYPRE_AMSSETSMOOTHINGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *relax_times,
  hypre_F90_Real *relax_weight,
  hypre_F90_Real *omega,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetSmoothingOptions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type),
                hypre_F90_PassInt (relax_times),
                hypre_F90_PassReal (relax_weight),
                hypre_F90_PassReal (omega) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetAlphaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetalphaamgoptions, HYPRE_AMSSETALPHAAMGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *alpha_coarsen_type,
  hypre_F90_Int *alpha_agg_levels,
  hypre_F90_Int *alpha_relax_type,
  hypre_F90_Real *alpha_strength_threshold,
  hypre_F90_Int *alpha_interp_type,
  hypre_F90_Int *alpha_Pmax,
  hypre_F90_Int *ierr)

{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetAlphaAMGOptions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (alpha_coarsen_type),
                hypre_F90_PassInt (alpha_agg_levels),
                hypre_F90_PassInt (alpha_relax_type),
                hypre_F90_PassReal (alpha_strength_threshold),
                hypre_F90_PassInt (alpha_interp_type),
                hypre_F90_PassInt (alpha_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSSetBetaAMGOptions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amssetbetaamgoptions, HYPRE_AMSSETBETAAMGOPTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *beta_coarsen_type,
  hypre_F90_Int *beta_agg_levels,
  hypre_F90_Int *beta_relax_type,
  hypre_F90_Real *beta_strength_threshold,
  hypre_F90_Int *beta_interp_type,
  hypre_F90_Int *beta_Pmax,
  hypre_F90_Int *ierr)

{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSSetBetaAMGOptions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (beta_coarsen_type),
                hypre_F90_PassInt (beta_agg_levels),
                hypre_F90_PassInt (beta_relax_type),
                hypre_F90_PassReal (beta_strength_threshold),
                hypre_F90_PassInt (beta_interp_type),
                hypre_F90_PassInt (beta_Pmax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetnumiterations, HYPRE_AMSGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsgetfinalrelativeresidualnorm, HYPRE_AMSGETFINALRELATIVERESIDUALNORM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *rel_resid_norm,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_AMSConstructDiscreteGradient
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_amsconstructdiscretegradient, HYPRE_AMSCONSTRUCTDISCRETEGRADIENT)
( hypre_F90_Obj *A,
  hypre_F90_Obj *x_coord,
  hypre_F90_BigIntArray *edge_vertex,
  hypre_F90_Int *edge_orientation,
  hypre_F90_Obj *G,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_AMSConstructDiscreteGradient(
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, x_coord),
                hypre_F90_PassBigIntArray (edge_vertex),
                hypre_F90_PassInt (edge_orientation),
                hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, G) ) );
}

#ifdef __cplusplus
}
#endif
