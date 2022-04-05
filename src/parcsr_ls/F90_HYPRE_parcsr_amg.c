/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParAMG Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgcreate, HYPRE_BOOMERAMGCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgdestroy, HYPRE_BOOMERAMGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetup, HYPRE_BOOMERAMGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsolve, HYPRE_BOOMERAMGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsolvet, HYPRE_BOOMERAMGSOLVET)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSolveT(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrestriction, HYPRE_BOOMERAMGSETRESTRICTION)
( hypre_F90_Obj *solver,
  hypre_F90_Int *restr_par,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRestriction(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (restr_par) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxLevels, HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxlevels, HYPRE_BOOMERAMGSETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMaxLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_levels) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxlevels, HYPRE_BOOMERAMGGETMAXLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMaxLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_levels) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxCoarseSize, HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetmaxcoarsesize, HYPRE_BOOMERAMGSETMAXCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMaxCoarseSize(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_coarse_size) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxcoarsesize, HYPRE_BOOMERAMGGETMAXCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMaxCoarseSize(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_coarse_size) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinCoarseSize, HYPRE_BoomerAMGGetMinCoarseSize
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetmincoarsesize, HYPRE_BOOMERAMGSETMINCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMinCoarseSize(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_coarse_size) ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmincoarsesize, HYPRE_BOOMERAMGGETMINCOARSESIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_coarse_size,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMinCoarseSize(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (min_coarse_size) ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetstrongthrshld, HYPRE_BOOMERAMGSETSTRONGTHRSHLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *strong_threshold,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetStrongThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (strong_threshold) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetstrongthrshld, HYPRE_BOOMERAMGGETSTRONGTHRSHLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *strong_threshold,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetStrongThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (strong_threshold) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxrowsum, HYPRE_BOOMERAMGSETMAXROWSUM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *max_row_sum,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMaxRowSum(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (max_row_sum) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxrowsum, HYPRE_BOOMERAMGGETMAXROWSUM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *max_row_sum,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMaxRowSum(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (max_row_sum) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettruncfactor, HYPRE_BOOMERAMGSETTRUNCFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetTruncFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettruncfactor, HYPRE_BOOMERAMGGETTRUNCFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetTruncFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpmaxelmts, HYPRE_BOOMERAMGSETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetPMaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetpmaxelmts, HYPRE_BOOMERAMGGETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetPMaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetJacobiTruncThreshold, HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetjacobitrunc, HYPRE_BOOMERAMGSETJACOBITRUNC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetJacobiTruncThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetjacobitrunc, HYPRE_BOOMERAMGGETJACOBITRUNC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetJacobiTruncThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPostInterpType, HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpostinterp, HYPRE_BOOMERAMGSETPOSTINTERP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *type,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetPostInterpType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (type) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterptype, HYPRE_BOOMERAMGSETINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *interp_type,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetInterpType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (interp_type) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetsepweight, HYPRE_BOOMERAMGSETSEPWEIGHT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sep_weight,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSepWeight(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (sep_weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetminiter, HYPRE_BOOMERAMGSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMinIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxIter, HYPRE_BoomerAMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxiter, HYPRE_BOOMERAMGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxiter, HYPRE_BOOMERAMGGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType, HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcoarsentype, HYPRE_BOOMERAMGSETCOARSENTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetCoarsenType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (coarsen_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcoarsentype, HYPRE_BOOMERAMGGETCOARSENTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetCoarsenType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (coarsen_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType, HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmeasuretype, HYPRE_BOOMERAMGSETMEASURETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *measure_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMeasureType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (measure_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmeasuretype, HYPRE_BOOMERAMGGETMEASURETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *measure_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMeasureType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (measure_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOldDefault
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetolddefault, HYPRE_BOOMERAMGSETOLDDEFAULT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetOldDefault(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsetuptype, HYPRE_BOOMERAMGSETSETUPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *setup_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSetupType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (setup_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleType, HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcycletype, HYPRE_BOOMERAMGSETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetCycleType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cycle_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcycletype, HYPRE_BOOMERAMGGETCYCLETYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cycle_type,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetCycleType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (cycle_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol, HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettol, HYPRE_BOOMERAMGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettol, HYPRE_BOOMERAMGGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 * DEPRECATED.  Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumgridsweeps, HYPRE_BOOMERAMGSETNUMGRIDSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_grid_sweeps,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNumGridSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_grid_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsweeps, HYPRE_BOOMERAMGSETNUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNumSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleNumSweeps, HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclenumsweeps, HYPRE_BOOMERAMGSETCYCLENUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetCycleNumSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (num_sweeps),
                hypre_F90_PassInt (k) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclenumsweeps, HYPRE_BOOMERAMGGETCYCLENUMSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_sweeps,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetCycleNumSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_sweeps),
                hypre_F90_PassInt (k) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGInitGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays that are allocated.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramginitgridrelaxatn, HYPRE_BOOMERAMGINITGRIDRELAXATN)
( hypre_F90_Obj *num_grid_sweeps,
  hypre_F90_Obj *grid_relax_type,
  hypre_F90_Obj *grid_relax_points,
  hypre_F90_Int *coarsen_type,
  hypre_F90_Obj *relax_weights,
  hypre_F90_Int *max_levels,
  hypre_F90_Int *ierr               )
{
   *num_grid_sweeps   = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int*,  1, HYPRE_MEMORY_HOST);
   *grid_relax_type   = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int*,  1, HYPRE_MEMORY_HOST);
   *grid_relax_points = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int**,  1, HYPRE_MEMORY_HOST);
   *relax_weights     = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Real*,  1, HYPRE_MEMORY_HOST);

   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGInitGridRelaxation(
                (HYPRE_Int **)    *num_grid_sweeps,
                (HYPRE_Int **)    *grid_relax_type,
                (HYPRE_Int ***)   *grid_relax_points,
                hypre_F90_PassInt (coarsen_type),
                (HYPRE_Real **) *relax_weights,
                hypre_F90_PassInt (max_levels)         ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGFinalizeGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgfingridrelaxatn, HYPRE_BOOMERAMGFINGRIDRELAXATN)
( hypre_F90_Obj *num_grid_sweeps,
  hypre_F90_Obj *grid_relax_type,
  hypre_F90_Obj *grid_relax_points,
  hypre_F90_Obj *relax_weights,
  hypre_F90_Int *ierr               )
{
   char *ptr_num_grid_sweeps   = (char *) *num_grid_sweeps;
   char *ptr_grid_relax_type   = (char *) *grid_relax_type;
   char *ptr_grid_relax_points = (char *) *grid_relax_points;
   char *ptr_relax_weights     = (char *) *relax_weights;

   hypre_TFree(ptr_num_grid_sweeps, HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_grid_relax_type, HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_grid_relax_points, HYPRE_MEMORY_HOST);
   hypre_TFree(ptr_relax_weights, HYPRE_MEMORY_HOST);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxtype, HYPRE_BOOMERAMGSETGRIDRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *grid_relax_type,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetGridRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (grid_relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxtype, HYPRE_BOOMERAMGSETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleRelaxType, HYPRE_BoomerAMGGetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclerelaxtype, HYPRE_BOOMERAMGSETCYCLERELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetCycleRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type),
                hypre_F90_PassInt (k) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclerelaxtype, HYPRE_BOOMERAMGGETCYCLERELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *k,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetCycleRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (relax_type),
                hypre_F90_PassInt (k)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxorder, HYPRE_BOOMERAMGSETRELAXORDER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_order,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRelaxOrder(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_order) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There is no alternative function.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxpnts, HYPRE_BOOMERAMGSETGRIDRELAXPNTS)
( hypre_F90_Obj *solver,
  HYPRE_Int      **grid_relax_points,
  hypre_F90_Int *ierr               )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetGridRelaxPoints(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                (HYPRE_Int **)        grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxweight, HYPRE_BOOMERAMGSETRELAXWEIGHT)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *relax_weights,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRelaxWeight(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealArray (relax_weights) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxwt, HYPRE_BOOMERAMGSETRELAXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *relax_weight,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRelaxWt(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (relax_weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelrelaxwt, HYPRE_BOOMERAMGSETLEVELRELAXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *relax_weight,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetLevelRelaxWt(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (relax_weight),
                hypre_F90_PassInt (level) ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetouterwt, HYPRE_BOOMERAMGSETOUTERWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *outer_wt,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetOuterWt(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (outer_wt) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelouterwt, HYPRE_BOOMERAMGSETLEVELOUTERWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *outer_wt,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetLevelOuterWt(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (outer_wt),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothType, HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothtype, HYPRE_BOOMERAMGSETSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSmoothType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothtype, HYPRE_BOOMERAMGGETSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetSmoothType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumLvls, HYPRE_BoomerAMGGetSmoothNumLvls
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumlvls, HYPRE_BOOMERAMGSETSMOOTHNUMLVLS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSmoothNumLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_num_levels) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumlvls, HYPRE_BOOMERAMGGETSMOOTHNUMLVLS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetSmoothNumLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_num_levels) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSwps, HYPRE_BoomerAMGGetSmoothNumSwps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumswps, HYPRE_BOOMERAMGSETSMOOTHNUMSWPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_sweeps,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSmoothNumSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (smooth_num_sweeps) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumswps, HYPRE_BOOMERAMGGETSMOOTHNUMSWPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *smooth_num_sweeps,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetSmoothNumSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (smooth_num_sweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogging, HYPRE_BOOMERAMGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetlogging, HYPRE_BOOMERAMGGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintLevel, HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintlevel, HYPRE_BOOMERAMGSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetprintlevel, HYPRE_BOOMERAMGGETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintfilename, HYPRE_BOOMERAMGSETPRINTFILENAME)
( hypre_F90_Obj *solver,
  char     *print_file_name,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetPrintFileName(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                (char *)        print_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag, HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdebugflag, HYPRE_BOOMERAMGSETDEBUGFLAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *debug_flag,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetDebugFlag(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (debug_flag) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdebugflag, HYPRE_BOOMERAMGGETDEBUGFLAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *debug_flag,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetDebugFlag(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (debug_flag) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetnumiterations, HYPRE_BOOMERAMGGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetcumnumiterati, HYPRE_BOOMERAMGGETCUMNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cum_num_iterations,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetCumNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (cum_num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetresidual, HYPRE_BOOMERAMGGETRESIDUAL)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *residual,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetResidual(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObjRef (HYPRE_ParVector, residual)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetFinalRelativeResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetfinalreltvres, HYPRE_BOOMERAMGGETFINALRELTVRES)
( hypre_F90_Obj *solver,
  hypre_F90_Real *rel_resid_norm,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (rel_resid_norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant, HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetvariant, HYPRE_BOOMERAMGSETVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetVariant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (variant) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetvariant, HYPRE_BOOMERAMGGETVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetVariant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (variant) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetoverlap, HYPRE_BOOMERAMGSETOVERLAP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *overlap,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetOverlap(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (overlap) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetoverlap, HYPRE_BOOMERAMGGETOVERLAP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *overlap,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetOverlap(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (overlap) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType, HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdomaintype, HYPRE_BOOMERAMGSETDOMAINTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *domain_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetDomainType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (domain_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdomaintype, HYPRE_BOOMERAMGGETDOMAINTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *domain_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetDomainType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (domain_type) ) );
}

void
hypre_F90_IFACE(hypre_boomeramgsetschwarznonsym, HYPRE_BOOMERAMGSETSCHWARZNONSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *schwarz_non_symm,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSchwarzUseNonSymm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (schwarz_non_symm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWt, HYPRE_BoomerAMGGetSchwarzRlxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetschwarzrlxwt, HYPRE_BOOMERAMGSETSCHWARZRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *schwarz_rlx_weight,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSchwarzRlxWeight(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (schwarz_rlx_weight)) );
}

void
hypre_F90_IFACE(hypre_boomeramggetschwarzrlxwt, HYPRE_BOOMERAMGGETSCHWARZRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *schwarz_rlx_weight,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetSchwarzRlxWeight(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (schwarz_rlx_weight)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsym, HYPRE_BOOMERAMGSETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSym(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (sym) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevel, HYPRE_BOOMERAMGSETLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetthreshold, HYPRE_BOOMERAMGSETTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetfilter, HYPRE_BOOMERAMGSETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetFilter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (filter)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdroptol, HYPRE_BOOMERAMGSETDROPTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *drop_tol,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetDropTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (drop_tol)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxnzperrow, HYPRE_BOOMERAMGSETMAXNZPERROW)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_nz_per_row,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMaxNzPerRow(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_nz_per_row) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteubj, HYPRE_BOOMERAMGSETEUBJ)
( hypre_F90_Obj *solver,
  hypre_F90_Int *eu_bj,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetEuBJ(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (eu_bj) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteulevel, HYPRE_BOOMERAMGSETEULEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *eu_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetEuLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (eu_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteusparsea, HYPRE_BOOMERAMGSETEUSPARSEA)
( hypre_F90_Obj *solver,
  hypre_F90_Real *eu_sparse_a,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetEuSparseA(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (eu_sparse_a)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteuclidfile, HYPRE_BOOMERAMGSETEUCLIDFILE)
( hypre_F90_Obj *solver,
  char     *euclidfile,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetEuclidFile(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                (char *)        euclidfile ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions, HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnumfunctions, HYPRE_BOOMERAMGSETNUMFUNCTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_functions,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNumFunctions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (num_functions) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetnumfunctions, HYPRE_BOOMERAMGGETNUMFUNCTIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_functions,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetNumFunctions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodal, HYPRE_BOOMERAMGSETNODAL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nodal,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNodal(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nodal) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodaldiag, HYPRE_BOOMERAMGSETNODALDIAG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nodal_diag,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNodalDiag(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nodal_diag) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdoffunc, HYPRE_BOOMERAMGSETDOFFUNC)
( hypre_F90_Obj *solver,
  hypre_F90_IntArray *dof_func,
  hypre_F90_Int *ierr             )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetDofFunc(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (dof_func) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumpaths, HYPRE_BOOMERAMGSETNUMPATHS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_paths,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNumPaths(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (num_paths) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggnumlevels, HYPRE_BOOMERAMGSETAGGNUMLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *agg_num_levels,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggNumLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (agg_num_levels) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetagginterptype, HYPRE_BOOMERAMGSETAGGINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *agg_interp_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggInterpType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (agg_interp_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggtrfactor, HYPRE_BOOMERAMGSETAGGTRFACTOR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggTruncFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggp12trfac, HYPRE_BOOMERAMGSETAGGP12TRFAC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *trunc_factor,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggP12TruncFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (trunc_factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggpmaxelmts, HYPRE_BOOMERAMGSETAGGPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggPMaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggp12maxelmt, HYPRE_BOOMERAMGSETAGGP12MAXELMT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *p_max_elmts,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAggP12MaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (p_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVectors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecs, HYPRE_BOOMERAMGSETINTERPVECS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_vectors,
  hypre_F90_Obj *interp_vectors,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetInterpVectors(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (num_vectors),
                hypre_F90_PassObjRef (HYPRE_ParVector, interp_vectors) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecvar, HYPRE_BOOMERAMGSETINTERPVECVAR)
( hypre_F90_Obj *solver,
  hypre_F90_Int *var,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetInterpVecVariant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (var) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecQMax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecqmx, HYPRE_BOOMERAMGSETINTERPVECQMX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *q_max,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetInterpVecQMax(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (q_max) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpVecAbsQTrunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterpvecqtr, HYPRE_BOOMERAMGSETINTERPVECQTR)
( hypre_F90_Obj *solver,
  hypre_F90_Real *q_trunc,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetInterpVecAbsQTrunc(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (q_trunc) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyorder, HYPRE_BOOMERAMGSETCHEBYORDER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_order,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetChebyOrder(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_order) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyFraction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyfract, HYPRE_BOOMERAMGSETCHEBYFRACT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *cheby_fraction,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetChebyFraction(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (cheby_fraction) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyscale, HYPRE_BOOMERAMGSETCHEBYSCALE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_scale,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetChebyScale(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_scale) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyvariant, HYPRE_BOOMERAMGSETCHEBYVARIANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_variant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetChebyVariant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_variant) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetChebyEigEst
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetchebyeigest, HYPRE_BOOMERAMGSETCHEBYEIGEST)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cheby_eig_est,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetChebyEigEst(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cheby_eig_est) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetKeepTranspose
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetkeeptransp, HYPRE_BOOMERAMGSETKEEPTRANSP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *keep_transpose,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetKeepTranspose(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (keep_transpose) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRAP2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrap2, HYPRE_BOOMERAMGSETRAP2)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rap2,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRAP2(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (rap2) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAdditive, HYPRE_BoomerAMGGetAdditive
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetadditive, HYPRE_BOOMERAMGSETADDITIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAdditive(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetadditive, HYPRE_BOOMERAMGGETADDITIVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetAdditive(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMultAdditive, HYPRE BoomerAMGGetMultAdditive
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultadd, HYPRE_BOOMERAMGSETMULTADD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMultAdditive(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmultadd, HYPRE_BOOMERAMGGETMULTADD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetMultAdditive(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSimple, HYPRE_BoomerAMGGetSimple
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsimple, HYPRE_BOOMERAMGSETSIMPLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSimple(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_lvl) ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsimple, HYPRE_BOOMERAMGGETSIMPLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGGetSimple(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (add_lvl) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAddLastLvl
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddlastlvl, HYPRE_BOOMERAMGSETADDLASTLVL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_last_lvl,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAddLastLvl(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_last_lvl) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMultAddTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultaddtrf, HYPRE_BOOMERAMGSETMULTADDTRF)
( hypre_F90_Obj *solver,
  hypre_F90_Real *add_tr,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMultAddTruncFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (add_tr) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMultAddPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmultaddpmx, HYPRE_BOOMERAMGSETMULTADDPMX)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_pmx,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetMultAddPMaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_pmx) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAddRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddrlxtype, HYPRE_BOOMERAMGSETADDRLXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *add_rlx_type,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAddRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (add_rlx_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAddRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaddrlxwt, HYPRE_BOOMERAMGSETADDRLXWT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *add_rlx_wt,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetAddRelaxWt(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (add_rlx_wt) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSeqThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetseqthrshold, HYPRE_BOOMERAMGSETSEQTHRSHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *seq_th,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetSeqThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (seq_th) ) );
}

#ifdef HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDSLUThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdsluthrshold, HYPRE_BOOMERAMGSETDSLUTHRSHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dslu_th,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetDSLUThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (dslu_th) ) );
}
#endif

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRedundant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetredundant, HYPRE_BOOMERAMGSETREDUNDANT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *redundant,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetRedundant(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (redundant) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnongaltol, HYPRE_BOOMERAMGSETNONGALTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *nongal_tol,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNonGalerkinTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (nongal_tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelNonGalerkinTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlvlnongaltol, HYPRE_BOOMERAMGSETLVLNONGALTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *nongal_tol,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetLevelNonGalerkinTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (nongal_tol),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgsmg, HYPRE_BOOMERAMGSETGSMG)
( hypre_F90_Obj *solver,
  hypre_F90_Int *gsmg,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetGSMG(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsamples, HYPRE_BOOMERAMGSETNUMSAMPLES)
( hypre_F90_Obj *solver,
  hypre_F90_Int *gsmg,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetNumSamples(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (gsmg) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcgcits, HYPRE_BOOMERAMGSETCGCITS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *its,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_BoomerAMGSetCGCIts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (its) ) );
}

#ifdef __cplusplus
}
#endif
