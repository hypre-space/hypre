/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructSSAMG interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgcreate, HYPRE_SSTRUCTSSAMGCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgdestroy, HYPRE_SSTRUCTSSAMGDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetup, HYPRE_SSTRUCTSSAMGSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetup(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsolve, HYPRE_SSTRUCTSSAMGSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSolve(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsettol, HYPRE_SSTRUCTSSAMGSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetmaxiter, HYPRE_SSTRUCTSSAMGSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetMaxIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetrelchang, HYPRE_SSTRUCTSSAMGSETRELCHANG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *rel_change,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetRelChange(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetzerogues, HYPRE_SSTRUCTSSAMGSETZEROGUES)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetnonzerog, HYPRE_SSTRUCTSSAMGSETNONZEROG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetNonZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetrelaxtyp, HYPRE_SSTRUCTSSAMGSETRELAXTYP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetRelaxType(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetweight, HYPRE_SSTRUCTSSAMGSETWEIGHT)
   (hypre_F90_Obj  *solver,
    hypre_F90_Real *weight,
    hypre_F90_Int  *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetWeight(
          hypre_F90_PassObj  (HYPRE_SStructSolver, solver),
          hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetnumprere, HYPRE_SSTRUCTSSAMGSETNUMPRERE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_pre_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetNumPreRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetnumpostr, HYPRE_SSTRUCTSSAMGSETNUMPOSTR)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_post_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetNumPostRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetskiprela, HYPRE_SSTRUCTSSAMGSETSKIPRELA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *skip_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetSkipRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (skip_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetDxyz
 *
 * TODO: hypre_F90_PassRealArray (dxyz) might be wrong
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetdxyz, HYPRE_SSTRUCTSSAMGSETDXYZ)
   (hypre_F90_Obj        *solver,
    hypre_F90_Int         nparts,
    hypre_F90_RealArray **dxyz,
    hypre_F90_Int        *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetDxyz(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (skip_relax),
          hypre_F90_PassRealArray (dxyz)   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetlogging, HYPRE_SSTRUCTSSAMGSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetLogging(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
  HYPRE_SStructSSAMGSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamgsetprintlev, HYPRE_SSTRUCTSSAMGSETPRINTLEV)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGSetPrintLevel(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamggetnumitera, HYPRE_SSTRUCTSSAMGGETNUMITERA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGGetNumIterations(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSSAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructssamggetfinalrel, HYPRE_SSTRUCTSSAMGGETFINALREL)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassRealRef (norm)   ));
}

#ifdef __cplusplus
}
#endif
