/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructLGMRES interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmrescreate, HYPRE_SSTRUCTLGMRESCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresdestroy, HYPRE_SSTRUCTLGMRESDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESDestroy(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetup, HYPRE_SSTRUCTLGMRESSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetup(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, b),
               hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressolve, HYPRE_SSTRUCTLGMRESSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSolve(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, b),
               hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetkdim, HYPRE_SSTRUCTLGMRESSETKDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *k_dim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetKDim(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (k_dim) ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetaugdim, HYPRE_SSTRUCTLGMRESSETAUGDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *aug_dim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetAugDim(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (aug_dim) ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressettol, HYPRE_SSTRUCTLGMRESSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetTol(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetabsolutetol, HYPRE_SSTRUCTLGMRESSETABSOLUTETOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetAbsoluteTol(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetminiter, HYPRE_SSTRUCTLGMRESSETMINITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *min_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetMinIter(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetmaxiter, HYPRE_SSTRUCTLGMRESSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetMaxIter(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter) ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprecond, HYPRE_SSTRUCTLGMRESSETPRECOND)
(hypre_F90_Obj *solver,
 hypre_F90_Int *precond_id,
 hypre_F90_Obj *precond_solver,
 hypre_F90_Int *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
              (HYPRE_SStructLGMRESSetPrecond(
                  hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                  HYPRE_SStructSplitSolve,
                  HYPRE_SStructSplitSetup,
                  hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              (HYPRE_SStructLGMRESSetPrecond(
                  hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                  HYPRE_SStructSysPFMGSolve,
                  HYPRE_SStructSysPFMGSetup,
                  hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              (HYPRE_SStructLGMRESSetPrecond(
                  hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                  HYPRE_SStructDiagScale,
                  HYPRE_SStructDiagScaleSetup,
                  hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }

   else
   {
      *ierr = -1;
   }

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetlogging, HYPRE_SSTRUCTLGMRESSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetLogging(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprintlevel, HYPRE_SSTRUCTLGMRESSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESSetPrintLevel(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetnumiterati, HYPRE_SSTRUCTLGMRESGETNUMITERATI)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESGetNumIterations(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetfinalrelat, HYPRE_SSTRUCTLGMRESGETFINALRELAT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetresidual, HYPRE_SSTRUCTLGMRESGETRESIDUAL)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *residual,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructLGMRESGetResidual(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

#ifdef __cplusplus
}
#endif
