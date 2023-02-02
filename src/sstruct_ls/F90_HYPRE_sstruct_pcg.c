/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructPCG interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgcreate, HYPRE_SSTRUCTPCGCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgdestroy, HYPRE_SSTRUCTPCGDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGDestroy(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetup, HYPRE_SSTRUCTPCGSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetup(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, b),
               hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsolve, HYPRE_SSTRUCTPCGSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSolve(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, b),
               hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettol, HYPRE_SSTRUCTPCGSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetTol(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetabsolutetol, HYPRE_SSTRUCTPCGSETABSOLUTETOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetAbsoluteTol(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassReal (tol) ) );
}
/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetmaxiter, HYPRE_SSTRUCTPCGSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetMaxIter(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettwonorm, HYPRE_SSTRUCTPCGSETTWONORM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *two_norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetTwoNorm(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetrelchange, HYPRE_SSTRUCTPCGSETRELCHANGE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetRelChange(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprecond, HYPRE_SSTRUCTPCGSETPRECOND)
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
              (HYPRE_SStructPCGSetPrecond(
                  hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                  HYPRE_SStructSplitSolve,
                  HYPRE_SStructSplitSetup,
                  hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              (HYPRE_SStructPCGSetPrecond(
                  hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                  HYPRE_SStructSysPFMGSolve,
                  HYPRE_SStructSysPFMGSetup,
                  hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              (HYPRE_SStructPCGSetPrecond(
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
 *  HYPRE_SStructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetlogging, HYPRE_SSTRUCTPCGSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetLogging(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprintlevel, HYPRE_SSTRUCTPCGSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGSetPrintLevel(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetnumiteration, HYPRE_SSTRUCTPCGGETNUMITERATION)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGGetNumIterations(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetfinalrelativ, HYPRE_SSTRUCTPCGGETFINALRELATIV)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGGetFinalRelativeResidualNorm(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassRealRef (norm) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetresidual, HYPRE_SSTRUCTPCGGETRESIDUAL)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *residual,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructPCGGetResidual(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               (void **)              *residual ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructdiagscalesetup, HYPRE_SSTRUCTDIAGSCALESETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *y,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructDiagScaleSetup(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, y),
               hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructdiagscale, HYPRE_SSTRUCTDIAGSCALE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *y,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructDiagScale(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassObj (HYPRE_SStructMatrix, A),
               hypre_F90_PassObj (HYPRE_SStructVector, y),
               hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

#ifdef __cplusplus
}
#endif
