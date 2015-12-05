/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructFlexGMRES interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmrescreate, HYPRE_SSTRUCTFLEXGMRESCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresdestroy, HYPRE_SSTRUCTFLEXGMRESDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetup, HYPRE_SSTRUCTFLEXGMRESSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetup(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressolve, HYPRE_SSTRUCTFLEXGMRESSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSolve(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetkdim, HYPRE_SSTRUCTFLEXGMRESSETKDIM)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *k_dim,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetKDim(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (k_dim) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressettol, HYPRE_SSTRUCTFLEXGMRESSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (tol) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetabsolutetol, HYPRE_SSTRUCTFLEXGMRESSETABSOLUTETOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetAbsoluteTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (tol) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetminiter, HYPRE_SSTRUCTFLEXGMRESSETMINITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *min_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetMinIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetmaxiter, HYPRE_SSTRUCTFLEXGMRESSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetMaxIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (max_iter) ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetprecond, HYPRE_SSTRUCTFLEXGMRESSETPRECOND)
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
   if(*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_SStructFlexGMRESSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructSplitSolve,
             HYPRE_SStructSplitSetup,
             hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if(*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_SStructFlexGMRESSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructSysPFMGSolve,
             HYPRE_SStructSysPFMGSetup,
             hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if(*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_SStructFlexGMRESSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructDiagScale,
             HYPRE_SStructDiagScaleSetup,
             hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }
   else if(*precond_id == 9)
   {
      *ierr = 0;
   }

   else
   {
      *ierr = -1;
   }

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetlogging, HYPRE_SSTRUCTFLEXGMRESSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetLogging(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetprintlevel, HYPRE_SSTRUCTFLEXGMRESSETPRINTLEVEL)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESSetPrintLevel(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetnumiterati, HYPRE_SSTRUCTFLEXGMRESGETNUMITERATI)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESGetNumIterations(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetfinalrelat, HYPRE_SSTRUCTFLEXGMRESGETFINALRELAT)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDblRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetresidual, HYPRE_SSTRUCTFLEXGMRESGETRESIDUAL)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *residual,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructFlexGMRESGetResidual(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          (void *)              *residual ) );
}
