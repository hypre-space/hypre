/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructSysPFMG interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgcreate, HYPRE_SSTRUCTSYSPFMGCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgdestroy, HYPRE_SSTRUCTSYSPFMGDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetup, HYPRE_SSTRUCTSYSPFMGSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetup(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsolve, HYPRE_SSTRUCTSYSPFMGSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSolve(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsettol, HYPRE_SSTRUCTSYSPFMGSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetmaxiter, HYPRE_SSTRUCTSYSPFMGSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetMaxIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelchang, HYPRE_SSTRUCTSYSPFMGSETRELCHANG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *rel_change,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetRelChange(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetzerogues, HYPRE_SSTRUCTSYSPFMGSETZEROGUES)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnonzerog, HYPRE_SSTRUCTSYSPFMGSETNONZEROG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetNonZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelaxtyp, HYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetRelaxType(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
void
hypre_F90_IFACE(hypre_sstructsyspfmgsetjacobiweigh, HYPRE_SSTRUCTSYSPFMGSETJACOBIWEIGH)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *weight,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetJacobiWeight(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumprere, HYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_pre_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetNumPreRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumpostr, HYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_post_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetNumPostRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetskiprela, HYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *skip_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetSkipRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (skip_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetdxyz, HYPRE_SSTRUCTSYSPFMGSETDXYZ)
   (hypre_F90_Obj *solver,
    hypre_F90_DblArray *dxyz,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetDxyz(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDblArray (dxyz)   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetlogging, HYPRE_SSTRUCTSYSPFMGSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetLogging(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
  HYPRE_SStructSysPFMGSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetprintlev, HYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGSetPrintLevel(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetnumitera, HYPRE_SSTRUCTSYSPFMGGETNUMITERA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGGetNumIterations(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetfinalrel, HYPRE_SSTRUCTSYSPFMGGETFINALREL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDblRef (norm)   ));
}
