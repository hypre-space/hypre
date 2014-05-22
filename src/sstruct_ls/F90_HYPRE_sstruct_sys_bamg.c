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
 * HYPRE_SStructSysBAMG interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgcreate, HYPRE_SSTRUCTSYSBAMGCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgdestroy, HYPRE_SSTRUCTSYSBAMGDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetup, HYPRE_SSTRUCTSYSBAMGSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetup(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsolve, HYPRE_SSTRUCTSYSBAMGSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSolve(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsettol, HYPRE_SSTRUCTSYSBAMGSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetmaxiter, HYPRE_SSTRUCTSYSBAMGSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetMaxIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetrelchang, HYPRE_SSTRUCTSYSBAMGSETRELCHANG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *rel_change,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetRelChange(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetzerogues, HYPRE_SSTRUCTSYSBAMGSETZEROGUES)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetnonzerog, HYPRE_SSTRUCTSYSBAMGSETNONZEROG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetNonZeroGuess(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetrelaxtyp, HYPRE_SSTRUCTSYSBAMGSETRELAXTYP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetRelaxType(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
void
hypre_F90_IFACE(hypre_sstructsysbamgsetjacobiweigh, HYPRE_SSTRUCTSYSBAMGSETJACOBIWEIGH)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *weight,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetJacobiWeight(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetnumprere, HYPRE_SSTRUCTSYSBAMGSETNUMPRERE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_pre_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetNumPreRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetnumpostr, HYPRE_SSTRUCTSYSBAMGSETNUMPOSTR)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_post_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetNumPostRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetskiprela, HYPRE_SSTRUCTSYSBAMGSETSKIPRELA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *skip_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetSkipRelax(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (skip_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetdxyz, HYPRE_SSTRUCTSYSBAMGSETDXYZ)
   (hypre_F90_Obj *solver,
    hypre_F90_RealArray *dxyz,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetDxyz(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassRealArray (dxyz)   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetlogging, HYPRE_SSTRUCTSYSBAMGSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetLogging(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
  HYPRE_SStructSysBAMGSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamgsetprintlev, HYPRE_SSTRUCTSYSBAMGSETPRINTLEV)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGSetPrintLevel(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamggetnumitera, HYPRE_SSTRUCTSYSBAMGGETNUMITERA)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGGetNumIterations(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysBAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsysbamggetfinalrel, HYPRE_SSTRUCTSYSBAMGGETFINALREL)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructSysBAMGGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassRealRef (norm)   ));
}

