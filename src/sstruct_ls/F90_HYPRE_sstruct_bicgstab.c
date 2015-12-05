/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabcreate, HYPRE_SSTRUCTBICGSTABCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) )) ;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabdestroy, HYPRE_SSTRUCTBICGSTABDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetup, HYPRE_SSTRUCTBICGSTABSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetup(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsolve, HYPRE_SSTRUCTBICGSTABSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSolve(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsettol, HYPRE_SSTRUCTBICGSTABSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (tol) ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetAnsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetabsolutetol, HYPRE_SSTRUCTBICGSTABSETABSOLUTETOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetAbsoluteTol(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDbl (tol) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetminiter, HYPRE_SSTRUCTBICGSTABSETMINITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *min_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetMinIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (min_iter) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetmaxiter, HYPRE_SSTRUCTBICGSTABSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetMaxIter(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetstopcri, HYPRE_SSTRUCTBICGSTABSETSTOPCRI)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *stop_crit,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetStopCrit(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (stop_crit) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprecond, HYPRE_SSTRUCTBICGSTABSETPRECOND)
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
         (HYPRE_SStructBiCGSTABSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructSplitSolve,
             HYPRE_SStructSplitSetup,
             hypre_F90_PassObjRef (HYPRE_SStructSolver, precond_solver)));
   }

   else if(*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_SStructBiCGSTABSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructSysPFMGSolve,
             HYPRE_SStructSysPFMGSetup,
             hypre_F90_PassObj (HYPRE_SStructSolver, precond_solver)));
   }

   else if(*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_SStructBiCGSTABSetPrecond(
             hypre_F90_PassObj (HYPRE_SStructSolver, solver),
             HYPRE_SStructDiagScale,
             HYPRE_SStructDiagScaleSetup,
             hypre_F90_PassObj (HYPRE_SStructSolver, precond_solver)));
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
 * HYPRE_SStructBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetlogging, HYPRE_SSTRUCTBICGSTABSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetLogging(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprintle, HYPRE_SSTRUCTBICGSTABSETPRINTLE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABSetPrintLevel(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetnumiter, HYPRE_SSTRUCTBICGSTABGETNUMITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABGetNumIterations(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetfinalre, HYPRE_SSTRUCTBICGSTABGETFINALRE)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassDblRef (norm) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetresidua, HYPRE_SSTRUCTBICGSTABGETRESIDUA)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *residual,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructBiCGSTABGetResidual(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          (void *)              *residual));
}
