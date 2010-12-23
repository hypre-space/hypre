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
 * HYPRE_SStructGMRES interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmrescreate, HYPRE_SSTRUCTGMRESCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj  *solver,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_SStructGMRESCreate(
      hypre_F90_PassComm(comm), hypre_F90_PassObjRef(HYPRE_SStructSolver,solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresdestroy, HYPRE_SSTRUCTGMRESDESTROY)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetup, HYPRE_SSTRUCTGMRESSETUP)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *A,
                                                        hypre_F90_Obj *b,
                                                        hypre_F90_Obj *x,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressolve, HYPRE_SSTRUCTGMRESSOLVE)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *A,
                                                        hypre_F90_Obj *b,
                                                        hypre_F90_Obj *x,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetkdim, HYPRE_SSTRUCTGMRESSETKDIM)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *k_dim,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_Int)                 *k_dim ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressettol, HYPRE_SSTRUCTGMRESSETTOL)
                                                       (hypre_F90_Obj *solver,
                                                        double    *tol,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetabsolutetol, HYPRE_SSTRUCTGMRESSETABSOLUTETOL)
                                                       (hypre_F90_Obj *solver,
                                                        double    *tol,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetminiter, HYPRE_SSTRUCTGMRESSETMINITER)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *min_iter,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetmaxiter, HYPRE_SSTRUCTGMRESSETMAXITER)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *max_iter,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetstopcrit, HYPRE_SSTRUCTGMRESSETSTOPCRIT)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *stop_crit,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetStopCrit( (HYPRE_SStructSolver) *solver,
                                                  (HYPRE_Int)                *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprecond, HYPRE_SSTRUCTGMRESSETPRECOND)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *precond_id,
                                                      hypre_F90_Obj *precond_solver,
                                                      HYPRE_Int      *ierr)
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
       *ierr = (HYPRE_Int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructDiagScale,
                                               HYPRE_SStructDiagScaleSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
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
 * HYPRE_SStructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetlogging, HYPRE_SSTRUCTGMRESSETLOGGING)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *logging,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprintlevel, HYPRE_SSTRUCTGMRESSETPRINTLEVEL)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *level,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetnumiterati, HYPRE_SSTRUCTGMRESGETNUMITERATI)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int *num_iterations,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (HYPRE_Int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetfinalrelat, HYPRE_SSTRUCTGMRESGETFINALRELAT)
                                                       (hypre_F90_Obj *solver,
                                                        double    *norm,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetresidual, HYPRE_SSTRUCTGMRESGETRESIDUAL)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *residual,
                                                        HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
