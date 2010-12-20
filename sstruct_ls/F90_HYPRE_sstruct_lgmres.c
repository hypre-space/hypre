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
 * HYPRE_SStructLGMRES interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmrescreate, HYPRE_SSTRUCTLGMRESCREATE)
                                                       (HYPRE_Int     *comm,
                                                        hypre_F90_Obj *solver,
                                                        HYPRE_Int     *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESCreate( (MPI_Comm)              *comm,
                                            (HYPRE_SStructSolver *)  solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresdestroy, HYPRE_SSTRUCTLGMRESDESTROY)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESDestroy( (HYPRE_SStructSolver) *solver ));
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
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
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
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetkdim, HYPRE_SSTRUCTLGMRESSETKDIM)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *k_dim,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_Int)                 *k_dim ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetaugdim, HYPRE_SSTRUCTLGMRESSETAUGDIM)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *aug_dim,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetAugDim( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_Int)                 *aug_dim ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressettol, HYPRE_SSTRUCTLGMRESSETTOL)
                                                       (hypre_F90_Obj *solver,
                                                        double    *tol,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetabsolutetol, HYPRE_SSTRUCTLGMRESSETABSOLUTETOL)
                                                       (hypre_F90_Obj *solver,
                                                        double    *tol,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetminiter, HYPRE_SSTRUCTLGMRESSETMINITER)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *min_iter,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetmaxiter, HYPRE_SSTRUCTLGMRESSETMAXITER)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *max_iter,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprecond, HYPRE_SSTRUCTLGMRESSETPRECOND)
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
               (HYPRE_SStructLGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructLGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructLGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
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
 * HYPRE_SStructLGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetlogging, HYPRE_SSTRUCTLGMRESSETLOGGING)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *logging,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprintlevel, HYPRE_SSTRUCTLGMRESSETPRINTLEVEL)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *level,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetnumiterati, HYPRE_SSTRUCTLGMRESGETNUMITERATI)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *num_iterations,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (HYPRE_Int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetfinalrelat, HYPRE_SSTRUCTLGMRESGETFINALRELAT)
                                                       (hypre_F90_Obj *solver,
                                                        double    *norm,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetresidual, HYPRE_SSTRUCTLGMRESGETRESIDUAL)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *residual,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructLGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
