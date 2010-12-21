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
 * HYPRE_SStructPCG interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgcreate, HYPRE_SSTRUCTPCGCREATE)
                                                     (hypre_F90_Comm *comm,
                                                      hypre_F90_Obj *solver,
                                                      HYPRE_Int     *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGCreate( (MPI_Comm)             *comm,
                                          (HYPRE_SStructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgdestroy, HYPRE_SSTRUCTPCGDESTROY)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGDestroy( (HYPRE_SStructSolver) *solver ));
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
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetup( (HYPRE_SStructSolver) *solver,
                                         (HYPRE_SStructMatrix) *A,
                                         (HYPRE_SStructVector) *b,
                                         (HYPRE_SStructVector) *x ) );
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
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSolve( (HYPRE_SStructSolver) *solver,
                                         (HYPRE_SStructMatrix) *A,
                                         (HYPRE_SStructVector) *b,
                                         (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettol, HYPRE_SSTRUCTPCGSETTOL)
                                                     (hypre_F90_Obj *solver,
                                                      double   *tol,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetTol( (HYPRE_SStructSolver) *solver,
                                          (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetabsolutetol, HYPRE_SSTRUCTPCGSETABSOLUTETOL)
                                                     (hypre_F90_Obj *solver,
                                                      double   *tol,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                          (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetmaxiter, HYPRE_SSTRUCTPCGSETMAXITER)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *max_iter,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetMaxIter( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_Int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettwonorm, HYPRE_SSTRUCTPCGSETTWONORM)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *two_norm,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetTwoNorm( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_Int)                 *two_norm ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetrelchange, HYPRE_SSTRUCTPCGSETRELCHANGE)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *rel_change,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetRelChange( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *rel_change ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprecond, HYPRE_SSTRUCTPCGSETPRECOND)
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
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSplitSolve,
                                             HYPRE_SStructSplitSetup,
                                            (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSysPFMGSolve,
                                             HYPRE_SStructSysPFMGSetup,
                                            (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
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
 *  HYPRE_SStructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetlogging, HYPRE_SSTRUCTPCGSETLOGGING)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *logging,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetLogging( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_Int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprintlevel, HYPRE_SSTRUCTPCGSETPRINTLEVEL)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *level,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                 (HYPRE_Int)                 *level ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetnumiteration, HYPRE_SSTRUCTPCGGETNUMITERATION)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *num_iterations,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetfinalrelativ, HYPRE_SSTRUCTPCGGETFINALRELATIV)
                                                     (hypre_F90_Obj *solver,
                                                      double   *norm,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetresidual, HYPRE_SSTRUCTPCGGETRESIDUAL)
                                                     (hypre_F90_Obj *solver,
                                                      hypre_F90_Obj *residual,
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructPCGGetResidual( (HYPRE_SStructSolver) *solver,
                                               (void *)              *residual ) );
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
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructDiagScaleSetup( (HYPRE_SStructSolver) *solver,
                                               (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructVector) *y,
                                               (HYPRE_SStructVector) *x    ) );
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
                                                      HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructDiagScale( (HYPRE_SStructSolver) *solver,
                                          (HYPRE_SStructMatrix) *A,
                                          (HYPRE_SStructVector) *y,
                                          (HYPRE_SStructVector) *x    ) );
}
