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
                                                       (long int  *comm,
                                                        long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESCreate( (MPI_Comm)              *comm,
                                            (HYPRE_SStructSolver *)  solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresdestroy, HYPRE_SSTRUCTLGMRESDESTROY)
                                                       (long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetup, HYPRE_SSTRUCTLGMRESSETUP)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressolve, HYPRE_SSTRUCTLGMRESSOLVE)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetkdim, HYPRE_SSTRUCTLGMRESSETKDIM)
                                                       (long int  *solver,
                                                        int       *k_dim,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (int)                 *k_dim ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetaugdim, HYPRE_SSTRUCTLGMRESSETAUGDIM)
                                                       (long int  *solver,
                                                        int       *aug_dim,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetAugDim( (HYPRE_SStructSolver) *solver,
                                             (int)                 *aug_dim ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressettol, HYPRE_SSTRUCTLGMRESSETTOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetabsolutetol, HYPRE_SSTRUCTLGMRESSETABSOLUTETOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetminiter, HYPRE_SSTRUCTLGMRESSETMINITER)
                                                       (long int  *solver,
                                                        int       *min_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetmaxiter, HYPRE_SSTRUCTLGMRESSETMAXITER)
                                                       (long int  *solver,
                                                        int       *max_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprecond, HYPRE_SSTRUCTLGMRESSETPRECOND)
                                                     (long int *solver,
                                                      int      *precond_id,
                                                      long int *precond_solver,
                                                      int      *ierr)
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
       *ierr = (int)
               (HYPRE_SStructLGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructLGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
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
                                                       (long int  *solver,
                                                        int       *logging,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmressetprintlevel, HYPRE_SSTRUCTLGMRESSETPRINTLEVEL)
                                                       (long int  *solver,
                                                        int       *level,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetnumiterati, HYPRE_SSTRUCTLGMRESGETNUMITERATI)
                                                       (long int  *solver,
                                                        int       *num_iterations,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetfinalrelat, HYPRE_SSTRUCTLGMRESGETFINALRELAT)
                                                       (long int  *solver,
                                                        double    *norm,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructlgmresgetresidual, HYPRE_SSTRUCTLGMRESGETRESIDUAL)
                                                       (long int  *solver,
                                                        long int  *residual,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructLGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
