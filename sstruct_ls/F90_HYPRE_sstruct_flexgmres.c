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
                                                       (long int  *comm,
                                                        long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESCreate( (MPI_Comm)              *comm,
                                            (HYPRE_SStructSolver *)  solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresdestroy, HYPRE_SSTRUCTFLEXGMRESDESTROY)
                                                       (long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetup, HYPRE_SSTRUCTFLEXGMRESSETUP)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressolve, HYPRE_SSTRUCTFLEXGMRESSOLVE)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetkdim, HYPRE_SSTRUCTFLEXGMRESSETKDIM)
                                                       (long int  *solver,
                                                        int       *k_dim,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (int)                 *k_dim ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressettol, HYPRE_SSTRUCTFLEXGMRESSETTOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetabsolutetol, HYPRE_SSTRUCTFLEXGMRESSETABSOLUTETOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetminiter, HYPRE_SSTRUCTFLEXGMRESSETMINITER)
                                                       (long int  *solver,
                                                        int       *min_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetmaxiter, HYPRE_SSTRUCTFLEXGMRESSETMAXITER)
                                                       (long int  *solver,
                                                        int       *max_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetprecond, HYPRE_SSTRUCTFLEXGMRESSETPRECOND)
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
               (HYPRE_SStructFlexGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructFlexGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
               (HYPRE_SStructFlexGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
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
 * HYPRE_SStructFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetlogging, HYPRE_SSTRUCTFLEXGMRESSETLOGGING)
                                                       (long int  *solver,
                                                        int       *logging,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmressetprintlevel, HYPRE_SSTRUCTFLEXGMRESSETPRINTLEVEL)
                                                       (long int  *solver,
                                                        int       *level,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetnumiterati, HYPRE_SSTRUCTFLEXGMRESGETNUMITERATI)
                                                       (long int  *solver,
                                                        int       *num_iterations,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetfinalrelat, HYPRE_SSTRUCTFLEXGMRESGETFINALRELAT)
                                                       (long int  *solver,
                                                        double    *norm,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructflexgmresgetresidual, HYPRE_SSTRUCTFLEXGMRESGETRESIDUAL)
                                                       (long int  *solver,
                                                        long int  *residual,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructFlexGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
