/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
                                                       (long int  *comm,
                                                        long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESCreate( (MPI_Comm)              *comm,
                                            (HYPRE_SStructSolver *)  solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresdestroy, HYPRE_SSTRUCTGMRESDESTROY)
                                                       (long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetup, HYPRE_SSTRUCTGMRESSETUP)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressolve, HYPRE_SSTRUCTGMRESSOLVE)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetkdim, HYPRE_SSTRUCTGMRESSETKDIM)
                                                       (long int  *solver,
                                                        int       *k_dim,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (int)                 *k_dim ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressettol, HYPRE_SSTRUCTGMRESSETTOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetminiter, HYPRE_SSTRUCTGMRESSETMINITER)
                                                       (long int  *solver,
                                                        int       *min_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetmaxiter, HYPRE_SSTRUCTGMRESSETMAXITER)
                                                       (long int  *solver,
                                                        int       *max_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetstopcrit, HYPRE_SSTRUCTGMRESSETSTOPCRIT)
                                                       (long int  *solver,
                                                        int       *stop_crit,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetStopCrit( (HYPRE_SStructSolver) *solver,
                                                  (int)                *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprecond, HYPRE_SSTRUCTGMRESSETPRECOND)
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
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
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
                                                       (long int  *solver,
                                                        int       *logging,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprintlevel, HYPRE_SSTRUCTGMRESSETPRINTLEVEL)
                                                       (long int  *solver,
                                                        int       *level,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetnumiterati, HYPRE_SSTRUCTGMRESGETNUMITERATI)
                                                       (long int  *solver,
                                                        int       *num_iterations,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetfinalrelat, HYPRE_SSTRUCTGMRESGETFINALRELAT)
                                                       (long int  *solver,
                                                        double    *norm,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetresidual, HYPRE_SSTRUCTGMRESGETRESIDUAL)
                                                       (long int  *solver,
                                                        long int  *residual,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
