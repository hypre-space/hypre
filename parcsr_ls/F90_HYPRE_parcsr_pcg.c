/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgcreate, HYPRE_PARCSRPCGCREATE)( int      *comm,
                                            long int *solver,
                                            int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPCGCreate( (MPI_Comm)       *comm,
                                              (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgdestroy, HYPRE_PARCSRPCGDESTROY)( long int *solver,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsetup, HYPRE_PARCSRPCGSETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetup( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsolve, HYPRE_PARCSRPCGSOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSolve( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettol, HYPRE_PARCSRPCGSETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetTol( (HYPRE_Solver) *solver,
                                          (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetmaxiter, HYPRE_PARCSRPCGSETMAXITER)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetMaxIter( (HYPRE_Solver) *solver,
                                              (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetstopcrit, HYPRE_PARCSRPCGSETSTOPCRIT)
                                          ( long int *solver,
                                            int      *stop_crit,
                                            int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetStopCrit( (HYPRE_Solver) *solver,
                                              (int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettwonorm, HYPRE_PARCSRPCGSETTWONORM)( long int *solver,
                                            int      *two_norm,
                                            int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetTwoNorm( (HYPRE_Solver) *solver,
                                              (int)          *two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetrelchange, HYPRE_PARCSRPCGSETRELCHANGE)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr        )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetRelChange( (HYPRE_Solver) *solver,
                                                (int)          *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprecond, HYPRE_PARCSRPCGSETPRECOND)( long int *solver,
                                            int      *precond_id,
                                            long int *precond_solver,
                                            int      *ierr            )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - do not set up a preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    * 4 - set up a ParaSails preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRDiagScale,
                               HYPRE_ParCSRDiagScaleSetup,
                               NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_BoomerAMGSolve,
                               HYPRE_BoomerAMGSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRPilutSolve,
                               HYPRE_ParCSRPilutSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParaSailsSolve,
                               HYPRE_ParaSailsSetup,
                               (void *)       *precond_solver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetprecond, HYPRE_PARCSRPCGGETPRECOND)( long int *solver,
                                            long int *precond_solver_ptr,
                                            int      *ierr                )
{
    *ierr = (int)
            ( HYPRE_ParCSRPCGGetPrecond( (HYPRE_Solver)   *solver,
                                         (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprintlevel, HYPRE_PARCSRPCGSETPRINTLEVEL)( long int *solver,
                                            int      *level,
                                            int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetPrintLevel( (HYPRE_Solver) *solver,
                                                 (int)       *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetnumiterations, HYPRE_PARCSRPCGGETNUMITERATIONS)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRPCGGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetfinalrelative, HYPRE_PARCSRPCGGETFINALRELATIVE)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscalesetup, HYPRE_PARCSRDIAGSCALESETUP)( long int *solver,
                                             long int *A,
                                             long int *y,
                                             long int *x,
                                             int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRDiagScaleSetup( (HYPRE_Solver)       *solver,
                                               (HYPRE_ParCSRMatrix) *A,
                                               (HYPRE_ParVector)    *y,
                                               (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscale, HYPRE_PARCSRDIAGSCALE)( long int *solver,
                                        long int *HA,
                                        long int *Hy,
                                        long int *Hx,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRDiagScale( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *HA,
                                          (HYPRE_ParVector)    *Hy,
                                          (HYPRE_ParVector)    *Hx      ) );
}
