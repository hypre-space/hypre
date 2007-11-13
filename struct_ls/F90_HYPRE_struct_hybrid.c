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





#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridcreate, HYPRE_STRUCTHYBRIDCREATE)( int      *comm,
                                               long int *solver,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridCreate( (MPI_Comm)             *comm,
                                             (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybriddestroy, HYPRE_STRUCTHYBRIDDESTROY)( long int *solver,
                                             int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetup, HYPRE_STRUCTHYBRIDSETUP)( long int *solver,
                                          long int *A,
                                          long int *b,
                                          long int *x,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSetup( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsolve, HYPRE_STRUCTHYBRIDSOLVE)( long int *solver,
                                          long int *A,
                                          long int *b,
                                          long int *x,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSolve( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettol, HYPRE_STRUCTHYBRIDSETTOL)( long int *solver,
                                           double   *tol,
                                           int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSetTol( (HYPRE_StructSolver) *solver,
                                             (double)             *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetconvergenc, HYPRE_STRUCTHYBRIDSETCONVERGENC)( long int *solver,
                                                  double   *cf_tol,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetConvergenceTol( (HYPRE_StructSolver) *solver,
                                             (double)             *cf_tol  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetdscgmaxite, HYPRE_STRUCTHYBRIDSETDSCGMAXITE)( long int *solver,
                                                  int      *dscg_max_its,
                                                  int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetDSCGMaxIter(
         (HYPRE_StructSolver) *solver,
         (int)                *dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgmaxiter, HYPRE_STRUCTHYBRIDSETPCGMAXITER)( long int *solver,
                                                  int      *pcg_max_its,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetPCGMaxIter( (HYPRE_StructSolver) *solver,
                                         (int)                *pcg_max_its ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgabsolut, HYPRE_STRUCTHYBRIDSETPCGABSOLUT)
                                                ( long int *solver,
                                                  double   *pcg_atolf,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetPCGAbsoluteTolFactor( (HYPRE_StructSolver) *solver,
                                                   (double)             *pcg_atolf ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettwonorm, HYPRE_STRUCTHYBRIDSETTWONORM)( long int *solver,
                                               int      *two_norm,
                                               int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetTwoNorm( (HYPRE_StructSolver) *solver,
                                      (int)                *two_norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetstopcrit, HYPRE_STRUCTHYBRIDSETSTOPCRIT)
                                             ( long int *solver,
                                               int      *stop_crit,
                                               int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetStopCrit( (HYPRE_StructSolver) *solver,
                                       (int)                *stop_crit   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetrelchange, HYPRE_STRUCTHYBRIDSETRELCHANGE)
                                               ( long int *solver,
                                                 int      *rel_change,
                                                 int      *ierr       )
{
   *ierr = (int) 
           ( HYPRE_StructHybridSetRelChange( (HYPRE_StructSolver) *solver,
                                             (int)                *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetsolvertype, HYPRE_STRUCTHYBRIDSETSOLVERTYPE)
                                             ( long int *solver,
                                               int      *solver_type,
                                               int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetSolverType( (HYPRE_StructSolver) *solver,
                                         (int)                *solver_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetkdim, HYPRE_STRUCTHYBRIDSETKDIM)
                                               ( long int *solver,
                                                 int      *k_dim,
                                                 int      *ierr    )
{
   *ierr = (int) (HYPRE_StructHybridSetKDim( (HYPRE_StructSolver) *solver,
                                             (int)                *k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprecond, HYPRE_STRUCTHYBRIDSETPRECOND)( long int *solver,
                                               int      *precond_id,
                                               long int *precond_solver,
                                               int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (int)
         ( HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (int)
         ( HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetlogging, HYPRE_STRUCTHYBRIDSETLOGGING)( long int *solver,
                                               int      *logging,
                                               int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetLogging(
         (HYPRE_StructSolver) *solver,
         (int)                *logging    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprintlevel, HYPRE_STRUCTHYBRIDSETPRINTLEVEL)( long int *solver,
                                               int      *print_level,
                                               int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetPrintLevel(
         (HYPRE_StructSolver) *solver,
         (int)                *print_level  ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetnumiterati, HYPRE_STRUCTHYBRIDGETNUMITERATI)( long int *solver,
                                                  int      *num_its,
                                                  int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_its    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetdscgnumite, HYPRE_STRUCTHYBRIDGETDSCGNUMITE)( long int *solver,
                                                  int      *dscg_num_its,
                                                  int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetDSCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetpcgnumiter, HYPRE_STRUCTHYBRIDGETPCGNUMITER)( long int *solver,
                                                  int      *pcg_num_its,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetPCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetfinalrelat, HYPRE_STRUCTHYBRIDGETFINALRELAT)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm    ) );
}
