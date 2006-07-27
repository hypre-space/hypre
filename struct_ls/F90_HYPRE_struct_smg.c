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
 * HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgcreate, HYPRE_STRUCTSMGCREATE)( int      *comm,
                                        long int *solver,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGCreate( (MPI_Comm)             *comm,
                                          (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgdestroy, HYPRE_STRUCTSMGDESTROY)( long int *solver,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsetup, HYPRE_STRUCTSMGSETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsolve, HYPRE_STRUCTSMGSOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse, HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse, HYPRE_STRUCTSMGSETMEMORYUSE)( long int *solver,
                                              int      *memory_use,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetMemoryUse( (HYPRE_StructSolver) *solver,
                                     (int)                *memory_use ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmemoryuse, HYPRE_STRUCTSMGGETMEMORYUSE)
                                            ( long int *solver,
                                              int      *memory_use,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetMemoryUse( (HYPRE_StructSolver) *solver,
                                     (int *)               memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol, HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol, HYPRE_STRUCTSMGSETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

void
hypre_F90_IFACE(hypre_structsmggettol, HYPRE_STRUCTSMGGETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGGetTol( (HYPRE_StructSolver) *solver,
                                          (double *)            tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter, HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter, HYPRE_STRUCTSMGSETMAXITER)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int)                *max_iter ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmaxiter, HYPRE_STRUCTSMGGETMAXITER)
                                          ( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int *)               max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange, HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange, HYPRE_STRUCTSMGSETRELCHANGE)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (int)                *rel_change ) );
}

void
hypre_F90_IFACE(hypre_structsmggetrelchange, HYPRE_STRUCTSMGGETRELCHANGE)
                                            ( long int *solver,
                                              int      *rel_change,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetRelChange( (HYPRE_StructSolver) *solver,
                                     (int *)               rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess, HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetzeroguess, HYPRE_STRUCTSMGSETZEROGUESS)( long int *solver,
                                              int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}
 
void
hypre_F90_IFACE(hypre_structsmggetzeroguess, HYPRE_STRUCTSMGGETZEROGUESS)
                                            ( long int *solver,
                                              int      *zeroguess,
                                              int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetZeroGuess( (HYPRE_StructSolver) *solver,
                                     (int *)               zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess, HYPRE_STRUCTSMGSETNONZEROGUESS)( long int *solver,
                                                 int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax, HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax, HYPRE_STRUCTSMGSETNUMPRERELAX)( long int *solver,
                                                int      *num_pre_relax,
                                                int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                       (int)                *num_pre_relax) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumprerelax, HYPRE_STRUCTSMGGETNUMPRERELAX)
                                              ( long int *solver,
                                                int      *num_pre_relax,
                                                int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetNumPreRelax( (HYPRE_StructSolver) *solver,
                                       (int *)               num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax, HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax, HYPRE_STRUCTSMGSETNUMPOSTRELAX)
                                               ( long int *solver,
                                                 int      *num_post_relax,
                                                 int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                        (int)                *num_post_relax) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumpostrelax, HYPRE_STRUCTSMGGETNUMPOSTRELAX)
                                               ( long int *solver,
                                                 int      *num_post_relax,
                                                 int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetNumPostRelax( (HYPRE_StructSolver) *solver,
                                        (int *)               num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging, HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging, HYPRE_STRUCTSMGSETLOGGING)
                                          ( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetLogging( (HYPRE_StructSolver) *solver,
                                   (int)                *logging) );
}

void
hypre_F90_IFACE(hypre_structsmggetlogging, HYPRE_STRUCTSMGGETLOGGING)
                                          ( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetLogging( (HYPRE_StructSolver) *solver,
                                   (int *)               logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetPrintLevel, HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetprintlevel, HYPRE_STRUCTSMGSETPRINTLEVEL)
                                          ( long int *solver,
                                            int      *print_level,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                      (int)                *print_level) );
}

void
hypre_F90_IFACE(hypre_structsmggetprintlevel, HYPRE_STRUCTSMGGETPRINTLEVEL)
                                          ( long int *solver,
                                            int      *print_level,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetPrintLevel( (HYPRE_StructSolver) *solver,
                                      (int *)               print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations, HYPRE_STRUCTSMGGETNUMITERATIONS)
                                                ( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetNumIterations( (HYPRE_StructSolver) *solver,
                                         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative, HYPRE_STRUCTSMGGETFINALRELATIVE)
                                                ( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
