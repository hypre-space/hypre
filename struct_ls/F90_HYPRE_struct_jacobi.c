/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
 * HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobicreate, HYPRE_STRUCTJACOBICREATE)( int      *comm,
                                            long int *solver,
                                            int      *ierr   )

{
   *ierr = (int)
      ( HYPRE_StructJacobiCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structjacobidestroy, HYPRE_STRUCTJACOBIDESTROY)( long int *solver,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructJacobiDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structjacobisetup, HYPRE_STRUCTJACOBISETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructJacobiSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structjacobisolve, HYPRE_STRUCTJACOBISOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructJacobiSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisettol, HYPRE_STRUCTJACOBISETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructJacobiSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigettol, HYPRE_STRUCTJACOBIGETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructJacobiGetTol( (HYPRE_StructSolver) *solver,
                                          (double *)             tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetmaxiter, HYPRE_STRUCTJACOBISETMAXITER)
                                          ( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructJacobiSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetmaxiter, HYPRE_STRUCTJACOBIGETMAXITER)
                                          ( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructJacobiGetMaxIter( (HYPRE_StructSolver) *solver,
                                      (int *)               max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetzeroguess, HYPRE_STRUCTJACOBISETZEROGUESS)( long int *solver,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructJacobiSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetzeroguess, HYPRE_STRUCTJACOBIGETZEROGUESS)
                                            ( long int *solver,
                                              int      *zeroguess,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructJacobiGetZeroGuess( (HYPRE_StructSolver) *solver,
                                        (int *)               zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetnonzerogue, HYPRE_STRUCTJACOBISETNONZEROGUE)( long int *solver,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructJacobiSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetnumiterati, HYPRE_STRUCTJACOBIGETNUMITERATI)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructJacobiGetNumIterations( (HYPRE_StructSolver) *solver,
         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetfinalrelat, HYPRE_STRUCTJACOBIGETFINALRELAT)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructJacobiGetFinalRelativeResidualNorm( (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
