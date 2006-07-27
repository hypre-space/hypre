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
 * HYPRE_SStructSplit solver interface
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"


/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitcreate, HYPRE_SSTRUCTSPLITCREATE)
                                                       (long int *comm,
                                                        long int *solver_ptr,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitCreate( (MPI_Comm)             *comm,
                                            (HYPRE_SStructSolver *) solver_ptr ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitdestroy, HYPRE_SSTRUCTSPLITDESTROY)
                                                       (long int *solver,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitDestroy( (HYPRE_SStructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetup, HYPRE_SSTRUCTSPLITSETUP)
                                                       (long int *solver,
                                                        long int *A,
                                                        long int *b,
                                                        long int *x,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsolve, HYPRE_SSTRUCTSPLITSOLVE)
                                                       (long int *solver,
                                                        long int *A,
                                                        long int *b,
                                                        long int *x,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsettol, HYPRE_SSTRUCTSPLITSETTOL)
                                                       (long int *solver,
                                                        double   *tol,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetmaxiter, HYPRE_SSTRUCTSPLITSETMAXITER)
                                                       (long int *solver,
                                                        int      *max_iter,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetzeroguess, HYPRE_SSTRUCTSPLITSETZEROGUESS)
                                                       (long int *solver,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetZeroGuess( (HYPRE_SStructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetnonzerogue, HYPRE_SSTRUCTSPLITSETNONZEROGUE)
                                                       (long int *solver,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetStructSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetstructsolv, HYPRE_SSTRUCTSPLITSETSTRUCTSOLV)
                                                       (long int *solver,
                                                        int      *ssolver,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitSetStructSolver( (HYPRE_SStructSolver) *solver,
                                                     (int)                 *ssolver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetnumiterati, HYPRE_SSTRUCTSPLITGETNUMITERATI)
                                                       (long int *solver,
                                                        int      *num_iterations,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetfinalrelat, HYPRE_SSTRUCTSPLITGETFINALRELAT)
                                                       (long int *solver,
                                                        double   *norm,
                                                        int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSplitGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}
