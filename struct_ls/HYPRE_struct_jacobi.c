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



/******************************************************************************
 *
 * HYPRE_StructJacobi interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiCreate( MPI_Comm            comm,
                          HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_JacobiCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiDestroy( HYPRE_StructSolver solver )
{
   return( hypre_JacobiDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_JacobiSetup( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructJacobiSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_JacobiSolve( (void *) solver,
                              (hypre_StructMatrix *) A,
                              (hypre_StructVector *) b,
                              (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver,
                          double             tol    )
{
   return( hypre_JacobiSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetTol( HYPRE_StructSolver solver,
                          double           * tol    )
{
   return( hypre_JacobiGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver,
                              int                max_iter  )
{
   return( hypre_JacobiSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetMaxIter( HYPRE_StructSolver solver,
                              int              * max_iter  )
{
   return( hypre_JacobiGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructJacobiGetZeroGuess( HYPRE_StructSolver solver,
                                int * zeroguess )
{
   return( hypre_JacobiGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_JacobiSetZeroGuess( (void *) solver, 0 ) );
}





/* NOT YET IMPLEMENTED */

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver  solver,
                                    int                *num_iterations )
{
   return( hypre_JacobiGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                double             *norm   )
{
#if 0
   return( hypre_JacobiGetFinalRelativeResidualNorm( (void *) solver, norm ) );
#endif
   return 0;
}
