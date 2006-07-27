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
 * HYPRE_StructSMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_SMGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGDestroy( HYPRE_StructSolver solver )
{
   return( hypre_SMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_SMGSetup( (void *) solver,
                           (hypre_StructMatrix *) A,
                           (hypre_StructVector *) b,
                           (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSMGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_SMGSolve( (void *) solver,
                           (hypre_StructMatrix *) A,
                           (hypre_StructVector *) b,
                           (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse, HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetMemoryUse( HYPRE_StructSolver solver,
                             int                memory_use )
{
   return( hypre_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

int
HYPRE_StructSMGGetMemoryUse( HYPRE_StructSolver solver,
                             int              * memory_use )
{
   return( hypre_SMGGetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol, HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( hypre_SMGSetTol( (void *) solver, tol ) );
}

int
HYPRE_StructSMGGetTol( HYPRE_StructSolver solver,
                       double           * tol    )
{
   return( hypre_SMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter, HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter  )
{
   return( hypre_SMGSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_StructSMGGetMaxIter( HYPRE_StructSolver solver,
                           int              * max_iter  )
{
   return( hypre_SMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange, HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetRelChange( HYPRE_StructSolver solver,
                             int                rel_change  )
{
   return( hypre_SMGSetRelChange( (void *) solver, rel_change ) );
}

int
HYPRE_StructSMGGetRelChange( HYPRE_StructSolver solver,
                             int              * rel_change  )
{
   return( hypre_SMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess, HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructSMGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SMGSetZeroGuess( (void *) solver, 1 ) );
}

int
HYPRE_StructSMGGetZeroGuess( HYPRE_StructSolver solver,
                             int * zeroguess )
{
   return( hypre_SMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess,  also see HYPRE_StructSMGGetZeroGuess above
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructSMGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax, HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetNumPreRelax( HYPRE_StructSolver solver,
                               int                num_pre_relax )
{
   return( hypre_SMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

int
HYPRE_StructSMGGetNumPreRelax( HYPRE_StructSolver solver,
                               int              * num_pre_relax )
{
   return( hypre_SMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax, HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetNumPostRelax( HYPRE_StructSolver solver,
                                int                num_post_relax )
{
   return( hypre_SMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

int
HYPRE_StructSMGGetNumPostRelax( HYPRE_StructSolver solver,
                                int              * num_post_relax )
{
   return( hypre_SMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging, HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( hypre_SMGSetLogging( (void *) solver, logging) );
}

int
HYPRE_StructSMGGetLogging( HYPRE_StructSolver solver,
                           int              * logging )
{
   return( hypre_SMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetPrintLevel, HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGSetPrintLevel( HYPRE_StructSolver solver,
                           int        print_level )
{
   return( hypre_SMGSetPrintLevel( (void *) solver, print_level) );
}

int
HYPRE_StructSMGGetPrintLevel( HYPRE_StructSolver solver,
                           int      * print_level )
{
   return( hypre_SMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
   return( hypre_SMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSMGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( hypre_SMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

