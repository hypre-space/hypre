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
 * HYPRE_StructSparseMSG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_SparseMSGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSparseMSGDestroy( HYPRE_StructSolver solver )
{
   return( hypre_SparseMSGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSparseMSGSetup( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_SparseMSGSetup( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructSparseMSGSolve( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix A,
                       HYPRE_StructVector b,
                       HYPRE_StructVector x      )
{
   return( hypre_SparseMSGSolve( (void *) solver,
                            (hypre_StructMatrix *) A,
                            (hypre_StructVector *) b,
                            (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetTol( HYPRE_StructSolver solver,
                        double             tol    )
{
   return( hypre_SparseMSGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetMaxIter( HYPRE_StructSolver solver,
                                 int                max_iter  )
{
   return( hypre_SparseMSGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetJump( HYPRE_StructSolver solver,
                              int                    jump )
{
   return( hypre_SparseMSGSetJump( (void *) solver, jump ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetRelChange( HYPRE_StructSolver solver,
                                   int                rel_change  )
{
   return( hypre_SparseMSGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructSparseMSGSetZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SparseMSGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructSparseMSGSetNonZeroGuess( HYPRE_StructSolver solver )
{
   return( hypre_SparseMSGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetRelaxType( HYPRE_StructSolver solver,
                              int                relax_type )
{
   return( hypre_SparseMSGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
int
HYPRE_StructSparseMSGSetJacobiWeight(HYPRE_StructSolver solver,
                                     double             weight)
{
   return( hypre_SparseMSGSetJacobiWeight( (void *) solver, weight) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetNumPreRelax( HYPRE_StructSolver solver,
                                     int                num_pre_relax )
{
   return( hypre_SparseMSGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetNumPostRelax( HYPRE_StructSolver solver,
                                 int                num_post_relax )
{
   return( hypre_SparseMSGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetNumFineRelax( HYPRE_StructSolver solver,
                                 int                num_fine_relax )
{
   return( hypre_SparseMSGSetNumFineRelax( (void *) solver, num_fine_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetLogging( HYPRE_StructSolver solver,
                                 int                logging )
{
   return( hypre_SparseMSGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGSetPrintLevel( HYPRE_StructSolver solver,
                                    int              print_level )
{
   return( hypre_SparseMSGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGGetNumIterations( HYPRE_StructSolver  solver,
                                  int                *num_iterations )
{
   return( hypre_SparseMSGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructSparseMSGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                   double             *norm   )
{
   return( hypre_SparseMSGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

