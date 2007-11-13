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
 * HYPRE_SStructSysPFMG interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_SysPFMGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSysPFMGDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSysPFMGSetup( HYPRE_SStructSolver  solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysPFMGSetup( (void *) solver,
                               (hypre_SStructMatrix *) A,
                               (hypre_SStructVector *) b,
                               (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSysPFMGSolve( HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_SysPFMGSolve( (void *) solver,
                            (hypre_SStructMatrix *) A,
                            (hypre_SStructVector *) b,
                            (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetTol( HYPRE_SStructSolver solver,
                            double             tol    )
{
   return( hypre_SysPFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetMaxIter( HYPRE_SStructSolver solver,
                                int                max_iter  )
{
   return( hypre_SysPFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetRelChange( HYPRE_SStructSolver solver,
                                  int                rel_change  )
{
   return( hypre_SysPFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SStructSysPFMGSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SStructSysPFMGSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_SysPFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetRelaxType( HYPRE_SStructSolver solver,
                                  int                relax_type )
{
   return( hypre_SysPFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
int
HYPRE_SStructSysPFMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                    double              weight)
{
   return( hypre_SysPFMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetNumPreRelax( HYPRE_SStructSolver solver,
                                    int                num_pre_relax )
{
   return( hypre_SysPFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetNumPostRelax( HYPRE_SStructSolver solver,
                                     int                num_post_relax )
{
   return( hypre_SysPFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetSkipRelax( HYPRE_SStructSolver solver,
                                  int                skip_relax )
{
   return( hypre_SysPFMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetDxyz( HYPRE_SStructSolver  solver,
                         double             *dxyz   )
{
   return( hypre_SysPFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetLogging( HYPRE_SStructSolver solver,
                                int                logging )
{
   return( hypre_SysPFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
HYPRE_SStructSysPFMGSetPrintLevel
*--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGSetPrintLevel( HYPRE_SStructSolver solver,
                                int               print_level )
{
   return( hypre_SysPFMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGGetNumIterations( HYPRE_SStructSolver  solver,
                                      int                *num_iterations )
{
   return( hypre_SysPFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  double             *norm   )
{
   return( hypre_SysPFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

