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
 * HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_FACCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver )
{
   return( hypre_FACDestroy2( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACAMR_RAP( HYPRE_SStructMatrix  A,
                         int                (*rfactors)[3], 
                         HYPRE_SStructMatrix *fac_A )
{
   return( hypre_AMR_RAP(A, rfactors, fac_A) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetup2( HYPRE_SStructSolver  solver,
                        HYPRE_SStructMatrix  A,
                        HYPRE_SStructVector  b,
                        HYPRE_SStructVector  x )
{
   return( hypre_FacSetup2( (void *) solver,
                           (hypre_SStructMatrix *)  A,
                           (hypre_SStructVector *)  b,
                           (hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSolve3(HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x)
{
   return( hypre_FACSolve3((void *) solver,
                           (hypre_SStructMatrix *)  A,
                           (hypre_SStructVector *)  b,
                           (hypre_SStructVector *)  x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetTol( HYPRE_SStructSolver solver,
                        double             tol    )
{
   return( hypre_FACSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetPLevels( HYPRE_SStructSolver  solver,
                            int                  nparts,
                            int                 *plevels)
{
   return( hypre_FACSetPLevels( (void *) solver, nparts, plevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACZeroCFSten( HYPRE_SStructMatrix  A,
                            HYPRE_SStructGrid    grid,
                            int                  part,
                            int                  rfactors[3] )
{
    hypre_SStructPMatrix   *Af= hypre_SStructMatrixPMatrix(A, part);
    hypre_SStructPMatrix   *Ac= hypre_SStructMatrixPMatrix(A, part-1);

    return( hypre_FacZeroCFSten(Af, Ac, (hypre_SStructGrid *)grid,
                                part, rfactors) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACZeroFCSten( HYPRE_SStructMatrix  A,
                            HYPRE_SStructGrid    grid,
                            int                  part )
{
    hypre_SStructPMatrix   *Af= hypre_SStructMatrixPMatrix(A, part);

    return( hypre_FacZeroFCSten(Af, (hypre_SStructGrid *)grid,
                                part) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACZeroAMRMatrixData( HYPRE_SStructMatrix  A,
                                   int                  part_crse,
                                   int                  rfactors[3] )
{
    return( hypre_ZeroAMRMatrixData(A, part_crse, rfactors) );
}
                                                                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACZeroAMRVectorData( HYPRE_SStructVector  b,
                                   int                 *plevels,
                                   int                (*rfactors)[3] )
{
    return( hypre_ZeroAMRVectorData(b, plevels, rfactors) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetPRefinements( HYPRE_SStructSolver  solver,
                                 int                  nparts,
                                 int                (*rfactors)[3] )
{
   return( hypre_FACSetPRefinements( (void *)         solver,
                                                      nparts,
                                                      rfactors ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver,
                              int                 max_levels  )
{
   return( hypre_FACSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetMaxIter( HYPRE_SStructSolver solver,
                            int                max_iter  )
{
   return( hypre_FACSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetRelChange( HYPRE_SStructSolver solver,
                              int                rel_change  )
{
   return( hypre_FACSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetRelaxType( HYPRE_SStructSolver solver,
                              int                relax_type )
{
   return( hypre_FACSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructFACSetNumPreRelax( HYPRE_SStructSolver solver,
                                int                num_pre_relax )
{
   return( hypre_FACSetNumPreSmooth( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetNumPostRelax( HYPRE_SStructSolver solver,
                                 int                num_post_relax )
{
   return( hypre_FACSetNumPostSmooth( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetCoarseSolverType( HYPRE_SStructSolver solver,
                                     int                 csolver_type)
{
   return( hypre_FACSetCoarseSolverType( (void *) solver, csolver_type) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACSetLogging( HYPRE_SStructSolver solver,
                            int                logging )
{
   return( hypre_FACSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACGetNumIterations( HYPRE_SStructSolver  solver,
                                  int                *num_iterations )
{
   return( hypre_FACGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructFACGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                              double             *norm   )
{
   return( hypre_FACGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


