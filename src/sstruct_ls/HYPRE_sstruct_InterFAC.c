/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
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

HYPRE_Int
HYPRE_SStructFACCreate( MPI_Comm comm, HYPRE_SStructSolver *solver )
{
   *solver = ( (HYPRE_SStructSolver) hypre_FACCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver )
{
   return( hypre_FACDestroy2( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACAMR_RAP( HYPRE_SStructMatrix  A,
                         HYPRE_Int          (*rfactors)[3], 
                         HYPRE_SStructMatrix *fac_A )
{
   return( hypre_AMR_RAP(A, rfactors, fac_A) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/
HYPRE_Int
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
HYPRE_Int
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

HYPRE_Int
HYPRE_SStructFACSetTol( HYPRE_SStructSolver solver,
                        double             tol    )
{
   return( hypre_FACSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACSetPLevels( HYPRE_SStructSolver  solver,
                            HYPRE_Int            nparts,
                            HYPRE_Int           *plevels)
{
   return( hypre_FACSetPLevels( (void *) solver, nparts, plevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACZeroCFSten( HYPRE_SStructMatrix  A,
                            HYPRE_SStructGrid    grid,
                            HYPRE_Int            part,
                            HYPRE_Int            rfactors[3] )
{
    hypre_SStructPMatrix   *Af= hypre_SStructMatrixPMatrix(A, part);
    hypre_SStructPMatrix   *Ac= hypre_SStructMatrixPMatrix(A, part-1);

    return( hypre_FacZeroCFSten(Af, Ac, (hypre_SStructGrid *)grid,
                                part, rfactors) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACZeroFCSten( HYPRE_SStructMatrix  A,
                            HYPRE_SStructGrid    grid,
                            HYPRE_Int            part )
{
    hypre_SStructPMatrix   *Af= hypre_SStructMatrixPMatrix(A, part);

    return( hypre_FacZeroFCSten(Af, (hypre_SStructGrid *)grid,
                                part) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACZeroAMRMatrixData( HYPRE_SStructMatrix  A,
                                   HYPRE_Int            part_crse,
                                   HYPRE_Int            rfactors[3] )
{
    return( hypre_ZeroAMRMatrixData(A, part_crse, rfactors) );
}
                                                                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACZeroAMRVectorData( HYPRE_SStructVector  b,
                                   HYPRE_Int           *plevels,
                                   HYPRE_Int          (*rfactors)[3] )
{
    return( hypre_ZeroAMRVectorData(b, plevels, rfactors) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACSetPRefinements( HYPRE_SStructSolver  solver,
                                 HYPRE_Int            nparts,
                                 HYPRE_Int          (*rfactors)[3] )
{
   return( hypre_FACSetPRefinements( (void *)         solver,
                                                      nparts,
                                                      rfactors ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_levels  )
{
   return( hypre_FACSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetMaxIter( HYPRE_SStructSolver solver,
                            HYPRE_Int          max_iter  )
{
   return( hypre_FACSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetRelChange( HYPRE_SStructSolver solver,
                              HYPRE_Int          rel_change  )
{
   return( hypre_FACSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   return( hypre_FACSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetRelaxType( HYPRE_SStructSolver solver,
                              HYPRE_Int          relax_type )
{
   return( hypre_FACSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetJacobiWeight( HYPRE_SStructSolver solver,
                                 double              weight)
{
   return( hypre_FACSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructFACSetNumPreRelax( HYPRE_SStructSolver solver,
                                HYPRE_Int          num_pre_relax )
{
   return( hypre_FACSetNumPreSmooth( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetNumPostRelax( HYPRE_SStructSolver solver,
                                 HYPRE_Int          num_post_relax )
{
   return( hypre_FACSetNumPostSmooth( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetCoarseSolverType( HYPRE_SStructSolver solver,
                                     HYPRE_Int           csolver_type)
{
   return( hypre_FACSetCoarseSolverType( (void *) solver, csolver_type) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACSetLogging( HYPRE_SStructSolver solver,
                            HYPRE_Int          logging )
{
   return( hypre_FACSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACGetNumIterations( HYPRE_SStructSolver  solver,
                                  HYPRE_Int          *num_iterations )
{
   return( hypre_FACGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFACGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                              double             *norm   )
{
   return( hypre_FACGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


