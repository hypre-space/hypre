/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructMaxwell interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMaxwellCreate(MPI_Comm comm, HYPRE_SStructSolver *solver)
{
   *solver = ( (HYPRE_SStructSolver) hypre_MaxwellTVCreate(comm) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMaxwellDestroy(HYPRE_SStructSolver solver)
{
   return ( hypre_MaxwellTVDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMaxwellSetup( HYPRE_SStructSolver  solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x )
{
   return ( hypre_MaxwellTV_Setup( (void *) solver,
                                   (hypre_SStructMatrix *) A,
                                   (hypre_SStructVector *) b,
                                   (hypre_SStructVector *) x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMaxwellSolve( HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return ( hypre_MaxwellSolve( (void *) solver,
                                (hypre_SStructMatrix *) A,
                                (hypre_SStructVector *) b,
                                (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMaxwellSolve2( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x      )
{
   return ( hypre_MaxwellSolve2( (void *) solver,
                                 (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) b,
                                 (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_MaxwellGrad( HYPRE_SStructGrid   grid,
                   HYPRE_ParCSRMatrix *T )

{
   *T = ( (HYPRE_ParCSRMatrix) hypre_Maxwell_Grad( (hypre_SStructGrid *) grid));
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetGrad( HYPRE_SStructSolver  solver,
                             HYPRE_ParCSRMatrix   T )
{
   return ( hypre_MaxwellSetGrad( (void *)               solver,
                                  (hypre_ParCSRMatrix *) T) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetRfactors( HYPRE_SStructSolver  solver,
                                 HYPRE_Int            rfactors[3] )
{
   return ( hypre_MaxwellSetRfactors( (void *)         solver,
                                      rfactors ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetTol( HYPRE_SStructSolver solver,
                            HYPRE_Real          tol    )
{
   return ( hypre_MaxwellSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetConstantCoef( HYPRE_SStructSolver solver,
                                     HYPRE_Int           constant_coef)
{
   return ( hypre_MaxwellSetConstantCoef( (void *) solver, constant_coef) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetMaxIter( HYPRE_SStructSolver solver,
                                HYPRE_Int           max_iter  )
{
   return ( hypre_MaxwellSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetRelChange( HYPRE_SStructSolver solver,
                                  HYPRE_Int           rel_change  )
{
   return ( hypre_MaxwellSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetNumPreRelax( HYPRE_SStructSolver solver,
                                    HYPRE_Int           num_pre_relax )
{
   return ( hypre_MaxwellSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetNumPostRelax( HYPRE_SStructSolver solver,
                                     HYPRE_Int           num_post_relax )
{
   return ( hypre_MaxwellSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetLogging( HYPRE_SStructSolver solver,
                                HYPRE_Int           logging )
{
   return ( hypre_MaxwellSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
HYPRE_SStructMaxwellSetPrintLevel
*--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellSetPrintLevel( HYPRE_SStructSolver solver,
                                   HYPRE_Int           print_level )
{
   return ( hypre_MaxwellSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellPrintLogging( HYPRE_SStructSolver solver,
                                  HYPRE_Int           myid)
{
   return ( hypre_MaxwellPrintLogging( (void *) solver, myid) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellGetNumIterations( HYPRE_SStructSolver  solver,
                                      HYPRE_Int           *num_iterations )
{
   return ( hypre_MaxwellGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  HYPRE_Real          *norm   )
{
   return ( hypre_MaxwellGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellPhysBdy( HYPRE_SStructGrid  *grid_l,
                             HYPRE_Int           num_levels,
                             HYPRE_Int           rfactors[3],
                             HYPRE_Int        ***BdryRanks_ptr,
                             HYPRE_Int         **BdryRanksCnt_ptr )
{
   return ( hypre_Maxwell_PhysBdy( (hypre_SStructGrid  **) grid_l,
                                   num_levels,
                                   rfactors,
                                   BdryRanks_ptr,
                                   BdryRanksCnt_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_SStructMaxwellEliminateRowsCols( HYPRE_ParCSRMatrix  parA,
                                       HYPRE_Int           nrows,
                                       HYPRE_Int          *rows )
{
   return ( hypre_ParCSRMatrixEliminateRowsCols( (hypre_ParCSRMatrix *) parA,
                                                 nrows,
                                                 rows ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/
HYPRE_Int HYPRE_SStructMaxwellZeroVector(HYPRE_ParVector  v,
                                         HYPRE_Int       *rows,
                                         HYPRE_Int        nrows)
{
   return ( hypre_ParVectorZeroBCValues( (hypre_ParVector *) v,
                                         rows,
                                         nrows ) );
}


