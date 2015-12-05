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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_SStructMaxwell interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMaxwellCreate(MPI_Comm comm, HYPRE_SStructSolver *solver)
{
   *solver = ( (HYPRE_SStructSolver) hypre_MaxwellTVCreate(comm) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMaxwellDestroy(HYPRE_SStructSolver solver)
{
   return( hypre_MaxwellTVDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMaxwellSetup( HYPRE_SStructSolver  solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x )
{
   return( hypre_MaxwellTV_Setup( (void *) solver,
                                  (hypre_SStructMatrix *) A,
                                  (hypre_SStructVector *) b,
                                  (hypre_SStructVector *) x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMaxwellSolve( HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x      )
{
   return( hypre_MaxwellSolve( (void *) solver,
                               (hypre_SStructMatrix *) A,
                               (hypre_SStructVector *) b,
                               (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/
                                                                                                              
int
HYPRE_SStructMaxwellSolve2( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x      )
{
   return( hypre_MaxwellSolve2( (void *) solver,
                                (hypre_SStructMatrix *) A,
                                (hypre_SStructVector *) b,
                                (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/
int
HYPRE_MaxwellGrad( HYPRE_SStructGrid   grid,
                   HYPRE_ParCSRMatrix *T )
                   
{
   *T= ( (HYPRE_ParCSRMatrix) hypre_Maxwell_Grad( (hypre_SStructGrid *) grid));
    return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetGrad( HYPRE_SStructSolver  solver,
                             HYPRE_ParCSRMatrix   T )
{
   return( hypre_MaxwellSetGrad( (void *)               solver,
                                 (hypre_ParCSRMatrix *) T) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetRfactors( HYPRE_SStructSolver  solver,
                                 int                  rfactors[3] )
{
   return( hypre_MaxwellSetRfactors( (void *)         solver,
                                                      rfactors ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetTol( HYPRE_SStructSolver solver,
                            double              tol    )
{
   return( hypre_MaxwellSetTol( (void *) solver, tol ) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetConstantCoef( HYPRE_SStructSolver solver,
                                     int                 constant_coef)
{
   return( hypre_MaxwellSetConstantCoef( (void *) solver, constant_coef) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetMaxIter( HYPRE_SStructSolver solver,
                                int                 max_iter  )
{
   return( hypre_MaxwellSetMaxIter( (void *) solver, max_iter ) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetRelChange( HYPRE_SStructSolver solver,
                                  int                 rel_change  )
{
   return( hypre_MaxwellSetRelChange( (void *) solver, rel_change ) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetNumPreRelax( HYPRE_SStructSolver solver,
                                    int                 num_pre_relax )
{
   return( hypre_MaxwellSetNumPreRelax( (void *) solver, num_pre_relax) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetNumPostRelax( HYPRE_SStructSolver solver,
                                     int                 num_post_relax )
{
   return( hypre_MaxwellSetNumPostRelax( (void *) solver, num_post_relax) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetLogging( HYPRE_SStructSolver solver,
                                int                 logging )
{
   return( hypre_MaxwellSetLogging( (void *) solver, logging) );
}
                                                                                                             
/*--------------------------------------------------------------------------
HYPRE_SStructMaxwellSetPrintLevel
*--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellSetPrintLevel( HYPRE_SStructSolver solver,
                                   int                 print_level )
{
   return( hypre_MaxwellSetPrintLevel( (void *) solver, print_level) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellPrintLogging( HYPRE_SStructSolver solver,
                                  int                 myid)
{
   return( hypre_MaxwellPrintLogging( (void *) solver, myid) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellGetNumIterations( HYPRE_SStructSolver  solver,
                                      int                 *num_iterations )
{
   return( hypre_MaxwellGetNumIterations( (void *) solver, num_iterations ) );
}
                                                                                                             
/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                  double              *norm   )
{
   return( hypre_MaxwellGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellPhysBdy( HYPRE_SStructGrid  *grid_l,
                             int                 num_levels,
                             int                 rfactors[3],
                             int              ***BdryRanks_ptr,
                             int               **BdryRanksCnt_ptr )
{
    return( hypre_Maxwell_PhysBdy( (hypre_SStructGrid  **) grid_l,
                                                           num_levels,
                                                           rfactors,
                                                           BdryRanks_ptr,
                                                           BdryRanksCnt_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMaxwellEliminateRowsCols( HYPRE_ParCSRMatrix  parA,
                                     int                 nrows,
                                     int                *rows )
{
    return( hypre_ParCSRMatrixEliminateRowsCols( (hypre_ParCSRMatrix *) parA,
                                                                        nrows,
                                                                        rows ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/
int HYPRE_SStructMaxwellZeroVector(HYPRE_ParVector  v,
                                   int             *rows,
                                   int              nrows)
{
    return( hypre_ParVectorZeroBCValues( (hypre_ParVector *) v,
                                                                rows,
                                                                nrows ) );
}


