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
 * HYPRE_SStructMaxwell interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellcreate, HYPRE_SSTRUCTMAXWELLCREATE)
                                                (long int *comm,
                                                 long int *solver,
                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructMaxwellCreate( (MPI_Comm) *comm,
                                              (HYPRE_SStructSolver *) solver) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelldestroy, HYPRE_SSTRUCTMAXWELLDESTROY)
                                                (long int *solver,
                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructMaxwellDestroy((HYPRE_SStructSolver) *solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetup, HYPRE_SSTRUCTMAXWELLSETUP)
                                                (long int *solver,
                                                 long int *A,
                                                 long int *b,
                                                 long int *x,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetup( 
                                            (HYPRE_SStructSolver) *solver,
                                            (HYPRE_SStructMatrix) *A,
                                            (HYPRE_SStructVector) *b,
                                            (HYPRE_SStructVector) *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsolve, HYPRE_SSTRUCTMAXWELLSOLVE)
                                                (long int *solver,
                                                 long int *A,
                                                 long int *b,
                                                 long int *x,
                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructMaxwellSolve( 
                                           (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSolve2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsolve2, HYPRE_SSTRUCTMAXWELLSOLVE2)
                                                (long int *solver,
                                                 long int *A,
                                                 long int *b,
                                                 long int *x,
                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructMaxwellSolve2( 
                                            (HYPRE_SStructSolver) *solver,
                                            (HYPRE_SStructMatrix) *A,
                                            (HYPRE_SStructVector) *b,
                                            (HYPRE_SStructVector) *x     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_maxwellgrad, HYPRE_MAXWELLGRAD)
                                                (long int *grid,
                                                 long int *T,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_MaxwellGrad( (HYPRE_SStructGrid)   *grid,
                                      (HYPRE_ParCSRMatrix *) T ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetgrad, HYPRE_SSTRUCTMAXWELLSETGRAD)
                                                (long int *solver,
                                                 long int *T,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetGrad( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_ParCSRMatrix) *T ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrfactors, HYPRE_SSTRUCTMAXWELLSETRFACTORS)
                                                (long int *solver,
                                                 int     (*rfactors)[3],
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetRfactors( (HYPRE_SStructSolver) *solver,
                                                                           rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsettol, HYPRE_SSTRUCTMAXWELLSETTOL)
                                                (long int *solver,
                                                 double   *tol,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetTol( (HYPRE_SStructSolver) *solver,
                                               (double)              *tol    ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetconstant, HYPRE_SSTRUCTMAXWELLSETCONSTANT)
                                                (long int *solver,
                                                 int      *constant_coef,
                                                 int      *ierr)
{
   *ierr = (int ) ( HYPRE_SStructMaxwellSetConstantCoef( 
                                                 (HYPRE_SStructSolver ) *solver,
                                                 (int)                  *constant_coef) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetmaxiter, HYPRE_SSTRUCTMAXWELLSETMAXITER)
                                                (long int *solver,
                                                 int      *max_iter,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *max_iter  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrelchang, HYPRE_SSTRUCTMAXWELLSETRELCHANG)
                                                (long int *solver,
                                                 int      *rel_change,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetRelChange( (HYPRE_SStructSolver) *solver,
                                                     (int)                 *rel_change  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumprere, HYPRE_SSTRUCTMAXWELLSETNUMPRERE)
                                                (long int *solver,
                                                 int      *num_pre_relax,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetNumPreRelax( 
                                          (HYPRE_SStructSolver) *solver,
                                          (int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumpostr, HYPRE_SSTRUCTMAXWELLSETNUMPOSTR)
                                                (long int *solver,
                                                 int      *num_post_relax,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetNumPostRelax( 
                                          (HYPRE_SStructSolver) *solver,
                                          (int)                 *num_post_relax ));

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetlogging, HYPRE_SSTRUCTMAXWELLSETLOGGING)
                                                (long int *solver,
                                                 int      *logging,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetLogging( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *logging));
}

/*--------------------------------------------------------------------------
HYPRE_SStructMaxwellSetPrintLevel
*--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetprintlev, HYPRE_SSTRUCTMAXWELLSETPRINTLEV)
                                                (long int *solver,
                                                 int      *print_level,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellSetPrintLevel( 
                                          (HYPRE_SStructSolver) *solver,
                                          (int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellprintloggin, HYPRE_SSTRUCTMAXWELLPRINTLOGGIN)
                                                (long int *solver,
                                                 int      *myid,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellPrintLogging( 
                                       (HYPRE_SStructSolver) *solver,
                                       (int)                 *myid));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetnumitera, HYPRE_SSTRUCTMAXWELLGETNUMITERA) 
                                                (long int *solver, 
                                                 int      *num_iterations,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellGetNumIterations( 
                                       (HYPRE_SStructSolver) *solver,
                                       (int *)                num_iterations ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetfinalrel, HYPRE_SSTRUCTMAXWELLGETFINALREL) 
                                                (long int *solver, 
                                                 double   *norm,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( 
                                       (HYPRE_SStructSolver) *solver,
                                       (double *)             norm   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellphysbdy, HYPRE_SSTRUCTMAXWELLPHYSBDY) 
                                                (long int  *grid_l, 
                                                 int       *num_levels,
                                                 int      (*rfactors)[3],
                                                 int      (***BdryRanks_ptr),
                                                 int      (**BdryRanksCnt_ptr),
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellPhysBdy( 
                                       (HYPRE_SStructGrid *)  grid_l,
                                       (int)                 *num_levels,
                                                              rfactors[3],
                                                              BdryRanks_ptr,
                                                              BdryRanksCnt_ptr ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelleliminaterowscols, HYPRE_SSTRUCTMAXWELLELIMINATEROWSCOLS) 
                                                (long int *A, 
                                                 int      *nrows,
                                                 int      *rows,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellEliminateRowsCols( (HYPRE_ParCSRMatrix) *A,
                                                          (int)                *nrows,
                                                          (int *)               rows ));
}      


/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellzerovector, HYPRE_SSTRUCTMAXWELLZEROVECTOR) 
                                                (long int *b, 
                                                 int      *rows,
                                                 int      *nrows,
                                                 int      *ierr)
{
   *ierr = (int) ( HYPRE_SStructMaxwellZeroVector( (HYPRE_ParVector) *b,
                                                   (int *)            rows,
                                                   (int)             *nrows ));
}      

