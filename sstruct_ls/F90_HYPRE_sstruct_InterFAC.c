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
#include "fortran.h"


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaccreate, HYPRE_SSTRUCTFACCREATE)
               (int *comm, long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACCreate( (MPI_Comm)             *comm,
                                           (HYPRE_SStructSolver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacdestroy2, HYPRE_SSTRUCTFACDESTROY2)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACDestroy2( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacamrrap, HYPRE_SSTRUCTFACAMRRAP)
               (long int *A, int (*rfactors)[3], long int *facA, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACAMR_RAP( (HYPRE_SStructMatrix) *A,
                                                            rfactors,
                                            (HYPRE_SStructMatrix *) facA ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetup2, HYPRE_SSTRUCTFACSETUP2)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetup2( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix)  *A,
                                           (HYPRE_SStructVector)  *b,
                                           (HYPRE_SStructVector)  *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsolve3, HYPRE_SSTRUCTFACSOLVE3)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSolve3( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsettol, HYPRE_SSTRUCTFACSETTOL)
               (long int *solver, double *tol, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetTol( (HYPRE_SStructSolver) *solver,
                                           (double)              *tol ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetplevels, HYPRE_SSTRUCTFACSETPLEVELS)
               (long int *solver, int *nparts, int *plevels, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetPLevels( (HYPRE_SStructSolver) *solver,
                                               (int)                 *nparts,
                                               (int *)                plevels));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerocfsten, HYPRE_SSTRUCTFACZEROCFSTEN)
               (long int *A, long int *grid, int *part, int (*rfactors)[3], int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACZeroCFSten( (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructGrid)   *grid,
                                               (int)                 *part,
                                                                      rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerofcsten, HYPRE_SSTRUCTFACZEROFCSTEN)
               (long int *A, long int *grid, int *part, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACZeroFCSten( (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructGrid)   *grid,
                                               (int)                 *part ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrmatrixdata, HYPRE_SSTRUCTFACZEROAMRMATRIXDATA)
               (long int *A, int *part_crse, int (*rfactors)[3], int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACZeroAMRMatrixData( (HYPRE_SStructMatrix) *A,
                                                      (int)                 *part_crse,
                                                                             rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrvectordata, HYPRE_SSTRUCTFACZEROAMRVECTORDATA)
               (long int *b, int *plevels, int (*rfactors)[3], int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACZeroAMRVectorData( (HYPRE_SStructVector) *b,
                                                      (int *)                plevels,
                                                              rfactors ));
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetprefinements, HYPRE_SSTRUCTFACSETPREFINEMENTS)
               (long int *solver, int *nparts, int (*rfactors)[3], int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetPRefinements( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *nparts,
                                                            rfactors ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxlevels, HYPRE_SSTRUCTFACSETMAXLEVELS)
               (long int *solver, int *max_levels, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetMaxLevels( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *max_levels ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxiter, HYPRE_SSTRUCTFACSETMAXITER)
               (long int *solver, int *max_iter, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetMaxIter( (HYPRE_SStructSolver) *solver,
                                               (int)                 *max_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelchange, HYPRE_SSTRUCTFACSETRELCHANGE)
               (long int *solver, int *rel_change, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetRelChange( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *rel_change ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetzeroguess, HYPRE_SSTRUCTFACSETZEROGUESS)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnonzeroguess, HYPRE_SSTRUCTFACSETNONZEROGUESS)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelaxtype, HYPRE_SSTRUCTFACSETRELAXTYPE)
               (long int *solver, int *relax_type, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetRelaxType( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumprerelax, HYPRE_SSTRUCTFACSETNUMPRERELAX)
               (long int *solver, int *num_pre_relax, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetNumPreRelax( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumpostrelax, HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
               (long int *solver, int *num_post_relax, int *ierr)
{
   *ierr = (int) (HYPRE_SStructFACSetNumPostRelax((HYPRE_SStructSolver) *solver,
                                                  (int)                  *num_post_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetcoarsesolver, HYPRE_SSTRUCTFACSETCOARSESOLVER)
               (long int *solver, int * csolver_type, int *ierr)
{
   *ierr = (int) 
           (HYPRE_SStructFACSetCoarseSolverType( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *csolver_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetlogging, HYPRE_SSTRUCTFACSETLOGGING)
               (long int *solver, int *logging, int *ierr)
{
   *ierr = (int) (HYPRE_SStructFACSetLogging( (HYPRE_SStructSolver) *solver,
                                              (int)                 *logging ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetnumiteration, HYPRE_SSTRUCTFACGETNUMITERATION)
               (long int *solver, int *num_iterations, int *ierr)
{
   *ierr = (int)  
           ( HYPRE_SStructFACGetNumIterations( (HYPRE_SStructSolver) *solver,
                                               (int *)                num_iterations));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetfinalrelativ, HYPRE_SSTRUCTFACGETFINALRELATIV)
               (long int *solver, double *norm, int *ierr)
{
   *ierr = (int) 
           ( HYPRE_SStructFACGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                           (double *)             norm ));
}
