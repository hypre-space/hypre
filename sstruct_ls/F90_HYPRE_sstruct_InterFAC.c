/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
               (hypre_F90_Comm *comm, hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACCreate( (MPI_Comm)             *comm,
                                           (HYPRE_SStructSolver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacdestroy2, HYPRE_SSTRUCTFACDESTROY2)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACDestroy2( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacamrrap, HYPRE_SSTRUCTFACAMRRAP)
               (hypre_F90_Obj *A, HYPRE_Int (*rfactors)[3], hypre_F90_Obj *facA, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACAMR_RAP( (HYPRE_SStructMatrix) *A,
                                                            rfactors,
                                            (HYPRE_SStructMatrix *) facA ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetup2, HYPRE_SSTRUCTFACSETUP2)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetup2( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix)  *A,
                                           (HYPRE_SStructVector)  *b,
                                           (HYPRE_SStructVector)  *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsolve3, HYPRE_SSTRUCTFACSOLVE3)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSolve3( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsettol, HYPRE_SSTRUCTFACSETTOL)
               (hypre_F90_Obj *solver, double *tol, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetTol( (HYPRE_SStructSolver) *solver,
                                           (double)              *tol ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetplevels, HYPRE_SSTRUCTFACSETPLEVELS)
               (hypre_F90_Obj *solver, HYPRE_Int *nparts, HYPRE_Int *plevels, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetPLevels( (HYPRE_SStructSolver) *solver,
                                               (HYPRE_Int)                 *nparts,
                                               (HYPRE_Int *)                plevels));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerocfsten, HYPRE_SSTRUCTFACZEROCFSTEN)
               (hypre_F90_Obj *A, hypre_F90_Obj *grid, HYPRE_Int *part, HYPRE_Int (*rfactors)[3], HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACZeroCFSten( (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructGrid)   *grid,
                                               (HYPRE_Int)                 *part,
                                                                      rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerofcsten, HYPRE_SSTRUCTFACZEROFCSTEN)
               (hypre_F90_Obj *A, hypre_F90_Obj *grid, HYPRE_Int *part, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACZeroFCSten( (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructGrid)   *grid,
                                               (HYPRE_Int)                 *part ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrmatrixdata, HYPRE_SSTRUCTFACZEROAMRMATRIXDATA)
               (hypre_F90_Obj *A, HYPRE_Int *part_crse, HYPRE_Int (*rfactors)[3], HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACZeroAMRMatrixData( (HYPRE_SStructMatrix) *A,
                                                      (HYPRE_Int)                 *part_crse,
                                                                             rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrvectordata, HYPRE_SSTRUCTFACZEROAMRVECTORDATA)
               (hypre_F90_Obj *b, HYPRE_Int *plevels, HYPRE_Int (*rfactors)[3], HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACZeroAMRVectorData( (HYPRE_SStructVector) *b,
                                                      (HYPRE_Int *)                plevels,
                                                              rfactors ));
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetprefinements, HYPRE_SSTRUCTFACSETPREFINEMENTS)
               (hypre_F90_Obj *solver, HYPRE_Int *nparts, HYPRE_Int (*rfactors)[3], HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetPRefinements( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int)                 *nparts,
                                                            rfactors ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxlevels, HYPRE_SSTRUCTFACSETMAXLEVELS)
               (hypre_F90_Obj *solver, HYPRE_Int *max_levels, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetMaxLevels( (HYPRE_SStructSolver) *solver,
                                                 (HYPRE_Int)                 *max_levels ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxiter, HYPRE_SSTRUCTFACSETMAXITER)
               (hypre_F90_Obj *solver, HYPRE_Int *max_iter, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetMaxIter( (HYPRE_SStructSolver) *solver,
                                               (HYPRE_Int)                 *max_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelchange, HYPRE_SSTRUCTFACSETRELCHANGE)
               (hypre_F90_Obj *solver, HYPRE_Int *rel_change, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetRelChange( (HYPRE_SStructSolver) *solver,
                                                 (HYPRE_Int)                 *rel_change ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetzeroguess, HYPRE_SSTRUCTFACSETZEROGUESS)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnonzeroguess, HYPRE_SSTRUCTFACSETNONZEROGUESS)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelaxtype, HYPRE_SSTRUCTFACSETRELAXTYPE)
               (hypre_F90_Obj *solver, HYPRE_Int *relax_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetRelaxType( (HYPRE_SStructSolver) *solver,
                                                 (HYPRE_Int)                 *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_sstructfacsetjacobiweigh, HYPRE_SSTRUCTFACSETJACOBIWEIGH)
                                                  (hypre_F90_Obj *solver,
                                                   double   *weight,
                                                   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructFACSetJacobiWeight( (HYPRE_SStructSolver) *solver,
                                                   (double)              *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumprerelax, HYPRE_SSTRUCTFACSETNUMPRERELAX)
               (hypre_F90_Obj *solver, HYPRE_Int *num_pre_relax, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructFACSetNumPreRelax( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumpostrelax, HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
               (hypre_F90_Obj *solver, HYPRE_Int *num_post_relax, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructFACSetNumPostRelax((HYPRE_SStructSolver) *solver,
                                                  (HYPRE_Int)                  *num_post_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetcoarsesolver, HYPRE_SSTRUCTFACSETCOARSESOLVER)
               (hypre_F90_Obj *solver, HYPRE_Int * csolver_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) 
           (HYPRE_SStructFACSetCoarseSolverType( (HYPRE_SStructSolver) *solver,
                                                 (HYPRE_Int)                 *csolver_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetlogging, HYPRE_SSTRUCTFACSETLOGGING)
               (hypre_F90_Obj *solver, HYPRE_Int *logging, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructFACSetLogging( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_Int)                 *logging ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetnumiteration, HYPRE_SSTRUCTFACGETNUMITERATION)
               (hypre_F90_Obj *solver, HYPRE_Int *num_iterations, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int)  
           ( HYPRE_SStructFACGetNumIterations( (HYPRE_SStructSolver) *solver,
                                               (HYPRE_Int *)                num_iterations));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetfinalrelativ, HYPRE_SSTRUCTFACGETFINALRELATIV)
               (hypre_F90_Obj *solver, double *norm, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) 
           ( HYPRE_SStructFACGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                           (double *)             norm ));
}
