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
                                                (HYPRE_Int     *comm,
                                                 hypre_F90_Obj *solver,
                                                 HYPRE_Int     *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMaxwellCreate( (MPI_Comm) *comm,
                                              (HYPRE_SStructSolver *) solver) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelldestroy, HYPRE_SSTRUCTMAXWELLDESTROY)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMaxwellDestroy((HYPRE_SStructSolver) *solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetup, HYPRE_SSTRUCTMAXWELLSETUP)
                                                (hypre_F90_Obj *solver,
                                                 hypre_F90_Obj *A,
                                                 hypre_F90_Obj *b,
                                                 hypre_F90_Obj *x,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetup( 
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
                                                (hypre_F90_Obj *solver,
                                                 hypre_F90_Obj *A,
                                                 hypre_F90_Obj *b,
                                                 hypre_F90_Obj *x,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMaxwellSolve( 
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
                                                (hypre_F90_Obj *solver,
                                                 hypre_F90_Obj *A,
                                                 hypre_F90_Obj *b,
                                                 hypre_F90_Obj *x,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMaxwellSolve2( 
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
                                                (hypre_F90_Obj *grid,
                                                 hypre_F90_Obj *T,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_MaxwellGrad( (HYPRE_SStructGrid)   *grid,
                                      (HYPRE_ParCSRMatrix *) T ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetgrad, HYPRE_SSTRUCTMAXWELLSETGRAD)
                                                (hypre_F90_Obj *solver,
                                                 hypre_F90_Obj *T,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetGrad( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_ParCSRMatrix) *T ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrfactors, HYPRE_SSTRUCTMAXWELLSETRFACTORS)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int     (*rfactors)[3],
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetRfactors( (HYPRE_SStructSolver) *solver,
                                                                           rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsettol, HYPRE_SSTRUCTMAXWELLSETTOL)
                                                (hypre_F90_Obj *solver,
                                                 double   *tol,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetTol( (HYPRE_SStructSolver) *solver,
                                               (double)              *tol    ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetconstant, HYPRE_SSTRUCTMAXWELLSETCONSTANT)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *constant_coef,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int ) ( HYPRE_SStructMaxwellSetConstantCoef( 
                                                 (HYPRE_SStructSolver ) *solver,
                                                 (HYPRE_Int)                  *constant_coef) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetmaxiter, HYPRE_SSTRUCTMAXWELLSETMAXITER)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *max_iter,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *max_iter  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrelchang, HYPRE_SSTRUCTMAXWELLSETRELCHANG)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *rel_change,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetRelChange( (HYPRE_SStructSolver) *solver,
                                                     (HYPRE_Int)                 *rel_change  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumprere, HYPRE_SSTRUCTMAXWELLSETNUMPRERE)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_pre_relax,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetNumPreRelax( 
                                          (HYPRE_SStructSolver) *solver,
                                          (HYPRE_Int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumpostr, HYPRE_SSTRUCTMAXWELLSETNUMPOSTR)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_post_relax,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetNumPostRelax( 
                                          (HYPRE_SStructSolver) *solver,
                                          (HYPRE_Int)                 *num_post_relax ));

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetlogging, HYPRE_SSTRUCTMAXWELLSETLOGGING)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *logging,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetLogging( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *logging));
}

/*--------------------------------------------------------------------------
HYPRE_SStructMaxwellSetPrintLevel
*--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetprintlev, HYPRE_SSTRUCTMAXWELLSETPRINTLEV)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *print_level,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellSetPrintLevel( 
                                          (HYPRE_SStructSolver) *solver,
                                          (HYPRE_Int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellprintloggin, HYPRE_SSTRUCTMAXWELLPRINTLOGGIN)
                                                (hypre_F90_Obj *solver,
                                                 HYPRE_Int      *myid,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellPrintLogging( 
                                       (HYPRE_SStructSolver) *solver,
                                       (HYPRE_Int)                 *myid));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetnumitera, HYPRE_SSTRUCTMAXWELLGETNUMITERA) 
                                                (hypre_F90_Obj *solver, 
                                                 HYPRE_Int      *num_iterations,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellGetNumIterations( 
                                       (HYPRE_SStructSolver) *solver,
                                       (HYPRE_Int *)                num_iterations ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetfinalrel, HYPRE_SSTRUCTMAXWELLGETFINALREL) 
                                                (hypre_F90_Obj *solver, 
                                                 double   *norm,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( 
                                       (HYPRE_SStructSolver) *solver,
                                       (double *)             norm   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellphysbdy, HYPRE_SSTRUCTMAXWELLPHYSBDY) 
                                                (hypre_F90_Obj *grid_l, 
                                                 HYPRE_Int       *num_levels,
                                                 HYPRE_Int      (*rfactors)[3],
                                                 HYPRE_Int      (***BdryRanks_ptr),
                                                 HYPRE_Int      (**BdryRanksCnt_ptr),
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellPhysBdy( 
                                       (HYPRE_SStructGrid *)  grid_l,
                                       (HYPRE_Int)                 *num_levels,
                                                              rfactors[3],
                                                              BdryRanks_ptr,
                                                              BdryRanksCnt_ptr ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellEliminateRowsCols
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelleliminatero, HYPRE_SSTRUCTMAXWELLELIMINATERO) 
                                                (hypre_F90_Obj *A, 
                                                 HYPRE_Int      *nrows,
                                                 HYPRE_Int      *rows,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellEliminateRowsCols( (HYPRE_ParCSRMatrix) *A,
                                                          (HYPRE_Int)                *nrows,
                                                          (HYPRE_Int *)               rows ));
}      


/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellzerovector, HYPRE_SSTRUCTMAXWELLZEROVECTOR) 
                                                (hypre_F90_Obj *b, 
                                                 HYPRE_Int      *rows,
                                                 HYPRE_Int      *nrows,
                                                 HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) ( HYPRE_SStructMaxwellZeroVector( (HYPRE_ParVector) *b,
                                                   (HYPRE_Int *)            rows,
                                                   (HYPRE_Int)             *nrows ));
}      

