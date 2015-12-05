/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
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
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMaxwellCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObjRef (HYPRE_SStructSolver, solver)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwelldestroy, HYPRE_SSTRUCTMAXWELLDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMaxwellDestroy(
          hypre_F90_PassObj (HYPRE_SStructSolver, solver)));
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
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetup( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassObj (HYPRE_SStructMatrix, A),
           hypre_F90_PassObj (HYPRE_SStructVector, b),
           hypre_F90_PassObj (HYPRE_SStructVector, x) ));
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
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMaxwellSolve( 
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)     ) );
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
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMaxwellSolve2( 
          hypre_F90_PassObj (HYPRE_SStructSolver, solver),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, b),
          hypre_F90_PassObj (HYPRE_SStructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MaxwellGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_maxwellgrad, HYPRE_MAXWELLGRAD)
   (hypre_F90_Obj *grid,
    hypre_F90_Obj *T,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_MaxwellGrad(
           hypre_F90_PassObj (HYPRE_SStructGrid, grid),
           hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, T) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetGrad
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetgrad, HYPRE_SSTRUCTMAXWELLSETGRAD)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *T,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetGrad(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, T) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRfactors
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrfactors, HYPRE_SSTRUCTMAXWELLSETRFACTORS)
   (hypre_F90_Obj *solver,
    HYPRE_Int     (*rfactors)[3],
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetRfactors(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           rfactors[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsettol, HYPRE_SSTRUCTMAXWELLSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetTol(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassDbl (tol)    ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetConstantCoef
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetconstant, HYPRE_SSTRUCTMAXWELLSETCONSTANT)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *constant_coef,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetConstantCoef( 
           (HYPRE_SStructSolver ) *solver,
           hypre_F90_PassInt (constant_coef)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetmaxiter, HYPRE_SSTRUCTMAXWELLSETMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_iter,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetMaxIter(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (max_iter)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetrelchang, HYPRE_SSTRUCTMAXWELLSETRELCHANG)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *rel_change,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetRelChange(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (rel_change)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumprere, HYPRE_SSTRUCTMAXWELLSETNUMPRERE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_pre_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetNumPreRelax( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetnumpostr, HYPRE_SSTRUCTMAXWELLSETNUMPOSTR)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_post_relax,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetNumPostRelax( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (num_post_relax) ));

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetlogging, HYPRE_SSTRUCTMAXWELLSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetLogging(
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (logging)));
}

/*--------------------------------------------------------------------------
  HYPRE_SStructMaxwellSetPrintLevel
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellsetprintlev, HYPRE_SSTRUCTMAXWELLSETPRINTLEV)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellSetPrintLevel( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (print_level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellprintloggin, HYPRE_SSTRUCTMAXWELLPRINTLOGGIN)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *myid,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellPrintLogging( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassInt (myid)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetnumitera, HYPRE_SSTRUCTMAXWELLGETNUMITERA) 
   (hypre_F90_Obj *solver, 
    hypre_F90_Int *num_iterations,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellGetNumIterations( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellgetfinalrel, HYPRE_SSTRUCTMAXWELLGETFINALREL) 
   (hypre_F90_Obj *solver, 
    hypre_F90_Dbl *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellGetFinalRelativeResidualNorm( 
           hypre_F90_PassObj (HYPRE_SStructSolver, solver),
           hypre_F90_PassDblRef (norm)   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellPhysBdy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellphysbdy, HYPRE_SSTRUCTMAXWELLPHYSBDY) 
   (hypre_F90_Obj *grid_l, 
    hypre_F90_Int *num_levels,
    HYPRE_Int      (*rfactors)[3],
    HYPRE_Int      (***BdryRanks_ptr),
    HYPRE_Int      (**BdryRanksCnt_ptr),
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellPhysBdy( 
           hypre_F90_PassObjRef (HYPRE_SStructGrid, grid_l),
           hypre_F90_PassInt (num_levels),
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
    hypre_F90_Int *nrows,
    hypre_F90_IntArray *rows,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellEliminateRowsCols(
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassInt (nrows),
           hypre_F90_PassIntArray (rows) ));
}      


/*--------------------------------------------------------------------------
 * HYPRE_SStructMaxwellZeroVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmaxwellzerovector, HYPRE_SSTRUCTMAXWELLZEROVECTOR) 
   (hypre_F90_Obj *b, 
    hypre_F90_IntArray *rows,
    hypre_F90_Int *nrows,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_SStructMaxwellZeroVector(
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassIntArray (rows),
           hypre_F90_PassInt (nrows) ));
}      

