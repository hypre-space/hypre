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
 * HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgcreate, HYPRE_STRUCTSMGCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgdestroy, HYPRE_STRUCTSMGDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGDestroy(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsetup, HYPRE_STRUCTSMGSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsolve, HYPRE_STRUCTSMGSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSolve(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse, HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse, HYPRE_STRUCTSMGSETMEMORYUSE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *memory_use,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetMemoryUse(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (memory_use) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmemoryuse, HYPRE_STRUCTSMGGETMEMORYUSE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *memory_use,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetMemoryUse(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (memory_use) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol, HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol, HYPRE_STRUCTSMGSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDbl (tol) ) );
}

void
hypre_F90_IFACE(hypre_structsmggettol, HYPRE_STRUCTSMGGETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter, HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter, HYPRE_STRUCTSMGSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_iter) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmaxiter, HYPRE_STRUCTSMGGETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange, HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange, HYPRE_STRUCTSMGSETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rel_change) ) );
}

void
hypre_F90_IFACE(hypre_structsmggetrelchange, HYPRE_STRUCTSMGGETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess, HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetzeroguess, HYPRE_STRUCTSMGSETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}
 
void
hypre_F90_IFACE(hypre_structsmggetzeroguess, HYPRE_STRUCTSMGGETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *zeroguess,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess, HYPRE_STRUCTSMGSETNONZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetNonZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax, HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax, HYPRE_STRUCTSMGSETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_pre_relax)) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumprerelax, HYPRE_STRUCTSMGGETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_pre_relax)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax, HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax, HYPRE_STRUCTSMGSETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_post_relax)) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumpostrelax, HYPRE_STRUCTSMGGETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_post_relax)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging, HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging, HYPRE_STRUCTSMGSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (logging)) );
}

void
hypre_F90_IFACE(hypre_structsmggetlogging, HYPRE_STRUCTSMGGETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (logging)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetPrintLevel, HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetprintlevel, HYPRE_STRUCTSMGSETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGSetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (print_level)) );
}

void
hypre_F90_IFACE(hypre_structsmggetprintlevel, HYPRE_STRUCTSMGGETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (print_level)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations, HYPRE_STRUCTSMGGETNUMITERATIONS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetNumIterations(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative, HYPRE_STRUCTSMGGETFINALRELATIVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructSMGGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (norm) ) );
}
