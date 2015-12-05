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

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgcreate, HYPRE_STRUCTPFMGCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgdestroy, HYPRE_STRUCTPFMGDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGDestroy(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsetup, HYPRE_STRUCTPFMGSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsolve, HYPRE_STRUCTPFMGSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSolve(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetTol, HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsettol, HYPRE_STRUCTPFMGSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDbl (tol)    ) );
}

void
hypre_F90_IFACE(hypre_structpfmggettol, HYPRE_STRUCTPFMGGETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter, HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxiter, HYPRE_STRUCTPFMGSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_iter)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxiter, HYPRE_STRUCTPFMGGETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxLevels, HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxlevels, HYPRE_STRUCTPFMGSETMAXLEVELS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_levels,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetMaxLevels(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_levels)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxlevels, HYPRE_STRUCTPFMGGETMAXLEVELS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_levels,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetMaxLevels(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (max_levels)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange, HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelchange, HYPRE_STRUCTPFMGSETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rel_change)  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelchange, HYPRE_STRUCTPFMGGETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess, HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetzeroguess, HYPRE_STRUCTPFMGSETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}
 
void
hypre_F90_IFACE(hypre_structpfmggetzeroguess, HYPRE_STRUCTPFMGGETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *zeroguess,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetnonzeroguess, HYPRE_STRUCTPFMGSETNONZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetNonZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType, HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelaxtype, HYPRE_STRUCTPFMGSETRELAXTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *relax_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetRelaxType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (relax_type) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelaxtype, HYPRE_STRUCTPFMGGETRELAXTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *relax_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetRelaxType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structpfmgsetjacobiweigh, HYPRE_STRUCTPFMGSETJACOBIWEIGH)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *weight,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_StructPFMGSetJacobiWeight(
          hypre_F90_PassObj (HYPRE_StructSolver, solver),
          hypre_F90_PassDbl (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRAPType, HYPRE_StructPFMGSetRapType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetraptype, HYPRE_STRUCTPFMGSETRAPTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rap_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetRAPType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rap_type) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetraptype, HYPRE_STRUCTPFMGGETRAPTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rap_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetRAPType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (rap_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax, HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumprerelax, HYPRE_STRUCTPFMGSETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_pre_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumprerelax, HYPRE_STRUCTPFMGGETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax, HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumpostrelax, HYPRE_STRUCTPFMGSETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_post_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumpostrelax, HYPRE_STRUCTPFMGGETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetSkipRelax, HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetskiprelax, HYPRE_STRUCTPFMGSETSKIPRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *skip_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetSkipRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (skip_relax) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetskiprelax, HYPRE_STRUCTPFMGGETSKIPRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *skip_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetSkipRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (skip_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetdxyz, HYPRE_STRUCTPFMGSETDXYZ)
   ( hypre_F90_Obj *solver,
     hypre_F90_DblArray *dxyz,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetDxyz(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblArray (dxyz)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging, HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetlogging, HYPRE_STRUCTPFMGSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetlogging, HYPRE_STRUCTPFMGGETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetPrintLevel, HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetprintlevel, HYPRE_STRUCTPFMGSETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGSetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (print_level) ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetprintlevel, HYPRE_STRUCTPFMGGETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetnumiteration, HYPRE_STRUCTPFMGGETNUMITERATION)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetNumIterations(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetfinalrelativ, HYPRE_STRUCTPFMGGETFINALRELATIV)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPFMGGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (norm)   ) );
}
