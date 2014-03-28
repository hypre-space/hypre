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

#include "_hypre_struct_ls.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgcreate, HYPRE_STRUCTBAMGCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbamgdestroy, HYPRE_STRUCTBAMGDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGDestroy(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbamgsetup, HYPRE_STRUCTBAMGSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbamgsolve, HYPRE_STRUCTBAMGSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSolve(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetTol, HYPRE_StructBAMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsettol, HYPRE_STRUCTBAMGSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassReal (tol)    ) );
}

void
hypre_F90_IFACE(hypre_structbamggettol, HYPRE_STRUCTBAMGGETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassRealRef (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetMaxIter, HYPRE_StructBAMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetmaxiter, HYPRE_STRUCTBAMGSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_iter)  ) );
}

void
hypre_F90_IFACE(hypre_structbamggetmaxiter, HYPRE_STRUCTBAMGGETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (max_iter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetMaxLevels, HYPRE_StructBAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetmaxlevels, HYPRE_STRUCTBAMGSETMAXLEVELS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_levels,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetMaxLevels(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_levels)  ) );
}

void
hypre_F90_IFACE(hypre_structbamggetmaxlevels, HYPRE_STRUCTBAMGGETMAXLEVELS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_levels,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetMaxLevels(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (max_levels)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetRelChange, HYPRE_StructBAMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetrelchange, HYPRE_STRUCTBAMGSETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rel_change)  ) );
}

void
hypre_F90_IFACE(hypre_structbamggetrelchange, HYPRE_STRUCTBAMGGETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetZeroGuess, HYPRE_StructBAMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structbamgsetzeroguess, HYPRE_STRUCTBAMGSETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}
 
void
hypre_F90_IFACE(hypre_structbamggetzeroguess, HYPRE_STRUCTBAMGGETZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *zeroguess,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structbamgsetnonzeroguess, HYPRE_STRUCTBAMGSETNONZEROGUESS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetNonZeroGuess(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetRelaxType, HYPRE_StructBAMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetrelaxtype, HYPRE_STRUCTBAMGSETRELAXTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *relax_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetRelaxType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (relax_type) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetrelaxtype, HYPRE_STRUCTBAMGGETRELAXTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *relax_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetRelaxType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structbamgsetjacobiweigh, HYPRE_STRUCTBAMGSETJACOBIWEIGH)
   (hypre_F90_Obj *solver,
    hypre_F90_Real *weight,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_StructBAMGSetJacobiWeight(
          hypre_F90_PassObj (HYPRE_StructSolver, solver),
          hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetRAPType, HYPRE_StructBAMGSetRapType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetraptype, HYPRE_STRUCTBAMGSETRAPTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rap_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetRAPType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rap_type) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetraptype, HYPRE_STRUCTBAMGGETRAPTYPE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rap_type,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetRAPType(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (rap_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetNumPreRelax, HYPRE_StructBAMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetnumprerelax, HYPRE_STRUCTBAMGSETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_pre_relax) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetnumprerelax, HYPRE_STRUCTBAMGGETNUMPRERELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_pre_relax,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetNumPreRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetNumPostRelax, HYPRE_StructBAMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetnumpostrelax, HYPRE_STRUCTBAMGSETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (num_post_relax) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetnumpostrelax, HYPRE_STRUCTBAMGGETNUMPOSTRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_post_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetNumPostRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetSkipRelax, HYPRE_StructBAMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetskiprelax, HYPRE_STRUCTBAMGSETSKIPRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *skip_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetSkipRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (skip_relax) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetskiprelax, HYPRE_STRUCTBAMGGETSKIPRELAX)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *skip_relax,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetSkipRelax(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (skip_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetdxyz, HYPRE_STRUCTBAMGSETDXYZ)
   ( hypre_F90_Obj *solver,
     hypre_F90_RealArray *dxyz,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetDxyz(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassRealArray (dxyz)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetLogging, HYPRE_StructBAMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetlogging, HYPRE_STRUCTBAMGSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetlogging, HYPRE_STRUCTBAMGGETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGSetPrintLevel, HYPRE_StructBAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamgsetprintlevel, HYPRE_STRUCTBAMGSETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGSetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (print_level) ) );
}

void
hypre_F90_IFACE(hypre_structbamggetprintlevel, HYPRE_STRUCTBAMGGETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamggetnumiteration, HYPRE_STRUCTBAMGGETNUMITERATION)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetNumIterations(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructBAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbamggetfinalrelativ, HYPRE_STRUCTBAMGGETFINALRELATIV)
   ( hypre_F90_Obj *solver,
     hypre_F90_Real *norm,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBAMGGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassRealRef (norm)   ) );
}
