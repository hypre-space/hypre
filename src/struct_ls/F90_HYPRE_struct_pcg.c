/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgcreate, HYPRE_STRUCTPCGCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgdestroy, HYPRE_STRUCTPCGDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGDestroy(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgsetup, HYPRE_STRUCTPCGSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgsolve, HYPRE_STRUCTPCGSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSolve(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsettol, HYPRE_STRUCTPCGSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDbl (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetmaxiter, HYPRE_STRUCTPCGSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsettwonorm, HYPRE_STRUCTPCGSETTWONORM)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *two_norm,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetTwoNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetrelchange, HYPRE_STRUCTPCGSETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetRelChange(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetprecond, HYPRE_STRUCTPCGSETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *precond_id,
     hypre_F90_Obj *precond_solver,
     hypre_F90_Int *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 7 - setup a jacobi preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructSMGSolve,
              HYPRE_StructSMGSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructPFMGSolve,
              HYPRE_StructPFMGSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructJacobiSolve,
              HYPRE_StructJacobiSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructDiagScale,
              HYPRE_StructDiagScaleSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetlogging, HYPRE_STRUCTPCGSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetprintlevel, HYPRE_STRUCTPCGSETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGSetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcggetnumiterations, HYPRE_STRUCTPCGGETNUMITERATIONS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGGetNumIterations(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcggetfinalrelative, HYPRE_STRUCTPCGGETFINALRELATIVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructPCGGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScaleSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structdiagscalesetup, HYPRE_STRUCTDIAGSCALESETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *y,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructDiagScaleSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, y),
           hypre_F90_PassObj (HYPRE_StructVector, x)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScale
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structdiagscale, HYPRE_STRUCTDIAGSCALE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *HA,
     hypre_F90_Obj *Hy,
     hypre_F90_Obj *Hx,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructDiagScale(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, HA),
           hypre_F90_PassObj (HYPRE_StructVector, Hy),
           hypre_F90_PassObj (HYPRE_StructVector, Hx)     ) );
}
