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
 * HYPRE_BiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabcreate, HYPRE_STRUCTBICGSTABCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabdestroy, HYPRE_STRUCTBICGSTABDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABDestroy(
           hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabsetup, HYPRE_STRUCTBICGSTABSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSetup(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabsolve, HYPRE_STRUCTBICGSTABSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSolve(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassObj (HYPRE_StructMatrix, A),
           hypre_F90_PassObj (HYPRE_StructVector, b),
           hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsettol, HYPRE_STRUCTBICGSTABSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSetTol(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDbl (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetmaxiter, HYPRE_STRUCTBICGSTABSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSetMaxIter(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprecond, HYPRE_STRUCTBICGSTABSETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *precond_id,
     hypre_F90_Obj *precond_solver,
     hypre_F90_Int *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructBiCGSTABSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructSMGSolve,
              HYPRE_StructSMGSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructBiCGSTABSetPrecond(
              hypre_F90_PassObj (HYPRE_StructSolver, solver),
              HYPRE_StructPFMGSolve,
              HYPRE_StructPFMGSetup,
              hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_StructBiCGSTABSetPrecond(
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
 * HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetlogging, HYPRE_STRUCTBICGSTABSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSetLogging(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprintlev, HYPRE_STRUCTBICGSTABSETPRINTLEV)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *print_level,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABSetPrintLevel(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetnumitera, HYPRE_STRUCTBICGSTABGETNUMITERA)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABGetNumIterations(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetresidual, HYPRE_STRUCTBICGSTABGETRESIDUAL)
   ( hypre_F90_Obj *solver,
     void *residual,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABGetResidual(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           (void *)          residual ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetfinalrel, HYPRE_STRUCTBICGSTABGETFINALREL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_StructSolver, solver),
           hypre_F90_PassDblRef (norm) ) );
}
