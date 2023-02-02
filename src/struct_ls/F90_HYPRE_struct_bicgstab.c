/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
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
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsettol, HYPRE_STRUCTBICGSTABSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructBiCGSTABSetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetabstol, HYPRE_STRUCTBICGSTABSETABSTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructBiCGSTABSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
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
    * 7 - setup a jacobi preconditioner
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
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructJacobiSolve,
                   HYPRE_StructJacobiSetup,
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
                (void **)          residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetfinalrel, HYPRE_STRUCTBICGSTABGETFINALREL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
