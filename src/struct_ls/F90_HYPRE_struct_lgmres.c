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
hypre_F90_IFACE(hypre_structlgmrescreate, HYPRE_STRUCTLGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmresdestroy, HYPRE_STRUCTLGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESDestroy(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetup, HYPRE_STRUCTLGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetup(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressolve, HYPRE_STRUCTLGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSolve(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressettol, HYPRE_STRUCTLGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetabstol, HYPRE_STRUCTLGMRESSETABSTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetmaxiter, HYPRE_STRUCTLGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetkdim, HYPRE_STRUCTLGMRESSETKDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *k_dim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_StructLGMRESSetKDim(
               hypre_F90_PassObj (HYPRE_StructSolver, solver),
               hypre_F90_PassInt (k_dim) ));
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetaugdim, HYPRE_STRUCTLGMRESSETAUGDIM)
(hypre_F90_Obj *solver,
 hypre_F90_Int *aug_dim,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_StructLGMRESSetAugDim(
               hypre_F90_PassObj (HYPRE_StructSolver, solver),
               hypre_F90_PassInt (aug_dim) ));
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetprecond, HYPRE_STRUCTLGMRESSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 6 - setup a jacobi preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructSMGSolve,
                   HYPRE_StructSMGSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructPFMGSolve,
                   HYPRE_StructPFMGSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructJacobiSolve,
                   HYPRE_StructJacobiSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructLGMRESSetPrecond(
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
hypre_F90_IFACE(hypre_structlgmressetlogging, HYPRE_STRUCTLGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetLogging(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmressetprintlevel, HYPRE_STRUCTLGMRESSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESSetPrintLevel(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmresgetnumiter, HYPRE_STRUCTLGMRESGETNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESGetNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structlgmresgetfinalrel, HYPRE_STRUCTLGMRESGETFINALREL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructLGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
