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
 * HYPRE_StructJacobiCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobicreate, HYPRE_STRUCTJACOBICREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobidestroy, HYPRE_STRUCTJACOBIDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiDestroy(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetup, HYPRE_STRUCTJACOBISETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSetup(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisolve, HYPRE_STRUCTJACOBISOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSolve(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisettol, HYPRE_STRUCTJACOBISETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigettol, HYPRE_STRUCTJACOBIGETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiGetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetmaxiter, HYPRE_STRUCTJACOBISETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSetMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetmaxiter, HYPRE_STRUCTJACOBIGETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiGetMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetzeroguess, HYPRE_STRUCTJACOBISETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSetZeroGuess(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetzeroguess, HYPRE_STRUCTJACOBIGETZEROGUESS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *zeroguess,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiGetZeroGuess(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (zeroguess) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobisetnonzerogue, HYPRE_STRUCTJACOBISETNONZEROGUE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiSetNonZeroGuess(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetnumiterati, HYPRE_STRUCTJACOBIGETNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiGetNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structjacobigetfinalrelat, HYPRE_STRUCTJACOBIGETFINALRELAT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructJacobiGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
