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
 * HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgcreate, HYPRE_STRUCTSPARSEMSGCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgdestroy, HYPRE_STRUCTSPARSEMSGDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGDestroy(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetup, HYPRE_STRUCTSPARSEMSGSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetup(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsolve, HYPRE_STRUCTSPARSEMSGSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSolve(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsettol, HYPRE_STRUCTSPARSEMSGSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetmaxiter, HYPRE_STRUCTSPARSEMSGSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetjump, HYPRE_STRUCTSPARSEMSGSETJUMP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *jump,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetJump(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (jump) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelchan, HYPRE_STRUCTSPARSEMSGSETRELCHAN)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetRelChange(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetzerogue, HYPRE_STRUCTSPARSEMSGSETZEROGUE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetZeroGuess(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnonzero, HYPRE_STRUCTSPARSEMSGSETNONZERO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetNonZeroGuess(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelaxty, HYPRE_STRUCTSPARSEMSGSETRELAXTY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetRelaxType(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structsparsemsgsetjacobiweigh, HYPRE_STRUCTSPARSEMSGSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_StructSparseMSGSetJacobiWeight(
               hypre_F90_PassObj (HYPRE_StructSolver, solver),
               hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumprer, HYPRE_STRUCTSPARSEMSGSETNUMPRER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_pre_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetNumPreRelax(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_pre_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumpost, HYPRE_STRUCTSPARSEMSGSETNUMPOST)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_post_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetNumPostRelax(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_post_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumfine, HYPRE_STRUCTSPARSEMSGSETNUMFINE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_fine_relax,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetNumFineRelax(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (num_fine_relax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetlogging, HYPRE_STRUCTSPARSEMSGSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetLogging(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetprintle, HYPRE_STRUCTSPARSEMSGSETPRINTLE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGSetPrintLevel(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetnumiter, HYPRE_STRUCTSPARSEMSGGETNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGGetNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetfinalre, HYPRE_STRUCTSPARSEMSGGETFINALRE)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm) ) );
}

#ifdef __cplusplus
}
#endif
