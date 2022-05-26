/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmrescreate, HYPRE_PARCSRGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresdestroy, HYPRE_PARCSRGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetup, HYPRE_PARCSRGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressolve, HYPRE_PARCSRGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetkdim, HYPRE_PARCSRGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetKDim(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressettol, HYPRE_PARCSRGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetabsolutetol, HYPRE_PARCSRGMRESSETABSOLUTETOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetminiter, HYPRE_PARCSRGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetMinIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetmaxiter, HYPRE_PARCSRGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetstopcrit, HYPRE_PARCSRGMRESSETSTOPCRIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetStopCrit(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprecond, HYPRE_PARCSRGMRESSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    *  0 - no preconditioner
    *  1 - set up a ds preconditioner
    *  2 - set up an amg preconditioner
    *  3 - set up a pilut preconditioner
    *  4 - set up a parasails preconditioner
    *  5 - set up a Euclid preconditioner
    *  6 - set up a ILU preconditioner
    *  7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRDiagScale,
                   HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_BoomerAMGSolve,
                   HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRPilutSolve,
                   HYPRE_ParCSRPilutSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRParaSailsSolve,
                   HYPRE_ParCSRParaSailsSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_EuclidSolve,
                   HYPRE_EuclidSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ILUSolve,
                   HYPRE_ILUSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_MGRSolve,
                   HYPRE_MGRSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetprecond, HYPRE_PARCSRGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESGetPrecond(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetlogging, HYPRE_PARCSRGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprintlevel, HYPRE_PARCSRGMRESSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetnumiteratio, HYPRE_PARCSRGMRESGETNUMITERATIO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetfinalrelati, HYPRE_PARCSRGMRESGETFINALRELATI)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
