/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRLGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmrescreate, HYPRE_PARCSRLGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresdestroy, HYPRE_PARCSRLGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetup, HYPRE_PARCSRLGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressolve, HYPRE_PARCSRLGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetkdim, HYPRE_PARCSRLGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetKDim(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressettol, HYPRE_PARCSRLGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetabsolutetol, HYPRE_PARCSRLGMRESSETABSOLUTETOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetminiter, HYPRE_PARCSRLGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetMinIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetmaxiter, HYPRE_PARCSRLGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprecond, HYPRE_PARCSRLGMRESSETPRECOND)
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
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRDiagScale,
                   HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_BoomerAMGSolve,
                   HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRPilutSolve,
                   HYPRE_ParCSRPilutSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRParaSailsSolve,
                   HYPRE_ParCSRParaSailsSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_EuclidSolve,
                   HYPRE_EuclidSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ILUSolve,
                   HYPRE_ILUSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond(
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
 * HYPRE_ParCSRLGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetprecond, HYPRE_PARCSRLGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESGetPrecond(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetlogging, HYPRE_PARCSRLGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprintlevel, HYPRE_PARCSRLGMRESSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetnumiteratio, HYPRE_PARCSRLGMRESGETNUMITERATIO)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetfinalrelati, HYPRE_PARCSRLGMRESGETFINALRELATI)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
