/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRCOGMRES Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmrescreate, HYPRE_PARCSRCOGMRESCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresdestroy, HYPRE_PARCSRCOGMRESDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetup, HYPRE_PARCSRCOGMRESSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressolve, HYPRE_PARCSRCOGMRESSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetkdim, HYPRE_PARCSRCOGMRESSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *kdim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetKDim(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (kdim)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetunroll, HYPRE_PARCSRCOGMRESSETUNROLL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *unroll,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetUnroll(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (unroll)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetcgs, HYPRE_PARCSRCOGMRESSETCGS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *cgs,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetCGS(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (cgs)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressettol, HYPRE_PARCSRCOGMRESSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetabsolutet, HYPRE_PARCSRCOGMRESSETABSOLUTET)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetminiter, HYPRE_PARCSRCOGMRESSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetMinIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetmaxiter, HYPRE_PARCSRCOGMRESSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetprecond, HYPRE_PARCSRCOGMRESSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - no preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    * 4 - set up a parasails preconditioner
    * 5 - set up a Euclid preconditioner
    * 6 - set up a ILU preconditioner
    * 7 - set up a MGR preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRDiagScale,
                   HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_BoomerAMGSolve,
                   HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRPilutSolve,
                   HYPRE_ParCSRPilutSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRParaSailsSolve,
                   HYPRE_ParCSRParaSailsSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_EuclidSolve,
                   HYPRE_EuclidSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ILUSolve,
                   HYPRE_ILUSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRCOGMRESSetPrecond(
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
 * HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetprecond, HYPRE_PARCSRCOGMRESGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESGetPrecond(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetlogging, HYPRE_PARCSRCOGMRESSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmressetprintleve, HYPRE_PARCSRCOGMRESSETPRINTLEVE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetnumiterat, HYPRE_PARCSRCOGMRESGETNUMITERAT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcogmresgetfinalrela, HYPRE_PARCSRCOGMRESGETFINALRELA)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
