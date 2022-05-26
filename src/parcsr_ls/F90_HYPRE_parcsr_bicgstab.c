/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParCSRBiCGSTAB Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabcreate, HYPRE_PARCSRBICGSTABCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabdestroy, HYPRE_PARCSRBICGSTABDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetup, HYPRE_PARCSRBICGSTABSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsolve, HYPRE_PARCSRBICGSTABSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsettol, HYPRE_PARCSRBICGSTABSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetatol, HYPRE_PARCSRBICGSTABSETATOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetAbsoluteTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetminiter, HYPRE_PARCSRBICGSTABSETMINITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *min_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetMinIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetmaxiter, HYPRE_PARCSRBICGSTABSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSeStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetstopcrit, HYPRE_PARCSRBICGSTABSETSTOP)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetStopCrit(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetprecond, HYPRE_PARCSRBICGSTABSETPRECOND)
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
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRDiagScale,
                   HYPRE_ParCSRDiagScaleSetup,
                   NULL                        ) );
   }
   else if (*precond_id == 2)
   {

      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_BoomerAMGSolve,
                   HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver)      * precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRPilutSolve,
                   HYPRE_ParCSRPilutSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ParCSRParaSailsSolve,
                   HYPRE_ParCSRParaSailsSetup,
                   (HYPRE_Solver)      * precond_solver ) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_EuclidSolve,
                   HYPRE_EuclidSetup,
                   (HYPRE_Solver)      * precond_solver ) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   HYPRE_ILUSolve,
                   HYPRE_ILUSetup,
                   (HYPRE_Solver)       * precond_solver ) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond(
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
 * HYPRE_ParCSRBiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetprecond, HYPRE_PARCSRBICGSTABGETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *precond_solver_ptr,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABGetPrecond(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetlogging, HYPRE_PARCSRBICGSTABSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetprintlev, HYPRE_PARCSRBICGSTABSETPRINTLEV)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetnumiter, HYPRE_PARCSRBICGSTABGETNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetfinalrel, HYPRE_PARCSRBICGSTABGETFINALREL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
