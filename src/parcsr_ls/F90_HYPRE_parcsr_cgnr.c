/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRCGNR Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrcreate, HYPRE_PARCSRCGNRCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )

{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrdestroy, HYPRE_PARCSRCGNRDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRDestroy(
           hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsetup, HYPRE_PARCSRCGNRSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetup(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsolve, HYPRE_PARCSRCGNRSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSolve(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsettol, HYPRE_PARCSRCGNRSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetTol(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassDbl (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetminiter, HYPRE_PARCSRCGNRSETMINITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *min_iter,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetMinIter(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (min_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetmaxiter, HYPRE_PARCSRCGNRSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetMaxIter(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetstopcrit, HYPRE_PARCSRCGNRSETSTOPCRIT)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *stop_crit,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetStopCrit(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetprecond, HYPRE_PARCSRCGNRSETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *precond_id,
     hypre_F90_Obj *precond_solver,
     hypre_F90_Int *ierr            )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - do not set up a preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCGNRSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRDiagScale,
              HYPRE_ParCSRDiagScale,
              HYPRE_ParCSRDiagScaleSetup,
              NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCGNRSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_BoomerAMGSolve,
              HYPRE_BoomerAMGSolve,
              HYPRE_BoomerAMGSetup,
              (void *)       *precond_solver ) );
   }
   if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCGNRSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRPilutSolve,
              HYPRE_ParCSRPilutSolve,
              HYPRE_ParCSRPilutSetup,
              (void *)       *precond_solver ) );
   }
   if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCGNRSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRParaSailsSolve,
              HYPRE_ParCSRParaSailsSolve,
              HYPRE_ParCSRParaSailsSetup,
              (void *)       *precond_solver ) );
   }
   if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRCGNRSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_EuclidSolve,
              HYPRE_EuclidSolve,
              HYPRE_EuclidSetup,
              (void *)       *precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetprecond, HYPRE_PARCSRCGNRGETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *precond_solver_ptr,
     hypre_F90_Int *ierr                 )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRGetPrecond(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetlogging, HYPRE_PARCSRCGNRSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *logging,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRSetLogging(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIteration
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetnumiteration, HYPRE_PARCSRCGNRGETNUMITERATION)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRGetNumIterations(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetfinalrelativ, HYPRE_PARCSRCGNRGETFINALRELATIV)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassDblRef (norm)     ) );
}
