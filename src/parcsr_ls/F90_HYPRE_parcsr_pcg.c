/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgcreate, HYPRE_PARCSRPCGCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgdestroy, HYPRE_PARCSRPCGDESTROY)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGDestroy(
           hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsetup, HYPRE_PARCSRPCGSETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetup(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsolve, HYPRE_PARCSRPCGSOLVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *b,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSolve(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, b),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettol, HYPRE_PARCSRPCGSETTOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetTol(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassDbl (tol)     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetAbsoluteTol
 *-------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetatol, HYPRE_PARCSRPCGSETATOL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *tol,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetAbsoluteTol(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassDbl (tol)     ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetmaxiter, HYPRE_PARCSRPCGSETMAXITER)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *max_iter,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetMaxIter(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetstopcrit, HYPRE_PARCSRPCGSETSTOPCRIT)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *stop_crit,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetStopCrit(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettwonorm, HYPRE_PARCSRPCGSETTWONORM)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *two_norm,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetTwoNorm(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (two_norm) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetrelchange, HYPRE_PARCSRPCGSETRELCHANGE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *rel_change,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetRelChange(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (rel_change) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprecond, HYPRE_PARCSRPCGSETPRECOND)
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
    * 4 - set up a ParaSails preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRDiagScale,
              HYPRE_ParCSRDiagScaleSetup,
              NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_BoomerAMGSolve,
              HYPRE_BoomerAMGSetup,
              (void *)       *precond_solver) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParCSRPilutSolve,
              HYPRE_ParCSRPilutSetup,
              (void *)       *precond_solver) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_ParaSailsSolve,
              HYPRE_ParaSailsSetup,
              (void *)       *precond_solver) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
         ( HYPRE_ParCSRPCGSetPrecond(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              HYPRE_EuclidSolve,
              HYPRE_EuclidSetup,
              (void *)       *precond_solver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetprecond, HYPRE_PARCSRPCGGETPRECOND)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *precond_solver_ptr,
     hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGGetPrecond(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObjRef (HYPRE_Solver, precond_solver_ptr) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprintlevel, HYPRE_PARCSRPCGSETPRINTLEVEL)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *level,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetPrintLevel(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetlogging, HYPRE_PARCSRPCGSETLOGGING)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *level,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGSetLogging(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetnumiterations, HYPRE_PARCSRPCGGETNUMITERATIONS)
   ( hypre_F90_Obj *solver,
     hypre_F90_Int *num_iterations,
     hypre_F90_Int *ierr            )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGGetNumIterations(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetfinalrelative, HYPRE_PARCSRPCGGETFINALRELATIVE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Dbl *norm,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassDblRef (norm)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscalesetup, HYPRE_PARCSRDIAGSCALESETUP)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *A,
     hypre_F90_Obj *y,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRDiagScaleSetup(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
           hypre_F90_PassObj (HYPRE_ParVector, y),
           hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscale, HYPRE_PARCSRDIAGSCALE)
   ( hypre_F90_Obj *solver,
     hypre_F90_Obj *HA,
     hypre_F90_Obj *Hy,
     hypre_F90_Obj *Hx,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParCSRDiagScale(
           hypre_F90_PassObj (HYPRE_Solver, solver),
           hypre_F90_PassObj (HYPRE_ParCSRMatrix, HA),
           hypre_F90_PassObj (HYPRE_ParVector, Hy),
           hypre_F90_PassObj (HYPRE_ParVector, Hx)      ) );
}
