/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRHybrid Fortran Interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *    HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridcreate, HYPRE_PARCSRHYBRIDCREATE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridCreate(
          hypre_F90_PassObjRef (HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybriddestroy, HYPRE_PARCSRHYBRIDDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridDestroy(
          hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetup, HYPRE_PARCSRHYBRIDSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetup(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
          hypre_F90_PassObj (HYPRE_ParVector, b),
          hypre_F90_PassObj (HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsolve, HYPRE_PARCSRHYBRIDSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSolve(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
          hypre_F90_PassObj (HYPRE_ParVector, b),
          hypre_F90_PassObj (HYPRE_ParVector, x)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettol, HYPRE_PARCSRHYBRIDSETTOL)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetTol(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetconvergenc, HYPRE_PARCSRHYBRIDSETCONVERGENC)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *cf_tol,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetConvergenceTol(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (cf_tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetdscgmaxite, HYPRE_PARCSRHYBRIDSETDSCGMAXITE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *dscg_max_its,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetDSCGMaxIter(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetpcgmaxiter, HYPRE_PARCSRHYBRIDSETPCGMAXITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *pcg_max_its,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetPCGMaxIter(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (pcg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetsolvertype, HYPRE_PARCSRHYBRIDSETSOLVERTYPE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *solver_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetSolverType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetkdim, HYPRE_PARCSRHYBRIDSETKDIM)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *kdim,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetKDim(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (kdim)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettwonorm, HYPRE_PARCSRHYBRIDSETTWONORM)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *two_norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetTwoNorm(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstopcrit, HYPRE_PARCSRHYBRIDSETSTOPCRIT)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *stop_crit,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetStopCrit(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (stop_crit) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelchange, HYPRE_PARCSRHYBRIDSETRELCHANGE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *rel_change,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetRelChange(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprecond, HYPRE_PARCSRHYBRIDSETPRECOND)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *precond_id,
    hypre_F90_Obj *precond_solver,
    hypre_F90_Int *ierr)
{
   /*----------------------------------------------------------------
    * precond_id definitions
    * 0 - no preconditioner
    * 1 - use diagscale preconditioner
    * 2 - use amg preconditioner
    * 3 - use pilut preconditioner
    * 4 - use parasails preconditioner
    *---------------------------------------------------------------*/
   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_ParCSRHybridSetPrecond(
             hypre_F90_PassObj (HYPRE_Solver, solver),
             HYPRE_ParCSRDiagScale,
             HYPRE_ParCSRDiagScaleSetup,
             NULL                      ));
   }
   else if (*precond_id == 2)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_ParCSRHybridSetPrecond(
             hypre_F90_PassObj (HYPRE_Solver, solver),
             HYPRE_BoomerAMGSolve,
             HYPRE_BoomerAMGSetup,
             (void *)          *precond_solver ));
   }
   else if (*precond_id == 3)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_ParCSRHybridSetPrecond(
             hypre_F90_PassObj (HYPRE_Solver, solver),
             HYPRE_ParCSRPilutSolve,
             HYPRE_ParCSRPilutSetup,
             (void *)          *precond_solver ));
   }
   else if (*precond_id == 4)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_ParCSRHybridSetPrecond(
             hypre_F90_PassObj (HYPRE_Solver, solver),
             HYPRE_ParCSRParaSailsSolve,
             HYPRE_ParCSRParaSailsSetup,
             (void *)          *precond_solver ));
   }
   else if (*precond_id == 5)
   {
      *ierr = (hypre_F90_Int)
         (HYPRE_ParCSRHybridSetPrecond(
             hypre_F90_PassObj (HYPRE_Solver, solver),
             HYPRE_EuclidSolve,
             HYPRE_EuclidSetup,
             (void *)          *precond_solver ));
   }
   else
   {
      *ierr = -1;
   }    
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlogging, HYPRE_PARCSRHYBRIDSETLOGGING)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *logging,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetLogging(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (logging)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprintlevel, HYPRE_PARCSRHYBRIDSETPRINTLEVEL)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetPrintLevel(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (print_level)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstrongthre, HYPRE_PARCSRHYBRIDSETSTRONGTHRE)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *strong_threshold,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetStrongThreshold(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (strong_threshold) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxrowsum, HYPRE_PARCSRHYBRIDSETMAXROWSUM)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *max_row_sum,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetMaxRowSum(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (max_row_sum)   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettruncfacto, HYPRE_PARCSRHYBRIDSETTRUNCFACTO)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *trunc_factor,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetTruncFactor(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (trunc_factor) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxlevels, HYPRE_PARCSRHYBRIDSETMAXLEVELS)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *max_levels,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetMaxLevels(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (max_levels)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmeasuretyp, HYPRE_PARCSRHYBRIDSETMEASURETYP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *measure_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetMeasureType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (measure_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcoarsentyp, HYPRE_PARCSRHYBRIDSETCOARSENTYP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *coarsen_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetCoarsenType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (coarsen_type)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcycletype, HYPRE_PARCSRHYBRIDSETCYCLETYPE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *cycle_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetCycleType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (cycle_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumgridswe, HYPRE_PARCSRHYBRIDSETNUMGRIDSWE)
   (hypre_F90_Obj *solver,
    hypre_F90_IntArray *num_grid_sweeps,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetNumGridSweeps(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassIntArray (num_grid_sweeps) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxt, HYPRE_PARCSRHYBRIDSETGRIDRELAXT)
   (hypre_F90_Obj *solver,
    hypre_F90_IntArray *grid_relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetGridRelaxType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassIntArray (grid_relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxp, HYPRE_PARCSRHYBRIDSETGRIDRELAXP)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *grid_relax_points,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetGridRelaxPoints(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          (HYPRE_Int **)        grid_relax_points  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumsweeps, HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_sweeps,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetNumSweeps(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (num_sweeps)  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclenumsw, HYPRE_PARCSRHYBRIDSETCYCLENUMSW)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_sweeps,
    hypre_F90_Int *k,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetCycleNumSweeps(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (num_sweeps),
          hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxtype, HYPRE_PARCSRHYBRIDSETRELAXTYPE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetRelaxType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclerelax, HYPRE_PARCSRHYBRIDSETCYCLERELAX)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *k,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetCycleRelaxType(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (relax_type),
          hypre_F90_PassInt (k) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxorder, HYPRE_PARCSRHYBRIDSETRELAXORDER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_order,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetRelaxOrder(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassInt (relax_order) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxwt, HYPRE_PARCSRHYBRIDSETRELAXWT)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *relax_wt,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetRelaxWt(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (relax_wt) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelrelax, HYPRE_PARCSRHYBRIDSETLEVELRELAX)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *relax_wt,
    hypre_F90_Int *level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetLevelRelaxWt(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (relax_wt),
          hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetouterwt, HYPRE_PARCSRHYBRIDSETOUTERWT)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *outer_wt,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetOuterWt(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (outer_wt) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelouter, HYPRE_PARCSRHYBRIDSETLEVELOUTER)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *outer_wt,
    hypre_F90_Int *level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetLevelOuterWt(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDbl (outer_wt),
          hypre_F90_PassInt (level) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxweigh, HYPRE_PARCSRHYBRIDSETRELAXWEIGH)
   (hypre_F90_Obj *solver,
    hypre_F90_DblArray *relax_weight,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetRelaxWeight(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDblArray (relax_weight) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetomega, HYPRE_PARCSRHYBRIDSETOMEGA)
   (hypre_F90_Obj *solver,
    hypre_F90_DblArray *omega,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridSetOmega(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDblArray (omega) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetnumiterati, HYPRE_PARCSRHYBRIDGETNUMITERATI)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_its,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridGetNumIterations(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassIntRef (num_its) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetdscgnumite, HYPRE_PARCSRHYBRIDGETDSCGNUMITE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *dscg_num_its,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridGetDSCGNumIterations(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassIntRef (dscg_num_its) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetpcgnumiter, HYPRE_PARCSRHYBRIDGETPCGNUMITER)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *pcg_num_its,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridGetPCGNumIterations(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassIntRef (pcg_num_its) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetfinalrelat, HYPRE_PARCSRHYBRIDGETFINALRELAT)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *norm,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(
          hypre_F90_PassObj (HYPRE_Solver, solver),
          hypre_F90_PassDblRef (norm) ));
}
