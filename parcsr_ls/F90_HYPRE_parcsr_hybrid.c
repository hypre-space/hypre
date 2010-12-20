/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
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
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridCreate( (HYPRE_Solver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybriddestroy, HYPRE_PARCSRHYBRIDDESTROY)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetup, HYPRE_PARCSRHYBRIDSETUP)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsolve, HYPRE_PARCSRHYBRIDSOLVE)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettol, HYPRE_PARCSRHYBRIDSETTOL)
               (hypre_F90_Obj *solver, double *tol, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetconvergenc, HYPRE_PARCSRHYBRIDSETCONVERGENC)
               (hypre_F90_Obj *solver, double *cf_tol, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetConvergenceTol( (HYPRE_Solver) *solver,
                                                       (double)       *cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetdscgmaxite, HYPRE_PARCSRHYBRIDSETDSCGMAXITE)
               (hypre_F90_Obj *solver, HYPRE_Int *dscg_max_its, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetDSCGMaxIter( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetpcgmaxiter, HYPRE_PARCSRHYBRIDSETPCGMAXITER)
               (hypre_F90_Obj *solver, HYPRE_Int *pcg_max_its, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPCGMaxIter( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetsolvertype, HYPRE_PARCSRHYBRIDSETSOLVERTYPE)
               (hypre_F90_Obj *solver, HYPRE_Int *solver_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetSolverType( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *solver_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetkdim, HYPRE_PARCSRHYBRIDSETKDIM)
               (hypre_F90_Obj *solver, HYPRE_Int *kdim, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetKDim( (HYPRE_Solver) *solver,
                                             (HYPRE_Int)          *kdim  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettwonorm, HYPRE_PARCSRHYBRIDSETTWONORM)
               (hypre_F90_Obj *solver, HYPRE_Int *two_norm, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetTwoNorm( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *two_norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstopcrit, HYPRE_PARCSRHYBRIDSETSTOPCRIT)
               (hypre_F90_Obj *solver, HYPRE_Int *stop_crit, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetStopCrit( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelchange, HYPRE_PARCSRHYBRIDSETRELCHANGE)
               (hypre_F90_Obj *solver, HYPRE_Int *rel_change, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetRelChange( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprecond, HYPRE_PARCSRHYBRIDSETPRECOND)
               (hypre_F90_Obj *solver, HYPRE_Int *precond_id, hypre_F90_Obj *precond_solver,  HYPRE_Int *ierr)
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
       *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_ParCSRDiagScale,
                                                     HYPRE_ParCSRDiagScaleSetup,
                                                     NULL                      ));
      }
   else if (*precond_id == 2)
      {
       *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_BoomerAMGSolve,
                                                     HYPRE_BoomerAMGSetup,
                                                     (void *)          *precond_solver ));
      }
   else if (*precond_id == 3)
      {
       *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_ParCSRPilutSolve,
                                                     HYPRE_ParCSRPilutSetup,
                                                     (void *)          *precond_solver ));
      }
   else if (*precond_id == 4)
      {
       *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_ParCSRParaSailsSolve,
                                                     HYPRE_ParCSRParaSailsSetup,
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
               (hypre_F90_Obj *solver, HYPRE_Int *logging, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetLogging( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *logging  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprintlevel, HYPRE_PARCSRHYBRIDSETPRINTLEVEL)
               (hypre_F90_Obj *solver, HYPRE_Int *print_level, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *print_level  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstrongthre, HYPRE_PARCSRHYBRIDSETSTRONGTHRE)
               (hypre_F90_Obj *solver, double *strong_threshold, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetStrongThreshold( (HYPRE_Solver) *solver,
                                                        (double)       *strong_threshold ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxrowsum, HYPRE_PARCSRHYBRIDSETMAXROWSUM)
               (hypre_F90_Obj *solver, double *max_row_sum, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetMaxRowSum( (HYPRE_Solver) *solver,
                                                  (double)       *max_row_sum   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettruncfacto, HYPRE_PARCSRHYBRIDSETTRUNCFACTO)
               (hypre_F90_Obj *solver, double *trunc_factor, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetTruncFactor( (HYPRE_Solver) *solver,
                                                    (double)       *trunc_factor ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxlevels, HYPRE_PARCSRHYBRIDSETMAXLEVELS)
               (hypre_F90_Obj *solver, HYPRE_Int *max_levels, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetMaxLevels( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *max_levels  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmeasuretyp, HYPRE_PARCSRHYBRIDSETMEASURETYP)
               (hypre_F90_Obj *solver, HYPRE_Int *measure_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetMeasureType( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *measure_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcoarsentyp, HYPRE_PARCSRHYBRIDSETCOARSENTYP)
               (hypre_F90_Obj *solver, HYPRE_Int *coarsen_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetCoarsenType( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *coarsen_type  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcycletype, HYPRE_PARCSRHYBRIDSETCYCLETYPE)
               (hypre_F90_Obj *solver, HYPRE_Int *cycle_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetCycleType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *cycle_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumgridswe, HYPRE_PARCSRHYBRIDSETNUMGRIDSWE)
               (hypre_F90_Obj *solver, HYPRE_Int *num_grid_sweeps, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetNumGridSweeps( (HYPRE_Solver) *solver,
                                                      (HYPRE_Int *)         num_grid_sweeps ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxt, HYPRE_PARCSRHYBRIDSETGRIDRELAXT)
               (hypre_F90_Obj *solver,  HYPRE_Int *grid_relax_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetGridRelaxType( (HYPRE_Solver) *solver,
                                                      (HYPRE_Int *)         grid_relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxp, HYPRE_PARCSRHYBRIDSETGRIDRELAXP)
               (hypre_F90_Obj *solver, HYPRE_Int *grid_relax_points, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetGridRelaxPoints( (HYPRE_Solver) *solver,
                                                        (HYPRE_Int **)        grid_relax_points  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumsweeps, HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
               (hypre_F90_Obj *solver, HYPRE_Int *num_sweeps, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetNumSweeps( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *num_sweeps  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclenumsw, HYPRE_PARCSRHYBRIDSETCYCLENUMSW)
               (hypre_F90_Obj *solver, HYPRE_Int *num_sweeps, HYPRE_Int *k, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetCycleNumSweeps( (HYPRE_Solver) *solver,
                                                       (HYPRE_Int)          *num_sweeps,
                                                       (HYPRE_Int)          *k ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxtype, HYPRE_PARCSRHYBRIDSETRELAXTYPE)
               (hypre_F90_Obj *solver, HYPRE_Int *relax_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetRelaxType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclerelax, HYPRE_PARCSRHYBRIDSETCYCLERELAX)
               (hypre_F90_Obj *solver, HYPRE_Int *relax_type, HYPRE_Int *k, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetCycleRelaxType( (HYPRE_Solver) *solver,
                                                       (HYPRE_Int)          *relax_type,
                                                       (HYPRE_Int)          *k ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxorder, HYPRE_PARCSRHYBRIDSETRELAXORDER)
               (hypre_F90_Obj *solver, HYPRE_Int *relax_order, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetRelaxOrder( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *relax_order ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxwt, HYPRE_PARCSRHYBRIDSETRELAXWT)
               (hypre_F90_Obj *solver, double *relax_wt, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetRelaxWt( (HYPRE_Solver) *solver,
                                                (double)       *relax_wt ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelrelax, HYPRE_PARCSRHYBRIDSETLEVELRELAX)
               (hypre_F90_Obj *solver, double *relax_wt, HYPRE_Int *level, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetLevelRelaxWt( (HYPRE_Solver) *solver,
                                                     (double)       *relax_wt,
                                                     (HYPRE_Int)          *level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetouterwt, HYPRE_PARCSRHYBRIDSETOUTERWT)
               (hypre_F90_Obj *solver, double *outer_wt, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetOuterWt( (HYPRE_Solver) *solver,
                                                (double)      *outer_wt ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelouter, HYPRE_PARCSRHYBRIDSETLEVELOUTER)
               (hypre_F90_Obj *solver, double *outer_wt, HYPRE_Int *level, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetLevelOuterWt( (HYPRE_Solver) *solver,
                                                     (double)       *outer_wt,
                                                     (HYPRE_Int)          *level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxweigh, HYPRE_PARCSRHYBRIDSETRELAXWEIGH)
               (hypre_F90_Obj *solver, double *relax_weight, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetRelaxWeight( (HYPRE_Solver) *solver,
                                                    (double *)      relax_weight ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetomega, HYPRE_PARCSRHYBRIDSETOMEGA)
               (hypre_F90_Obj *solver, double *omega, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridSetOmega( (HYPRE_Solver) *solver,
                                              (double *)  omega ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetnumiterati, HYPRE_PARCSRHYBRIDGETNUMITERATI)
               (hypre_F90_Obj *solver, HYPRE_Int *num_its, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridGetNumIterations( (HYPRE_Solver) *solver,
                                                      (HYPRE_Int *)         num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetdscgnumite, HYPRE_PARCSRHYBRIDGETDSCGNUMITE)
               (hypre_F90_Obj *solver, HYPRE_Int *dscg_num_its, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridGetDSCGNumIterations( (HYPRE_Solver) *solver,
                                                          (HYPRE_Int *)         dscg_num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetpcgnumiter, HYPRE_PARCSRHYBRIDGETPCGNUMITER)
               (hypre_F90_Obj *solver, HYPRE_Int *pcg_num_its, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridGetPCGNumIterations( (HYPRE_Solver) *solver,
                                                         (HYPRE_Int *)         pcg_num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetfinalrelat, HYPRE_PARCSRHYBRIDGETFINALRELAT)
               (hypre_F90_Obj *solver, double *norm, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                                  (double *)      norm ));
}
