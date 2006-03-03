/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
               (long int *solver, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridCreate( (HYPRE_Solver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybriddestroy, HYPRE_PARCSRHYBRIDDESTROY)
               (long int *solver, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetup, HYPRE_PARCSRHYBRIDSETUP)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsolve, HYPRE_PARCSRHYBRIDSOLVE)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettol, HYPRE_PARCSRHYBRIDSETTOL)
               (long int *solver, double *tol, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetconvergenc, HYPRE_PARCSRHYBRIDSETCONVERGENC)
               (long int *solver, double *cf_tol, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetConvergenceTol( (HYPRE_Solver) *solver,
                                                       (double)       *cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetdscgmaxite, HYPRE_PARCSRHYBRIDSETDSCGMAXITE)
               (long int *solver, int *dscg_max_its, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetDSCGMaxIter( (HYPRE_Solver) *solver,
                                                    (int)          *dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetpcgmaxiter, HYPRE_PARCSRHYBRIDSETPCGMAXITER)
               (long int *solver, int *pcg_max_its, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetPCGMaxIter( (HYPRE_Solver) *solver,
                                                   (int)          *pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetsolvertype, HYPRE_PARCSRHYBRIDSETSOLVERTYPE)
               (long int *solver, int *solver_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetSolverType( (HYPRE_Solver) *solver,
                                                   (int)          *solver_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetkdim, HYPRE_PARCSRHYBRIDSETKDIM)
               (long int *solver, int *kdim, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetKDim( (HYPRE_Solver) *solver,
                                             (int)          *kdim  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettwonorm, HYPRE_PARCSRHYBRIDSETTWONORM)
               (long int *solver, int *two_norm, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetTwoNorm( (HYPRE_Solver) *solver,
                                                (int)          *two_norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstopcrit, HYPRE_PARCSRHYBRIDSETSTOPCRIT)
               (long int *solver, int *stop_crit, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetStopCrit( (HYPRE_Solver) *solver,
                                                 (int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelchange, HYPRE_PARCSRHYBRIDSETRELCHANGE)
               (long int *solver, int *rel_change, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetRelChange( (HYPRE_Solver) *solver,
                                                  (int)          *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprecond, HYPRE_PARCSRHYBRIDSETPRECOND)
               (long int *solver, int *precond_id, long int *precond_solver,  int *ierr)
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
       *ierr = (int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_ParCSRDiagScale,
                                                     HYPRE_ParCSRDiagScaleSetup,
                                                     NULL                      ));
      }
   else if (*precond_id == 2)
      {
       *ierr = (int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_BoomerAMGSolve,
                                                     HYPRE_BoomerAMGSetup,
                                                     (void *)          *precond_solver ));
      }
   else if (*precond_id == 3)
      {
       *ierr = (int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
                                                     HYPRE_ParCSRPilutSolve,
                                                     HYPRE_ParCSRPilutSetup,
                                                     (void *)          *precond_solver ));
      }
   else if (*precond_id == 4)
      {
       *ierr = (int) (HYPRE_ParCSRHybridSetPrecond( (HYPRE_Solver) *solver,
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
               (long int *solver, int *logging, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetLogging( (HYPRE_Solver) *solver,
                                                (int)          *logging  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetprintlevel, HYPRE_PARCSRHYBRIDSETPRINTLEVEL)
               (long int *solver, int *print_level, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (int)          *print_level  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetstrongthre, HYPRE_PARCSRHYBRIDSETSTRONGTHRE)
               (long int *solver, double *strong_threshold, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetStrongThreshold( (HYPRE_Solver) *solver,
                                                        (double)       *strong_threshold ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxrowsum, HYPRE_PARCSRHYBRIDSETMAXROWSUM)
               (long int *solver, double *max_row_sum, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetMaxRowSum( (HYPRE_Solver) *solver,
                                                  (double)       *max_row_sum   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsettruncfacto, HYPRE_PARCSRHYBRIDSETTRUNCFACTO)
               (long int *solver, double *trunc_factor, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetTruncFactor( (HYPRE_Solver) *solver,
                                                    (double)       *trunc_factor ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmaxlevels, HYPRE_PARCSRHYBRIDSETMAXLEVELS)
               (long int *solver, int *max_levels, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetMaxLevels( (HYPRE_Solver) *solver,
                                                  (int)          *max_levels  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetmeasuretyp, HYPRE_PARCSRHYBRIDSETMEASURETYP)
               (long int *solver, int *measure_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetMeasureType( (HYPRE_Solver) *solver,
                                                    (int)          *measure_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcoarsentyp, HYPRE_PARCSRHYBRIDSETCOARSENTYP)
               (long int *solver, int *coarsen_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetCoarsenType( (HYPRE_Solver) *solver,
                                                    (int)          *coarsen_type  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcycletype, HYPRE_PARCSRHYBRIDSETCYCLETYPE)
               (long int *solver, int *cycle_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetCycleType( (HYPRE_Solver) *solver,
                                                  (int)          *cycle_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumgridswe, HYPRE_PARCSRHYBRIDSETNUMGRIDSWE)
               (long int *solver, int *num_grid_sweeps, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetNumGridSweeps( (HYPRE_Solver) *solver,
                                                      (int *)         num_grid_sweeps ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxt, HYPRE_PARCSRHYBRIDSETGRIDRELAXT)
               (long int *solver,  int *grid_relax_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetGridRelaxType( (HYPRE_Solver) *solver,
                                                      (int *)         grid_relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetgridrelaxp, HYPRE_PARCSRHYBRIDSETGRIDRELAXP)
               (long int *solver, int *grid_relax_points, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetGridRelaxPoints( (HYPRE_Solver) *solver,
                                                        (int **)        grid_relax_points  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetnumsweeps, HYPRE_PARCSRHYBRIDSETNUMSWEEPS)
               (long int *solver, int *num_sweeps, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetNumSweeps( (HYPRE_Solver) *solver,
                                                  (int)          *num_sweeps  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclenumsw, HYPRE_PARCSRHYBRIDSETCYCLENUMSW)
               (long int *solver, int *num_sweeps, int *k, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetCycleNumSweeps( (HYPRE_Solver) *solver,
                                                       (int)          *num_sweeps,
                                                       (int)          *k ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxtype, HYPRE_PARCSRHYBRIDSETRELAXTYPE)
               (long int *solver, int *relax_type, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetRelaxType( (HYPRE_Solver) *solver,
                                                  (int)          *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetcyclerelax, HYPRE_PARCSRHYBRIDSETCYCLERELAX)
               (long int *solver, int *relax_type, int *k, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetCycleRelaxType( (HYPRE_Solver) *solver,
                                                       (int)          *relax_type,
                                                       (int)          *k ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxorder, HYPRE_PARCSRHYBRIDSETRELAXORDER)
               (long int *solver, int *relax_order, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetRelaxOrder( (HYPRE_Solver) *solver,
                                                   (int)          *relax_order ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxwt, HYPRE_PARCSRHYBRIDSETRELAXWT)
               (long int *solver, double *relax_wt, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetRelaxWt( (HYPRE_Solver) *solver,
                                                (double)       *relax_wt ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelrelax, HYPRE_PARCSRHYBRIDSETLEVELRELAX)
               (long int *solver, double *relax_wt, int *level, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetLevelRelaxWt( (HYPRE_Solver) *solver,
                                                     (double)       *relax_wt,
                                                     (int)          *level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetouterwt, HYPRE_PARCSRHYBRIDSETOUTERWT)
               (long int *solver, double *outer_wt, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetOuterWt( (HYPRE_Solver) *solver,
                                                (double)      *outer_wt ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetlevelouter, HYPRE_PARCSRHYBRIDSETLEVELOUTER)
               (long int *solver, double *outer_wt, int *level, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetLevelOuterWt( (HYPRE_Solver) *solver,
                                                     (double)       *outer_wt,
                                                     (int)          *level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetrelaxweigh, HYPRE_PARCSRHYBRIDSETRELAXWEIGH)
               (long int *solver, double *relax_weight, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetRelaxWeight( (HYPRE_Solver) *solver,
                                                    (double *)      relax_weight ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridsetomega, HYPRE_PARCSRHYBRIDSETOMEGA)
               (long int *solver, double *omega, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridSetOmega( (HYPRE_Solver) *solver,
                                              (double *)  omega ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetnumiterati, HYPRE_PARCSRHYBRIDGETNUMITERATI)
               (long int *solver, int *num_its, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridGetNumIterations( (HYPRE_Solver) *solver,
                                                      (int *)         num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetdscgnumite, HYPRE_PARCSRHYBRIDGETDSCGNUMITE)
               (long int *solver, int *dscg_num_its, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridGetDSCGNumIterations( (HYPRE_Solver) *solver,
                                                          (int *)         dscg_num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetpcgnumiter, HYPRE_PARCSRHYBRIDGETPCGNUMITER)
               (long int *solver, int *pcg_num_its, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridGetPCGNumIterations( (HYPRE_Solver) *solver,
                                                         (int *)         pcg_num_its ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrhybridgetfinalrelat, HYPRE_PARCSRHYBRIDGETFINALRELAT)
               (long int *solver, double *norm, int *ierr)
{
   *ierr = (int) (HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                                  (double *)      norm ));
}
