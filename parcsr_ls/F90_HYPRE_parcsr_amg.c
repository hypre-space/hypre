/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParAMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgcreate, HYPRE_BOOMERAMGCREATE)( long int *solver,
                                         int      *ierr    )

{
   *ierr = (int) ( HYPRE_BoomerAMGCreate( (HYPRE_Solver *) solver) );

}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgdestroy, HYPRE_BOOMERAMGDESTROY)( long int *solver,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsetup, HYPRE_BOOMERAMGSETUP)( long int *solver,
                                    long int *A,
                                    long int *b,
                                    long int *x,
                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetup( (HYPRE_Solver)       *solver,
                                      (HYPRE_ParCSRMatrix) *A,
                                      (HYPRE_ParVector)    *b,
                                      (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsolve, HYPRE_BOOMERAMGSOLVE)( long int *solver,
                                    long int *A,
                                    long int *b,
                                    long int *x,
                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGSolve( (HYPRE_Solver)       *solver,
                                      (HYPRE_ParCSRMatrix) *A,
                                      (HYPRE_ParVector)    *b,
                                      (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsolvet, HYPRE_BOOMERAMGSOLVET)( long int *solver,
                                     long int *A,
                                     long int *b,
                                     long int *x,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGSolveT( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrestriction, HYPRE_BOOMERAMGSETRESTRICTION)( long int *solver,
                                             int      *restr_par,
                                             int      *ierr       )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetRestriction( (HYPRE_Solver) *solver,
                                               (int)          *restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxlevels, HYPRE_BOOMERAMGSETMAXLEVELS)( long int *solver,
                                           int      *max_levels,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMaxLevels( (HYPRE_Solver) *solver,
                                             (int)          *max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetstrongthrshld, HYPRE_BOOMERAMGSETSTRONGTHRSHLD)( long int *solver,
                                                 double   *strong_threshold,
                                                 int      *ierr              )
{
   *ierr = (int)
           ( HYPRE_BoomerAMGSetStrongThreshold( (HYPRE_Solver) *solver,
                                             (double)       *strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxrowsum, HYPRE_BOOMERAMGSETMAXROWSUM)( long int *solver,
                                           double   *max_row_sum,
                                           int      *ierr              )
{
   *ierr = (int)
           ( HYPRE_BoomerAMGSetMaxRowSum( (HYPRE_Solver) *solver,
                                       (double)       *max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettruncfactor, HYPRE_BOOMERAMGSETTRUNCFACTOR)( long int *solver,
                                             double   *trunc_factor,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetTruncFactor( (HYPRE_Solver) *solver,
                                               (double)       *trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterptype, HYPRE_BOOMERAMGSETINTERPTYPE)( long int *solver,
                                            int      *interp_type,
                                            int      *ierr         )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetInterpType( (HYPRE_Solver) *solver,
                                              (int)          *interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetminiter, HYPRE_BOOMERAMGSETMINITER)( long int *solver,
                                         int      *min_iter,
                                         int      *ierr      )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMinIter( (HYPRE_Solver) *solver,
                                           (int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxiter, HYPRE_BOOMERAMGSETMAXITER)( long int *solver,
                                         int      *max_iter,
                                         int      *ierr      )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMaxIter( (HYPRE_Solver) *solver,
                                           (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcoarsentype, HYPRE_BOOMERAMGSETCOARSENTYPE)( long int *solver,
                                             int      *coarsen_type,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetCoarsenType( (HYPRE_Solver) *solver,
                                               (int)          *coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmeasuretype, HYPRE_BOOMERAMGSETMEASURETYPE)( long int *solver,
                                                int      *measure_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMeasureType( (HYPRE_Solver) *solver,
                                                  (int)          *measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsetuptype, HYPRE_BOOMERAMGSETSETUPTYPE)( long int *solver,
                                              int      *setup_type,
                                              int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSetupType( (HYPRE_Solver) *solver,
                                                (int)          *setup_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcycletype, HYPRE_BOOMERAMGSETCYCLETYPE)( long int *solver,
                                           int      *cycle_type,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetCycleType( (HYPRE_Solver) *solver,
                                             (int)          *cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettol, HYPRE_BOOMERAMGSETTOL)( long int *solver,
                                     double   *tol,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetTol( (HYPRE_Solver) *solver,
                                       (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumgridsweeps, HYPRE_BOOMERAMGSETNUMGRIDSWEEPS)( long int *solver,
                                               long int *num_grid_sweeps,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNumGridSweeps(
                        (HYPRE_Solver) *solver,
                        (int *)        *((int **)(*num_grid_sweeps)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramginitgridrelaxatn, HYPRE_BOOMERAMGINITGRIDRELAXATN)( long int *num_grid_sweeps,
                                                 long int *grid_relax_type,
                                                 long int *grid_relax_points,
                                                 int      *coarsen_type,
                                                 long int *relax_weights,
                                                 int      *max_levels,
                                                 int      *ierr               )
{
   *num_grid_sweeps   = (long int) hypre_CTAlloc(int*, 1);
   *grid_relax_type   = (long int) hypre_CTAlloc(int*, 1);
   *grid_relax_points = (long int) hypre_CTAlloc(int**, 1);
   *relax_weights     = (long int) hypre_CTAlloc(double*, 1);

   *ierr = (int) ( HYPRE_BoomerAMGInitGridRelaxation( (int **)    *num_grid_sweeps,
                                                   (int **)    *grid_relax_type,
                                                   (int ***)   *grid_relax_points,
                                                   (int)       *coarsen_type,
                                                   (double **) *relax_weights,
                                                   (int)       *max_levels         ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGFinalizeGridRelaxation
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgfingridrelaxatn,
                HYPRE_BOOMERAMGFINGRIDRELAXATN)( long int *num_grid_sweeps,
                                                 long int *grid_relax_type,
                                                 long int *grid_relax_points,
                                                 long int *relax_weights,
                                                 int      *ierr               )
{
   hypre_TFree(*num_grid_sweeps);
   hypre_TFree(*grid_relax_type);
   hypre_TFree(*grid_relax_points);
   hypre_TFree(*relax_weights);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxtype, HYPRE_BOOMERAMGSETGRIDRELAXTYPE)( long int *solver,
                                               long int *grid_relax_type,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetGridRelaxType(
                       (HYPRE_Solver) *solver,
                       (int *)        *((int **)(*grid_relax_type)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxpnts, HYPRE_BOOMERAMGSETGRIDRELAXPNTS)( long int *solver,
                                                 long int *grid_relax_points,
                                                 int      *ierr               )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetGridRelaxPoints(
                       (HYPRE_Solver) *solver,
                       (int **)       *((int ***)(*grid_relax_points)) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxweight, HYPRE_BOOMERAMGSETRELAXWEIGHT)( long int *solver,
                                             long int *relax_weights,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetRelaxWeight(
                       (HYPRE_Solver) *solver,
                       (double *)     *((double **)(*relax_weights)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetIOutDat
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetioutdat, HYPRE_BOOMERAMGSETIOUTDAT)( long int *solver,
                                         int      *ioutdat,
                                         int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetIOutDat( (HYPRE_Solver) *solver,
                                           (int)          *ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogfilename, HYPRE_BOOMERAMGSETLOGFILENAME)( long int *solver,
                                             char     *log_file_name,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLogFileName( (HYPRE_Solver) *solver,
                                               (char *)       log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogging, HYPRE_BOOMERAMGSETLOGGING)( long int *solver,
                                         int      *ioutdat,
                                         char     *log_file_name,
                                         int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLogging( (HYPRE_Solver) *solver,
                                           (int)          *ioutdat,
                                           (char *)       log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdebugflag, HYPRE_BOOMERAMGSETDEBUGFLAG)( long int *solver,
                                           int      *debug_flag,
                                           int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetDebugFlag( (HYPRE_Solver) *solver,
                                             (int)          *debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetnumiterations, HYPRE_BOOMERAMGGETNUMITERATIONS)( long int *solver,
                                               int      *num_iterations,
                                               int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetNumIterations( (HYPRE_Solver) *solver,
                                                 (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetFinalRelativeRes
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetfinalreltvres, HYPRE_BOOMERAMGGETFINALRELTVRES)( long int *solver,
                                                  double   *rel_resid_norm,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetFinalRelativeResidualNorm(
                                (HYPRE_Solver) *solver,
                                (double *)      rel_resid_norm ) );
}
