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
 * HYPRE_ParAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgcreate)( long int *solver,
                                         int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParAMGCreate( (HYPRE_Solver *) solver) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paramgdestroy)( long int *solver,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParAMGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paramgsetup)( long int *solver,
                                    long int *A,
                                    long int *b,
                                    long int *x,
                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParAMGSetup( (HYPRE_Solver)       *solver,
                                      (HYPRE_ParCSRMatrix) *A,
                                      (HYPRE_ParVector)    *b,
                                      (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paramgsolve)( long int *solver,
                                    long int *A,
                                    long int *b,
                                    long int *x,
                                    int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParAMGSolve( (HYPRE_Solver)       *solver,
                                      (HYPRE_ParCSRMatrix) *A,
                                      (HYPRE_ParVector)    *b,
                                      (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSolveT
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paramgsolvet)( long int *solver,
                                     long int *A,
                                     long int *b,
                                     long int *x,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParAMGSolveT( (HYPRE_Solver)       *solver,
                                       (HYPRE_ParCSRMatrix) *A,
                                       (HYPRE_ParVector)    *b,
                                       (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetrestriction)( long int *solver,
                                             int      *restr_par,
                                             int      *ierr       )
{
   *ierr = (int) ( HYPRE_ParAMGSetRestriction( (HYPRE_Solver) *solver,
                                               (int)          *restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetmaxlevels)( long int *solver,
                                           int      *max_levels,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_ParAMGSetMaxLevels( (HYPRE_Solver) *solver,
                                             (int)          *max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetstrongthreshold)( long int *solver,
                                                 double   *strong_threshold,
                                                 int      *ierr              )
{
   *ierr = (int)
           ( HYPRE_ParAMGSetStrongThreshold( (HYPRE_Solver) *solver,
                                             (double)       *strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsettruncfactor)( long int *solver,
                                             double   *trunc_factor,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_ParAMGSetTruncFactor( (HYPRE_Solver) *solver,
                                               (double)       *trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetinterptype)( long int *solver,
                                            int      *interp_type,
                                            int      *ierr         )
{
   *ierr = (int) ( HYPRE_ParAMGSetInterpType( (HYPRE_Solver) *solver,
                                              (int)          *interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetmaxiter)( long int *solver,
                                         int      *max_iter,
                                         int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParAMGSetMaxIter( (HYPRE_Solver) *solver,
                                           (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetcoarsentype)( long int *solver,
                                             int      *coarsen_type,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_ParAMGSetCoarsenType( (HYPRE_Solver) *solver,
                                               (int)          *coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetmeasuretype)( long int *solver,
                                             int      *measure_type,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_ParAMGSetMeasureType( (HYPRE_Solver) *solver,
                                               (int)          *measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetcycletype)( long int *solver,
                                           int      *cycle_type,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_ParAMGSetCycleType( (HYPRE_Solver) *solver,
                                             (int)          *cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsettol)( long int *solver,
                                     double   *tol,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParAMGSetTol( (HYPRE_Solver) *solver,
                                       (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetNumGridSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetnumgridsweeps)( long int *solver,
                                               long int *num_grid_sweeps,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_ParAMGSetNumGridSweeps(
                        (HYPRE_Solver) *solver,
                        (int *)        *((int **)(*num_grid_sweeps)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramginitgridrelaxation)( long int *num_grid_sweeps,
                                                 long int *grid_relax_type,
                                                 long int *grid_relax_points,
                                                 int      *coarsen_type,
                                                 long int *relax_weights,
                                                 int      *max_levels,
                                                 int      *ierr               )
{
   *num_grid_sweeps   = (long int) hypre_CTAlloc(int**, 1);
   *grid_relax_type   = (long int) hypre_CTAlloc(int**, 1);
   *grid_relax_points = (long int) hypre_CTAlloc(int***, 1);
   *relax_weights     = (long int) hypre_CTAlloc(double**, 1);

   *ierr = (int) ( HYPRE_ParAMGInitGridRelaxation( (int **)     *num_grid_sweeps,
                                                   (int **)     *grid_relax_type,
                                                   (int ***)    *grid_relax_points,
                                                   (int)        *coarsen_type,
                                                   (double **)  *relax_weights,
                                                   (int)        *max_levels         ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetGridRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetgridrelaxtype)( long int *solver,
                                               long int *grid_relax_type,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_ParAMGSetGridRelaxType(
                       (HYPRE_Solver) *solver,
                       (int *)        *((int **)(*grid_relax_type)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetgridrelaxpoints)( long int *solver,
                                                 long int *grid_relax_points,
                                                 int      *ierr               )
{
   *ierr = (int) ( HYPRE_ParAMGSetGridRelaxPoints(
                       (HYPRE_Solver) *solver,
                       (int **)       *((int ***)(*grid_relax_points)) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetrelaxweight)( long int *solver,
                                             long int *relax_weights,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGSetRelaxWeight(
                       (HYPRE_Solver) *solver,
                       (double *)     *((double **)(*relax_weights)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetIOutDat
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetioutdat)( long int *solver,
                                         int      *ioutdat,
                                         int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGSetIOutDat( (HYPRE_Solver) *solver,
                                           (int)          *ioutdat ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetLogFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetlogfilename)( long int *solver,
                                             char     *log_file_name,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGSetLogFileName( (HYPRE_Solver) *solver,
                                               (char *)       log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetlogging)( long int *solver,
                                         int      *ioutdat,
                                         char     *log_file_name,
                                         int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGSetLogging( (HYPRE_Solver) *solver,
                                           (int)          *ioutdat,
                                           (char *)       log_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGSetDebugFlag
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramgsetdebugflag)( long int *solver,
                                           int      *debug_flag,
                                           int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGSetDebugFlag( (HYPRE_Solver) *solver,
                                             (int)          *debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramggetnumiterations)( long int *solver,
                                               int      *num_iterations,
                                               int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParAMGGetNumIterations( (HYPRE_Solver) *solver,
                                                 (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetFinalRelativeRes
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paramggetfinalrelativeres)( long int *solver,
                                                  double   *rel_resid_norm,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParAMGGetFinalRelativeResidualNorm(
                                (HYPRE_Solver) *solver,
                                (double *)      rel_resid_norm ) );
}
