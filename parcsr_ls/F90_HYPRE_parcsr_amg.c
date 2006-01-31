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
 * HYPRE_BoomerAMGSetMaxLevels, HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxlevels, HYPRE_BOOMERAMGSETMAXLEVELS)( long int *solver,
                                           int      *max_levels,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMaxLevels( (HYPRE_Solver) *solver,
                                             (int)          *max_levels ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxlevels, HYPRE_BOOMERAMGGETMAXLEVELS)( long int *solver,
                                           int      *max_levels,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetMaxLevels( (HYPRE_Solver) *solver,
                                             (int *)          max_levels ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
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


void
hypre_F90_IFACE(hypre_boomeramggetstrongthrshld, HYPRE_BOOMERAMGGETSTRONGTHRSHLD)( long int *solver,
                                                 double   *strong_threshold,
                                                 int      *ierr              )
{
   *ierr = (int)
           ( HYPRE_BoomerAMGGetStrongThreshold( (HYPRE_Solver) *solver,
                                             (double *)       strong_threshold ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
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

void
hypre_F90_IFACE(hypre_boomeramggetmaxrowsum, HYPRE_BOOMERAMGGETMAXROWSUM)( long int *solver,
                                           double   *max_row_sum,
                                           int      *ierr              )
{
   *ierr = (int)
           ( HYPRE_BoomerAMGGetMaxRowSum( (HYPRE_Solver) *solver,
                                       (double *)       max_row_sum ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettruncfactor, HYPRE_BOOMERAMGSETTRUNCFACTOR)( long int *solver,
                                             double   *trunc_factor,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetTruncFactor( (HYPRE_Solver) *solver,
                                               (double)       *trunc_factor ) );
}


void
hypre_F90_IFACE(hypre_boomeramggettruncfactor, HYPRE_BOOMERAMGGETTRUNCFACTOR)( long int *solver,
                                             double   *trunc_factor,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetTruncFactor( (HYPRE_Solver) *solver,
                                               (double *)       trunc_factor ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSCommPkgSwitch
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetscommpkgswitch, HYPRE_BOOMERAMGSETSCOMMPKGSWITCH)
                                            ( long int *solver,
                                            double      *S_commpkg_switch,
                                            int         *ierr         )


{
   *ierr = (int) ( HYPRE_BoomerAMGSetSCommPkgSwitch( (HYPRE_Solver) *solver,
                                             (double) *S_commpkg_switch ) );
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

void
hypre_F90_IFACE(hypre_boomeramggetmaxiter, HYPRE_BOOMERAMGGETMAXITER)( long int *solver,
                                         int      *max_iter,
                                         int      *ierr      )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetMaxIter( (HYPRE_Solver) *solver,
                                           (int *)          max_iter ) );
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

void
hypre_F90_IFACE(hypre_boomeramggetcoarsentype, HYPRE_BOOMERAMGGETCOARSENTYPE)( long int *solver,
                                             int      *coarsen_type,
                                             int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetCoarsenType( (HYPRE_Solver) *solver,
                                               (int *)          coarsen_type ) );
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


void
hypre_F90_IFACE(hypre_boomeramggetmeasuretype, HYPRE_BOOMERAMGGETMEASURETYPE)( long int *solver,
                                                int      *measure_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetMeasureType( (HYPRE_Solver) *solver,
                                                  (int *)          measure_type ) );
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


void
hypre_F90_IFACE(hypre_boomeramggetcycletype, HYPRE_BOOMERAMGGETCYCLETYPE)( long int *solver,
                                           int      *cycle_type,
                                           int      *ierr        )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetCycleType( (HYPRE_Solver) *solver,
                                             (int *)          cycle_type ) );
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



void
hypre_F90_IFACE(hypre_boomeramggettol, HYPRE_BOOMERAMGGETTOL)( long int *solver,
                                     double   *tol,
                                     int      *ierr    )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetTol( (HYPRE_Solver) *solver,
                                       (double *)       tol     ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 * DEPRECATED.  Use SetNumSweeps and SetCycleNumSweeps instead.
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
 * HYPRE_BoomerAMGSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsweeps, HYPRE_BOOMERAMGSETNUMSWEEPS)( long int *solver,
                                               int *num_sweeps,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNumSweeps(
                        (HYPRE_Solver) *solver,
                        (int)        *num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclenumsweeps, HYPRE_BOOMERAMGSETCYCLENUMSWEEPS)( long int *solver,
                                               int *num_sweeps,
                                               int *k,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetCycleNumSweeps(
                        (HYPRE_Solver) *solver,
                        (int)        *num_sweeps,
                        (int)        *k ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetcyclenumsweeps, HYPRE_BOOMERAMGGETCYCLENUMSWEEPS)( long int *solver,
                                               int *num_sweeps,
                                               int *k,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetCycleNumSweeps(
                        (HYPRE_Solver) *solver,
                        (int *)        num_sweeps,
                        (int)        *k ) );
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
   hypre_TFree(num_grid_sweeps);
   hypre_TFree(grid_relax_type);
   hypre_TFree(grid_relax_points);
   hypre_TFree(relax_weights);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  Use SetRelaxType and SetCycleRelaxType instead.
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
 * HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxtype, HYPRE_BOOMERAMGSETRELAXTYPE)( long int *solver,
                                               int   *relax_type,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetRelaxType(
                       (HYPRE_Solver) *solver,
                       (int)        *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclerelaxtype, HYPRE_BOOMERAMGSETCYCLERELAXTYPE)( long int *solver,
                                               int   *relax_type,
                                               int   *k,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetCycleRelaxType(
                       (HYPRE_Solver) *solver,
                       (int)        *relax_type,
                       (int)        *k ) );
}


void
hypre_F90_IFACE(hypre_boomeramggetcyclerelaxtype, HYPRE_BOOMERAMGGETCYCLERELAXTYPE)( long int *solver,
                                               int   *relax_type,
                                               int   *k,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetCycleRelaxType(
                       (HYPRE_Solver) *solver,
                       (int *)        relax_type,
                       (int)           *k  ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetrelaxorder, HYPRE_BOOMERAMGSETRELAXORDER)( long int *solver,
                                               int   *relax_order,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetRelaxOrder(
                       (HYPRE_Solver) *solver,
                       (int)        *relax_order ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There is no alternative function.
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
 * DEPRECATED.  Use SetRelaxWt and SetLevelRelaxWt instead.
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
 * HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxwt, HYPRE_BOOMERAMGSETRELAXWT)( long int *solver,
                                             double *relax_weight,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetRelaxWt(
                       (HYPRE_Solver) *solver,
                       (double)     *relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelrelaxwt, HYPRE_BOOMERAMGSETLEVELRELAXWT)( long int *solver,
                                             double *relax_weight,
                                             int    *level,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLevelRelaxWt(
                       (HYPRE_Solver) *solver,
                       (double)     *relax_weight,
                       (int)        *level ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOmega - this is old interface - don't need it
 *--------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetouterwt, HYPRE_BOOMERAMGSETOUTERWT)( long int *solver,
                                                double      *outer_wt,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetOuterWt( (HYPRE_Solver) *solver,
                                                  (double)       *outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelouterwt, HYPRE_BOOMERAMGSETLEVELOUTERWT)( long int *solver,
                                                double   *outer_wt,
                                                int      *level,                                    
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLevelOuterWt( (HYPRE_Solver) *solver,
                                                   (double)       *outer_wt,
                                                   (int)           *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothType, HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetsmoothtype, HYPRE_BOOMERAMGSETSMOOTHTYPE)( long int *solver,
                                                int      *smooth_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSmoothType( (HYPRE_Solver) *solver,
                                                  (int)          *smooth_type ) );
}


void
hypre_F90_IFACE(hypre_boomeramggetsmoothtype, HYPRE_BOOMERAMGGETSMOOTHTYPE)( long int *solver,
                                                int      *smooth_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetSmoothType( (HYPRE_Solver) *solver,
                                                  (int *)        smooth_type ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumLvls, HYPRE_BoomerAMGGetSmoothNumLvls
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumlvls, HYPRE_BOOMERAMGSETSMOOTHNUMLVLS)( long int *solver,
                                                int      *smooth_num_levels,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSmoothNumLevels( (HYPRE_Solver) *solver,
                                                  (int)          *smooth_num_levels ) );
}


void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumlvls, HYPRE_BOOMERAMGGETSMOOTHNUMLVLS)( long int *solver,
                                                int      *smooth_num_levels,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetSmoothNumLevels( (HYPRE_Solver) *solver,
                                                  (int *)          smooth_num_levels ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSwps, HYPRE_BoomerAMGGetSmoothNumSwps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumswps, HYPRE_BOOMERAMGSETSMOOTHNUMSWPS)( long int *solver,
                                                int      *smooth_num_sweeps,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSmoothNumSweeps( (HYPRE_Solver) *solver,
                                                  (int)          *smooth_num_sweeps ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumswps, HYPRE_BOOMERAMGGETSMOOTHNUMSWPS)( long int *solver,
                                                int      *smooth_num_sweeps,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetSmoothNumSweeps( (HYPRE_Solver) *solver,
                                                  (int *)          smooth_num_sweeps ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogging, HYPRE_BOOMERAMGSETLOGGING)( long int *solver,
                                                int      *logging,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLogging( (HYPRE_Solver) *solver,
                                                  (int)          *logging ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetlogging, HYPRE_BOOMERAMGGETLOGGING)( long int *solver,
                                                int      *logging,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetLogging( (HYPRE_Solver) *solver,
                                                  (int *)     logging ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintlevel, HYPRE_BOOMERAMGSETPRINTLEVEL)( long int *solver,
                                         int      *print_level,
                                         int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetPrintLevel( (HYPRE_Solver) *solver,
                                           (int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintfilename, HYPRE_BOOMERAMGSETPRINTFILENAME)( long int *solver,
                                             char     *print_file_name,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetPrintFileName( (HYPRE_Solver) *solver,
                                               (char *)       print_file_name ) );
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
 * HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramggetcumnumiterations, HYPRE_BOOMERAMGGETCUMNUMITERATIONS)( long int *solver,
                                               int      *cum_num_iterations,
                                               int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetCumNumIterations( (HYPRE_Solver) *solver,
                                                 (int *)         cum_num_iterations ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetresidual, HYPRE_BOOMERAMGGETRESIDUAL)( long int *solver,
                                                  long int   *residual,
                                                  int      *ierr     )
{
   *ierr = (int) (HYPRE_BoomerAMGGetResidual((HYPRE_Solver) *solver,
                                            (HYPRE_ParVector *) residual));
}



/*--------------------------------------------------------------------------
 * HYPRE_ParAMGGetFinalRelativeResNorm
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

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant, HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetvariant, HYPRE_BOOMERAMGSETVARIANT)( long int *solver,
                                                int      *variant,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetVariant( (HYPRE_Solver) *solver,
                                                  (int)      *variant ) );
}


void
hypre_F90_IFACE(hypre_boomeramggetvariant, HYPRE_BOOMERAMGGETVARIANT)( long int *solver,
                                                int      *variant,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetVariant( (HYPRE_Solver) *solver,
                                                  (int *)      variant ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetoverlap, HYPRE_BOOMERAMGSETOVERLAP)( long int *solver,
                                                int      *overlap,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetOverlap( (HYPRE_Solver) *solver,
                                                  (int)          *overlap ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetoverlap, HYPRE_BOOMERAMGGETOVERLAP)( long int *solver,
                                                int      *overlap,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetOverlap( (HYPRE_Solver) *solver,
                                                  (int *)     overlap ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType, HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdomaintype, HYPRE_BOOMERAMGSETDOMAINTYPE)( long int *solver,
                                                int      *domain_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetDomainType( (HYPRE_Solver) *solver,
                                                  (int)          *domain_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdomaintype, HYPRE_BOOMERAMGGETDOMAINTYPE)( long int *solver,
                                                int      *domain_type,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetDomainType( (HYPRE_Solver) *solver,
                                                  (int *)        domain_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWt, HYPRE_BoomerAMGGetSchwarzRlxWt
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetschwarzrlxwt, HYPRE_BOOMERAMGSETSCHWARZRLXWT)( long int *solver,
                                                double      *schwarz_rlx_weight,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSchwarzRlxWeight( (HYPRE_Solver) *solver,
                                                  (double)        *schwarz_rlx_weight) );
}

void
hypre_F90_IFACE(hypre_boomeramggetschwarzrlxwt, HYPRE_BOOMERAMGGETSCHWARZRLXWT)( long int *solver,
                                                double      *schwarz_rlx_weight,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetSchwarzRlxWeight( (HYPRE_Solver) *solver,
                                                  (double *)        schwarz_rlx_weight) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsym, HYPRE_BOOMERAMGSETSYM)( long int *solver,
                                                int      *sym,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetSym( (HYPRE_Solver) *solver,
                                                  (int)          *sym ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevel, HYPRE_BOOMERAMGSETLEVEL)( long int *solver,
                                                int      *level,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetLevel( (HYPRE_Solver) *solver,
                                                  (int)          *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetthreshold, HYPRE_BOOMERAMGSETTHRESHOLD)( long int *solver,
                                                double      *threshold,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetThreshold( (HYPRE_Solver) *solver,
                                                  (double)        *threshold) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetfilter, HYPRE_BOOMERAMGSETFILTER)( long int *solver,
                                                double      *filter,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetFilter( (HYPRE_Solver) *solver,
                                                  (double)        *filter) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdroptol, HYPRE_BOOMERAMGSETDROPTOL)( long int *solver,
                                                double      *drop_tol,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetDropTol( (HYPRE_Solver) *solver,
                                                  (double)        *drop_tol) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxnzperrow, HYPRE_BOOMERAMGSETMAXNZPERROW)( long int *solver,
                                                int      *max_nz_per_row,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetMaxNzPerRow( (HYPRE_Solver) *solver,
                                                  (int)          *max_nz_per_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteuclidfile, HYPRE_BOOMERAMGSETEUCLIDFILE)( long int *solver,
                                             char     *euclidfile,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetEuclidFile( (HYPRE_Solver) *solver,
                                               (char *)       euclidfile ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions, HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnumfunctions, HYPRE_BOOMERAMGSETNUMFUNCTIONS)( long int *solver,
                                                int      *num_functions,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNumFunctions( (HYPRE_Solver) *solver,
                                                  (int)          *num_functions ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetnumfunctions, HYPRE_BOOMERAMGGETNUMFUNCTIONS)( long int *solver,
                                                int      *num_functions,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGGetNumFunctions( (HYPRE_Solver) *solver,
                                                  (int *)          num_functions ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodal, HYPRE_BOOMERAMGSETNODAL)( long int *solver,
                                                int      *nodal,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNodal( (HYPRE_Solver) *solver,
                                                  (int)          *nodal ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdoffunc, HYPRE_BOOMERAMGSETDOFFUNC)( long int *solver,
                                               long int *dof_func,
                                               int      *ierr             )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetDofFunc(
                        (HYPRE_Solver) *solver,
                        (int *)        *((int **)(*dof_func)) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetnumpaths, HYPRE_BOOMERAMGSETNUMPATHS)( long int *solver,
                                                int      *num_paths,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNumPaths( (HYPRE_Solver) *solver,
                                                  (int)          *num_paths ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggnumlevels, HYPRE_BOOMERAMGSETAGGNUMLEVELS)( long int *solver,
                                                int      *agg_num_levels,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetAggNumLevels( (HYPRE_Solver) *solver,
                                                  (int)          *agg_num_levels ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgsmg, HYPRE_BOOMERAMGSETGSMG)( long int *solver,
                                                  int      *gsmg,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetGSMG(
                                (HYPRE_Solver) *solver,
                                (int)          *gsmg ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnumsamples, HYPRE_BOOMERAMGSETNUMSAMPLES)( long int *solver,
                                                  int      *gsmg,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_BoomerAMGSetNumSamples(
                                (HYPRE_Solver) *solver,
                                (int)          *gsmg ) );
}
