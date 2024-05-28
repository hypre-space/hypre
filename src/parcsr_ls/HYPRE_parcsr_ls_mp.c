/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_parcsr_ls mixed precision functions
 *
 *****************************************************************************/

#include "HYPRE_parcsr_ls_mp.h"
#include "_hypre_parcsr_ls.h"
#include "hypre_parcsr_ls_mup.h"
#include "hypre_parcsr_mv_mup.h"
#include "HYPRE_parcsr_mv_mp.h"
#include "hypre_utilities_mup.h"


#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BoomerAMGSetup_mp( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );   

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */        
   HYPRE_BoomerAMGSetup_flt( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

/* free data */   
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;

}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_BoomerAMGSolve_mp( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );   

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */        
   HYPRE_BoomerAMGSolve_flt( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

/* free data */   
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;
}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_MPAMGPrecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_MPAMGPrecSetup_mp( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );   

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */        
   HYPRE_MPAMGSetup_mp( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

/* free data */   
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;

}

/*--------------------------------------------------------------------------
 * Mixed-precision HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_MPAMGPrecSolve_mp( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   hypre_ParVector *btemp = NULL;
   hypre_ParVector *xtemp = NULL;

   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &btemp);
   HYPRE_ParVectorInitialize_flt( btemp );
   HYPRE_ParVectorCreate_flt(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 &xtemp);
   HYPRE_ParVectorInitialize_flt( xtemp );   

/* copy from double precision {b,x} to single precision {btemp,xtemp} */
   HYPRE_ParVectorCopy_mp(b, btemp);
   HYPRE_ParVectorCopy_mp(x, xtemp);

/* call setup */        
   HYPRE_MPAMGSolve_mp( solver, A, btemp, xtemp );

/* copy from single precision {btemp,xtemp} to double precision {b,x} */
   HYPRE_ParVectorCopy_mp(btemp, b);
   HYPRE_ParVectorCopy_mp(xtemp, x);

/* free data */   
   HYPRE_ParVectorDestroy_flt(btemp);
   HYPRE_ParVectorDestroy_flt(xtemp);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetup_mp( HYPRE_Solver solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector b,
                     HYPRE_ParVector x      )
{
  if (!A)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_MPAMGSetup_mp( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSolve_mp( HYPRE_Solver solver,
                     HYPRE_ParCSRMatrix A,
                     HYPRE_ParVector b,
                     HYPRE_ParVector x      )
{
  if (!A)
   {
      //hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   return ( hypre_MPAMGSolve_mp( (void *) solver,
                                 (hypre_ParCSRMatrix *) A,
                                 (hypre_ParVector *) b,
                                 (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGCreate_mp( HYPRE_Solver *solver)
{
   if (!solver)
   {
      //hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (HYPRE_Solver) hypre_MPAMGCreate_mp( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGDestroy_mp( HYPRE_Solver solver )
{
   return ( hypre_MPAMGDestroy_mp( (void *) solver ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerMPAMGSetMaxLevels, HYPRE_BoomerMPAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMaxLevels_mp( HYPRE_Solver solver,
                             HYPRE_Int          max_levels  )
{
   return ( hypre_MPAMGSetMaxLevels_mp( (void *) solver, max_levels ) );
}

HYPRE_Int
HYPRE_MPAMGGetMaxLevels_mp( HYPRE_Solver solver,
                             HYPRE_Int        * max_levels  )
{
   return ( hypre_MPAMGGetMaxLevels_mp( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMaxCoarseSize, HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMaxCoarseSize_mp( HYPRE_Solver solver,
                                 HYPRE_Int          max_coarse_size  )
{
   return( hypre_MPAMGSetMaxCoarseSize_mp( (void *) solver, max_coarse_size ) );
}

HYPRE_Int
HYPRE_MPAMGGetMaxCoarseSize_mp( HYPRE_Solver solver,
                                 HYPRE_Int        * max_coarse_size  )
{
   return( hypre_MPAMGGetMaxCoarseSize_mp( (void *) solver, max_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMinCoarseSize, HYPRE_BoomerAMGGetMinCoarseSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMinCoarseSize_mp( HYPRE_Solver solver,
                                 HYPRE_Int          min_coarse_size  )
{
   return( hypre_MPAMGSetMinCoarseSize_mp( (void *) solver, min_coarse_size ) );
}

HYPRE_Int
HYPRE_MPAMGGetMinCoarseSize_mp( HYPRE_Solver solver,
                                 HYPRE_Int        * min_coarse_size  )
{
   return( hypre_MPAMGGetMinCoarseSize_mp( (void *) solver, min_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetCoarsenCutFactor, HYPRE_BoomerAMGGetCoarsenCutFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetCoarsenCutFactor_mp( HYPRE_Solver solver,
                                    HYPRE_Int    coarsen_cut_factor )
{
   return( hypre_MPAMGSetCoarsenCutFactor_mp( (void *) solver, coarsen_cut_factor ) );
}

HYPRE_Int
HYPRE_MPAMGGetCoarsenCutFactor_mp( HYPRE_Solver  solver,
                                    HYPRE_Int    *coarsen_cut_factor )
{
   return( hypre_MPAMGGetCoarsenCutFactor_mp( (void *) solver, coarsen_cut_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetStrongThreshold_mp( HYPRE_Solver solver,
                                   HYPRE_Real   strong_threshold  )
{
   return( hypre_MPAMGSetStrongThreshold_mp( (void *) solver,
                                               strong_threshold ) );
}

HYPRE_Int
HYPRE_MPAMGGetStrongThreshold_mp( HYPRE_Solver solver,
                                   HYPRE_Real * strong_threshold  )
{
   return( hypre_MPAMGGetStrongThreshold_mp( (void *) solver,
                                               strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMaxRowSum_mp( HYPRE_Solver solver,
                             HYPRE_Real   max_row_sum  )
{
   return( hypre_MPAMGSetMaxRowSum_mp( (void *) solver,
                                         max_row_sum ) );
}

HYPRE_Int
HYPRE_MPAMGGetMaxRowSum_mp( HYPRE_Solver solver,
                             HYPRE_Real * max_row_sum  )
{
   return( hypre_MPAMGGetMaxRowSum_mp( (void *) solver,
                                         max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetTruncFactor_mp( HYPRE_Solver solver,
                               HYPRE_Real   trunc_factor  )
{
   return( hypre_MPAMGSetTruncFactor_mp( (void *) solver,
                                           trunc_factor ) );
}

HYPRE_Int
HYPRE_MPAMGGetTruncFactor_mp( HYPRE_Solver solver,
                               HYPRE_Real * trunc_factor  )
{
   return( hypre_MPAMGGetTruncFactor_mp( (void *) solver,
                                           trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetPMaxElmts, HYPRE_BoomerAMGGetPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetPMaxElmts_mp( HYPRE_Solver solver,
                             HYPRE_Int   P_max_elmts  )
{
   return( hypre_MPAMGSetPMaxElmts_mp( (void *) solver,
                                         P_max_elmts ) );
}

HYPRE_Int
HYPRE_MPAMGGetPMaxElmts_mp( HYPRE_Solver solver,
                             HYPRE_Int   * P_max_elmts  )
{
   return( hypre_MPAMGGetPMaxElmts_mp( (void *) solver,
                                         P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetInterpType_mp( HYPRE_Solver solver,
                              HYPRE_Int          interp_type  )
{
   return( hypre_MPAMGSetInterpType_mp( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetSepWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetSepWeight_mp( HYPRE_Solver solver,
                             HYPRE_Int          sep_weight  )
{
   return( hypre_MPAMGSetSepWeight_mp( (void *) solver, sep_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMinIter_mp( HYPRE_Solver solver,
                           HYPRE_Int          min_iter  )
{
   return( hypre_MPAMGSetMinIter_mp( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMaxIter_mp( HYPRE_Solver solver,
                           HYPRE_Int          max_iter  )
{
   return( hypre_MPAMGSetMaxIter_mp( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_MPAMGGetMaxIter_mp( HYPRE_Solver solver,
                           HYPRE_Int        * max_iter  )
{
   return( hypre_MPAMGGetMaxIter_mp( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetCoarsenType, HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetCoarsenType_mp( HYPRE_Solver solver,
                               HYPRE_Int          coarsen_type  )
{
   return( hypre_MPAMGSetCoarsenType_mp( (void *) solver, coarsen_type ) );
}

HYPRE_Int
HYPRE_MPAMGGetCoarsenType_mp( HYPRE_Solver solver,
                               HYPRE_Int        * coarsen_type  )
{
   return( hypre_MPAMGGetCoarsenType_mp( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetMeasureType, HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetMeasureType_mp( HYPRE_Solver solver,
                               HYPRE_Int          measure_type  )
{
   return( hypre_MPAMGSetMeasureType_mp( (void *) solver, measure_type ) );
}

HYPRE_Int
HYPRE_MPAMGGetMeasureType_mp( HYPRE_Solver solver,
                               HYPRE_Int        * measure_type  )
{
   return( hypre_MPAMGGetMeasureType_mp( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetCycleType, HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetCycleType_mp( HYPRE_Solver solver,
                             HYPRE_Int          cycle_type  )
{
   return( hypre_MPAMGSetCycleType_mp( (void *) solver, cycle_type ) );
}

HYPRE_Int
HYPRE_MPAMGGetCycleType_mp( HYPRE_Solver solver,
                             HYPRE_Int        * cycle_type  )
{
   return( hypre_MPAMGGetCycleType_mp( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetFCycle, HYPRE_BoomerAMGGetFCycle
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetFCycle_mp( HYPRE_Solver solver,
                          HYPRE_Int    fcycle  )
{
   return( hypre_MPAMGSetFCycle_mp( (void *) solver, fcycle ) );
}

HYPRE_Int
HYPRE_MPAMGGetFCycle_mp( HYPRE_Solver solver,
                          HYPRE_Int   *fcycle  )
{
   return( hypre_MPAMGGetFCycle_mp( (void *) solver, fcycle ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetTol, HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetTol_mp( HYPRE_Solver solver,
                       HYPRE_Real   tol    )
{
   return( hypre_MPAMGSetTol_mp( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_MPAMGGetTol_mp( HYPRE_Solver solver,
                       HYPRE_Real * tol    )
{
   return( hypre_MPAMGGetTol_mp( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNumGridSweeps
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array _mp(who releases it?).
 * Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNumGridSweeps_mp( HYPRE_Solver  solver,
                                 HYPRE_Int          *num_grid_sweeps  )
{
   return( hypre_MPAMGSetNumGridSweeps_mp( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNumSweeps
 * There is no corresponding Get function.  Use GetCycleNumSweeps.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNumSweeps_mp( HYPRE_Solver  solver,
                             HYPRE_Int          num_sweeps  )
{
   return( hypre_MPAMGSetNumSweeps_mp( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetCycleNumSweeps, HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetCycleNumSweeps_mp( HYPRE_Solver  solver,
                                  HYPRE_Int          num_sweeps, HYPRE_Int k  )
{
   return( hypre_MPAMGSetCycleNumSweeps_mp( (void *) solver, num_sweeps, k ) );
}

HYPRE_Int
HYPRE_MPAMGGetCycleNumSweeps_mp( HYPRE_Solver  solver,
                                  HYPRE_Int        * num_sweeps, HYPRE_Int k  )
{
   return( hypre_MPAMGGetCycleNumSweeps_mp( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetGridRelaxType
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array _mp(who releases it?).
 * Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetGridRelaxType_mp( HYPRE_Solver  solver,
                                 HYPRE_Int          *grid_relax_type  )
{
   return( hypre_MPAMGSetGridRelaxType_mp( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetRelaxType_mp( HYPRE_Solver  solver,
                             HYPRE_Int          relax_type  )
{
   return( hypre_MPAMGSetRelaxType_mp( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetCycleRelaxType, HYPRE_BoomerAMGetCycleRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetCycleRelaxType_mp( HYPRE_Solver  solver,
                                  HYPRE_Int          relax_type, HYPRE_Int k  )
{
   return( hypre_MPAMGSetCycleRelaxType_mp( (void *) solver, relax_type, k ) );
}

HYPRE_Int
HYPRE_MPAMGGetCycleRelaxType_mp( HYPRE_Solver  solver,
                                  HYPRE_Int        * relax_type, HYPRE_Int k  )
{
   return( hypre_MPAMGGetCycleRelaxType_mp( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetRelaxOrder_mp( HYPRE_Solver  solver,
                              HYPRE_Int           relax_order)
{
   return( hypre_MPAMGSetRelaxOrder_mp( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetGridRelaxPoints
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array _mp(who releases it?).
 * Ulrike Yang suspects that nobody uses this function.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetGridRelaxPoints_mp( HYPRE_Solver   solver,
                                   HYPRE_Int          **grid_relax_points  )
{
   return( hypre_MPAMGSetGridRelaxPoints_mp( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetRelaxWeight
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array _mp(who releases it?).
 * Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetRelaxWeight_mp( HYPRE_Solver  solver,
                               HYPRE_Real   *relax_weight  )
{
   return( hypre_MPAMGSetRelaxWeight_mp( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetRelaxWt_mp( HYPRE_Solver  solver,
                           HYPRE_Real    relax_wt  )
{
   return( hypre_MPAMGSetRelaxWt_mp( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetLevelRelaxWt_mp( HYPRE_Solver  solver,
                                HYPRE_Real    relax_wt,
                                HYPRE_Int         level  )
{
   return( hypre_MPAMGSetLevelRelaxWt_mp( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetOmega
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetOmega_mp( HYPRE_Solver  solver,
                         HYPRE_Real   *omega  )
{
   return( hypre_MPAMGSetOmega_mp( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetOuterWt_mp( HYPRE_Solver  solver,
                           HYPRE_Real    outer_wt  )
{
   return( hypre_MPAMGSetOuterWt_mp( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetLevelOuterWt_mp( HYPRE_Solver  solver,
                                HYPRE_Real    outer_wt,
                                HYPRE_Int         level  )
{
   return( hypre_MPAMGSetLevelOuterWt_mp( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetLogging_mp( HYPRE_Solver solver,
                           HYPRE_Int          logging  )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   return( hypre_MPAMGSetLogging_mp( (void *) solver, logging ) );
}

HYPRE_Int
HYPRE_MPAMGGetLogging_mp( HYPRE_Solver solver,
                           HYPRE_Int        * logging  )
{
   return( hypre_MPAMGGetLogging_mp( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetPrintLevel, HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetPrintLevel_mp( HYPRE_Solver solver,
                              HYPRE_Int        print_level  )
{
   return( hypre_MPAMGSetPrintLevel_mp( (void *) solver, print_level ) );
}

HYPRE_Int
HYPRE_MPAMGGetPrintLevel_mp( HYPRE_Solver solver,
                              HYPRE_Int      * print_level  )
{
   return( hypre_MPAMGGetPrintLevel_mp( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetDebugFlag, HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetDebugFlag_mp( HYPRE_Solver solver,
                             HYPRE_Int          debug_flag  )
{
   return( hypre_MPAMGSetDebugFlag_mp( (void *) solver, debug_flag ) );
}

HYPRE_Int
HYPRE_MPAMGGetDebugFlag_mp( HYPRE_Solver solver,
                             HYPRE_Int        * debug_flag  )
{
   return( hypre_MPAMGGetDebugFlag_mp( (void *) solver, debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGGetNumIterations_mp( HYPRE_Solver  solver,
                                 HYPRE_Int          *num_iterations  )
{
   return( hypre_MPAMGGetNumIterations_mp( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGGetCumNumIterations_mp( HYPRE_Solver  solver,
                                    HYPRE_Int          *cum_num_iterations  )
{
   return( hypre_MPAMGGetCumNumIterations_mp( (void *) solver, cum_num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGGetResidual_mp( HYPRE_Solver solver, HYPRE_ParVector * residual )
{
   return hypre_MPAMGGetResidual_mp( (void *) solver,
                                     (hypre_ParVector **) residual );
}


/*--------------------------------------------------------------------------
 * HYPRE_MPAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGGetFinalRelativeResidualNorm_mp( HYPRE_Solver  solver,
                                             HYPRE_Real   *rel_resid_norm  )
{
   return( hypre_MPAMGGetRelResidualNorm_mp( (void *) solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNumFunctions, HYPRE_MPAMGGetNumFunctions
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNumFunctions_mp( HYPRE_Solver  solver,
                                HYPRE_Int          num_functions  )
{
   return( hypre_MPAMGSetNumFunctions_mp( (void *) solver, num_functions ) );
}

HYPRE_Int
HYPRE_MPAMGGetNumFunctions_mp( HYPRE_Solver  solver,
                                HYPRE_Int        * num_functions  )
{
   return( hypre_MPAMGGetNumFunctions_mp( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNodal
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNodal_mp( HYPRE_Solver  solver,
                         HYPRE_Int          nodal  )
{
   return( hypre_MPAMGSetNodal_mp( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNodalLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNodalLevels_mp( HYPRE_Solver  solver,
                               HYPRE_Int          nodal_levels  )
{
   return( hypre_MPAMGSetNodalLevels_mp( (void *) solver, nodal_levels ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNodalDiag
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNodalDiag_mp( HYPRE_Solver  solver,
                             HYPRE_Int          nodal  )
{
   return( hypre_MPAMGSetNodalDiag_mp( (void *) solver, nodal ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetKeepSameSign
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetKeepSameSign_mp( HYPRE_Solver  solver,
                                HYPRE_Int     keep_same_sign  )
{
   return( hypre_MPAMGSetKeepSameSign_mp( (void *) solver, keep_same_sign ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetDofFunc
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetDofFunc_mp( HYPRE_Solver  solver,
                           HYPRE_Int          *dof_func  )
/* Warning about a possible memory problem: When the MPAMG object is destroyed
   in hypre_MPAMGDestroy, dof_func aka DofFunc will be destroyed _mp(currently
   line 246 of par_amg.c).  Normally this is what we want.  But if the user provided
   dof_func by calling HYPRE_MPAMGSetDofFunc, this could be an unwanted surprise.
   As hypre is currently commonly used, this situation is likely to be rare. */
{
   return( hypre_MPAMGSetDofFunc_mp( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetNumPaths
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetNumPaths_mp( HYPRE_Solver  solver,
                            HYPRE_Int          num_paths  )
{
   return( hypre_MPAMGSetNumPaths_mp( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggNumLevels_mp( HYPRE_Solver  solver,
                                HYPRE_Int          agg_num_levels  )
{
   return( hypre_MPAMGSetAggNumLevels_mp( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggInterpType
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggInterpType_mp( HYPRE_Solver  solver,
                                 HYPRE_Int          agg_interp_type  )
{
   return( hypre_MPAMGSetAggInterpType_mp( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggTruncFactor_mp( HYPRE_Solver  solver,
                                  HYPRE_Real    agg_trunc_factor  )
{
   return( hypre_MPAMGSetAggTruncFactor_mp( (void *) solver, agg_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggP12TruncFactor_mp( HYPRE_Solver  solver,
                                     HYPRE_Real    agg_P12_trunc_factor  )
{
   return( hypre_MPAMGSetAggP12TruncFactor_mp( (void *) solver, agg_P12_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggPMaxElmts_mp( HYPRE_Solver  solver,
                                HYPRE_Int          agg_P_max_elmts  )
{
   return( hypre_MPAMGSetAggPMaxElmts_mp( (void *) solver, agg_P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetAggP12MaxElmts_mp( HYPRE_Solver  solver,
                                  HYPRE_Int          agg_P12_max_elmts  )
{
   return( hypre_MPAMGSetAggP12MaxElmts_mp( (void *) solver, agg_P12_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetRAP2
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetRAP2_mp(HYPRE_Solver solver,
                        HYPRE_Int    rap2)
{
   return (hypre_MPAMGSetRAP2_mp ( (void *) solver, rap2 ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetKeepTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetKeepTranspose_mp(HYPRE_Solver solver,
                                 HYPRE_Int    keepTranspose)
{
   return (hypre_MPAMGSetKeepTranspose_mp ( (void *) solver, keepTranspose ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_MPAMGSetPrecisionArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_MPAMGSetPrecisionArray_mp(HYPRE_Solver solver,
                                HYPRE_Precision *precision_array)
{
   return (hypre_MPAMGSetPrecisionArray_mp ( (void *) solver, precision_array ) );
}

#endif
