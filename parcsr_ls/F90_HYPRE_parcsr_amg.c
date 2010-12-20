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
 * HYPRE_ParAMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgcreate, HYPRE_BOOMERAMGCREATE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGCreate( (HYPRE_Solver *) solver) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgdestroy, HYPRE_BOOMERAMGDESTROY)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsetup, HYPRE_BOOMERAMGSETUP)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A,
   hypre_F90_Obj *b,
   hypre_F90_Obj *x,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetup( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsolve, HYPRE_BOOMERAMGSOLVE)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A,
   hypre_F90_Obj *b,
   hypre_F90_Obj *x,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSolve( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_boomeramgsolvet, HYPRE_BOOMERAMGSOLVET)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *A,
   hypre_F90_Obj *b,
   hypre_F90_Obj *x,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSolveT( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b,
                                          (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrestriction, HYPRE_BOOMERAMGSETRESTRICTION)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *restr_par,
   HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetRestriction( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *restr_par ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxLevels, HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxlevels, HYPRE_BOOMERAMGSETMAXLEVELS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_levels,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMaxLevels( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *max_levels ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxlevels, HYPRE_BOOMERAMGGETMAXLEVELS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_levels,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetMaxLevels( (HYPRE_Solver) *solver,
                                                (HYPRE_Int *)          max_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxCoarseSize, HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetmaxcoarsesize, HYPRE_BOOMERAMGSETMAXCOARSESIZE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_coarse_size,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMaxCoarseSize( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *max_coarse_size ) );
}



void
hypre_F90_IFACE(hypre_boomeramggetmaxcoarsesize, HYPRE_BOOMERAMGGETMAXCOARSESIZE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_coarse_size,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetMaxCoarseSize( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int *)          max_coarse_size ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetStrongThreshold, HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetstrongthrshld, HYPRE_BOOMERAMGSETSTRONGTHRSHLD)(
   hypre_F90_Obj *solver,
   double   *strong_threshold,
   HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_BoomerAMGSetStrongThreshold( (HYPRE_Solver) *solver,
                                           (double)       *strong_threshold ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetstrongthrshld, HYPRE_BOOMERAMGGETSTRONGTHRSHLD)(
   hypre_F90_Obj *solver,
   double   *strong_threshold,
   HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_BoomerAMGGetStrongThreshold( (HYPRE_Solver) *solver,
                                           (double *)       strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxRowSum, HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxrowsum, HYPRE_BOOMERAMGSETMAXROWSUM)(
   hypre_F90_Obj *solver,
   double   *max_row_sum,
   HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_BoomerAMGSetMaxRowSum( (HYPRE_Solver) *solver,
                                     (double)       *max_row_sum ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxrowsum, HYPRE_BOOMERAMGGETMAXROWSUM)(
   hypre_F90_Obj *solver,
   double   *max_row_sum,
   HYPRE_Int      *ierr              )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_BoomerAMGGetMaxRowSum( (HYPRE_Solver) *solver,
                                     (double *)       max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTruncFactor, HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettruncfactor, HYPRE_BOOMERAMGSETTRUNCFACTOR)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetTruncFactor( (HYPRE_Solver) *solver,
                                                  (double)       *trunc_factor ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettruncfactor, HYPRE_BOOMERAMGGETTRUNCFACTOR)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetTruncFactor( (HYPRE_Solver) *solver,
                                                  (double *)       trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpmaxelmts, HYPRE_BOOMERAMGSETPMAXELMTS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *p_max_elmts,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetPMaxElmts( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *p_max_elmts ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetpmaxelmts, HYPRE_BOOMERAMGGETPMAXELMTS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *p_max_elmts,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetPMaxElmts( (HYPRE_Solver) *solver,
                                               (HYPRE_Int *)          p_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetJacobiTruncThreshold, HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetjacobitrunc, HYPRE_BOOMERAMGSETJACOBITRUNC)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetJacobiTruncThreshold( (HYPRE_Solver) *solver,
                                                           (double)       *trunc_factor ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetjacobitrunc, HYPRE_BOOMERAMGGETJACOBITRUNC)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetJacobiTruncThreshold( (HYPRE_Solver) *solver,
                                                           (double *)       trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPostInterpType, HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetpostinterp, HYPRE_BOOMERAMGSETPOSTINTERP)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *type,
   HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetPostInterpType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *type ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSCommPkgSwitch
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetscommpkgswitc, HYPRE_BOOMERAMGSETSCOMMPKGSWITC)(
   hypre_F90_Obj *solver,
   double      *S_commpkg_switch,
   HYPRE_Int         *ierr         )


{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSCommPkgSwitch( (HYPRE_Solver) *solver,
                                                     (double) *S_commpkg_switch ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetinterptype, HYPRE_BOOMERAMGSETINTERPTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *interp_type,
   HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetInterpType( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *interp_type ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetsepweight, HYPRE_BOOMERAMGSETSEPWEIGHT)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *sep_weight,
   HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSepWeight( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *sep_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetminiter, HYPRE_BOOMERAMGSETMINITER)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *min_iter,
   HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMinIter( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxIter, HYPRE_BoomerAMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxiter, HYPRE_BOOMERAMGSETMAXITER)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_iter,
   HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMaxIter( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *max_iter ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmaxiter, HYPRE_BOOMERAMGGETMAXITER)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_iter,
   HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetMaxIter( (HYPRE_Solver) *solver,
                                              (HYPRE_Int *)          max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCoarsenType, HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcoarsentype, HYPRE_BOOMERAMGSETCOARSENTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *coarsen_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetCoarsenType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *coarsen_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcoarsentype, HYPRE_BOOMERAMGGETCOARSENTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *coarsen_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetCoarsenType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int *)          coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMeasureType, HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmeasuretype, HYPRE_BOOMERAMGSETMEASURETYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *measure_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMeasureType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *measure_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetmeasuretype, HYPRE_BOOMERAMGGETMEASURETYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *measure_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetMeasureType( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int *)          measure_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsetuptype, HYPRE_BOOMERAMGSETSETUPTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *setup_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSetupType( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *setup_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleType, HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcycletype, HYPRE_BOOMERAMGSETCYCLETYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *cycle_type,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetCycleType( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *cycle_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcycletype, HYPRE_BOOMERAMGGETCYCLETYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *cycle_type,
   HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetCycleType( (HYPRE_Solver) *solver,
                                                (HYPRE_Int *)          cycle_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetTol, HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsettol, HYPRE_BOOMERAMGSETTOL)(
   hypre_F90_Obj *solver,
   double   *tol,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetTol( (HYPRE_Solver) *solver,
                                          (double)       *tol     ) );
}

void
hypre_F90_IFACE(hypre_boomeramggettol, HYPRE_BOOMERAMGGETTOL)(
   hypre_F90_Obj *solver,
   double   *tol,
   HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetTol( (HYPRE_Solver) *solver,
                                          (double *)       tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 * DEPRECATED.  Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumgridsweeps, HYPRE_BOOMERAMGSETNUMGRIDSWEEPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_grid_sweeps,
   HYPRE_Int      *ierr             )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNumGridSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsweeps, HYPRE_BOOMERAMGSETNUMSWEEPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_sweeps,
   HYPRE_Int      *ierr             )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNumSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)        *num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleNumSweeps, HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclenumsweeps, HYPRE_BOOMERAMGSETCYCLENUMSWEEPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_sweeps,
   HYPRE_Int      *k,
   HYPRE_Int      *ierr             )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetCycleNumSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)        *num_sweeps,
                      (HYPRE_Int)        *k ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclenumsweeps, HYPRE_BOOMERAMGGETCYCLENUMSWEEPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_sweeps,
   HYPRE_Int      *k,
   HYPRE_Int      *ierr             )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetCycleNumSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)        num_sweeps,
                      (HYPRE_Int)        *k ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGInitGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays that are allocated.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramginitgridrelaxatn, HYPRE_BOOMERAMGINITGRIDRELAXATN)(
   hypre_F90_Obj *num_grid_sweeps,
   hypre_F90_Obj *grid_relax_type,
   hypre_F90_Obj *grid_relax_points,
   HYPRE_Int      *coarsen_type,
   hypre_F90_Obj *relax_weights,
   HYPRE_Int      *max_levels,
   HYPRE_Int      *ierr               )
{
   *num_grid_sweeps   = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int*, 1);
   *grid_relax_type   = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int*, 1);
   *grid_relax_points = (hypre_F90_Obj) hypre_CTAlloc(HYPRE_Int**, 1);
   *relax_weights     = (hypre_F90_Obj) hypre_CTAlloc(double*, 1);

   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGInitGridRelaxation( (HYPRE_Int **)    *num_grid_sweeps,
                                                      (HYPRE_Int **)    *grid_relax_type,
                                                      (HYPRE_Int ***)   *grid_relax_points,
                                                      (HYPRE_Int)       *coarsen_type,
                                                      (double **) *relax_weights,
                                                      (HYPRE_Int)       *max_levels         ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGFinalizeGridRelaxation
 *
 * RDF: This is probably not a very useful Fortran routine because you can't do
 * anything with the pointers to arrays.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgfingridrelaxatn, HYPRE_BOOMERAMGFINGRIDRELAXATN)(
   hypre_F90_Obj *num_grid_sweeps,
   hypre_F90_Obj *grid_relax_type,
   hypre_F90_Obj *grid_relax_points,
   hypre_F90_Obj *relax_weights,
   HYPRE_Int      *ierr               )
{
   char *ptr_num_grid_sweeps   = (char *) *num_grid_sweeps;
   char *ptr_grid_relax_type   = (char *) *grid_relax_type;
   char *ptr_grid_relax_points = (char *) *grid_relax_points;
   char *ptr_relax_weights     = (char *) *relax_weights;

   hypre_TFree(ptr_num_grid_sweeps);
   hypre_TFree(ptr_grid_relax_type);
   hypre_TFree(ptr_grid_relax_points);
   hypre_TFree(ptr_relax_weights);

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxtype, HYPRE_BOOMERAMGSETGRIDRELAXTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *grid_relax_type,
   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetGridRelaxType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxtype, HYPRE_BOOMERAMGSETRELAXTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_type,
   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetRelaxType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCycleRelaxType, HYPRE_BoomerAMGGetCycleRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetcyclerelaxtype, HYPRE_BOOMERAMGSETCYCLERELAXTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_type,
   HYPRE_Int      *k,
   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetCycleRelaxType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *relax_type,
                      (HYPRE_Int)          *k ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetcyclerelaxtype, HYPRE_BOOMERAMGGETCYCLERELAXTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_type,
   HYPRE_Int      *k,
   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetCycleRelaxType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         relax_type,
                      (HYPRE_Int)          *k  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxorder, HYPRE_BOOMERAMGSETRELAXORDER)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_order,
   HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetRelaxOrder(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *relax_order ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There is no alternative function.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgridrelaxpnts, HYPRE_BOOMERAMGSETGRIDRELAXPNTS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      **grid_relax_points,
   HYPRE_Int       *ierr               )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetGridRelaxPoints(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int **)        grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxweight, HYPRE_BOOMERAMGSETRELAXWEIGHT)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *relax_weights,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetRelaxWeight(
                      (HYPRE_Solver) *solver,
                      (double *)      relax_weights ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetrelaxwt, HYPRE_BOOMERAMGSETRELAXWT)(
   hypre_F90_Obj *solver,
   double   *relax_weight,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetRelaxWt(
                      (HYPRE_Solver) *solver,
                      (double)       *relax_weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelrelaxwt, HYPRE_BOOMERAMGSETLEVELRELAXWT)(
   hypre_F90_Obj *solver,
   double   *relax_weight,
   HYPRE_Int      *level,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetLevelRelaxWt(
                      (HYPRE_Solver) *solver,
                      (double)       *relax_weight,
                      (HYPRE_Int)          *level ) );
}




/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetouterwt, HYPRE_BOOMERAMGSETOUTERWT)(
   hypre_F90_Obj *solver,
   double   *outer_wt,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetOuterWt( (HYPRE_Solver) *solver,
                                              (double)       *outer_wt ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevelouterwt, HYPRE_BOOMERAMGSETLEVELOUTERWT)(
   hypre_F90_Obj *solver,
   double   *outer_wt,
   HYPRE_Int      *level,                                    
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetLevelOuterWt( (HYPRE_Solver) *solver,
                                                   (double)       *outer_wt,
                                                   (HYPRE_Int)          *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothType, HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothtype, HYPRE_BOOMERAMGSETSMOOTHTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSmoothType( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)         *smooth_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothtype, HYPRE_BOOMERAMGGETSMOOTHTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetSmoothType( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int *)        smooth_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumLvls, HYPRE_BoomerAMGGetSmoothNumLvls
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumlvls, HYPRE_BOOMERAMGSETSMOOTHNUMLVLS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_num_levels,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSmoothNumLevels(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *smooth_num_levels ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumlvls, HYPRE_BOOMERAMGGETSMOOTHNUMLVLS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_num_levels,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetSmoothNumLevels(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         smooth_num_levels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSmoothNumSwps, HYPRE_BoomerAMGGetSmoothNumSwps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsmoothnumswps, HYPRE_BOOMERAMGSETSMOOTHNUMSWPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_num_sweeps,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSmoothNumSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *smooth_num_sweeps ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetsmoothnumswps, HYPRE_BOOMERAMGGETSMOOTHNUMSWPS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *smooth_num_sweeps,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetSmoothNumSweeps(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         smooth_num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLogging, HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlogging, HYPRE_BOOMERAMGSETLOGGING)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *logging,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetLogging( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *logging ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetlogging, HYPRE_BOOMERAMGGETLOGGING)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *logging,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetLogging( (HYPRE_Solver) *solver,
                                              (HYPRE_Int *)         logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintLevel, HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintlevel, HYPRE_BOOMERAMGSETPRINTLEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *print_level,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetPrintLevel( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *print_level ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetprintlevel, HYPRE_BOOMERAMGGETPRINTLEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *print_level,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetPrintLevel( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int *)         print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetprintfilename, HYPRE_BOOMERAMGSETPRINTFILENAME)(
   hypre_F90_Obj *solver,
   char     *print_file_name,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetPrintFileName(
                      (HYPRE_Solver) *solver,
                      (char *)        print_file_name ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDebugFlag, HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdebugflag, HYPRE_BOOMERAMGSETDEBUGFLAG)
(
   hypre_F90_Obj *solver,
   HYPRE_Int      *debug_flag,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetDebugFlag( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *debug_flag ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdebugflag, HYPRE_BOOMERAMGGETDEBUGFLAG)
(
   hypre_F90_Obj *solver,
   HYPRE_Int      *debug_flag,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetDebugFlag( (HYPRE_Solver) *solver,
                                                (HYPRE_Int *)         debug_flag ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetnumiterations, HYPRE_BOOMERAMGGETNUMITERATIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_iterations,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetNumIterations(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetcumnumiterati, HYPRE_BOOMERAMGGETCUMNUMITERATI)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *cum_num_iterations,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetCumNumIterations(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         cum_num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetresidual, HYPRE_BOOMERAMGGETRESIDUAL)(
   hypre_F90_Obj *solver,
   hypre_F90_Obj *residual,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) (HYPRE_BoomerAMGGetResidual(
                     (HYPRE_Solver)      *solver,
                     (HYPRE_ParVector *)  residual));
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGGetFinalRelativeResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramggetfinalreltvres, HYPRE_BOOMERAMGGETFINALRELTVRES)(
   hypre_F90_Obj *solver,
   double   *rel_resid_norm,
   HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetFinalRelativeResidualNorm(
                      (HYPRE_Solver) *solver,
                      (double *)      rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetVariant, HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetvariant, HYPRE_BOOMERAMGSETVARIANT)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *variant,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetVariant( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *variant ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetvariant, HYPRE_BOOMERAMGGETVARIANT)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *variant,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetVariant( (HYPRE_Solver) *solver,
                                              (HYPRE_Int *)         variant ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetoverlap, HYPRE_BOOMERAMGSETOVERLAP)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *overlap,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetOverlap( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *overlap ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetoverlap, HYPRE_BOOMERAMGGETOVERLAP)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *overlap,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetOverlap( (HYPRE_Solver) *solver,
                                              (HYPRE_Int *)         overlap ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDomainType, HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdomaintype, HYPRE_BOOMERAMGSETDOMAINTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *domain_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetDomainType( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *domain_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetdomaintype, HYPRE_BOOMERAMGGETDOMAINTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *domain_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetDomainType( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int *)        domain_type ) );
}

void
hypre_F90_IFACE(hypre_boomeramgsetschwarznonsym, HYPRE_BOOMERAMGSETSCHWARZNONSYM)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *schwarz_non_symm,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSchwarzUseNonSymm( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)        *schwarz_non_symm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSchwarzRlxWt, HYPRE_BoomerAMGGetSchwarzRlxWt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetschwarzrlxwt, HYPRE_BOOMERAMGSETSCHWARZRLXWT)(
   hypre_F90_Obj *solver,
   double   *schwarz_rlx_weight,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSchwarzRlxWeight(
                      (HYPRE_Solver) *solver,
                      (double)       *schwarz_rlx_weight) );
}

void
hypre_F90_IFACE(hypre_boomeramggetschwarzrlxwt, HYPRE_BOOMERAMGGETSCHWARZRLXWT)(
   hypre_F90_Obj *solver,
   double   *schwarz_rlx_weight,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetSchwarzRlxWeight(
                      (HYPRE_Solver) *solver,
                      (double *)      schwarz_rlx_weight) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetsym, HYPRE_BOOMERAMGSETSYM)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *sym,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetSym( (HYPRE_Solver) *solver,
                                          (HYPRE_Int)          *sym ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetlevel, HYPRE_BOOMERAMGSETLEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *level,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetLevel( (HYPRE_Solver) *solver,
                                            (HYPRE_Int)          *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetthreshold, HYPRE_BOOMERAMGSETTHRESHOLD)(
   hypre_F90_Obj *solver,
   double   *threshold,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetThreshold( (HYPRE_Solver) *solver,
                                                (double)       *threshold) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetfilter, HYPRE_BOOMERAMGSETFILTER)(
   hypre_F90_Obj *solver,
   double   *filter,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetFilter( (HYPRE_Solver) *solver,
                                             (double)       *filter) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdroptol, HYPRE_BOOMERAMGSETDROPTOL)(
   hypre_F90_Obj *solver,
   double   *drop_tol,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetDropTol( (HYPRE_Solver) *solver,
                                              (double)       *drop_tol) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetmaxnzperrow, HYPRE_BOOMERAMGSETMAXNZPERROW)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *max_nz_per_row,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetMaxNzPerRow(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *max_nz_per_row ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteubj, HYPRE_BOOMERAMGSETEUBJ)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *eu_bj,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetEuBJ( (HYPRE_Solver) *solver,
                                           (HYPRE_Int)          *eu_bj ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteulevel, HYPRE_BOOMERAMGSETEULEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *eu_level,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetEuLevel( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *eu_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteusparsea, HYPRE_BOOMERAMGSETEUSPARSEA)(
   hypre_F90_Obj *solver,
   double   *eu_sparse_a,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetEuSparseA( (HYPRE_Solver) *solver,
                                              (double)       *eu_sparse_a) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgseteuclidfile, HYPRE_BOOMERAMGSETEUCLIDFILE)(
   hypre_F90_Obj *solver,
   char     *euclidfile,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetEuclidFile( (HYPRE_Solver) *solver,
                                                 (char *)        euclidfile ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumFunctions, HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnumfunctions, HYPRE_BOOMERAMGSETNUMFUNCTIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_functions,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNumFunctions(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *num_functions ) );
}

void
hypre_F90_IFACE(hypre_boomeramggetnumfunctions, HYPRE_BOOMERAMGGETNUMFUNCTIONS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_functions,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGGetNumFunctions(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         num_functions ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetnodal, HYPRE_BOOMERAMGSETNODAL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *nodal,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNodal( (HYPRE_Solver) *solver,
                                            (HYPRE_Int)          *nodal ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetdoffunc, HYPRE_BOOMERAMGSETDOFFUNC)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *dof_func,
   HYPRE_Int      *ierr             )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetDofFunc(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int *)         dof_func ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumpaths, HYPRE_BOOMERAMGSETNUMPATHS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *num_paths,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNumPaths( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *num_paths ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggnumlevels, HYPRE_BOOMERAMGSETAGGNUMLEVELS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *agg_num_levels,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggNumLevels(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *agg_num_levels ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/


void
hypre_F90_IFACE(hypre_boomeramgsetagginterptype, HYPRE_BOOMERAMGSETAGGINTERPTYPE)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *agg_interp_type,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggInterpType(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggtruncfactor, HYPRE_BOOMERAMGSETAGGTRUNCFACTOR)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggTruncFactor( (HYPRE_Solver) *solver,
                                                     (double)       *trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_boomeramgsetaggptruncftr, HYPRE_BOOMERAMGSETAGGPTRUNCFTR)(
   hypre_F90_Obj *solver,
   double   *trunc_factor,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggP12TruncFactor( (HYPRE_Solver) *solver,
                                                        (double)       *trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggpmaxelmts, HYPRE_BOOMERAMGSETAGGPMAXELMTS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *p_max_elmts,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggPMaxElmts( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *p_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetaggp12maxelmts, HYPRE_BOOMERAMGSETAGGP12MAXELMTS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *p_max_elmts,
   HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetAggP12MaxElmts( (HYPRE_Solver) *solver,
                                                     (HYPRE_Int)          *p_max_elmts ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetgsmg, HYPRE_BOOMERAMGSETGSMG)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *gsmg,
   HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetGSMG(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *gsmg ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_boomeramgsetnumsamples, HYPRE_BOOMERAMGSETNUMSAMPLES)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *gsmg,
   HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetNumSamples(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *gsmg ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/
                                                                    
void
hypre_F90_IFACE(hypre_boomeramgsetcgcits, HYPRE_BOOMERAMGSETCGCITS)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *its,
   HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_BoomerAMGSetCGCIts(
                      (HYPRE_Solver) *solver,
                      (HYPRE_Int)          *its ) );
}
