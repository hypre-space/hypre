/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ILU Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilucreate, HYPRE_ILUCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_iludestroy, HYPRE_ILUDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetup, HYPRE_ILUSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusolve, HYPRE_ILUSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetprintlevel, HYPRE_ILUSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetlogging, HYPRE_ILUSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetmaxiter, HYPRE_ILUSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusettol, HYPRE_ILUSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetdropthreshold, HYPRE_ILUSETDROPTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetDropThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetdropthresholdarray, HYPRE_ILUSETDROPTHRESHOLDARRAY)
( hypre_F90_Obj *solver,
  hypre_F90_RealArray *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetDropThresholdArray(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealArray (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetnshdropthreshold, HYPRE_ILUSETNSHDROPTHRESHOLD)
( hypre_F90_Obj *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetNSHDropThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetschurmaxiter, HYPRE_ILUSETSCHURMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ss_max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (ss_max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetmaxnnzperrow, HYPRE_ILUSETMAXNNZPERROW)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nzmax,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetMaxNnzPerRow(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nzmax) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetleveloffill, HYPRE_ILUSETLEVELOFFILL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *lfil,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetMaxNnzPerRow(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (lfil) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusettype, HYPRE_ILUSETTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ilu_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (ilu_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilusetlocalreordering, HYPRE_ILUSETLOCALREORDERING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ordering_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUSetType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (ordering_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilugetnumiterations, HYPRE_ILUGETNUMITERATIONS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_iterations,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ILUGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ilugetfinalrelresnorm, HYPRE_ILUGETFINALRELRESNORM)
( hypre_F90_Obj *solver,
  hypre_F90_Real *res_norm,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ILUGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (res_norm) ) );
}


#ifdef __cplusplus
}
#endif
