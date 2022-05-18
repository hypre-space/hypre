/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_MGR Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrcreate, HYPRE_MGRCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdestroy, HYPRE_MGRDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetup, HYPRE_MGRSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsolve, HYPRE_MGRSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

#ifdef HYPRE_USING_DSUPERLU

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolvercreate, HYPRE_MGRDIRECTSOLVERCREATE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRDirectSolverCreate(
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolverdestroy, HYPRE_MGRDIRECTSOLVERDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRDirectSolverDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolversetup, HYPRE_MGRDIRECTSOLVERSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRDirectSolverSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRDirectSolverSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrdirectsolversolve, HYPRE_MGRDIRECTSOLVERSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRDirectSolverSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

#endif

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCptsByCtgBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcptsbyctgblock, HYPRE_MGRSETCPTSBYCTGBLOCK)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_BigIntArray   *idx_array,
  hypre_F90_IntArray      *block_num_coarse_points,
  hypre_F90_IntArrayArray *block_coarse_indexes,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetCpointsByContiguousBlock(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassBigIntArray (idx_array),
                hypre_F90_PassIntArray (block_num_coarse_points),
                hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCpointsByBlock
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcpointsbyblock, HYPRE_MGRSETCPOINTSBYBLOCK)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_IntArray      *block_num_coarse_points,
  hypre_F90_IntArrayArray *block_coarse_indexes,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetCpointsByBlock(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassIntArray (block_num_coarse_points),
                hypre_F90_PassIntArrayArray (block_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCptsByMarkerArray
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcptsbymarkerarray, HYPRE_MGRSETCPTSBYMARKERARRAY)
( hypre_F90_Obj           *solver,
  hypre_F90_Int           *block_size,
  hypre_F90_Int           *max_num_levels,
  hypre_F90_IntArray      *num_block_coarse_points,
  hypre_F90_IntArrayArray *lvl_block_coarse_indexes,
  hypre_F90_IntArray      *point_marker_array,
  hypre_F90_Int           *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetCpointsByPointMarkerArray(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (block_size),
                hypre_F90_PassInt (max_num_levels),
                hypre_F90_PassIntArray (num_block_coarse_points),
                hypre_F90_PassIntArrayArray (lvl_block_coarse_indexes),
                hypre_F90_PassIntArray (point_marker_array) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNonCptsToFpts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnoncptstofpts, HYPRE_MGRSETNONCPTSTOFPTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nonCptToFptFlag,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetNonCpointsToFpoints(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nonCptToFptFlag) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfsolver, HYPRE_MGRSETFSOLVER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *fsolver_id,
  hypre_F90_Obj *fsolver,
  hypre_F90_Int *ierr )
{
   /*------------------------------------------------------------
    * The fsolver_id flag means:
    *   0 - do not setup a F-solver.
    *   1 - BoomerAMG.
    *------------------------------------------------------------*/

   if (*fsolver_id == 0)
   {
      *ierr = 0;
   }
   else if (*fsolver_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_MGRSetFSolver(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
                   (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver) * fsolver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRBuildAff
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrbuildaff, HYPRE_MGRBUILDAFF)
( hypre_F90_Obj      *A,
  hypre_F90_IntArray *CF_marker,
  hypre_F90_Int      *debug_flag,
  hypre_F90_Obj      *A_ff,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRBuildAff(
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassIntArray (CF_marker),
                hypre_F90_PassInt (debug_flag),
                hypre_F90_PassObjRef (HYPRE_ParCSRMatrix, A_ff) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcoarsesolver, HYPRE_MGRSETCOARSESOLVER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *csolver_id,
  hypre_F90_Obj *csolver,
  hypre_F90_Int *ierr )
{
   /*------------------------------------------------------------
    * The csolver_id flag means:
    *   0 - do not setup a coarse solver.
    *   1 - BoomerAMG.
    *------------------------------------------------------------*/

   if (*csolver_id == 0)
   {
      *ierr = 0;
   }
   else if (*csolver_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_MGRSetCoarseSolver(
                   hypre_F90_PassObj (HYPRE_Solver, solver),
                   (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
                   (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
                   (HYPRE_Solver) * csolver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxCoarseLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetmaxcoarselevels, HYPRE_MGRSETMAXCOARSELEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *maxlev,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetMaxCoarseLevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (maxlev) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetblocksize, HYPRE_MGRSETBLOCKSIZE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *bsize,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetBlockSize(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (bsize) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetreservedcoarsenodes, HYPRE_MGRSETRESERVEDCOARSENODES)
( hypre_F90_Obj         *solver,
  hypre_F90_Int         *reserved_coarse_size,
  hypre_F90_BigIntArray *reserved_coarse_indexes,
  hypre_F90_Int         *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetReservedCoarseNodes(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (reserved_coarse_size),
                hypre_F90_PassBigIntArray (reserved_coarse_indexes) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetReservedCptsLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetreservedcptslevel, HYPRE_MGRSETRESERVEDCPTSLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetReservedCpointsLevelToKeep(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetrestricttype, HYPRE_MGRSETRESTRICTTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *restrict_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetRestrictType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelRestrictType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelrestricttype, HYPRE_MGRSETLEVELRESTRICTTYPE)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *restrict_type,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetLevelRestrictType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (restrict_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFRelaxMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfrelaxmethod, HYPRE_MGRSETFRELAXMETHOD)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_method,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetFRelaxMethod(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelFRelaxMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelfrelaxmethod, HYPRE_MGRSETLEVELFRELAXMETHOD)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *relax_method,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetLevelFRelaxMethod(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (relax_method) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCoarseGridMethod
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcoarsegridmethod, HYPRE_MGRSETCOARSEGRIDMETHOD)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *cg_method,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetCoarseGridMethod(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (cg_method) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelFRelaxNumFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelfrelaxnumfunc, HYPRE_MGRSETLEVELFRELAXNUMFUNC)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *num_functions,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetLevelFRelaxNumFunctions(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (num_functions) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetrelaxtype, HYPRE_MGRSETRELAXTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *relax_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetRelaxType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (relax_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnumrelaxsweeps, HYPRE_MGRSETNUMRELAXSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetNumRelaxSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetinterptype, HYPRE_MGRSETINTERPTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *interpType,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetInterpType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (interpType) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLevelInterpType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlevelinterptype, HYPRE_MGRSETLEVELINTERPTYPE)
( hypre_F90_Obj      *solver,
  hypre_F90_IntArray *interpType,
  hypre_F90_Int      *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetLevelInterpType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntArray (interpType) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnuminterpsweeps, HYPRE_MGRSETNUMINTERPSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetNumInterpSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetNumRestrictSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetnumrestrictsweeps, HYPRE_MGRSETNUMRESTRICTSWEEPS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nsweeps,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetNumRestrictSweeps(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nsweeps) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCGridThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcgridthreshold, HYPRE_MGRSETCGRIDTHRESHOLD)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *threshold,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetTruncateCoarseGridThreshold(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (threshold) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetFrelaxPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetfrelaxprintlevel, HYPRE_MGRSETFRELAXPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetFrelaxPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetCgridPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetcgridprintlevel, HYPRE_MGRSETCGRIDPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetCoarseGridPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetprintlevel, HYPRE_MGRSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetPrintLevel(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (print_level) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetlogging, HYPRE_MGRSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetmaxiter, HYPRE_MGRSETMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetMaxIter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsettol, HYPRE_MGRSETTOL)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetTol(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (tol) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetMaxGlobalsmoothIt
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetmaxglobalsmoothit, HYPRE_MGRSETMAXGLOBALSMOOTHIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *max_iter,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetMaxGlobalSmoothIters(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (max_iter) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetglobalsmoothtype, HYPRE_MGRSETGLOBALSMOOTHTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *iter_type,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetGlobalSmoothType(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (iter_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRSetPMaxElmts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrsetpmaxelmts, HYPRE_MGRSETPMAXELMTS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *P_max_elmts,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRSetPMaxElmts(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (P_max_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetCoarseGridConvFac
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetcoarsegridconvfac, HYPRE_MGRGETCOARSEGRIDCONVFAC)
( hypre_F90_Obj  *solver,
  hypre_F90_Real *conv_factor,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRGetCoarseGridConvergenceFactor(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (conv_factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetnumiterations, HYPRE_MGRGETNUMITERATIONS)
( hypre_F90_Obj  *solver,
  hypre_F90_Int  *num_iterations,
  hypre_F90_Int  *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRGetNumIterations(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (num_iterations) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_MGRGetFinalRelResNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_mgrgetfinalrelresnorm, HYPRE_MGRGETFINALRELRESNORM)
( hypre_F90_Obj   *solver,
  hypre_F90_Real  *res_norm,
  hypre_F90_Int   *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_MGRGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (res_norm) ) );
}

#ifdef __cplusplus
}
#endif
