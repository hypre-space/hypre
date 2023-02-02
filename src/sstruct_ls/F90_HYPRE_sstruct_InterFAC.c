/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaccreate, HYPRE_SSTRUCTFACCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacdestroy2, HYPRE_SSTRUCTFACDESTROY2)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACDestroy2(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACAMR_RAP
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacamrrap, HYPRE_SSTRUCTFACAMRRAP)
(hypre_F90_Obj *A,
 HYPRE_Int (*rfactors)[HYPRE_MAXDIM],
 hypre_F90_Obj *facA,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACAMR_RAP(
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                rfactors,
                hypre_F90_PassObjRef (HYPRE_SStructMatrix, facA) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetup2, HYPRE_SSTRUCTFACSETUP2)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetup2(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (HYPRE_SStructVector, b),
                hypre_F90_PassObj (HYPRE_SStructVector, x) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsolve3, HYPRE_SSTRUCTFACSOLVE3)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSolve3(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (HYPRE_SStructVector, b),
                hypre_F90_PassObj (HYPRE_SStructVector, x)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsettol, HYPRE_SSTRUCTFACSETTOL)
(hypre_F90_Obj *solver,
 hypre_F90_Real *tol,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetTol(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassReal (tol) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetplevels, HYPRE_SSTRUCTFACSETPLEVELS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *nparts,
 hypre_F90_IntArray *plevels,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetPLevels(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (nparts),
                hypre_F90_PassIntArray (plevels)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroCFSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerocfsten, HYPRE_SSTRUCTFACZEROCFSTEN)
(hypre_F90_Obj *A,
 hypre_F90_Obj *grid,
 hypre_F90_Int *part,
 HYPRE_Int (*rfactors)[HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACZeroCFSten(
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (HYPRE_SStructGrid, grid),
                hypre_F90_PassInt (part),
                rfactors[HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroFCSten
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczerofcsten, HYPRE_SSTRUCTFACZEROFCSTEN)
(hypre_F90_Obj *A,
 hypre_F90_Obj *grid,
 hypre_F90_Int *part,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACZeroFCSten(
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                hypre_F90_PassObj (HYPRE_SStructGrid, grid),
                hypre_F90_PassInt (part) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRMatrixData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrmatrixdata, HYPRE_SSTRUCTFACZEROAMRMATRIXDATA)
(hypre_F90_Obj *A,
 hypre_F90_Int *part_crse,
 HYPRE_Int (*rfactors)[HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACZeroAMRMatrixData(
                hypre_F90_PassObj (HYPRE_SStructMatrix, A),
                hypre_F90_PassInt (part_crse),
                rfactors[HYPRE_MAXDIM] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACZeroAMRVectorData
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaczeroamrvectordata, HYPRE_SSTRUCTFACZEROAMRVECTORDATA)
(hypre_F90_Obj *b,
 hypre_F90_IntArray *plevels,
 HYPRE_Int (*rfactors)[HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACZeroAMRVectorData(
                hypre_F90_PassObj (HYPRE_SStructVector, b),
                hypre_F90_PassIntArray (plevels),
                rfactors ));
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetprefinements, HYPRE_SSTRUCTFACSETPREFINEMENTS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *nparts,
 HYPRE_Int (*rfactors)[HYPRE_MAXDIM],
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetPRefinements(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (nparts),
                rfactors ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxlevels, HYPRE_SSTRUCTFACSETMAXLEVELS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_levels,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetMaxLevels(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (max_levels) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxiter, HYPRE_SSTRUCTFACSETMAXITER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *max_iter,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetMaxIter(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (max_iter) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelchange, HYPRE_SSTRUCTFACSETRELCHANGE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *rel_change,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetRelChange(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (rel_change) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetzeroguess, HYPRE_SSTRUCTFACSETZEROGUESS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetZeroGuess(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnonzeroguess, HYPRE_SSTRUCTFACSETNONZEROGUESS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetNonZeroGuess(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelaxtype, HYPRE_SSTRUCTFACSETRELAXTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetRelaxType(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassInt (relax_type) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_sstructfacsetjacobiweigh, HYPRE_SSTRUCTFACSETJACOBIWEIGH)
(hypre_F90_Obj *solver,
 hypre_F90_Real *weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructFACSetJacobiWeight( hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                                             hypre_F90_PassReal (weight) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumprerelax, HYPRE_SSTRUCTFACSETNUMPRERELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_pre_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACSetNumPreRelax( hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                                             hypre_F90_PassInt (num_pre_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumpostrelax, HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_post_relax,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructFACSetNumPostRelax(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (num_post_relax) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetcoarsesolver, HYPRE_SSTRUCTFACSETCOARSESOLVER)
(hypre_F90_Obj *solver,
 hypre_F90_Int *csolver_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructFACSetCoarseSolverType(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (csolver_type)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetlogging, HYPRE_SSTRUCTFACSETLOGGING)
(hypre_F90_Obj *solver,
 hypre_F90_Int *logging,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (HYPRE_SStructFACSetLogging(
               hypre_F90_PassObj (HYPRE_SStructSolver, solver),
               hypre_F90_PassInt (logging) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetnumiteration, HYPRE_SSTRUCTFACGETNUMITERATION)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_iterations,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACGetNumIterations(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassIntRef (num_iterations)));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetfinalrelativ, HYPRE_SSTRUCTFACGETFINALRELATIV)
(hypre_F90_Obj *solver,
 hypre_F90_Real *norm,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructFACGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_SStructSolver, solver),
                hypre_F90_PassRealRef (norm) ));
}

#ifdef __cplusplus
}
#endif
