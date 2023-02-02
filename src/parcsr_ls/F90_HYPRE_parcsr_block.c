/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_BlockTridiag Fortran interface
 *
 *****************************************************************************/

#include "block_tridiag.h"
#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagcreate, HYPRE_BLOCKTRIDIAGCREATE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagCreate(
              hypre_F90_PassObjRef (HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, HYPRE_BLOCKTRIDIAGDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagDestroy(
              hypre_F90_PassObj (HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, HYPRE_BLOCKTRIDIAGSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetup(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (HYPRE_ParVector, b),
              hypre_F90_PassObj (HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, HYPRE_BLOCKTRIDIAGSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSolve(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
              hypre_F90_PassObj (HYPRE_ParVector, b),
              hypre_F90_PassObj (HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, HYPRE_BLOCKTRIDIAGSETINDEXSET)
(hypre_F90_Obj *solver,
 hypre_F90_Int *n,
 hypre_F90_IntArray *inds,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetIndexSet(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassInt (n),
              hypre_F90_PassIntArray (inds));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *thresh,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGStrengthThreshold(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassReal (thresh));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_sweeps,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGNumSweeps(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassInt (num_sweeps));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *relax_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGRelaxType(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassInt (relax_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
(hypre_F90_Obj *solver,
 hypre_F90_Int *print_level,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetPrintLevel(
              hypre_F90_PassObj (HYPRE_Solver, solver),
              hypre_F90_PassInt (print_level));
}

#ifdef __cplusplus
}
#endif
