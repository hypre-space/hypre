/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParaSails Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailscreate, HYPRE_PARASAILSCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailsdestroy, HYPRE_PARASAILSDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsDestroy(
                hypre_F90_PassObj (HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetup, HYPRE_PARASAILSSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetup(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssolve, HYPRE_PARASAILSSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSolve(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (HYPRE_ParVector, b),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetParams
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetparams, HYPRE_PARASAILSSETPARAMS)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetParams(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (thresh),
                hypre_F90_PassInt (nlevels) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetThresh,  HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetthresh, HYPRE_PARASAILSSETTHRESH)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetThresh(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (thresh) ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetthresh, HYPRE_PARASAILSGETTHRESH)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetThresh(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (thresh) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetNlevels,  HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetnlevels, HYPRE_PARASAILSSETNLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetNlevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (nlevels)) );
}


void
hypre_F90_IFACE(hypre_parasailsgetnlevels, HYPRE_PARASAILSGETNLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetNlevels(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (nlevels)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetFilter, HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetfilter, HYPRE_PARASAILSSETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetFilter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (filter)  ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetfilter, HYPRE_PARASAILSGETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetFilter(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (filter)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym, HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetsym, HYPRE_PARASAILSSETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetSym(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (sym)     ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetsym, HYPRE_PARASAILSGETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetSym(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (sym)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLoadbal, HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetloadbal, HYPRE_PARASAILSSETLOADBAL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *loadbal,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetLoadbal(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassReal (loadbal) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetloadbal, HYPRE_PARASAILSGETLOADBAL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *loadbal,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetLoadbal(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassRealRef (loadbal) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetReuse, HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetreuse, HYPRE_PARASAILSSETREUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *reuse,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetReuse(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (reuse) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetreuse, HYPRE_PARASAILSGETREUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *reuse,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetReuse(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (reuse) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging, HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetlogging, HYPRE_PARASAILSSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsSetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetlogging, HYPRE_PARASAILSGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParaSailsGetLogging(
                hypre_F90_PassObj (HYPRE_Solver, solver),
                hypre_F90_PassIntRef (logging) ) );
}

#ifdef __cplusplus
}
#endif
