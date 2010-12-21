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
 * HYPRE_ParaSails Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailscreate, HYPRE_PARASAILSCREATE)(
                                               hypre_F90_Comm *comm,
                                               hypre_F90_Obj *solver,
                                               HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsCreate( (MPI_Comm)       *comm,
                                          (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailsdestroy, HYPRE_PARASAILSDESTROY)(
                                                hypre_F90_Obj *solver,
                                                HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetup, HYPRE_PARASAILSSETUP)(
                                                hypre_F90_Obj *solver,
                                                hypre_F90_Obj *A,
                                                hypre_F90_Obj *b,
                                                hypre_F90_Obj *x,
                                                HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetup( (HYPRE_Solver)       *solver, 
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssolve, HYPRE_PARASAILSSOLVE)(
                                                hypre_F90_Obj *solver,
                                                hypre_F90_Obj *A,
                                                hypre_F90_Obj *b,
                                                hypre_F90_Obj *x,
                                                HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSolve( (HYPRE_Solver)       *solver, 
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetparams, HYPRE_PARASAILSSETPARAMS)(
                                                 hypre_F90_Obj *solver,
                                                 double   *thresh,
                                                 HYPRE_Int      *nlevels,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetParams( (HYPRE_Solver) *solver, 
                                             (double)       *thresh,
                                             (HYPRE_Int)          *nlevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetThresh,  HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetthresh, HYPRE_PARASAILSSETTHRESH)(
                                                 hypre_F90_Obj *solver,
                                                 double   *thresh,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetThresh( (HYPRE_Solver) *solver, 
                                             (double)       *thresh ) );
}


void 
hypre_F90_IFACE(hypre_parasailsgetthresh, HYPRE_PARASAILSGETTHRESH)(
                                                 hypre_F90_Obj *solver,
                                                 double   *thresh,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetThresh( (HYPRE_Solver) *solver, 
                                             (double *)      thresh ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetNlevels,  HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetnlevels, HYPRE_PARASAILSSETNLEVELS)(
                                                 hypre_F90_Obj *solver,
                                                 HYPRE_Int      *nlevels,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetNlevels( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int)          *nlevels) );
}


void 
hypre_F90_IFACE(hypre_parasailsgetnlevels, HYPRE_PARASAILSGETNLEVELS)(
                                                 hypre_F90_Obj *solver,
                                                 HYPRE_Int      *nlevels,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetNlevels( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int *)         nlevels) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetFilter, HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetfilter, HYPRE_PARASAILSSETFILTER)(
                                                 hypre_F90_Obj *solver,
                                                 double   *filter,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetFilter( (HYPRE_Solver) *solver, 
                                             (double)       *filter  ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetfilter, HYPRE_PARASAILSGETFILTER)(
                                                 hypre_F90_Obj *solver,
                                                 double   *filter,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetFilter( (HYPRE_Solver) *solver, 
                                             (double *)      filter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym, HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetsym, HYPRE_PARASAILSSETSYM)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *sym,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetSym( (HYPRE_Solver) *solver, 
                                          (HYPRE_Int)          *sym     ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetsym, HYPRE_PARASAILSGETSYM)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *sym,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetSym( (HYPRE_Solver) *solver, 
                                          (HYPRE_Int *)         sym     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLoadbal, HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetloadbal, HYPRE_PARASAILSSETLOADBAL)(
                                              hypre_F90_Obj *solver,
                                              double   *loadbal,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetLoadbal( (HYPRE_Solver) *solver, 
                                              (double)       *loadbal ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetloadbal, HYPRE_PARASAILSGETLOADBAL)(
                                              hypre_F90_Obj *solver,
                                              double   *loadbal,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetLoadbal( (HYPRE_Solver) *solver, 
                                              (double *)      loadbal ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetReuse, HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetreuse, HYPRE_PARASAILSSETREUSE)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *reuse,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetReuse( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int)          *reuse ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetreuse, HYPRE_PARASAILSGETREUSE)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *reuse,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetReuse( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int *)         reuse ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging, HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetlogging, HYPRE_PARASAILSSETLOGGING)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsSetLogging( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int)          *logging ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetlogging, HYPRE_PARASAILSGETLOGGING)(
                                              hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParaSailsGetLogging( (HYPRE_Solver) *solver, 
                                              (HYPRE_Int *)         logging ) );
}
