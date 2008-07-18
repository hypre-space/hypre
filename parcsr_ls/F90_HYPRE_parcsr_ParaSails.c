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
                                               int      *comm,
                                               long int *solver,
                                               int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParaSailsCreate( (MPI_Comm)       *comm,
                                          (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailsdestroy, HYPRE_PARASAILSDESTROY)(
                                                long int *solver,
                                                int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParaSailsDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetup, HYPRE_PARASAILSSETUP)(
                                                long int *solver,
                                                long int *A,
                                                long int *b,
                                                long int *x,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetup( (HYPRE_Solver)       *solver, 
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssolve, HYPRE_PARASAILSSOLVE)(
                                                long int *solver,
                                                long int *A,
                                                long int *b,
                                                long int *x,
                                                int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSolve( (HYPRE_Solver)       *solver, 
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetparams, HYPRE_PARASAILSSETPARAMS)(
                                                 long int *solver,
                                                 double   *thresh,
                                                 int      *nlevels,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetParams( (HYPRE_Solver) *solver, 
                                             (double)       *thresh,
                                             (int)          *nlevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetThresh,  HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetthresh, HYPRE_PARASAILSSETTHRESH)(
                                                 long int *solver,
                                                 double   *thresh,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetThresh( (HYPRE_Solver) *solver, 
                                             (double)       *thresh ) );
}


void 
hypre_F90_IFACE(hypre_parasailsgetthresh, HYPRE_PARASAILSGETTHRESH)(
                                                 long int *solver,
                                                 double   *thresh,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetThresh( (HYPRE_Solver) *solver, 
                                             (double *)      thresh ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetNlevels,  HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parasailssetnlevels, HYPRE_PARASAILSSETNLEVELS)(
                                                 long int *solver,
                                                 int      *nlevels,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetNlevels( (HYPRE_Solver) *solver, 
                                              (int)          *nlevels) );
}


void 
hypre_F90_IFACE(hypre_parasailsgetnlevels, HYPRE_PARASAILSGETNLEVELS)(
                                                 long int *solver,
                                                 int      *nlevels,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetNlevels( (HYPRE_Solver) *solver, 
                                              (int *)         nlevels) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetFilter, HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetfilter, HYPRE_PARASAILSSETFILTER)(
                                                 long int *solver,
                                                 double   *filter,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetFilter( (HYPRE_Solver) *solver, 
                                             (double)       *filter  ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetfilter, HYPRE_PARASAILSGETFILTER)(
                                                 long int *solver,
                                                 double   *filter,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetFilter( (HYPRE_Solver) *solver, 
                                             (double *)      filter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym, HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetsym, HYPRE_PARASAILSSETSYM)(
                                              long int *solver,
                                              int      *sym,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetSym( (HYPRE_Solver) *solver, 
                                          (int)          *sym     ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetsym, HYPRE_PARASAILSGETSYM)(
                                              long int *solver,
                                              int      *sym,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetSym( (HYPRE_Solver) *solver, 
                                          (int *)         sym     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLoadbal, HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetloadbal, HYPRE_PARASAILSSETLOADBAL)(
                                              long int *solver,
                                              double   *loadbal,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetLoadbal( (HYPRE_Solver) *solver, 
                                              (double)       *loadbal ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetloadbal, HYPRE_PARASAILSGETLOADBAL)(
                                              long int *solver,
                                              double   *loadbal,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetLoadbal( (HYPRE_Solver) *solver, 
                                              (double *)      loadbal ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetReuse, HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetreuse, HYPRE_PARASAILSSETREUSE)(
                                              long int *solver,
                                              int      *reuse,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetReuse( (HYPRE_Solver) *solver, 
                                              (int)          *reuse ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetreuse, HYPRE_PARASAILSGETREUSE)(
                                              long int *solver,
                                              int      *reuse,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetReuse( (HYPRE_Solver) *solver, 
                                              (int *)         reuse ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging, HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetlogging, HYPRE_PARASAILSSETLOGGING)(
                                              long int *solver,
                                              int      *logging,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsSetLogging( (HYPRE_Solver) *solver, 
                                              (int)          *logging ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetlogging, HYPRE_PARASAILSGETLOGGING)(
                                              long int *solver,
                                              int      *logging,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParaSailsGetLogging( (HYPRE_Solver) *solver, 
                                              (int *)         logging ) );
}
