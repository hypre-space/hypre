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
 * HYPRE_ParCSRParaSailsSetFilter
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

/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetSym
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
/*--------------------------------------------------------------------------
 * HYPRE_ParaSailsSetLogging
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
