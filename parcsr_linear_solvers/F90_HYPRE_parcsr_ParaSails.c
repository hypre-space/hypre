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
 * HYPRE_ParCSRParaSails Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrparasailscreate, HYPRE_PARCSRPARASAILSCREATE)( int      *comm,
                                              long int *solver,
                                              int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRParaSailsCreate( (MPI_Comm)       *comm,
                                                (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrparasailsdestroy, HYPRE_PARCSRPARASAILSDESTROY)( long int *solver,
                                               int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRParaSailsDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrparasailssetup, HYPRE_PARCSRPARASAILSSETUP)( long int *solver,
                                             long int *A,
                                             long int *b,
                                             long int *x,
                                             int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSetup( (HYPRE_Solver)       *solver, 
                                               (HYPRE_ParCSRMatrix) *A,
                                               (HYPRE_ParVector)    *b,
                                               (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrparasailssolve, HYPRE_PARCSRPARASAILSSOLVE)( long int *solver,
                                             long int *A,
                                             long int *b,
                                             long int *x,
                                             int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSolve( (HYPRE_Solver)       *solver, 
                                               (HYPRE_ParCSRMatrix) *A,
                                               (HYPRE_ParVector)    *b,
                                               (HYPRE_ParVector)    *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrparasailssetparams, HYPRE_PARCSRPARASAILSSETPARAMS)( long int *solver,
                                                 double   *thresh,
                                                 int      *nlevels,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSetParams( (HYPRE_Solver) *solver, 
                                                   (double)       *thresh,
                                                   (int)          *nlevels ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrparasailssetfilter, HYPRE_PARCSRPARASAILSSETFILTER)( long int *solver,
                                                 double   *filter,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSetFilter( (HYPRE_Solver) *solver, 
                                                   (double)       *filter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrparasailssetsym, HYPRE_PARCSRPARASAILSSETSYM)( long int *solver,
                                              int      *sym,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSetSym( (HYPRE_Solver) *solver, 
                                                (int)          *sym     ) );
}
