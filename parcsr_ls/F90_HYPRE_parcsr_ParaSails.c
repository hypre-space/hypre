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
hypre_F90_IFACE(hypre_parcsrparasailscreate)( int      *comm,
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
hypre_F90_IFACE(hypre_parcsrparasailsdestroy)( long int *solver,
                                               int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRParaSailsDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRParaSailsSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrparasailssetup)( long int *solver,
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
hypre_F90_IFACE(hypre_parcsrparasailssolve)( long int *solver,
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
hypre_F90_IFACE(hypre_parcsrparasailssetparams)( long int *solver,
                                                 double   *thresh,
                                                 int      *nlevels,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRParaSailsSetParams( (HYPRE_Solver) *solver, 
                                                   (double)       *thresh,
                                                   (int)          *nlevels ) );
}
