/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutinitialize)( int      *comm,
                                              long int *solver,
                                              int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPilutInitialize( (MPI_Comm)       *comm,
                                                (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutFinalize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpilutfinalize)( long int *solver,
                                            int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPilutFinalize( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetup)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetup( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsolve)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSolve( (HYPRE_Solver)       *solver, 
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetmaxiter)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetMaxIter( (HYPRE_Solver) *solver, 
                                                (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTol
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetdroptol)( long int *solver,
                                              double   *tol,
                                              int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetDropTolerance( (HYPRE_Solver) *solver, 
                                                      (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFacRowSize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpilutsetfacrowsize)( long int *solver,
                                                 int      *size,
                                                 int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPilutSetFactorRowSize( (HYPRE_Solver) *solver,
                                                      (int)          *size    ) );
}

