/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmginitialize)( int      *comm,
                                            long int *solver,
                                            int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGInitialize( (MPI_Comm)             *comm,
                                              (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGFinalize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgfinalize)( long int *solver,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGFinalize( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsetup)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsolve)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse)( long int *solver,
                                              int      *memory_use,
                                              int      *ierr       )
{
   *ierr = (int) ( HYPRE_StructSMGSetMemoryUse( (HYPRE_StructSolver) *solver,
                                                (int)                *memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int) ( HYPRE_StructSMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                              (int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr       )
{
   *ierr = (int) ( HYPRE_StructSMGSetRelChange( (HYPRE_StructSolver) *solver,
                                                (int)                *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetzeroguess)( long int *solver,
                                              int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess)( long int *solver,
                                                 int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax)( long int *solver,
                                                int      *num_pre_relax,
                                                int      *ierr         )
{
   *ierr = (int) ( HYPRE_StructSMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                                  (int)                *num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax)( long int *solver,
                                                 int      *num_post_relax,
                                                 int      *ierr           )
{
   *ierr = (int) ( HYPRE_StructSMGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                                   (int)                *num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging)( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_StructSMGSetLogging( (HYPRE_StructSolver) *solver,
                                              (int)                *logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr           )
{
   *ierr = (int) ( HYPRE_StructSMGGetNumIterations( (HYPRE_StructSolver) *solver,
                                                    (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSMGGetFinalRelativeResidualNorm( (HYPRE_StructSolver) *solver,
                                                                (double *)           norm ) );
}
