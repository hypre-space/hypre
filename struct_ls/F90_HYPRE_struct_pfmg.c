/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgcreate)( int      *comm,
                                             long int *solver,
                                             int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGCreate( (MPI_Comm)             *comm,
                                (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgdestroy)( long int *solver,
                                           int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsetup)( long int *solver,
                                        long int *A,
                                        long int *b,
                                        long int *x,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSetup( (HYPRE_StructSolver) *solver,
                                          (HYPRE_StructMatrix) *A,
                                          (HYPRE_StructVector) *b,
                                          (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsolve)( long int *solver,
                                        long int *A,
                                        long int *b,
                                        long int *x,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSolve( (HYPRE_StructSolver) *solver,
                                          (HYPRE_StructMatrix) *A,
                                          (HYPRE_StructVector) *b,
                                          (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsettol)( long int *solver,
                                         double   *tol,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSetTol( (HYPRE_StructSolver) *solver,
                                           (double)             *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxiter)( long int *solver,
                                             int      *max_iter,
                                             int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                    (int)                *max_iter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelchange)( long int *solver,
                                               int      *rel_change,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetRelChange( (HYPRE_StructSolver) *solver,
                                      (int)                *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetzeroguess)( long int *solver,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetnonzeroguess)( long int *solver,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelaxtype)( long int *solver,
                                               int      *relax_type,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetRelaxType( (HYPRE_StructSolver) *solver,
                                      (int)                *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumprerelax)( long int *solver,
                                                 int      *num_pre_relax,
                                                 int      *ierr          )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNumPreRelax(
         (HYPRE_StructSolver) *solver,
         (int)                *num_pre_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumpostrelax)( long int *solver,
                                                  int      *num_post_relax,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNumPostRelax(
         (HYPRE_StructSolver) *solver,
         (int)                *num_post_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetdxyz)( long int *solver,
                                          double   *dxyz,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSetDxyz( (HYPRE_StructSolver) *solver,
                                            (double *)           dxyz   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetlogging)( long int *solver,
                                             int      *logging,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetLogging( (HYPRE_StructSolver) *solver,
                                    (int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetnumiteration)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetfinalrelativ)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm   ) );
}
