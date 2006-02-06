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
hypre_F90_IFACE(hypre_structpfmgcreate, HYPRE_STRUCTPFMGCREATE)( int      *comm,
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
hypre_F90_IFACE(hypre_structpfmgdestroy, HYPRE_STRUCTPFMGDESTROY)( long int *solver,
                                           int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsetup, HYPRE_STRUCTPFMGSETUP)( long int *solver,
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
hypre_F90_IFACE(hypre_structpfmgsolve, HYPRE_STRUCTPFMGSOLVE)( long int *solver,
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
 * HYPRE_StructPFMGSetTol, HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsettol, HYPRE_STRUCTPFMGSETTOL)( long int *solver,
                                         double   *tol,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSetTol( (HYPRE_StructSolver) *solver,
                                           (double)             *tol    ) );
}

void
hypre_F90_IFACE(hypre_structpfmggettol, HYPRE_STRUCTPFMGGETTOL)( long int *solver,
                                         double   *tol,
                                         int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGGetTol( (HYPRE_StructSolver) *solver,
                                           (double)             *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter, HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxiter, HYPRE_STRUCTPFMGSETMAXITER)( long int *solver,
                                             int      *max_iter,
                                             int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                    (int)                *max_iter  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxiter, HYPRE_STRUCTPFMGGETMAXITER)( long int *solver,
                                             int      *max_iter,
                                             int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetMaxIter( (HYPRE_StructSolver) *solver,
                                    (int)                *max_iter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxLevels, HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxlevels, HYPRE_STRUCTPFMGSETMAXLEVELS)
                                           ( long int *solver,
                                             int      *max_levels,
                                             int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetMaxLevels( (HYPRE_StructSolver) *solver,
                                      (int)                *max_levels  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxlevels, HYPRE_STRUCTPFMGGETMAXLEVELS)( long int *solver,
                                             int      *max_levels,
                                             int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetMaxLevels( (HYPRE_StructSolver) *solver,
                                      (int)                *max_levels  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange, HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelchange, HYPRE_STRUCTPFMGSETRELCHANGE)( long int *solver,
                                               int      *rel_change,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetRelChange( (HYPRE_StructSolver) *solver,
                                      (int)                *rel_change  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelchange, HYPRE_STRUCTPFMGGETRELCHANGE)( long int *solver,
                                               int      *rel_change,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetRelChange( (HYPRE_StructSolver) *solver,
                                      (int)                *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess, HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetzeroguess, HYPRE_STRUCTPFMGSETZEROGUESS)( long int *solver,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}
 
void
hypre_F90_IFACE(hypre_structpfmggetzeroguess, HYPRE_STRUCTPFMGGETZEROGUESS)( long int *solver,
                                               int      *zeroguess,
                                               int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetZeroGuess( (HYPRE_StructSolver) *solver,
                                      (int)                *zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetnonzeroguess, HYPRE_STRUCTPFMGSETNONZEROGUESS)( long int *solver,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType, HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelaxtype, HYPRE_STRUCTPFMGSETRELAXTYPE)( long int *solver,
                                               int      *relax_type,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetRelaxType( (HYPRE_StructSolver) *solver,
                                      (int)                *relax_type ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelaxtype, HYPRE_STRUCTPFMGGETRELAXTYPE)( long int *solver,
                                               int      *relax_type,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetRelaxType( (HYPRE_StructSolver) *solver,
                                      (int)                *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRAPType, HYPRE_StructPFMGSetRapType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetraptype, HYPRE_STRUCTPFMGSETRAPTYPE)( long int *solver,
                                               int      *rap_type,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetRAPType( (HYPRE_StructSolver) *solver,
                                      (int)              *rap_type ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetraptype, HYPRE_STRUCTPFMGGETRAPTYPE)( long int *solver,
                                               int      *rap_type,
                                               int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetRAPType( (HYPRE_StructSolver) *solver,
                                      (int)              *rap_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax, HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumprerelax, HYPRE_STRUCTPFMGSETNUMPRERELAX)( long int *solver,
                                                 int      *num_pre_relax,
                                                 int      *ierr          )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                        (int)                *num_pre_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumprerelax, HYPRE_STRUCTPFMGGETNUMPRERELAX)
                                               ( long int *solver,
                                                 int      *num_pre_relax,
                                                 int      *ierr          )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetNumPreRelax( (HYPRE_StructSolver) *solver,
                                        (int)                *num_pre_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax, HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumpostrelax, HYPRE_STRUCTPFMGSETNUMPOSTRELAX)( long int *solver,
                                                  int      *num_post_relax,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                         (int)                *num_post_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumpostrelax, HYPRE_STRUCTPFMGGETNUMPOSTRELAX)( long int *solver,
                                                  int      *num_post_relax,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetNumPostRelax( (HYPRE_StructSolver) *solver,
                                         (int)                *num_post_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetSkipRelax, HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetskiprelax, HYPRE_STRUCTPFMGSETSKIPRELAX)( long int *solver,
                                                  int      *skip_relax,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetSkipRelax( (HYPRE_StructSolver) *solver,
                                         (int)                *skip_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetskiprelax, HYPRE_STRUCTPFMGGETSKIPRELAX)( long int *solver,
                                                  int      *skip_relax,
                                                  int      *ierr           )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetSkipRelax( (HYPRE_StructSolver) *solver,
                                         (int)                *skip_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz, HYPRE_StructPFMGGetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetdxyz, HYPRE_STRUCTPFMGSETDXYZ)( long int *solver,
                                          double   *dxyz,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGSetDxyz( (HYPRE_StructSolver) *solver,
                                            (double *)           dxyz   ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetdxyz, HYPRE_STRUCTPFMGGETDXYZ)( long int *solver,
                                          double   *dxyz,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructPFMGGetDxyz( (HYPRE_StructSolver) *solver,
                                            (double *)           dxyz   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging, HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetlogging, HYPRE_STRUCTPFMGSETLOGGING)( long int *solver,
                                             int      *logging,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetLogging( (HYPRE_StructSolver) *solver,
                                    (int)                *logging ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetlogging, HYPRE_STRUCTPFMGGETLOGGING)( long int *solver,
                                             int      *logging,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetLogging( (HYPRE_StructSolver) *solver,
                                    (int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetPrintLevel, HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetprintlevel, HYPRE_STRUCTPFMGSETPRINTLEVEL)( long int *solver,
                                             int      *print_level,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructPFMGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                        (int)             *print_level ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetprintlevel, HYPRE_STRUCTPFMGGETPRINTLEVEL)( long int *solver,
                                             int      *print_level,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetPrintLevel( (HYPRE_StructSolver) *solver,
                                        (int)             *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetnumiteration, HYPRE_STRUCTPFMGGETNUMITERATION)( long int *solver,
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
hypre_F90_IFACE(hypre_structpfmggetfinalrelativ, HYPRE_STRUCTPFMGGETFINALRELATIV)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructPFMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm   ) );
}
