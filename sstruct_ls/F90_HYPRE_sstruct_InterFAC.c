/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/


/******************************************************************************
 *
 * HYPRE_SStructFAC Routines
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"


/*--------------------------------------------------------------------------
 * HYPRE_SStructFACCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfaccreate, HYPRE_SSTRUCTFACCREATE)
               (int *comm, long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACCreate( (MPI_Comm)             *comm,
                                           (HYPRE_SStructSolver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACDestroy2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacdestroy2, HYPRE_SSTRUCTFACDESTROY2)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACDestroy2( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetup2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetup2, HYPRE_SSTRUCTFACSETUP2)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetup2( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix)  *A,
                                           (HYPRE_SStructVector)  *b,
                                           (HYPRE_SStructVector)  *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSolve3
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsolve3, HYPRE_SSTRUCTFACSOLVE3)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSolve3( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsettol, HYPRE_SSTRUCTFACSETTOL)
               (long int *solver, double *tol, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetTol( (HYPRE_SStructSolver) *solver,
                                           (double)              *tol ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetplevels, HYPRE_SSTRUCTFACSETPLEVELS)
               (long int *solver, int *nparts, int *plevels, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetPLevels( (HYPRE_SStructSolver) *solver,
                                               (int)                 *nparts,
                                               (int *)                plevels));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetPRefinements
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetprefinements, HYPRE_SSTRUCTFACSETPREFINEMENTS)
               (long int *solver, int *nparts, int (*rfactors)[3], int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetPRefinements( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *nparts,
                                                    (int)                 (*rfactors)[3] ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxlevels, HYPRE_SSTRUCTFACSETMAXLEVELS)
               (long int *solver, int *max_levels, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetMaxLevels( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *max_levels ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetmaxiter, HYPRE_SSTRUCTFACSETMAXITER)
               (long int *solver, int *max_iter, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetMaxIter( (HYPRE_SStructSolver) *solver,
                                               (int)                 *max_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelchange, HYPRE_SSTRUCTFACSETRELCHANGE)
               (long int *solver, int *rel_change, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetRelChange( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *rel_change ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetzeroguess, HYPRE_SSTRUCTFACSETZEROGUESS)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnonzeroguess, HYPRE_SSTRUCTFACSETNONZEROGUESS)
               (long int *solver, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetrelaxtype, HYPRE_SSTRUCTFACSETRELAXTYPE)
               (long int *solver, int *relax_type, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetRelaxType( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumprerelax, HYPRE_SSTRUCTFACSETNUMPRERELAX)
               (long int *solver, int *num_pre_relax, int *ierr)
{
   *ierr = (int) ( HYPRE_SStructFACSetNumPreRelax( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetnumpostrelax, HYPRE_SSTRUCTFACSETNUMPOSTRELAX)
               (long int *solver, int *num_post_relax, int *ierr)
{
   *ierr = (int) (HYPRE_SStructFACSetNumPostRelax((HYPRE_SStructSolver) *solver,
                                                  (int)                  *num_post_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetcoarsesolver, HYPRE_SSTRUCTFACSETCOARSESOLVER)
               (long int *solver, int * csolver_type, int *ierr)
{
   *ierr = (int) 
           (HYPRE_SStructFACSetCoarseSolverType( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *csolver_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacsetlogging, HYPRE_SSTRUCTFACSETLOGGING)
               (long int *solver, int *logging, int *ierr)
{
   *ierr = (int) (HYPRE_SStructFACSetLogging( (HYPRE_SStructSolver) *solver,
                                              (int)                 *logging ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetnumiteration, HYPRE_SSTRUCTFACGETNUMITERATION)
               (long int *solver, int *num_iterations, int *ierr)
{
   *ierr = (int)  
           ( HYPRE_SStructFACGetNumIterations( (HYPRE_SStructSolver) *solver,
                                               (int *)                num_iterations));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructfacgetfinalrelativ, HYPRE_SSTRUCTFACGETFINALRELATIV)
               (long int *solver, double *norm, int *ierr)
{
   *ierr = (int) 
           ( HYPRE_SStructFACGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                           (double *)             norm ));
}
