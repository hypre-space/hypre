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
 * HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridcreate, HYPRE_STRUCTHYBRIDCREATE)( int      *comm,
                                               long int *solver,
                                               int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridCreate( (MPI_Comm)             *comm,
                                             (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybriddestroy, HYPRE_STRUCTHYBRIDDESTROY)( long int *solver,
                                             int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetup, HYPRE_STRUCTHYBRIDSETUP)( long int *solver,
                                          long int *A,
                                          long int *b,
                                          long int *x,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSetup( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsolve, HYPRE_STRUCTHYBRIDSOLVE)( long int *solver,
                                          long int *A,
                                          long int *b,
                                          long int *x,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSolve( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettol, HYPRE_STRUCTHYBRIDSETTOL)( long int *solver,
                                           double   *tol,
                                           int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructHybridSetTol( (HYPRE_StructSolver) *solver,
                                             (double)             *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetconvergenc, HYPRE_STRUCTHYBRIDSETCONVERGENC)( long int *solver,
                                                  double   *cf_tol,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetConvergenceTol( (HYPRE_StructSolver) *solver,
                                             (double)             *cf_tol  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetdscgmaxite, HYPRE_STRUCTHYBRIDSETDSCGMAXITE)( long int *solver,
                                                  int      *dscg_max_its,
                                                  int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetDSCGMaxIter(
         (HYPRE_StructSolver) *solver,
         (int)                *dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgmaxiter, HYPRE_STRUCTHYBRIDSETPCGMAXITER)( long int *solver,
                                                  int      *pcg_max_its,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetPCGMaxIter( (HYPRE_StructSolver) *solver,
                                         (int)                *pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettwonorm, HYPRE_STRUCTHYBRIDSETTWONORM)( long int *solver,
                                               int      *two_norm,
                                               int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetTwoNorm( (HYPRE_StructSolver) *solver,
                                      (int)                *two_norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetrelchange, HYPRE_STRUCTHYBRIDSETRELCHANGE)( long int *solver,
                                                 int      *rel_change,
                                                 int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetRelChange(
         (HYPRE_StructSolver) *solver,
         (int)                *rel_change    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprecond, HYPRE_STRUCTHYBRIDSETPRECOND)( long int *solver,
                                               int      *precond_id,
                                               long int *precond_solver,
                                               int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetlogging, HYPRE_STRUCTHYBRIDSETLOGGING)( long int *solver,
                                               int      *logging,
                                               int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructHybridSetLogging(
         (HYPRE_StructSolver) *solver,
         (int)                *logging    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetnumiterati, HYPRE_STRUCTHYBRIDGETNUMITERATI)( long int *solver,
                                                  int      *num_its,
                                                  int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_its    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetdscgnumite, HYPRE_STRUCTHYBRIDGETDSCGNUMITE)( long int *solver,
                                                  int      *dscg_num_its,
                                                  int      *ierr         )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetDSCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetpcgnumiter, HYPRE_STRUCTHYBRIDGETPCGNUMITER)( long int *solver,
                                                  int      *pcg_num_its,
                                                  int      *ierr        )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetPCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetfinalrelat, HYPRE_STRUCTHYBRIDGETFINALRELAT)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructHybridGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm    ) );
}
