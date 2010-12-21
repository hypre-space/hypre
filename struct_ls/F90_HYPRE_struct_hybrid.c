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





#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridcreate, HYPRE_STRUCTHYBRIDCREATE)( hypre_F90_Comm *comm,
                                               hypre_F90_Obj *solver,
                                               HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructHybridCreate( (MPI_Comm)             *comm,
                                             (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybriddestroy, HYPRE_STRUCTHYBRIDDESTROY)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructHybridDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetup, HYPRE_STRUCTHYBRIDSETUP)( hypre_F90_Obj *solver,
                                          hypre_F90_Obj *A,
                                          hypre_F90_Obj *b,
                                          hypre_F90_Obj *x,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructHybridSetup( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsolve, HYPRE_STRUCTHYBRIDSOLVE)( hypre_F90_Obj *solver,
                                          hypre_F90_Obj *A,
                                          hypre_F90_Obj *b,
                                          hypre_F90_Obj *x,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructHybridSolve( (HYPRE_StructSolver) *solver,
                                            (HYPRE_StructMatrix) *A,
                                            (HYPRE_StructVector) *b,
                                            (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettol, HYPRE_STRUCTHYBRIDSETTOL)( hypre_F90_Obj *solver,
                                           double   *tol,
                                           HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructHybridSetTol( (HYPRE_StructSolver) *solver,
                                             (double)             *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetconvergenc, HYPRE_STRUCTHYBRIDSETCONVERGENC)( hypre_F90_Obj *solver,
                                                  double   *cf_tol,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetConvergenceTol( (HYPRE_StructSolver) *solver,
                                             (double)             *cf_tol  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetdscgmaxite, HYPRE_STRUCTHYBRIDSETDSCGMAXITE)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *dscg_max_its,
                                                  HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetDSCGMaxIter(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int)                *dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgmaxiter, HYPRE_STRUCTHYBRIDSETPCGMAXITER)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *pcg_max_its,
                                                  HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetPCGMaxIter( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int)                *pcg_max_its ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgabsolut, HYPRE_STRUCTHYBRIDSETPCGABSOLUT)
                                                ( hypre_F90_Obj *solver,
                                                  double   *pcg_atolf,
                                                  HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetPCGAbsoluteTolFactor( (HYPRE_StructSolver) *solver,
                                                   (double)             *pcg_atolf ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettwonorm, HYPRE_STRUCTHYBRIDSETTWONORM)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *two_norm,
                                               HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetTwoNorm( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)                *two_norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetstopcrit, HYPRE_STRUCTHYBRIDSETSTOPCRIT)
                                             ( hypre_F90_Obj *solver,
                                               HYPRE_Int      *stop_crit,
                                               HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetStopCrit( (HYPRE_StructSolver) *solver,
                                       (HYPRE_Int)                *stop_crit   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetrelchange, HYPRE_STRUCTHYBRIDSETRELCHANGE)
                                               ( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *rel_change,
                                                 HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int) 
           ( HYPRE_StructHybridSetRelChange( (HYPRE_StructSolver) *solver,
                                             (HYPRE_Int)                *rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetsolvertype, HYPRE_STRUCTHYBRIDSETSOLVERTYPE)
                                             ( hypre_F90_Obj *solver,
                                               HYPRE_Int      *solver_type,
                                               HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetSolverType( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int)                *solver_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetkdim, HYPRE_STRUCTHYBRIDSETKDIM)
                                               ( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *k_dim,
                                                 HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) (HYPRE_StructHybridSetKDim( (HYPRE_StructSolver) *solver,
                                             (HYPRE_Int)                *k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprecond, HYPRE_STRUCTHYBRIDSETPRECOND)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *precond_id,
                                               hypre_F90_Obj *precond_solver,
                                               HYPRE_Int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructHybridSetPrecond( (HYPRE_StructSolver) *solver,
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
hypre_F90_IFACE(hypre_structhybridsetlogging, HYPRE_STRUCTHYBRIDSETLOGGING)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *logging,
                                               HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetLogging(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int)                *logging    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprintlevel, HYPRE_STRUCTHYBRIDSETPRINTLEVEL)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *print_level,
                                               HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridSetPrintLevel(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int)                *print_level  ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetnumiterati, HYPRE_STRUCTHYBRIDGETNUMITERATI)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_its,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              num_its    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetdscgnumite, HYPRE_STRUCTHYBRIDGETDSCGNUMITE)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *dscg_num_its,
                                                  HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridGetDSCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetpcgnumiter, HYPRE_STRUCTHYBRIDGETPCGNUMITER)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *pcg_num_its,
                                                  HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridGetPCGNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetfinalrelat, HYPRE_STRUCTHYBRIDGETFINALRELAT)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructHybridGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm    ) );
}
