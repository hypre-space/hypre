/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridcreate, HYPRE_STRUCTHYBRIDCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybriddestroy, HYPRE_STRUCTHYBRIDDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridDestroy(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetup, HYPRE_STRUCTHYBRIDSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetup(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsolve, HYPRE_STRUCTHYBRIDSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSolve(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettol, HYPRE_STRUCTHYBRIDSETTOL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (tol)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetconvergenc, HYPRE_STRUCTHYBRIDSETCONVERGENC)
( hypre_F90_Obj *solver,
  hypre_F90_Real *cf_tol,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetConvergenceTol(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (cf_tol)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetdscgmaxite, HYPRE_STRUCTHYBRIDSETDSCGMAXITE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dscg_max_its,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetDSCGMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (dscg_max_its) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgmaxiter, HYPRE_STRUCTHYBRIDSETPCGMAXITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *pcg_max_its,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetPCGMaxIter(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (pcg_max_its) ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetpcgabsolut, HYPRE_STRUCTHYBRIDSETPCGABSOLUT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *pcg_atolf,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetPCGAbsoluteTolFactor(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassReal (pcg_atolf) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsettwonorm, HYPRE_STRUCTHYBRIDSETTWONORM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *two_norm,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetTwoNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (two_norm)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetstopcrit, HYPRE_STRUCTHYBRIDSETSTOPCRIT)
( hypre_F90_Obj *solver,
  hypre_F90_Int *stop_crit,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetStopCrit(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (stop_crit)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetrelchange, HYPRE_STRUCTHYBRIDSETRELCHANGE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *rel_change,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetRelChange(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (rel_change)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetSolverType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetsolvertype, HYPRE_STRUCTHYBRIDSETSOLVERTYPE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *solver_type,
  hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetSolverType(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (solver_type) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetkdim, HYPRE_STRUCTHYBRIDSETKDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *k_dim,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           (HYPRE_StructHybridSetKDim(
               hypre_F90_PassObj (HYPRE_StructSolver, solver),
               hypre_F90_PassInt (k_dim) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprecond, HYPRE_STRUCTHYBRIDSETPRECOND)
( hypre_F90_Obj *solver,
  hypre_F90_Int *precond_id,
  hypre_F90_Obj *precond_solver,
  hypre_F90_Int *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 7 - setup a jacobi preconditioner
    * 8 - setup a ds preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructSMGSolve,
                   HYPRE_StructSMGSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructPFMGSolve,
                   HYPRE_StructPFMGSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructJacobiSolve,
                   HYPRE_StructJacobiSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (hypre_F90_Int)
              ( HYPRE_StructHybridSetPrecond(
                   hypre_F90_PassObj (HYPRE_StructSolver, solver),
                   HYPRE_StructDiagScale,
                   HYPRE_StructDiagScaleSetup,
                   hypre_F90_PassObj (HYPRE_StructSolver, precond_solver)) );
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
hypre_F90_IFACE(hypre_structhybridsetlogging, HYPRE_STRUCTHYBRIDSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetLogging(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (logging)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridsetprintlevel, HYPRE_STRUCTHYBRIDSETPRINTLEVEL)
( hypre_F90_Obj *solver,
  hypre_F90_Int *print_level,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridSetPrintLevel(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (print_level)  ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetnumiterati, HYPRE_STRUCTHYBRIDGETNUMITERATI)
( hypre_F90_Obj *solver,
  hypre_F90_Int *num_its,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridGetNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (num_its)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetdscgnumite, HYPRE_STRUCTHYBRIDGETDSCGNUMITE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *dscg_num_its,
  hypre_F90_Int *ierr         )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridGetDSCGNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (dscg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetpcgnumiter, HYPRE_STRUCTHYBRIDGETPCGNUMITER)
( hypre_F90_Obj *solver,
  hypre_F90_Int *pcg_num_its,
  hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridGetPCGNumIterations(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassIntRef (pcg_num_its) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structhybridgetfinalrelat, HYPRE_STRUCTHYBRIDGETFINALRELAT)
( hypre_F90_Obj *solver,
  hypre_F90_Real *norm,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructHybridGetFinalRelativeResidualNorm(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassRealRef (norm)    ) );
}

#ifdef __cplusplus
}
#endif
