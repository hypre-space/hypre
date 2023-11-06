/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * hypre_HybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   HYPRE_Real            tol;
   HYPRE_Real            cf_tol;
   HYPRE_Real            pcg_atolf;
   HYPRE_Int             dscg_max_its;
   HYPRE_Int             krylov_max_its;
   HYPRE_Int             two_norm;
   HYPRE_Int             stop_crit;
   HYPRE_Int             rel_change;
   HYPRE_Int             recompute_residual;
   HYPRE_Int             recompute_residual_p;
   HYPRE_Int             k_dim;
   HYPRE_Int             solver_type;

   HYPRE_Int             krylov_default;              /* boolean */
   HYPRE_Int           (*krylov_precond_solve)(void*, void*, void*, void*);
   HYPRE_Int           (*krylov_precond_setup)(void*, void*, void*, void*);
   void                 *krylov_precond;

   /* log info (always logged) */
   HYPRE_Int             dscg_num_its;
   HYPRE_Int             krylov_num_its;
   HYPRE_Real            final_rel_res_norm;
   HYPRE_Int             time_index;

   HYPRE_Int             print_level;
   /* additional information (place-holder currently used to print norms) */
   HYPRE_Int             logging;

} hypre_HybridData;

/*--------------------------------------------------------------------------
 * hypre_HybridCreate
 *--------------------------------------------------------------------------*/

void *
hypre_HybridCreate( MPI_Comm  comm )
{
   hypre_HybridData *hybrid_data;

   hybrid_data = hypre_CTAlloc(hypre_HybridData,  1, HYPRE_MEMORY_HOST);

   (hybrid_data -> comm)        = comm;
   (hybrid_data -> time_index)  = hypre_InitializeTiming("Hybrid");

   /* set defaults */
   (hybrid_data -> tol)               = 1.0e-06;
   (hybrid_data -> cf_tol)            = 0.90;
   (hybrid_data -> pcg_atolf)         = 0.0;
   (hybrid_data -> dscg_max_its)      = 1000;
   (hybrid_data -> krylov_max_its)    = 200;
   (hybrid_data -> two_norm)          = 0;
   (hybrid_data -> stop_crit)          = 0;
   (hybrid_data -> rel_change)        = 0;
   (hybrid_data -> solver_type)       = 1;
   (hybrid_data -> k_dim)             = 5;
   (hybrid_data -> krylov_default)       = 1;
   (hybrid_data -> krylov_precond_solve) = NULL;
   (hybrid_data -> krylov_precond_setup) = NULL;
   (hybrid_data -> krylov_precond)       = NULL;

   /* initialize */
   (hybrid_data -> dscg_num_its)      = 0;
   (hybrid_data -> krylov_num_its)    = 0;
   (hybrid_data -> logging)           = 0;
   (hybrid_data -> print_level)       = 0;

   return (void *) hybrid_data;
}

/*--------------------------------------------------------------------------
 * hypre_HybridDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridDestroy( void  *hybrid_vdata )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *) hybrid_vdata;

   if (hybrid_data)
   {
      hypre_TFree(hybrid_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetTol( void       *hybrid_vdata,
                    HYPRE_Real  tol       )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetConvergenceTol( void       *hybrid_vdata,
                               HYPRE_Real  cf_tol       )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> cf_tol) = cf_tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetDSCGMaxIter( void      *hybrid_vdata,
                            HYPRE_Int  dscg_max_its )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> dscg_max_its) = dscg_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPCGMaxIter( void      *hybrid_vdata,
                           HYPRE_Int  krylov_max_its  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> krylov_max_its) = krylov_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPCGAbsoluteTolFactor( void       *hybrid_vdata,
                                     HYPRE_Real  pcg_atolf  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> pcg_atolf) = pcg_atolf;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetTwoNorm( void      *hybrid_vdata,
                        HYPRE_Int  two_norm  )
{
   hypre_HybridData *hybrid_data = ( hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> two_norm) = two_norm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetStopCrit( void      *hybrid_vdata,
                         HYPRE_Int  stop_crit  )
{
   hypre_HybridData *hybrid_data = ( hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> stop_crit) = stop_crit;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetRelChange( void      *hybrid_vdata,
                          HYPRE_Int  rel_change  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetSolverType( void      *hybrid_vdata,
                           HYPRE_Int  solver_type  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> solver_type) = solver_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetRecomputeResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetRecomputeResidual( void      *hybrid_vdata,
                                  HYPRE_Int  recompute_residual )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> recompute_residual) = recompute_residual;

   return hypre_error_flag;
}

HYPRE_Int
hypre_HybridGetRecomputeResidual( void      *hybrid_vdata,
                                  HYPRE_Int *recompute_residual )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *recompute_residual = (hybrid_data -> recompute_residual);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetRecomputeResidualP
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetRecomputeResidualP( void      *hybrid_vdata,
                                   HYPRE_Int  recompute_residual_p )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> recompute_residual_p) = recompute_residual_p;

   return hypre_error_flag;
}

HYPRE_Int
hypre_HybridGetRecomputeResidualP( void      *hybrid_vdata,
                                   HYPRE_Int *recompute_residual_p )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *recompute_residual_p = (hybrid_data -> recompute_residual_p);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetKDim( void      *hybrid_vdata,
                     HYPRE_Int  k_dim  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> k_dim) = k_dim;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPrecond( void  *krylov_vdata,
                        HYPRE_Int  (*krylov_precond_solve)(void*, void*, void*, void*),
                        HYPRE_Int  (*krylov_precond_setup)(void*, void*, void*, void*),
                        void  *krylov_precond          )
{
   hypre_HybridData *krylov_data = (hypre_HybridData *)krylov_vdata;

   (krylov_data -> krylov_default)       = 0;
   (krylov_data -> krylov_precond_solve) = krylov_precond_solve;
   (krylov_data -> krylov_precond_setup) = krylov_precond_setup;
   (krylov_data -> krylov_precond)       = krylov_precond;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetLogging( void       *hybrid_vdata,
                        HYPRE_Int   logging  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> logging) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPrintLevel( void      *hybrid_vdata,
                           HYPRE_Int  print_level  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> print_level) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetNumIterations( void       *hybrid_vdata,
                              HYPRE_Int  *num_its      )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *num_its = (hybrid_data -> dscg_num_its) + (hybrid_data -> krylov_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetDSCGNumIterations( void       *hybrid_vdata,
                                  HYPRE_Int  *dscg_num_its )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *dscg_num_its = (hybrid_data -> dscg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetPCGNumIterations( void       *hybrid_vdata,
                                 HYPRE_Int  *krylov_num_its  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *krylov_num_its = (hybrid_data -> krylov_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetFinalRelativeResidualNorm( void        *hybrid_vdata,
                                          HYPRE_Real  *final_rel_res_norm )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *final_rel_res_norm = (hybrid_data -> final_rel_res_norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetup( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x )
{
   HYPRE_UNUSED_VAR(hybrid_vdata);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a hybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If sufficient
 * progress is not made, the algorithm switches to preconditioned
 * conjugate gradients with user-specified preconditioner.
 *
 *--------------------------------------------------------------------------*/

/* Local helper function for creating default PCG solver */
void *
hypre_HybridSolveUsePCG( hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   HYPRE_Real  tol            = (hybrid_data -> tol);
   HYPRE_Real  pcg_atolf      = (hybrid_data -> pcg_atolf);
   HYPRE_Int   two_norm       = (hybrid_data -> two_norm);
   HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   HYPRE_Int   rel_change     = (hybrid_data -> rel_change);
   HYPRE_Int   recompute_residual   = (hybrid_data -> recompute_residual);
   HYPRE_Int   recompute_residual_p = (hybrid_data -> recompute_residual_p);
   HYPRE_Int   logging        = (hybrid_data -> logging);
   HYPRE_Int   print_level    = (hybrid_data -> print_level);

   hypre_PCGFunctions  *pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
   krylov_solver = hypre_PCGCreate( pcg_functions );

   hypre_PCGSetTol(krylov_solver, tol);
   hypre_PCGSetAbsoluteTolFactor(krylov_solver, pcg_atolf);
   hypre_PCGSetTwoNorm(krylov_solver, two_norm);
   hypre_PCGSetStopCrit(krylov_solver, stop_crit);
   hypre_PCGSetRelChange(krylov_solver, rel_change);
   hypre_PCGSetRecomputeResidual(krylov_solver, recompute_residual);
   hypre_PCGSetRecomputeResidualP(krylov_solver, recompute_residual_p);
   hypre_PCGSetPrintLevel(krylov_solver, print_level);
   hypre_PCGSetLogging(krylov_solver, logging);

   return krylov_solver;
}

/* Local helper function for setting up GMRES */
void *
hypre_HybridSolveUseGMRES( hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   HYPRE_Real  tol            = (hybrid_data -> tol);
   HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   HYPRE_Int   rel_change     = (hybrid_data -> rel_change);
   HYPRE_Int   logging        = (hybrid_data -> logging);
   HYPRE_Int   print_level    = (hybrid_data -> print_level);
   HYPRE_Int   k_dim          = (hybrid_data -> k_dim);

   hypre_GMRESFunctions  *gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovCreateVectorArray,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
   krylov_solver = hypre_GMRESCreate( gmres_functions );

   hypre_GMRESSetTol(krylov_solver, tol);
   hypre_GMRESSetKDim(krylov_solver, k_dim);
   hypre_GMRESSetStopCrit(krylov_solver, stop_crit);
   hypre_GMRESSetRelChange(krylov_solver, rel_change);
   hypre_GMRESSetPrintLevel(krylov_solver, print_level);
   hypre_GMRESSetLogging(krylov_solver, logging);

   return krylov_solver;
}

/* Local helper function for setting up BiCGSTAB */
void *
hypre_HybridSolveUseBiCGSTAB( hypre_HybridData  *hybrid_data )
{
   void       *krylov_solver;
   HYPRE_Real  tol            = (hybrid_data -> tol);
   HYPRE_Int   stop_crit      = (hybrid_data -> stop_crit);
   HYPRE_Int   logging        = (hybrid_data -> logging);
   HYPRE_Int   print_level    = (hybrid_data -> print_level);

   hypre_BiCGSTABFunctions  *bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
   krylov_solver = hypre_BiCGSTABCreate( bicgstab_functions );

   hypre_BiCGSTABSetTol(krylov_solver, tol);
   hypre_BiCGSTABSetStopCrit(krylov_solver, stop_crit);
   hypre_BiCGSTABSetPrintLevel(krylov_solver, print_level);
   hypre_BiCGSTABSetLogging(krylov_solver, logging);

   return krylov_solver;
}

HYPRE_Int
hypre_HybridSolve( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_HybridData  *hybrid_data    = (hypre_HybridData *)hybrid_vdata;

   MPI_Comm           comm           = (hybrid_data -> comm);

   HYPRE_Real         cf_tol         = (hybrid_data -> cf_tol);
   HYPRE_Int          dscg_max_its   = (hybrid_data -> dscg_max_its);
   HYPRE_Int          krylov_max_its    = (hybrid_data -> krylov_max_its);
   HYPRE_Int          logging        = (hybrid_data -> logging);
   HYPRE_Int          solver_type    = (hybrid_data -> solver_type);

   HYPRE_Int          krylov_default = (hybrid_data -> krylov_default);
   HYPRE_Int        (*krylov_precond_solve)(void*, void*, void*, void*);
   HYPRE_Int        (*krylov_precond_setup)(void*, void*, void*, void*);
   void              *krylov_precond;
   void              *krylov_solver;

   HYPRE_Int          dscg_num_its;
   HYPRE_Int          krylov_num_its;
   HYPRE_Int          converged;

   HYPRE_Real         res_norm;
   HYPRE_Int          myid;

   if (solver_type == 1)
   {
      /*--------------------------------------------------------------------
       * Setup DSCG.
       *--------------------------------------------------------------------*/
      krylov_solver = hypre_HybridSolveUsePCG(hybrid_data);
      hypre_PCGSetMaxIter(krylov_solver, dscg_max_its);
      hypre_PCGSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      hypre_PCGSetPrecond((void*) krylov_solver,
                          (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScale,
                          (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScaleSetup,
                          (void*) krylov_precond);
      hypre_PCGSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with DSCG.
       *--------------------------------------------------------------------*/
      hypre_PCGSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for DSCG.
       *--------------------------------------------------------------------*/
      hypre_PCGGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_PCGGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * Get additional information from PCG if logging on for hybrid solver.
       * Currently used as debugging flag to print norms.
       *--------------------------------------------------------------------*/
      if ( logging > 1 )
      {
         hypre_MPI_Comm_rank(comm, &myid );
         hypre_PCGPrintLogging(krylov_solver, myid);
      }

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_PCGGetConverged(krylov_solver, &converged);
   }
   else if (solver_type == 2)
   {
      /*--------------------------------------------------------------------
       * Setup GMRES
       *--------------------------------------------------------------------*/
      krylov_solver = hypre_HybridSolveUseGMRES(hybrid_data);
      hypre_GMRESSetMaxIter(krylov_solver, dscg_max_its);
      hypre_GMRESSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      hypre_GMRESSetPrecond((void*) krylov_solver,
                            (HYPRE_Int (*)(void*, void*, void*, void*))HYPRE_StructDiagScale,
                            (HYPRE_Int (*)(void*, void*, void*, void*))HYPRE_StructDiagScaleSetup,
                            (void*) krylov_precond);
      hypre_GMRESSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with GMRES
       *--------------------------------------------------------------------*/
      hypre_GMRESSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for GMRES
       *--------------------------------------------------------------------*/
      hypre_GMRESGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_GMRESGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_GMRESGetConverged(krylov_solver, &converged);
   }

   else
   {
      /*--------------------------------------------------------------------
       * Setup BiCGSTAB
       *--------------------------------------------------------------------*/
      krylov_solver = hypre_HybridSolveUseBiCGSTAB(hybrid_data);
      hypre_BiCGSTABSetMaxIter(krylov_solver, dscg_max_its);
      hypre_BiCGSTABSetConvergenceFactorTol(krylov_solver, cf_tol);

      krylov_precond = NULL;

      hypre_BiCGSTABSetPrecond((void*) krylov_solver,
                               (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScale,
                               (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScaleSetup,
                               (void*) krylov_precond);
      hypre_BiCGSTABSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Solve with BiCGSTAB
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for BiCGSTAB
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABGetNumIterations(krylov_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_BiCGSTABGetFinalRelativeResidualNorm(krylov_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABGetConverged(krylov_solver, &converged);
   }

   /*-----------------------------------------------------------------------
    * if converged, done...
    *-----------------------------------------------------------------------*/
   if ( converged )
   {
      (hybrid_data -> final_rel_res_norm) = res_norm;
      if (solver_type == 1)
      {
         hypre_PCGDestroy(krylov_solver);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESDestroy(krylov_solver);
      }
      else
      {
         hypre_BiCGSTABDestroy(krylov_solver);
      }
   }

   /*-----------------------------------------------------------------------
    * ... otherwise, use solver+precond
    *-----------------------------------------------------------------------*/
   else
   {
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      if (solver_type == 1)
      {
         hypre_PCGDestroy(krylov_solver);

         krylov_solver = hypre_HybridSolveUsePCG(hybrid_data);
         hypre_PCGSetMaxIter(krylov_solver, krylov_max_its);
         hypre_PCGSetConvergenceFactorTol(krylov_solver, 0.0);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESDestroy(krylov_solver);

         krylov_solver = hypre_HybridSolveUseGMRES(hybrid_data);
         hypre_GMRESSetMaxIter(krylov_solver, krylov_max_its);
         hypre_GMRESSetConvergenceFactorTol(krylov_solver, 0.0);
      }
      else
      {
         hypre_BiCGSTABDestroy(krylov_solver);

         krylov_solver = hypre_HybridSolveUseBiCGSTAB(hybrid_data);
         hypre_BiCGSTABSetMaxIter(krylov_solver, krylov_max_its);
         hypre_BiCGSTABSetConvergenceFactorTol(krylov_solver, 0.0);
      }

      /* Setup preconditioner */
      if (krylov_default)
      {
         krylov_precond = hypre_SMGCreate(comm);
         hypre_SMGSetMaxIter(krylov_precond, 1);
         hypre_SMGSetTol(krylov_precond, 0.0);
         hypre_SMGSetNumPreRelax(krylov_precond, 1);
         hypre_SMGSetNumPostRelax(krylov_precond, 1);
         hypre_SMGSetLogging(krylov_precond, 0);
         krylov_precond_solve = (HYPRE_Int (*)(void*, void*, void*, void*))hypre_SMGSolve;
         krylov_precond_setup = (HYPRE_Int (*)(void*, void*, void*, void*))hypre_SMGSetup;
      }
      else
      {
         krylov_precond       = (hybrid_data -> krylov_precond);
         krylov_precond_solve = (hybrid_data -> krylov_precond_solve);
         krylov_precond_setup = (hybrid_data -> krylov_precond_setup);
      }

      /* Complete setup of solver+precond */
      if (solver_type == 1)
      {
         hypre_PCGSetPrecond((void*) krylov_solver,
                             (HYPRE_Int (*)(void*, void*, void*, void*)) krylov_precond_solve,
                             (HYPRE_Int (*)(void*, void*, void*, void*)) krylov_precond_setup,
                             (void*) krylov_precond);
         hypre_PCGSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_PCGSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from PCG that is always logged in hybrid solver*/
         hypre_PCGGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         hypre_PCGGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /*-----------------------------------------------------------------
          * Get additional information from PCG if logging on for hybrid solver.
          * Currently used as debugging flag to print norms.
          *-----------------------------------------------------------------*/
         if ( logging > 1 )
         {
            hypre_MPI_Comm_rank(comm, &myid );
            hypre_PCGPrintLogging(krylov_solver, myid);
         }

         /* Free PCG and preconditioner */
         hypre_PCGDestroy(krylov_solver);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESSetPrecond(krylov_solver,
                               krylov_precond_solve, krylov_precond_setup, krylov_precond);
         hypre_GMRESSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_GMRESSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from GMRES that is always logged in hybrid solver*/
         hypre_GMRESGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         hypre_GMRESGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free GMRES and preconditioner */
         hypre_GMRESDestroy(krylov_solver);
      }
      else
      {
         hypre_BiCGSTABSetPrecond(krylov_solver, krylov_precond_solve,
                                  krylov_precond_setup, krylov_precond);
         hypre_BiCGSTABSetup(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_BiCGSTABSolve(krylov_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from BiCGSTAB that is always logged in hybrid solver*/
         hypre_BiCGSTABGetNumIterations(krylov_solver, &krylov_num_its);
         (hybrid_data -> krylov_num_its)  = krylov_num_its;
         hypre_BiCGSTABGetFinalRelativeResidualNorm(krylov_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free BiCGSTAB and preconditioner */
         hypre_BiCGSTABDestroy(krylov_solver);
      }

      if (krylov_default)
      {
         hypre_SMGDestroy(krylov_precond);
      }
   }

   return hypre_error_flag;
}
