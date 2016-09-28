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
   HYPRE_Int             pcg_max_its;
   HYPRE_Int             two_norm;
   HYPRE_Int             stop_crit;
   HYPRE_Int             rel_change;
   HYPRE_Int             k_dim;
   HYPRE_Int 			 solver_type;

   HYPRE_Int             pcg_default;              /* boolean */
   HYPRE_Int           (*pcg_precond_solve)(void*, void*, void*, void*);
   HYPRE_Int           (*pcg_precond_setup)(void*, void*, void*, void*);
   void                 *pcg_precond;

   /* log info (always logged) */
   HYPRE_Int             dscg_num_its;
   HYPRE_Int             pcg_num_its;
   HYPRE_Real            final_rel_res_norm;
   HYPRE_Int             time_index;

   HYPRE_Int           print_level;
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

   hybrid_data = hypre_CTAlloc(hypre_HybridData, 1);

   (hybrid_data -> comm)        = comm;
   (hybrid_data -> time_index)  = hypre_InitializeTiming("Hybrid");

   /* set defaults */
   (hybrid_data -> tol)               = 1.0e-06;
   (hybrid_data -> cf_tol)            = 0.90;
   (hybrid_data -> pcg_atolf)         = 0.0;
   (hybrid_data -> dscg_max_its)      = 1000;
   (hybrid_data -> pcg_max_its)       = 200;
   (hybrid_data -> two_norm)          = 0;
   (hybrid_data -> stop_crit)          = 0;
   (hybrid_data -> rel_change)        = 0;
   (hybrid_data -> solver_type)       = 1;
   (hybrid_data -> k_dim)             = 5;
   (hybrid_data -> pcg_default)       = 1;
   (hybrid_data -> pcg_precond_solve) = NULL;
   (hybrid_data -> pcg_precond_setup) = NULL;
   (hybrid_data -> pcg_precond)       = NULL;

   /* initialize */
   (hybrid_data -> dscg_num_its)      = 0; 
   (hybrid_data -> pcg_num_its)       = 0;
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
      hypre_TFree(hybrid_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetTol( void   *hybrid_vdata,
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
hypre_HybridSetConvergenceTol( void   *hybrid_vdata,
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
hypre_HybridSetDSCGMaxIter( void   *hybrid_vdata,
                            HYPRE_Int     dscg_max_its )
{
	hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> dscg_max_its) = dscg_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPCGMaxIter( void   *hybrid_vdata,
                           HYPRE_Int     pcg_max_its  )
{
	hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> pcg_max_its) = pcg_max_its;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPCGAbsoluteTolFactor( void   *hybrid_vdata,
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
hypre_HybridSetTwoNorm( void *hybrid_vdata,
                        HYPRE_Int   two_norm  )
{
	hypre_HybridData *hybrid_data = ( hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> two_norm) = two_norm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetStopCrit( void *hybrid_vdata,
                         HYPRE_Int   stop_crit  )
{
	hypre_HybridData *hybrid_data = ( hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> stop_crit) = stop_crit;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetRelChange( void *hybrid_vdata,
                          HYPRE_Int   rel_change  )
{
	hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetSolverType( void *hybrid_vdata,
                           HYPRE_Int   solver_type  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> solver_type) = solver_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetKDim( void *hybrid_vdata,
                     HYPRE_Int   k_dim  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> k_dim) = k_dim;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetPrecond( void  *pcg_vdata,
                        HYPRE_Int  (*pcg_precond_solve)(void*, void*, void*, void*),
                        HYPRE_Int  (*pcg_precond_setup)(void*, void*, void*, void*),
                        void  *pcg_precond          )
{
   hypre_HybridData *pcg_data = (hypre_HybridData *)pcg_vdata;
 
   (pcg_data -> pcg_default)       = 0;
   (pcg_data -> pcg_precond_solve) = pcg_precond_solve;
   (pcg_data -> pcg_precond_setup) = pcg_precond_setup;
   (pcg_data -> pcg_precond)       = pcg_precond;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridSetLogging( void *hybrid_vdata,
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
hypre_HybridSetPrintLevel( void *hybrid_vdata,
                           HYPRE_Int   print_level  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   (hybrid_data -> print_level) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetNumIterations( void   *hybrid_vdata,
                              HYPRE_Int    *num_its      )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *num_its = (hybrid_data -> dscg_num_its) + (hybrid_data -> pcg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetDSCGNumIterations( void   *hybrid_vdata,
                                  HYPRE_Int    *dscg_num_its )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *dscg_num_its = (hybrid_data -> dscg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetPCGNumIterations( void   *hybrid_vdata,
                                 HYPRE_Int    *pcg_num_its  )
{
   hypre_HybridData *hybrid_data = (hypre_HybridData *)hybrid_vdata;

   *pcg_num_its = (hybrid_data -> pcg_num_its);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_HybridGetFinalRelativeResidualNorm( void   *hybrid_vdata,
                                          HYPRE_Real *final_rel_res_norm )
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
                   hypre_StructVector *x            )
{
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

HYPRE_Int
hypre_HybridSolve( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_HybridData  *hybrid_data    = (hypre_HybridData *)hybrid_vdata;

   MPI_Comm           comm           = (hybrid_data -> comm);

   HYPRE_Real         tol            = (hybrid_data -> tol);
   HYPRE_Real         cf_tol         = (hybrid_data -> cf_tol);
   HYPRE_Real         pcg_atolf      = (hybrid_data -> pcg_atolf);
   HYPRE_Int          dscg_max_its   = (hybrid_data -> dscg_max_its);
   HYPRE_Int          pcg_max_its    = (hybrid_data -> pcg_max_its);
   HYPRE_Int          two_norm       = (hybrid_data -> two_norm);
   HYPRE_Int          stop_crit      = (hybrid_data -> stop_crit);
   HYPRE_Int          rel_change     = (hybrid_data -> rel_change);
   HYPRE_Int          logging        = (hybrid_data -> logging);
   HYPRE_Int          print_level    = (hybrid_data -> print_level);
   HYPRE_Int          solver_type    = (hybrid_data -> solver_type);
   HYPRE_Int          k_dim          = (hybrid_data -> k_dim);
  
   HYPRE_Int          pcg_default    = (hybrid_data -> pcg_default);
   HYPRE_Int        (*pcg_precond_solve)(void*, void*, void*, void*);
   HYPRE_Int        (*pcg_precond_setup)(void*, void*, void*, void*);
   void              *pcg_precond;

   void              *pcg_solver;
   hypre_PCGFunctions * pcg_functions;
   hypre_GMRESFunctions * gmres_functions;
   hypre_BiCGSTABFunctions * bicgstab_functions;

   HYPRE_Int          dscg_num_its;
   HYPRE_Int          pcg_num_its;
   HYPRE_Int          converged;

   HYPRE_Real         res_norm;
   HYPRE_Int          myid;

   if (solver_type == 1)
   {
      /*--------------------------------------------------------------------
       * Setup DSCG.
       *--------------------------------------------------------------------*/
      pcg_functions =
         hypre_PCGFunctionsCreate(
            hypre_CAlloc, hypre_StructKrylovFree,
            hypre_StructKrylovCommInfo,
            hypre_StructKrylovCreateVector,
            hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
            hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
            hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
            hypre_StructKrylovClearVector,
            hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
            hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
      pcg_solver = hypre_PCGCreate( pcg_functions );

      hypre_PCGSetMaxIter(pcg_solver, dscg_max_its);
      hypre_PCGSetTol(pcg_solver, tol);
      hypre_PCGSetAbsoluteTolFactor(pcg_solver, pcg_atolf);
      hypre_PCGSetConvergenceFactorTol(pcg_solver, cf_tol);
      hypre_PCGSetTwoNorm(pcg_solver, two_norm);
      hypre_PCGSetStopCrit(pcg_solver, stop_crit);
      hypre_PCGSetRelChange(pcg_solver, rel_change);
      hypre_PCGSetPrintLevel(pcg_solver, print_level);
      hypre_PCGSetLogging(pcg_solver, logging);

      pcg_precond = NULL;

      hypre_PCGSetPrecond((void*) pcg_solver,
                          (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScale,
                          (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScaleSetup,
                          (void*) pcg_precond);
      hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);


      /*--------------------------------------------------------------------
       * Solve with DSCG.
       *--------------------------------------------------------------------*/
      hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for DSCG.
       *--------------------------------------------------------------------*/
      hypre_PCGGetNumIterations(pcg_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

      /*--------------------------------------------------------------------
       * Get additional information from PCG if logging on for hybrid solver.
       * Currently used as debugging flag to print norms.
       *--------------------------------------------------------------------*/
      if( logging > 1 )
      {
         hypre_MPI_Comm_rank(comm, &myid );
         hypre_PCGPrintLogging(pcg_solver, myid);
      }

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_PCGGetConverged(pcg_solver, &converged);
   }
   else if (solver_type == 2)
   {
      /*--------------------------------------------------------------------
       * Setup GMRES
       *--------------------------------------------------------------------*/
      gmres_functions =
         hypre_GMRESFunctionsCreate(
            hypre_CAlloc, hypre_StructKrylovFree,
            hypre_StructKrylovCommInfo,
            hypre_StructKrylovCreateVector,
            hypre_StructKrylovCreateVectorArray,
            hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
            hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
            hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
            hypre_StructKrylovClearVector,
            hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
            hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
      pcg_solver = hypre_GMRESCreate( gmres_functions );

      hypre_GMRESSetMaxIter(pcg_solver, dscg_max_its);
      hypre_GMRESSetTol(pcg_solver, tol);
      hypre_GMRESSetKDim(pcg_solver, k_dim);
      hypre_GMRESSetConvergenceFactorTol(pcg_solver, cf_tol);
      hypre_GMRESSetStopCrit(pcg_solver, stop_crit);
      hypre_GMRESSetRelChange(pcg_solver, rel_change);
      hypre_GMRESSetPrintLevel(pcg_solver, print_level);
      hypre_GMRESSetPrintLevel(pcg_solver, print_level);
      hypre_GMRESSetLogging(pcg_solver, logging);

      pcg_precond = NULL;

      hypre_GMRESSetPrecond((void*) pcg_solver,
                            (HYPRE_Int (*)(void*, void*, void*, void*))HYPRE_StructDiagScale,
                            (HYPRE_Int (*)(void*, void*, void*, void*))HYPRE_StructDiagScaleSetup,
                            (void*) pcg_precond);
      hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);


      /*--------------------------------------------------------------------
       * Solve with GMRES
       *--------------------------------------------------------------------*/
      hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for GMRES
       *--------------------------------------------------------------------*/
      hypre_GMRESGetNumIterations(pcg_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_GMRESGetConverged(pcg_solver, &converged);
   }

   else 
   {
      /*--------------------------------------------------------------------
       * Setup BiCGSTAB
       *--------------------------------------------------------------------*/
      bicgstab_functions =
         hypre_BiCGSTABFunctionsCreate(
            hypre_StructKrylovCreateVector,
            hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
            hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
            hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
            hypre_StructKrylovClearVector,
            hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
            hypre_StructKrylovCommInfo,
            hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
      pcg_solver = hypre_BiCGSTABCreate( bicgstab_functions );

      hypre_BiCGSTABSetMaxIter(pcg_solver, dscg_max_its);
      hypre_BiCGSTABSetTol(pcg_solver, tol);
      hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, cf_tol);
      hypre_BiCGSTABSetStopCrit(pcg_solver, stop_crit);
      hypre_BiCGSTABSetPrintLevel(pcg_solver, print_level);
      hypre_BiCGSTABSetLogging(pcg_solver, logging);

      pcg_precond = NULL;

      hypre_BiCGSTABSetPrecond((void*) pcg_solver,
                               (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScale,
                               (HYPRE_Int (*)(void*, void*, void*, void*)) HYPRE_StructDiagScaleSetup,
                               (void*) pcg_precond);
      hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);


      /*--------------------------------------------------------------------
       * Solve with BiCGSTAB
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

      /*--------------------------------------------------------------------
       * Get information for BiCGSTAB
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABGetNumIterations(pcg_solver, &dscg_num_its);
      (hybrid_data -> dscg_num_its) = dscg_num_its;
      hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

      /*--------------------------------------------------------------------
       * check if converged.
       *--------------------------------------------------------------------*/
      hypre_BiCGSTABGetConverged(pcg_solver, &converged);
   }


   /*-----------------------------------------------------------------------
    * if converged, done... 
    *-----------------------------------------------------------------------*/
   if( converged )
   {
      (hybrid_data -> final_rel_res_norm) = res_norm;
      if (solver_type == 1)
	 hypre_PCGDestroy(pcg_solver);
      else if (solver_type == 2)
	 hypre_GMRESDestroy(pcg_solver);
      else
	 hypre_BiCGSTABDestroy(pcg_solver);
   }

   /*-----------------------------------------------------------------------
    * ... otherwise, use SMG+solver
    *-----------------------------------------------------------------------*/
   else
   {
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      if (solver_type == 1)
      {
         hypre_PCGDestroy(pcg_solver);

         pcg_functions =
            hypre_PCGFunctionsCreate(
               hypre_CAlloc, hypre_StructKrylovFree,
               hypre_StructKrylovCommInfo,
               hypre_StructKrylovCreateVector,
               hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
               hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
               hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
               hypre_StructKrylovClearVector,
               hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
               hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
         pcg_solver = hypre_PCGCreate( pcg_functions );

         hypre_PCGSetMaxIter(pcg_solver, pcg_max_its);
         hypre_PCGSetTol(pcg_solver, tol);
         hypre_PCGSetAbsoluteTolFactor(pcg_solver, pcg_atolf);
         hypre_PCGSetTwoNorm(pcg_solver, two_norm);
         hypre_PCGSetStopCrit(pcg_solver, stop_crit);
         hypre_PCGSetRelChange(pcg_solver, rel_change);
         hypre_PCGSetPrintLevel(pcg_solver, print_level);
         hypre_PCGSetLogging(pcg_solver, logging);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESDestroy(pcg_solver);

         gmres_functions =
            hypre_GMRESFunctionsCreate(
               hypre_CAlloc, hypre_StructKrylovFree,
               hypre_StructKrylovCommInfo,
               hypre_StructKrylovCreateVector,
               hypre_StructKrylovCreateVectorArray,
               hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
               hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
               hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
               hypre_StructKrylovClearVector,
               hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
               hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
         pcg_solver = hypre_GMRESCreate( gmres_functions );

         hypre_GMRESSetMaxIter(pcg_solver, pcg_max_its);
         hypre_GMRESSetTol(pcg_solver, tol);
         hypre_GMRESSetKDim(pcg_solver, k_dim);
         hypre_GMRESSetStopCrit(pcg_solver, stop_crit);
         hypre_GMRESSetRelChange(pcg_solver, rel_change);
         hypre_GMRESSetPrintLevel(pcg_solver, print_level);
         hypre_GMRESSetLogging(pcg_solver, logging);
         hypre_GMRESSetConvergenceFactorTol(pcg_solver, 0.0);
      }
      else
      {
         hypre_BiCGSTABDestroy(pcg_solver);

         bicgstab_functions =
            hypre_BiCGSTABFunctionsCreate(
               hypre_StructKrylovCreateVector,
               hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
               hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
               hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
               hypre_StructKrylovClearVector,
               hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
               hypre_StructKrylovCommInfo,
               hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );
         pcg_solver = hypre_BiCGSTABCreate( bicgstab_functions );

         hypre_BiCGSTABSetMaxIter(pcg_solver, pcg_max_its);
         hypre_BiCGSTABSetTol(pcg_solver, tol);
         hypre_BiCGSTABSetStopCrit(pcg_solver, stop_crit);
         hypre_BiCGSTABSetPrintLevel(pcg_solver, print_level);
         hypre_BiCGSTABSetLogging(pcg_solver, logging);
         hypre_BiCGSTABSetConvergenceFactorTol(pcg_solver, 0.0);
      }

      /* Setup preconditioner */
      if (pcg_default)
      {
         pcg_precond = hypre_SMGCreate(comm);
         hypre_SMGSetMaxIter(pcg_precond, 1);
         hypre_SMGSetTol(pcg_precond, 0.0);
         hypre_SMGSetNumPreRelax(pcg_precond, 1);
         hypre_SMGSetNumPostRelax(pcg_precond, 1);
         hypre_SMGSetLogging(pcg_precond, 0);
         pcg_precond_solve = (HYPRE_Int (*)(void*, void*, void*, void*))hypre_SMGSolve;
         pcg_precond_setup = (HYPRE_Int (*)(void*, void*, void*, void*))hypre_SMGSetup;
      }
      else
      {
         pcg_precond       = (hybrid_data -> pcg_precond);
         pcg_precond_solve = (hybrid_data -> pcg_precond_solve);
         pcg_precond_setup = (hybrid_data -> pcg_precond_setup);
      }

      /* Complete setup of solver+SMG */
      if (solver_type == 1)
      {
		  hypre_PCGSetPrecond((void*) pcg_solver,
                             (HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_solve,
							 (HYPRE_Int (*)(void*, void*, void*, void*)) pcg_precond_setup,
							 (void*) pcg_precond);
         hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from PCG that is always logged in hybrid solver*/
         hypre_PCGGetNumIterations(pcg_solver, &pcg_num_its);
         (hybrid_data -> pcg_num_its)  = pcg_num_its;
         hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /*-----------------------------------------------------------------
          * Get additional information from PCG if logging on for hybrid solver.
          * Currently used as debugging flag to print norms.
          *-----------------------------------------------------------------*/
         if( logging > 1 )
         {
            hypre_MPI_Comm_rank(comm, &myid );
            hypre_PCGPrintLogging(pcg_solver, myid);
         }

         /* Free PCG and preconditioner */
         hypre_PCGDestroy(pcg_solver);
      }
      else if (solver_type == 2)
      {
         hypre_GMRESSetPrecond(pcg_solver,
                               pcg_precond_solve, pcg_precond_setup, pcg_precond);
         hypre_GMRESSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_GMRESSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from GMRES that is always logged in hybrid solver*/
         hypre_GMRESGetNumIterations(pcg_solver, &pcg_num_its);
         (hybrid_data -> pcg_num_its)  = pcg_num_its;
         hypre_GMRESGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free GMRES and preconditioner */
         hypre_GMRESDestroy(pcg_solver);
      }
      else
      {
         hypre_BiCGSTABSetPrecond(pcg_solver, pcg_precond_solve,
                                  pcg_precond_setup, pcg_precond);
         hypre_BiCGSTABSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Solve */
         hypre_BiCGSTABSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

         /* Get information from BiCGSTAB that is always logged in hybrid solver*/
         hypre_BiCGSTABGetNumIterations(pcg_solver, &pcg_num_its);
         (hybrid_data -> pcg_num_its)  = pcg_num_its;
         hypre_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
         (hybrid_data -> final_rel_res_norm) = res_norm;

         /* Free BiCGSTAB and preconditioner */
         hypre_BiCGSTABDestroy(pcg_solver);
      }

      if (pcg_default)
      {
         hypre_SMGDestroy(pcg_precond);
      }
   }

   return hypre_error_flag;
}

