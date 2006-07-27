/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_HybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   double                tol;
   double                cf_tol;
   double                pcg_atolf;
   int                   dscg_max_its;
   int                   pcg_max_its;
   int                   two_norm;
   int                   stop_crit;
   int                   rel_change;
   int                   k_dim;
   int 			 solver_type;

   int                   pcg_default;              /* boolean */
   int                 (*pcg_precond_solve)();
   int                 (*pcg_precond_setup)();
   void                 *pcg_precond;

   /* log info (always logged) */
   int                   dscg_num_its;
   int                   pcg_num_its;
   double                final_rel_res_norm;
   int                   time_index;

   int                 print_level;
   /* additional information (place-holder currently used to print norms) */
   int                   logging;

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

int
hypre_HybridDestroy( void  *hybrid_vdata )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int ierr = 0;

   if (hybrid_data)
   {
      hypre_TFree(hybrid_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetTol
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetTol( void   *hybrid_vdata,
                    double  tol       )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetConvergenceTol( void   *hybrid_vdata,
                               double  cf_tol       )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> cf_tol) = cf_tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetDSCGMaxIter( void   *hybrid_vdata,
                            int     dscg_max_its )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> dscg_max_its) = dscg_max_its;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetPCGMaxIter( void   *hybrid_vdata,
                           int     pcg_max_its  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> pcg_max_its) = pcg_max_its;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPCGAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetPCGAbsoluteTolFactor( void   *hybrid_vdata,
                                     double  pcg_atolf  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> pcg_atolf) = pcg_atolf;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetTwoNorm
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetTwoNorm( void *hybrid_vdata,
                        int   two_norm  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> two_norm) = two_norm;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetStopCrit
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetStopCrit( void *hybrid_vdata,
                        int   stop_crit  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> stop_crit) = stop_crit;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetRelChange( void *hybrid_vdata,
                          int   rel_change  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetSolverType
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetSolverType( void *hybrid_vdata,
                          int   solver_type  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> solver_type) = solver_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetKDim
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetKDim( void *hybrid_vdata,
                          int   k_dim  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> k_dim) = k_dim;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetPrecond( void  *pcg_vdata,
                        int  (*pcg_precond_solve)(),
                        int  (*pcg_precond_setup)(),
                        void  *pcg_precond          )
{
   hypre_HybridData *pcg_data = pcg_vdata;
   int               ierr = 0;
 
   (pcg_data -> pcg_default)       = 0;
   (pcg_data -> pcg_precond_solve) = pcg_precond_solve;
   (pcg_data -> pcg_precond_setup) = pcg_precond_setup;
   (pcg_data -> pcg_precond)       = pcg_precond;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetLogging( void *hybrid_vdata,
                        int   logging  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetPrintLevel
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetPrintLevel( void *hybrid_vdata,
                        int   print_level  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> print_level) = print_level;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetNumIterations( void   *hybrid_vdata,
                              int    *num_its      )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *num_its = (hybrid_data -> dscg_num_its) + (hybrid_data -> pcg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetDSCGNumIterations( void   *hybrid_vdata,
                                  int    *dscg_num_its )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *dscg_num_its = (hybrid_data -> dscg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetPCGNumIterations( void   *hybrid_vdata,
                                 int    *pcg_num_its  )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *pcg_num_its = (hybrid_data -> pcg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetFinalRelativeResidualNorm( void   *hybrid_vdata,
                                          double *final_rel_res_norm )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *final_rel_res_norm = (hybrid_data -> final_rel_res_norm);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetup
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetup( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   int ierr = 0;
    
   return ierr;
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

int
hypre_HybridSolve( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_HybridData  *hybrid_data    = hybrid_vdata;

   MPI_Comm           comm           = (hybrid_data -> comm);

   double             tol            = (hybrid_data -> tol);
   double             cf_tol         = (hybrid_data -> cf_tol);
   double             pcg_atolf      = (hybrid_data -> pcg_atolf);
   int                dscg_max_its   = (hybrid_data -> dscg_max_its);
   int                pcg_max_its    = (hybrid_data -> pcg_max_its);
   int                two_norm       = (hybrid_data -> two_norm);
   int                stop_crit      = (hybrid_data -> stop_crit);
   int                rel_change     = (hybrid_data -> rel_change);
   int                logging        = (hybrid_data -> logging);
   int                print_level    = (hybrid_data -> print_level);
   int                solver_type    = (hybrid_data -> solver_type);
   int                k_dim          = (hybrid_data -> k_dim);
  
   int                pcg_default    = (hybrid_data -> pcg_default);
   int              (*pcg_precond_solve)();
   int              (*pcg_precond_setup)();
   void              *pcg_precond;

   void              *pcg_solver;
   hypre_PCGFunctions * pcg_functions;
   hypre_GMRESFunctions * gmres_functions;
   hypre_BiCGSTABFunctions * bicgstab_functions;

   int                dscg_num_its;
   int                pcg_num_its;
   int                converged;

   double             res_norm;
   int                myid;

   int                ierr = 0;


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

      hypre_PCGSetPrecond(pcg_solver,
                       HYPRE_StructDiagScale,
                       HYPRE_StructDiagScaleSetup,
                       pcg_precond);
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
         MPI_Comm_rank(comm, &myid );
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

      hypre_GMRESSetPrecond(pcg_solver,
                       HYPRE_StructDiagScale,
                       HYPRE_StructDiagScaleSetup,
                       pcg_precond);
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

      hypre_BiCGSTABSetPrecond(pcg_solver,
                       HYPRE_StructDiagScale,
                       HYPRE_StructDiagScaleSetup,
                       pcg_precond);
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
         pcg_precond_solve = hypre_SMGSolve;
         pcg_precond_setup = hypre_SMGSetup;
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
         hypre_PCGSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
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
            MPI_Comm_rank(comm, &myid );
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
         hypre_BiCGSTABSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
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

   return ierr;
   
}

