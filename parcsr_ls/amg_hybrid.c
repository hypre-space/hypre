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
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_AMGHybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{

   double                tol;
   double                cf_tol;
   int                   dscg_max_its;
   int                   pcg_max_its;
   int                   two_norm;
   int                   rel_change;

   int                   pcg_default;              /* boolean */
   int                 (*pcg_precond_solve)();
   int                 (*pcg_precond_setup)();
   void                 *pcg_precond;

   /* log info (always logged) */
   int                   dscg_num_its;
   int                   pcg_num_its;
   double                final_rel_res_norm;
   int                   time_index;

   /* additional information (place-holder currently used to print norms) */
   int                   logging;

} hypre_AMGHybridData;

/*--------------------------------------------------------------------------
 * hypre_AMGHybridCreate
 *--------------------------------------------------------------------------*/

void *
hypre_AMGHybridCreate( )
{
   hypre_AMGHybridData *AMGhybrid_data;

   AMGhybrid_data = hypre_CTAlloc(hypre_AMGHybridData, 1);

   (AMGhybrid_data -> time_index)  = hypre_InitializeTiming("AMGHybrid");

   /* set defaults */
   (AMGhybrid_data -> tol)               = 1.0e-06;
   (AMGhybrid_data -> cf_tol)            = 0.90;
   (AMGhybrid_data -> dscg_max_its)      = 1000;
   (AMGhybrid_data -> pcg_max_its)       = 200;
   (AMGhybrid_data -> two_norm)          = 0;
   (AMGhybrid_data -> rel_change)        = 0;
   (AMGhybrid_data -> pcg_default)       = 1;
   (AMGhybrid_data -> pcg_precond_solve) = NULL;
   (AMGhybrid_data -> pcg_precond_setup) = NULL;
   (AMGhybrid_data -> pcg_precond)       = NULL;
   
   /* initialize */ 
   (AMGhybrid_data -> dscg_num_its)      = 0; 
   (AMGhybrid_data -> pcg_num_its)       = 0; 
   (AMGhybrid_data -> logging)           = 0; 

   return (void *) AMGhybrid_data; 
}

/*-------------------------------------------------------------------------- *
  hypre_AMGHybridDestroy 
*--------------------------------------------------------------------------*/ 

int
hypre_AMGHybridDestroy( void  *AMGhybrid_vdata )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int ierr = 0;

   if (AMGhybrid_data)
   {
      hypre_TFree(AMGhybrid_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTol
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetTol( void   *AMGhybrid_vdata,
                    double  tol       )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetConvergenceTol( void   *AMGhybrid_vdata,
                               double  cf_tol       )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> cf_tol) = cf_tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetDSCGMaxIter( void   *AMGhybrid_vdata,
                            int     dscg_max_its )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> dscg_max_its) = dscg_max_its;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetPCGMaxIter( void   *AMGhybrid_vdata,
                           int     pcg_max_its  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> pcg_max_its) = pcg_max_its;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetTwoNorm( void *AMGhybrid_vdata,
                        int   two_norm  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> two_norm) = two_norm;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetRelChange( void *AMGhybrid_vdata,
                          int   rel_change  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetPrecond( void  *pcg_vdata,
                        int  (*pcg_precond_solve)(),
                        int  (*pcg_precond_setup)(),
                        void  *pcg_precond          )
{
   hypre_AMGHybridData *pcg_data = pcg_vdata;
   int               ierr = 0;
 
   (pcg_data -> pcg_default)       = 0;
   (pcg_data -> pcg_precond_solve) = pcg_precond_solve;
   (pcg_data -> pcg_precond_setup) = pcg_precond_setup;
   (pcg_data -> pcg_precond)       = pcg_precond;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetLogging( void *AMGhybrid_vdata,
                        int   logging  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridGetNumIterations( void   *AMGhybrid_vdata,
                              int    *num_its      )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   *num_its = (AMGhybrid_data -> dscg_num_its) + (AMGhybrid_data -> pcg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridGetDSCGNumIterations( void   *AMGhybrid_vdata,
                                  int    *dscg_num_its )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   *dscg_num_its = (AMGhybrid_data -> dscg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridGetPCGNumIterations( void   *AMGhybrid_vdata,
                                 int    *pcg_num_its  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   *pcg_num_its = (AMGhybrid_data -> pcg_num_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridGetFinalRelativeResidualNorm( void   *AMGhybrid_vdata,
                                          double *final_rel_res_norm )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   *final_rel_res_norm = (AMGhybrid_data -> final_rel_res_norm);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetup
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetup( void               *AMGhybrid_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x            )
{
   int ierr = 0;
    
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a AMGhybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If sufficient
 * progress is not made, the algorithm switches to preconditioned
 * conjugate gradients with user-specified preconditioner.
 *
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSolve( void               *AMGhybrid_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector *b,
                   hypre_ParVector *x            )
{
   hypre_AMGHybridData  *AMGhybrid_data    = AMGhybrid_vdata;

   MPI_Comm           comm           = hypre_ParCSRMatrixComm(A);

   double             tol            = (AMGhybrid_data -> tol);
   double             cf_tol         = (AMGhybrid_data -> cf_tol);
   int                dscg_max_its   = (AMGhybrid_data -> dscg_max_its);
   int                pcg_max_its    = (AMGhybrid_data -> pcg_max_its);
   int                two_norm       = (AMGhybrid_data -> two_norm);
   int                rel_change     = (AMGhybrid_data -> rel_change);
   int                logging        = (AMGhybrid_data -> logging);
  
   int                pcg_default    = (AMGhybrid_data -> pcg_default);
   int              (*pcg_precond_solve)();
   int              (*pcg_precond_setup)();
   void              *pcg_precond;

   void              *pcg_solver;
   hypre_PCGFunctions * pcg_functions;

   int                dscg_num_its;
   int                pcg_num_its;

   double             res_norm;
   int                myid;

   int                ierr = 0;


   /*-----------------------------------------------------------------------
    * Setup DSCG.
    *-----------------------------------------------------------------------*/
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, 
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   pcg_solver = hypre_PCGCreate( pcg_functions );

   hypre_PCGSetMaxIter(pcg_solver, dscg_max_its);
   hypre_PCGSetTol(pcg_solver, tol);
   hypre_PCGSetConvergenceFactorTol(pcg_solver, cf_tol);
   hypre_PCGSetTwoNorm(pcg_solver, two_norm);
   hypre_PCGSetRelChange(pcg_solver, rel_change);
   hypre_PCGSetLogging(pcg_solver, 1);

   pcg_precond = NULL;

   hypre_PCGSetPrecond(pcg_solver,
                       HYPRE_ParCSRDiagScale,
                       HYPRE_ParCSRDiagScaleSetup,
                       pcg_precond);
   hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);


   /*-----------------------------------------------------------------------
    * Solve with DSCG.
    *-----------------------------------------------------------------------*/
   hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

   /*-----------------------------------------------------------------------
    * Get information for DSCG.
    *-----------------------------------------------------------------------*/
   hypre_PCGGetNumIterations(pcg_solver, &dscg_num_its);
   (AMGhybrid_data -> dscg_num_its) = dscg_num_its;
   hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

   /*-----------------------------------------------------------------------
    * Get additional information from PCG if logging on for AMGhybrid solver.
    * Currently used as debugging flag to print norms.
    *-----------------------------------------------------------------------*/
   if( logging > 1 )
   {
      MPI_Comm_rank(comm, &myid );
      hypre_PCGPrintLogging(pcg_solver, myid);
   }

   /*-----------------------------------------------------------------------
    * If converged, done... 
    *-----------------------------------------------------------------------*/
   if( res_norm < tol )
   {
      (AMGhybrid_data -> final_rel_res_norm) = res_norm;
      hypre_PCGDestroy(pcg_solver);
   }

   /*-----------------------------------------------------------------------
    * ... otherwise, use AMG+PCG.
    *-----------------------------------------------------------------------*/
   else
   {
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      hypre_PCGDestroy(pcg_solver);

      pcg_functions =
         hypre_PCGFunctionsCreate(
            hypre_CAlloc, hypre_ParKrylovFree,
            hypre_ParKrylovCommInfo,
            hypre_ParKrylovCreateVector,
            hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
            hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
            hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
            hypre_ParKrylovClearVector,
            hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
            hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
      pcg_solver = hypre_PCGCreate( pcg_functions );

      hypre_PCGSetMaxIter(pcg_solver, pcg_max_its);
      hypre_PCGSetTol(pcg_solver, tol);
      hypre_PCGSetTwoNorm(pcg_solver, two_norm);
      hypre_PCGSetRelChange(pcg_solver, rel_change);
      hypre_PCGSetLogging(pcg_solver, 1);

      /* Setup preconditioner */
      if (pcg_default)
      {
         pcg_precond = hypre_BoomerAMGCreate();
         hypre_BoomerAMGSetMaxIter(pcg_precond, 1);
         hypre_BoomerAMGSetTol(pcg_precond, 0.0);
   	 /* if (boomeramg_data)
	 {
            hypre_BoomerAMGSetCoarsenType(pcg_precond, 
		hypre_ParAMGDataCoarsenType(boomeramg_data));
            hypre_BoomerAMGSetMeasureType(pcg_precond, 
		hypre_ParAMGDataMeasureType(boomeramg_data));
            hypre_BoomerAMGSetStrongThreshold(pcg_precond, 
		hypre_ParAMGDataStrongThreshold(boomeramg_data));
            hypre_BoomerAMGSetTruncFactor(pcg_precond, 
		hypre_ParAMGDataTruncFactor(boomeramg_data));
            hypre_BoomerAMGSetLogging(pcg_precond, 
		hypre_ParAMGDataIOutDat(boomeramg_data), NULL);
            hypre_BoomerAMGSetCycleType(pcg_precond, 
		hypre_ParAMGDataCycleType(boomeramg_data));
            hypre_BoomerAMGSetNumGridSweeps(pcg_precond, 
		hypre_ParAMGDataNumGridSweeps(boomeramg_data));
            hypre_BoomerAMGSetGridRelaxType(pcg_precond, 
		hypre_ParAMGDataGridRelaxType(boomeramg_data));
            hypre_BoomerAMGSetRelaxWeight(pcg_precond, 
		hypre_ParAMGDataRelaxWeight(boomeramg_data));
            hypre_BoomerAMGSetSmoothOption(pcg_precond, 
		hypre_ParAMGDataSmoothOption(boomeramg_data));
            hypre_BoomerAMGSetSmoothNumSweep(pcg_precond, 
		hypre_ParAMGDataSmoothNumSweep(boomeramg_data));
            hypre_BoomerAMGSetGridRelaxPoints(pcg_precond, 
		hypre_ParAMGDataGridRelaxPoints(boomeramg_data));
            hypre_BoomerAMGSetMaxLevels(pcg_precond, 
		hypre_ParAMGDataMaxLevels(boomeramg_data));
            hypre_BoomerAMGSetMaxRowSum(pcg_precond, 
		hypre_ParAMGDataMaxRowSum(boomeramg_data));
            hypre_BoomerAMGSetVariant(pcg_precond, 
		hypre_ParAMGDataVariant(boomeramg_data));
            hypre_BoomerAMGSetOverlap(pcg_precond, 
		hypre_ParAMGDataOverlap(boomeramg_data));
            hypre_BoomerAMGSetDomainType(pcg_precond, 
		hypre_ParAMGDataDomainType(boomeramg_data));
            hypre_BoomerAMGSetSchwarzRlxWeight(pcg_precond, 
		hypre_ParAMGDataSchwarzRlxWeight(boomeramg_data));
            num_functions = hypre_ParAMGDataNumFunctions(boomeramg_data));
            hypre_BoomerAMGSetNumFunctions(pcg_precond, num_functions ));
            if (num_functions > 1)
               hypre_BoomerAMGSetDofFunc(pcg_precond, 
		  hypre_ParAMGDataSofFunc(boomeramg_data));
         } */
         pcg_precond_solve = hypre_BoomerAMGSolve;
         pcg_precond_setup = hypre_BoomerAMGSetup;
      }
      else
      {
         pcg_precond       = (AMGhybrid_data -> pcg_precond);
         pcg_precond_solve = (AMGhybrid_data -> pcg_precond_solve);
         pcg_precond_setup = (AMGhybrid_data -> pcg_precond_setup);
      }

      /* Complete setup of PCG+SMG */
      hypre_PCGSetPrecond(pcg_solver,
                          pcg_precond_solve, pcg_precond_setup, pcg_precond);
      hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

      /* Solve */
      hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

      /* Get information from PCG that is always logged in AMGhybrid solver*/
      hypre_PCGGetNumIterations(pcg_solver, &pcg_num_its);
      (AMGhybrid_data -> pcg_num_its)  = pcg_num_its;
      hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
      (AMGhybrid_data -> final_rel_res_norm) = res_norm;

      /*--------------------------------------------------------------------
       * Get additional information from PCG if logging on for hybrid solver.
       * Currently used as debugging flag to print norms.
       *--------------------------------------------------------------------*/
      if( logging > 1 )
      {
         MPI_Comm_rank(comm, &myid );
         hypre_PCGPrintLogging(pcg_solver, myid);
      }

      /* Free PCG and preconditioner */
      hypre_PCGDestroy(pcg_solver);
      if (pcg_default)
      {
         hypre_BoomerAMGDestroy(pcg_precond);
      }
   }

   return ierr;
   
}

