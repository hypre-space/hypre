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

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif
#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_HybridData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   double                tol;
   double                cf_tol;
   int                   max_ds_its;
   int                   max_mg_its;
   int                   two_norm;
   int                   rel_change;

   void                 *pcg_solver;
   void                 *pcg_precond;

   /* log info (always logged) */
   int                   num_ds_its;
   int                   num_mg_its;
   double                final_rel_res_norm;
   int                   time_index;

   /* additional information (place-holder currently used to print norms) */
   int                   logging;

} hypre_HybridData;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_HybridData
 *--------------------------------------------------------------------------*/

#define hypre_HybridPCGSolver(data)           ((data) -> pcg_solver)
#define hypre_HybridPCGPrecond(data)          ((data) -> pcg_precond)
#define hypre_HybridNumDSIterations(data)     ((data) -> num_ds_its)
#define hypre_HybridNumMGIterations(data)     ((data) -> num_mg_its)
#define hypre_HybridFinalRelResNorm(data)     ((data) -> final_rel_res_norm)


/*--------------------------------------------------------------------------
 * hypre_HybridInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_HybridInitialize( MPI_Comm  comm )
{
   hypre_HybridData *hybrid_data;

   hybrid_data = hypre_CTAlloc(hypre_HybridData, 1);

   (hybrid_data -> comm)        = comm;
   (hybrid_data -> time_index)  = hypre_InitializeTiming("HYBRID");

   /* set defaults */
   (hybrid_data -> tol)               = 1.0e-06;
   (hybrid_data -> cf_tol)            = 0.90;
   (hybrid_data -> max_ds_its)        = 1000;
   (hybrid_data -> max_mg_its)        = 200;
   (hybrid_data -> two_norm)          = 0;
   (hybrid_data -> rel_change)        = 0;
   (hybrid_data -> logging)           = 0;

   /* initialize */
   (hybrid_data -> num_ds_its)        = 0; 
   (hybrid_data -> num_mg_its)        = 0;
   (hybrid_data -> pcg_solver)        = NULL;
   (hybrid_data -> pcg_precond)       = NULL;
   
   return (void *) hybrid_data;
}

/*--------------------------------------------------------------------------
 * hypre_HybridFinalize
 *--------------------------------------------------------------------------*/

int
hypre_HybridFinalize( void  *hybrid_vdata )
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
 * hypre_HybridSetMaxDSIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetMaxDSIterations( void   *hybrid_vdata,
                                int     max_ds_its   )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> max_ds_its) = max_ds_its;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSetMaxMGIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridSetMaxMGIterations( void   *hybrid_vdata,
                                int     max_mg_its   )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   (hybrid_data -> max_mg_its) = max_mg_its;

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
 * hypre_HybridGetNumDSIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetNumIterations( void   *hybrid_vdata,
                              int    *num_its   )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *num_its = (hybrid_data -> num_ds_its) + (hybrid_data -> num_mg_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetNumDSIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetNumDSIterations( void   *hybrid_vdata,
                                int    *num_ds_its   )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *num_ds_its = (hybrid_data -> num_ds_its);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridGetNumMGIterations
 *--------------------------------------------------------------------------*/

int
hypre_HybridGetNumMGIterations( void   *hybrid_vdata,
                                int    *num_mg_its   )
{
   hypre_HybridData *hybrid_data = hybrid_vdata;
   int               ierr = 0;

   *num_mg_its = (hybrid_data -> num_mg_its);

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
                   hypre_StructVector *x         )
{
   hypre_HybridData  *hybrid_data    = hybrid_vdata;
   int                logging        = (hybrid_data -> logging);
   int                max_ds_its     = (hybrid_data -> max_ds_its);
   int                max_mg_its     = (hybrid_data -> max_mg_its);

   int                ierr = 0;
    
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_HybridSolve
 *--------------------------------------------------------------------------
 *
 * This solver is designed to solve Ax=b using a hybrid algorithm. First
 * the solver uses diagonally scaled conjugate gradients. If suffiecient
 * progress is not made, the algorithm switches to SMG preconditioned
 * conjugate gradients.
 *
 *--------------------------------------------------------------------------*/

int
hypre_HybridSolve( void               *hybrid_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x         )
{
   hypre_HybridData  *hybrid_data    = hybrid_vdata;

   MPI_Comm           comm           = (hybrid_data -> comm);

   double             tol            = (hybrid_data -> tol);
   double             cf_tol         = (hybrid_data -> cf_tol);
   int                max_ds_its     = (hybrid_data -> max_ds_its);
   int                max_mg_its     = (hybrid_data -> max_mg_its);
   int                two_norm       = (hybrid_data -> two_norm);
   int                rel_change     = (hybrid_data -> rel_change);
   int                logging        = (hybrid_data -> logging);
  
   void              *pcg_solver     = (hybrid_data -> pcg_solver);
   void              *pcg_precond    = (hybrid_data -> pcg_precond);

   int                num_its;
   int                num_ds_its;
   int                num_mg_its;

   double             res_norm;
   int                myid;

   int                ierr = 0;


   /*-----------------------------------------------------------------------
    * Setup DSCG.
    *-----------------------------------------------------------------------*/
   pcg_solver = hypre_PCGInitialize();
   hypre_HybridPCGSolver(hybrid_data) = pcg_solver;
   hypre_PCGSetMaxIter(pcg_solver, max_ds_its);
   hypre_PCGSetTol(pcg_solver, tol);
   hypre_PCGSetConvergenceFactorTol(pcg_solver, cf_tol);
   hypre_PCGSetTwoNorm(pcg_solver, two_norm);
   hypre_PCGSetRelChange(pcg_solver, rel_change);
   hypre_PCGSetLogging(pcg_solver, 1);

   pcg_precond = NULL;
   hypre_HybridPCGPrecond(hybrid_data) = pcg_precond;

   hypre_PCGSetPrecond(pcg_solver,
                       HYPRE_StructDiagScale,
                       HYPRE_StructDiagScaleSetup,
                       pcg_precond);
   hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);


   /*-----------------------------------------------------------------------
    * Solve with DSCG.
    *-----------------------------------------------------------------------*/
   hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

   /*-----------------------------------------------------------------------
    * Get information for DSCG.
    *-----------------------------------------------------------------------*/
   hypre_PCGGetNumIterations(pcg_solver, &num_ds_its);
   hypre_HybridNumDSIterations(hybrid_data)  = num_ds_its;
   hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);

   /*-----------------------------------------------------------------------
    * Get additional information from PCG if logging on for hybrid solver.
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
      hypre_HybridFinalRelResNorm( hybrid_data ) = res_norm;
      hypre_PCGFinalize(pcg_solver);
   }

   /*-----------------------------------------------------------------------
    * ... otherwise, use SMG+PCG.
    *-----------------------------------------------------------------------*/
   else
   {
      
      /*--------------------------------------------------------------------
       * Free up previous PCG solver structure and set up a new one.
       *--------------------------------------------------------------------*/
      hypre_PCGFinalize(pcg_solver);

      pcg_solver = hypre_PCGInitialize();
      hypre_HybridPCGSolver(hybrid_data) = pcg_solver;
      hypre_PCGSetMaxIter(pcg_solver, max_mg_its);
      hypre_PCGSetTol(pcg_solver, tol);
      hypre_PCGSetTwoNorm(pcg_solver, two_norm);
      hypre_PCGSetRelChange(pcg_solver, rel_change);
      hypre_PCGSetLogging(pcg_solver, 1);



      /* Setup SMG preconditioner */
      pcg_precond = hypre_SMGInitialize(comm);
      hypre_HybridPCGPrecond(hybrid_data) = pcg_precond;
      hypre_SMGSetMaxIter(pcg_precond, 1);
      hypre_SMGSetTol(pcg_precond, 0.0);
      hypre_SMGSetNumPreRelax(pcg_precond, 1);
      hypre_SMGSetNumPostRelax(pcg_precond, 1);
      hypre_SMGSetLogging(pcg_precond, 0);

      /* Complete setup of PCG+SMG */
      hypre_PCGSetPrecond(pcg_solver,
                          hypre_SMGSolve,
                          hypre_SMGSetup,
                          pcg_precond);
      hypre_PCGSetup(pcg_solver, (void*) A, (void*) b, (void*) x);

      /* Solve */
      hypre_PCGSolve(pcg_solver, (void*) A, (void*) b, (void*) x);

      /* Get information from PCG that is always logged in hybrid solver*/
      hypre_PCGGetNumIterations(pcg_solver, &num_mg_its);
      hypre_HybridNumMGIterations(hybrid_data)  = num_mg_its;
      hypre_PCGGetFinalRelativeResidualNorm(pcg_solver, &res_norm);
      hypre_HybridFinalRelResNorm( hybrid_data ) = res_norm;

      /*--------------------------------------------------------------------
       * Get additional information from PCG if logging on for hybrid solver.
       * Currently used as debugging flag to print norms.
       *--------------------------------------------------------------------*/
      if( logging > 1 )
      {
         MPI_Comm_rank(comm, &myid );
         hypre_PCGPrintLogging( pcg_solver, myid);
      }

      /* Free PCG and SMG preconditioner */
      hypre_PCGFinalize(pcg_solver);
      hypre_SMGFinalize(pcg_precond);
   }

   return ierr;
   
}

