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
   int                   plogging; /* for preconditioner */

   /* info for BoomerAMG */
   double		strong_threshold;
   double		max_row_sum;
   double		trunc_factor;
   int			max_levels;
   int			measure_type;
   int			coarsen_type;
   int			cycle_type;
   int		       *num_grid_sweeps;
   int		       *grid_relax_type;
   int		      **grid_relax_points;
   double	       *relax_weight;

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
   (AMGhybrid_data -> plogging)           = 0; 

   /* BoomerAMG info */
   (AMGhybrid_data -> strong_threshold)  = 0.25;
   (AMGhybrid_data -> max_row_sum)  = 0.9;
   (AMGhybrid_data -> trunc_factor)  = 0.0;
   (AMGhybrid_data -> max_levels)  = 25;
   (AMGhybrid_data -> measure_type)  = 0;
   (AMGhybrid_data -> coarsen_type)  = 6;
   (AMGhybrid_data -> cycle_type)  = 1;
   (AMGhybrid_data -> num_grid_sweeps)  = NULL;
   (AMGhybrid_data -> grid_relax_type)  = NULL;
   (AMGhybrid_data -> grid_relax_points)  = NULL;
   (AMGhybrid_data -> relax_weight)  = NULL;

   return (void *) AMGhybrid_data; 
}

/*-------------------------------------------------------------------------- *
  hypre_AMGHybridDestroy 
*--------------------------------------------------------------------------*/ 

int
hypre_AMGHybridDestroy( void  *AMGhybrid_vdata )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int i, ierr = 0;

   if (AMGhybrid_data -> num_grid_sweeps)  
   {
      hypre_TFree( (AMGhybrid_data -> num_grid_sweeps) );
      (AMGhybrid_data -> num_grid_sweeps) = NULL;
   }
   if (AMGhybrid_data -> grid_relax_type)  
   {
      hypre_TFree( (AMGhybrid_data -> grid_relax_type) );
      (AMGhybrid_data -> grid_relax_type) = NULL;
   }
   if (AMGhybrid_data -> grid_relax_points)  
   {
      for (i=0; i < 4; i++)
         hypre_TFree( (AMGhybrid_data -> grid_relax_points)[i] );
      hypre_TFree( (AMGhybrid_data -> grid_relax_points) );
      (AMGhybrid_data -> grid_relax_points) = NULL;
   }
   if (AMGhybrid_data -> relax_weight)  
   {
      hypre_TFree( (AMGhybrid_data -> relax_weight) );
      (AMGhybrid_data -> relax_weight) = NULL;
   }
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
 * hypre_AMGHybridSetPLogging
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetPLogging( void *AMGhybrid_vdata,
                        int   plogging  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> plogging) = plogging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetStrongThreshold( void *AMGhybrid_vdata,
                        double strong_threshold)
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> strong_threshold) = strong_threshold;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetMaxRowSum( void *AMGhybrid_vdata,
                        double   max_row_sum  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> max_row_sum) = max_row_sum;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetTruncFactor( void *AMGhybrid_vdata,
                        double   trunc_factor  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> trunc_factor) = trunc_factor;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetMaxLevels( void *AMGhybrid_vdata,
                        int   max_levels  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> max_levels) = max_levels;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetMeasureType
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetMeasureType( void *AMGhybrid_vdata,
                        int   measure_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> measure_type) = measure_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetCoarsenType( void *AMGhybrid_vdata,
                        int   coarsen_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> coarsen_type) = coarsen_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetCycleType
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetCycleType( void *AMGhybrid_vdata,
                        int   cycle_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   (AMGhybrid_data -> cycle_type) = cycle_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetNumGridSweeps( void *AMGhybrid_vdata,
                        int  *num_grid_sweeps  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   if ((AMGhybrid_data -> num_grid_sweeps) != NULL)
      hypre_TFree((AMGhybrid_data -> num_grid_sweeps));
   (AMGhybrid_data -> num_grid_sweeps) = num_grid_sweeps;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetGridRelaxType( void *AMGhybrid_vdata,
                        int  *grid_relax_type  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   if ((AMGhybrid_data -> grid_relax_type) != NULL )
      hypre_TFree((AMGhybrid_data -> grid_relax_type));
   (AMGhybrid_data -> grid_relax_type) = grid_relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetGridRelaxPoints( void *AMGhybrid_vdata,
                        int  **grid_relax_points  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   if ((AMGhybrid_data -> grid_relax_points) != NULL )
      hypre_TFree((AMGhybrid_data -> grid_relax_points));
   (AMGhybrid_data -> grid_relax_points) = grid_relax_points;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AMGHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

int
hypre_AMGHybridSetRelaxWeight( void *AMGhybrid_vdata,
                        double *relax_weight  )
{
   hypre_AMGHybridData *AMGhybrid_data = AMGhybrid_vdata;
   int               ierr = 0;

   if ((AMGhybrid_data -> relax_weight) != NULL )
      hypre_TFree((AMGhybrid_data -> relax_weight));
   (AMGhybrid_data -> relax_weight) = relax_weight;

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

   double             tol            = (AMGhybrid_data -> tol);
   double             cf_tol         = (AMGhybrid_data -> cf_tol);
   int                dscg_max_its   = (AMGhybrid_data -> dscg_max_its);
   int                pcg_max_its    = (AMGhybrid_data -> pcg_max_its);
   int                two_norm       = (AMGhybrid_data -> two_norm);
   int                rel_change     = (AMGhybrid_data -> rel_change);
   int                logging        = (AMGhybrid_data -> logging);
   int                plogging       = (AMGhybrid_data -> plogging);
  
   /* BoomerAMG info */
   double 	strong_threshold = (AMGhybrid_data -> strong_threshold);
   double      	max_row_sum = (AMGhybrid_data -> max_row_sum);
   double	trunc_factor = (AMGhybrid_data -> trunc_factor);
   int		max_levels = (AMGhybrid_data -> max_levels);
   int		measure_type = (AMGhybrid_data -> measure_type);
   int		coarsen_type = (AMGhybrid_data -> coarsen_type);
   int		cycle_type = (AMGhybrid_data -> cycle_type);
   int	       *num_grid_sweeps = (AMGhybrid_data -> num_grid_sweeps);
   int	       *grid_relax_type = (AMGhybrid_data -> grid_relax_type);
   int	      **grid_relax_points = (AMGhybrid_data -> grid_relax_points);
   double      *relax_weight = (AMGhybrid_data -> relax_weight);

   int	       *boom_ngs;
   int	       *boom_grt;
   int	      **boom_grp;
   double      *boom_rlxw;

   int                pcg_default    = (AMGhybrid_data -> pcg_default);
   int              (*pcg_precond_solve)();
   int              (*pcg_precond_setup)();
   void              *pcg_precond;

   void              *pcg_solver;
   hypre_PCGFunctions * pcg_functions;

   int                dscg_num_its;
   int                pcg_num_its;

   double             res_norm;

   int                ierr = 0;
   int                i, j;


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
   hypre_PCGSetLogging(pcg_solver, logging);

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
      hypre_PCGSetLogging(pcg_solver, logging);
      hypre_PCGSetConvergenceFactorTol(pcg_solver, 0.0);

      /* Setup preconditioner */
      if (pcg_default)
      {
         pcg_precond = hypre_BoomerAMGCreate();
         hypre_BoomerAMGSetMaxIter(pcg_precond, 1);
         hypre_BoomerAMGSetTol(pcg_precond, 0.0);
         hypre_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         hypre_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         hypre_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         hypre_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         hypre_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         hypre_BoomerAMGSetLogging(pcg_precond, plogging, NULL);
         hypre_BoomerAMGSetMaxLevels(pcg_precond,  max_levels);
         hypre_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
   	 if (num_grid_sweeps)
         {	
	    boom_ngs = hypre_CTAlloc(int,4);
	    for (i=0; i < 4; i++)
	       boom_ngs[i] = num_grid_sweeps[i];
            hypre_BoomerAMGSetNumGridSweeps(pcg_precond, boom_ngs);
         }
   	 if (grid_relax_type)
         {
	    boom_grt = hypre_CTAlloc(int,4);
	    for (i=0; i < 4; i++)
	       boom_grt[i] = grid_relax_type[i];
   	    hypre_BoomerAMGSetGridRelaxType(pcg_precond, boom_grt);
         }
   	 if (relax_weight)
         {
	    boom_rlxw = hypre_CTAlloc(double,max_levels);
	    for (i=0; i < max_levels; i++)
	       boom_rlxw[i] = relax_weight[i];
            hypre_BoomerAMGSetRelaxWeight(pcg_precond, boom_rlxw);
         }
   	 if (grid_relax_points)
         {
	    boom_grp = hypre_CTAlloc(int*,4);
	    for (i=0; i < 4; i++)
 	    {
	       boom_grp[i] = hypre_CTAlloc(int, num_grid_sweeps[i]);
	       for (j=0; j < num_grid_sweeps[i]; j++)
		  boom_grp[i][j] = grid_relax_points[i][j];
    	    }
            hypre_BoomerAMGSetGridRelaxPoints(pcg_precond, boom_grp);
         }
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

      /* Free PCG and preconditioner */
      hypre_PCGDestroy(pcg_solver);
      if (pcg_default)
      {
         hypre_BoomerAMGDestroy(pcg_precond);
      }
   }

   return ierr;
   
}

