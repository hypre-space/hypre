/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * ParAMG functions
 *
 *****************************************************************************/

#include "headers.h"
#include "par_amg.h"
#include <assert.h>

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_BoomerAMGCreate()
{
   hypre_ParAMGData  *amg_data;

   /* setup params */
   int      max_levels;
   double   strong_threshold;
   double   max_row_sum;
   double   trunc_factor;
   double   S_commpkg_switch;
   int      interp_type;
   int      coarsen_type;
   int      measure_type;
   int      setup_type;
   int 	    num_functions;
   int 	    nodal;

   /* solve params */
   int      min_iter;
   int      max_iter;
   int      cycle_type;    
 
   double   tol;

   int      num_sweeps;  
   int      relax_type;   
   int      relax_order;   
   double   relax_wt;
   double   outer_wt;
   int      smooth_type;
   int      smooth_num_levels;
   int      smooth_num_sweeps;

   int      variant, overlap, domain_type;
   double   schwarz_rlx_weight;
   int	    level, sym;
   double   thresh, filter;
   double   drop_tol;
   int	    max_nz_per_row;
   char    *euclidfile;

   /* log info */
   int      num_iterations;
   int      cum_num_iterations;

   /* output params */
   int      print_level;
   int      logging;
   /* int      cycle_op_count; */
   char     log_file_name[256];
   int      debug_flag;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   strong_threshold = 0.25;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   S_commpkg_switch = 1.0;
   interp_type = 200;
   coarsen_type = 6;
   measure_type = 0;
   setup_type = 1;
   num_functions = 1;
   nodal = 0;

   variant = 0;
   overlap = 1;
   domain_type = 2;
   schwarz_rlx_weight = 1.0;
   smooth_num_sweeps = 1;
   smooth_num_levels = 0;
   smooth_type = 6;

   level = 1;
   sym = 0;
   thresh = 0.1;
   filter = 0.05;
   drop_tol = 0.0001;
   max_nz_per_row = 20;
   euclidfile = NULL;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   cycle_type = 1;
   tol = 1.0e-7;

   num_sweeps = 1;
   relax_type = 3;
   relax_order = 1;
   relax_wt = 1.0;
   outer_wt = 1.0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;

   /* output params */
   print_level = 0;
   logging = 0;
   sprintf(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1);

   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   hypre_BoomerAMGSetMaxLevels(amg_data, max_levels);
   hypre_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_BoomerAMGSetMaxRowSum(amg_data, max_row_sum);
   hypre_BoomerAMGSetTruncFactor(amg_data, trunc_factor);
   hypre_BoomerAMGSetSCommPkgSwitch(amg_data, S_commpkg_switch);
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetSetupType(amg_data, setup_type);
   hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   hypre_BoomerAMGSetNodal(amg_data, nodal);
   hypre_BoomerAMGSetVariant(amg_data, variant);
   hypre_BoomerAMGSetOverlap(amg_data, overlap);
   hypre_BoomerAMGSetSchwarzRlxWeight(amg_data, schwarz_rlx_weight);
   hypre_BoomerAMGSetDomainType(amg_data, domain_type);
   hypre_BoomerAMGSetSym(amg_data, sym);
   hypre_BoomerAMGSetLevel(amg_data, level);
   hypre_BoomerAMGSetThreshold(amg_data, thresh);
   hypre_BoomerAMGSetFilter(amg_data, filter);
   hypre_BoomerAMGSetDropTol(amg_data, drop_tol);
   hypre_BoomerAMGSetMaxNzPerRow(amg_data, max_nz_per_row);
   hypre_BoomerAMGSetEuclidFile(amg_data, euclidfile);

   hypre_BoomerAMGSetMinIter(amg_data, min_iter);
   hypre_BoomerAMGSetMaxIter(amg_data, max_iter);
   hypre_BoomerAMGSetCycleType(amg_data, cycle_type);
   hypre_BoomerAMGSetTol(amg_data, tol); 
   hypre_BoomerAMGSetNumSweeps(amg_data, num_sweeps);
   hypre_BoomerAMGSetRelaxType(amg_data, relax_type);
   hypre_BoomerAMGSetRelaxOrder(amg_data, relax_order);
   hypre_BoomerAMGSetRelaxWt(amg_data, relax_wt);
   hypre_BoomerAMGSetOuterWt(amg_data, outer_wt);
   hypre_BoomerAMGSetSmoothType(amg_data, smooth_type);
   hypre_BoomerAMGSetSmoothNumLevels(amg_data, smooth_num_levels);
   hypre_BoomerAMGSetSmoothNumSweeps(amg_data, smooth_num_sweeps);

   hypre_BoomerAMGSetNumIterations(amg_data, num_iterations);
#ifdef CUMNUMIT
   hypre_ParAMGDataCumNumIterations(amg_data) = cum_num_iterations;
#endif
   hypre_BoomerAMGSetPrintLevel(amg_data, print_level);
   hypre_BoomerAMGSetLogging(amg_data, logging);
   hypre_BoomerAMGSetPrintFileName(amg_data, log_file_name); 
   hypre_BoomerAMGSetDebugFlag(amg_data, debug_flag);

   hypre_BoomerAMGSetRestriction(amg_data, 0);

   hypre_BoomerAMGSetGSMG(amg_data, 0);
   hypre_BoomerAMGSetNumSamples(amg_data, 0);
   
   hypre_ParAMGDataAArray(amg_data) = NULL;
   hypre_ParAMGDataPArray(amg_data) = NULL;
   hypre_ParAMGDataRArray(amg_data) = NULL;
   hypre_ParAMGDataCFMarkerArray(amg_data) = NULL;
   hypre_ParAMGDataVtemp(amg_data)  = NULL;
   hypre_ParAMGDataRtemp(amg_data)  = NULL;
   hypre_ParAMGDataPtemp(amg_data)  = NULL;
   hypre_ParAMGDataZtemp(amg_data)  = NULL;
   hypre_ParAMGDataFArray(amg_data) = NULL;
   hypre_ParAMGDataUArray(amg_data) = NULL;
   hypre_ParAMGDataDofFunc(amg_data) = NULL;
   hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
   hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
   hypre_ParAMGDataSmoother(amg_data) = NULL;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGDestroy( void *data )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
   int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   int smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   HYPRE_Solver *smoother = hypre_ParAMGDataSmoother(amg_data);
   int i;

   if (hypre_ParAMGDataNumGridSweeps(amg_data))
   {
      hypre_TFree (hypre_ParAMGDataNumGridSweeps(amg_data));
      hypre_ParAMGDataNumGridSweeps(amg_data) = NULL; 
   }
   if (hypre_ParAMGDataGridRelaxType(amg_data))
   {
      hypre_TFree (hypre_ParAMGDataGridRelaxType(amg_data));
      hypre_ParAMGDataGridRelaxType(amg_data) = NULL; 
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data))
   {
      hypre_TFree (hypre_ParAMGDataRelaxWeight(amg_data));
      hypre_ParAMGDataRelaxWeight(amg_data) = NULL; 
   }
   if (hypre_ParAMGDataOmega(amg_data))
   {
      hypre_TFree (hypre_ParAMGDataOmega(amg_data));
      hypre_ParAMGDataOmega(amg_data) = NULL; 
   }
   if (hypre_ParAMGDataDofFunc(amg_data))
   {
      hypre_TFree (hypre_ParAMGDataDofFunc(amg_data));
      hypre_ParAMGDataDofFunc(amg_data) = NULL; 
   }
   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
   	hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data));
      hypre_ParAMGDataGridRelaxPoints(amg_data) = NULL; 
   }
   for (i=1; i < num_levels; i++)
   {
	hypre_ParVectorDestroy(hypre_ParAMGDataFArray(amg_data)[i]);
	hypre_ParVectorDestroy(hypre_ParAMGDataUArray(amg_data)[i]);
	hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(amg_data)[i]);
	hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(amg_data)[i-1]);
	hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data)[i-1]);
   }
   /* see comments in par_coarsen.c regarding special case for CF_marker */
   if (num_levels == 1)
   {
      hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
   }
   hypre_ParVectorDestroy(hypre_ParAMGDataVtemp(amg_data));
   hypre_TFree(hypre_ParAMGDataFArray(amg_data));
   hypre_TFree(hypre_ParAMGDataUArray(amg_data));
   hypre_TFree(hypre_ParAMGDataAArray(amg_data));
   hypre_TFree(hypre_ParAMGDataPArray(amg_data));
   hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data));
   if (hypre_ParAMGDataRtemp(amg_data))
      hypre_ParVectorDestroy(hypre_ParAMGDataRtemp(amg_data));
   if (hypre_ParAMGDataPtemp(amg_data))
      hypre_ParVectorDestroy(hypre_ParAMGDataPtemp(amg_data));
   if (hypre_ParAMGDataZtemp(amg_data))
      hypre_ParVectorDestroy(hypre_ParAMGDataZtemp(amg_data));

   if (hypre_ParAMGDataDofFuncArray(amg_data))
   {
      for (i=1; i < num_levels; i++)
	 hypre_TFree(hypre_ParAMGDataDofFuncArray(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataDofFuncArray(amg_data));
      hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
   }
   if (hypre_ParAMGDataRestriction(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataRArray(amg_data));
      hypre_ParAMGDataRArray(amg_data) = NULL;
   }
   if (hypre_ParAMGDataDofPointArray(amg_data))
   {
      for (i=0; i < num_levels; i++)
	 hypre_TFree(hypre_ParAMGDataDofPointArray(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataDofPointArray(amg_data));
      hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   }
   if (hypre_ParAMGDataPointDofMapArray(amg_data))
   {
      for (i=0; i < num_levels; i++)
	 hypre_TFree(hypre_ParAMGDataPointDofMapArray(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataPointDofMapArray(amg_data));
      hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
   }
   if (smooth_num_levels)
   {
      if (smooth_num_levels > num_levels-1)
	smooth_num_levels = num_levels -1;
      if (hypre_ParAMGDataSmoothType(amg_data) == 7)
      {
         for (i=0; i < smooth_num_levels; i++)
         {
	    HYPRE_ParCSRPilutDestroy(smoother[i]);
         }
      }
      else if (hypre_ParAMGDataSmoothType(amg_data) == 8)
      {
         for (i=0; i < smooth_num_levels; i++)
         {
	    HYPRE_ParCSRParaSailsDestroy(smoother[i]);
         }
      }
      else if (hypre_ParAMGDataSmoothType(amg_data) == 9)
      {
         for (i=0; i < smooth_num_levels; i++)
	 {
	    HYPRE_EuclidDestroy(smoother[i]);
         }
      }
      else if (hypre_ParAMGDataSmoothType(amg_data) == 6)
      {
         for (i=0; i < smooth_num_levels; i++)
	 {
	    HYPRE_SchwarzDestroy(smoother[i]);
         } 
      }
      hypre_TFree (hypre_ParAMGDataSmoother(amg_data));
   }
   if ( hypre_ParAMGDataResidual(amg_data) ) {
      /* jfp: was... hypre_TFree( hypre_ParAMGDataResidual(amg_data) );*/
      hypre_ParVectorDestroy( hypre_ParAMGDataResidual(amg_data) );
      hypre_ParAMGDataResidual(amg_data) = NULL;
   }

   hypre_TFree(amg_data);
   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetRestriction( void *data,
                            int   restr_par )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return (ierr);
}

int
hypre_BoomerAMGSetMaxLevels( void *data,
                          int   max_levels )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return (ierr);
}

int
hypre_BoomerAMGSetStrongThreshold( void     *data,
                                double    strong_threshold )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return (ierr);
}

int
hypre_BoomerAMGSetMaxRowSum( void     *data,
                          double    max_row_sum )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxRowSum(amg_data) = max_row_sum;

   return (ierr);
}

int
hypre_BoomerAMGSetTruncFactor( void     *data,
                            double    trunc_factor )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataTruncFactor(amg_data) = trunc_factor;

   return (ierr);
}

int
hypre_BoomerAMGSetSCommPkgSwitch( void     *data,
                                  double    S_commpkg_switch )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataSCommPkgSwitch(amg_data) = S_commpkg_switch;

   return (ierr);
}

int
hypre_BoomerAMGSetInterpType( void     *data,
                           int       interp_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return (ierr);
}

int
hypre_BoomerAMGSetMinIter( void     *data,
                        int       min_iter )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return (ierr);
} 

int
hypre_BoomerAMGSetMaxIter( void     *data,
                        int       max_iter )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return (ierr);
} 

int
hypre_BoomerAMGSetCoarsenType( void  *data,
                          int    coarsen_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return (ierr);
}

int
hypre_BoomerAMGSetMeasureType( void  *data,
                            int    measure_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return (ierr);
}

int
hypre_BoomerAMGSetSetupType( void  *data,
                             int    setup_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataSetupType(amg_data) = setup_type;

   return (ierr);
}

int
hypre_BoomerAMGSetCycleType( void  *data,
                          int    cycle_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return (ierr);
}

int
hypre_BoomerAMGSetTol( void     *data,
                    double    tol  )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataTol(amg_data) = tol;

   return (ierr);
}

int
hypre_BoomerAMGSetNumSweeps( void     *data,
                              int      num_sweeps )
{
   int ierr = 0, i;
   int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
       hypre_ParAMGDataNumGridSweeps(amg_data) = hypre_CTAlloc(int,4);
       
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   for (i=0; i < 3; i++)
      num_grid_sweeps[i] = num_sweeps;
   num_grid_sweeps[3] = 1;

   return (ierr);
}
 
int
hypre_BoomerAMGSetCycleNumSweeps( void     *data,
                                  int      num_sweeps,
                                  int      k )
{
   int ierr = 0, i;
   int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (k < 0 || k > 3)
   {
      printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      return -99;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
       num_grid_sweeps = hypre_CTAlloc(int,4);
       for (i=0; i < 4; i++)
	  num_grid_sweeps[i] = 1;
       hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }
       
   hypre_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return (ierr);
}
 
int
hypre_BoomerAMGSetNumGridSweeps( void     *data,
                              int      *num_grid_sweeps )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataNumGridSweeps(amg_data))
      hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data));
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return (ierr);
}
 
int
hypre_BoomerAMGSetRelaxType( void     *data,
                              int      relax_type )
{
   int ierr = 0, i;
   int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
       hypre_ParAMGDataGridRelaxType(amg_data) = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   for (i=0; i < 3; i++)
      grid_relax_type[i] = relax_type;
   grid_relax_type[3] = 9;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;

   return (ierr);
}

int
hypre_BoomerAMGSetCycleRelaxType( void     *data,
                                  int      relax_type,
                                  int      k )
{
   int ierr = 0, i;
   int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (k < 0 || k > 3)
   {
      printf (" Warning! Invalid cycle! relax_type not set!\n");
      return -99;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      grid_relax_type = hypre_CTAlloc(int,4);
      for (i=0; i < 3; i++)
         grid_relax_type[i] = 3;
      grid_relax_type[3] = 9;
      hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   }
      
   hypre_ParAMGDataGridRelaxType(amg_data)[k] = relax_type;
   if (k == 3)
      hypre_ParAMGDataUserCoarseRelaxType(amg_data) = relax_type;

   return (ierr);
}

int
hypre_BoomerAMGSetRelaxOrder( void     *data,
                              int       relax_order)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return (ierr);
}

int
hypre_BoomerAMGSetGridRelaxType( void     *data,
                              int      *grid_relax_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataGridRelaxType(amg_data))
      hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data));
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return (ierr);
}

int
hypre_BoomerAMGSetGridRelaxPoints( void     *data,
                                int      **grid_relax_points )
{
   int i, ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
   	hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data));
   }
   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return (ierr);
}

int
hypre_BoomerAMGSetRelaxWeight( void     *data,
                               double   *relax_weight )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataRelaxWeight(amg_data))
      hypre_TFree(hypre_ParAMGDataRelaxWeight(amg_data));
   hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;

   return (ierr);
}

int
hypre_BoomerAMGSetRelaxWt( void     *data,
                           double    relax_weight )
{
   int ierr = 0, i, num_levels;
   double *relax_weight_array;
   hypre_ParAMGData  *amg_data = data;

   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(double,num_levels);
                     
   relax_weight_array = hypre_ParAMGDataRelaxWeight(amg_data);
   for (i=0; i < num_levels; i++)
      relax_weight_array[i] = relax_weight;
   
   return (ierr);
}

int
hypre_BoomerAMGSetLevelRelaxWt( void    *data,
                                double   relax_weight,
                                int      level )
{
   int ierr = 0, i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      printf (" Warning! Invalid level! Relax weight not set!\n");
      return -99;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         hypre_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
   }
               
   hypre_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;
   
   return (ierr);
}

int
hypre_BoomerAMGSetOmega( void     *data,
                         double   *omega )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   if (hypre_ParAMGDataOmega(amg_data))
      hypre_TFree(hypre_ParAMGDataOmega(amg_data));
   hypre_ParAMGDataOmega(amg_data) = omega;

   return (ierr);
}

int
hypre_BoomerAMGSetOuterWt( void     *data,
                           double    omega )
{
   int ierr = 0, i, num_levels;
   double *omega_array;
   hypre_ParAMGData  *amg_data = data;

   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(double,num_levels);
                     
   omega_array = hypre_ParAMGDataOmega(amg_data);
   for (i=0; i < num_levels; i++)
      omega_array[i] = omega;
   
   return (ierr);
}

int
hypre_BoomerAMGSetLevelOuterWt( void    *data,
                                double   omega,
                                int      level )
{
   int ierr = 0, i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      printf (" Warning! Invalid level! Outer weight not set!\n");
      return -99;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         hypre_ParAMGDataOmega(amg_data)[i] = 1.0;
   }
               
   hypre_ParAMGDataOmega(amg_data)[level] = omega;
   
   return (ierr);
}

int
hypre_BoomerAMGSetSmoothType( void     *data,
                              int   smooth_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
               
   hypre_ParAMGDataSmoothType(amg_data) = smooth_type;
   
   return (ierr);
}

int
hypre_BoomerAMGSetSmoothNumLevels( void     *data,
                            int   smooth_num_levels )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
               
   hypre_ParAMGDataSmoothNumLevels(amg_data) = smooth_num_levels;
   
   return (ierr);
}

int
hypre_BoomerAMGSetSmoothNumSweeps( void     *data,
                            int   smooth_num_sweeps )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
               
   hypre_ParAMGDataSmoothNumSweeps(amg_data) = smooth_num_sweeps;
   
   return (ierr);
}

int
hypre_BoomerAMGSetLogging( void     *data,
                            int       logging )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataLogging(amg_data) = logging;

   return (ierr);
}

int
hypre_BoomerAMGSetPrintLevel( void     *data,
                        int print_level )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return (ierr);
}

int
hypre_BoomerAMGSetPrintFileName( void       *data,
                               const char *print_file_name )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
   assert( strlen(print_file_name)<=256 );

   sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return (ierr);
}

int
hypre_BoomerAMGSetNumIterations( void    *data,
                              int      num_iterations )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return (ierr);
}

int
hypre_BoomerAMGSetDebugFlag( void     *data,
                          int       debug_flag )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetGSMG( void *data,
                        int   par )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   amg_data->gsmg = par;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumSamples( void *data,
                        int   par )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   amg_data->num_samples = par;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumFunctions( void     *data,
                            int       num_functions )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumFunctions(amg_data) = num_functions;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNodal( void     *data,
                          int    nodal )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNodal(amg_data) = nodal;

   return (ierr);
}

int
hypre_BoomerAMGSetNumPoints( void     *data,
                          int       num_points )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return (ierr);
}

int
hypre_BoomerAMGSetDofFunc( void     *data,
                           int      *dof_func )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataDofFunc(amg_data));
   hypre_ParAMGDataDofFunc(amg_data) = dof_func;

   return (ierr);
}

int
hypre_BoomerAMGSetPointDofMap( void     *data,
                         int      *point_dof_map )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataPointDofMap(amg_data));
   hypre_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return (ierr);
}

int
hypre_BoomerAMGSetDofPoint( void     *data,
                         int      *dof_point )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataDofPoint(amg_data));
   hypre_ParAMGDataDofPoint(amg_data) = dof_point;

   return (ierr);
}

int
hypre_BoomerAMGGetNumIterations( void     *data,
                              int      *num_iterations )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   *num_iterations = hypre_ParAMGDataNumIterations(amg_data);

   return (ierr);
}

int
hypre_BoomerAMGGetCumNumIterations( void     *data,
                                    int      *cum_num_iterations )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

#ifdef CUMNUMIT
   *cum_num_iterations = hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return (ierr);
}

int
hypre_BoomerAMGGetResidual( void * data, hypre_ParVector ** resid )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
   *resid = hypre_ParAMGDataResidual( amg_data );
   return ierr;
}
                            

int
hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                     double   *rel_resid_norm )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return (ierr);
}

int
hypre_BoomerAMGSetVariant( void     *data,
                            int       variant)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataVariant(amg_data) = variant;

   return (ierr);
}

int
hypre_BoomerAMGSetOverlap( void     *data,
                            int       overlap)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataOverlap(amg_data) = overlap;

   return (ierr);
}

int
hypre_BoomerAMGSetDomainType( void     *data,
                            int       domain_type)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataDomainType(amg_data) = domain_type;

   return (ierr);
}

int
hypre_BoomerAMGSetSchwarzRlxWeight( void     *data,
                            double     schwarz_rlx_weight)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return (ierr);
}

int
hypre_BoomerAMGSetSym( void     *data,
                            int       sym)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataSym(amg_data) = sym;

   return (ierr);
}

int
hypre_BoomerAMGSetLevel( void     *data,
                            int       level)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataLevel(amg_data) = level;

   return (ierr);
}

int
hypre_BoomerAMGSetThreshold( void     *data,
                             double    thresh)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataThreshold(amg_data) = thresh;

   return (ierr);
}

int
hypre_BoomerAMGSetFilter( void     *data,
                          double    filter)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataFilter(amg_data) = filter;

   return (ierr);
}

int
hypre_BoomerAMGSetDropTol( void     *data,
                           double    drop_tol)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataDropTol(amg_data) = drop_tol;

   return (ierr);
}

int
hypre_BoomerAMGSetMaxNzPerRow( void     *data,
                               int       max_nz_per_row)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxNzPerRow(amg_data) = max_nz_per_row;

   return (ierr);
}

int
hypre_BoomerAMGSetEuclidFile( void     *data,
                              char     *euclidfile)
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return (ierr);
}

