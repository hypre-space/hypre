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
   int      interp_type;
   int      coarsen_type;
   int      measure_type;

   /* solve params */
   int      min_iter;
   int      max_iter;
   int      cycle_type;    
 
   double   tol;

   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   double  *relax_weight;

   /* log info */
   int      num_iterations;

   /* output params */
   int      ioutdat;
   int      cycle_op_count;
   char     log_file_name[256];
   int      debug_flag;

   int      j;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   strong_threshold = 0.25;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   interp_type = 200;
   coarsen_type = 6;
   measure_type = 0;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   cycle_type = 1;
   tol = 1.0e-7;

   num_grid_sweeps = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight = hypre_CTAlloc(double,max_levels);

   for (j = 0; j < max_levels; j++)
   {
      relax_weight[j] = 1.0;
   }

   for (j = 0; j < 3; j++)
   {
      num_grid_sweeps[j] = 2;
      grid_relax_type[j] = 3; 
      grid_relax_points[j] = hypre_CTAlloc(int,2); 
      grid_relax_points[j][0] = 1;
      grid_relax_points[j][1] = -1;
   }
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int,1);
   grid_relax_points[3][0] = 0;

   /* log info */
   num_iterations = 0;

   /* output params */
   ioutdat = 0;
   sprintf(log_file_name, "%s", "amg.out.log");
   cycle_op_count = 0;
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
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);

   hypre_BoomerAMGSetMinIter(amg_data, min_iter);
   hypre_BoomerAMGSetMaxIter(amg_data, max_iter);
   hypre_BoomerAMGSetCycleType(amg_data, cycle_type);
   hypre_BoomerAMGSetTol(amg_data, tol); 
   hypre_BoomerAMGSetNumGridSweeps(amg_data, num_grid_sweeps);
   hypre_BoomerAMGSetGridRelaxType(amg_data, grid_relax_type);
   hypre_BoomerAMGSetGridRelaxPoints(amg_data, grid_relax_points);
   hypre_BoomerAMGSetRelaxWeight(amg_data, relax_weight);

   hypre_BoomerAMGSetNumIterations(amg_data, num_iterations);
   hypre_BoomerAMGSetIOutDat(amg_data, ioutdat);
   hypre_BoomerAMGSetLogFileName(amg_data, log_file_name); 
   hypre_BoomerAMGSetDebugFlag(amg_data, debug_flag);

   hypre_BoomerAMGSetRestriction(amg_data, 0);
   
   hypre_ParAMGDataAArray(amg_data) = NULL;
   hypre_ParAMGDataPArray(amg_data) = NULL;
   hypre_ParAMGDataRArray(amg_data) = NULL;
   hypre_ParAMGDataCFMarkerArray(amg_data) = NULL;
   hypre_ParAMGDataVtemp(amg_data)  = NULL;
   hypre_ParAMGDataFArray(amg_data) = NULL;
   hypre_ParAMGDataUArray(amg_data) = NULL;

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
   int i;

/*   hypre_ParAMGFreeData(amg_data);*/
   if (hypre_ParAMGDataNumGridSweeps(amg_data))
      hypre_TFree (hypre_ParAMGDataNumGridSweeps(amg_data));
   if (hypre_ParAMGDataGridRelaxType(amg_data))
      hypre_TFree (hypre_ParAMGDataGridRelaxType(amg_data));
   if (hypre_ParAMGDataRelaxWeight(amg_data))
      hypre_TFree (hypre_ParAMGDataRelaxWeight(amg_data));
   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
   	hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data));
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
   if (hypre_ParAMGDataRestriction(amg_data))
	hypre_TFree(hypre_ParAMGDataRArray(amg_data));
   hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data));
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
hypre_BoomerAMGSetIOutDat( void     *data,
                        int       ioutdat )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataIOutDat(amg_data) = ioutdat;

   return (ierr);
}

int
hypre_BoomerAMGSetLogFileName( void   *data,
                            char   *log_file_name )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", log_file_name);

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
hypre_BoomerAMGSetLogging( void     *data,
                        int       ioutdat,
                        char     *log_file_name )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   FILE *fp;

   hypre_ParAMGDataIOutDat(amg_data) = ioutdat;
/*   if (ioutdat > 0)
   {
      if (*log_file_name == 0)  
         sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", "amg.out.log");
      else
         sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", log_file_name); 
       
      fp = fopen(hypre_ParAMGDataLogFileName(amg_data),"w");
      fclose(fp);
   }
*/
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
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumUnknowns( void     *data,
                            int       num_unknowns )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumUnknowns(amg_data) = num_unknowns;

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
hypre_BoomerAMGSetUnknownMap( void     *data,
                           int      *unknown_map )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataUnknownMap(amg_data));
   hypre_ParAMGDataUnknownMap(amg_data) = unknown_map;

   return (ierr);
}

int
hypre_BoomerAMGSetPointMap( void     *data,
                         int      *point_map )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataPointMap(amg_data));
   hypre_ParAMGDataPointMap(amg_data) = point_map;

   return (ierr);
}

int
hypre_BoomerAMGSetVatPoint( void     *data,
                         int      *v_at_point )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataVatPoint(amg_data));
   hypre_ParAMGDataVatPoint(amg_data) = v_at_point;

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
hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                     double   *rel_resid_norm )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return (ierr);
}
