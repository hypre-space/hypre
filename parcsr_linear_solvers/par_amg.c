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
 * hypre_ParAMGInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_ParAMGInitialize()
{
   hypre_ParAMGData  *amg_data;

   /* setup params */
   int      max_levels;
   double   strong_threshold;
   int      interp_type;
   int      coarsen_type;
   int      measure_type;

   /* solve params */
   int      max_iter;
   int      cycle_type;    
 
   double   tol;

   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   double   relax_weight;


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
   interp_type = 200;
   coarsen_type = 0;
   measure_type = 0;

   /* solve params */
   max_iter  = 20;
   cycle_type = 1;
   tol = 1.0e-7;

   num_grid_sweeps = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);
   relax_weight = 2.0/3.0;

   for (j = 0; j < 3; j++)
   {
      num_grid_sweeps[j] = 2;
      grid_relax_type[j] = 0; 
      grid_relax_points[j] = hypre_CTAlloc(int,2); 
      grid_relax_points[j][0] = 1;
      grid_relax_points[j][1] = -1;
   }
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(int,1);
   grid_relax_points[3][0] = 9;

   /* output params */
   ioutdat = 0;
   sprintf(log_file_name, "%s", "amg.out.log");
   cycle_op_count = 0;
   debug_flag = 0;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1);

   hypre_ParAMGSetMaxLevels(amg_data, max_levels);
   hypre_ParAMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_ParAMGSetInterpType(amg_data, interp_type);
   hypre_ParAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_ParAMGSetMeasureType(amg_data, measure_type);

   hypre_ParAMGSetMaxIter(amg_data, max_iter);
   hypre_ParAMGSetCycleType(amg_data, cycle_type);
   hypre_ParAMGSetTol(amg_data, tol); 
   hypre_ParAMGSetNumGridSweeps(amg_data, num_grid_sweeps);
   hypre_ParAMGSetGridRelaxType(amg_data, grid_relax_type);
   hypre_ParAMGSetGridRelaxPoints(amg_data, grid_relax_points);
   hypre_ParAMGSetRelaxWeight(amg_data, relax_weight);

   hypre_ParAMGSetIOutDat(amg_data, ioutdat);
   hypre_ParAMGSetLogFileName(amg_data, log_file_name); 
   hypre_ParAMGSetDebugFlag(amg_data, debug_flag);

   hypre_ParAMGSetRestriction(amg_data, 0);
   
   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGFinalize
 *--------------------------------------------------------------------------*/

int
hypre_ParAMGFinalize( void *data )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
   int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   int i;

/*   hypre_ParAMGFreeData(amg_data);*/
   hypre_TFree (hypre_ParAMGDataNumGridSweeps(amg_data));
   hypre_TFree (hypre_ParAMGDataGridRelaxType(amg_data));
   for (i=0; i < 4; i++)
   	hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data)[i]);
   hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data));
   for (i=1; i < num_levels; i++)
   {
	hypre_DestroyParVector(hypre_ParAMGDataFArray(amg_data)[i]);
	hypre_DestroyParVector(hypre_ParAMGDataUArray(amg_data)[i]);
	hypre_DestroyParCSRMatrix(hypre_ParAMGDataAArray(amg_data)[i]);
	hypre_DestroyParCSRMatrix(hypre_ParAMGDataPArray(amg_data)[i-1]);
	hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data)[i-1]);
   }
   hypre_DestroyParVector(hypre_ParAMGDataVtemp(amg_data));
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
hypre_ParAMGSetRestriction( void *data,
                            int   restr_par )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return (ierr);
}

int
hypre_ParAMGSetMaxLevels( void *data,
                          int   max_levels )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return (ierr);
}

int
hypre_ParAMGSetStrongThreshold( void     *data,
                                double    strong_threshold )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return (ierr);
}

int
hypre_ParAMGSetInterpType( void     *data,
                           int       interp_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return (ierr);
}

int
hypre_ParAMGSetMaxIter( void     *data,
                        int       max_iter )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return (ierr);
} 

int
hypre_ParAMGSetCoarsenType( void  *data,
                          int    coarsen_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return (ierr);
}

int
hypre_ParAMGSetMeasureType( void  *data,
                            int    measure_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return (ierr);
}

int
hypre_ParAMGSetCycleType( void  *data,
                          int    cycle_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return (ierr);
}

int
hypre_ParAMGSetTol( void     *data,
                    double    tol  )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataTol(amg_data) = tol;

   return (ierr);
}

int
hypre_ParAMGSetNumGridSweeps( void     *data,
                              int      *num_grid_sweeps )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data));
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return (ierr);
}
 
int
hypre_ParAMGSetGridRelaxType( void     *data,
                              int      *grid_relax_type )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data));
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;

   return (ierr);
}

int
hypre_ParAMGSetGridRelaxPoints( void     *data,
                                int      **grid_relax_points )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return (ierr);
}

int
hypre_ParAMGSetRelaxWeight( void     *data,
                            double    relax_weight )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
               
   hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;
   
   return (ierr);
}

int
hypre_ParAMGSetIOutDat( void     *data,
                        int       ioutdat )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataIOutDat(amg_data) = ioutdat;

   return (ierr);
}

int
hypre_ParAMGSetLogFileName( void   *data,
                            char   *log_file_name )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;

   sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", log_file_name);

   return (ierr);
}

int
hypre_ParAMGSetLogging( void     *data,
                        int       ioutdat,
                        char     *log_file_name )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   FILE *fp;

   hypre_ParAMGDataIOutDat(amg_data) = ioutdat;
   if (ioutdat > 0)
   {
      if (*log_file_name == 0)  
         sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", "amg.out.log");
      else
         sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", log_file_name); 
       
      fp = fopen(hypre_ParAMGDataLogFileName(amg_data),"w");
      fclose(fp);
   }

   return (ierr);
}

int
hypre_ParAMGSetDebugFlag( void     *data,
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
hypre_ParAMGSetNumUnknowns( void     *data,
                            int       num_unknowns )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumUnknowns(amg_data) = num_unknowns;

   return (ierr);
}

int
hypre_ParAMGSetNumPoints( void     *data,
                          int       num_points )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return (ierr);
}

int
hypre_ParAMGSetUnknownMap( void     *data,
                           int      *unknown_map )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataUnknownMap(amg_data));
   hypre_ParAMGDataUnknownMap(amg_data) = unknown_map;

   return (ierr);
}

int
hypre_ParAMGSetPointMap( void     *data,
                         int      *point_map )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataPointMap(amg_data));
   hypre_ParAMGDataPointMap(amg_data) = point_map;

   return (ierr);
}

int
hypre_ParAMGSetVatPoint( void     *data,
                         int      *v_at_point )
{
   int ierr = 0;
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataVatPoint(amg_data));
   hypre_ParAMGDataVatPoint(amg_data) = v_at_point;

   return (ierr);
}

