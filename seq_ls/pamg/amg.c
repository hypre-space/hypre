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
 * AMG functions
 *
 *****************************************************************************/

#include "headers.h"
#include "amg.h"

/*--------------------------------------------------------------------------
 * hypre_AMGInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_AMGInitialize()
{
   hypre_AMGData  *amg_data;

   /* setup params */
   int      max_levels;
   double   strong_threshold;
   int      interp_type;

   /* solve params */
   int      max_iter;
   int      cycle_type;    
 
   double   tol;

   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 


   /* output params */
   int      ioutdat;
   int      cycle_op_count;
   char     log_file_name[256];

   int      j;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   strong_threshold = 0.25;
   interp_type = 200;

   /* solve params */
   max_iter  = 20;
   cycle_type = 1;
   tol = 1.0e-7;

   num_grid_sweeps = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_CTAlloc(int,4);
   grid_relax_points = hypre_CTAlloc(int *,4);

   for (j = 0; j < 3; j++)
   {
      num_grid_sweeps[j] = 2;
      grid_relax_type[j] = 1; 
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

   /*-----------------------------------------------------------------------
    * Create the hypre_AMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_AMGData, 1);

   hypre_AMGSetMaxLevels(amg_data, max_levels);
   hypre_AMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_AMGSetInterpType(amg_data, interp_type);

   hypre_AMGSetMaxIter(amg_data, max_iter);
   hypre_AMGSetCycleType(amg_data, cycle_type);
   hypre_AMGSetTol(amg_data, tol); 
   hypre_AMGSetNumGridSweeps(amg_data, num_grid_sweeps);
   hypre_AMGSetGridRelaxType(amg_data, grid_relax_type);
   hypre_AMGSetGridRelaxPoints(amg_data, grid_relax_points);

   hypre_AMGSetIOutDat(amg_data, ioutdat);
   hypre_AMGSetLogFileName(amg_data, log_file_name); 
   
   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMGFinalize
 *--------------------------------------------------------------------------*/

int
hypre_AMGFinalize( void *data )
{
   int ierr = 0;
/*   hypre_AMGData  *amg_data = data;*/

/*   hypre_AMGFreeData(amg_data);*/

   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

int
hypre_AMGSetMaxLevels( void *data,
                       int   max_levels )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxLevels(amg_data) = max_levels;

   return (ierr);
}

int
hypre_AMGSetStrongThreshold( void     *data,
                             double    strong_threshold )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataStrongThreshold(amg_data) = strong_threshold;

   return (ierr);
}

int
hypre_AMGSetInterpType( void     *data,
                        int       interp_type )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataInterpType(amg_data) = interp_type;

   return (ierr);
}

int
hypre_AMGSetMaxIter( void     *data,
                     int       max_iter )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxIter(amg_data) = max_iter;

   return (ierr);
} 

int
hypre_AMGSetCycleType( void  *data,
                       int    cycle_type )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataCycleType(amg_data) = cycle_type;

   return (ierr);
}

int
hypre_AMGSetTol( void     *data,
                 double    tol  )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataTol(amg_data) = tol;

   return (ierr);
}

int
hypre_AMGSetNumGridSweeps( void     *data,
                           int      *num_grid_sweeps )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataNumGridSweeps(amg_data));
   hypre_AMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return (ierr);
}
 
int
hypre_AMGSetGridRelaxType( void     *data,
                           int      *grid_relax_type )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataGridRelaxType(amg_data));
   hypre_AMGDataGridRelaxType(amg_data) = grid_relax_type;

   return (ierr);
}

int
hypre_AMGSetGridRelaxPoints( void     *data,
                             int      **grid_relax_points )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return (ierr);
}

int
hypre_AMGSetIOutDat( void     *data,
                     int       ioutdat )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataIOutDat(amg_data) = ioutdat;

   return (ierr);
}

int
hypre_AMGSetLogFileName( void   *data,
                         char   *log_file_name )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;

   sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name);

   return (ierr);
}

int
hypre_AMGSetLogging( void     *data,
                     int       ioutdat,
                     char     *log_file_name )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   FILE *fp;

   hypre_AMGDataIOutDat(amg_data) = ioutdat;
   if (ioutdat > 0)
   {
      if (*log_file_name == 0)  
         sprintf(hypre_AMGDataLogFileName(amg_data), "%s", "amg.out.log");
      else
         sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name); 
       
   fp = fopen(hypre_AMGDataLogFileName(amg_data),"w");
   fclose(fp);
   }

   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

int
hypre_AMGSetNumUnknowns( void     *data,
                         int       num_unknowns )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumUnknowns(amg_data) = num_unknowns;

   return (ierr);
}

int
hypre_AMGSetNumPoints( void     *data,
                       int       num_points )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumPoints(amg_data) = num_points;

   return (ierr);
}

int
hypre_AMGSetUnknownMap( void     *data,
                        int      *unknown_map )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataUnknownMap(amg_data));
   hypre_AMGDataUnknownMap(amg_data) = unknown_map;

   return (ierr);
}

int
hypre_AMGSetPointMap( void     *data,
                      int      *point_map )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataPointMap(amg_data));
   hypre_AMGDataPointMap(amg_data) = point_map;

   return (ierr);
}

int
hypre_AMGSetVatPoint( void     *data,
                      int      *v_at_point )
{
   int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataVatPoint(amg_data));
   hypre_AMGDataVatPoint(amg_data) = v_at_point;

   return (ierr);
}

