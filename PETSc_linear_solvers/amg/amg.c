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

/*--------------------------------------------------------------------------
 * hypre_AMGInitialize
 *--------------------------------------------------------------------------*/

void  *hypre_AMGInitialize()

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

   amg_data = hypre_AMGNewData(max_levels, strong_threshold, interp_type,
                               max_iter, cycle_type, tol, num_grid_sweeps,
                               grid_relax_type, grid_relax_points, 
                               ioutdat, log_file_name);

                     
   return (void *) amg_data;
}





/*--------------------------------------------------------------------------
 * hypre_AMGNewData
 *--------------------------------------------------------------------------*/
 
hypre_AMGData *hypre_AMGNewData(max_levels, strong_threshold, interp_type,
                               max_iter, cycle_type, tol, num_grid_sweeps,
                               grid_relax_type, grid_relax_points, 
                               ioutdat, log_file_name)

int      max_levels;
double   strong_threshold;
int      interp_type;
int      max_iter;
int      cycle_type;    
double   tol;
int     *num_grid_sweeps;  
int     *grid_relax_type;   
int     *grid_relax_points[4]; 
int      ioutdat;
char     log_file_name[256];

{
   hypre_AMGData  *amg_data;

   amg_data = hypre_CTAlloc(hypre_AMGData, 1);

   hypre_AMGSetMaxLevels(max_levels, amg_data);
   hypre_AMGSetStrongThreshold(strong_threshold, amg_data);
   hypre_AMGSetInterpType(interp_type, amg_data);

   hypre_AMGSetMaxIter(max_iter, amg_data);
   hypre_AMGSetCycleType(cycle_type, amg_data);
   hypre_AMGSetTol(tol, amg_data); 
   hypre_AMGSetNumGridSweeps(num_grid_sweeps, amg_data);
   hypre_AMGSetGridRelaxType(grid_relax_type, amg_data);
   hypre_AMGSetGridRelaxPoints(grid_relax_points, amg_data);

   hypre_AMGSetIOutDat(ioutdat, amg_data);
   hypre_AMGSetLogFileName(log_file_name, amg_data); 
   
   return amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMGFinalize
 *--------------------------------------------------------------------------*/

void   hypre_AMGFinalize(data)
void  *data;
{
   hypre_AMGData  *amg_data = data;


   /*  hypre_AMGFreeData(amg_data); */
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      hypre_AMGSetMaxLevels(max_levels, data)
int       max_levels;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxLevels(amg_data) = max_levels;
}



void      hypre_AMGSetStrongThreshold(strong_threshold, data)
double    strong_threshold;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataStrongThreshold(amg_data) = strong_threshold;
}


void      hypre_AMGSetInterpType(interp_type, data)
int       interp_type;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataInterpType(amg_data) = interp_type;
}

void hypre_AMGSetMaxIter(max_iter, data)
int       max_iter;
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxIter(amg_data) = max_iter;
} 


void  hypre_AMGSetCycleType(cycle_type, data)
int    cycle_type;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataCycleType(amg_data) = cycle_type;
}


void   hypre_AMGSetTol(tol, data) 
double    tol;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataTol(amg_data) = tol;
}


void   hypre_AMGSetNumGridSweeps(num_grid_sweeps, data)
int      *num_grid_sweeps;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataNumGridSweeps(amg_data));
   hypre_AMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
}
 
  
void   hypre_AMGSetGridRelaxType(grid_relax_type, data)
int      *grid_relax_type;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataGridRelaxType(amg_data));
   hypre_AMGDataGridRelaxType(amg_data) = grid_relax_type;
}

void   hypre_AMGSetGridRelaxPoints(grid_relax_points, data)
int      **grid_relax_points;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataGridRelaxPoints(amg_data) = grid_relax_points; 
}

   
void   hypre_AMGSetIOutDat(ioutdat, data)
int       ioutdat;
void     *data;
{
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataIOutDat(amg_data) = ioutdat;
}

void   hypre_AMGSetLogFileName(log_file_name, data) 
char   *log_file_name;
void   *data;
{
   hypre_AMGData  *amg_data = data;

   sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name);
}

 


void      hypre_AMGSetLogging(ioutdat, log_file_name, data)
int       ioutdat;
char     *log_file_name;
void     *data;
{
   FILE *fp;

   hypre_AMGData  *amg_data = data;
 
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
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      hypre_AMGSetNumUnknowns(num_unknowns, data)
int       num_unknowns;  
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumUnknowns(amg_data) = num_unknowns;
}

void      hypre_AMGSetNumPoints(num_points, data)
int       num_points;    
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumPoints(amg_data) = num_points;
}


void      hypre_AMGSetUnknownMap(unknown_map, data)
int      *unknown_map;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataUnknownMap(amg_data));
   hypre_AMGDataUnknownMap(amg_data) = unknown_map;
}

void      hypre_AMGSetPointMap(point_map, data)
int      *point_map;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataPointMap(amg_data));
   hypre_AMGDataPointMap(amg_data) = point_map;
}

void      hypre_AMGSetVatPoint(v_at_point, data)
int      *v_at_point;            
void     *data;
{
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataVatPoint(amg_data));
   hypre_AMGDataVatPoint(amg_data) = v_at_point;
}


