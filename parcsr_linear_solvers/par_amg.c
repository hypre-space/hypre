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
 * hypre_ParAMGInitialize
 *--------------------------------------------------------------------------*/

void  *hypre_ParAMGInitialize()

{
   hypre_ParAMGData  *amg_data;

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
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_ParAMGNewData(max_levels, strong_threshold, interp_type,
                               max_iter, cycle_type, tol, num_grid_sweeps,
                               grid_relax_type, grid_relax_points, 
                               ioutdat, log_file_name);

                     
   return (void *) amg_data;
}





/*--------------------------------------------------------------------------
 * hypre_ParAMGNewData
 *--------------------------------------------------------------------------*/
 
hypre_ParAMGData *hypre_ParAMGNewData(max_levels, strong_threshold, interp_type,
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
   hypre_ParAMGData  *amg_data;

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1);

   hypre_ParAMGSetMaxLevels(max_levels, amg_data);
   hypre_ParAMGSetStrongThreshold(strong_threshold, amg_data);
   hypre_ParAMGSetInterpType(interp_type, amg_data);

   hypre_ParAMGSetMaxIter(max_iter, amg_data);
   hypre_ParAMGSetCycleType(cycle_type, amg_data);
   hypre_ParAMGSetTol(tol, amg_data); 
   hypre_ParAMGSetNumGridSweeps(num_grid_sweeps, amg_data);
   hypre_ParAMGSetGridRelaxType(grid_relax_type, amg_data);
   hypre_ParAMGSetGridRelaxPoints(grid_relax_points, amg_data);

   hypre_ParAMGSetIOutDat(ioutdat, amg_data);
   hypre_ParAMGSetLogFileName(log_file_name, amg_data); 
   
   return amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_ParAMGFinalize
 *--------------------------------------------------------------------------*/

void   hypre_ParAMGFinalize(data)
void  *data;
{
/*   hypre_ParAMGData  *amg_data = data;*/

/*   hypre_ParAMGFreeData(amg_data);*/
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

void      hypre_ParAMGSetMaxLevels(max_levels, data)
int       max_levels;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;
}



void      hypre_ParAMGSetStrongThreshold(strong_threshold, data)
double    strong_threshold;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;
}


void      hypre_ParAMGSetInterpType(interp_type, data)
int       interp_type;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataInterpType(amg_data) = interp_type;
}

void hypre_ParAMGSetMaxIter(max_iter, data)
int       max_iter;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataMaxIter(amg_data) = max_iter;
} 


void  hypre_ParAMGSetCycleType(cycle_type, data)
int    cycle_type;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;
}


void   hypre_ParAMGSetTol(tol, data) 
double    tol;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataTol(amg_data) = tol;
}


void   hypre_ParAMGSetNumGridSweeps(num_grid_sweeps, data)
int      *num_grid_sweeps;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data));
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
}
 
  
void   hypre_ParAMGSetGridRelaxType(grid_relax_type, data)
int      *grid_relax_type;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data));
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
}

void   hypre_ParAMGSetGridRelaxPoints(grid_relax_points, data)
int      **grid_relax_points;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points; 
}

   
void   hypre_ParAMGSetIOutDat(ioutdat, data)
int       ioutdat;
void     *data;
{
   hypre_ParAMGData  *amg_data = data;

   hypre_ParAMGDataIOutDat(amg_data) = ioutdat;
}

void   hypre_ParAMGSetLogFileName(log_file_name, data) 
char   *log_file_name;
void   *data;
{
   hypre_ParAMGData  *amg_data = data;

   sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", log_file_name);
}

 


void      hypre_ParAMGSetLogging(ioutdat, log_file_name, data)
int       ioutdat;
char     *log_file_name;
void     *data;
{
   FILE *fp;

   hypre_ParAMGData  *amg_data = data;
 
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
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

void      hypre_ParAMGSetNumUnknowns(num_unknowns, data)
int       num_unknowns;  
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumUnknowns(amg_data) = num_unknowns;
}

void      hypre_ParAMGSetNumPoints(num_points, data)
int       num_points;    
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_ParAMGDataNumPoints(amg_data) = num_points;
}


void      hypre_ParAMGSetUnknownMap(unknown_map, data)
int      *unknown_map;            
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataUnknownMap(amg_data));
   hypre_ParAMGDataUnknownMap(amg_data) = unknown_map;
}

void      hypre_ParAMGSetPointMap(point_map, data)
int      *point_map;            
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataPointMap(amg_data));
   hypre_ParAMGDataPointMap(amg_data) = point_map;
}

void      hypre_ParAMGSetVatPoint(v_at_point, data)
int      *v_at_point;            
void     *data;
{
   hypre_ParAMGData  *amg_data = data;
 
   hypre_TFree(hypre_ParAMGDataVatPoint(amg_data));
   hypre_ParAMGDataVatPoint(amg_data) = v_at_point;
}


