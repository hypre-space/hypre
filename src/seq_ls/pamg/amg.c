/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/





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
   HYPRE_Int      max_levels;
   double   strong_threshold;
   double   A_trunc_factor;
   double   P_trunc_factor;
   HYPRE_Int      A_max_elmts;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      agg_coarsen_type;
   HYPRE_Int      interp_type;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      num_functions;
   HYPRE_Int      agg_levels;
   HYPRE_Int      num_relax_steps;
   HYPRE_Int      num_jacs;
   HYPRE_Int use_block_flag;

   /* solve params */
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_type;    
 
   double  *relax_weight;
   double   tol;

   HYPRE_Int     *num_grid_sweeps;  
   HYPRE_Int     *grid_relax_type;   
   HYPRE_Int    **grid_relax_points; 
   HYPRE_Int     *schwarz_option;
   HYPRE_Int      mode;


   /* output params */
   HYPRE_Int      ioutdat;
   char     log_file_name[256];

   HYPRE_Int      j;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   strong_threshold = 0.25;
   A_trunc_factor = 0;
   P_trunc_factor = 0;
   A_max_elmts = 0;
   P_max_elmts = 0;
   coarsen_type = 0;
   agg_coarsen_type = 8;
   agg_levels = 0;
   interp_type = 200;
   agg_interp_type = 5;
   num_functions = 1;
   num_relax_steps = 1;
   num_jacs = 1;
   use_block_flag = 0;
   mode = 0;

   /* solve params */
   max_iter  = 100;
   cycle_type = 1;
   tol = 1.0e-7;
   relax_weight = hypre_CTAlloc(double, max_levels);
   schwarz_option = hypre_CTAlloc(HYPRE_Int, max_levels);
   for (j = 0; j < max_levels; j++)
   {
      relax_weight[j] = 1.0;
      schwarz_option[j] = -1;
   }

   num_grid_sweeps = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_type = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_points = hypre_CTAlloc(HYPRE_Int *,4);

   for (j = 0; j < 3; j++)
   {
      num_grid_sweeps[j] = 2;
      grid_relax_type[j] = 0; 
      grid_relax_points[j] = hypre_CTAlloc(HYPRE_Int,2); 
      grid_relax_points[j][0] = 1;
      grid_relax_points[j][1] = -1;
   }
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 9;
   grid_relax_points[3] = hypre_CTAlloc(HYPRE_Int,1);
   grid_relax_points[3][0] = 9;

   /* output params */
   ioutdat = 0;
   hypre_sprintf(log_file_name, "%s", "amg.out.log");

   /*-----------------------------------------------------------------------
    * Create the hypre_AMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_AMGData, 1);

   hypre_AMGSetMaxLevels(amg_data, max_levels);
   hypre_AMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_AMGSetCoarsenType(amg_data, coarsen_type);
   hypre_AMGSetAggCoarsenType(amg_data, agg_coarsen_type);
   hypre_AMGSetAggLevels(amg_data, agg_levels);
   hypre_AMGSetInterpType(amg_data, interp_type);
   hypre_AMGSetAggInterpType(amg_data, agg_interp_type);
   hypre_AMGSetNumFunctions(amg_data, num_functions);
   hypre_AMGSetNumRelaxSteps(amg_data, num_relax_steps);
   hypre_AMGSetNumJacs(amg_data, num_jacs);
   hypre_AMGSetATruncFactor(amg_data, A_trunc_factor);
   hypre_AMGSetPTruncFactor(amg_data, P_trunc_factor);
   hypre_AMGSetAMaxElmts(amg_data, A_max_elmts);
   hypre_AMGSetPMaxElmts(amg_data, P_max_elmts);
   hypre_AMGSetUseBlockFlag(amg_data, use_block_flag);

   hypre_AMGSetMaxIter(amg_data, max_iter);
   hypre_AMGSetCycleType(amg_data, cycle_type);
   hypre_AMGSetTol(amg_data, tol); 
   hypre_AMGSetNumGridSweeps(amg_data, num_grid_sweeps);
   hypre_AMGSetGridRelaxType(amg_data, grid_relax_type);
   hypre_AMGSetGridRelaxPoints(amg_data, grid_relax_points);
   hypre_AMGSetRelaxWeight(amg_data, relax_weight);
   hypre_AMGSetSchwarzOption(amg_data, schwarz_option);
   hypre_AMGSetMode(amg_data, mode);

   hypre_AMGSetIOutDat(amg_data, ioutdat);
   hypre_AMGSetLogFileName(amg_data, log_file_name); 
   
   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMGFinalize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGFinalize( void *data )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int i;
   hypre_AMGData  *amg_data = data;
   HYPRE_Int num_levels = hypre_AMGDataNumLevels(amg_data);

   if (hypre_AMGDataNumGridSweeps(amg_data))
      hypre_TFree (hypre_AMGDataNumGridSweeps(amg_data));
   if (hypre_AMGDataGridRelaxType(amg_data))
      hypre_TFree (hypre_AMGDataGridRelaxType(amg_data));
   if (hypre_AMGDataRelaxWeight(amg_data))
      hypre_TFree (hypre_AMGDataRelaxWeight(amg_data));
   if (hypre_AMGDataSchwarzOption(amg_data)[0] > -1)
   {
      hypre_TFree (hypre_AMGDataNumDomains(amg_data));
      for (i=0; i < num_levels; i++)
      {
         hypre_TFree (hypre_AMGDataIDomainDof(amg_data)[i]);
         hypre_TFree (hypre_AMGDataJDomainDof(amg_data)[i]);
         hypre_TFree (hypre_AMGDataDomainMatrixInverse(amg_data)[i]);
      }
      hypre_TFree (hypre_AMGDataIDomainDof(amg_data));
      hypre_TFree (hypre_AMGDataJDomainDof(amg_data));
      hypre_TFree (hypre_AMGDataDomainMatrixInverse(amg_data));
   }
   if (hypre_AMGDataSchwarzOption(amg_data))
      hypre_TFree (hypre_AMGDataSchwarzOption(amg_data));
   if (hypre_AMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
        hypre_TFree (hypre_AMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree (hypre_AMGDataGridRelaxPoints(amg_data));
   }

   for (i=1; i < num_levels; i++)
   {
      hypre_SeqVectorDestroy(hypre_AMGDataFArray(amg_data)[i]);
      hypre_SeqVectorDestroy(hypre_AMGDataUArray(amg_data)[i]);
      hypre_CSRMatrixDestroy(hypre_AMGDataAArray(amg_data)[i]);
      if(hypre_AMGDataUseBlockFlag(amg_data)) {
	hypre_BCSRMatrixDestroy(hypre_AMGDataBArray(amg_data)[i]);
	hypre_BCSRMatrixDestroy(hypre_AMGDataPBArray(amg_data)[i]);
      }
      hypre_CSRMatrixDestroy(hypre_AMGDataPArray(amg_data)[i-1]);
      hypre_TFree(hypre_AMGDataCFMarkerArray(amg_data)[i-1]);
      hypre_TFree(hypre_AMGDataDofFuncArray(amg_data)[i-1]);
   }
   hypre_SeqVectorDestroy(hypre_AMGDataVtemp(amg_data));
   hypre_TFree(hypre_AMGDataFArray(amg_data));
   hypre_TFree(hypre_AMGDataUArray(amg_data));
   hypre_TFree(hypre_AMGDataAArray(amg_data));
   hypre_TFree(hypre_AMGDataBArray(amg_data));
   hypre_TFree(hypre_AMGDataPArray(amg_data));
   hypre_TFree(hypre_AMGDataPBArray(amg_data));
   hypre_TFree(hypre_AMGDataCFMarkerArray(amg_data));
   hypre_TFree(hypre_AMGDataDofFuncArray(amg_data)[num_levels-1]);
   hypre_TFree(hypre_AMGDataDofFuncArray(amg_data));

   hypre_TFree(amg_data);
   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGSetMaxLevels( void *data,
                       HYPRE_Int   max_levels )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxLevels(amg_data) = max_levels;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetStrongThreshold( void     *data,
                             double    strong_threshold )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataStrongThreshold(amg_data) = strong_threshold;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetMode( void     *data,
                             HYPRE_Int    mode )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMode(amg_data) = mode;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetATruncFactor( void     *data,
                          double    A_trunc_factor)
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataATruncFactor(amg_data) = A_trunc_factor;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetPTruncFactor( void     *data,
                          double    P_trunc_factor)
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataPTruncFactor(amg_data) = P_trunc_factor;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetAMaxElmts( void     *data,
                       HYPRE_Int       A_max_elmts)
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataAMaxElmts(amg_data) = A_max_elmts;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetPMaxElmts( void     *data,
                       HYPRE_Int       P_max_elmts)
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataPMaxElmts(amg_data) = P_max_elmts;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetCoarsenType( void     *data,
                        HYPRE_Int       coarsen_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataCoarsenType(amg_data) = coarsen_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetAggCoarsenType( void     *data,
                        HYPRE_Int       agg_coarsen_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataAggCoarsenType(amg_data) = agg_coarsen_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetAggLevels( void     *data,
                        HYPRE_Int       agg_levels )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataAggLevels(amg_data) = agg_levels;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetInterpType( void     *data,
                        HYPRE_Int       interp_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataInterpType(amg_data) = interp_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetAggInterpType( void     *data,
                        HYPRE_Int       agg_interp_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataAggInterpType(amg_data) = agg_interp_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetNumJacs( void     *data,
                     HYPRE_Int       num_jacs )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumJacs(amg_data) = num_jacs;

   return (ierr);
} 

HYPRE_Int
hypre_AMGSetMaxIter( void     *data,
                     HYPRE_Int       max_iter )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataMaxIter(amg_data) = max_iter;

   return (ierr);
} 

HYPRE_Int
hypre_AMGSetCycleType( void  *data,
                       HYPRE_Int    cycle_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataCycleType(amg_data) = cycle_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetTol( void     *data,
                 double    tol  )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataTol(amg_data) = tol;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetNumRelaxSteps( void     *data,
                           HYPRE_Int      num_relax_steps )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataNumRelaxSteps(amg_data) = num_relax_steps;

   return (ierr);
}
 
HYPRE_Int
hypre_AMGSetNumGridSweeps( void     *data,
                           HYPRE_Int      *num_grid_sweeps )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataNumGridSweeps(amg_data));
   hypre_AMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return (ierr);
}
 
HYPRE_Int
hypre_AMGSetGridRelaxType( void     *data,
                           HYPRE_Int      *grid_relax_type )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataGridRelaxType(amg_data));
   hypre_AMGDataGridRelaxType(amg_data) = grid_relax_type;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetGridRelaxPoints( void     *data,
                             HYPRE_Int      **grid_relax_points )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int i;
   hypre_AMGData  *amg_data = data;

   
   if (hypre_AMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
         hypre_TFree(hypre_AMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree(hypre_AMGDataGridRelaxPoints(amg_data));
   }
   hypre_AMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return (ierr);
}

HYPRE_Int
hypre_AMGSetRelaxWeight( void     *data,
                         double   *relax_weight )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataRelaxWeight(amg_data));
   hypre_AMGDataRelaxWeight(amg_data) = relax_weight; 

   return (ierr);
}

HYPRE_Int
hypre_AMGSetSchwarzOption( void     *data,
                         HYPRE_Int   *schwarz_option )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_TFree(hypre_AMGDataSchwarzOption(amg_data));
   hypre_AMGDataSchwarzOption(amg_data) = schwarz_option;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetIOutDat( void     *data,
                     HYPRE_Int       ioutdat )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_AMGDataIOutDat(amg_data) = ioutdat;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetLogFileName( void   *data,
                         char   *log_file_name )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;

   hypre_sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name);

   return (ierr);
}

HYPRE_Int
hypre_AMGSetLogging( void     *data,
                     HYPRE_Int       ioutdat,
                     char     *log_file_name )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   FILE *fp;

   hypre_AMGDataIOutDat(amg_data) = ioutdat;
   if (ioutdat > 0)
   {
         hypre_sprintf(hypre_AMGDataLogFileName(amg_data), "%s", log_file_name); 
       
   	fp = fopen(hypre_AMGDataLogFileName(amg_data),"w");
   	fclose(fp);
   }

   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMGSetUseBlockFlag( void     *data,
                          HYPRE_Int       use_block_flag )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataUseBlockFlag(amg_data) = use_block_flag;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetNumFunctions( void     *data,
                         HYPRE_Int       num_functions )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumFunctions(amg_data) = num_functions;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetNumPoints( void     *data,
                       HYPRE_Int       num_points )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_AMGDataNumPoints(amg_data) = num_points;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetDofFunc( void     *data,
                        HYPRE_Int      *dof_func )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataDofFunc(amg_data));
   hypre_AMGDataDofFunc(amg_data) = dof_func;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetDofPoint( void     *data,
                      HYPRE_Int      *dof_point )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataDofPoint(amg_data));
   hypre_AMGDataDofPoint(amg_data) = dof_point;

   return (ierr);
}

HYPRE_Int
hypre_AMGSetPointDofMap( void     *data,
                         HYPRE_Int      *point_dof_map  )
{
   HYPRE_Int ierr = 0;
   hypre_AMGData  *amg_data = data;
 
   hypre_TFree(hypre_AMGDataPointDofMap(amg_data));
   hypre_AMGDataPointDofMap(amg_data) = point_dof_map;

   return (ierr);
}

