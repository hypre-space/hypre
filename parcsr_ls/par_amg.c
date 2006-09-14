/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


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
   double   jacobi_trunc_threshold;
   double   S_commpkg_switch;
   double   CR_rate;
   int      interp_type;
   int      coarsen_type;
   int      measure_type;
   int      setup_type;
   int      P_max_elmts;
   int 	    num_functions;
   int 	    nodal;
   int 	    num_paths;
   int 	    agg_num_levels;
   int      post_interp_type;
   int 	    num_CR_relax_steps;
   int 	    IS_type;
   int 	    CR_use_CG;

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

   int block_mode;
   

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
   jacobi_trunc_threshold = 0.01;
   S_commpkg_switch = 1.0;
   interp_type = 0;
   coarsen_type = 6;
   measure_type = 0;
   setup_type = 1;
   P_max_elmts = 0;
   num_functions = 1;
   nodal = 0;
   num_paths = 1;
   agg_num_levels = 0;
   post_interp_type = 0;
   num_CR_relax_steps = 2;
   CR_rate = 0.7;
   IS_type = 1;
   CR_use_CG = 0;

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

   block_mode = 0;

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
   hypre_BoomerAMGSetJacobiTruncThreshold(amg_data, jacobi_trunc_threshold);
   hypre_BoomerAMGSetSCommPkgSwitch(amg_data, S_commpkg_switch);
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetSetupType(amg_data, setup_type);
   hypre_BoomerAMGSetPMaxElmts(amg_data, P_max_elmts);
   hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   hypre_BoomerAMGSetNodal(amg_data, nodal);
   hypre_BoomerAMGSetNumPaths(amg_data, num_paths);
   hypre_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
   hypre_BoomerAMGSetPostInterpType(amg_data, post_interp_type);
   hypre_BoomerAMGSetNumCRRelaxSteps(amg_data, num_CR_relax_steps);
   hypre_BoomerAMGSetCRRate(amg_data, CR_rate);
   hypre_BoomerAMGSetISType(amg_data, IS_type);
   hypre_BoomerAMGSetCRUseCG(amg_data, CR_use_CG);
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
  
   hypre_ParAMGDataABlockArray(amg_data) = NULL;
   hypre_ParAMGDataPBlockArray(amg_data) = NULL;
   hypre_ParAMGDataRBlockArray(amg_data) = NULL;

   /* this can not be set by the user currently */
   hypre_ParAMGDataBlockMode(amg_data) = block_mode;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGDestroy( void *data )
{
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

        if (hypre_ParAMGDataAArray(amg_data)[i])
           hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(amg_data)[i]);

        if (hypre_ParAMGDataPArray(amg_data)[i-1])
           hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(amg_data)[i-1]);

	hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data)[i-1]);

        /* get rid of any block structures */ 
        if (hypre_ParAMGDataABlockArray(amg_data)[i])
           hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataABlockArray(amg_data)[i]);
    
        if (hypre_ParAMGDataPBlockArray(amg_data)[i-1])
           hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataPBlockArray(amg_data)[i-1]);

   }

   /* get rid of a fine level block matrix */
   if (hypre_ParAMGDataABlockArray(amg_data))
      if (hypre_ParAMGDataABlockArray(amg_data)[0])
           hypre_ParCSRBlockMatrixDestroy(hypre_ParAMGDataABlockArray(amg_data)[0]);


   /* see comments in par_coarsen.c regarding special case for CF_marker */
   if (num_levels == 1)
   {
      hypre_TFree(hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
   }
   hypre_ParVectorDestroy(hypre_ParAMGDataVtemp(amg_data));
   hypre_TFree(hypre_ParAMGDataFArray(amg_data));
   hypre_TFree(hypre_ParAMGDataUArray(amg_data));
   hypre_TFree(hypre_ParAMGDataAArray(amg_data));
   hypre_TFree(hypre_ParAMGDataABlockArray(amg_data));
   hypre_TFree(hypre_ParAMGDataPBlockArray(amg_data));
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
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetRestriction( void *data,
                            int   restr_par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetMaxLevels( void *data,
                          int   max_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (max_levels < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetMaxLevels( void *data,
                             int *  max_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_levels = hypre_ParAMGDataMaxLevels(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetStrongThreshold( void     *data,
                                double    strong_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetStrongThreshold( void     *data,
                                double *  strong_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetMaxRowSum( void     *data,
                          double    max_row_sum )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (max_row_sum <= 0 || max_row_sum > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxRowSum(amg_data) = max_row_sum;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetMaxRowSum( void     *data,
                          double *  max_row_sum )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetTruncFactor( void     *data,
                            double    trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (trunc_factor < 0 || trunc_factor >= 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataTruncFactor(amg_data) = trunc_factor;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetTruncFactor( void     *data,
                            double *  trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetPMaxElmts( void     *data,
                            int    P_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataPMaxElmts(amg_data) = P_max_elmts;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetPMaxElmts( void     *data,
                            int *  P_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetJacobiTruncThreshold( void     *data,
                            double    jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (jacobi_trunc_threshold < 0 || jacobi_trunc_threshold >= 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataJacobiTruncThreshold(amg_data) = jacobi_trunc_threshold;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetJacobiTruncThreshold( void     *data,
                            double *  jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetPostInterpType( void     *data,
                                  int    post_interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (post_interp_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataPostInterpType(amg_data) = post_interp_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetPostInterpType( void     *data,
                                  int  * post_interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSCommPkgSwitch( void     *data,
                                  double    S_commpkg_switch )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataSCommPkgSwitch(amg_data) = S_commpkg_switch;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSCommPkgSwitch( void     *data,
                                  double *  S_commpkg_switch )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *S_commpkg_switch = hypre_ParAMGDataSCommPkgSwitch(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetInterpType( void     *data,
                           int       interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (interp_type < 0 || interp_type > 11)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetInterpType( void     *data,
                           int *     interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *interp_type = hypre_ParAMGDataInterpType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetMinIter( void     *data,
                        int       min_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return hypre_error_flag;
} 

int
hypre_BoomerAMGGetMinIter( void     *data,
                        int *     min_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *min_iter = hypre_ParAMGDataMinIter(amg_data);

   return hypre_error_flag;
} 

int
hypre_BoomerAMGSetMaxIter( void     *data,
                        int     max_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (max_iter < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return hypre_error_flag;
} 

int
hypre_BoomerAMGGetMaxIter( void     *data,
                        int *   max_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_iter = hypre_ParAMGDataMaxIter(amg_data);

   return hypre_error_flag;
} 

int
hypre_BoomerAMGSetCoarsenType( void  *data,
                          int    coarsen_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetCoarsenType( void  *data,
                          int *  coarsen_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetMeasureType( void  *data,
                            int    measure_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetMeasureType( void  *data,
                            int *  measure_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *measure_type = hypre_ParAMGDataMeasureType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSetupType( void  *data,
                             int    setup_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataSetupType(amg_data) = setup_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSetupType( void  *data,
                             int  *  setup_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *setup_type = hypre_ParAMGDataSetupType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetCycleType( void  *data,
                          int    cycle_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (cycle_type < 0 || cycle_type > 2)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetCycleType( void  *data,
                          int *  cycle_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *cycle_type = hypre_ParAMGDataCycleType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetTol( void     *data,
                    double    tol  )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (tol < 0 || tol > 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataTol(amg_data) = tol;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetTol( void     *data,
                    double *  tol  )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *tol = hypre_ParAMGDataTol(amg_data);

   return hypre_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
int
hypre_BoomerAMGSetNumSweeps( void     *data,
                              int      num_sweeps )
{
   int i;
   int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
       hypre_ParAMGDataNumGridSweeps(amg_data) = hypre_CTAlloc(int,4);
       
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   for (i=0; i < 3; i++)
      num_grid_sweeps[i] = num_sweeps;
   num_grid_sweeps[3] = 1;

   return hypre_error_flag;
}
 
int
hypre_BoomerAMGSetCycleNumSweeps( void     *data,
                                  int      num_sweeps,
                                  int      k )
{
   int i;
   int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
       num_grid_sweeps = hypre_CTAlloc(int,4);
       for (i=0; i < 4; i++)
	  num_grid_sweeps[i] = 1;
       hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }
       
   hypre_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return hypre_error_flag;
}
 
int
hypre_BoomerAMGGetCycleNumSweeps( void     *data,
                                  int *    num_sweeps,
                                  int      k )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      printf (" Warning! Invalid cycle! No num_sweeps to get!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
       
   *num_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data)[k];

   return hypre_error_flag;
}
 
int
hypre_BoomerAMGSetNumGridSweeps( void     *data,
                              int      *num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!num_grid_sweeps)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data))
      hypre_TFree(hypre_ParAMGDataNumGridSweeps(amg_data));
   hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return hypre_error_flag;
}
 
int
hypre_BoomerAMGGetNumGridSweeps( void     *data,
                              int    ** num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   return hypre_error_flag;
}
 
/* The "Get" function for SetRelaxType is GetCycleRelaxType. */
int
hypre_BoomerAMGSetRelaxType( void     *data,
                              int      relax_type )
{
   int i;
   int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (relax_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
       hypre_ParAMGDataGridRelaxType(amg_data) = hypre_CTAlloc(int,4);
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   for (i=0; i < 3; i++)
      grid_relax_type[i] = relax_type;
   grid_relax_type[3] = 9;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetCycleRelaxType( void     *data,
                                  int      relax_type,
                                  int      k )
{
   int i;
   int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      printf (" Warning! Invalid cycle! relax_type not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (relax_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
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

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetCycleRelaxType( void     *data,
                                  int    * relax_type,
                                  int      k )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      printf (" Warning! Invalid cycle! relax_type not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
      
   *relax_type = hypre_ParAMGDataGridRelaxType(amg_data)[k];

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetRelaxOrder( void     *data,
                              int       relax_order)
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetRelaxOrder( void     *data,
                              int     * relax_order)
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *relax_order = hypre_ParAMGDataRelaxOrder(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetGridRelaxType( void     *data,
                              int      *grid_relax_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!grid_relax_type)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data))
      hypre_TFree(hypre_ParAMGDataGridRelaxType(amg_data));
   hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetGridRelaxType( void     *data,
                              int    ** grid_relax_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetGridRelaxPoints( void     *data,
                                int      **grid_relax_points )
{
   int i;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!grid_relax_points)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i=0; i < 4; i++)
   	hypre_TFree (hypre_ParAMGDataGridRelaxPoints(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataGridRelaxPoints(amg_data));
   }
   hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetGridRelaxPoints( void     *data,
                                int    *** grid_relax_points )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetRelaxWeight( void     *data,
                               double   *relax_weight )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!relax_weight)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataRelaxWeight(amg_data))
      hypre_TFree(hypre_ParAMGDataRelaxWeight(amg_data));
   hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetRelaxWeight( void     *data,
                               double ** relax_weight )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetRelaxWt( void     *data,
                           double    relax_weight )
{
   int i, num_levels;
   double *relax_weight_array;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(double,num_levels);
                     
   relax_weight_array = hypre_ParAMGDataRelaxWeight(amg_data);
   for (i=0; i < num_levels; i++)
      relax_weight_array[i] = relax_weight;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetLevelRelaxWt( void    *data,
                                double   relax_weight,
                                int      level )
{
   int i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (level > num_levels-1 || level < 0) 
   {
      printf (" Warning! Invalid level! Relax weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         hypre_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
   }
               
   hypre_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGGetLevelRelaxWt( void    *data,
                                double * relax_weight,
                                int      level )
{
   int num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1 || level < 0) 
   {
      printf (" Warning! Invalid level! Relax weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
               
   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[level];
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetOmega( void     *data,
                         double   *omega )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!omega)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   if (hypre_ParAMGDataOmega(amg_data))
      hypre_TFree(hypre_ParAMGDataOmega(amg_data));
   hypre_ParAMGDataOmega(amg_data) = omega;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetOmega( void     *data,
                         double ** omega )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *omega = hypre_ParAMGDataOmega(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetOuterWt( void     *data,
                           double    omega )
{
   int i, num_levels;
   double *omega_array;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(double,num_levels);
                     
   omega_array = hypre_ParAMGDataOmega(amg_data);
   for (i=0; i < num_levels; i++)
      omega_array[i] = omega;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetLevelOuterWt( void    *data,
                                double   omega,
                                int      level )
{
   int i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      printf (" Warning! Invalid level! Outer weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_ParAMGDataOmega(amg_data) = hypre_CTAlloc(double,num_levels);
      for (i=0; i < num_levels; i++)
         hypre_ParAMGDataOmega(amg_data)[i] = 1.0;
   }
               
   hypre_ParAMGDataOmega(amg_data)[level] = omega;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGGetLevelOuterWt( void    *data,
                                double * omega,
                                int      level )
{
   int num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      printf (" Warning! Invalid level! Outer weight not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   if (hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
               
   *omega = hypre_ParAMGDataOmega(amg_data)[level];
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSmoothType( void     *data,
                              int   smooth_type )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
               
   hypre_ParAMGDataSmoothType(amg_data) = smooth_type;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSmoothType( void     *data,
                              int * smooth_type )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSmoothNumLevels( void     *data,
                            int   smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (smooth_num_levels < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSmoothNumLevels(amg_data) = smooth_num_levels;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSmoothNumLevels( void     *data,
                            int * smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSmoothNumSweeps( void     *data,
                            int   smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (smooth_num_sweeps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSmoothNumSweeps(amg_data) = smooth_num_sweeps;
   
   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSmoothNumSweeps( void     *data,
                            int * smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_num_sweeps = hypre_ParAMGDataSmoothNumSweeps(amg_data);
   
   return hypre_error_flag;
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
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataLogging(amg_data) = logging;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetLogging( void     *data,
                            int     * logging )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *logging = hypre_ParAMGDataLogging(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetPrintLevel( void     *data,
                        int print_level )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetPrintLevel( void     *data,
                        int * print_level )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *print_level =  hypre_ParAMGDataPrintLevel(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetPrintFileName( void       *data,
                               const char *print_file_name )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if( strlen(print_file_name)<=256 );
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 

   sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetPrintFileName( void       *data,
                                 char ** print_file_name )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   sprintf( *print_file_name, "%s", hypre_ParAMGDataLogFileName(amg_data) );

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetNumIterations( void    *data,
                              int      num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetDebugFlag( void     *data,
                          int       debug_flag )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetDebugFlag( void     *data,
                          int     * debug_flag )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetGSMG( void *data,
                        int   par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   amg_data->gsmg = par;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumSamples( void *data,
                        int   par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   amg_data->num_samples = par;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumFunctions( void     *data,
                            int       num_functions )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (num_functions < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumFunctions(amg_data) = num_functions;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetNumFunctions( void     *data,
                            int     * num_functions )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_functions = hypre_ParAMGDataNumFunctions(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNodal( void     *data,
                          int    nodal )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNodal(amg_data) = nodal;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate the degree of aggressive coarsening
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumPaths( void     *data,
                            int       num_paths )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (num_paths < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumPaths(amg_data) = num_paths;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetAggNumLevels( void     *data,
                            int       agg_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_num_levels < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggNumLevels(amg_data) = agg_num_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of relaxation steps for Compatible relaxation
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetNumCRRelaxSteps( void     *data,
                            int       num_CR_relax_steps )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (num_CR_relax_steps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumCRRelaxSteps(amg_data) = num_CR_relax_steps;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetCRRate( void     *data,
                          double    CR_rate )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataCRRate(amg_data) = CR_rate;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates which independent set algorithm is used for CR
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetISType( void     *data,
                            int      IS_type )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (IS_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataISType(amg_data) = IS_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates whether to use CG for compatible relaxation
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGSetCRUseCG( void     *data,
                            int      CR_use_CG )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataCRUseCG(amg_data) = CR_use_CG;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetNumPoints( void     *data,
                          int       num_points )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetDofFunc( void     *data,
                           int      *dof_func )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (!dof_func)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataDofFunc(amg_data));
   hypre_ParAMGDataDofFunc(amg_data) = dof_func;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetPointDofMap( void     *data,
                         int      *point_dof_map )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataPointDofMap(amg_data));
   hypre_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetDofPoint( void     *data,
                         int      *dof_point )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataDofPoint(amg_data));
   hypre_ParAMGDataDofPoint(amg_data) = dof_point;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetNumIterations( void     *data,
                              int      *num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_iterations = hypre_ParAMGDataNumIterations(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetCumNumIterations( void     *data,
                                    int      *cum_num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
#ifdef CUMNUMIT
   *cum_num_iterations = hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetResidual( void * data, hypre_ParVector ** resid )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *resid = hypre_ParAMGDataResidual( amg_data );
   return hypre_error_flag;
}
                            

int
hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                     double   *rel_resid_norm )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetVariant( void     *data,
                            int       variant)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (variant < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataVariant(amg_data) = variant;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetVariant( void     *data,
                            int     * variant)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *variant = hypre_ParAMGDataVariant(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetOverlap( void     *data,
                            int       overlap)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (overlap < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataOverlap(amg_data) = overlap;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetOverlap( void     *data,
                            int     * overlap)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *overlap = hypre_ParAMGDataOverlap(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetDomainType( void     *data,
                            int       domain_type)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (domain_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataDomainType(amg_data) = domain_type;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetDomainType( void     *data,
                            int     * domain_type)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *domain_type = hypre_ParAMGDataDomainType(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSchwarzRlxWeight( void     *data,
                            double     schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return hypre_error_flag;
}

int
hypre_BoomerAMGGetSchwarzRlxWeight( void     *data,
                            double   * schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *schwarz_rlx_weight = hypre_ParAMGDataSchwarzRlxWeight(amg_data);

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetSym( void     *data,
                            int       sym)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSym(amg_data) = sym;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetLevel( void     *data,
                            int       level)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataLevel(amg_data) = level;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetThreshold( void     *data,
                             double    thresh)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataThreshold(amg_data) = thresh;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetFilter( void     *data,
                          double    filter)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataFilter(amg_data) = filter;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetDropTol( void     *data,
                           double    drop_tol)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataDropTol(amg_data) = drop_tol;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetMaxNzPerRow( void     *data,
                               int       max_nz_per_row)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (max_nz_per_row < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataMaxNzPerRow(amg_data) = max_nz_per_row;

   return hypre_error_flag;
}

int
hypre_BoomerAMGSetEuclidFile( void     *data,
                              char     *euclidfile)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return hypre_error_flag;
}

