/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.70 $
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
   HYPRE_Int      max_levels;
   HYPRE_Int      max_coarse_size;
   double   strong_threshold;
   double   max_row_sum;
   double   trunc_factor;
   double   agg_trunc_factor;
   double   agg_P12_trunc_factor;
   double   jacobi_trunc_threshold;
   double   S_commpkg_switch;
   double   CR_rate;
   double   CR_strong_th;
   HYPRE_Int      interp_type;
   HYPRE_Int      sep_weight;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      measure_type;
   HYPRE_Int      setup_type;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int 	    num_functions;
   HYPRE_Int 	    nodal, nodal_levels, nodal_diag;
   HYPRE_Int 	    num_paths;
   HYPRE_Int 	    agg_num_levels;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      agg_P_max_elmts;
   HYPRE_Int      agg_P12_max_elmts;
   HYPRE_Int      post_interp_type;
   HYPRE_Int 	    num_CR_relax_steps;
   HYPRE_Int 	    IS_type;
   HYPRE_Int 	    CR_use_CG;
   HYPRE_Int 	    cgc_its;
   HYPRE_Int 	    seq_threshold;

   /* solve params */
   HYPRE_Int      min_iter;
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_type;    
 
   double   tol;

   HYPRE_Int      num_sweeps;  
   HYPRE_Int      relax_type;   
   HYPRE_Int      relax_order;   
   double   relax_wt;
   double   outer_wt;
   HYPRE_Int      smooth_type;
   HYPRE_Int      smooth_num_levels;
   HYPRE_Int      smooth_num_sweeps;

   HYPRE_Int      variant, overlap, domain_type, schwarz_use_nonsymm;
   double   schwarz_rlx_weight;
   HYPRE_Int	    level, sym;
   HYPRE_Int	    eu_level, eu_bj;
   HYPRE_Int	    max_nz_per_row;
   double   thresh, filter;
   double   drop_tol;
   double   eu_sparse_A;
   char    *euclidfile;

   HYPRE_Int cheby_order;
   double cheby_eig_ratio;

   HYPRE_Int block_mode;
   

   /* log info */
   HYPRE_Int      num_iterations;
   HYPRE_Int      cum_num_iterations;

   /* output params */
   HYPRE_Int      print_level;
   HYPRE_Int      logging;
   /* HYPRE_Int      cycle_op_count; */
   char     log_file_name[256];
   HYPRE_Int      debug_flag;

   char     plot_file_name[251] = {0};

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   max_coarse_size = 9;
   seq_threshold = 0;
   strong_threshold = 0.25;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   agg_trunc_factor = 0.0;
   agg_P12_trunc_factor = 0.0;
   jacobi_trunc_threshold = 0.01;
   S_commpkg_switch = 1.0;
   interp_type = 0;
   sep_weight = 0;
   coarsen_type = 6;
   measure_type = 0;
   setup_type = 1;
   P_max_elmts = 0;
   agg_P_max_elmts = 0;
   agg_P12_max_elmts = 0;
   num_functions = 1;
   nodal = 0;
   nodal_levels = max_levels;
   nodal_diag = 0;
   num_paths = 1;
   agg_num_levels = 0;
   post_interp_type = 0;
   agg_interp_type = 4;
   num_CR_relax_steps = 2;
   CR_rate = 0.7;
   CR_strong_th = 0;
   IS_type = 1;
   CR_use_CG = 0;
   cgc_its = 1;

   variant = 0;
   overlap = 1;
   domain_type = 2;
   schwarz_rlx_weight = 1.0;
   smooth_num_sweeps = 1;
   smooth_num_levels = 0;
   smooth_type = 6;
   schwarz_use_nonsymm = 0;
   
   level = 1;
   sym = 0;
   thresh = 0.1;
   filter = 0.05;
   drop_tol = 0.0001;
   max_nz_per_row = 20;
   euclidfile = NULL;
   eu_level = 0;
   eu_sparse_A = 0.0;
   eu_bj = 0;

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

   cheby_order = 2;
   cheby_eig_ratio = .3;

   block_mode = 0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;

   /* output params */
   print_level = 0;
   logging = 0;
   hypre_sprintf(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = hypre_CTAlloc(hypre_ParAMGData, 1);

   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   hypre_ParAMGDataUserRelaxType(amg_data) = -1;
   hypre_ParAMGDataUserNumSweeps(amg_data) = -1;
   hypre_ParAMGDataUserRelaxWeight(amg_data) = 1.0;
   hypre_BoomerAMGSetMaxLevels(amg_data, max_levels);
   hypre_BoomerAMGSetMaxCoarseSize(amg_data, max_coarse_size);
   hypre_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
   hypre_BoomerAMGSetMaxRowSum(amg_data, max_row_sum);
   hypre_BoomerAMGSetTruncFactor(amg_data, trunc_factor);
   hypre_BoomerAMGSetAggTruncFactor(amg_data, agg_trunc_factor);
   hypre_BoomerAMGSetAggP12TruncFactor(amg_data, agg_P12_trunc_factor);
   hypre_BoomerAMGSetJacobiTruncThreshold(amg_data, jacobi_trunc_threshold);
   hypre_BoomerAMGSetSCommPkgSwitch(amg_data, S_commpkg_switch);
   hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   hypre_BoomerAMGSetSepWeight(amg_data, sep_weight);
   hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   hypre_BoomerAMGSetSetupType(amg_data, setup_type);
   hypre_BoomerAMGSetPMaxElmts(amg_data, P_max_elmts);
   hypre_BoomerAMGSetAggPMaxElmts(amg_data, agg_P_max_elmts);
   hypre_BoomerAMGSetAggP12MaxElmts(amg_data, agg_P12_max_elmts);
   hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   hypre_BoomerAMGSetNodal(amg_data, nodal);
   hypre_BoomerAMGSetNodalLevels(amg_data, nodal_levels);
   hypre_BoomerAMGSetNodal(amg_data, nodal_diag);
   hypre_BoomerAMGSetNumPaths(amg_data, num_paths);
   hypre_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
   hypre_BoomerAMGSetAggInterpType(amg_data, agg_interp_type);
   hypre_BoomerAMGSetPostInterpType(amg_data, post_interp_type);
   hypre_BoomerAMGSetNumCRRelaxSteps(amg_data, num_CR_relax_steps);
   hypre_BoomerAMGSetCRRate(amg_data, CR_rate);
   hypre_BoomerAMGSetCRStrongTh(amg_data, CR_strong_th);
   hypre_BoomerAMGSetISType(amg_data, IS_type);
   hypre_BoomerAMGSetCRUseCG(amg_data, CR_use_CG);
   hypre_BoomerAMGSetCGCIts(amg_data, cgc_its);
   hypre_BoomerAMGSetVariant(amg_data, variant);
   hypre_BoomerAMGSetOverlap(amg_data, overlap);
   hypre_BoomerAMGSetSchwarzRlxWeight(amg_data, schwarz_rlx_weight);
   hypre_BoomerAMGSetSchwarzUseNonSymm(amg_data, schwarz_use_nonsymm);
   hypre_BoomerAMGSetDomainType(amg_data, domain_type);
   hypre_BoomerAMGSetSym(amg_data, sym);
   hypre_BoomerAMGSetLevel(amg_data, level);
   hypre_BoomerAMGSetThreshold(amg_data, thresh);
   hypre_BoomerAMGSetFilter(amg_data, filter);
   hypre_BoomerAMGSetDropTol(amg_data, drop_tol);
   hypre_BoomerAMGSetMaxNzPerRow(amg_data, max_nz_per_row);
   hypre_BoomerAMGSetEuclidFile(amg_data, euclidfile);
   hypre_BoomerAMGSetEuLevel(amg_data, eu_level);
   hypre_BoomerAMGSetEuSparseA(amg_data, eu_sparse_A);
   hypre_BoomerAMGSetEuBJ(amg_data, eu_bj);

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

   hypre_BoomerAMGSetChebyOrder(amg_data, cheby_order);
   hypre_BoomerAMGSetChebyFraction(amg_data, cheby_eig_ratio);

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
   hypre_ParAMGDataL1Norms(amg_data) = NULL;
  
   hypre_ParAMGDataABlockArray(amg_data) = NULL;
   hypre_ParAMGDataPBlockArray(amg_data) = NULL;
   hypre_ParAMGDataRBlockArray(amg_data) = NULL;

   /* this can not be set by the user currently */
   hypre_ParAMGDataBlockMode(amg_data) = block_mode;

   hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
   hypre_ParAMGDataMinEigEst(amg_data) = NULL;

   /* BM Oct 22, 2006 */
   hypre_ParAMGDataPlotGrids(amg_data) = 0;
   hypre_BoomerAMGSetPlotFileName (amg_data, plot_file_name);
                                                                                                       
   /* BM Oct 17, 2006 */
   hypre_ParAMGDataCoordDim(amg_data) = 0;
   hypre_ParAMGDataCoordinates(amg_data) = NULL;

  /* for fitting vectors for interp */ 
   hypre_BoomerAMGSetInterpVecVariant(amg_data, 0);
   hypre_BoomerAMGSetInterpVectors(amg_data, 0, NULL); 
   hypre_ParAMGNumLevelsInterpVectors(amg_data) = max_levels;
   hypre_ParAMGInterpVectorsArray(amg_data) = NULL;
   hypre_ParAMGInterpVecQMax(amg_data) = 0;
   hypre_ParAMGInterpVecAbsQTrunc(amg_data) = 0.0;
   hypre_ParAMGInterpRefine(amg_data) = 0;
   hypre_ParAMGInterpVecFirstLevel(amg_data) = 0;
   hypre_ParAMGNumInterpVectors(amg_data) = 0;
   hypre_ParAMGSmoothInterpVectors(amg_data) = 0;
   hypre_ParAMGDataExpandPWeights(amg_data) = NULL;

   /* for redundant coarse grid solve */
   hypre_ParAMGDataCoarseSolver(amg_data) = NULL;
   hypre_ParAMGDataACoarse(amg_data) = NULL;
   hypre_ParAMGDataFCoarse(amg_data) = NULL;
   hypre_ParAMGDataUCoarse(amg_data) = NULL;
   hypre_ParAMGDataNewComm(amg_data) = hypre_MPI_COMM_NULL;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGDestroy( void *data )
{
   hypre_ParAMGData  *amg_data = data;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   HYPRE_Solver *smoother = hypre_ParAMGDataSmoother(amg_data);
   void *amg = hypre_ParAMGDataCoarseSolver(amg_data);
   MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
   HYPRE_Int i;

   if (hypre_ParAMGDataMaxEigEst(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataMaxEigEst(amg_data));
      hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
   }
   if (hypre_ParAMGDataMinEigEst(amg_data))
   {
      hypre_TFree(hypre_ParAMGDataMinEigEst(amg_data));
      hypre_ParAMGDataMinEigEst(amg_data) = NULL;
   }
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

   if (hypre_ParAMGDataL1Norms(amg_data))
   {
      for (i=0; i < num_levels; i++)
         if (hypre_ParAMGDataL1Norms(amg_data)[i])
           hypre_TFree(hypre_ParAMGDataL1Norms(amg_data)[i]);
      hypre_TFree(hypre_ParAMGDataL1Norms(amg_data));
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

   
   if (hypre_ParAMGInterpVecVariant(amg_data) > 0 
        &&  hypre_ParAMGNumInterpVectors(amg_data) > 0)
   {

      HYPRE_Int j;
      HYPRE_Int num_vecs =  hypre_ParAMGNumInterpVectors(amg_data);
      hypre_ParVector **sm_vecs;
      HYPRE_Int num_il;
      num_il = hypre_min(hypre_ParAMGNumLevelsInterpVectors(amg_data),num_levels);

      /* don't destroy lev = 0 - this was user input */
      for (i = 1; i< num_il; i++)
      {
         sm_vecs = hypre_ParAMGInterpVectorsArray(amg_data)[i];
         for (j = 0; j< num_vecs; j++)
         {
            hypre_ParVectorDestroy(sm_vecs[j]);
         }
         hypre_TFree(sm_vecs);
      }
      hypre_TFree( hypre_ParAMGInterpVectorsArray(amg_data));
   
   }
   
   if (amg) hypre_BoomerAMGDestroy(amg);

   if (hypre_ParAMGDataACoarse(amg_data))
      hypre_ParCSRMatrixDestroy(hypre_ParAMGDataACoarse(amg_data));

   if (hypre_ParAMGDataUCoarse(amg_data))
      hypre_ParVectorDestroy(hypre_ParAMGDataUCoarse(amg_data));

   if (hypre_ParAMGDataFCoarse(amg_data))
      hypre_ParVectorDestroy(hypre_ParAMGDataFCoarse(amg_data));

   if (new_comm != hypre_MPI_COMM_NULL) 
   {
       MPI_Comm_free (&new_comm);
   }
   hypre_TFree(amg_data);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetRestriction( void *data,
                            HYPRE_Int   restr_par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxLevels( void *data,
                          HYPRE_Int   max_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetMaxLevels( void *data,
                             HYPRE_Int *  max_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_levels = hypre_ParAMGDataMaxLevels(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxCoarseSize( void *data,
                          HYPRE_Int   max_coarse_size )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (max_coarse_size < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataMaxCoarseSize(amg_data) = max_coarse_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMaxCoarseSize( void *data,
                             HYPRE_Int *  max_coarse_size )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_coarse_size = hypre_ParAMGDataMaxCoarseSize(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSeqThreshold( void *data,
                          HYPRE_Int   seq_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (seq_threshold < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataSeqThreshold(amg_data) = seq_threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSeqThreshold( void *data,
                             HYPRE_Int *  seq_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *seq_threshold = hypre_ParAMGDataSeqThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetStrongThreshold( void     *data,
                                double    strong_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetStrongThreshold( void     *data,
                                double *  strong_threshold )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxRowSum( void     *data,
                          double    max_row_sum )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetMaxRowSum( void     *data,
                          double *  max_row_sum )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetTruncFactor( void     *data,
                            double    trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetTruncFactor( void     *data,
                            double *  trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPMaxElmts( void     *data,
                            HYPRE_Int    P_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetPMaxElmts( void     *data,
                            HYPRE_Int *  P_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetJacobiTruncThreshold( void     *data,
                            double    jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetJacobiTruncThreshold( void     *data,
                            double *  jacobi_trunc_threshold )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPostInterpType( void     *data,
                                  HYPRE_Int    post_interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetPostInterpType( void     *data,
                                  HYPRE_Int  * post_interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSCommPkgSwitch( void     *data,
                                  double    S_commpkg_switch )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataSCommPkgSwitch(amg_data) = S_commpkg_switch;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSCommPkgSwitch( void     *data,
                                  double *  S_commpkg_switch )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *S_commpkg_switch = hypre_ParAMGDataSCommPkgSwitch(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpType( void     *data,
                           HYPRE_Int       interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 


   if (interp_type < 0 || interp_type > 25)

   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetInterpType( void     *data,
                           HYPRE_Int *     interp_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *interp_type = hypre_ParAMGDataInterpType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSepWeight( void     *data,
                           HYPRE_Int       sep_weight )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataSepWeight(amg_data) = sep_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMinIter( void     *data,
                        HYPRE_Int       min_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return hypre_error_flag;
} 

HYPRE_Int
hypre_BoomerAMGGetMinIter( void     *data,
                        HYPRE_Int *     min_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *min_iter = hypre_ParAMGDataMinIter(amg_data);

   return hypre_error_flag;
} 

HYPRE_Int
hypre_BoomerAMGSetMaxIter( void     *data,
                        HYPRE_Int     max_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetMaxIter( void     *data,
                        HYPRE_Int *   max_iter )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *max_iter = hypre_ParAMGDataMaxIter(amg_data);

   return hypre_error_flag;
} 

HYPRE_Int
hypre_BoomerAMGSetCoarsenType( void  *data,
                          HYPRE_Int    coarsen_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCoarsenType( void  *data,
                          HYPRE_Int *  coarsen_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMeasureType( void  *data,
                            HYPRE_Int    measure_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetMeasureType( void  *data,
                            HYPRE_Int *  measure_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *measure_type = hypre_ParAMGDataMeasureType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSetupType( void  *data,
                             HYPRE_Int    setup_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGDataSetupType(amg_data) = setup_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSetupType( void  *data,
                             HYPRE_Int  *  setup_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *setup_type = hypre_ParAMGDataSetupType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCycleType( void  *data,
                          HYPRE_Int    cycle_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetCycleType( void  *data,
                          HYPRE_Int *  cycle_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *cycle_type = hypre_ParAMGDataCycleType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetTol( void     *data,
                    double    tol  )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetTol( void     *data,
                    double *  tol  )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   *tol = hypre_ParAMGDataTol(amg_data);

   return hypre_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
HYPRE_Int
hypre_BoomerAMGSetNumSweeps( void     *data,
                              HYPRE_Int      num_sweeps )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (num_sweeps < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
       hypre_ParAMGDataNumGridSweeps(amg_data) = hypre_CTAlloc(HYPRE_Int,4);
       
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   for (i=0; i < 3; i++)
      num_grid_sweeps[i] = num_sweeps;
   num_grid_sweeps[3] = 1;

   hypre_ParAMGDataUserNumSweeps(amg_data) = num_sweeps;

   return hypre_error_flag;
}
 
HYPRE_Int
hypre_BoomerAMGSetCycleNumSweeps( void     *data,
                                  HYPRE_Int      num_sweeps,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *num_grid_sweeps;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (num_sweeps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
       num_grid_sweeps = hypre_CTAlloc(HYPRE_Int,4);
       for (i=0; i < 4; i++)
	  num_grid_sweeps[i] = 1;
       hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }
       
   hypre_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return hypre_error_flag;
}
 
HYPRE_Int
hypre_BoomerAMGGetCycleNumSweeps( void     *data,
                                  HYPRE_Int *    num_sweeps,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! No num_sweeps to get!\n");
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
 
HYPRE_Int
hypre_BoomerAMGSetNumGridSweeps( void     *data,
                              HYPRE_Int      *num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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
 
HYPRE_Int
hypre_BoomerAMGGetNumGridSweeps( void     *data,
                              HYPRE_Int    ** num_grid_sweeps )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);

   return hypre_error_flag;
}
 
/* The "Get" function for SetRelaxType is GetCycleRelaxType. */
HYPRE_Int
hypre_BoomerAMGSetRelaxType( void     *data,
                              HYPRE_Int      relax_type )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (relax_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
       hypre_ParAMGDataGridRelaxType(amg_data) = hypre_CTAlloc(HYPRE_Int,4);
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   for (i=0; i < 3; i++)
      grid_relax_type[i] = relax_type;
   grid_relax_type[3] = 9;
   hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   hypre_ParAMGDataUserRelaxType(amg_data) = relax_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetCycleRelaxType( void     *data,
                                  HYPRE_Int      relax_type,
                                  HYPRE_Int      k )
{
   HYPRE_Int i;
   HYPRE_Int *grid_relax_type;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! relax_type not set!\n");
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
      grid_relax_type = hypre_CTAlloc(HYPRE_Int,4);
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

HYPRE_Int
hypre_BoomerAMGGetCycleRelaxType( void     *data,
                                  HYPRE_Int    * relax_type,
                                  HYPRE_Int      k )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      hypre_printf (" Warning! Invalid cycle! relax_type not set!\n");
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

HYPRE_Int
hypre_BoomerAMGSetRelaxOrder( void     *data,
                              HYPRE_Int       relax_order)
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetRelaxOrder( void     *data,
                              HYPRE_Int     * relax_order)
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *relax_order = hypre_ParAMGDataRelaxOrder(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetGridRelaxType( void     *data,
                              HYPRE_Int      *grid_relax_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetGridRelaxType( void     *data,
                              HYPRE_Int    ** grid_relax_type )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetGridRelaxPoints( void     *data,
                                HYPRE_Int      **grid_relax_points )
{
   HYPRE_Int i;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetGridRelaxPoints( void     *data,
                                HYPRE_Int    *** grid_relax_points )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRelaxWeight( void     *data,
                               double   *relax_weight )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetRelaxWeight( void     *data,
                               double ** relax_weight )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetRelaxWt( void     *data,
                           double    relax_weight )
{
   HYPRE_Int i, num_levels;
   double *relax_weight_array;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
      hypre_ParAMGDataRelaxWeight(amg_data) = hypre_CTAlloc(double,num_levels);
                     
   relax_weight_array = hypre_ParAMGDataRelaxWeight(amg_data);
   for (i=0; i < num_levels; i++)
      relax_weight_array[i] = relax_weight;

   hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_weight;
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevelRelaxWt( void    *data,
                                double   relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (level > num_levels-1 || level < 0) 
   {
      hypre_printf (" Warning! Invalid level! Relax weight not set!\n");
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

HYPRE_Int
hypre_BoomerAMGGetLevelRelaxWt( void    *data,
                                double * relax_weight,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1 || level < 0) 
   {
      hypre_printf (" Warning! Invalid level! Relax weight not set!\n");
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

HYPRE_Int
hypre_BoomerAMGSetOmega( void     *data,
                         double   *omega )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetOmega( void     *data,
                         double ** omega )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *omega = hypre_ParAMGDataOmega(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetOuterWt( void     *data,
                           double    omega )
{
   HYPRE_Int i, num_levels;
   double *omega_array;
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGSetLevelOuterWt( void    *data,
                                double   omega,
                                HYPRE_Int      level )
{
   HYPRE_Int i, num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      hypre_printf (" Warning! Invalid level! Outer weight not set!\n");
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

HYPRE_Int
hypre_BoomerAMGGetLevelOuterWt( void    *data,
                                double * omega,
                                HYPRE_Int      level )
{
   HYPRE_Int num_levels;
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   num_levels = hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      hypre_printf (" Warning! Invalid level! Outer weight not set!\n");
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

HYPRE_Int
hypre_BoomerAMGSetSmoothType( void     *data,
                              HYPRE_Int   smooth_type )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
               
   hypre_ParAMGDataSmoothType(amg_data) = smooth_type;
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSmoothType( void     *data,
                              HYPRE_Int * smooth_type )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSmoothNumLevels( void     *data,
                            HYPRE_Int   smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetSmoothNumLevels( void     *data,
                            HYPRE_Int * smooth_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSmoothNumSweeps( void     *data,
                            HYPRE_Int   smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetSmoothNumSweeps( void     *data,
                            HYPRE_Int * smooth_num_sweeps )
{
   hypre_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *smooth_num_sweeps = hypre_ParAMGDataSmoothNumSweeps(amg_data);
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLogging( void     *data,
                            HYPRE_Int       logging )
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
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataLogging(amg_data) = logging;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetLogging( void     *data,
                            HYPRE_Int     * logging )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *logging = hypre_ParAMGDataLogging(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPrintLevel( void     *data,
                        HYPRE_Int print_level )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPrintLevel( void     *data,
                        HYPRE_Int * print_level )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *print_level =  hypre_ParAMGDataPrintLevel(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPrintFileName( void       *data,
                               const char *print_file_name )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if( strlen(print_file_name) > 256 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 

   hypre_sprintf(hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetPrintFileName( void       *data,
                                 char ** print_file_name )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_sprintf( *print_file_name, "%s", hypre_ParAMGDataLogFileName(amg_data) );

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNumIterations( void    *data,
                              HYPRE_Int      num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDebugFlag( void     *data,
                          HYPRE_Int       debug_flag )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetDebugFlag( void     *data,
                          HYPRE_Int     * debug_flag )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *debug_flag = hypre_ParAMGDataDebugFlag(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetGSMG( void *data,
                        HYPRE_Int   par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   amg_data->gsmg = par;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumSamples( void *data,
                        HYPRE_Int   par )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   amg_data->num_samples = par;

   return hypre_error_flag;
}

/* BM Aug 25, 2006 */

HYPRE_Int
hypre_BoomerAMGSetCGCIts( void *data,
                          HYPRE_Int  its)
{
  HYPRE_Int ierr = 0;
  hypre_ParAMGData *amg_data = data;

  hypre_ParAMGDataCGCIts(amg_data) = its;
  return (ierr);
}

/* BM Oct 22, 2006 */
HYPRE_Int
hypre_BoomerAMGSetPlotGrids( void *data,
                          HYPRE_Int plotgrids)
{
  HYPRE_Int ierr = 0;
  hypre_ParAMGData *amg_data = data;

  hypre_ParAMGDataPlotGrids(amg_data) = plotgrids;
  return (ierr);
}

HYPRE_Int
hypre_BoomerAMGSetPlotFileName( void       *data,
                              const char *plot_file_name )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if( strlen(plot_file_name)>251 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (strlen(plot_file_name)==0 )
     hypre_sprintf(hypre_ParAMGDataPlotFileName(amg_data), "%s", "AMGgrids.CF.dat");
   else
     hypre_sprintf(hypre_ParAMGDataPlotFileName(amg_data), "%s", plot_file_name);

   return hypre_error_flag;
}

/* BM Oct 17, 2006 */
HYPRE_Int
hypre_BoomerAMGSetCoordDim( void *data,
                          HYPRE_Int coorddim)
{
  HYPRE_Int ierr = 0;
  hypre_ParAMGData *amg_data = data;

  hypre_ParAMGDataCoordDim(amg_data) = coorddim;
  return (ierr);
}

HYPRE_Int
hypre_BoomerAMGSetCoordinates( void *data,
                             float *coordinates)
{
  HYPRE_Int ierr = 0;
  hypre_ParAMGData *amg_data = data;

  hypre_ParAMGDataCoordinates(amg_data) = coordinates;
  return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumFunctions( void     *data,
                            HYPRE_Int       num_functions )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetNumFunctions( void     *data,
                            HYPRE_Int     * num_functions )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_functions = hypre_ParAMGDataNumFunctions(amg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodal( void     *data,
                          HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNodal(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate number of levels for nodal coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodalLevels( void     *data,
                          HYPRE_Int    nodal_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNodalLevels(amg_data) = nodal_levels;

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Indicate how to treat diag for primary matrix with  nodal systems function
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNodalDiag( void     *data,
                          HYPRE_Int    nodal )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNodalDiag(amg_data) = nodal;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate the degree of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumPaths( void     *data,
                            HYPRE_Int       num_paths )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGSetAggNumLevels( void     *data,
                            HYPRE_Int       agg_num_levels )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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
 * Indicates the interpolation used with aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggInterpType( void     *data,
                            HYPRE_Int       agg_interp_type )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_interp_type < 0 || agg_interp_type > 4)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggInterpType(amg_data) = agg_interp_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggPMaxElmts( void     *data,
                            HYPRE_Int       agg_P_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_P_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggPMaxElmts(amg_data) = agg_P_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggP12MaxElmts( void     *data,
                            HYPRE_Int       agg_P12_max_elmts )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_P12_max_elmts < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggP12MaxElmts(amg_data) = agg_P12_max_elmts;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggTruncFactor( void     *data,
                            double      agg_trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_trunc_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggTruncFactor(amg_data) = agg_trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetAggP12TruncFactor( void     *data,
                            double      agg_P12_trunc_factor )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (agg_P12_trunc_factor < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataAggP12TruncFactor(amg_data) = agg_P12_trunc_factor;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of relaxation steps for Compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetNumCRRelaxSteps( void     *data,
                            HYPRE_Int       num_CR_relax_steps )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGSetCRRate( void     *data,
                          double    CR_rate )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataCRRate(amg_data) = CR_rate;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetCRStrongTh( void     *data,
                          double    CR_strong_th )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataCRStrongTh(amg_data) = CR_strong_th;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates which independent set algorithm is used for CR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGSetISType( void     *data,
                            HYPRE_Int      IS_type )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGSetCRUseCG( void     *data,
                            HYPRE_Int      CR_use_CG )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataCRUseCG(amg_data) = CR_use_CG;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetNumPoints( void     *data,
                          HYPRE_Int       num_points )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDofFunc( void     *data,
                           HYPRE_Int      *dof_func )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataDofFunc(amg_data));
   hypre_ParAMGDataDofFunc(amg_data) = dof_func;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetPointDofMap( void     *data,
                         HYPRE_Int      *point_dof_map )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataPointDofMap(amg_data));
   hypre_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDofPoint( void     *data,
                         HYPRE_Int      *dof_point )
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_TFree(hypre_ParAMGDataDofPoint(amg_data));
   hypre_ParAMGDataDofPoint(amg_data) = dof_point;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetNumIterations( void     *data,
                              HYPRE_Int      *num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *num_iterations = hypre_ParAMGDataNumIterations(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetCumNumIterations( void     *data,
                                    HYPRE_Int      *cum_num_iterations )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
#ifdef CUMNUMIT
   *cum_num_iterations = hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetResidual( void * data, hypre_ParVector ** resid )
{
   hypre_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *resid = hypre_ParAMGDataResidual( amg_data );
   return hypre_error_flag;
}
                            

HYPRE_Int
hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                     double   *rel_resid_norm )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *rel_resid_norm = hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetVariant( void     *data,
                            HYPRE_Int       variant)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetVariant( void     *data,
                            HYPRE_Int     * variant)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *variant = hypre_ParAMGDataVariant(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetOverlap( void     *data,
                            HYPRE_Int       overlap)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetOverlap( void     *data,
                            HYPRE_Int     * overlap)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *overlap = hypre_ParAMGDataOverlap(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDomainType( void     *data,
                            HYPRE_Int       domain_type)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGGetDomainType( void     *data,
                            HYPRE_Int     * domain_type)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *domain_type = hypre_ParAMGDataDomainType(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSchwarzRlxWeight( void     *data,
                            double     schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGGetSchwarzRlxWeight( void     *data,
                            double   * schwarz_rlx_weight)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   *schwarz_rlx_weight = hypre_ParAMGDataSchwarzRlxWeight(amg_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSchwarzUseNonSymm( void     *data,
                                     HYPRE_Int use_nonsymm)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSchwarzUseNonSymm(amg_data) = use_nonsymm;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetSym( void     *data,
                            HYPRE_Int       sym)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataSym(amg_data) = sym;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetLevel( void     *data,
                            HYPRE_Int       level)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataLevel(amg_data) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetThreshold( void     *data,
                             double    thresh)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataThreshold(amg_data) = thresh;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetFilter( void     *data,
                          double    filter)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataFilter(amg_data) = filter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetDropTol( void     *data,
                           double    drop_tol)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataDropTol(amg_data) = drop_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetMaxNzPerRow( void     *data,
                               HYPRE_Int       max_nz_per_row)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
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

HYPRE_Int
hypre_BoomerAMGSetEuclidFile( void     *data,
                              char     *euclidfile)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuLevel( void     *data,
                            HYPRE_Int      eu_level)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataEuLevel(amg_data) = eu_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuSparseA( void     *data,
                             double    eu_sparse_A)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataEuSparseA(amg_data) = eu_sparse_A;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetEuBJ( void     *data,
                        HYPRE_Int       eu_bj)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataEuBJ(amg_data) = eu_bj;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyOrder( void     *data,
                              HYPRE_Int       order)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (order < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataChebyOrder(amg_data) = order;

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGSetChebyFraction( void     *data,
                                 double      ratio)
{
   hypre_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   if (ratio <= 0.0 || ratio > 1.0 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   } 
   hypre_ParAMGDataChebyFraction(amg_data) = ratio;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGSetInterpVectors
 * -used for post-interpolation fitting of smooth vectors
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGSetInterpVectors(void *solver,
                                    HYPRE_Int  num_vectors,
                                    hypre_ParVector **interp_vectors)

{
   hypre_ParAMGData *amg_data = solver;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGInterpVectors(amg_data) =  interp_vectors;
   hypre_ParAMGNumInterpVectors(amg_data) = num_vectors;
   
   return hypre_error_flag;
}

HYPRE_Int hypre_BoomerAMGSetInterpVecVariant(void *solver,
                                       HYPRE_Int  var)


{
   hypre_ParAMGData *amg_data = solver;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   if (var < 1)
      var = 0;
   if (var > 3)
      var = 3;

   hypre_ParAMGInterpVecVariant(amg_data) = var;
   
   return hypre_error_flag;
  
}

HYPRE_Int
hypre_BoomerAMGSetInterpVecQMax( void     *data,
                                 HYPRE_Int    q_max)
{
   hypre_ParAMGData  *amg_data = data;
   
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGInterpVecQMax(amg_data) = q_max;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpVecAbsQTrunc( void     *data,
                                      double    q_trunc)
{
   hypre_ParAMGData  *amg_data = data;
   
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 
   hypre_ParAMGInterpVecAbsQTrunc(amg_data) = q_trunc;

   return hypre_error_flag;
}

HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors(void *solver,
                                          HYPRE_Int  smooth_interp_vectors)

{
   hypre_ParAMGData *amg_data = solver;
   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGSmoothInterpVectors(amg_data) = smooth_interp_vectors;
   
   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpRefine( void     *data,
                                HYPRE_Int       num_refine )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGInterpRefine(amg_data) = num_refine;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGSetInterpVecFirstLevel( void     *data,
                                       HYPRE_Int  level )
{
   hypre_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      hypre_printf("Warning! BoomerAMG object empty!\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   } 

   hypre_ParAMGInterpVecFirstLevel(amg_data) = level;

   return hypre_error_flag;
}
