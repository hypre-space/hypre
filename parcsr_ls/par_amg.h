/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_ParAMG_DATA_HEADER
#define hypre_ParAMG_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParAMGData
 *--------------------------------------------------------------------------*/

typedef struct
{

   /* setup params */
   int      max_levels;
   double   strong_threshold;
   double   max_row_sum;
   double   trunc_factor;
   int      measure_type;
   int      setup_type;
   int      coarsen_type;
   int      interp_type;
   int      restr_par;

   /* solve params */
   int      max_iter;
   int      min_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points;
   int      user_coarse_relax_type;   
   double  *relax_weight; 
   double  *omega;
   double   tol;

   /* problem data */
   hypre_ParCSRMatrix  *A;
   int      num_variables;
   int      num_functions;
   int      num_points;
   int     *dof_func;
   int     *dof_point;           
   int     *point_dof_map;

   /* data generated in the setup phase */
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   int                **CF_marker_array;
   int                **dof_func_array;
   int                **dof_point_array;
   int                **point_dof_map_array;
   int                  num_levels;

   /* data for more complex smoothers */
   int                 *smooth_option;
   HYPRE_Solver        *smoother;
   int			smooth_num_sweep;
   int                  variant;
   int                  overlap;
   int                  domain_type;
   double		schwarz_rlx_weight;
   int			sym;
   int			level;
   int			max_nz_per_row;
   double		threshold;
   double		filter;
   double		drop_tol;
   char		       *euclidfile;

   /* data generated in the solve phase */
   hypre_ParVector   *Vtemp;
   hypre_Vector      *Vtemp_local;
   double            *Vtemp_local_data;
   int                cycle_op_count;                                                   
   /* fields used by GSMG and LS interpolation */
   int                 gsmg;        /* nonzero indicates use of GSMG */
   int                 num_samples; /* number of sample vectors */

   /* log info */
   int      log_level;
   int      num_iterations;
   double   rel_resid_norm;
   hypre_ParVector *residual; /* available if log_level>2 */

   /* output params */
   int      print_level;
   char     log_file_name[256];
   int      debug_flag;

} hypre_ParAMGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_AMGData structure
 *--------------------------------------------------------------------------*/

/* setup params */
		  		      
#define hypre_ParAMGDataRestriction(amg_data) ((amg_data)->restr_par)
#define hypre_ParAMGDataMaxLevels(amg_data) ((amg_data)->max_levels)
#define hypre_ParAMGDataStrongThreshold(amg_data) \
((amg_data)->strong_threshold)
#define hypre_ParAMGDataMaxRowSum(amg_data) ((amg_data)->max_row_sum)
#define hypre_ParAMGDataTruncFactor(amg_data) ((amg_data)->trunc_factor)
#define hypre_ParAMGDataInterpType(amg_data) ((amg_data)->interp_type)
#define hypre_ParAMGDataCoarsenType(amg_data) ((amg_data)->coarsen_type)
#define hypre_ParAMGDataMeasureType(amg_data) ((amg_data)->measure_type)
#define hypre_ParAMGDataSetupType(amg_data) ((amg_data)->setup_type)

/* solve params */

#define hypre_ParAMGDataMinIter(amg_data) ((amg_data)->min_iter)
#define hypre_ParAMGDataMaxIter(amg_data) ((amg_data)->max_iter)
#define hypre_ParAMGDataCycleType(amg_data) ((amg_data)->cycle_type)
#define hypre_ParAMGDataTol(amg_data) ((amg_data)->tol)
#define hypre_ParAMGDataNumGridSweeps(amg_data) ((amg_data)->num_grid_sweeps)
#define hypre_ParAMGDataUserCoarseRelaxType(amg_data) ((amg_data)->user_coarse_relax_type)
#define hypre_ParAMGDataGridRelaxType(amg_data) ((amg_data)->grid_relax_type)
#define hypre_ParAMGDataGridRelaxPoints(amg_data) \
((amg_data)->grid_relax_points)
#define hypre_ParAMGDataRelaxWeight(amg_data) ((amg_data)->relax_weight)
#define hypre_ParAMGDataOmega(amg_data) ((amg_data)->omega)

/* problem data parameters */
#define  hypre_ParAMGDataNumVariables(amg_data)  ((amg_data)->num_variables)
#define hypre_ParAMGDataNumFunctions(amg_data) ((amg_data)->num_functions)
#define hypre_ParAMGDataNumPoints(amg_data) ((amg_data)->num_points)
#define hypre_ParAMGDataDofFunc(amg_data) ((amg_data)->dof_func)
#define hypre_ParAMGDataDofPoint(amg_data) ((amg_data)->dof_point)
#define hypre_ParAMGDataPointDofMap(amg_data) ((amg_data)->point_dof_map)

/* data generated by the setup phase */
#define hypre_ParAMGDataCFMarkerArray(amg_data) ((amg_data)-> CF_marker_array)
#define hypre_ParAMGDataAArray(amg_data) ((amg_data)->A_array)
#define hypre_ParAMGDataFArray(amg_data) ((amg_data)->F_array)
#define hypre_ParAMGDataUArray(amg_data) ((amg_data)->U_array)
#define hypre_ParAMGDataPArray(amg_data) ((amg_data)->P_array)
#define hypre_ParAMGDataRArray(amg_data) ((amg_data)->R_array)
#define hypre_ParAMGDataDofFuncArray(amg_data) ((amg_data)->dof_func_array)
#define hypre_ParAMGDataDofPointArray(amg_data) ((amg_data)->dof_point_array)
#define hypre_ParAMGDataPointDofMapArray(amg_data) \
((amg_data)->point_dof_map_array) 
#define hypre_ParAMGDataNumLevels(amg_data) ((amg_data)->num_levels)	
#define hypre_ParAMGDataSmoothOption(amg_data) ((amg_data)->smooth_option)
#define hypre_ParAMGDataSmoother(amg_data) ((amg_data)->smoother)	
#define hypre_ParAMGDataSmoothNumSweep(amg_data) ((amg_data)->smooth_num_sweep)	
#define hypre_ParAMGDataVariant(amg_data) ((amg_data)->variant)	
#define hypre_ParAMGDataOverlap(amg_data) ((amg_data)->overlap)	
#define hypre_ParAMGDataDomainType(amg_data) ((amg_data)->domain_type)	
#define hypre_ParAMGDataSchwarzRlxWeight(amg_data) \
((amg_data)->schwarz_rlx_weight)
#define hypre_ParAMGDataSym(amg_data) ((amg_data)->sym)	
#define hypre_ParAMGDataLevel(amg_data) ((amg_data)->level)	
#define hypre_ParAMGDataMaxNzPerRow(amg_data) ((amg_data)->max_nz_per_row)
#define hypre_ParAMGDataThreshold(amg_data) ((amg_data)->threshold)	
#define hypre_ParAMGDataFilter(amg_data) ((amg_data)->filter)	
#define hypre_ParAMGDataDropTol(amg_data) ((amg_data)->drop_tol)	
#define hypre_ParAMGDataEuclidFile(amg_data) ((amg_data)->euclidfile)	

/* data generated in the solve phase */
#define hypre_ParAMGDataVtemp(amg_data) ((amg_data)->Vtemp)
#define hypre_ParAMGDataVtempLocal(amg_data) ((amg_data)->Vtemp_local)
#define hypre_ParAMGDataVtemplocalData(amg_data) ((amg_data)->Vtemp_local_data)
#define hypre_ParAMGDataCycleOpCount(amg_data) ((amg_data)->cycle_op_count)

/* fields used by GSMG */
#define hypre_ParAMGDataGSMG(amg_data) ((amg_data)->gsmg)
#define hypre_ParAMGDataNumSamples(amg_data) ((amg_data)->num_samples)

/* log info data */
#define hypre_ParAMGDataLogLevel(amg_data) ((amg_data)->log_level)
#define hypre_ParAMGDataNumIterations(amg_data) ((amg_data)->num_iterations)
#define hypre_ParAMGDataRelativeResidualNorm(amg_data) ((amg_data)->rel_resid_norm)
#define hypre_ParAMGDataResidual(amg_data) ((amg_data)->residual)

/* output parameters */
#define hypre_ParAMGDataPrintLevel(amg_data) ((amg_data)->print_level)
#define hypre_ParAMGDataLogFileName(amg_data) ((amg_data)->log_file_name)
#define hypre_ParAMGDataDebugFlag(amg_data)   ((amg_data)->debug_flag)

#endif



