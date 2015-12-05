/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




/********************************************************************* */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */   
/* ******************************************************************** */

/********************************************************************* */
/*          some information about the aggregates                     */
/********************************************************************* */

#ifndef __MLAGGINFO__
#define __MLAGGINFO__


#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

  /* function ML_Aggregate_VizAndStats_Setup and ML_Aggregate_VizAndStats_Clean
     are declared in ml_aggregate.h */
     
  extern void ML_Aggregate_ComputeRadius( ML_Aggregate_Viz_Stats finer_level,
 					  ML_Aggregate_Viz_Stats coarser_level,
					  double R[] );
  extern void ML_Aggregate_ComputeBox( ML_Aggregate_Viz_Stats finer_level,int,
				       double R[], int,ML_Comm * comm );
  extern void ML_Aggregate_ComputeCenterOfGravity( ML_Aggregate_Viz_Stats finer_level,
					    ML_Aggregate_Viz_Stats coarser_level,
						   ML_Comm * comm);
  extern void ML_Aggregate_ComputeVolume( int N_fine,
					  int N_aggregates,
					  int graph_decomposition[],
					  int local_or_global,
					  double volume[],
					  double V[] );
  extern void ML_Aggregate_AnalyzeLocalGraphDec( int N_aggregates,
					  int *nodes_per_aggregate,
					  int *min_loc_aggre,
					  int *max_loc_aggre,
					  double *avg_loc_aggre,
					  double *std_loc_aggre,
						 ML_Comm *comm );
  extern void ML_Aggregate_AnalyzeVector( int Nlocal,
					  double vector[],
					  double *min_vec,
					  double *max_vec,
					  double *avg_vec,
					  double *std_vec,
					  ML_Comm *comm );
  extern void ML_Aggregate_CountLocal( int N_fine, int graph_decomposition[],
				       int N_aggregates, int nodes_per_aggregate[] );
  extern int ML_Aggregate_Viz_Stats_SetUpLevel( ML_Aggregate_Viz_Stats finer_level,
						ML_Aggregate_Viz_Stats *coarser_level,
						int dim );
  extern int ML_Aggregate_VizAndStats_Compute( ML *ml, ML_Aggregate *ag, int MaxMgLevels,
				     double *x, double *y, double *z, int Ndimensions,
				     char *base_filename );
  extern int ML_Info_DomainDecomp( ML_Aggregate_Viz_Stats info,
				   ML_Comm *comm, double *H, double *h );
  extern int ML_Compute_AggregateGraphRadius( int Nrows, int ia[], int ja[],
					      int dep [],
					      int *pradius, int *pNcenter );
  extern int ML_Aggregate_Stats_ComputeCoordinates( ML *ml, ML_Aggregate *ag,
						   double *x, double *y, double *z);
  extern int ML_Aggregate_Stats_CopyCoordinates( ML *ml, ML_Aggregate *ag,
						   double *x, double *y, double *z);
  extern int ML_Aggregate_Stats_Analyze( ML *ml, ML_Aggregate *ag);
  extern int ML_Aggregate_Viz( ML *ml, ML_Aggregate *ag, int choice, 
			      double *, char * base_filename, int level);
  extern int ML_Aggregate_Viz_Amalgamate( ML *ml, ML_Aggregate *ag);
  extern int ML_Aggregate_Viz_UnAmalgamate( ML *ml, ML_Aggregate *ag);
  extern int ML_Aggregate_Stats_CleanUp_Info( ML *ml);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /* #ifndef __MLAGGMETIS__ */
