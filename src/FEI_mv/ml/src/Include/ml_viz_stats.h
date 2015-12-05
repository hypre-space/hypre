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

#ifndef __MLVIZSTATS__
#define __MLVIZSTATS__

/*MS*/
#define ML_AGGREGATE_VIZ_STATS_ID 24680

typedef struct ML_Aggregate_Viz_Stats_Struct
{
  int id;
  double *x;
  double *y;
  double *z;
  int Ndim;
  int *graph_decomposition;
  int Nlocal;
  int Naggregates;
  int local_or_global;
  int is_filled;
  int MaxNodesPerAgg;
  void *Amatrix;  /* void * so that I do not have to include
		     ml_operator.h */
  
} ML_Aggregate_Viz_Stats;
/*ms*/

#endif /* #ifndef __MLAGGMETIS__ */
