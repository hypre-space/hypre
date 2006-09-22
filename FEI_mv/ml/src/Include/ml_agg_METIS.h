/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/********************************************************************* */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */   
/* ******************************************************************** */

/********************************************************************* */
/*          Decomposition with METIS                                  */
/********************************************************************* */

#ifndef __MLAGGMETIS__
#define __MLAGGMETIS__

/*MS*/
#define ML_AGGREGATE_OPTIONS_ID 13579

/* undefined will be a negative number */
#define ML_NUM_LOCAL_AGGREGATES     0
#define ML_NUM_GLOBAL_AGGREGATES    1
#define ML_NUM_NODES_PER_AGGREGATE  2

typedef struct ML_Aggregate_Options_Struct
{
  int id;
  int Naggregates_local;
  int Nnodes_per_aggregate;
  int Naggregates_global;
  int choice;
  int reordering_flag;
  int desired_aggre_per_proc; /* for ParMETIS */
} ML_Aggregate_Options;
/*ms*/

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

  extern int ML_Aggregate_Options_Defaults( ML_Aggregate_Options * pointer,
					    int NumLevels );
  
  extern int ML_Aggregate_Set_NodesPerAggr( ML *ml, ML_Aggregate *ag, 
					    int level, int nodes_per_aggre );
  extern int ML_Aggregate_Set_LocalNumber( ML *ml, ML_Aggregate *ag, 
					   int level, int Nlocal  );
  extern int ML_Aggregate_Set_GlobalNumber( ML *ml, ML_Aggregate *ag, 
					    int level, int Nglobal  );
  extern int ML_Aggregate_Set_ReorderingFlag( ML *ml, ML_Aggregate *ag, 
					      int level, int reordering_flag);
  extern int ML_Aggregate_CoarsenMETIS( ML_Aggregate *ml_ag,
					ML_Operator *Amatrix, 
					ML_Operator **Pmatrix, ML_Comm *comm);
  extern int ML_DecomposeGraph_BuildOffsets( int N_parts,
					     int offsets[],
					     int N_procs,
					     USR_COMM comm);
  extern int ML_Aggregate_Set_OptimalNumberOfNodesPerAggregate( int optimal_value );
  extern int ML_Aggregate_Get_OptimalNumberOfNodesPerAggregate( );
  
  extern int ML_Aggregate_Set_UseDropping(int i);

  extern int ML_Aggregate_Get_UseDropping();
  

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /* #ifndef __MLAGGMETIS__ */
