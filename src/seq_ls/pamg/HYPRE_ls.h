/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header file for HYPRE_ls library
 *
 *****************************************************************************/

#ifndef HYPRE_LS_HEADER
#define HYPRE_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_SOLVER_STRUCT
#define HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize( void );
HYPRE_Int HYPRE_AMGFinalize( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_AMGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_AMGSetMaxLevels( HYPRE_Solver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
HYPRE_Int HYPRE_AMGSetCoarsenType( HYPRE_Solver solver , HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_AMGSetInterpType( HYPRE_Solver solver , HYPRE_Int interp_type );
HYPRE_Int HYPRE_AMGSetMaxIter( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_AMGSetCycleType( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMGSetTol( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_AMGSetNumGridSweeps( HYPRE_Solver solver , HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_AMGSetGridRelaxType( HYPRE_Solver solver , HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver solver , HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_AMGSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
HYPRE_Int HYPRE_AMGSetIOutDat( HYPRE_Solver solver , HYPRE_Int ioutdat );
HYPRE_Int HYPRE_AMGSetLogFileName( HYPRE_Solver solver , char *log_file_name );
HYPRE_Int HYPRE_AMGSetLogging( HYPRE_Solver solver , HYPRE_Int ioutdat , char *log_file_name );
HYPRE_Int HYPRE_AMGSetNumFunctions( HYPRE_Solver solver , HYPRE_Int num_functions );
HYPRE_Int HYPRE_AMGSetDofFunc( HYPRE_Solver solver , HYPRE_Int *dof_func );
HYPRE_Int HYPRE_AMGSetUseBlockFlag( HYPRE_Solver solver , HYPRE_Int use_block_flag );

#ifdef __cplusplus
}
#endif

#endif

