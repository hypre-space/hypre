/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
int HYPRE_AMGFinalize( HYPRE_Solver solver );
int HYPRE_AMGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSetMaxLevels( HYPRE_Solver solver , int max_levels );
int HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
int HYPRE_AMGSetCoarsenType( HYPRE_Solver solver , int coarsen_type );
int HYPRE_AMGSetInterpType( HYPRE_Solver solver , int interp_type );
int HYPRE_AMGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_AMGSetCycleType( HYPRE_Solver solver , int cycle_type );
int HYPRE_AMGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_AMGSetNumGridSweeps( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_AMGSetGridRelaxType( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_AMGSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
int HYPRE_AMGSetIOutDat( HYPRE_Solver solver , int ioutdat );
int HYPRE_AMGSetLogFileName( HYPRE_Solver solver , char *log_file_name );
int HYPRE_AMGSetLogging( HYPRE_Solver solver , int ioutdat , char *log_file_name );
int HYPRE_AMGSetNumFunctions( HYPRE_Solver solver , int num_functions );
int HYPRE_AMGSetDofFunc( HYPRE_Solver solver , int *dof_func );

#ifdef __cplusplus
}
#endif

#endif

