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
#include "HYPRE_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {int opaque;} *HYPRE_Solver;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize P((void ));
int HYPRE_AMGFinalize P((HYPRE_Solver solver ));
int HYPRE_AMGSetup P((HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSolve P((HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_AMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_AMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_AMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_AMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_AMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_AMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_AMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_AMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_AMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_AMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_AMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

