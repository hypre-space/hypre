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
#include "HYPRE_parcsr_mv.h"

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


/* HYPRE_parcsr_amg.c */
HYPRE_Solver HYPRE_ParAMGInitialize P((void ));
int HYPRE_ParAMGFinalize P((HYPRE_Solver solver ));
int HYPRE_ParAMGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_ParAMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_ParAMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_ParAMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParAMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_ParAMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParAMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_ParAMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_ParAMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_ParAMGSetRelaxWeight P((HYPRE_Solver solver , double relax_weight ));
int HYPRE_ParAMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_ParAMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_ParAMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));

/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGInitialize P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRPCGFinalize P((HYPRE_Solver solver ));
int HYPRE_ParCSRPCGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRPCGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRPCGSetTwoNorm P((HYPRE_Solver solver , int two_norm ));
int HYPRE_ParCSRPCGSetRelChange P((HYPRE_Solver solver , int rel_change )) ;
int HYPRE_ParCSRPCGSetPrecond P((HYPRE_Solver solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_ParCSRPCGSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRPCGGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));
int HYPRE_ParCSRDiagScaleSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x ));
int HYPRE_ParCSRDiagScale P((HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

