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

#ifndef HYPRE_PARCSR_LS_HEADER
#define HYPRE_PARCSR_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"
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

# define        P(s) s

/* HYPRE_parcsr_amg.c */
int HYPRE_ParAMGCreate P((HYPRE_Solver *solver ));
int HYPRE_ParAMGDestroy P((HYPRE_Solver solver ));
int HYPRE_ParAMGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSolveT P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_ParAMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_ParAMGSetTruncFactor P((HYPRE_Solver solver , double trunc_factor ));
int HYPRE_ParAMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_ParAMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParAMGSetCoarsenType P((HYPRE_Solver solver , int coarsen_type ));
int HYPRE_ParAMGSetMeasureType P((HYPRE_Solver solver , int measure_type ));
int HYPRE_ParAMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_ParAMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParAMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_ParAMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_ParAMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_ParAMGSetRelaxWeight P((HYPRE_Solver solver , double *relax_weight ));
int HYPRE_ParAMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_ParAMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_ParAMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));
int HYPRE_ParAMGSetDebugFlag P((HYPRE_Solver solver , int debug_flag ));

/* HYPRE_parcsr_gmres.c */
int HYPRE_ParCSRGMRESCreate P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRGMRESDestroy P((HYPRE_Solver solver ));
int HYPRE_ParCSRGMRESSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRGMRESSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRGMRESSetKDim P((HYPRE_Solver solver , int k_dim ));
int HYPRE_ParCSRGMRESSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRGMRESSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRGMRESSetPrecond P((HYPRE_Solver solver , int (*precond )(
HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, HYPRE_ParVector b, HYPRE_ParVector
x), int (*precond_setup )(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, 
HYPRE_ParVector b, HYPRE_ParVector x), void *precond_data ));
int HYPRE_ParCSRGMRESSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRGMRESGetNumIterations P((HYPRE_Solver solver , int *num_iterations));
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));

/* HYPRE_parcsr_cgnr.c */
int HYPRE_ParCSRCGNRCreate P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRCGNRDestroy P((HYPRE_Solver solver ));
int HYPRE_ParCSRCGNRSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRCGNRSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRCGNRSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRCGNRSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRCGNRSetPrecond P((HYPRE_Solver solver , int (*precond )(
HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, HYPRE_ParVector b, HYPRE_ParVector
x), int (*precondT )(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, 
HYPRE_ParVector b, HYPRE_ParVector x), int (*precond_setup )(HYPRE_Solver sol, 
HYPRE_ParCSRMatrix matrix, HYPRE_ParVector b, HYPRE_ParVector x), void 
*precond_data ));
int HYPRE_ParCSRCGNRSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRCGNRGetNumIterations P((HYPRE_Solver solver , int *num_iterations));
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));
 
/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGCreate P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRPCGDestroy P((HYPRE_Solver solver ));
int HYPRE_ParCSRPCGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRPCGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRPCGSetTwoNorm P((HYPRE_Solver solver , int two_norm ));
int HYPRE_ParCSRPCGSetRelChange P((HYPRE_Solver solver , int rel_change )) ;
int HYPRE_ParCSRPCGSetPrecond P((HYPRE_Solver solver , int (*precond )(
HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, HYPRE_ParVector b, HYPRE_ParVector
x), int (*precond_setup )(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix, 
HYPRE_ParVector b, HYPRE_ParVector x), void *precond_data ));
int HYPRE_ParCSRPCGSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRPCGGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));
int HYPRE_ParCSRDiagScaleSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x ));
int HYPRE_ParCSRDiagScale P((HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx ));

/* HYPRE_parcsr_pilut.c */
int HYPRE_ParCSRPilutCreate P(( MPI_Comm comm, HYPRE_Solver *solver ));
int HYPRE_ParCSRPilutDestroy P((HYPRE_Solver solver ));
int HYPRE_ParCSRPilutSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPilutSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPilutSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRPilutSetDropTolerance P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRPilutSetFactorRowSize P((HYPRE_Solver solver , int size ));

/* HYPRE_parcsr_ParaSails.c */
int HYPRE_ParCSRParaSailsCreate P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRParaSailsDestroy P((HYPRE_Solver solver ));
int HYPRE_ParCSRParaSailsSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRParaSailsSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRParaSailsSetParams P((HYPRE_Solver solver , double thresh , int nlevels ));

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian P((MPI_Comm comm , int nx , int ny , int nz
 , int P , int Q , int R , int p , int q , int r , double *value ));
int map P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , 
int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt P((MPI_Comm comm , int nx , int ny , 
int nz , int P , int Q , int R , int p , int q , int r , double *value ));
int map3 P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , 
int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt P((MPI_Comm comm , int nx , int ny , int
 P , int Q , int p , int q , double *value ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int
 *ny_part , int *global_part ));

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv P((MPI_Comm comm , int nx , int ny , int nz ,
 int P , int Q , int R , int p , int q , int r , double *value ));
#undef P

#undef P

#ifdef __cplusplus
}
#endif

#endif

