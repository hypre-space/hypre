
#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "struct_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_hybrid.c */
int HYPRE_StructHybridInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructHybridFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructHybridSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructHybridSetConvergenceTol P((HYPRE_StructSolver solver , double cf_tol ));
int HYPRE_StructHybridSetMaxDSIterations P((HYPRE_StructSolver solver , int max_ds_its ));
int HYPRE_StructHybridSetMaxMGIterations P((HYPRE_StructSolver solver , int max_mg_its ));
int HYPRE_StructHybridSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructHybridSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructHybridSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructHybridGetNumIterations P((HYPRE_StructSolver solver , int *num_its ));
int HYPRE_StructHybridGetNumDSIterations P((HYPRE_StructSolver solver , int *num_ds_its ));
int HYPRE_StructHybridGetNumMGIterations P((HYPRE_StructSolver solver , int *num_mg_its ));
int HYPRE_StructHybridGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPCGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPCGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPCGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPCGSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructPCGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPCGSetPrecond P((HYPRE_StructSolver solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_StructPCGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPCGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPCGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));
int HYPRE_StructDiagScaleSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x ));
int HYPRE_StructDiagScale P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));

/* HYPRE_struct_smg.c */
int HYPRE_StructSMGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructSMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSetMemoryUse P((HYPRE_StructSolver solver , int memory_use ));
int HYPRE_StructSMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructSMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructSMGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructSMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* cyclic_reduction.c */
void *hypre_CyclicReductionInitialize P((MPI_Comm comm ));
hypre_StructMatrix *hypre_CycRedNewCoarseOp P((hypre_StructMatrix *A , hypre_StructGrid *coarse_grid , int cdir ));
int hypre_CycRedSetupCoarseOp P((hypre_StructMatrix *A , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride ));
int hypre_CyclicReductionSetup P((void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_CyclicReduction P((void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_CyclicReductionSetBase P((void *cyc_red_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_CyclicReductionFinalize P((void *cyc_red_vdata ));

/* general.c */
int hypre_Log2 P((int p ));

/* hybrid.c */
void *hypre_HybridInitialize P((MPI_Comm comm ));
int hypre_HybridFinalize P((void *hybrid_vdata ));
int hypre_HybridSetTol P((void *hybrid_vdata , double tol ));
int hypre_HybridSetConvergenceTol P((void *hybrid_vdata , double cf_tol ));
int hypre_HybridSetMaxDSIterations P((void *hybrid_vdata , int max_ds_its ));
int hypre_HybridSetMaxMGIterations P((void *hybrid_vdata , int max_mg_its ));
int hypre_HybridSetTwoNorm P((void *hybrid_vdata , int two_norm ));
int hypre_HybridSetRelChange P((void *hybrid_vdata , int rel_change ));
int hypre_HybridSetLogging P((void *hybrid_vdata , int logging ));
int hypre_HybridGetNumIterations P((void *hybrid_vdata , int *num_its ));
int hypre_HybridGetNumDSIterations P((void *hybrid_vdata , int *num_ds_its ));
int hypre_HybridGetNumMGIterations P((void *hybrid_vdata , int *num_mg_its ));
int hypre_HybridGetFinalRelativeResidualNorm P((void *hybrid_vdata , double *final_rel_res_norm ));
int hypre_HybridSetup P((void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_HybridSolve P((void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* pcg.c */
int hypre_PCGIdentitySetup P((void *vdata , void *A , void *b , void *x ));
int hypre_PCGIdentity P((void *vdata , void *A , void *b , void *x ));
void *hypre_PCGInitialize P((void ));
int hypre_PCGFinalize P((void *pcg_vdata ));
int hypre_PCGSetup P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSolve P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSetTol P((void *pcg_vdata , double tol ));
int hypre_PCGSetConvergenceFactorTol P((void *pcg_vdata , double cf_tol ));
int hypre_PCGSetMaxIter P((void *pcg_vdata , int max_iter ));
int hypre_PCGSetTwoNorm P((void *pcg_vdata , int two_norm ));
int hypre_PCGSetRelChange P((void *pcg_vdata , int rel_change ));
int hypre_PCGSetPrecond P((void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int hypre_PCGSetLogging P((void *pcg_vdata , int logging ));
int hypre_PCGGetNumIterations P((void *pcg_vdata , int *num_iterations ));
int hypre_PCGPrintLogging P((void *pcg_vdata , int myid ));
int hypre_PCGGetFinalRelativeResidualNorm P((void *pcg_vdata , double *relative_residual_norm ));

/* pcg_struct.c */
char *hypre_PCGCAlloc P((int count , int elt_size ));
int hypre_PCGFree P((char *ptr ));
void *hypre_PCGNewVector P((void *vvector ));
int hypre_PCGFreeVector P((void *vvector ));
void *hypre_PCGMatvecInitialize P((void *A , void *x ));
int hypre_PCGMatvec P((void *matvec_data , double alpha , void *A , void *x , double beta , void *y ));
int hypre_PCGMatvecFinalize P((void *matvec_data ));
double hypre_PCGInnerProd P((void *x , void *y ));
int hypre_PCGCopyVector P((void *x , void *y ));
int hypre_PCGClearVector P((void *x ));
int hypre_PCGScaleVector P((double alpha , void *x ));
int hypre_PCGAxpy P((double alpha , void *x , void *y ));

/* smg.c */
void *hypre_SMGInitialize P((MPI_Comm comm ));
int hypre_SMGFinalize P((void *smg_vdata ));
int hypre_SMGSetMemoryUse P((void *smg_vdata , int memory_use ));
int hypre_SMGSetTol P((void *smg_vdata , double tol ));
int hypre_SMGSetMaxIter P((void *smg_vdata , int max_iter ));
int hypre_SMGSetRelChange P((void *smg_vdata , int rel_change ));
int hypre_SMGSetZeroGuess P((void *smg_vdata ));
int hypre_SMGSetNumPreRelax P((void *smg_vdata , int num_pre_relax ));
int hypre_SMGSetNumPostRelax P((void *smg_vdata , int num_post_relax ));
int hypre_SMGSetBase P((void *smg_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_SMGSetLogging P((void *smg_vdata , int logging ));
int hypre_SMGGetNumIterations P((void *smg_vdata , int *num_iterations ));
int hypre_SMGPrintLogging P((void *smg_vdata , int myid ));
int hypre_SMGGetFinalRelativeResidualNorm P((void *smg_vdata , double *relative_residual_norm ));
int hypre_SMGSetStructVectorConstantValues P((hypre_StructVector *vector , double values , hypre_BoxArray *box_array , hypre_Index stride ));

/* smg2_setup_rap.c */
hypre_StructMatrix *hypre_SMG2NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMG2BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG2BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG2RAPPeriodicSym P((hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG2RAPPeriodicNoSym P((hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));

/* smg3_setup_rap.c */
hypre_StructMatrix *hypre_SMG3NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMG3BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG3BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG3RAPPeriodicSym P((hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG3RAPPeriodicNoSym P((hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));

/* smg_intadd.c */
void *hypre_SMGIntAddInitialize P((void ));
int hypre_SMGIntAddSetup P((void *intadd_vdata , hypre_StructMatrix *PT , hypre_StructVector *xc , hypre_StructVector *e , hypre_StructVector *x , hypre_Index cindex , hypre_Index findex , hypre_Index stride ));
int hypre_SMGIntAdd P((void *intadd_vdata , hypre_StructMatrix *PT , hypre_StructVector *xc , hypre_StructVector *e , hypre_StructVector *x ));
int hypre_SMGIntAddFinalize P((void *intadd_vdata ));
int hypre_AppendBoxArrayArrayAndProcs P((int **processes_0 , int **processes_1 , hypre_BoxArrayArray *box_array_array_0 , hypre_BoxArrayArray *box_array_array_1 , int ***processes_ptr ));

/* smg_relax.c */
void *hypre_SMGRelaxInitialize P((MPI_Comm comm ));
int hypre_SMGRelaxFreeTempVec P((void *relax_vdata ));
int hypre_SMGRelaxFreeARem P((void *relax_vdata ));
int hypre_SMGRelaxFreeASol P((void *relax_vdata ));
int hypre_SMGRelaxFinalize P((void *relax_vdata ));
int hypre_SMGRelax P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SMGRelaxSetup P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SMGRelaxSetupTempVec P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SMGRelaxSetupARem P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SMGRelaxSetupASol P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SMGRelaxSetTempVec P((void *relax_vdata , hypre_StructVector *temp_vec ));
int hypre_SMGRelaxSetMemoryUse P((void *relax_vdata , int memory_use ));
int hypre_SMGRelaxSetTol P((void *relax_vdata , double tol ));
int hypre_SMGRelaxSetMaxIter P((void *relax_vdata , int max_iter ));
int hypre_SMGRelaxSetNonZeroGuess P((void *relax_vdata ));
int hypre_SMGRelaxSetZeroGuess P((void *relax_vdata ));
int hypre_SMGRelaxSetNumSpaces P((void *relax_vdata , int num_spaces ));
int hypre_SMGRelaxSetNumPreSpaces P((void *relax_vdata , int num_pre_spaces ));
int hypre_SMGRelaxSetNumRegSpaces P((void *relax_vdata , int num_reg_spaces ));
int hypre_SMGRelaxSetSpace P((void *relax_vdata , int i , int space_index , int space_stride ));
int hypre_SMGRelaxSetRegSpaceRank P((void *relax_vdata , int i , int reg_space_rank ));
int hypre_SMGRelaxSetPreSpaceRank P((void *relax_vdata , int i , int pre_space_rank ));
int hypre_SMGRelaxSetBase P((void *relax_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_SMGRelaxSetNumPreRelax P((void *relax_vdata , int num_pre_relax ));
int hypre_SMGRelaxSetNumPostRelax P((void *relax_vdata , int num_post_relax ));
int hypre_SMGRelaxSetNewMatrixStencil P((void *relax_vdata , hypre_StructStencil *diff_stencil ));
int hypre_SMGRelaxSetupBaseBoxArray P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* smg_residual.c */
void *hypre_SMGResidualInitialize P((void ));
int hypre_SMGResidualSetup P((void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r ));
int hypre_SMGResidual P((void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r ));
int hypre_SMGResidualSetBase P((void *residual_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_SMGResidualFinalize P((void *residual_vdata ));

/* smg_residual_unrolled.c */
void *hypre_SMGResidualInitialize P((void ));
int hypre_SMGResidualSetup P((void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r ));
int hypre_SMGResidual P((void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r ));
int hypre_SMGResidualSetBase P((void *residual_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_SMGResidualFinalize P((void *residual_vdata ));

/* smg_restrict.c */
void *hypre_SMGRestrictInitialize P((void ));
int hypre_SMGRestrictSetup P((void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride ));
int hypre_SMGRestrict P((void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc ));
int hypre_SMGRestrictFinalize P((void *restrict_vdata ));

/* smg_setup.c */
int hypre_SMGSetup P((void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* smg_setup_interp.c */
hypre_StructMatrix *hypre_SMGNewInterpOp P((hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir ));
int hypre_SMGSetupInterpOp P((void *relax_data , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x , hypre_StructMatrix *PT , int cdir , hypre_Index cindex , hypre_Index findex , hypre_Index stride ));

/* smg_setup_rap.c */
hypre_StructMatrix *hypre_SMGNewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMGSetupRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride ));

/* smg_setup_restrict.c */
hypre_StructMatrix *hypre_SMGNewRestrictOp P((hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir ));
int hypre_SMGSetupRestrictOp P((hypre_StructMatrix *A , hypre_StructMatrix *R , hypre_StructVector *temp_vec , int cdir , hypre_Index cindex , hypre_Index cstride ));

/* smg_solve.c */
int hypre_SMGSolve P((void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* thread_wrappers.c */
void HYPRE_StructPCGInitializeVoidPtr P((void *argptr ));
int HYPRE_StructPCGInitializePush P((MPI_Comm comm , HYPRE_StructSolverArray *solver ));
void HYPRE_StructPCGFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructPCGFinalizePush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructPCGSetupVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructPCGSolveVoidPtr P((void *argptr ));
int HYPRE_StructPCGSolvePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructPCGSetTolVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetTolPush P((HYPRE_StructSolverArray solver , double tol ));
void HYPRE_StructPCGSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetMaxIterPush P((HYPRE_StructSolverArray solver , int max_iter ));
void HYPRE_StructPCGSetTwoNormVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetTwoNormPush P((HYPRE_StructSolverArray solver , int two_norm ));
void HYPRE_StructPCGSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetRelChangePush P((HYPRE_StructSolverArray solver , int rel_change ));
void HYPRE_StructPCGSetPrecondVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetPrecondPush P((HYPRE_StructSolverArray solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
void HYPRE_StructPCGSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetLoggingPush P((HYPRE_StructSolverArray solver , int logging ));
void HYPRE_StructPCGGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructPCGGetNumIterationsPush P((HYPRE_StructSolverArray solver , int *num_iterations ));
void HYPRE_StructPCGGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructPCGGetFinalRelativeResidualNormPush P((HYPRE_StructSolverArray solver , double *norm ));
void HYPRE_StructDiagScaleSetupVoidPtr P((void *argptr ));
int HYPRE_StructDiagScaleSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray y , HYPRE_StructVectorArray x ));
void HYPRE_StructDiagScaleVoidPtr P((void *argptr ));
int HYPRE_StructDiagScalePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray HA , HYPRE_StructVectorArray Hy , HYPRE_StructVectorArray Hx ));
void HYPRE_StructSMGInitializeVoidPtr P((void *argptr ));
int HYPRE_StructSMGInitializePush P((MPI_Comm comm , HYPRE_StructSolverArray *solver ));
void HYPRE_StructSMGFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructSMGFinalizePush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructSMGSetupVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructSMGSolveVoidPtr P((void *argptr ));
int HYPRE_StructSMGSolvePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructSMGSetMemoryUseVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetMemoryUsePush P((HYPRE_StructSolverArray solver , int memory_use ));
void HYPRE_StructSMGSetTolVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetTolPush P((HYPRE_StructSolverArray solver , double tol ));
void HYPRE_StructSMGSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetMaxIterPush P((HYPRE_StructSolverArray solver , int max_iter ));
void HYPRE_StructSMGSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetRelChangePush P((HYPRE_StructSolverArray solver , int rel_change ));
void HYPRE_StructSMGSetZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetZeroGuessPush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructSMGSetNumPreRelaxVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetNumPreRelaxPush P((HYPRE_StructSolverArray solver , int num_pre_relax ));
void HYPRE_StructSMGSetNumPostRelaxVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetNumPostRelaxPush P((HYPRE_StructSolverArray solver , int num_post_relax ));
void HYPRE_StructSMGSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetLoggingPush P((HYPRE_StructSolverArray solver , int logging ));
void HYPRE_StructSMGGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructSMGGetNumIterationsPush P((HYPRE_StructSolverArray solver , int *num_iterations ));
void HYPRE_StructSMGGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructSMGGetFinalRelativeResidualNormPush P((HYPRE_StructSolverArray solver , double *norm ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

