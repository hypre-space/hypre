
#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "struct_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

# define	P(s) s

/* F90_HYPRE_struct_hybrid.c */
void hypre_F90_IFACE P((int hypre_structhybridinitialize ));
void hypre_F90_IFACE P((int hypre_structhybridfinalize ));
void hypre_F90_IFACE P((int hypre_structhybridsetup ));
void hypre_F90_IFACE P((int hypre_structhybridsolve ));
void hypre_F90_IFACE P((int hypre_structhybridsettol ));
void hypre_F90_IFACE P((int hypre_structhybridsetconvergenc ));
void hypre_F90_IFACE P((int hypre_structhybridsetdscgmaxite ));
void hypre_F90_IFACE P((int hypre_structhybridsetpcgmaxiter ));
void hypre_F90_IFACE P((int hypre_structhybridsettwonorm ));
void hypre_F90_IFACE P((int hypre_structhybridsetrelchange ));
void hypre_F90_IFACE P((int hypre_structhybridsetprecond ));
void hypre_F90_IFACE P((int hypre_structhybridsetlogging ));
void hypre_F90_IFACE P((int hypre_structhybridgetnumiterati ));
void hypre_F90_IFACE P((int hypre_structhybridgetdscgnumite ));
void hypre_F90_IFACE P((int hypre_structhybridgetpcgnumiter ));
void hypre_F90_IFACE P((int hypre_structhybridgetfinalrelat ));

/* F90_HYPRE_struct_pcg.c */
void hypre_F90_IFACE P((int hypre_structpcginitialize ));
void hypre_F90_IFACE P((int hypre_structpcgfinalize ));
void hypre_F90_IFACE P((int hypre_structpcgsetup ));
void hypre_F90_IFACE P((int hypre_structpcgsolve ));
void hypre_F90_IFACE P((int hypre_structpcgsettol ));
void hypre_F90_IFACE P((int hypre_structpcgsetmaxiter ));
void hypre_F90_IFACE P((int hypre_structpcgsettwonorm ));
void hypre_F90_IFACE P((int hypre_structpcgsetrelchange ));
void hypre_F90_IFACE P((int hypre_structpcgsetprecond ));
void hypre_F90_IFACE P((int hypre_structpcgsetlogging ));
void hypre_F90_IFACE P((int hypre_structpcggetnumiterations ));
void hypre_F90_IFACE P((int hypre_structpcggetfinalrelative ));
void hypre_F90_IFACE P((int hypre_structdiagscalesetup ));
void hypre_F90_IFACE P((int hypre_structdiagscale ));

/* F90_HYPRE_struct_pfmg.c */
void hypre_F90_IFACE P((int hypre_structpfmginitialize ));
void hypre_F90_IFACE P((int hypre_structpfmgfinalize ));
void hypre_F90_IFACE P((int hypre_structpfmgsetup ));
void hypre_F90_IFACE P((int hypre_structpfmgsolve ));
void hypre_F90_IFACE P((int hypre_structpfmgsettol ));
void hypre_F90_IFACE P((int hypre_structpfmgsetmaxiter ));
void hypre_F90_IFACE P((int hypre_structpfmgsetrelchange ));
void hypre_F90_IFACE P((int hypre_structpfmgsetzeroguess ));
void hypre_F90_IFACE P((int hypre_structpfmgsetnonzeroguess ));
void hypre_F90_IFACE P((int hypre_structpfmgsetrelaxtype ));
void hypre_F90_IFACE P((int hypre_structpfmgsetnumprerelax ));
void hypre_F90_IFACE P((int hypre_structpfmgsetnumpostrelax ));
void hypre_F90_IFACE P((int hypre_structpfmgsetdxyz ));
void hypre_F90_IFACE P((int hypre_structpfmgsetlogging ));
void hypre_F90_IFACE P((int hypre_structpfmggetnumiteration ));
void hypre_F90_IFACE P((int hypre_structpfmggetfinalrelativ ));

/* F90_HYPRE_struct_smg.c */
void hypre_F90_IFACE P((int hypre_structsmginitialize ));
void hypre_F90_IFACE P((int hypre_structsmgfinalize ));
void hypre_F90_IFACE P((int hypre_structsmgsetup ));
void hypre_F90_IFACE P((int hypre_structsmgsolve ));
void hypre_F90_IFACE P((int hypre_structsmgsetmemoryuse ));
void hypre_F90_IFACE P((int hypre_structsmgsettol ));
void hypre_F90_IFACE P((int hypre_structsmgsetmaxiter ));
void hypre_F90_IFACE P((int hypre_structsmgsetrelchange ));
void hypre_F90_IFACE P((int hypre_structsmgsetzeroguess ));
void hypre_F90_IFACE P((int hypre_structsmgsetnonzeroguess ));
void hypre_F90_IFACE P((int hypre_structsmgsetnumprerelax ));
void hypre_F90_IFACE P((int hypre_structsmgsetnumpostrelax ));
void hypre_F90_IFACE P((int hypre_structsmgsetlogging ));
void hypre_F90_IFACE P((int hypre_structsmggetnumiterations ));
void hypre_F90_IFACE P((int hypre_structsmggetfinalrelative ));

/* HYPRE_struct_hybrid.c */
int HYPRE_StructHybridInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructHybridFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructHybridSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructHybridSetConvergenceTol P((HYPRE_StructSolver solver , double cf_tol ));
int HYPRE_StructHybridSetDSCGMaxIter P((HYPRE_StructSolver solver , int dscg_max_its ));
int HYPRE_StructHybridSetPCGMaxIter P((HYPRE_StructSolver solver , int pcg_max_its ));
int HYPRE_StructHybridSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructHybridSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructHybridSetPrecond P((HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver ));
int HYPRE_StructHybridSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructHybridGetNumIterations P((HYPRE_StructSolver solver , int *num_its ));
int HYPRE_StructHybridGetDSCGNumIterations P((HYPRE_StructSolver solver , int *dscg_num_its ));
int HYPRE_StructHybridGetPCGNumIterations P((HYPRE_StructSolver solver , int *pcg_num_its ));
int HYPRE_StructHybridGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_jacobi.c */
int HYPRE_StructJacobiInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructJacobiFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructJacobiSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructJacobiSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructJacobiSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructJacobiSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructJacobiGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPCGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPCGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPCGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPCGSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructPCGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPCGSetPrecond P((HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver ));
int HYPRE_StructPCGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPCGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPCGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));
int HYPRE_StructDiagScaleSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x ));
int HYPRE_StructDiagScale P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));

/* HYPRE_struct_pfmg.c */
int HYPRE_StructPFMGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPFMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPFMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPFMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPFMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPFMGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPFMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetRelaxType P((HYPRE_StructSolver solver , int relax_type ));
int HYPRE_StructPFMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructPFMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructPFMGSetSkipRelax P((HYPRE_StructSolver solver , int skip_relax ));
int HYPRE_StructPFMGSetDxyz P((HYPRE_StructSolver solver , double *dxyz ));
int HYPRE_StructPFMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPFMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPFMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

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
int HYPRE_StructSMGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_sparse_msg.c */
int HYPRE_StructSparseMSGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructSparseMSGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSparseMSGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSparseMSGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructSparseMSGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructSparseMSGSetJump P((HYPRE_StructSolver solver , int jump ));
int HYPRE_StructSparseMSGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructSparseMSGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetRelaxType P((HYPRE_StructSolver solver , int relax_type ));
int HYPRE_StructSparseMSGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSparseMSGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSparseMSGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSparseMSGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

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
int hypre_HybridSetDSCGMaxIter P((void *hybrid_vdata , int dscg_max_its ));
int hypre_HybridSetPCGMaxIter P((void *hybrid_vdata , int pcg_max_its ));
int hypre_HybridSetTwoNorm P((void *hybrid_vdata , int two_norm ));
int hypre_HybridSetRelChange P((void *hybrid_vdata , int rel_change ));
int hypre_HybridSetPrecond P((void *pcg_vdata , int (*pcg_precond_solve )(), int (*pcg_precond_setup )(), void *pcg_precond ));
int hypre_HybridSetLogging P((void *hybrid_vdata , int logging ));
int hypre_HybridGetNumIterations P((void *hybrid_vdata , int *num_its ));
int hypre_HybridGetDSCGNumIterations P((void *hybrid_vdata , int *dscg_num_its ));
int hypre_HybridGetPCGNumIterations P((void *hybrid_vdata , int *pcg_num_its ));
int hypre_HybridGetFinalRelativeResidualNorm P((void *hybrid_vdata , double *final_rel_res_norm ));
int hypre_HybridSetup P((void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_HybridSolve P((void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* jacobi.c */
void *hypre_JacobiInitialize P((MPI_Comm comm ));
int hypre_JacobiFinalize P((void *jacobi_vdata ));
int hypre_JacobiSetup P((void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_JacobiSolve P((void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_JacobiSetTol P((void *jacobi_vdata , double tol ));
int hypre_JacobiSetMaxIter P((void *jacobi_vdata , int max_iter ));
int hypre_JacobiSetZeroGuess P((void *jacobi_vdata , int zero_guess ));
int hypre_JacobiSetTempVec P((void *jacobi_vdata , hypre_StructVector *t ));

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

/* pfmg.c */
void *hypre_PFMGInitialize P((MPI_Comm comm ));
int hypre_PFMGFinalize P((void *pfmg_vdata ));
int hypre_PFMGSetTol P((void *pfmg_vdata , double tol ));
int hypre_PFMGSetMaxIter P((void *pfmg_vdata , int max_iter ));
int hypre_PFMGSetRelChange P((void *pfmg_vdata , int rel_change ));
int hypre_PFMGSetZeroGuess P((void *pfmg_vdata , int zero_guess ));
int hypre_PFMGSetRelaxType P((void *pfmg_vdata , int relax_type ));
int hypre_PFMGSetNumPreRelax P((void *pfmg_vdata , int num_pre_relax ));
int hypre_PFMGSetNumPostRelax P((void *pfmg_vdata , int num_post_relax ));
int hypre_PFMGSetSkipRelax P((void *pfmg_vdata , int skip_relax ));
int hypre_PFMGSetDxyz P((void *pfmg_vdata , double *dxyz ));
int hypre_PFMGSetLogging P((void *pfmg_vdata , int logging ));
int hypre_PFMGGetNumIterations P((void *pfmg_vdata , int *num_iterations ));
int hypre_PFMGPrintLogging P((void *pfmg_vdata , int myid ));
int hypre_PFMGGetFinalRelativeResidualNorm P((void *pfmg_vdata , double *relative_residual_norm ));

/* pfmg2_setup_rap.c */
hypre_StructMatrix *hypre_PFMG2NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir ));
int hypre_PFMG2BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP ));
int hypre_PFMG2BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP ));

/* pfmg3_setup_rap.c */
hypre_StructMatrix *hypre_PFMG3NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir ));
int hypre_PFMG3BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP ));
int hypre_PFMG3BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP ));

/* pfmg_interp.c */
void *hypre_PFMGInterpInitialize P((void ));
int hypre_PFMGInterpSetup P((void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride ));
int hypre_PFMGInterp P((void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e ));
int hypre_PFMGInterpFinalize P((void *interp_vdata ));

/* pfmg_relax.c */
void *hypre_PFMGRelaxInitialize P((MPI_Comm comm ));
int hypre_PFMGRelaxFinalize P((void *pfmg_relax_vdata ));
int hypre_PFMGRelax P((void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_PFMGRelaxSetup P((void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_PFMGRelaxSetType P((void *pfmg_relax_vdata , int relax_type ));
int hypre_PFMGRelaxSetPreRelax P((void *pfmg_relax_vdata ));
int hypre_PFMGRelaxSetPostRelax P((void *pfmg_relax_vdata ));
int hypre_PFMGRelaxSetTol P((void *pfmg_relax_vdata , double tol ));
int hypre_PFMGRelaxSetMaxIter P((void *pfmg_relax_vdata , int max_iter ));
int hypre_PFMGRelaxSetZeroGuess P((void *pfmg_relax_vdata , int zero_guess ));
int hypre_PFMGRelaxSetTempVec P((void *pfmg_relax_vdata , hypre_StructVector *t ));

/* pfmg_restrict.c */
void *hypre_PFMGRestrictInitialize P((void ));
int hypre_PFMGRestrictSetup P((void *restrict_vdata , hypre_StructMatrix *RT , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride ));
int hypre_PFMGRestrict P((void *restrict_vdata , hypre_StructMatrix *RT , hypre_StructVector *r , hypre_StructVector *rc ));
int hypre_PFMGRestrictFinalize P((void *restrict_vdata ));

/* pfmg_setup.c */
int hypre_PFMGSetup P((void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_PFMGComputeDxyz P((hypre_StructMatrix *A , double *dxyz ));

/* pfmg_setup_interp.c */
hypre_StructMatrix *hypre_PFMGNewInterpOp P((hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir ));
int hypre_PFMGSetupInterpOp P((hypre_StructMatrix *A , int cdir , hypre_Index findex , hypre_Index stride , hypre_StructMatrix *P ));

/* pfmg_setup_rap.c */
hypre_StructMatrix *hypre_PFMGNewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir ));
int hypre_PFMGSetupRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *Ac ));

/* pfmg_solve.c */
int hypre_PFMGSolve P((void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* point_relax.c */
void *hypre_PointRelaxInitialize P((MPI_Comm comm ));
int hypre_PointRelaxFinalize P((void *relax_vdata ));
int hypre_PointRelaxSetup P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_PointRelax P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_PointRelaxSetTol P((void *relax_vdata , double tol ));
int hypre_PointRelaxSetMaxIter P((void *relax_vdata , int max_iter ));
int hypre_PointRelaxSetZeroGuess P((void *relax_vdata , int zero_guess ));
int hypre_PointRelaxSetWeight P((void *relax_vdata , double weight ));
int hypre_PointRelaxSetNumPointsets P((void *relax_vdata , int num_pointsets ));
int hypre_PointRelaxSetPointset P((void *relax_vdata , int pointset , int pointset_size , hypre_Index pointset_stride , hypre_Index *pointset_indices ));
int hypre_PointRelaxSetPointsetRank P((void *relax_vdata , int pointset , int pointset_rank ));
int hypre_PointRelaxSetTempVec P((void *relax_vdata , hypre_StructVector *t ));

/* smg.c */
void *hypre_SMGInitialize P((MPI_Comm comm ));
int hypre_SMGFinalize P((void *smg_vdata ));
int hypre_SMGSetMemoryUse P((void *smg_vdata , int memory_use ));
int hypre_SMGSetTol P((void *smg_vdata , double tol ));
int hypre_SMGSetMaxIter P((void *smg_vdata , int max_iter ));
int hypre_SMGSetRelChange P((void *smg_vdata , int rel_change ));
int hypre_SMGSetZeroGuess P((void *smg_vdata , int zero_guess ));
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
int hypre_SMGRelaxSetZeroGuess P((void *relax_vdata , int zero_guess ));
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

/* sparse_msg.c */
void *hypre_SparseMSGInitialize P((MPI_Comm comm ));
int hypre_SparseMSGFinalize P((void *SparseMSG_vdata ));
int hypre_SparseMSGSetTol P((void *SparseMSG_vdata , double tol ));
int hypre_SparseMSGSetMaxIter P((void *SparseMSG_vdata , int max_iter ));
int hypre_SparseMSGSetJump P((void *SparseMSG_vdata , int jump ));
int hypre_SparseMSGSetRelChange P((void *SparseMSG_vdata , int rel_change ));
int hypre_SparseMSGSetZeroGuess P((void *SparseMSG_vdata , int zero_guess ));
int hypre_SparseMSGSetRelaxType P((void *SparseMSG_vdata , int relax_type ));
int hypre_SparseMSGSetNumPreRelax P((void *SparseMSG_vdata , int num_pre_relax ));
int hypre_SparseMSGSetNumPostRelax P((void *SparseMSG_vdata , int num_post_relax ));
int hypre_SparseMSGSetLogging P((void *SparseMSG_vdata , int logging ));
int hypre_SparseMSGGetNumIterations P((void *SparseMSG_vdata , int *num_iterations ));
int hypre_SparseMSGPrintLogging P((void *SparseMSG_vdata , int myid ));
int hypre_SparseMSGGetFinalRelativeResidualNorm P((void *SparseMSG_vdata , double *relative_residual_norm ));

/* sparse_msg_setup.c */
int hypre_SparseMSGSetup P((void *SparseMSG_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_SparseMSGComputeRestrictWeights P((int *num_levels , double *restrict_weights ));
int hypre_SparseMSGComputeInterpWeights P((int *num_levels , double *interp_weights ));

/* sparse_msg_setup_interp.c */
int hypre_SparseMSGSetupInterpOp P((hypre_StructMatrix *Q , hypre_Index findex , hypre_Index stride , hypre_StructMatrix *P ));

/* sparse_msg_solve.c */
int hypre_SparseMSGSolve P((void *SparseMSG_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* thread_wrappers.c */
void HYPRE_StructHybridInitializeVoidPtr P((void *argptr ));
int HYPRE_StructHybridInitializePush P((MPI_Comm comm , HYPRE_StructSolverArray *solver ));
void HYPRE_StructHybridFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructHybridFinalizePush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructHybridSetupVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructHybridSolveVoidPtr P((void *argptr ));
int HYPRE_StructHybridSolvePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructHybridSetTolVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetTolPush P((HYPRE_StructSolverArray solver , double tol ));
void HYPRE_StructHybridSetConvergenceTolVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetConvergenceTolPush P((HYPRE_StructSolverArray solver , double cf_tol ));
void HYPRE_StructHybridSetDSCGMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetDSCGMaxIterPush P((HYPRE_StructSolverArray solver , int dscg_max_its ));
void HYPRE_StructHybridSetPCGMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetPCGMaxIterPush P((HYPRE_StructSolverArray solver , int pcg_max_its ));
void HYPRE_StructHybridSetTwoNormVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetTwoNormPush P((HYPRE_StructSolverArray solver , int two_norm ));
void HYPRE_StructHybridSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetRelChangePush P((HYPRE_StructSolverArray solver , int rel_change ));
void HYPRE_StructHybridSetPrecondVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetPrecondPush P((HYPRE_StructSolverArray solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolverArray precond_solver ));
void HYPRE_StructHybridSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructHybridSetLoggingPush P((HYPRE_StructSolverArray solver , int logging ));
void HYPRE_StructHybridGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructHybridGetNumIterationsPush P((HYPRE_StructSolverArray solver , int *num_its ));
void HYPRE_StructHybridGetDSCGNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructHybridGetDSCGNumIterationsPush P((HYPRE_StructSolverArray solver , int *dscg_num_its ));
void HYPRE_StructHybridGetPCGNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructHybridGetPCGNumIterationsPush P((HYPRE_StructSolverArray solver , int *pcg_num_its ));
void HYPRE_StructHybridGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructHybridGetFinalRelativeResidualNormPush P((HYPRE_StructSolverArray solver , double *norm ));
void HYPRE_StructJacobiInitializeVoidPtr P((void *argptr ));
int HYPRE_StructJacobiInitializePush P((MPI_Comm comm , HYPRE_StructSolverArray *solver ));
void HYPRE_StructJacobiFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructJacobiFinalizePush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructJacobiSetupVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructJacobiSolveVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSolvePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructJacobiSetTolVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSetTolPush P((HYPRE_StructSolverArray solver , double tol ));
void HYPRE_StructJacobiSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSetMaxIterPush P((HYPRE_StructSolverArray solver , int max_iter ));
void HYPRE_StructJacobiSetZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSetZeroGuessPush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructJacobiSetNonZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructJacobiSetNonZeroGuessPush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructJacobiGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructJacobiGetNumIterationsPush P((HYPRE_StructSolverArray solver , int *num_iterations ));
void HYPRE_StructJacobiGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructJacobiGetFinalRelativeResidualNormPush P((HYPRE_StructSolverArray solver , double *norm ));
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
int HYPRE_StructPCGSetPrecondPush P((HYPRE_StructSolverArray solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolverArray precond_solver ));
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
void HYPRE_StructPFMGInitializeVoidPtr P((void *argptr ));
int HYPRE_StructPFMGInitializePush P((MPI_Comm comm , HYPRE_StructSolverArray *solver ));
void HYPRE_StructPFMGFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructPFMGFinalizePush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructPFMGSetupVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetupPush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructPFMGSolveVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSolvePush P((HYPRE_StructSolverArray solver , HYPRE_StructMatrixArray A , HYPRE_StructVectorArray b , HYPRE_StructVectorArray x ));
void HYPRE_StructPFMGSetTolVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetTolPush P((HYPRE_StructSolverArray solver , double tol ));
void HYPRE_StructPFMGSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetMaxIterPush P((HYPRE_StructSolverArray solver , int max_iter ));
void HYPRE_StructPFMGSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetRelChangePush P((HYPRE_StructSolverArray solver , int rel_change ));
void HYPRE_StructPFMGSetZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetZeroGuessPush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructPFMGSetNonZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetNonZeroGuessPush P((HYPRE_StructSolverArray solver ));
void HYPRE_StructPFMGSetRelaxTypeVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetRelaxTypePush P((HYPRE_StructSolverArray solver , int relax_type ));
void HYPRE_StructPFMGSetNumPreRelaxVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetNumPreRelaxPush P((HYPRE_StructSolverArray solver , int num_pre_relax ));
void HYPRE_StructPFMGSetNumPostRelaxVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetNumPostRelaxPush P((HYPRE_StructSolverArray solver , int num_post_relax ));
void HYPRE_StructPFMGSetSkipRelaxVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetSkipRelaxPush P((HYPRE_StructSolverArray solver , int skip_relax ));
void HYPRE_StructPFMGSetDxyzVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetDxyzPush P((HYPRE_StructSolverArray solver , double *dxyz ));
void HYPRE_StructPFMGSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructPFMGSetLoggingPush P((HYPRE_StructSolverArray solver , int logging ));
void HYPRE_StructPFMGGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructPFMGGetNumIterationsPush P((HYPRE_StructSolverArray solver , int *num_iterations ));
void HYPRE_StructPFMGGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructPFMGGetFinalRelativeResidualNormPush P((HYPRE_StructSolverArray solver , double *norm ));
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
void HYPRE_StructSMGSetNonZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetNonZeroGuessPush P((HYPRE_StructSolverArray solver ));
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

