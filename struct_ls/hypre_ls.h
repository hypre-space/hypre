
#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#ifndef HYPRE_SEQUENTIAL
#include "mpi.h"
#endif
#include "HYPRE_ls.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_pcg.c */
HYPRE_StructSolver HYPRE_StructPCGInitialize P((MPI_Comm comm ));
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
HYPRE_StructSolver HYPRE_StructSMGInitialize P((MPI_Comm comm ));
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

/* driver.c */
int main P((int argc , char *argv []));

/* driver_internal.c */
int main P((int argc , char *argv []));

/* driver_internal_cgsmg.c */
int main P((int argc , char *argv []));

/* driver_internal_const_coef.c */
int main P((int argc , char *argv []));

/* driver_internal_overlap.c */
int main P((int argc , char *argv []));

/* general.c */
int hypre_Log2 P((int p ));

/* pcg.c */
int hypre_PCGIdentitySetup P((void *vdata , void *A , void *b , void *x ));
int hypre_PCGIdentity P((void *vdata , void *A , void *b , void *x ));
void *hypre_PCGInitialize P((void ));
int hypre_PCGFinalize P((void *pcg_vdata ));
int hypre_PCGSetup P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSolve P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSetTol P((void *pcg_vdata , double tol ));
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
void hypre_PCGFree P((char *ptr ));
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
int hypre_SMGSetStructVectorConstantValues P((hypre_StructVector *vector , double values , hypre_SBoxArray *sbox_array ));

/* smg2_setup_rap.c */
hypre_StructMatrix *hypre_SMG2NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMG2BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG2BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));

/* smg3_setup_rap.c */
hypre_StructMatrix *hypre_SMG3NewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMG3BuildRAPSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));
int hypre_SMG3BuildRAPNoSym P((hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride ));

/* smg_intadd.c */
void *hypre_SMGIntAddInitialize P((void ));
int hypre_SMGIntAddSetup P((void *intadd_vdata , hypre_StructMatrix *PT , hypre_StructVector *xc , hypre_StructVector *e , hypre_StructVector *x , hypre_Index cindex , hypre_Index cstride , hypre_Index findex , hypre_Index fstride ));
int hypre_SMGIntAdd P((void *intadd_vdata , hypre_StructMatrix *PT , hypre_StructVector *xc , hypre_StructVector *e , hypre_StructVector *x ));
int hypre_SMGIntAddFinalize P((void *intadd_vdata ));
void hypre_AppendSBoxArrayArrayAndProcs P((int **processes_0 , int **processes_1 , hypre_SBoxArrayArray *sbox_array_array_0 , hypre_SBoxArrayArray *sbox_array_array_1 , int ***processes_ptr ));

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
int hypre_SMGRelaxSetupBaseSBoxArray P((void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

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
int hypre_SMGRestrictSetup P((void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index cstride , hypre_Index findex , hypre_Index fstride ));
int hypre_SMGRestrict P((void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc ));
int hypre_SMGRestrictFinalize P((void *restrict_vdata ));

/* smg_setup.c */
int hypre_SMGSetup P((void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

/* smg_setup_interp.c */
hypre_StructMatrix *hypre_SMGNewInterpOp P((hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir ));
int hypre_SMGSetupInterpOp P((void *relax_data , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x , hypre_StructMatrix *PT , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index findex , hypre_Index fstride ));

/* smg_setup_rap.c */
hypre_StructMatrix *hypre_SMGNewRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT ));
int hypre_SMGSetupRAPOp P((hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride ));

/* smg_setup_restrict.c */
hypre_StructMatrix *hypre_SMGNewRestrictOp P((hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir ));
int hypre_SMGSetupRestrictOp P((hypre_StructMatrix *A , hypre_StructMatrix *R , hypre_StructVector *temp_vec , int cdir , hypre_Index cindex , hypre_Index cstride ));

/* smg_solve.c */
int hypre_SMGSolve P((void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

