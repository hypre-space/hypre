#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_pcg.c */
void HYPRE_PCG P((Vector *x , Vector *b , double tol , void *data ));

/* HYPRE_struct_pcg.c */
int HYPRE_Matvec P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));
double HYPRE_InnerProd P((Vector *x , Vector *y ));
int HYPRE_CopyVector P((Vector *x , Vector *y ));
int HYPRE_InitVector P((Vector *x , double value ));
int HYPRE_ScaleVector P((double alpha , Vector *x ));
int HYPRE_Axpy P((double alpha , Vector *x , Vector *y ));
void HYPRE_PCGSetup P((Matrix *vA , int (*HYPRE_PCGPrecond )(), void *precond_data , void *data ));
void HYPRE_PCGSMGPrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int HYPRE_PCGSMGPrecond P((Vector *x , Vector *y , double dummy , void *precond_vdata ));
void HYPRE_FreePCGSMGData P((void *data ));
void HYPRE_PCGDiagScalePrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int HYPRE_PCGDiagScalePrecond P((Vector *vx , Vector *vy , double dummy , void *precond_vdata ));
void HYPRE_FreePCGDiagScaleData P((void *data ));

/* HYPRE_struct_smg.c */
HYPRE_StructSolver HYPRE_StructSMGInitialize P((MPI_Comm comm ));
int HYPRE_StructSMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_SMGSetMemoryUse P((HYPRE_StructSolver solver , int memory_use ));
int HYPRE_SMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_SMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_SMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_SMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_SMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_SMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_SMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* cyclic_reduction.c */
void *hypre_CyclicReductionInitialize P((MPI_Comm comm ));
hypre_StructMatrix *hypre_CycRedNewCoarseOp P((hypre_StructMatrix *A , hypre_StructGrid *coarse_grid , int cdir ));
int hypre_CycRedSetupCoarseOp P((hypre_StructMatrix *A , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride ));
int hypre_CyclicReductionSetup P((void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_CyclicReduction P((void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x ));
int hypre_CyclicReductionSetBase P((void *cyc_red_vdata , hypre_Index base_index , hypre_Index base_stride ));
int hypre_CyclicReductionFinalize P((void *cyc_red_vdata ));

/* driver_internal.c */
int main P((int argc , char *argv []));

/* driver_internal_cgsmg.c */
int main P((int argc , char *argv []));

/* driver_internal_const_coef.c */
int main P((int argc , char *argv []));

/* general.c */
int hypre_Log2 P((int p ));

/* linear_interface.c */

/* smg.c */
void *hypre_SMGInitialize P((MPI_Comm comm ));
int hypre_SMGFinalize P((void *smg_vdata ));
int hypre_SMGSetMemoryUse P((void *smg_vdata , int memory_use ));
int hypre_SMGSetTol P((void *smg_vdata , double tol ));
int hypre_SMGSetMaxIter P((void *smg_vdata , int max_iter ));
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
