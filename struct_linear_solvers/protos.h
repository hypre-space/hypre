#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* ZZZ_struct_smg.c */
ZZZ_StructSolver ZZZ_StructSMGInitialize P((MPI_Comm *comm ));
int ZZZ_StructSMGFinalize P((ZZZ_StructSolver solver ));
int ZZZ_StructSMGSetup P((ZZZ_StructSolver solver , ZZZ_StructMatrix A , ZZZ_StructVector b , ZZZ_StructVector x ));
int ZZZ_StructSMGSolve P((ZZZ_StructSolver solver , ZZZ_StructMatrix A , ZZZ_StructVector b , ZZZ_StructVector x ));
int ZZZ_SMGSetMemoryUse P((ZZZ_StructSolver solver , int memory_use ));
int ZZZ_SMGSetTol P((ZZZ_StructSolver solver , double tol ));
int ZZZ_SMGSetMaxIter P((ZZZ_StructSolver solver , int max_iter ));
int ZZZ_SMGSetZeroGuess P((ZZZ_StructSolver solver ));
int ZZZ_SMGGetNumIterations P((ZZZ_StructSolver solver , int *num_iterations ));
int ZZZ_SMGGetFinalRelativeResidualNorm P((ZZZ_StructSolver solver , double *relative_residual_norm ));

/* cyclic_reduction.c */
void *zzz_CyclicReductionInitialize P((MPI_Comm *comm ));
zzz_StructMatrix *zzz_CycRedNewCoarseOp P((zzz_StructMatrix *A , zzz_StructGrid *coarse_grid , int cdir ));
int zzz_CycRedSetupCoarseOp P((zzz_StructMatrix *A , zzz_StructMatrix *Ac , zzz_Index *cindex , zzz_Index *cstride ));
int zzz_CyclicReductionSetup P((void *cyc_red_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_CyclicReduction P((void *cyc_red_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_CyclicReductionSetBase P((void *cyc_red_vdata , zzz_Index *base_index , zzz_Index *base_stride ));
int zzz_CyclicReductionFinalize P((void *cyc_red_vdata ));

/* driver_internal.c */
int main P((int argc , char *argv []));

/* general.c */
int zzz_Log2 P((int p ));

/* linear_interface.c */

/* smg.c */
void *zzz_SMGInitialize P((MPI_Comm *comm ));
int zzz_SMGFinalize P((void *smg_vdata ));
int zzz_SMGSetMemoryUse P((void *smg_vdata , int memory_use ));
int zzz_SMGSetTol P((void *smg_vdata , double tol ));
int zzz_SMGSetMaxIter P((void *smg_vdata , int max_iter ));
int zzz_SMGSetZeroGuess P((void *smg_vdata ));
int zzz_SMGSetNumPreRelax P((void *smg_vdata , int num_pre_relax ));
int zzz_SMGSetNumPostRelax P((void *smg_vdata , int num_post_relax ));
int zzz_SMGSetBase P((void *smg_vdata , zzz_Index *base_index , zzz_Index *base_stride ));
int zzz_SMGSetLogging P((void *smg_vdata , int logging ));
int zzz_SMGGetNumIterations P((void *smg_vdata , int *num_iterations ));
int zzz_SMGPrintLogging P((void *smg_vdata , int myid ));
int zzz_SMGGetFinalRelativeResidualNorm P((void *smg_vdata , double *relative_residual_norm ));

/* smg2_setup_rap.c */
zzz_StructMatrix *zzz_SMG2NewRAPOp P((zzz_StructMatrix *R , zzz_StructMatrix *A , zzz_StructMatrix *PT ));
int zzz_SMG2BuildRAPSym P((zzz_StructMatrix *A , zzz_StructMatrix *PT , zzz_StructMatrix *R , zzz_StructMatrix *RAP , zzz_Index *cindex , zzz_Index *cstride ));
int zzz_SMG2BuildRAPNoSym P((zzz_StructMatrix *A , zzz_StructMatrix *PT , zzz_StructMatrix *R , zzz_StructMatrix *RAP , zzz_Index *cindex , zzz_Index *cstride ));

/* smg3_setup_rap.c */
zzz_StructMatrix *zzz_SMG3NewRAPOp P((zzz_StructMatrix *R , zzz_StructMatrix *A , zzz_StructMatrix *PT ));
int zzz_SMG3BuildRAPSym P((zzz_StructMatrix *A , zzz_StructMatrix *PT , zzz_StructMatrix *R , zzz_StructMatrix *RAP , zzz_Index *cindex , zzz_Index *cstride ));
int zzz_SMG3BuildRAPNoSym P((zzz_StructMatrix *A , zzz_StructMatrix *PT , zzz_StructMatrix *R , zzz_StructMatrix *RAP , zzz_Index *cindex , zzz_Index *cstride ));

/* smg_intadd.c */
void *zzz_SMGIntAddInitialize P((void ));
int zzz_SMGIntAddSetup P((void *intadd_vdata , zzz_StructMatrix *PT , zzz_StructVector *xc , zzz_StructVector *e , zzz_StructVector *x , zzz_Index *cindex , zzz_Index *cstride , zzz_Index *findex , zzz_Index *fstride ));
int zzz_SMGIntAdd P((void *intadd_vdata , zzz_StructMatrix *PT , zzz_StructVector *xc , zzz_StructVector *e , zzz_StructVector *x ));
int zzz_SMGIntAddFinalize P((void *intadd_vdata ));
void zzz_AppendSBoxArrayArrayAndRanks P((int **box_ranks_0 , int **box_ranks_1 , zzz_SBoxArrayArray *sbox_array_array_0 , zzz_SBoxArrayArray *sbox_array_array_1 , int ***box_ranks_ptr ));

/* smg_relax.c */
void *zzz_SMGRelaxInitialize P((MPI_Comm *comm ));
int zzz_SMGRelaxFreeTempVec P((void *relax_vdata ));
int zzz_SMGRelaxFreeARem P((void *relax_vdata ));
int zzz_SMGRelaxFreeASol P((void *relax_vdata ));
int zzz_SMGRelaxFinalize P((void *relax_vdata ));
int zzz_SMGRelax P((void *relax_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_SMGRelaxSetup P((void *relax_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_SMGRelaxSetupTempVec P((void *relax_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_SMGRelaxSetupARem P((void *relax_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_SMGRelaxSetupASol P((void *relax_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));
int zzz_SMGRelaxSetTempVec P((void *relax_vdata , zzz_StructVector *temp_vec ));
int zzz_SMGRelaxSetMemoryUse P((void *relax_vdata , int memory_use ));
int zzz_SMGRelaxSetTol P((void *relax_vdata , double tol ));
int zzz_SMGRelaxSetMaxIter P((void *relax_vdata , int max_iter ));
int zzz_SMGRelaxSetNonZeroGuess P((void *relax_vdata ));
int zzz_SMGRelaxSetZeroGuess P((void *relax_vdata ));
int zzz_SMGRelaxSetNumSpaces P((void *relax_vdata , int num_spaces ));
int zzz_SMGRelaxSetNumPreSpaces P((void *relax_vdata , int num_pre_spaces ));
int zzz_SMGRelaxSetNumRegSpaces P((void *relax_vdata , int num_reg_spaces ));
int zzz_SMGRelaxSetSpace P((void *relax_vdata , int i , int space_index , int space_stride ));
int zzz_SMGRelaxSetRegSpaceRank P((void *relax_vdata , int i , int reg_space_rank ));
int zzz_SMGRelaxSetPreSpaceRank P((void *relax_vdata , int i , int pre_space_rank ));
int zzz_SMGRelaxSetBase P((void *relax_vdata , zzz_Index *base_index , zzz_Index *base_stride ));
int zzz_SMGRelaxSetNewMatrixStencil P((void *relax_vdata , zzz_StructStencil *diff_stencil ));

/* smg_residual.c */
void *zzz_SMGResidualInitialize P((void ));
int zzz_SMGResidualSetup P((void *residual_vdata , zzz_StructMatrix *A , zzz_StructVector *x , zzz_StructVector *b , zzz_StructVector *r ));
int zzz_SMGResidual P((void *residual_vdata , zzz_StructMatrix *A , zzz_StructVector *x , zzz_StructVector *b , zzz_StructVector *r ));
int zzz_SMGResidualSetBase P((void *residual_vdata , zzz_Index *base_index , zzz_Index *base_stride ));
int zzz_SMGResidualFinalize P((void *residual_vdata ));

/* smg_restrict.c */
void *zzz_SMGRestrictInitialize P((void ));
int zzz_SMGRestrictSetup P((void *restrict_vdata , zzz_StructMatrix *R , zzz_StructVector *r , zzz_StructVector *rc , zzz_Index *cindex , zzz_Index *cstride , zzz_Index *findex , zzz_Index *fstride ));
int zzz_SMGRestrict P((void *restrict_vdata , zzz_StructMatrix *R , zzz_StructVector *r , zzz_StructVector *rc ));
int zzz_SMGRestrictFinalize P((void *restrict_vdata ));

/* smg_setup.c */
int zzz_SMGSetup P((void *smg_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));

/* smg_setup_interp.c */
zzz_StructMatrix *zzz_SMGNewInterpOp P((zzz_StructMatrix *A , zzz_StructGrid *cgrid , int cdir ));
int zzz_SMGSetupInterpOp P((void *relax_data , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x , zzz_StructMatrix *PT , int cdir , zzz_Index *cindex , zzz_Index *cstride , zzz_Index *findex , zzz_Index *fstride ));

/* smg_setup_rap.c */
zzz_StructMatrix *zzz_SMGNewRAPOp P((zzz_StructMatrix *R , zzz_StructMatrix *A , zzz_StructMatrix *PT ));
int zzz_SMGSetupRAPOp P((zzz_StructMatrix *R , zzz_StructMatrix *A , zzz_StructMatrix *PT , zzz_StructMatrix *Ac , zzz_Index *cindex , zzz_Index *cstride ));

/* smg_setup_restrict.c */
zzz_StructMatrix *zzz_SMGNewRestrictOp P((zzz_StructMatrix *A , zzz_StructGrid *cgrid , int cdir ));
int zzz_SMGSetupRestrictOp P((zzz_StructMatrix *A , zzz_StructMatrix *R , zzz_StructVector *temp_vec , int cdir , zzz_Index *cindex , zzz_Index *cstride ));

/* smg_solve.c */
int zzz_SMGSolve P((void *smg_vdata , zzz_StructMatrix *A , zzz_StructVector *b , zzz_StructVector *x ));

#undef P
