# define	P(s) s

/* HYPRE_DistributedMatrixPilutSolver.c */
int HYPRE_NewDistributedMatrixPilutSolver P((MPI_Comm comm , HYPRE_DistributedMatrix matrix, HYPRE_DistributedMatrixPilutSolver *solver ));
int HYPRE_FreeDistributedMatrixPilutSolver P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverInitialize P((HYPRE_DistributedMatrixPilutSolver solver ));
int HYPRE_DistributedMatrixPilutSolverSetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix ));
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow P((HYPRE_DistributedMatrixPilutSolver in_ptr , int FirstLocalRow ));
int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize P((HYPRE_DistributedMatrixPilutSolver in_ptr , int size ));
int HYPRE_DistributedMatrixPilutSolverSetDropTolerance P((HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance ));
int HYPRE_DistributedMatrixPilutSolverSetMaxIts P((HYPRE_DistributedMatrixPilutSolver in_ptr , int its ));
int HYPRE_DistributedMatrixPilutSolverSetup P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSolve P((HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , double *b ));

/* comm.c */
int hypre_GlobalSEMax P((int value , MPI_Comm MPI_Context ));
int hypre_GlobalSEMin P((int value , MPI_Comm MPI_Context ));
int hypre_GlobalSESum P((int value , MPI_Comm MPI_Context ));
double hypre_GlobalSEMaxDouble P((double value , MPI_Comm MPI_Context ));
double hypre_GlobalSEMinDouble P((double value , MPI_Comm MPI_Context ));
double hypre_GlobalSESumDouble P((double value , MPI_Comm MPI_Context ));

/* debug.c */
void hypre_PrintLine P((char *str , hypre_PilutSolverGlobals *globals ));
void hypre_CheckBounds P((int low , int i , int up , hypre_PilutSolverGlobals *globals ));
long hypre_IDX_Checksum P((const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long hypre_INT_Checksum P((const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long hypre_FP_Checksum P((const double *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long hypre_RMat_Checksum P((const ReduceMatType *rmat , hypre_PilutSolverGlobals *globals ));
long hypre_LDU_Checksum P((const FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void hypre_PrintVector P((int *v , int n , char *msg , hypre_PilutSolverGlobals *globals ));

/* hypre.c */

/* ilut.c */
int hypre_ILUT P((DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , int maxnz , double tol , hypre_PilutSolverGlobals *globals ));
void hypre_ComputeAdd2Nrms P((int num_rows , int *rowptr , double *values , double *nrm2s ));

/* parilut.c */
void hypre_ParILUT P((DataDistType *ddist , FactorMatType *ldu , ReduceMatType *rmat , int gmaxnz , double tol , hypre_PilutSolverGlobals *globals ));
void hypre_ComputeCommInfo P((ReduceMatType *rmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals ));
int hypre_Idx2PE P((int idx , hypre_PilutSolverGlobals *globals ));
int hypre_SelectSet P((ReduceMatType *rmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , hypre_PilutSolverGlobals *globals ));
void hypre_SendFactoredRows P((FactorMatType *ldu , CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals ));
void hypre_ComputeRmat P((FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals ));
void hypre_FactorLocal P((FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals ));
void hypre_SecondDropSmall P((double rtol , hypre_PilutSolverGlobals *globals ));
int hypre_SeperateLU_byDIAG P((int diag , int *newiperm , hypre_PilutSolverGlobals *globals ));
int hypre_SeperateLU_byMIS P((hypre_PilutSolverGlobals *globals ));
void hypre_UpdateL P((int lrow , int last , FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void hypre_FormNRmat P((int rrow , int first , ReduceMatType *nrmat , int max_rowlen, int in_rowlen , int *rcolind , double *rvalues , hypre_PilutSolverGlobals *globals ));
void hypre_FormDU P((int lrow , int first , FactorMatType *ldu , int *rcolind , double *rvalues , double tol , hypre_PilutSolverGlobals *globals ));
void hypre_EraseMap P((CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals ));
void hypre_ParINIT P((ReduceMatType *nrmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals ));

/* parutil.c */
void hypre_errexit (char *f_str , ...);
void hypre_my_abort P((int inSignal , hypre_PilutSolverGlobals *globals ));
int *hypre_idx_malloc P((int n , char *msg ));
int *hypre_idx_malloc_init P((int n , int ival , char *msg ));
double *hypre_fp_malloc P((int n , char *msg ));
double *hypre_fp_malloc_init P((int n , double ival , char *msg ));
void *hypre_mymalloc P((int nbytes , char *msg ));
void hypre_free_multi (void *ptr1 , ...);
void hypre_memcpy_int P((int *dest , const int *src , size_t n ));
void hypre_memcpy_idx P((int *dest , const int *src , size_t n ));
void hypre_memcpy_fp P((double *dest , const double *src , size_t n ));

/* pblas1.c */
double hypre_p_dnrm2 P((DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals ));
double hypre_p_ddot P((DataDistType *ddist , double *x , double *y , hypre_PilutSolverGlobals *globals ));
void hypre_p_daxy P((DataDistType *ddist , double alpha , double *x , double *y ));
void hypre_p_daxpy P((DataDistType *ddist , double alpha , double *x , double *y ));
void hypre_p_daxbyz P((DataDistType *ddist , double alpha , double *x , double beta , double *y , double *z ));
int hypre_p_vprintf P((DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals ));

/* qsort.c */

/* qsort_si.c */
void hypre_sincsort_fast P((int n , int *base ));
void hypre_sdecsort_fast P((int n , int *base ));

/* serilut.c */
int hypre_SerILUT P((DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , ReduceMatType *rmat , int maxnz , double tol , hypre_PilutSolverGlobals *globals ));
int hypre_SelectInterior( int local_num_rows, 
                    HYPRE_DistributedMatrix matrix, 
                    int *external_rows,
		    int *newperm, int *newiperm, 
                    hypre_PilutSolverGlobals *globals );
void hypre_SecondDrop P((int maxnz , double tol , int row , int *perm , int *iperm , FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void hypre_SecondDropUpdate P((int maxnz , int maxnzkeep , double tol , int row , int nlocal , int *perm , int *iperm , FactorMatType *ldu , ReduceMatType *rmat , hypre_PilutSolverGlobals *globals ));
void hypre_SetUpFactor P((DataDistType *ddist, FactorMatType *ldu, int maxnz, int *petotal, int *rind, int *imap, int *maxsendP, int DoingL, hypre_PilutSolverGlobals *globals ));

/* trifactor.c */
void hypre_LDUSolve P((DataDistType *ddist , FactorMatType *ldu , double *x , double *b , hypre_PilutSolverGlobals *globals ));
int hypre_SetUpLUFactor P((DataDistType *ddist , FactorMatType *ldu , int maxnz , hypre_PilutSolverGlobals *globals ));
/* util.c */
int hypre_ExtractMinLR P((hypre_PilutSolverGlobals *globals ));
void hypre_IdxIncSort P((int n , int *idx , double *val ));
void hypre_ValDecSort P((int n , int *idx , double *val ));
int hypre_CompactIdx P((int n , int *idx , double *val ));
void hypre_PrintIdxVal P((int n , int *idx , double *val ));
int hypre_DecKeyValueCmp P((const void *v1 , const void *v2 ));
void hypre_SortKeyValueNodesDec P((KeyValueType *nodes , int n ));
int hypre_sasum P((int n , int *x ));
void hypre_sincsort P((int n , int *a ));
void sdecsort P((int n , int *a ));

#undef P
