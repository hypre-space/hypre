#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_DistributedMatrixPilutSolver.c */
HYPRE_DistributedMatrixPilutSolver HYPRE_NewDistributedMatrixPilutSolver P((MPI_Comm comm , HYPRE_DistributedMatrix matrix ));
int HYPRE_FreeDistributedMatrixPilutSolver P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverInitialize P((HYPRE_DistributedMatrixPilutSolver solver ));
int HYPRE_DistributedMatrixPilutSolverSetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix ));
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow P((HYPRE_DistributedMatrixPilutSolver in_ptr , int FirstLocalRow ));
int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize P((HYPRE_DistributedMatrixPilutSolver in_ptr , int size ));
int HYPRE_DistributedMatrixPilutSolverSetDropTolerance P((HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance ));
int HYPRE_DistributedMatrixPilutSolverSetMaxIts P((HYPRE_DistributedMatrixPilutSolver in_ptr , int its ));
int HYPRE_DistributedMatrixPilutSolverSetup P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSolve P((HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , const double *b ));

/* comm.c */
int GlobalSEMax P((int value , MPI_Comm MPI_Context ));
int GlobalSEMin P((int value , MPI_Comm MPI_Context ));
int GlobalSESum P((int value , MPI_Comm MPI_Context ));
double GlobalSEMaxDouble P((double value , MPI_Comm MPI_Context ));
double GlobalSEMinDouble P((double value , MPI_Comm MPI_Context ));
double GlobalSESumDouble P((double value , MPI_Comm MPI_Context ));

/* debug.c */
void PrintLine P((char *str , hypre_PilutSolverGlobals *globals ));
void CheckBounds P((int low , int i , int up , hypre_PilutSolverGlobals *globals ));
long IDX_Checksum P((const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long INT_Checksum P((const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long FP_Checksum P((const double *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals ));
long RMat_Checksum P((const ReduceMatType *rmat , hypre_PilutSolverGlobals *globals ));
long LDU_Checksum P((const FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void PrintVector P((int *v , int n , char *msg , hypre_PilutSolverGlobals *globals ));

/* hypre.c */

/* ilut.c */
int ILUT P((DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , int maxnz , double tol , hypre_PilutSolverGlobals *globals ));
void ComputeAdd2Nrms P((int num_rows , int *rowptr , double *values , double *nrm2s ));

/* parilut.c */
void ParILUT P((DataDistType *ddist , FactorMatType *ldu , ReduceMatType *rmat , int gmaxnz , double tol , hypre_PilutSolverGlobals *globals ));
void ComputeCommInfo P((ReduceMatType *rmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals ));
int Idx2PE P((int idx , hypre_PilutSolverGlobals *globals ));
int SelectSet P((ReduceMatType *rmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , hypre_PilutSolverGlobals *globals ));
void SendFactoredRows P((FactorMatType *ldu , CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals ));
void ComputeRmat P((FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals ));
void FactorLocal P((FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals ));
void SecondDropSmall P((double rtol , hypre_PilutSolverGlobals *globals ));
int SeperateLU_byDIAG P((int diag , int *newiperm , hypre_PilutSolverGlobals *globals ));
int SeperateLU_byMIS P((hypre_PilutSolverGlobals *globals ));
void UpdateL P((int lrow , int last , FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void FormNRmat P((int rrow , int first , ReduceMatType *nrmat , int max_rowlen, int in_rowlen , int *rcolind , double *rvalues , hypre_PilutSolverGlobals *globals ));
void FormDU P((int lrow , int first , FactorMatType *ldu , int *rcolind , double *rvalues , double tol , hypre_PilutSolverGlobals *globals ));
void EraseMap P((CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals ));
void ParINIT P((ReduceMatType *nrmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals ));

/* parutil.c */
void errexit P((char *f_str , ...));
void my_abort P((int inSignal , hypre_PilutSolverGlobals *globals ));
int *idx_malloc P((int n , char *msg ));
int *idx_malloc_init P((int n , int ival , char *msg ));
double *fp_malloc P((int n , char *msg ));
double *fp_malloc_init P((int n , double ival , char *msg ));
void *mymalloc P((int nbytes , char *msg ));
void free_multi P((void *ptr1 , ...));
void memcpy_int P((int *dest , const int *src , size_t n ));
void memcpy_idx P((int *dest , const int *src , size_t n ));
void memcpy_fp P((double *dest , const double *src , size_t n ));

/* pblas1.c */
double p_dnrm2 P((DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals ));
double p_ddot P((DataDistType *ddist , double *x , double *y , hypre_PilutSolverGlobals *globals ));
void p_daxy P((DataDistType *ddist , double alpha , double *x , double *y ));
void p_daxpy P((DataDistType *ddist , double alpha , double *x , double *y ));
void p_daxbyz P((DataDistType *ddist , double alpha , double *x , double beta , double *y , double *z ));
int p_vprintf P((DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals ));

/* qsort.c */

/* qsort_si.c */
void sincsort_fast P((int n , int *base ));
void sdecsort_fast P((int n , int *base ));

/* serilut.c */
int SerILUT P((DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , ReduceMatType *rmat , int maxnz , double tol , hypre_PilutSolverGlobals *globals ));
int SelectInterior( int local_num_rows, 
                    HYPRE_DistributedMatrix matrix, 
                    int *external_rows,
		    int *newperm, int *newiperm, 
                    hypre_PilutSolverGlobals *globals );
void SecondDrop P((int maxnz , double tol , int row , int *perm , int *iperm , FactorMatType *ldu , hypre_PilutSolverGlobals *globals ));
void SecondDropUpdate P((int maxnz , int maxnzkeep , double tol , int row , int nlocal , int *perm , int *iperm , FactorMatType *ldu , ReduceMatType *rmat , hypre_PilutSolverGlobals *globals ));
void SetUpFactor P((DataDistType *ddist, FactorMatType *ldu, int maxnz, int *petotal, int *rind, int *imap, int *maxsendP, int DoingL, hypre_PilutSolverGlobals *globals ));

/* trifactor.c */
void LDUSolve P((DataDistType *ddist , FactorMatType *ldu , double *x , double *b , hypre_PilutSolverGlobals *globals ));
int SetUpLUFactor P((DataDistType *ddist , FactorMatType *ldu , int maxnz , hypre_PilutSolverGlobals *globals ));
/* util.c */
int ExtractMinLR P((hypre_PilutSolverGlobals *globals ));
void IdxIncSort P((int n , int *idx , double *val ));
void ValDecSort P((int n , int *idx , double *val ));
int CompactIdx P((int n , int *idx , double *val ));
void PrintIdxVal P((int n , int *idx , double *val ));
int DecKeyValueCmp P((const void *v1 , const void *v2 ));
void SortKeyValueNodesDec P((KeyValueType *nodes , int n ));
int sasum P((int n , int *x ));
void sincsort P((int n , int *a ));
void sdecsort P((int n , int *a ));

#undef P
