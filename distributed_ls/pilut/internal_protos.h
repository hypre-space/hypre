
/* HYPRE_DistributedMatrixPilutSolver.c */
int HYPRE_NewDistributedMatrixPilutSolver( MPI_Comm comm , HYPRE_DistributedMatrix matrix , HYPRE_DistributedMatrixPilutSolver *new_solver );
int HYPRE_FreeDistributedMatrixPilutSolver( HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverInitialize( HYPRE_DistributedMatrixPilutSolver solver );
int HYPRE_DistributedMatrixPilutSolverSetMatrix( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix );
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix( HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow( HYPRE_DistributedMatrixPilutSolver in_ptr , int FirstLocalRow );
int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize( HYPRE_DistributedMatrixPilutSolver in_ptr , int size );
int HYPRE_DistributedMatrixPilutSolverSetDropTolerance( HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance );
int HYPRE_DistributedMatrixPilutSolverSetMaxIts( HYPRE_DistributedMatrixPilutSolver in_ptr , int its );
int HYPRE_DistributedMatrixPilutSolverSetup( HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverSolve( HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , double *b );

/* comm.c */
int hypre_GlobalSEMax( int value , MPI_Comm MPI_Context );
int hypre_GlobalSEMin( int value , MPI_Comm MPI_Context );
int hypre_GlobalSESum( int value , MPI_Comm MPI_Context );
double hypre_GlobalSEMaxDouble( double value , MPI_Comm MPI_Context );
double hypre_GlobalSEMinDouble( double value , MPI_Comm MPI_Context );
double hypre_GlobalSESumDouble( double value , MPI_Comm MPI_Context );

/* debug.c */
void hypre_PrintLine( char *str , hypre_PilutSolverGlobals *globals );
void hypre_CheckBounds( int low , int i , int up , hypre_PilutSolverGlobals *globals );
long hypre_IDX_Checksum( const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals );
long hypre_INT_Checksum( const int *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals );
long hypre_FP_Checksum( const double *v , int len , const char *msg , int tag , hypre_PilutSolverGlobals *globals );
long hypre_RMat_Checksum( const ReduceMatType *rmat , hypre_PilutSolverGlobals *globals );
long hypre_LDU_Checksum( const FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_PrintVector( int *v , int n , char *msg , hypre_PilutSolverGlobals *globals );

/* hypre.c */

/* ilut.c */
int hypre_ILUT( DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , int maxnz , double tol , hypre_PilutSolverGlobals *globals );
void hypre_ComputeAdd2Nrms( int num_rows , int *rowptr , double *values , double *nrm2s );

/* parilut.c */
void hypre_ParILUT( DataDistType *ddist , FactorMatType *ldu , ReduceMatType *rmat , int gmaxnz , double tol , hypre_PilutSolverGlobals *globals );
void hypre_ComputeCommInfo( ReduceMatType *rmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals );
int hypre_Idx2PE( int idx , hypre_PilutSolverGlobals *globals );
int hypre_SelectSet( ReduceMatType *rmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , hypre_PilutSolverGlobals *globals );
void hypre_SendFactoredRows( FactorMatType *ldu , CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals );
void hypre_ComputeRmat( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals );
void hypre_FactorLocal( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , int *perm , int *iperm , int *newperm , int *newiperm , int nmis , double tol , hypre_PilutSolverGlobals *globals );
void hypre_SecondDropSmall( double rtol , hypre_PilutSolverGlobals *globals );
int hypre_SeperateLU_byDIAG( int diag , int *newiperm , hypre_PilutSolverGlobals *globals );
int hypre_SeperateLU_byMIS( hypre_PilutSolverGlobals *globals );
void hypre_UpdateL( int lrow , int last , FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_FormNRmat( int rrow , int first , ReduceMatType *nrmat , int max_rowlen , int in_rowlen , int *in_colind , double *in_values , hypre_PilutSolverGlobals *globals );
void hypre_FormDU( int lrow , int first , FactorMatType *ldu , int *rcolind , double *rvalues , double tol , hypre_PilutSolverGlobals *globals );
void hypre_EraseMap( CommInfoType *cinfo , int *newperm , int nmis , hypre_PilutSolverGlobals *globals );
void hypre_ParINIT( ReduceMatType *nrmat , CommInfoType *cinfo , int *rowdist , hypre_PilutSolverGlobals *globals );

/* parutil.c */
void hypre_errexit( char *f_str , ...);
void hypre_my_abort( int inSignal , hypre_PilutSolverGlobals *globals );
int *hypre_idx_malloc( int n , char *msg );
int *hypre_idx_malloc_init( int n , int ival , char *msg );
double *hypre_fp_malloc( int n , char *msg );
double *hypre_fp_malloc_init( int n , double ival , char *msg );
void *hypre_mymalloc( int nbytes , char *msg );
void hypre_free_multi( void *ptr1 , ...);
void hypre_memcpy_int( int *dest , const int *src , size_t n );
void hypre_memcpy_idx( int *dest , const int *src , size_t n );
void hypre_memcpy_fp( double *dest , const double *src , size_t n );

/* pblas1.c */
double hypre_p_dnrm2( DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals );
double hypre_p_ddot( DataDistType *ddist , double *x , double *y , hypre_PilutSolverGlobals *globals );
void hypre_p_daxy( DataDistType *ddist , double alpha , double *x , double *y );
void hypre_p_daxpy( DataDistType *ddist , double alpha , double *x , double *y );
void hypre_p_daxbyz( DataDistType *ddist , double alpha , double *x , double beta , double *y , double *z );
int hypre_p_vprintf( DataDistType *ddist , double *x , hypre_PilutSolverGlobals *globals );

/* qsort.c */

/* qsort_si.c */
void hypre_sincsort_fast( int n , int *base );
void hypre_sdecsort_fast( int n , int *base );

/* serilut.c */
int hypre_SerILUT( DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , ReduceMatType *rmat , int maxnz , double tol , hypre_PilutSolverGlobals *globals );
int hypre_SelectInterior( int local_num_rows , HYPRE_DistributedMatrix matrix , int *external_rows , int *newperm , int *newiperm , hypre_PilutSolverGlobals *globals );
int FindStructuralUnion( HYPRE_DistributedMatrix matrix , int **structural_union , hypre_PilutSolverGlobals *globals );
int ExchangeStructuralUnions( DataDistType *ddist , int **structural_union , hypre_PilutSolverGlobals *globals );
void hypre_SecondDrop( int maxnz , double tol , int row , int *perm , int *iperm , FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_SecondDropUpdate( int maxnz , int maxnzkeep , double tol , int row , int nlocal , int *perm , int *iperm , FactorMatType *ldu , ReduceMatType *rmat , hypre_PilutSolverGlobals *globals );

/* trifactor.c */
void hypre_LDUSolve( DataDistType *ddist , FactorMatType *ldu , double *x , double *b , hypre_PilutSolverGlobals *globals );
int hypre_SetUpLUFactor( DataDistType *ddist , FactorMatType *ldu , int maxnz , hypre_PilutSolverGlobals *globals );
void hypre_SetUpFactor( DataDistType *ddist , FactorMatType *ldu , int maxnz , int *petotal , int *rind , int *imap , int *maxsendP , int DoingL , hypre_PilutSolverGlobals *globals );

/* util.c */
int hypre_ExtractMinLR( hypre_PilutSolverGlobals *globals );
void hypre_IdxIncSort( int n , int *idx , double *val );
void hypre_ValDecSort( int n , int *idx , double *val );
int hypre_CompactIdx( int n , int *idx , double *val );
void hypre_PrintIdxVal( int n , int *idx , double *val );
int hypre_DecKeyValueCmp( const void *v1 , const void *v2 );
void hypre_SortKeyValueNodesDec( KeyValueType *nodes , int n );
int hypre_sasum( int n , int *x );
void hypre_sincsort( int n , int *a );
void sdecsort( int n , int *a );

