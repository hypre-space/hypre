/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* HYPRE_DistributedMatrixPilutSolver.c */
HYPRE_Int HYPRE_NewDistributedMatrixPilutSolver( MPI_Comm comm , HYPRE_DistributedMatrix matrix , HYPRE_DistributedMatrixPilutSolver *new_solver );
HYPRE_Int HYPRE_FreeDistributedMatrixPilutSolver( HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverInitialize( HYPRE_DistributedMatrixPilutSolver solver );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMatrix( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix );
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix( HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int FirstLocalRow );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int size );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetDropTolerance( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Real tolerance );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMaxIts( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int its );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetup( HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSolve( HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Real *x , HYPRE_Real *b );

/* comm.c */
HYPRE_Int hypre_GlobalSEMax( HYPRE_Int value , MPI_Comm hypre_MPI_Context );
HYPRE_Int hypre_GlobalSEMin( HYPRE_Int value , MPI_Comm hypre_MPI_Context );
HYPRE_Int hypre_GlobalSESum( HYPRE_Int value , MPI_Comm hypre_MPI_Context );
HYPRE_Real hypre_GlobalSEMaxDouble( HYPRE_Real value , MPI_Comm hypre_MPI_Context );
HYPRE_Real hypre_GlobalSEMinDouble( HYPRE_Real value , MPI_Comm hypre_MPI_Context );
HYPRE_Real hypre_GlobalSESumDouble( HYPRE_Real value , MPI_Comm hypre_MPI_Context );

/* debug.c */
void hypre_PrintLine( const char *str , hypre_PilutSolverGlobals *globals );
void hypre_CheckBounds( HYPRE_Int low , HYPRE_Int i , HYPRE_Int up , hypre_PilutSolverGlobals *globals );
hypre_longint hypre_IDX_Checksum( const HYPRE_Int *v , HYPRE_Int len , const char *msg , HYPRE_Int tag , hypre_PilutSolverGlobals *globals );
hypre_longint hypre_INT_Checksum( const HYPRE_Int *v , HYPRE_Int len , const char *msg , HYPRE_Int tag , hypre_PilutSolverGlobals *globals );
hypre_longint hypre_FP_Checksum( const HYPRE_Real *v , HYPRE_Int len , const char *msg , HYPRE_Int tag , hypre_PilutSolverGlobals *globals );
hypre_longint hypre_RMat_Checksum( const ReduceMatType *rmat , hypre_PilutSolverGlobals *globals );
hypre_longint hypre_LDU_Checksum( const FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_PrintVector( HYPRE_Int *v , HYPRE_Int n , char *msg , hypre_PilutSolverGlobals *globals );

/* hypre.c */

/* ilut.c */
HYPRE_Int hypre_ILUT( DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , HYPRE_Int maxnz , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
void hypre_ComputeAdd2Nrms( HYPRE_Int num_rows , HYPRE_Int *rowptr , HYPRE_Real *values , HYPRE_Real *nrm2s );

/* parilut.c */
void hypre_ParILUT( DataDistType *ddist , FactorMatType *ldu , ReduceMatType *rmat , HYPRE_Int gmaxnz , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
void hypre_ComputeCommInfo( ReduceMatType *rmat , CommInfoType *cinfo , HYPRE_Int *rowdist , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_Idx2PE( HYPRE_Int idx , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_SelectSet( ReduceMatType *rmat , CommInfoType *cinfo , HYPRE_Int *perm , HYPRE_Int *iperm , HYPRE_Int *newperm , HYPRE_Int *newiperm , hypre_PilutSolverGlobals *globals );
void hypre_SendFactoredRows( FactorMatType *ldu , CommInfoType *cinfo , HYPRE_Int *newperm , HYPRE_Int nmis , hypre_PilutSolverGlobals *globals );
void hypre_ComputeRmat( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , HYPRE_Int *perm , HYPRE_Int *iperm , HYPRE_Int *newperm , HYPRE_Int *newiperm , HYPRE_Int nmis , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
void hypre_FactorLocal( FactorMatType *ldu , ReduceMatType *rmat , ReduceMatType *nrmat , CommInfoType *cinfo , HYPRE_Int *perm , HYPRE_Int *iperm , HYPRE_Int *newperm , HYPRE_Int *newiperm , HYPRE_Int nmis , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
void hypre_SecondDropSmall( HYPRE_Real rtol , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_SeperateLU_byDIAG( HYPRE_Int diag , HYPRE_Int *newiperm , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_SeperateLU_byMIS( hypre_PilutSolverGlobals *globals );
void hypre_UpdateL( HYPRE_Int lrow , HYPRE_Int last , FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_FormNRmat( HYPRE_Int rrow , HYPRE_Int first , ReduceMatType *nrmat , HYPRE_Int max_rowlen , HYPRE_Int in_rowlen , HYPRE_Int *in_colind , HYPRE_Real *in_values , hypre_PilutSolverGlobals *globals );
void hypre_FormDU( HYPRE_Int lrow , HYPRE_Int first , FactorMatType *ldu , HYPRE_Int *rcolind , HYPRE_Real *rvalues , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
void hypre_EraseMap( CommInfoType *cinfo , HYPRE_Int *newperm , HYPRE_Int nmis , hypre_PilutSolverGlobals *globals );
void hypre_ParINIT( ReduceMatType *nrmat , CommInfoType *cinfo , HYPRE_Int *rowdist , hypre_PilutSolverGlobals *globals );

/* parutil.c */
void hypre_errexit(const char *f_str , ...);
void hypre_my_abort( HYPRE_Int inSignal , hypre_PilutSolverGlobals *globals );
HYPRE_Int *hypre_idx_malloc( HYPRE_Int n ,const char *msg );
HYPRE_Int *hypre_idx_malloc_init( HYPRE_Int n , HYPRE_Int ival ,const char *msg );
HYPRE_Real *hypre_fp_malloc( HYPRE_Int n ,const char *msg );
HYPRE_Real *hypre_fp_malloc_init( HYPRE_Int n , HYPRE_Real ival ,const char *msg );
void *hypre_mymalloc( HYPRE_Int nbytes ,const char *msg );
void hypre_free_multi( void *ptr1 , ...);
void hypre_memcpy_int( HYPRE_Int *dest , const HYPRE_Int *src , size_t n );
void hypre_memcpy_idx( HYPRE_Int *dest , const HYPRE_Int *src , size_t n );
void hypre_memcpy_fp( HYPRE_Real *dest , const HYPRE_Real *src , size_t n );

/* pblas1.c */
HYPRE_Real hypre_p_dnrm2( DataDistType *ddist , HYPRE_Real *x , hypre_PilutSolverGlobals *globals );
HYPRE_Real hypre_p_ddot( DataDistType *ddist , HYPRE_Real *x , HYPRE_Real *y , hypre_PilutSolverGlobals *globals );
void hypre_p_daxy( DataDistType *ddist , HYPRE_Real alpha , HYPRE_Real *x , HYPRE_Real *y );
void hypre_p_daxpy( DataDistType *ddist , HYPRE_Real alpha , HYPRE_Real *x , HYPRE_Real *y );
void hypre_p_daxbyz( DataDistType *ddist , HYPRE_Real alpha , HYPRE_Real *x , HYPRE_Real beta , HYPRE_Real *y , HYPRE_Real *z );
HYPRE_Int hypre_p_vprintf( DataDistType *ddist , HYPRE_Real *x , hypre_PilutSolverGlobals *globals );

/* distributed_qsort.c */
void hypre_tex_qsort(char *base, HYPRE_Int n, HYPRE_Int size, HYPRE_Int (*compar) (char*,char*));
/* distributed_qsort_si.c */
void hypre_sincsort_fast( HYPRE_Int n , HYPRE_Int *base );
void hypre_sdecsort_fast( HYPRE_Int n , HYPRE_Int *base );

/* serilut.c */
HYPRE_Int hypre_SerILUT( DataDistType *ddist , HYPRE_DistributedMatrix matrix , FactorMatType *ldu , ReduceMatType *rmat , HYPRE_Int maxnz , HYPRE_Real tol , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_SelectInterior( HYPRE_Int local_num_rows , HYPRE_DistributedMatrix matrix , HYPRE_Int *external_rows , HYPRE_Int *newperm , HYPRE_Int *newiperm , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_FindStructuralUnion( HYPRE_DistributedMatrix matrix , HYPRE_Int **structural_union , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_ExchangeStructuralUnions( DataDistType *ddist , HYPRE_Int **structural_union , hypre_PilutSolverGlobals *globals );
void hypre_SecondDrop( HYPRE_Int maxnz , HYPRE_Real tol , HYPRE_Int row , HYPRE_Int *perm , HYPRE_Int *iperm , FactorMatType *ldu , hypre_PilutSolverGlobals *globals );
void hypre_SecondDropUpdate( HYPRE_Int maxnz , HYPRE_Int maxnzkeep , HYPRE_Real tol , HYPRE_Int row , HYPRE_Int nlocal , HYPRE_Int *perm , HYPRE_Int *iperm , FactorMatType *ldu , ReduceMatType *rmat , hypre_PilutSolverGlobals *globals );

/* trifactor.c */
void hypre_LDUSolve( DataDistType *ddist , FactorMatType *ldu , HYPRE_Real *x , HYPRE_Real *b , hypre_PilutSolverGlobals *globals );
HYPRE_Int hypre_SetUpLUFactor( DataDistType *ddist , FactorMatType *ldu , HYPRE_Int maxnz , hypre_PilutSolverGlobals *globals );
void hypre_SetUpFactor( DataDistType *ddist , FactorMatType *ldu , HYPRE_Int maxnz , HYPRE_Int *petotal , HYPRE_Int *rind , HYPRE_Int *imap , HYPRE_Int *maxsendP , HYPRE_Int DoingL , hypre_PilutSolverGlobals *globals );

/* util.c */
HYPRE_Int hypre_ExtractMinLR( hypre_PilutSolverGlobals *globals );
void hypre_IdxIncSort( HYPRE_Int n , HYPRE_Int *idx , HYPRE_Real *val );
void hypre_ValDecSort( HYPRE_Int n , HYPRE_Int *idx , HYPRE_Real *val );
HYPRE_Int hypre_CompactIdx( HYPRE_Int n , HYPRE_Int *idx , HYPRE_Real *val );
void hypre_PrintIdxVal( HYPRE_Int n , HYPRE_Int *idx , HYPRE_Real *val );
HYPRE_Int hypre_DecKeyValueCmp( const void *v1 , const void *v2 );
void hypre_SortKeyValueNodesDec( KeyValueType *nodes , HYPRE_Int n );
HYPRE_Int hypre_sasum( HYPRE_Int n , HYPRE_Int *x );
void hypre_sincsort( HYPRE_Int n , HYPRE_Int *a );
void hypre_sdecsort( HYPRE_Int n , HYPRE_Int *a );

