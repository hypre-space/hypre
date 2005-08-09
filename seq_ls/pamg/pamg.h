
#include <HYPRE_config.h>

#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************************
 *
 * This code implements a class for block compressed sparse row matrices.
 *
 *****************************************************************************/

#ifndef hypre_BCSR_MATRIX_HEADER
#define hypre_BCSR_MATRIX_HEADER

#define hypre_BCSR_MATRIX_USE_DENSE_BLOCKS
#include "bcsr_matrix_dense_block.h"

typedef struct {
  hypre_BCSRMatrixBlock** blocks;
  int* i;
  int* j;
  int num_block_rows;
  int num_block_cols;
  int num_nonzero_blocks;
  int num_rows_per_block;
  int num_cols_per_block;
} hypre_BCSRMatrix;

/*****************************************************************************
 *
 * Accessors
 *
 *****************************************************************************/

#define hypre_BCSRMatrixBlocks(A) ((A) -> blocks)
#define hypre_BCSRMatrixI(A) ((A) -> i)
#define hypre_BCSRMatrixJ(A) ((A) -> j)
#define hypre_BCSRMatrixNumBlockRows(A) ((A) -> num_block_rows)
#define hypre_BCSRMatrixNumBlockCols(A) ((A) -> num_block_cols)
#define hypre_BCSRMatrixNumNonzeroBlocks(A) ((A) -> num_nonzero_blocks)
#define hypre_BCSRMatrixNumRowsPerBlock(A) ((A) -> num_rows_per_block)
#define hypre_BCSRMatrixNumColsPerBlock(A) ((A) -> num_cols_per_block)


#if 0

/*****************************************************************************
 *
 * Prototypes
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixCreate(int num_block_rows, int num_block_cols,
		       int num_nonzero_blocks,
		       int num_rows_per_block, int num_cols_per_block);

int
hypre_BCSRMatrixDestroy(hypre_BCSRMatrix* A);

int
hypre_BCSRMatrixInitialise(hypre_BCSRMatrix* A);

int
hypre_BCSRMatrixPrint(hypre_BCSRMatrix* A, char* file_name);

int
hypre_BCSRMatrixTranspose(hypre_BCSRMatrix* A, hypre_BCSRMatrix** AT);

hypre_BCSRMatrix*
hypre_BCSRMatrixFromCSRMatrix(hypre_CSRMatrix* A,
			      int num_rows_per_block, int num_cols_per_block);

hypre_CSRMatrix*
hypre_BCSRMatrixToCSRMatrix(hypre_BCSRMatrix* B);

hypre_CSRMatrix*
hypre_BCSRMatrixCompress(hypre_BCSRMatrix* A);

/*****************************************************************************
 *
 * Auxiliary function prototypes
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixBuildInterp(hypre_BCSRMatrix* A, int* CF_marker,
			    hypre_CSRMatrix* S, int coarse_size);

hypre_BCSRMatrix*
hypre_BCSRMatrixBuildInterpD(hypre_BCSRMatrix* A, int* CF_marker,
			     hypre_CSRMatrix* S, int coarse_size);

int
hypre_BCSRMatrixBuildCoarseOperator(hypre_BCSRMatrix* RT,
				    hypre_BCSRMatrix* A,
				    hypre_BCSRMatrix* P,
				    hypre_BCSRMatrix** RAP_ptr);

#endif

#endif
/*****************************************************************************
 *
 * This code implements a class for a dense block of a compressed sparse row
 * matrix.
 *
 *****************************************************************************/

#ifndef hypre_BCSR_MATRIX_DENSE_BLOCK_HEADER
#define hypre_BCSR_MATRIX_DENSE_BLOCK_HEADER

typedef struct {
  double* data;
  int num_rows;
  int num_cols;
} hypre_BCSRMatrixDenseBlock;

/*****************************************************************************
 *
 * Prototypes
 *
 *****************************************************************************/

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCreate(int num_rows, int num_cols);

int
hypre_BCSRMatrixDenseBlockDestroy(hypre_BCSRMatrixDenseBlock* A);

int
hypre_BCSRMatrixDenseBlockInitialise(hypre_BCSRMatrixDenseBlock* A);

int
hypre_BCSRMatrixDenseBlockFillData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

int
hypre_BCSRMatrixDenseBlockGetData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCopy(hypre_BCSRMatrixDenseBlock* A);

int
hypre_BCSRMatrixDenseBlockAdd(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

int
hypre_BCSRMatrixDenseBlockMultiply(hypre_BCSRMatrixDenseBlock* A,
				   hypre_BCSRMatrixDenseBlock* B);

int
hypre_BCSRMatrixDenseBlockNeg(hypre_BCSRMatrixDenseBlock* A);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockDiag(hypre_BCSRMatrixDenseBlock* A);

int
hypre_BCSRMatrixDenseBlockMulInv(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

int
hypre_BCSRMatrixDenseBlockTranspose(hypre_BCSRMatrixDenseBlock* A);

int
hypre_BCSRMatrixBlockMatvec(double alpha, hypre_BCSRMatrixDenseBlock* A,
			    double* x_data, double beta, double* y_data);

int
hypre_BCSRMatrixBlockMatvecT(double alpha, hypre_BCSRMatrixDenseBlock* A,
			     double* x_data, double beta, double* y_data);

double
hypre_BCSRMatrixDenseBlockNorm(hypre_BCSRMatrixDenseBlock* A,
			       const char* norm);

int
hypre_BCSRMatrixDenseBlockPrint(hypre_BCSRMatrixDenseBlock* A,
				FILE* out_file);

#ifdef hypre_BCSR_MATRIX_USE_DENSE_BLOCKS

#define hypre_BCSRMatrixBlock hypre_BCSRMatrixDenseBlock
#define hypre_BCSRMatrixBlockCreate hypre_BCSRMatrixDenseBlockCreate
#define hypre_BCSRMatrixBlockDestroy hypre_BCSRMatrixDenseBlockDestroy
#define hypre_BCSRMatrixBlockInitialise hypre_BCSRMatrixDenseBlockInitialise
#define hypre_BCSRMatrixBlockFillData hypre_BCSRMatrixDenseBlockFillData
#define hypre_BCSRMatrixBlockGetData hypre_BCSRMatrixDenseBlockGetData
#define hypre_BCSRMatrixBlockCopy hypre_BCSRMatrixDenseBlockCopy
#define hypre_BCSRMatrixBlockAdd hypre_BCSRMatrixDenseBlockAdd
#define hypre_BCSRMatrixBlockMultiply hypre_BCSRMatrixDenseBlockMultiply
#define hypre_BCSRMatrixBlockNeg hypre_BCSRMatrixDenseBlockNeg
#define hypre_BCSRMatrixBlockDiag hypre_BCSRMatrixDenseBlockDiag
#define hypre_BCSRMatrixBlockMulInv hypre_BCSRMatrixDenseBlockMulInv
#define hypre_BCSRMatrixBlockTranspose hypre_BCSRMatrixDenseBlockTranspose
#define hypre_BCSRMatrixBlockMatvec hypre_BCSRMatrixDenseBlockMatvec
#define hypre_BCSRMatrixBlockMatvecT hypre_BCSRMatrixDenseBlockMatvecT
#define hypre_BCSRMatrixBlockNorm hypre_BCSRMatrixDenseBlockNorm
#define hypre_BCSRMatrixBlockPrint hypre_BCSRMatrixDenseBlockPrint

#endif

#endif
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_AMG_DATA_HEADER
#define hypre_AMG_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_AMGData
 *--------------------------------------------------------------------------*/

typedef struct
{

   /* setup params */
   int      max_levels;
   double   strong_threshold;
   double   A_trunc_factor;
   double   P_trunc_factor;
   int      A_max_elmts;
   int      P_max_elmts;
   int      coarsen_type;
   int      agg_coarsen_type;
   int      interp_type;
   int      agg_interp_type;
   int      agg_levels;
   int      num_relax_steps;  
   int      num_jacs;
   int use_block_flag;

   /* solve params */
   int      max_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   double  *relax_weight;
   double   tol;
   /* problem data */
   hypre_CSRMatrix  *A;
   int      num_variables;
   int      num_functions;
   int      num_points;
   int     *dof_func;
   int     *dof_point;
   int     *point_dof_map;           

   /* data generated in the setup phase */
   hypre_CSRMatrix **A_array;
   hypre_BCSRMatrix **B_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;
   hypre_CSRMatrix **P_array;
   hypre_BCSRMatrix **PB_array;
   int             **CF_marker_array;
   int             **dof_func_array;
   int             **dof_point_array;
   int             **point_dof_map_array;
   int               num_levels;
   int      	    *schwarz_option;
   int      	    *num_domains;
   int     	   **i_domain_dof;
   int     	   **j_domain_dof;
   double  	   **domain_matrixinverse;
   int		     mode;

   /* data generated in the solve phase */
   hypre_Vector   *Vtemp;
   double   *vtmp;
   int       cycle_op_count;                                                   

   /* output params */
   int      ioutdat;
   char     log_file_name[256];

} hypre_AMGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_AMGData structure
 *--------------------------------------------------------------------------*/

/* setup params */
		  		      
#define hypre_AMGDataMaxLevels(amg_data) ((amg_data)->max_levels)
#define hypre_AMGDataStrongThreshold(amg_data) ((amg_data)->strong_threshold)
#define hypre_AMGDataATruncFactor(amg_data) ((amg_data)->A_trunc_factor)
#define hypre_AMGDataPTruncFactor(amg_data) ((amg_data)->P_trunc_factor)
#define hypre_AMGDataAMaxElmts(amg_data) ((amg_data)->A_max_elmts)
#define hypre_AMGDataPMaxElmts(amg_data) ((amg_data)->P_max_elmts)
#define hypre_AMGDataCoarsenType(amg_data) ((amg_data)->coarsen_type)
#define hypre_AMGDataAggCoarsenType(amg_data) ((amg_data)->agg_coarsen_type)
#define hypre_AMGDataInterpType(amg_data) ((amg_data)->interp_type)
#define hypre_AMGDataAggInterpType(amg_data) ((amg_data)->agg_interp_type)
#define hypre_AMGDataAggLevels(amg_data) ((amg_data)->agg_levels)
#define hypre_AMGDataNumRelaxSteps(amg_data) ((amg_data)->num_relax_steps)
#define hypre_AMGDataNumJacs(amg_data) ((amg_data)->num_jacs)
#define hypre_AMGDataUseBlockFlag(amg_data) ((amg_data)->use_block_flag)
/* solve params */

#define hypre_AMGDataMaxIter(amg_data) ((amg_data)->max_iter)
#define hypre_AMGDataCycleType(amg_data) ((amg_data)->cycle_type)
#define hypre_AMGDataTol(amg_data) ((amg_data)->tol)
#define hypre_AMGDataNumGridSweeps(amg_data) ((amg_data)->num_grid_sweeps)
#define hypre_AMGDataGridRelaxType(amg_data) ((amg_data)->grid_relax_type)
#define hypre_AMGDataGridRelaxPoints(amg_data) ((amg_data)->grid_relax_points)
#define hypre_AMGDataRelaxWeight(amg_data) ((amg_data)->relax_weight)

/* problem data parameters */
#define  hypre_AMGDataNumVariables(amg_data)  ((amg_data)->num_variables)
#define hypre_AMGDataNumFunctions(amg_data) ((amg_data)->num_functions)
#define hypre_AMGDataNumPoints(amg_data) ((amg_data)->num_points)
#define hypre_AMGDataDofFunc(amg_data) ((amg_data)->dof_func)
#define hypre_AMGDataDofPoint(amg_data) ((amg_data)->dof_point)
#define hypre_AMGDataPointDofMap(amg_data) ((amg_data)->point_dof_map)

/* data generated by the setup phase */
#define hypre_AMGDataCFMarkerArray(amg_data) ((amg_data)-> CF_marker_array)
#define hypre_AMGDataAArray(amg_data) ((amg_data)->A_array)
#define hypre_AMGDataBArray(amg_data) ((amg_data)->B_array)
#define hypre_AMGDataFArray(amg_data) ((amg_data)->F_array)
#define hypre_AMGDataUArray(amg_data) ((amg_data)->U_array)
#define hypre_AMGDataPArray(amg_data) ((amg_data)->P_array)
#define hypre_AMGDataPArray(amg_data) ((amg_data)->P_array)
#define hypre_AMGDataPBArray(amg_data) ((amg_data)->PB_array)
#define hypre_AMGDataDofFuncArray(amg_data) ((amg_data)->dof_func_array) 
#define hypre_AMGDataDofPointArray(amg_data) ((amg_data)->dof_point_array) 
#define hypre_AMGDataPointDofMapArray(amg_data) ((amg_data)->point_dof_map_array)
#define hypre_AMGDataNumLevels(amg_data) ((amg_data)->num_levels)
#define hypre_AMGDataSchwarzOption(amg_data) ((amg_data)->schwarz_option)
#define hypre_AMGDataNumDomains(amg_data) ((amg_data)->num_domains)
#define hypre_AMGDataIDomainDof(amg_data) ((amg_data)->i_domain_dof)
#define hypre_AMGDataJDomainDof(amg_data) ((amg_data)->j_domain_dof)
#define hypre_AMGDataDomainMatrixInverse(amg_data) ((amg_data)->domain_matrixinverse)
#define hypre_AMGDataMode(amg_data) ((amg_data)->mode)

/* data generated in the solve phase */
#define hypre_AMGDataVtemp(amg_data) ((amg_data)->Vtemp)
#define hypre_AMGDataCycleOpCount(amg_data) ((amg_data)->cycle_op_count)

/* output parameters */
#define hypre_AMGDataIOutDat(amg_data) ((amg_data)->ioutdat)
#define hypre_AMGDataLogFileName(amg_data) ((amg_data)->log_file_name)

#endif



/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef hypre_GENERAL_HEADER
#define hypre_GENERAL_HEADER


/*--------------------------------------------------------------------------
 * Define various flags
 *--------------------------------------------------------------------------*/

#ifndef NULL
#define NULL 0
#endif


/*--------------------------------------------------------------------------
 * Define max and min functions
 *--------------------------------------------------------------------------*/

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef min
#define min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef round
#define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif

#endif

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header for PCG
 *
 *****************************************************************************/

#ifndef _PCG_HEADER
#define _PCG_HEADER


/*--------------------------------------------------------------------------
 * PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      two_norm;

   hypre_CSRMatrix  *A;
   hypre_Vector  *p;
   hypre_Vector  *s;
   hypre_Vector  *r;

   int    (*precond)();
   void    *precond_data;

   char    *log_file_name;

} PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the PCGData structure
 *--------------------------------------------------------------------------*/

#define PCGDataMaxIter(pcg_data)      ((pcg_data) -> max_iter)
#define PCGDataTwoNorm(pcg_data)      ((pcg_data) -> two_norm)

#define PCGDataA(pcg_data)            ((pcg_data) -> A)
#define PCGDataP(pcg_data)            ((pcg_data) -> p)
#define PCGDataS(pcg_data)            ((pcg_data) -> s)
#define PCGDataR(pcg_data)            ((pcg_data) -> r)

#define PCGDataPrecond(pcg_data)      ((pcg_data) -> precond)
#define PCGDataPrecondData(pcg_data)  ((pcg_data) -> precond_data)

#define PCGDataLogFileName(pcg_data)  ((pcg_data) -> log_file_name)


#endif

/* amg.c */
void *hypre_AMGInitialize( void );
int hypre_AMGFinalize( void *data );
int hypre_AMGSetMaxLevels( void *data , int max_levels );
int hypre_AMGSetStrongThreshold( void *data , double strong_threshold );
int hypre_AMGSetMode( void *data , int mode );
int hypre_AMGSetATruncFactor( void *data , double A_trunc_factor );
int hypre_AMGSetPTruncFactor( void *data , double P_trunc_factor );
int hypre_AMGSetAMaxElmts( void *data , int A_max_elmts );
int hypre_AMGSetPMaxElmts( void *data , int P_max_elmts );
int hypre_AMGSetCoarsenType( void *data , int coarsen_type );
int hypre_AMGSetAggCoarsenType( void *data , int agg_coarsen_type );
int hypre_AMGSetAggLevels( void *data , int agg_levels );
int hypre_AMGSetInterpType( void *data , int interp_type );
int hypre_AMGSetAggInterpType( void *data , int agg_interp_type );
int hypre_AMGSetNumJacs( void *data , int num_jacs );
int hypre_AMGSetMaxIter( void *data , int max_iter );
int hypre_AMGSetCycleType( void *data , int cycle_type );
int hypre_AMGSetTol( void *data , double tol );
int hypre_AMGSetNumRelaxSteps( void *data , int num_relax_steps );
int hypre_AMGSetNumGridSweeps( void *data , int *num_grid_sweeps );
int hypre_AMGSetGridRelaxType( void *data , int *grid_relax_type );
int hypre_AMGSetGridRelaxPoints( void *data , int **grid_relax_points );
int hypre_AMGSetRelaxWeight( void *data , double *relax_weight );
int hypre_AMGSetSchwarzOption( void *data , int *schwarz_option );
int hypre_AMGSetIOutDat( void *data , int ioutdat );
int hypre_AMGSetLogFileName( void *data , char *log_file_name );
int hypre_AMGSetLogging( void *data , int ioutdat , char *log_file_name );
int hypre_AMGSetUseBlockFlag( void *data , int use_block_flag );
int hypre_AMGSetNumFunctions( void *data , int num_functions );
int hypre_AMGSetNumPoints( void *data , int num_points );
int hypre_AMGSetDofFunc( void *data , int *dof_func );
int hypre_AMGSetDofPoint( void *data , int *dof_point );
int hypre_AMGSetPointDofMap( void *data , int *point_dof_map );

/* amg_setup.c */
int hypre_AMGSetup( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amg_solve.c */
int hypre_AMGSolve( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amgstats.c */
int hypre_AMGSetupStats( void *amg_vdata );
void hypre_WriteSolverParams( void *data );

/* Atrunc.c */
int hypre_AMGOpTruncation( hypre_CSRMatrix *A , double trunc_factor , int max_elmts );

/* bcsr_interp.c */
hypre_BCSRMatrix *hypre_BCSRMatrixBuildInterp( hypre_BCSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int coarse_size );
hypre_BCSRMatrix *hypre_BCSRMatrixBuildInterpD( hypre_BCSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int coarse_size );

/* bcsr_matrix.c */
hypre_BCSRMatrix *hypre_BCSRMatrixCreate( int num_block_rows , int num_block_cols , int num_nonzero_blocks , int num_rows_per_block , int num_cols_per_block );
int hypre_BCSRMatrixDestroy( hypre_BCSRMatrix *A );
int hypre_BCSRMatrixInitialise( hypre_BCSRMatrix *A );
int hypre_BCSRMatrixPrint( hypre_BCSRMatrix *A , char *file_name );
int hypre_BCSRMatrixTranspose( hypre_BCSRMatrix *A , hypre_BCSRMatrix **AT );
hypre_BCSRMatrix *hypre_BCSRMatrixFromCSRMatrix( hypre_CSRMatrix *A , int num_rows_per_block , int num_cols_per_block );
hypre_CSRMatrix *hypre_BCSRMatrixToCSRMatrix( hypre_BCSRMatrix *B );
hypre_CSRMatrix *hypre_BCSRMatrixCompress( hypre_BCSRMatrix *A );

/* bcsr_matrix_dense_block.c */
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockCreate( int num_rows , int num_cols );
int hypre_BCSRMatrixDenseBlockDestroy( hypre_BCSRMatrixDenseBlock *A );
int hypre_BCSRMatrixDenseBlockInitialise( hypre_BCSRMatrixDenseBlock *A );
int hypre_BCSRMatrixDenseBlockFillData( hypre_BCSRMatrixDenseBlock *A , double *data );
int hypre_BCSRMatrixDenseBlockGetData( hypre_BCSRMatrixDenseBlock *A , double *data );
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockCopy( hypre_BCSRMatrixDenseBlock *A );
int hypre_BCSRMatrixDenseBlockAdd( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
int hypre_BCSRMatrixDenseBlockMultiply( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
int hypre_BCSRMatrixDenseBlockNeg( hypre_BCSRMatrixDenseBlock *A );
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockDiag( hypre_BCSRMatrixDenseBlock *A );
int hypre_BCSRMatrixDenseBlockMulInv( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
int hypre_BCSRMatrixDenseBlockTranspose( hypre_BCSRMatrixDenseBlock *A );
int hypre_BCSRMatrixDenseBlockMatvec( double alpha , hypre_BCSRMatrixBlock *A , double *x_data , double beta , double *y_data );
int hypre_BCSRMatrixDenseBlockMatvecT( double alpha , hypre_BCSRMatrixBlock *A , double *x_data , double beta , double *y_data );
double hypre_BCSRMatrixDenseBlockNorm( hypre_BCSRMatrixDenseBlock *A , const char *norm );
int hypre_BCSRMatrixDenseBlockPrint( hypre_BCSRMatrixDenseBlock *A , FILE *out_file );

/* bcsr_relax.c */
int hypre_BCSRMatrixRelax( hypre_BCSRMatrix *A , hypre_Vector *f , int *cf_marker , int relax_points , hypre_Vector *u );

/* cg_fun.c */
char *hypre_CGCAlloc( int count , int elt_size );
int hypre_CGFree( char *ptr );
void *hypre_CGCreateVector( void *vvector );
void *hypre_CGCreateVectorArray( int n , void *vvector );
int hypre_CGDestroyVector( void *vvector );
void *hypre_CGMatvecCreate( void *A , void *x );
int hypre_CGMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_CGMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_CGMatvecDestroy( void *matvec_data );
double hypre_CGInnerProd( void *x , void *y );
int hypre_CGCopyVector( void *x , void *y );
int hypre_CGClearVector( void *x );
int hypre_CGScaleVector( double alpha , void *x );
int hypre_CGAxpy( double alpha , void *x , void *y );
int hypre_CGCommInfo( void *A , int *my_id , int *num_procs );
int hypre_CGIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_CGIdentity( void *vdata , void *A , void *b , void *x );

/* coarsen.c */
int hypre_AMGCoarsen( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenRuge( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenRugeLoL( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenwLJP( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenRugeOnePass( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , int **CF_marker_ptr , int *coarse_size_ptr );

/* coarsenCR.c */
int hypre_AMGCoarsenCR( hypre_CSRMatrix *A , double strength_threshold , double relax_weight , int relax_type , int num_relax_steps , int **CF_marker_ptr , int *coarse_size_ptr );

/* cycle.c */
int hypre_AMGCycle( void *amg_vdata , hypre_Vector **F_array , hypre_Vector **U_array );

/* difconv.c */
hypre_CSRMatrix *hypre_GenerateDifConv( int nx , int ny , int nz , int P , int Q , int R , double *value );

/* driver.c */
int BuildFromFile( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildStencilMatrix( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian9pt( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian27pt( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildDifConv( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildRhsFromFile( int argc , char *argv [], int arg_index , hypre_CSRMatrix *A , hypre_Vector **b_ptr );
int BuildFuncsFromFile( int argc , char *argv [], int arg_index , int **dof_func_ptr );

/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize( void );
int HYPRE_AMGFinalize( HYPRE_Solver solver );
int HYPRE_AMGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSetMaxLevels( HYPRE_Solver solver , int max_levels );
int HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
int HYPRE_AMGSetMode( HYPRE_Solver solver , int mode );
int HYPRE_AMGSetATruncFactor( HYPRE_Solver solver , double A_trunc_factor );
int HYPRE_AMGSetAMaxElmts( HYPRE_Solver solver , int A_max_elmts );
int HYPRE_AMGSetPTruncFactor( HYPRE_Solver solver , double P_trunc_factor );
int HYPRE_AMGSetPMaxElmts( HYPRE_Solver solver , int P_max_elmts );
int HYPRE_AMGSetCoarsenType( HYPRE_Solver solver , int coarsen_type );
int HYPRE_AMGSetAggCoarsenType( HYPRE_Solver solver , int agg_coarsen_type );
int HYPRE_AMGSetAggLevels( HYPRE_Solver solver , int agg_levels );
int HYPRE_AMGSetInterpType( HYPRE_Solver solver , int interp_type );
int HYPRE_AMGSetAggInterpType( HYPRE_Solver solver , int agg_interp_type );
int HYPRE_AMGSetNumJacs( HYPRE_Solver solver , int num_jacs );
int HYPRE_AMGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_AMGSetCycleType( HYPRE_Solver solver , int cycle_type );
int HYPRE_AMGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_AMGSetNumRelaxSteps( HYPRE_Solver solver , int num_relax_steps );
int HYPRE_AMGSetNumGridSweeps( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_AMGSetGridRelaxType( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_AMGSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
int HYPRE_AMGSetSchwarzOption( HYPRE_Solver solver , int *schwarz_option );
int HYPRE_AMGSetIOutDat( HYPRE_Solver solver , int ioutdat );
int HYPRE_AMGSetLogFileName( HYPRE_Solver solver , char *log_file_name );
int HYPRE_AMGSetLogging( HYPRE_Solver solver , int ioutdat , char *log_file_name );
int HYPRE_AMGSetNumFunctions( HYPRE_Solver solver , int num_functions );
int HYPRE_AMGSetDofFunc( HYPRE_Solver solver , int *dof_func );
int HYPRE_AMGSetUseBlockFlag( HYPRE_Solver solver , int use_block_flag );

/* HYPRE_csr_gmres.c */
int HYPRE_CSRGMRESCreate( HYPRE_Solver *solver );
int HYPRE_CSRGMRESDestroy( HYPRE_Solver solver );
int HYPRE_CSRGMRESSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRGMRESSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRGMRESSetKDim( HYPRE_Solver solver , int k_dim );
int HYPRE_CSRGMRESSetTol( HYPRE_Solver solver , double tol );
int HYPRE_CSRGMRESSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_CSRGMRESSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_CSRGMRESSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_CSRGMRESSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
int HYPRE_CSRGMRESGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CSRGMRESSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_CSRGMRESGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_csr_pcg.c */
int HYPRE_CSRPCGCreate( HYPRE_Solver *solver );
int HYPRE_CSRPCGDestroy( HYPRE_Solver solver );
int HYPRE_CSRPCGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRPCGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRPCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_CSRPCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_CSRPCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_CSRPCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_CSRPCGSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
int HYPRE_CSRPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CSRPCGSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_CSRPCGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );
int HYPRE_CSRDiagScaleSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector y , HYPRE_Vector x );
int HYPRE_CSRDiagScale( HYPRE_Solver solver , HYPRE_CSRMatrix HA , HYPRE_Vector Hy , HYPRE_Vector Hx );

/* indepset.c */
int hypre_InitAMGIndepSet( hypre_CSRMatrix *S , double *measure_array , double cconst );
int hypre_AMGIndepSet( hypre_CSRMatrix *S , double *measure_array , double cconst , int *graph_array , int graph_array_size , int *IS_marker );

/* interp.c */
int hypre_AMGBuildInterp( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
int hypre_AMGBuildMultipass( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
int hypre_AMGJacobiIterate( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );

/* interpCR.c */
int hypre_AMGBuildCRInterp( hypre_CSRMatrix *A , int *CF_marker , int n_coarse , int num_relax_steps , int relax_type , double relax_weight , hypre_CSRMatrix **P_ptr );

/* interpRBM.c */
int hypre_AMGBuildRBMInterp( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int num_functions , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
int row_mat_rectmat_prod( double *a1 , double *a2 , double *a3 , int i_row , int m , int n );
int matinv( double *x , double *a , int k );

/* inx_part_of_u_interp.c */
int hypre_CreateDomain( int *CF_marker , hypre_CSRMatrix *A , int num_coarse , int *dof_func , int **coarse_dof_ptr , int **domain_i_ptr , int **domain_j_ptr );
int hypre_InexactPartitionOfUnityInterpolation( hypre_CSRMatrix **P_pointer , int *i_dof_dof , int *j_dof_dof , double *a_dof_dof , double *unit_vector , int *i_domain_dof , int *j_domain_dof , int num_domains , int num_dofs );
int compute_sym_GS_T_action( double *x , double *v , double *w , int *i_domain_dof , int *j_domain_dof , int nu_max , int *i_dof_dof , int *j_dof_dof , double *a_dof_dof , int *i_global_to_local , int num_domains , int num_dofs );
int compute_sum_A_i_action( double *w , double *v , int *i_domain_dof , int *j_domain_dof , int *i_dof_dof , int *j_dof_dof , double *a_dof_dof , int *i_global_to_local , int num_domains , int num_dofs );

/* laplace_27pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian27pt( int nx , int ny , int nz , int P , int Q , int R , double *value );
int map3( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* laplace_9pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian9pt( int nx , int ny , int P , int Q , double *value );
int map2( int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part );

/* laplace.c */
hypre_CSRMatrix *hypre_GenerateLaplacian( int nx , int ny , int nz , int P , int Q , int R , double *value );
int map( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );
hypre_CSRMatrix *hypre_GenerateSysLaplacian( int nx, int ny, int  nz, int P, int Q, int R, int num_fun, double  *mtrx, double  *value );
   
/* pcg.c */
void PCG( hypre_Vector *x , hypre_Vector *b , double tol , void *data );
void PCGSetup( hypre_CSRMatrix *A , int (*precond )(), void *precond_data , void *data );

/* random.c */
void hypre_SeedRand( int seed );
double hypre_Rand( void );

/* rap.c */
int hypre_AMGBuildCoarseOperator( hypre_CSRMatrix *RT , hypre_CSRMatrix *A , hypre_CSRMatrix *P , hypre_CSRMatrix **RAP_ptr );

/* relax.c */
int hypre_AMGRelax( hypre_CSRMatrix *A , hypre_Vector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_Vector *u , hypre_Vector *Vtemp );
int gselim( double *A , double *x , int n );

/* scaled_matnorm.c */
int hypre_CSRMatrixScaledNorm( hypre_CSRMatrix *A , double *scnorm );

/* schwarz.c */
int hypre_AMGNodalSchwarzSmoother( hypre_CSRMatrix *A , int *dof_func , int num_functions , int option , int **i_domain_dof_pointer , int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , int *num_domains_pointer );
int hypre_SchwarzSolve( hypre_CSRMatrix *A , hypre_Vector *rhs_vector , int num_domains , int *i_domain_dof , int *j_domain_dof , double *domain_matrixinverse , hypre_Vector *x_vector , hypre_Vector *aux_vector );
int transpose_matrix_create( int **i_face_element_pointer , int **j_face_element_pointer , int *i_element_face , int *j_element_face , int num_elements , int num_faces );
int matrix_matrix_product( int **i_element_edge_pointer , int **j_element_edge_pointer , int *i_element_face , int *j_element_face , int *i_face_edge , int *j_face_edge , int num_elements , int num_faces , int num_edges );
int hypre_AMGCreateDomainDof( hypre_CSRMatrix *A , int **i_domain_dof_pointer , int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , int *num_domains_pointer );
int hypre_AMGeAgglomerate( int *i_AE_element , int *j_AE_element , int *i_face_face , int *j_face_face , int *w_face_face , int *i_face_element , int *j_face_element , int *i_element_face , int *j_element_face , int *i_face_to_prefer_weight , int *i_face_weight , int num_faces , int num_elements , int *num_AEs_pointer );
int update_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int remove_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int move_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );

/* SPamg-pcg.c */

/* stencil_matrix.c */
hypre_CSRMatrix *hypre_GenerateStencilMatrix( int nx , int ny , int nz , char *infile );

/* strength.c */
int hypre_AMGCreateS( hypre_CSRMatrix *A , double strength_threshold , int mode , int *dof_func , hypre_CSRMatrix **S_ptr );
int hypre_AMGCompressS( hypre_CSRMatrix *S , int num_path );
int hypre_AMGCreate2ndS( hypre_CSRMatrix *A , int n_coarse , int *CF_marker , int num_paths , hypre_CSRMatrix **S_ptr );
int hypre_AMGCorrectCFMarker( int *CF_marker , int num_var , int *new_CF_marker );

/* trunc.c */
int hypre_AMGTruncation( hypre_CSRMatrix *A , double trunc_factor , int max_elmts );
void swap3( int *v , double *w , int i , int j );
void qsort2( int *v , double *w , int left , int right );


#ifdef __cplusplus
}
#endif

#endif

