/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.18 $
 ***********************************************************************EHEADER*/





#include <HYPRE_config.h>

#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "_hypre_utilities.h"
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
  HYPRE_Int* i;
  HYPRE_Int* j;
  HYPRE_Int num_block_rows;
  HYPRE_Int num_block_cols;
  HYPRE_Int num_nonzero_blocks;
  HYPRE_Int num_rows_per_block;
  HYPRE_Int num_cols_per_block;
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
hypre_BCSRMatrixCreate(HYPRE_Int num_block_rows, HYPRE_Int num_block_cols,
		       HYPRE_Int num_nonzero_blocks,
		       HYPRE_Int num_rows_per_block, HYPRE_Int num_cols_per_block);

HYPRE_Int
hypre_BCSRMatrixDestroy(hypre_BCSRMatrix* A);

HYPRE_Int
hypre_BCSRMatrixInitialise(hypre_BCSRMatrix* A);

HYPRE_Int
hypre_BCSRMatrixPrint(hypre_BCSRMatrix* A, char* file_name);

HYPRE_Int
hypre_BCSRMatrixTranspose(hypre_BCSRMatrix* A, hypre_BCSRMatrix** AT);

hypre_BCSRMatrix*
hypre_BCSRMatrixFromCSRMatrix(hypre_CSRMatrix* A,
			      HYPRE_Int num_rows_per_block, HYPRE_Int num_cols_per_block);

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
hypre_BCSRMatrixBuildInterp(hypre_BCSRMatrix* A, HYPRE_Int* CF_marker,
			    hypre_CSRMatrix* S, HYPRE_Int coarse_size);

hypre_BCSRMatrix*
hypre_BCSRMatrixBuildInterpD(hypre_BCSRMatrix* A, HYPRE_Int* CF_marker,
			     hypre_CSRMatrix* S, HYPRE_Int coarse_size);

HYPRE_Int
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
  HYPRE_Int num_rows;
  HYPRE_Int num_cols;
} hypre_BCSRMatrixDenseBlock;

/*****************************************************************************
 *
 * Prototypes
 *
 *****************************************************************************/

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCreate(HYPRE_Int num_rows, HYPRE_Int num_cols);

HYPRE_Int
hypre_BCSRMatrixDenseBlockDestroy(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockInitialise(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockFillData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

HYPRE_Int
hypre_BCSRMatrixDenseBlockGetData(hypre_BCSRMatrixDenseBlock* A,
				   double* data);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCopy(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockAdd(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiply(hypre_BCSRMatrixDenseBlock* A,
				   hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockNeg(hypre_BCSRMatrixDenseBlock* A);

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockDiag(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMulInv(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);

HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiplyInverse2(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B);


HYPRE_Int
hypre_BCSRMatrixDenseBlockTranspose(hypre_BCSRMatrixDenseBlock* A);

HYPRE_Int
hypre_BCSRMatrixBlockMatvec(double alpha, hypre_BCSRMatrixDenseBlock* A,
			    double* x_data, double beta, double* y_data);

HYPRE_Int
hypre_BCSRMatrixBlockMatvecT(double alpha, hypre_BCSRMatrixDenseBlock* A,
			     double* x_data, double beta, double* y_data);

double
hypre_BCSRMatrixDenseBlockNorm(hypre_BCSRMatrixDenseBlock* A,
			       const char* norm);

HYPRE_Int
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
#define hypre_BCSRMatrixBlockMultiplyInverse2 hypre_BCSRMatrixDenseBlockMultiplyInverse2
#define hypre_BCSRMatrixBlockTranspose hypre_BCSRMatrixDenseBlockTranspose
#define hypre_BCSRMatrixBlockMatvec hypre_BCSRMatrixDenseBlockMatvec
#define hypre_BCSRMatrixBlockMatvecT hypre_BCSRMatrixDenseBlockMatvecT
#define hypre_BCSRMatrixBlockNorm hypre_BCSRMatrixDenseBlockNorm
#define hypre_BCSRMatrixBlockPrint hypre_BCSRMatrixDenseBlockPrint

#endif

#endif

#ifndef hypre_AMG_DATA_HEADER
#define hypre_AMG_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_AMGData
 *--------------------------------------------------------------------------*/

typedef struct
{

   /* setup params */
   HYPRE_Int      max_levels;
   double   strong_threshold;
   double   A_trunc_factor;
   double   P_trunc_factor;
   HYPRE_Int      A_max_elmts;
   HYPRE_Int      P_max_elmts;
   HYPRE_Int      coarsen_type;
   HYPRE_Int      agg_coarsen_type;
   HYPRE_Int      interp_type;
   HYPRE_Int      agg_interp_type;
   HYPRE_Int      agg_levels;
   HYPRE_Int      num_relax_steps;  
   HYPRE_Int      num_jacs;
   HYPRE_Int use_block_flag;

   /* solve params */
   HYPRE_Int      max_iter;
   HYPRE_Int      cycle_type;    
   HYPRE_Int     *num_grid_sweeps;  
   HYPRE_Int     *grid_relax_type;   
   HYPRE_Int    **grid_relax_points; 
   double  *relax_weight;
   double   tol;
   /* problem data */
   hypre_CSRMatrix  *A;
   HYPRE_Int      num_variables;
   HYPRE_Int      num_functions;
   HYPRE_Int      num_points;
   HYPRE_Int     *dof_func;
   HYPRE_Int     *dof_point;
   HYPRE_Int     *point_dof_map;           

   /* data generated in the setup phase */
   hypre_CSRMatrix **A_array;
   hypre_BCSRMatrix **B_array;
   hypre_Vector    **F_array;
   hypre_Vector    **U_array;
   hypre_CSRMatrix **P_array;
   hypre_BCSRMatrix **PB_array;
   HYPRE_Int             **CF_marker_array;
   HYPRE_Int             **dof_func_array;
   HYPRE_Int             **dof_point_array;
   HYPRE_Int             **point_dof_map_array;
   HYPRE_Int               num_levels;
   HYPRE_Int      	    *schwarz_option;
   HYPRE_Int      	    *num_domains;
   HYPRE_Int     	   **i_domain_dof;
   HYPRE_Int     	   **j_domain_dof;
   double  	   **domain_matrixinverse;
   HYPRE_Int		     mode;

   /* data generated in the solve phase */
   hypre_Vector   *Vtemp;
   double   *vtmp;
   HYPRE_Int       cycle_op_count;                                                   

   /* output params */
   HYPRE_Int      ioutdat;
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
#define round(x)  ( ((x) < 0.0) ? ((HYPRE_Int)(x - 0.5)) : ((HYPRE_Int)(x + 0.5)) )
#endif

#endif


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
   HYPRE_Int      max_iter;
   HYPRE_Int      two_norm;

   hypre_CSRMatrix  *A;
   hypre_Vector  *p;
   hypre_Vector  *s;
   hypre_Vector  *r;

   HYPRE_Int    (*precond)();
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
void *hypre_AMGInitialize ( void );
HYPRE_Int hypre_AMGFinalize ( void *data );
HYPRE_Int hypre_AMGSetMaxLevels ( void *data , HYPRE_Int max_levels );
HYPRE_Int hypre_AMGSetStrongThreshold ( void *data , double strong_threshold );
HYPRE_Int hypre_AMGSetMode ( void *data , HYPRE_Int mode );
HYPRE_Int hypre_AMGSetATruncFactor ( void *data , double A_trunc_factor );
HYPRE_Int hypre_AMGSetPTruncFactor ( void *data , double P_trunc_factor );
HYPRE_Int hypre_AMGSetAMaxElmts ( void *data , HYPRE_Int A_max_elmts );
HYPRE_Int hypre_AMGSetPMaxElmts ( void *data , HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGSetCoarsenType ( void *data , HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGSetAggCoarsenType ( void *data , HYPRE_Int agg_coarsen_type );
HYPRE_Int hypre_AMGSetAggLevels ( void *data , HYPRE_Int agg_levels );
HYPRE_Int hypre_AMGSetInterpType ( void *data , HYPRE_Int interp_type );
HYPRE_Int hypre_AMGSetAggInterpType ( void *data , HYPRE_Int agg_interp_type );
HYPRE_Int hypre_AMGSetNumJacs ( void *data , HYPRE_Int num_jacs );
HYPRE_Int hypre_AMGSetMaxIter ( void *data , HYPRE_Int max_iter );
HYPRE_Int hypre_AMGSetCycleType ( void *data , HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGSetTol ( void *data , double tol );
HYPRE_Int hypre_AMGSetNumRelaxSteps ( void *data , HYPRE_Int num_relax_steps );
HYPRE_Int hypre_AMGSetNumGridSweeps ( void *data , HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGSetGridRelaxType ( void *data , HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGSetGridRelaxPoints ( void *data , HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGSetRelaxWeight ( void *data , double *relax_weight );
HYPRE_Int hypre_AMGSetSchwarzOption ( void *data , HYPRE_Int *schwarz_option );
HYPRE_Int hypre_AMGSetIOutDat ( void *data , HYPRE_Int ioutdat );
HYPRE_Int hypre_AMGSetLogFileName ( void *data , char *log_file_name );
HYPRE_Int hypre_AMGSetLogging ( void *data , HYPRE_Int ioutdat , char *log_file_name );
HYPRE_Int hypre_AMGSetUseBlockFlag ( void *data , HYPRE_Int use_block_flag );
HYPRE_Int hypre_AMGSetNumFunctions ( void *data , HYPRE_Int num_functions );
HYPRE_Int hypre_AMGSetNumPoints ( void *data , HYPRE_Int num_points );
HYPRE_Int hypre_AMGSetDofFunc ( void *data , HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGSetDofPoint ( void *data , HYPRE_Int *dof_point );
HYPRE_Int hypre_AMGSetPointDofMap ( void *data , HYPRE_Int *point_dof_map );

/* amg_setup.c */
HYPRE_Int hypre_AMGSetup ( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amg_solve.c */
HYPRE_Int hypre_AMGSolve ( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amgstats.c */
HYPRE_Int hypre_AMGSetupStats ( void *amg_vdata );
void hypre_WriteSolverParams ( void *data );

/* Atrunc.c */
HYPRE_Int hypre_AMGOpTruncation ( hypre_CSRMatrix *A , double trunc_factor , HYPRE_Int max_elmts );

/* bcsr_interp.c */
hypre_BCSRMatrix *hypre_BCSRMatrixBuildInterp ( hypre_BCSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int coarse_size );
hypre_BCSRMatrix *hypre_BCSRMatrixBuildInterpD ( hypre_BCSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int coarse_size );

/* bcsr_matrix.c */
hypre_BCSRMatrix *hypre_BCSRMatrixCreate ( HYPRE_Int num_block_rows , HYPRE_Int num_block_cols , HYPRE_Int num_nonzero_blocks , HYPRE_Int num_rows_per_block , HYPRE_Int num_cols_per_block );
HYPRE_Int hypre_BCSRMatrixDestroy ( hypre_BCSRMatrix *A );
HYPRE_Int hypre_BCSRMatrixInitialise ( hypre_BCSRMatrix *A );
HYPRE_Int hypre_BCSRMatrixPrint ( hypre_BCSRMatrix *A , char *file_name );
HYPRE_Int hypre_BCSRMatrixTranspose ( hypre_BCSRMatrix *A , hypre_BCSRMatrix **AT );
hypre_BCSRMatrix *hypre_BCSRMatrixFromCSRMatrix ( hypre_CSRMatrix *A , HYPRE_Int num_rows_per_block , HYPRE_Int num_cols_per_block );
hypre_CSRMatrix *hypre_BCSRMatrixToCSRMatrix ( hypre_BCSRMatrix *B );
hypre_CSRMatrix *hypre_BCSRMatrixCompress ( hypre_BCSRMatrix *A );

/* bcsr_matrix_dense_block.c */
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols );
HYPRE_Int hypre_BCSRMatrixDenseBlockDestroy ( hypre_BCSRMatrixDenseBlock *A );
HYPRE_Int hypre_BCSRMatrixDenseBlockInitialise ( hypre_BCSRMatrixDenseBlock *A );
HYPRE_Int hypre_BCSRMatrixDenseBlockFillData ( hypre_BCSRMatrixDenseBlock *A , double *data );
HYPRE_Int hypre_BCSRMatrixDenseBlockGetData ( hypre_BCSRMatrixDenseBlock *A , double *data );
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockCopy ( hypre_BCSRMatrixDenseBlock *A );
HYPRE_Int hypre_BCSRMatrixDenseBlockAdd ( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
HYPRE_Int hypre_BCSRMatrixDenseBlockMultiply ( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
HYPRE_Int hypre_BCSRMatrixDenseBlockNeg ( hypre_BCSRMatrixDenseBlock *A );
hypre_BCSRMatrixDenseBlock *hypre_BCSRMatrixDenseBlockDiag ( hypre_BCSRMatrixDenseBlock *A );
HYPRE_Int hypre_BCSRMatrixDenseBlockMulInv ( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
HYPRE_Int hypre_BCSRMatrixDenseBlockMultiplyInverse2 ( hypre_BCSRMatrixDenseBlock *A , hypre_BCSRMatrixDenseBlock *B );
HYPRE_Int hypre_BCSRMatrixDenseBlockTranspose ( hypre_BCSRMatrixDenseBlock *A );
HYPRE_Int hypre_BCSRMatrixDenseBlockMatvec ( double alpha , hypre_BCSRMatrixBlock *A , double *x_data , double beta , double *y_data );
HYPRE_Int hypre_BCSRMatrixDenseBlockMatvecT ( double alpha , hypre_BCSRMatrixBlock *A , double *x_data , double beta , double *y_data );
double hypre_BCSRMatrixDenseBlockNorm ( hypre_BCSRMatrixDenseBlock *A , const char *norm );
HYPRE_Int hypre_BCSRMatrixDenseBlockPrint ( hypre_BCSRMatrixDenseBlock *A , FILE *out_file );

/* bcsr_relax.c */
HYPRE_Int hypre_BCSRMatrixRelax ( hypre_BCSRMatrix *A , hypre_Vector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_points , hypre_Vector *u );

/* cg_fun.c */
char *hypre_CGCAlloc ( HYPRE_Int count , HYPRE_Int elt_size );
HYPRE_Int hypre_CGFree ( char *ptr );
void *hypre_CGCreateVector ( void *vvector );
void *hypre_CGCreateVectorArray ( HYPRE_Int n , void *vvector );
HYPRE_Int hypre_CGDestroyVector ( void *vvector );
void *hypre_CGMatvecCreate ( void *A , void *x );
HYPRE_Int hypre_CGMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
HYPRE_Int hypre_CGMatvecT ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
HYPRE_Int hypre_CGMatvecDestroy ( void *matvec_data );
double hypre_CGInnerProd ( void *x , void *y );
HYPRE_Int hypre_CGCopyVector ( void *x , void *y );
HYPRE_Int hypre_CGClearVector ( void *x );
HYPRE_Int hypre_CGScaleVector ( double alpha , void *x );
HYPRE_Int hypre_CGAxpy ( double alpha , void *x , void *y );
HYPRE_Int hypre_CGCommInfo ( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );
HYPRE_Int hypre_CGIdentitySetup ( void *vdata , void *A , void *b , void *x );
HYPRE_Int hypre_CGIdentity ( void *vdata , void *A , void *b , void *x );

/* coarsen.c */
HYPRE_Int hypre_AMGCoarsen ( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );
HYPRE_Int hypre_AMGCoarsenRuge ( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );
HYPRE_Int hypre_AMGCoarsenRugeLoL ( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );
HYPRE_Int hypre_AMGCoarsenwLJP ( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );
HYPRE_Int hypre_AMGCoarsenRugeOnePass ( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix *S , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );

/* coarsenCR.c */
HYPRE_Int hypre_AMGCoarsenCR ( hypre_CSRMatrix *A , double strength_threshold , double relax_weight , HYPRE_Int relax_type , HYPRE_Int num_relax_steps , HYPRE_Int **CF_marker_ptr , HYPRE_Int *coarse_size_ptr );

/* cycle.c */
HYPRE_Int hypre_AMGCycle ( void *amg_vdata , hypre_Vector **F_array , hypre_Vector **U_array );

/* difconv.c */
hypre_CSRMatrix *hypre_GenerateDifConv ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , double *value );

/* driver.c */
HYPRE_Int BuildFromFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildLaplacian ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildStencilMatrix ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildLaplacian9pt ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildLaplacian27pt ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildDifConv ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix **A_ptr );
HYPRE_Int BuildRhsFromFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , hypre_CSRMatrix *A , hypre_Vector **b_ptr );
HYPRE_Int BuildFuncsFromFile ( HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_Int **dof_func_ptr );
HYPRE_Int SetSysVcoefValues ( HYPRE_Int num_fun , HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , double vcx , double vcy , double vcz , HYPRE_Int mtx_entry , double *values );

/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize ( void );
HYPRE_Int HYPRE_AMGFinalize ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMGSetup ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_AMGSolve ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_AMGSetMaxLevels ( HYPRE_Solver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_AMGSetStrongThreshold ( HYPRE_Solver solver , double strong_threshold );
HYPRE_Int HYPRE_AMGSetMode ( HYPRE_Solver solver , HYPRE_Int mode );
HYPRE_Int HYPRE_AMGSetATruncFactor ( HYPRE_Solver solver , double A_trunc_factor );
HYPRE_Int HYPRE_AMGSetAMaxElmts ( HYPRE_Solver solver , HYPRE_Int A_max_elmts );
HYPRE_Int HYPRE_AMGSetPTruncFactor ( HYPRE_Solver solver , double P_trunc_factor );
HYPRE_Int HYPRE_AMGSetPMaxElmts ( HYPRE_Solver solver , HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_AMGSetCoarsenType ( HYPRE_Solver solver , HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_AMGSetAggCoarsenType ( HYPRE_Solver solver , HYPRE_Int agg_coarsen_type );
HYPRE_Int HYPRE_AMGSetAggLevels ( HYPRE_Solver solver , HYPRE_Int agg_levels );
HYPRE_Int HYPRE_AMGSetInterpType ( HYPRE_Solver solver , HYPRE_Int interp_type );
HYPRE_Int HYPRE_AMGSetAggInterpType ( HYPRE_Solver solver , HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_AMGSetNumJacs ( HYPRE_Solver solver , HYPRE_Int num_jacs );
HYPRE_Int HYPRE_AMGSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_AMGSetCycleType ( HYPRE_Solver solver , HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMGSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_AMGSetNumRelaxSteps ( HYPRE_Solver solver , HYPRE_Int num_relax_steps );
HYPRE_Int HYPRE_AMGSetNumGridSweeps ( HYPRE_Solver solver , HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_AMGSetGridRelaxType ( HYPRE_Solver solver , HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_AMGSetGridRelaxPoints ( HYPRE_Solver solver , HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_AMGSetRelaxWeight ( HYPRE_Solver solver , double *relax_weight );
HYPRE_Int HYPRE_AMGSetSchwarzOption ( HYPRE_Solver solver , HYPRE_Int *schwarz_option );
HYPRE_Int HYPRE_AMGSetIOutDat ( HYPRE_Solver solver , HYPRE_Int ioutdat );
HYPRE_Int HYPRE_AMGSetLogFileName ( HYPRE_Solver solver , char *log_file_name );
HYPRE_Int HYPRE_AMGSetLogging ( HYPRE_Solver solver , HYPRE_Int ioutdat , char *log_file_name );
HYPRE_Int HYPRE_AMGSetNumFunctions ( HYPRE_Solver solver , HYPRE_Int num_functions );
HYPRE_Int HYPRE_AMGSetDofFunc ( HYPRE_Solver solver , HYPRE_Int *dof_func );
HYPRE_Int HYPRE_AMGSetUseBlockFlag ( HYPRE_Solver solver , HYPRE_Int use_block_flag );

/* HYPRE_csr_gmres.c */
HYPRE_Int HYPRE_CSRGMRESCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_CSRGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CSRGMRESSetup ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CSRGMRESSolve ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CSRGMRESSetKDim ( HYPRE_Solver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_CSRGMRESSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_CSRGMRESSetMinIter ( HYPRE_Solver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_CSRGMRESSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_CSRGMRESSetStopCrit ( HYPRE_Solver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CSRGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_Int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), HYPRE_Int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
HYPRE_Int HYPRE_CSRGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CSRGMRESSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_CSRGMRESGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CSRGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_csr_pcg.c */
HYPRE_Int HYPRE_CSRPCGCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_CSRPCGDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CSRPCGSetup ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CSRPCGSolve ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
HYPRE_Int HYPRE_CSRPCGSetTol ( HYPRE_Solver solver , double tol );
HYPRE_Int HYPRE_CSRPCGSetMaxIter ( HYPRE_Solver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_CSRPCGSetTwoNorm ( HYPRE_Solver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_CSRPCGSetRelChange ( HYPRE_Solver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_CSRPCGSetPrecond ( HYPRE_Solver solver , HYPRE_Int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), HYPRE_Int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
HYPRE_Int HYPRE_CSRPCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CSRPCGSetLogging ( HYPRE_Solver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_CSRPCGGetNumIterations ( HYPRE_Solver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CSRPCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
HYPRE_Int HYPRE_CSRDiagScaleSetup ( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector y , HYPRE_Vector x );
HYPRE_Int HYPRE_CSRDiagScale ( HYPRE_Solver solver , HYPRE_CSRMatrix HA , HYPRE_Vector Hy , HYPRE_Vector Hx );

/* indepset.c */
HYPRE_Int hypre_InitAMGIndepSet ( hypre_CSRMatrix *S , double *measure_array , double cconst );
HYPRE_Int hypre_AMGIndepSet ( hypre_CSRMatrix *S , double *measure_array , double cconst , HYPRE_Int *graph_array , HYPRE_Int graph_array_size , HYPRE_Int *IS_marker );

/* interp.c */
HYPRE_Int hypre_AMGBuildInterp ( hypre_CSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
HYPRE_Int hypre_AMGBuildMultipass ( hypre_CSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
HYPRE_Int hypre_AMGJacobiIterate ( hypre_CSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );

/* interpCR.c */
HYPRE_Int hypre_AMGBuildCRInterp ( hypre_CSRMatrix *A , HYPRE_Int *CF_marker , HYPRE_Int n_coarse , HYPRE_Int num_relax_steps , HYPRE_Int relax_type , double relax_weight , hypre_CSRMatrix **P_ptr );

/* interpRBM.c */
HYPRE_Int hypre_AMGBuildRBMInterp ( hypre_CSRMatrix *A , HYPRE_Int *CF_marker , hypre_CSRMatrix *S , HYPRE_Int *dof_func , HYPRE_Int num_functions , HYPRE_Int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
HYPRE_Int row_mat_rectmat_prod ( double *a1 , double *a2 , double *a3 , HYPRE_Int i_row , HYPRE_Int m , HYPRE_Int n );
HYPRE_Int matinv ( double *x , double *a , HYPRE_Int k );

/* inx_part_of_u_interp.c */
HYPRE_Int hypre_CreateDomain ( HYPRE_Int *CF_marker , hypre_CSRMatrix *A , HYPRE_Int num_coarse , HYPRE_Int *dof_func , HYPRE_Int **coarse_dof_ptr , HYPRE_Int **domain_i_ptr , HYPRE_Int **domain_j_ptr );
HYPRE_Int hypre_InexactPartitionOfUnityInterpolation ( hypre_CSRMatrix **P_pointer , HYPRE_Int *i_dof_dof , HYPRE_Int *j_dof_dof , double *a_dof_dof , double *unit_vector , HYPRE_Int *i_domain_dof , HYPRE_Int *j_domain_dof , HYPRE_Int num_domains , HYPRE_Int num_dofs );
HYPRE_Int compute_sym_GS_T_action ( double *x , double *v , double *w , HYPRE_Int *i_domain_dof , HYPRE_Int *j_domain_dof , HYPRE_Int nu_max , HYPRE_Int *i_dof_dof , HYPRE_Int *j_dof_dof , double *a_dof_dof , HYPRE_Int *i_global_to_local , HYPRE_Int num_domains , HYPRE_Int num_dofs );
HYPRE_Int compute_sum_A_i_action ( double *w , double *v , HYPRE_Int *i_domain_dof , HYPRE_Int *j_domain_dof , HYPRE_Int *i_dof_dof , HYPRE_Int *j_dof_dof , double *a_dof_dof , HYPRE_Int *i_global_to_local , HYPRE_Int num_domains , HYPRE_Int num_dofs );

/* laplace_27pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian27pt ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , double *value );
HYPRE_Int map3 ( HYPRE_Int ix , HYPRE_Int iy , HYPRE_Int iz , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int *nx_part , HYPRE_Int *ny_part , HYPRE_Int *nz_part , HYPRE_Int *global_part );

/* laplace_9pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian9pt ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int P , HYPRE_Int Q , double *value );
HYPRE_Int map2 ( HYPRE_Int ix , HYPRE_Int iy , HYPRE_Int p , HYPRE_Int q , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int *nx_part , HYPRE_Int *ny_part , HYPRE_Int *global_part );

/* laplace.c */
hypre_CSRMatrix *hypre_GenerateLaplacian ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , double *value );
HYPRE_Int map ( HYPRE_Int ix , HYPRE_Int iy , HYPRE_Int iz , HYPRE_Int p , HYPRE_Int q , HYPRE_Int r , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int *nx_part , HYPRE_Int *ny_part , HYPRE_Int *nz_part , HYPRE_Int *global_part );
hypre_CSRMatrix *hypre_GenerateSysLaplacian ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int num_fun , double *mtrx , double *value );
hypre_CSRMatrix *hypre_GenerateSysLaplacianVCoef ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , HYPRE_Int P , HYPRE_Int Q , HYPRE_Int R , HYPRE_Int num_fun , double *mtrx , double *value );

/* pcg.c */
void PCG ( hypre_Vector *x , hypre_Vector *b , double tol , void *data );
void PCGSetup ( hypre_CSRMatrix *A , HYPRE_Int (*precond )(), void *precond_data , void *data );

/* random.c */
void hypre_SeedRand ( HYPRE_Int seed );
double hypre_Rand ( void );

/* rap.c */
HYPRE_Int hypre_AMGBuildCoarseOperator ( hypre_CSRMatrix *RT , hypre_CSRMatrix *A , hypre_CSRMatrix *P , hypre_CSRMatrix **RAP_ptr );

/* relax.c */
HYPRE_Int hypre_AMGRelax ( hypre_CSRMatrix *A , hypre_Vector *f , HYPRE_Int *cf_marker , HYPRE_Int relax_type , HYPRE_Int relax_points , double relax_weight , hypre_Vector *u , hypre_Vector *Vtemp );
HYPRE_Int gselim ( double *A , double *x , HYPRE_Int n );

/* scaled_matnorm.c */
HYPRE_Int hypre_CSRMatrixScaledNorm ( hypre_CSRMatrix *A , double *scnorm );

/* schwarz.c */
HYPRE_Int hypre_AMGNodalSchwarzSmoother ( hypre_CSRMatrix *A , HYPRE_Int *dof_func , HYPRE_Int num_functions , HYPRE_Int option , HYPRE_Int **i_domain_dof_pointer , HYPRE_Int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , HYPRE_Int *num_domains_pointer );
HYPRE_Int hypre_SchwarzSolve ( hypre_CSRMatrix *A , hypre_Vector *rhs_vector , HYPRE_Int num_domains , HYPRE_Int *i_domain_dof , HYPRE_Int *j_domain_dof , double *domain_matrixinverse , hypre_Vector *x_vector , hypre_Vector *aux_vector );
HYPRE_Int transpose_matrix_create ( HYPRE_Int **i_face_element_pointer , HYPRE_Int **j_face_element_pointer , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int num_elements , HYPRE_Int num_faces );
HYPRE_Int matrix_matrix_product ( HYPRE_Int **i_element_edge_pointer , HYPRE_Int **j_element_edge_pointer , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int *i_face_edge , HYPRE_Int *j_face_edge , HYPRE_Int num_elements , HYPRE_Int num_faces , HYPRE_Int num_edges );
HYPRE_Int hypre_AMGCreateDomainDof ( hypre_CSRMatrix *A , HYPRE_Int **i_domain_dof_pointer , HYPRE_Int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , HYPRE_Int *num_domains_pointer );
HYPRE_Int hypre_AMGeAgglomerate ( HYPRE_Int *i_AE_element , HYPRE_Int *j_AE_element , HYPRE_Int *i_face_face , HYPRE_Int *j_face_face , HYPRE_Int *w_face_face , HYPRE_Int *i_face_element , HYPRE_Int *j_face_element , HYPRE_Int *i_element_face , HYPRE_Int *j_element_face , HYPRE_Int *i_face_to_prefer_weight , HYPRE_Int *i_face_weight , HYPRE_Int num_faces , HYPRE_Int num_elements , HYPRE_Int *num_AEs_pointer );
HYPRE_Int update_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );
HYPRE_Int remove_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );
HYPRE_Int move_entry ( HYPRE_Int weight , HYPRE_Int *weight_max , HYPRE_Int *previous , HYPRE_Int *next , HYPRE_Int *first , HYPRE_Int *last , HYPRE_Int head , HYPRE_Int tail , HYPRE_Int i );

/* SPamg-pcg.c */

/* stencil_matrix.c */
hypre_CSRMatrix *hypre_GenerateStencilMatrix ( HYPRE_Int nx , HYPRE_Int ny , HYPRE_Int nz , char *infile );

/* strength.c */
HYPRE_Int hypre_AMGCreateS ( hypre_CSRMatrix *A , double strength_threshold , HYPRE_Int mode , HYPRE_Int *dof_func , hypre_CSRMatrix **S_ptr );
HYPRE_Int hypre_AMGCompressS ( hypre_CSRMatrix *S , HYPRE_Int num_path );
HYPRE_Int hypre_AMGCreate2ndS ( hypre_CSRMatrix *A , HYPRE_Int n_coarse , HYPRE_Int *CF_marker , HYPRE_Int num_paths , hypre_CSRMatrix **S_ptr );
HYPRE_Int hypre_AMGCorrectCFMarker ( HYPRE_Int *CF_marker , HYPRE_Int num_var , HYPRE_Int *new_CF_marker );

/* trunc.c */
HYPRE_Int hypre_AMGTruncation ( hypre_CSRMatrix *A , double trunc_factor , HYPRE_Int max_elmts );
void swap3 ( HYPRE_Int *v , double *w , HYPRE_Int i , HYPRE_Int j );
void qsort2 ( HYPRE_Int *v , double *w , HYPRE_Int left , HYPRE_Int right );

#ifdef __cplusplus
}
#endif

#endif

