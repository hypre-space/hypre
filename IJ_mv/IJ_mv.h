
#include <HYPRE_config.h>

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
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
 * Header info for Auxiliary Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PARCSR_MATRIX_HEADER
#define hypre_AUX_PARCSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      local_num_rows;   /* defines number of rows on this processors */
   int      local_num_cols;   /* defines number of cols of diag */

   int      need_aux; /* if need_aux = 1, aux_j, aux_data are used to
			generate the parcsr matrix (default),
			for need_aux = 0, data is put directly into
			parcsr structure (requires the knowledge of
			offd_i and diag_i ) */

   int     *row_length; /* row_length_diag[i] contains number of stored
				elements in i-th row */
   int     *row_space; /* row_space_diag[i] contains space allocated to
				i-th row */
   int    **aux_j;	/* contains collected column indices */
   double **aux_data; /* contains collected data */

   int     *indx_diag; /* indx_diag[i] points to first empty space of portion
			 in diag_j , diag_data assigned to row i */  
   int     *indx_offd; /* indx_offd[i] points to first empty space of portion
			 in offd_j , offd_data assigned to row i */  
} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)  ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)  ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixNeedAux(matrix)   ((matrix) -> need_aux)
#define hypre_AuxParCSRMatrixRowLength(matrix) ((matrix) -> row_length)
#define hypre_AuxParCSRMatrixRowSpace(matrix)  ((matrix) -> row_space)
#define hypre_AuxParCSRMatrixAuxJ(matrix)      ((matrix) -> aux_j)
#define hypre_AuxParCSRMatrixAuxData(matrix)   ((matrix) -> aux_data)

#define hypre_AuxParCSRMatrixIndxDiag(matrix)  ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)  ((matrix) -> indx_offd)

#endif
/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_MATRIX_HEADER
#define hypre_IJ_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   int M, N;                               /* number of rows and cols in matrix */


   void         *local_storage;            /* Structure for storing local portion */
   int      	 local_storage_type;       /* Indicates the type of "local storage" */
   void         *translator;               /* optional storage_type specfic structure
                                              for holding additional local info */

   int           insertion_semantics;      /* Flag that indicates for the current
                                              object to what extent values can be set
                                              from different processors than the one that
                                              stores the row. */
                                           /* 0: minimum definition, values can only be set on-processor. */
   int           ref_count;                /* reference count for memory management */
} hypre_IJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_IJMatrixContext(matrix)              ((matrix) -> context)
#define hypre_IJMatrixM(matrix)                    ((matrix) -> M)
#define hypre_IJMatrixN(matrix)                    ((matrix) -> N)

#define hypre_IJMatrixLocalStorageType(matrix)     ((matrix) -> local_storage_type)
#define hypre_IJMatrixTranslator(matrix)           ((matrix) -> translator)
#define hypre_IJMatrixLocalStorage(matrix)         ((matrix) -> local_storage)

#define hypre_IJMatrixInsertionSemantics(matrix)   ((matrix) -> insertion_semantics)
#define hypre_IJMatrixReferenceCount(matrix)       ((matrix) -> ref_count)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/

#ifdef PETSC_AVAILABLE
/* IJMatrix_petsc.c */
int
hypre_GetIJMatrixParCSRMatrix( HYPRE_IJMatrix IJmatrix, Mat *reference )
#endif
  
#ifdef ISIS_AVAILABLE
/* IJMatrix_isis.c */
int
hypre_GetIJMatrixISISMatrix( HYPRE_IJMatrix IJmatrix, RowMatrix *reference )
#endif

#endif
/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_IJMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_IJ_VECTOR_HEADER
#define hypre_IJ_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_IJVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm      context;

   int N;                                  /* number of rows in column vector */


   void         *local_storage;            /* Structure for storing local portio
n */
   int           local_storage_type;       /* Indicates the type of "local stora
ge" */
   int           ref_count;                /* reference count for memory managem
ent */
} hypre_IJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_IJVector
 *--------------------------------------------------------------------------*/

#define hypre_IJVectorContext(vector)              ((vector) -> context)
#define hypre_IJVectorN(vector)                    ((vector) -> N)

#define hypre_IJVectorLocalStorageType(vector)     ((vector) -> local_storage_type)
#define hypre_IJVectorLocalStorage(vector)         ((vector) -> local_storage)

#define hypre_IJVectorReferenceCount(vector)       ((vector) -> ref_count)

/*--------------------------------------------------------------------------
 * prototypes for operations on local objects
 *--------------------------------------------------------------------------*/
/* #include "./internal_protos.h" */

#endif

/* aux_parcsr_matrix.c */
int hypre_AuxParCSRMatrixCreate( hypre_AuxParCSRMatrix **aux_matrix , int local_num_rows , int local_num_cols , int *sizes );
int hypre_AuxParCSRMatrixDestroy( hypre_AuxParCSRMatrix *matrix );
int hypre_AuxParCSRMatrixInitialize( hypre_AuxParCSRMatrix *matrix );


/* hypre_IJMatrix_isis.c */
int hypre_IJMatrixSetLocalSizeISIS( hypre_IJMatrix *matrix , int local_m , int local_n );
int hypre_IJMatrixCreateISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetDiagRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetOffDiagRowSizesISIS( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixInitializeISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixInsertBlockISIS( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixAddToBlockISIS( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixInsertRowISIS( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAddToRowISIS( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAssembleISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixDistributeISIS( hypre_IJMatrix *matrix , int *row_starts , int *col_starts );
int hypre_IJMatrixApplyISIS( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
int hypre_IJMatrixDestroyISIS( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetTotalSizeISIS( hypre_IJMatrix *matrix , int size );

/* hypre_IJMatrix_parcsr.c */
int hypre_IJMatrixSetLocalSizeParCSR( hypre_IJMatrix *matrix , int local_m , int local_n );
int hypre_IJMatrixCreateParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesParCSR( hypre_IJMatrix *matrix , const int *sizes );
int hypre_IJMatrixSetDiagRowSizesParCSR( hypre_IJMatrix *matrix , const int *sizes );
int hypre_IJMatrixSetOffDiagRowSizesParCSR( hypre_IJMatrix *matrix , const int *sizes );
int hypre_IJMatrixInitializeParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixInsertRowParCSR( hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *coeffs );
int hypre_IJMatrixInsertBlockParCSR( hypre_IJMatrix *matrix , int m , int n , const int *rows , const int *cols , const double *coeffs );
int hypre_IJMatrixAddToRowParCSR( hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *coeffs );
int hypre_IJMatrixAddToBlockParCSR( hypre_IJMatrix *matrix , int m , int n , const int *rows , const int *cols , const double *coeffs );
int hypre_IJMatrixAddToRowAfterParCSR( hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *coeffs );
int hypre_IJMatrixSetValuesParCSR( hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *values , int add_to );
int hypre_IJMatrixSetBlockValuesParCSR( hypre_IJMatrix *matrix , int m , int n , const int *rows , const int *cols , const double *values , int add_to );
int hypre_IJMatrixAssembleParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixDistributeParCSR( hypre_IJMatrix *matrix , const int *row_starts , const int *col_starts );
int hypre_IJMatrixApplyParCSR( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
int hypre_IJMatrixDestroyParCSR( hypre_IJMatrix *matrix );
int hypre_IJMatrixGetRowPartitioningParCSR( hypre_IJMatrix *matrix , const int **row_partitioning );
int hypre_IJMatrixGetColPartitioningParCSR( hypre_IJMatrix *matrix , const int **col_partitioning );

/* hypre_IJMatrix_petsc.c */
int hypre_IJMatrixSetLocalSizePETSc( hypre_IJMatrix *matrix , int local_m , int local_n );
int hypre_IJMatrixCreatePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetDiagRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixSetOffDiagRowSizesPETSc( hypre_IJMatrix *matrix , int *sizes );
int hypre_IJMatrixInitializePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixInsertBlockPETSc( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixAddToBlockPETSc( hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs );
int hypre_IJMatrixInsertRowPETSc( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAddToRowPETSc( hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs );
int hypre_IJMatrixAssemblePETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixDistributePETSc( hypre_IJMatrix *matrix , int *row_starts , int *col_starts );
int hypre_IJMatrixApplyPETSc( hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b );
int hypre_IJMatrixDestroyPETSc( hypre_IJMatrix *matrix );
int hypre_IJMatrixSetTotalSizePETSc( hypre_IJMatrix *matrix , int size );

/* hypre_IJVector_parcsr.c */
int hypre_IJVectorCreatePar( hypre_IJVector *vector , const int *partitioning );
int hypre_IJVectorDestroyPar( hypre_IJVector *vector );
int hypre_IJVectorSetPartitioningPar( hypre_IJVector *vector , const int *partitioning );
int hypre_IJVectorSetLocalPartitioningPar( hypre_IJVector *vector , int vec_start_this_proc , int vec_start_next_proc );
int hypre_IJVectorInitializePar( hypre_IJVector *vector );
int hypre_IJVectorDistributePar( hypre_IJVector *vector , const int *vec_starts );
int hypre_IJVectorZeroLocalComponentsPar( hypre_IJVector *vector );
int hypre_IJVectorSetLocalComponentsPar( hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int hypre_IJVectorSetLocalComponentsInBlockPar( hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int hypre_IJVectorAddToLocalComponentsPar( hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int hypre_IJVectorAddToLocalComponentsInBlockPar( hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int hypre_IJVectorAssemblePar( hypre_IJVector *vector );
int hypre_IJVectorGetLocalComponentsPar( hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values );
int hypre_IJVectorGetLocalComponentsInBlockPar( hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values );


/* HYPRE_IJMatrix.c */
int HYPRE_IJMatrixCreate( MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n );
int HYPRE_IJMatrixDestroy( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixInitialize( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixAssemble( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixDistribute( HYPRE_IJMatrix IJmatrix , const int *row_starts , const int *col_starts );
int HYPRE_IJMatrixSetLocalStorageType( HYPRE_IJMatrix IJmatrix , int type );
int HYPRE_IJMatrixSetLocalSize( HYPRE_IJMatrix IJmatrix , int local_m , int local_n );
int HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixQueryInsertionSemantics( HYPRE_IJMatrix IJmatrix , int *level );
int HYPRE_IJMatrixInsertBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAddToBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToRowAfter( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixSetValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixSetBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAddToBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference );
void *HYPRE_IJMatrixGetLocalStorage( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixGetRowPartitioning( HYPRE_IJMatrix IJmatrix , const int **row_partitioning );
int HYPRE_IJMatrixGetColPartitioning( HYPRE_IJMatrix IJmatrix , const int **col_partitioning );

/* HYPRE_IJVector.c */
int HYPRE_IJVectorCreate( MPI_Comm comm , HYPRE_IJVector *in_vector_ptr , int global_n );
int HYPRE_IJVectorDestroy( HYPRE_IJVector IJvector );
int HYPRE_IJVectorSetPartitioning( HYPRE_IJVector IJvector , const int *partitioning );
int HYPRE_IJVectorSetLocalPartitioning( HYPRE_IJVector IJvector , int vec_start_this_proc , int vec_start_next_proc );
int HYPRE_IJVectorInitialize( HYPRE_IJVector IJvector );
int HYPRE_IJVectorDistribute( HYPRE_IJVector IJvector , const int *vec_starts );
int HYPRE_IJVectorSetLocalStorageType( HYPRE_IJVector IJvector , int type );
int HYPRE_IJVectorZeroLocalComponents( HYPRE_IJVector IJvector );
int HYPRE_IJVectorSetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int HYPRE_IJVectorSetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int HYPRE_IJVectorAddToLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int HYPRE_IJVectorAddToLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int HYPRE_IJVectorAssemble( HYPRE_IJVector IJvector );
int HYPRE_IJVectorGetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values );
int HYPRE_IJVectorGetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values );
int HYPRE_IJVectorGetLocalStorageType( HYPRE_IJVector IJvector , int *type );
void *HYPRE_IJVectorGetLocalStorage( HYPRE_IJVector IJvector );
int hypre_RefIJVector( HYPRE_IJVector IJvector , HYPRE_IJVector *reference );


#ifdef __cplusplus
}
#endif

#endif

