
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
 * Fortran <-> C interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_FORTRAN_HEADER
#define HYPRE_FORTRAN_HEADER

#if defined(IRIX) || defined(DEC)
#define hypre_NAME_C_FOR_FORTRAN(name) name##_
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#else
#define hypre_NAME_C_FOR_FORTRAN(name) name##__
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#endif

#define hypre_F90_IFACE(iface_name) hypre_NAME_FORTRAN_FOR_C(iface_name)

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
# define	P(s) s

/* aux_parcsr_matrix.c */
hypre_AuxParCSRMatrix *hypre_AuxParCSRMatrixCreate P((int local_num_rows , int local_num_cols , int *sizes ));
int hypre_AuxParCSRMatrixDestroy P((hypre_AuxParCSRMatrix *matrix ));
int hypre_AuxParCSRMatrixInitialize P((hypre_AuxParCSRMatrix *matrix ));

#undef P
# define	P(s) s

/* hypre_IJMatrix_isis.c */
int hypre_IJMatrixSetLocalSizeISIS P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_IJMatrixCreateISIS P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixSetRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixSetDiagRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixSetOffDiagRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixInitializeISIS P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixInsertBlockISIS P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_IJMatrixAddBlockISIS P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_IJMatrixInsertRowISIS P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_IJMatrixAddRowISIS P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_IJMatrixAssembleISIS P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixDistributeISIS P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_IJMatrixApplyISIS P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_IJMatrixDestroyISIS P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixSetTotalSizeISIS P((hypre_IJMatrix *matrix , int size ));

/* hypre_IJMatrix_parcsr.c */
int hypre_IJMatrixSetLocalSizeParCSR P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_IJMatrixCreateParCSR P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixSetRowSizesParCSR P((hypre_IJMatrix *matrix , const int *sizes ));
int hypre_IJMatrixSetDiagRowSizesParCSR P((hypre_IJMatrix *matrix , const int *sizes ));
int hypre_IJMatrixSetOffDiagRowSizesParCSR P((hypre_IJMatrix *matrix , const int *sizes ));
int hypre_IJMatrixInitializeParCSR P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixInsertBlockParCSR P((hypre_IJMatrix *matrix , int m , int n , const int *rows , const int *cols , const double *coeffs ));
int hypre_IJMatrixAddBlockParCSR P((hypre_IJMatrix *matrix , int m , int n , const int *rows , const int *cols , const double *coeffs ));
int hypre_IJMatrixInsertRowParCSR P((hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *coeffs ));
int hypre_IJMatrixAddRowParCSR P((hypre_IJMatrix *matrix , int n , int row , const int *indices , const double *coeffs ));
int hypre_IJMatrixAssembleParCSR P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixDistributeParCSR P((hypre_IJMatrix *matrix , const int *row_starts , const int *col_starts ));
int hypre_IJMatrixApplyParCSR P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_IJMatrixDestroyParCSR P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixGetRowPartitioningParCSR P((hypre_IJMatrix *matrix , const int **row_partitioning ));
int hypre_IJMatrixGetColPartitioningParCSR P((hypre_IJMatrix *matrix , const int **col_partitioning ));

/* hypre_IJMatrix_petsc.c */
int hypre_IJMatrixSetLocalSizePETSc P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_IJMatrixCreatePETSc P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixSetRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixSetDiagRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixSetOffDiagRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_IJMatrixInitializePETSc P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixInsertBlockPETSc P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_IJMatrixAddBlockPETSc P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_IJMatrixInsertRowPETSc P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_IJMatrixAddRowPETSc P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_IJMatrixAssemblePETSc P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixDistributePETSc P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_IJMatrixApplyPETSc P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_IJMatrixDestroyPETSc P((hypre_IJMatrix *matrix ));
int hypre_IJMatrixSetTotalSizePETSc P((hypre_IJMatrix *matrix , int size ));

/* hypre_IJVector_parcsr.c */
int hypre_IJVectorCreatePar P((hypre_IJVector *vector , const int *partitioning ));
int hypre_IJVectorDestroyPar P((hypre_IJVector *vector ));
int hypre_IJVectorSetPartitioningPar P((hypre_IJVector *vector , const int *partitioning ));
int hypre_IJVectorSetLocalPartitioningPar P((hypre_IJVector *vector , int vec_start_this_proc , int vec_start_next_proc ));
int hypre_IJVectorInitializePar P((hypre_IJVector *vector ));
int hypre_IJVectorDistributePar P((hypre_IJVector *vector , const int *vec_starts ));
int hypre_IJVectorZeroLocalComponentsPar P((hypre_IJVector *vector ));
int hypre_IJVectorSetLocalComponentsPar P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int hypre_IJVectorSetLocalComponentsInBlockPar P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int hypre_IJVectorAddLocalComponentsPar P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int hypre_IJVectorAddLocalComponentsInBlockPar P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int hypre_IJVectorAssemblePar P((hypre_IJVector *vector ));
int hypre_IJVectorGetLocalComponentsPar P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values ));
int hypre_IJVectorGetLocalComponentsInBlockPar P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values ));

#undef P
# define	P(s) s
# define	P(s) s

/* HYPRE_IJMatrix.c */
int HYPRE_IJMatrixCreate P((MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n ));
int HYPRE_IJMatrixDestroy P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixInitialize P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixAssemble P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixDistribute P((HYPRE_IJMatrix IJmatrix , const int *row_starts , const int *col_starts ));
int HYPRE_IJMatrixSetLocalStorageType P((HYPRE_IJMatrix IJmatrix , int type ));
int HYPRE_IJMatrixSetLocalSize P((HYPRE_IJMatrix IJmatrix , int local_m , int local_n ));
int HYPRE_IJMatrixSetRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixSetDiagRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixSetOffDiagRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixQueryInsertionSemantics P((HYPRE_IJMatrix IJmatrix , int *level ));
int HYPRE_IJMatrixInsertBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values ));
int HYPRE_IJMatrixAddBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values ));
int HYPRE_IJMatrixInsertRow P((HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values ));
int HYPRE_IJMatrixAddRow P((HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values ));
int hypre_RefIJMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference ));
void *HYPRE_IJMatrixGetLocalStorage P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixGetRowPartitioning P((HYPRE_IJMatrix IJmatrix , const int **row_partitioning ));
int HYPRE_IJMatrixGetColPartitioning P((HYPRE_IJMatrix IJmatrix , const int **col_partitioning ));

/* HYPRE_IJVector.c */
int HYPRE_IJVectorCreate P((MPI_Comm comm , HYPRE_IJVector *in_vector_ptr , int global_n ));
int HYPRE_IJVectorDestroy P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorSetPartitioning P((HYPRE_IJVector IJvector , const int *partitioning ));
int HYPRE_IJVectorSetLocalPartitioning P((HYPRE_IJVector IJvector , int vec_start_this_proc , int vec_start_next_proc ));
int HYPRE_IJVectorInitialize P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorDistribute P((HYPRE_IJVector IJvector , const int *vec_starts ));
int HYPRE_IJVectorSetLocalStorageType P((HYPRE_IJVector IJvector , int type ));
int HYPRE_IJVectorZeroLocalComponents P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorSetLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_IJVectorSetLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int HYPRE_IJVectorAddLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_IJVectorAddLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int HYPRE_IJVectorAssemble P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorGetLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values ));
int HYPRE_IJVectorGetLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values ));
int HYPRE_IJVectorGetLocalStorageType P((HYPRE_IJVector IJvector , int *type ));
void *HYPRE_IJVectorGetLocalStorage P((HYPRE_IJVector IJvector ));
int hypre_RefIJVector P((HYPRE_IJVector IJvector , HYPRE_IJVector *reference ));

#undef P
# define	P(s) s

/* F90_HYPRE_IJMatrix.c */
void hypre_F90_IFACE P((int hypre_ijmatrixcreate ));
void hypre_F90_IFACE P((int hypre_ijmatrixdestroy ));
void hypre_F90_IFACE P((int hypre_ijmatrixinitialize ));
void hypre_F90_IFACE P((int hypre_ijmatrixassemble ));
void hypre_F90_IFACE P((int hypre_ijmatrixdistribute ));
void hypre_F90_IFACE P((int hypre_ijmatrixsetlocalstoragety ));
void hypre_F90_IFACE P((int hypre_ijmatrixsetlocalsize ));
void hypre_F90_IFACE P((int hypre_ijmatrixsetrowsizes ));
void hypre_F90_IFACE P((int hypre_ijmatrixsetdiagrowsizes ));
void hypre_F90_IFACE P((int hypre_ijmatrixsetoffdiagrowsize ));
void hypre_F90_IFACE P((int hypre_ijmatrixqueryinsertionsem ));
void hypre_F90_IFACE P((int hypre_ijmatrixinsertblock ));
void hypre_F90_IFACE P((int hypre_ijmatrixaddblock ));
void hypre_F90_IFACE P((int hypre_ijmatrixinsertrow ));
void hypre_F90_IFACE P((int hypre_ijmatrixgetlocalstorage ));

/* F90_HYPRE_IJVector.c */
void hypre_F90_IFACE P((int hypre_ijvectorcreate ));
void hypre_F90_IFACE P((int hypre_ijvectordestroy ));
void hypre_F90_IFACE P((int hypre_ijvectorsetpartitioning ));
void hypre_F90_IFACE P((int hypre_ijvectorsetlocalpartition ));
void hypre_F90_IFACE P((int hypre_ijvectorinitialize ));
void hypre_F90_IFACE P((int hypre_ijvectordistribute ));
void hypre_F90_IFACE P((int hypre_ijvectorsetlocalstoragety ));
void hypre_F90_IFACE P((int hypre_ijvectorzerolocalcomps ));
void hypre_F90_IFACE P((int hypre_ijvectorsetlocalcomps ));
void hypre_F90_IFACE P((int hypre_ijvectorsetlocalcompsinbl ));
void hypre_F90_IFACE P((int hypre_ijvectoraddlocalcomps ));
void hypre_F90_IFACE P((int hypre_ijvectoraddlocalcompsinbl ));
void hypre_F90_IFACE P((int hypre_ijvectorassemble ));
void hypre_F90_IFACE P((int hypre_ijvectorgetlocalcomps ));
void hypre_F90_IFACE P((int hypre_ijvectorgetlocalcompsinbl ));
void hypre_F90_IFACE P((int hypre_ijvectorgetlocalstoragety ));
void hypre_F90_IFACE P((int hypre_ijvectorgetlocalstorage ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

