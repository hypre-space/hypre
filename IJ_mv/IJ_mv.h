
#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include "../utilities/utilities.h"
#include "../seq_matrix_vector/seq_matrix_vector.h"
#include "../parcsr_matrix_vector/parcsr_matrix_vector.h"
#include "./HYPRE_IJ_mv.h"

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

#include "../utilities/general.h"
#include "../utilities/utilities.h"

#include "../HYPRE.h"

/* #include "./HYPRE_IJ_matrix_types.h" */

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
/* #include "./internal_protos.h" */

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

#include "../utilities/general.h"
#include "../utilities/utilities.h"

#include "../HYPRE.h"

/* #include "./HYPRE_IJ_vector_types.h" */

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

/* F90_HYPRE_IJMatrix.c */
void hypre_F90_IFACE P((int hypre_newijmatrix ));
void hypre_F90_IFACE P((int hypre_freeijmatrix ));
void hypre_F90_IFACE P((int hypre_initializeijmatrix ));
void hypre_F90_IFACE P((int hypre_assembleijmatrix ));
void hypre_F90_IFACE P((int hypre_distributeijmatrix ));
void hypre_F90_IFACE P((int hypre_setijmatrixlocalstoragety ));
void hypre_F90_IFACE P((int hypre_setijmatrixlocalsize ));
void hypre_F90_IFACE P((int hypre_setijmatrixrowsizes ));
void hypre_F90_IFACE P((int hypre_setijmatrixdiagrowsizes ));
void hypre_F90_IFACE P((int hypre_setijmatrixoffdiagrowsize ));
void hypre_F90_IFACE P((int hypre_setijmatrixtotalsize ));
void hypre_F90_IFACE P((int hypre_queryijmatrixinsertionsem ));
void hypre_F90_IFACE P((int hypre_insertijmatrixblock ));
void hypre_F90_IFACE P((int hypre_addblocktoijmatrix ));
void hypre_F90_IFACE P((int hypre_insertijmatrixrow ));

/* F90_HYPRE_IJVector.c */
void hypre_F90_IFACE P((int hypre_newijvector ));
void hypre_F90_IFACE P((int hypre_freeijvector ));
void hypre_F90_IFACE P((int hypre_setijvectorpartitioning ));
void hypre_F90_IFACE P((int hypre_setijvectorlocalpartition ));
void hypre_F90_IFACE P((int hypre_initializeijvector ));
void hypre_F90_IFACE P((int hypre_distributeijvector ));
void hypre_F90_IFACE P((int hypre_setijvectorlocalstoragety ));
void hypre_F90_IFACE P((int hypre_setijvectorlocalsize ));
void hypre_F90_IFACE P((int hypre_zeroijveclocalcomps ));
void hypre_F90_IFACE P((int hypre_setijveclocalcomps ));
void hypre_F90_IFACE P((int hypre_setijveclocalcompsinblock ));
void hypre_F90_IFACE P((int hypre_addtoijveclocalcomps ));
void hypre_F90_IFACE P((int hypre_addtoijveclocalcompsinblo ));
void hypre_F90_IFACE P((int hypre_getijveclocalcomps ));
void hypre_F90_IFACE P((int hypre_getijveclocalcompsinblock ));
void hypre_F90_IFACE P((int hypre_getijveclocalstoragetype ));
void hypre_F90_IFACE P((int hypre_getijveclocalstorage ));

/* HYPRE_IJMatrix.c */
int HYPRE_NewIJMatrix P((MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n ));
int HYPRE_FreeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_InitializeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_AssembleIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_DistributeIJMatrix P((HYPRE_IJMatrix IJmatrix , int *row_starts , int *col_starts ));
int HYPRE_SetIJMatrixLocalStorageType P((HYPRE_IJMatrix IJmatrix , int type ));
int HYPRE_SetIJMatrixLocalSize P((HYPRE_IJMatrix IJmatrix , int local_m , int local_n ));
int HYPRE_SetIJMatrixRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixOffDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixTotalSize P((HYPRE_IJMatrix IJmatrix , int size ));
int HYPRE_QueryIJMatrixInsertionSemantics P((HYPRE_IJMatrix IJmatrix , int *level ));
int HYPRE_InsertIJMatrixBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_AddBlockToIJMatrix P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_InsertIJMatrixRow P((HYPRE_IJMatrix IJmatrix , int n , int row , int *cols , double *values ));
int hypre_RefIJMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference ));
void *HYPRE_GetIJMatrixLocalStorage P((HYPRE_IJMatrix IJmatrix ));

/* HYPRE_IJVector.c */
int HYPRE_NewIJVector P((MPI_Comm comm , HYPRE_IJVector *in_vector_ptr , int global_n ));
int HYPRE_FreeIJVector P((HYPRE_IJVector IJvector ));
int HYPRE_SetIJVectorPartitioning P((HYPRE_IJVector IJvector , const int *partitioning ));
int HYPRE_SetIJVectorLocalPartitioning P((HYPRE_IJVector IJvector , int vec_start , int vec_stop ));
int HYPRE_InitializeIJVector P((HYPRE_IJVector IJvector ));
int HYPRE_DistributeIJVector P((HYPRE_IJVector IJvector , const int *vec_starts ));
int HYPRE_SetIJVectorLocalStorageType P((HYPRE_IJVector IJvector , int type ));
int HYPRE_ZeroIJVectorLocalComponents P((HYPRE_IJVector IJvector ));
int HYPRE_SetIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_SetIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int HYPRE_AddToIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_AddToIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int HYPRE_GetIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values ));
int HYPRE_GetIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values ));
int HYPRE_GetIJVectorLocalStorageType P((HYPRE_IJVector IJvector , int *type ));
void *HYPRE_GetIJVectorLocalStorage P((HYPRE_IJVector IJvector ));
int hypre_RefIJVector P((HYPRE_IJVector IJvector , HYPRE_IJVector *reference ));

/* IJMatrix_isis.c */
int hypre_GetIJMatrixParCSRMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_ParCSRMatrix *reference ));

/* IJMatrix_parcsr.c */
int hypre_GetIJMatrixParCSRMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_ParCSRMatrix *reference ));

/* IJMatrix_petsc.c */
int hypre_GetIJMatrixParCSRMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_ParCSRMatrix *reference ));

/* IJVector_parcsr.c */
int hypre_GetIJVectorParVector P((HYPRE_IJVector IJvector , HYPRE_ParVector *reference ));

/* IJ_par_laplace_9pt.c */
int IJMatrixBuildParLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr , HYPRE_IJMatrix **ij_matrix , int ij_matrix_storage_type ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part ));

/* aux_parcsr_matrix.c */
hypre_AuxParCSRMatrix *hypre_CreateAuxParCSRMatrix P((int local_num_rows , int local_num_cols , int *sizes ));
int hypre_DestroyAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));
int hypre_InitializeAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));

/* hypre_IJMatrix_isis.c */
int hypre_SetIJMatrixLocalSizeISIS P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_NewIJMatrixISIS P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixDiagRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixOffDiagRowSizesISIS P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_InitializeIJMatrixISIS P((hypre_IJMatrix *matrix ));
int hypre_InsertIJMatrixBlockISIS P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_AddBlockToIJMatrixISIS P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_InsertIJMatrixRowISIS P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AddIJMatrixRowISIS P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AssembleIJMatrixISIS P((hypre_IJMatrix *matrix ));
int hypre_DistributeIJMatrixISIS P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_ApplyIJMatrixISIS P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_FreeIJMatrixISIS P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixTotalSizeISIS P((hypre_IJMatrix *matrix , int size ));

/* hypre_IJMatrix_parcsr.c */
int hypre_SetIJMatrixLocalSizeParCSR P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_NewIJMatrixParCSR P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixRowSizesParCSR P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixDiagRowSizesParCSR P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixOffDiagRowSizesParCSR P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_InitializeIJMatrixParCSR P((hypre_IJMatrix *matrix ));
int hypre_InsertIJMatrixBlockParCSR P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_AddBlockToIJMatrixParCSR P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_InsertIJMatrixRowParCSR P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AddIJMatrixRowParCSR P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AssembleIJMatrixParCSR P((hypre_IJMatrix *matrix ));
int hypre_DistributeIJMatrixParCSR P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_ApplyIJMatrixParCSR P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_FreeIJMatrixParCSR P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixTotalSizeParCSR P((hypre_IJMatrix *matrix , int size ));

/* hypre_IJMatrix_petsc.c */
int hypre_SetIJMatrixLocalSizePETSc P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_NewIJMatrixPETSc P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixDiagRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_SetIJMatrixOffDiagRowSizesPETSc P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_InitializeIJMatrixPETSc P((hypre_IJMatrix *matrix ));
int hypre_InsertIJMatrixBlockPETSc P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_AddBlockToIJMatrixPETSc P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_InsertIJMatrixRowPETSc P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AddIJMatrixRowPETSc P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AssembleIJMatrixPETSc P((hypre_IJMatrix *matrix ));
int hypre_DistributeIJMatrixPETSc P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_ApplyIJMatrixPETSc P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_FreeIJMatrixPETSc P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixTotalSizePETSc P((hypre_IJMatrix *matrix , int size ));

/* hypre_IJVector_parcsr.c */
int hypre_NewIJVectorPar P((hypre_IJVector *vector ));
int hypre_FreeIJVectorPar P((hypre_IJVector *vector ));
int hypre_SetIJVectorParPartitioning P((hypre_IJVector *vector , const int *partitioning ));
int hypre_SetIJVectorParLocalPartitioning P((hypre_IJVector *vector , int vec_start , int vec_stop ));
int hypre_InitializeIJVectorPar P((hypre_IJVector *vector ));
int hypre_DistributeIJVectorPar P((hypre_IJVector *vector , const int *vec_starts ));
int hypre_ZeroIJVectorParLocalComponents P((hypre_IJVector *vector ));
int hypre_SetIJVectorParLocalComponents P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int hypre_SetIJVectorParLocalComponentsInBlock P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int hypre_AddToIJVectorParLocalComponents P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int hypre_AddToIJVectorParLocalComponentsInBlock P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));
int hypre_GetIJVectorParLocalComponents P((hypre_IJVector *vector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values ));
int hypre_GetIJVectorParLocalComponentsInBlock P((hypre_IJVector *vector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values ));

/* qsort.c */
void swap P((int *v , int i , int j ));
void swap2 P((int *v , double *w , int i , int j ));
void qsort0 P((int *v , int left , int right ));
void qsort1 P((int *v , double *w , int left , int right ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

