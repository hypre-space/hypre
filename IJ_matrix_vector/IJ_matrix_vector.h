
#include "HYPRE_IJMatrix.h"

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"

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
   int     local_num_rows;   /* defines number of rows on this processors */
   int     local_num_cols;   /* defines number of cols of diag */

   int	   diag_size;	    /* size of aux_diag_j or aux_diag_data */
   int	   indx_diag;	   /* first empty element of aux_diag_j(data) */
   int    *row_start_diag; /* row_start_diag[i] points to first element 
				of i-th row */
   int    *row_end_diag; /* row_end_diag[i] points to last element 
				of i-th row */
   int    *aux_diag_j;	/* contains collected column indices */
   double *aux_diag_data; /* contains collected data */

   int	   offd_size; /* size of aux_offd_j or aux_offd_data */
   int	   indx_offd; /* first empty element of aux_offd_j(data) */
   int    *row_start_offd;  /* row_start_offd[i] points to first element 
                                of i-th row */
   int    *row_end_offd;  /* row_end_offd[i] points to last element 
                                of i-th row */
   int    *aux_offd_j;  /* contains collected column indices */
   double *aux_offd_data;  /* contains collected data */

} hypre_AuxParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParCSRMatrixLocalNumRows(matrix)  ((matrix) -> local_num_rows)
#define hypre_AuxParCSRMatrixLocalNumCols(matrix)  ((matrix) -> local_num_cols)

#define hypre_AuxParCSRMatrixDiagSize(matrix)      ((matrix) -> diag_size)
#define hypre_AuxParCSRMatrixIndxDiag(matrix)      ((matrix) -> indx_diag)
#define hypre_AuxParCSRMatrixRowStartDiag(matrix)  ((matrix) -> row_start_diag)
#define hypre_AuxParCSRMatrixRowEndDiag(matrix)    ((matrix) -> row_end_diag)
#define hypre_AuxParCSRMatrixAuxDiagJ(matrix)  	   ((matrix) -> aux_diag_j)
#define hypre_AuxParCSRMatrixAuxDiagData(matrix)   ((matrix) -> aux_diag_data)

#define hypre_AuxParCSRMatrixOffdSize(matrix)      ((matrix) -> offd_size)
#define hypre_AuxParCSRMatrixIndxOffd(matrix)      ((matrix) -> indx_offd)
#define hypre_AuxParCSRMatrixRowStartOffd(matrix)  ((matrix) -> row_start_offd)
#define hypre_AuxParCSRMatrixRowEndOffd(matrix)    ((matrix) -> row_end_offd)
#define hypre_AuxParCSRMatrixAuxOffdJ(matrix)  	   ((matrix) -> aux_offd_j)
#define hypre_AuxParCSRMatrixAuxOffdData(matrix)   ((matrix) -> aux_offd_data)

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
# define	P(s) s

/* HYPRE_IJMatrix.c */
int HYPRE_NewIJMatrix P((HYPRE_IJMatrix *in_matrix_ptr , MPI_Comm comm , int global_m , int global_n ));
int HYPRE_FreeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_InitializeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_AssembleIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_DistributeIJMatrix P((HYPRE_IJMatrix IJmatrix , int *row_starts , int *col_starts ));
int HYPRE_SetIJMatrixLocalStorageType P((HYPRE_IJMatrix IJmatrix , int type ));
int HYPRE_SetIJMatrixLocalSize P((HYPRE_IJMatrix IJmatrix , int local_m , int local_n ));
int HYPRE_SetIJMatrixDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixOffDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixTotalSize P((HYPRE_IJMatrix IJmatrix , int size ));
int HYPRE_QueryIJMatrixInsertionSemantics P((HYPRE_IJMatrix IJmatrix , int *level ));
int HYPRE_InsertIJMatrixBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_AddBlockToIJMatrix P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_InsertIJMatrixRow P((HYPRE_IJMatrix IJmatrix , int n , int row , int *cols , double *values ));


/* aux_parcsr_matrix.c */
hypre_AuxParCSRMatrix *hypre_CreateAuxParCSRMatrix P((int local_num_rows , int local_num_cols , int diag_size , int offd_size ));
int hypre_DestroyAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));
int hypre_InitializeAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));

/* hypre_IJMatrix_parcsr.c */
int hypre_SetIJMatrixLocalSizeParcsr P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_NewIJMatrixParcsr P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixDiagRowSizesParcsr P((hypre_IJMatrix *matrix , int *sizes ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

