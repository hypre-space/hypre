
#include "HYPRE_IJMatrix.h"

#ifndef hypre_IJ_HEADER
#define hypre_IJ_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"
#include "parcsr_linear_solvers.h"

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
<<<<<<< IJ_matrix_vector.h
 * $Revision$
=======
 * $Revision$
>>>>>>> 1.4
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

/* F90_HYPRE_IJMatrix.c */
void hypre_F90_IFACE P((int hypre_newijmatrix ));
void hypre_F90_IFACE P((int hypre_freeijmatrix ));
void hypre_F90_IFACE P((int hypre_initializeijmatrix ));
void hypre_F90_IFACE P((int hypre_assembleijmatrix ));
void hypre_F90_IFACE P((int hypre_distributeijmatrix ));
void hypre_F90_IFACE P((int hypre_setijmatrixlocalstoragety ));
void hypre_F90_IFACE P((int hypre_setijmatrixlocalsize ));
void hypre_F90_IFACE P((int hypre_setijmatrixdiagrowsizes ));
void hypre_F90_IFACE P((int hypre_setijmatrixoffdiagrowsize ));
void hypre_F90_IFACE P((int hypre_setijmatrixtotalsize ));
void hypre_F90_IFACE P((int hypre_queryijmatrixinsertionsem ));
void hypre_F90_IFACE P((int hypre_insertijmatrixblock ));
void hypre_F90_IFACE P((int hypre_addblocktoijmatrix ));
void hypre_F90_IFACE P((int hypre_insertijmatrixrow ));

/* HYPRE_IJMatrix.c */
int HYPRE_NewIJMatrix P((HYPRE_IJMatrix *in_matrix_ptr , MPI_Comm comm , int global_m , int global_n ));
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
int hypre_GetIJMatrixLocalStorage P((HYPRE_IJMatrix IJmatrix , void **local_storage ));

/* IJ_par_laplace_9pt.c */
int IJMatrixBuildParLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr , HYPRE_IJMatrix **ij_matrix , int ij_matrix_storage_type ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part ));

/* aux_parcsr_matrix.c */
hypre_AuxParCSRMatrix *hypre_CreateAuxParCSRMatrix P((int local_num_rows , int local_num_cols , int *sizes ));
int hypre_DestroyAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));
int hypre_InitializeAuxParCSRMatrix P((hypre_AuxParCSRMatrix *matrix ));

/* driver.c */
int main P((int argc , char *argv []));
int BuildParFromFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParDifConv P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParFromOneFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildRhsParFromOneFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix *A , hypre_ParVector **b_ptr ));
int BuildParLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian27pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));

/* hypre_IJMatrix_parcsr.c */
int hypre_SetIJMatrixLocalSizeParcsr P((hypre_IJMatrix *matrix , int local_m , int local_n ));
int hypre_NewIJMatrixParcsr P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixRowSizesParcsr P((hypre_IJMatrix *matrix , int *sizes ));
int hypre_InitializeIJMatrixParcsr P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixBlockParcsr P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_AddBlockToIJMatrixParcsr P((hypre_IJMatrix *matrix , int m , int n , int *rows , int *cols , double *coeffs ));
int hypre_InsertIJMatrixRowParcsr P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AddIJMatrixRowParcsr P((hypre_IJMatrix *matrix , int n , int row , int *indices , double *coeffs ));
int hypre_AssembleIJMatrixParcsr P((hypre_IJMatrix *matrix ));
int hypre_DistributeIJMatrixParcsr P((hypre_IJMatrix *matrix , int *row_starts , int *col_starts ));
int hypre_ApplyIJMatrixParcsr P((hypre_IJMatrix *matrix , hypre_ParVector *x , hypre_ParVector *b ));
int hypre_FreeIJMatrixParcsr P((hypre_IJMatrix *matrix ));
int hypre_SetIJMatrixTotalSizeParcsr P((hypre_IJMatrix *matrix , int size ));

/* qsort.c */
void qsort0 P((int *v , int left , int right ));
void qsort1 P((int *v , double *w , int left , int right ));
void swap P((int *v , int i , int j ));
void swap2 P((int *v , double *w , int i , int j ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

