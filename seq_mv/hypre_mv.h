
#include "HYPRE_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

#include "hypre_utilities.h"

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
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *i;
   int     *j;
   int      num_rows;
   int      num_cols;
   int      num_nonzeros;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   int      owns_data;

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)

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
 * Header info for Mapped Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_MAPPED_MATRIX_HEADER
#define hypre_MAPPED_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Mapped Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   void               *matrix;
   int               (*ColMap)(int, void *);
   void               *MapData;

} hypre_MappedMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Mapped Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MappedMatrixMatrix(matrix)           ((matrix) -> matrix)
#define hypre_MappedMatrixColMap(matrix)           ((matrix) -> ColMap)
#define hypre_MappedMatrixMapData(matrix)           ((matrix) -> MapData)

#define hypre_MappedMatrixColIndex(matrix,j) \
         (hypre_MappedMatrixColMap(matrix)(j,hypre_MappedMatrixMapData(matrix)))

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
 * Header info for Multiblock Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_MULTIBLOCK_MATRIX_HEADER
#define hypre_MULTIBLOCK_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Multiblock Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                  num_submatrices;
   int                 *submatrix_types;
   void                **submatrices;

} hypre_MultiblockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multiblock Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MultiblockMatrixSubmatrices(matrix)        ((matrix) -> submatrices)
#define hypre_MultiblockMatrixNumSubmatrices(matrix)     ((matrix) -> num_submatrices)
#define hypre_MultiblockMatrixSubmatrixTypes(matrix)     ((matrix) -> submatrix_types)

#define hypre_MultiblockMatrixSubmatrix(matrix,j) (hypre_MultiblockMatrixSubmatrices\
(matrix)[j])
#define hypre_MultiblockMatrixSubmatrixType(matrix,j) (hypre_MultiblockMatrixSubmatrixTypes\
(matrix)[j])

#endif
/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_VECTOR_HEADER
#define hypre_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int      size;

   /* Does the Vector create/destroy `data'? */
   int      owns_data;

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)
#define hypre_VectorOwnsData(vector)  ((vector) -> owns_data)

#endif
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CreateCSRMatrix P((int num_rows , int num_cols , int *row_sizes ));
int HYPRE_DestroyCSRMatrix P((HYPRE_CSRMatrix matrix ));
int HYPRE_InitializeCSRMatrix P((HYPRE_CSRMatrix matrix ));
void HYPRE_PrintCSRMatrix P((HYPRE_CSRMatrix matrix , char *file_name ));

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_NewMappedMatrix P((void ));
int HYPRE_FreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_LimitedFreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_InitializeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_AssembleMappedMatrix P((HYPRE_MappedMatrix matrix ));
void HYPRE_PrintMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_GetMappedMatrixColIndex P((HYPRE_MappedMatrix matrix , int j ));
void *HYPRE_GetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_SetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix , void *matrix_data ));
int HYPRE_SetMappedMatrixColMap P((HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *)));
int HYPRE_SetMappedMatrixMapData P((HYPRE_MappedMatrix matrix , void *MapData ));

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_NewMultiblockMatrix P((void ));
int HYPRE_FreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_LimitedFreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_InitializeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_AssembleMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
void HYPRE_PrintMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_SetMultiblockMatrixNumSubmatrices P((HYPRE_MultiblockMatrix matrix , int n ));
int HYPRE_SetMultiblockMatrixSubmatrixType P((HYPRE_MultiblockMatrix matrix , int j , int type ));

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_CreateVector P((int size ));
int HYPRE_DestroyVector P((HYPRE_Vector vector ));
int HYPRE_InitializeVector P((HYPRE_Vector vector ));
int HYPRE_PrintVector P((HYPRE_Vector vector , char *file_name ));

/* csr_matop.c */
hypre_CSRMatrix *hypre_Matadd P((hypre_CSRMatrix *A , hypre_CSRMatrix *B ));
hypre_CSRMatrix *hypre_Matmul P((hypre_CSRMatrix *A , hypre_CSRMatrix *B ));
hypre_CSRMatrix *hypre_DeleteZerosInMatrix P((hypre_CSRMatrix *A , double tol ));

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CreateCSRMatrix P((int num_rows , int num_cols , int num_nonzeros ));
int hypre_DestroyCSRMatrix P((hypre_CSRMatrix *matrix ));
int hypre_InitializeCSRMatrix P((hypre_CSRMatrix *matrix ));
int hypre_SetCSRMatrixDataOwner P((hypre_CSRMatrix *matrix , int owns_data ));
hypre_CSRMatrix *hypre_ReadCSRMatrix P((char *file_name ));
int hypre_PrintCSRMatrix P((hypre_CSRMatrix *matrix , char *file_name ));

/* csr_matvec.c */
int hypre_Matvec P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));
int hypre_MatvecT P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_NewMappedMatrix P((void ));
int hypre_FreeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_LimitedFreeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_InitializeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_AssembleMappedMatrix P((hypre_MappedMatrix *matrix ));
void hypre_PrintMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_GetMappedMatrixColIndex P((hypre_MappedMatrix *matrix , int j ));
void *hypre_GetMappedMatrixMatrix P((hypre_MappedMatrix *matrix ));
int hypre_SetMappedMatrixMatrix P((hypre_MappedMatrix *matrix , void *matrix_data ));
int hypre_SetMappedMatrixColMap P((hypre_MappedMatrix *matrix , int (*ColMap )(int ,void *)));
int hypre_SetMappedMatrixMapData P((hypre_MappedMatrix *matrix , void *map_data ));

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_NewMultiblockMatrix P((void ));
int hypre_FreeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_LimitedFreeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_InitializeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_AssembleMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
void hypre_PrintMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_SetMultiblockMatrixNumSubmatrices P((hypre_MultiblockMatrix *matrix , int n ));
int hypre_SetMultiblockMatrixSubmatrixType P((hypre_MultiblockMatrix *matrix , int j , int type ));
int hypre_SetMultiblockMatrixSubmatrix P((hypre_MultiblockMatrix *matrix , int j , void *submatrix ));

/* vector.c */
hypre_Vector *hypre_CreateVector P((int size ));
int hypre_DestroyVector P((hypre_Vector *vector ));
int hypre_InitializeVector P((hypre_Vector *vector ));
int hypre_SetVectorDataOwner P((hypre_Vector *vector , int owns_data ));
hypre_Vector *hypre_ReadVector P((char *file_name ));
int hypre_PrintVector P((hypre_Vector *vector , char *file_name ));
int hypre_SetVectorConstantValues P((hypre_Vector *v , double value ));
int hypre_CopyVector P((hypre_Vector *x , hypre_Vector *y ));
int hypre_ScaleVector P((double alpha , hypre_Vector *y ));
int hypre_Axpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_InnerProd P((hypre_Vector *x , hypre_Vector *y ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

