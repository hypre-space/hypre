
#include <HYPRE_config.h>

#include "HYPRE_seq_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

#include "utilities.h"

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
# define	P(s) s

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate P((int num_rows , int num_cols , int *row_sizes ));
int HYPRE_CSRMatrixDestroy P((HYPRE_CSRMatrix matrix ));
int HYPRE_CSRMatrixInitialize P((HYPRE_CSRMatrix matrix ));
HYPRE_CSRMatrix HYPRE_CSRMatrixRead P((char *file_name ));
void HYPRE_CSRMatrixPrint P((HYPRE_CSRMatrix matrix , char *file_name ));

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate P((void ));
int HYPRE_MappedMatrixDestroy P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixLimitedDestroy P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixInitialize P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixAssemble P((HYPRE_MappedMatrix matrix ));
void HYPRE_MappedMatrixPrint P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixGetColIndex P((HYPRE_MappedMatrix matrix , int j ));
void *HYPRE_MappedMatrixGetMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixSetMatrix P((HYPRE_MappedMatrix matrix , void *matrix_data ));
int HYPRE_MappedMatrixSetColMap P((HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *)));
int HYPRE_MappedMatrixSetMapData P((HYPRE_MappedMatrix matrix , void *MapData ));

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate P((void ));
int HYPRE_MultiblockMatrixDestroy P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixLimitedDestroy P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixInitialize P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixAssemble P((HYPRE_MultiblockMatrix matrix ));
void HYPRE_MultiblockMatrixPrint P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixSetNumSubmatrices P((HYPRE_MultiblockMatrix matrix , int n ));
int HYPRE_MultiblockMatrixSetSubmatrixType P((HYPRE_MultiblockMatrix matrix , int j , int type ));

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate P((int size ));
int HYPRE_VectorDestroy P((HYPRE_Vector vector ));
int HYPRE_VectorInitialize P((HYPRE_Vector vector ));
int HYPRE_VectorPrint P((HYPRE_Vector vector , char *file_name ));
HYPRE_Vector HYPRE_VectorRead P((char *file_name ));

/* csr_matop.c */
hypre_CSRMatrix *hypre_CSRMatrixAdd P((hypre_CSRMatrix *A , hypre_CSRMatrix *B ));
hypre_CSRMatrix *hypre_CSRMatrixMultiply P((hypre_CSRMatrix *A , hypre_CSRMatrix *B ));
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros P((hypre_CSRMatrix *A , double tol ));

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate P((int num_rows , int num_cols , int num_nonzeros ));
int hypre_CSRMatrixDestroy P((hypre_CSRMatrix *matrix ));
int hypre_CSRMatrixInitialize P((hypre_CSRMatrix *matrix ));
int hypre_CSRMatrixSetDataOwner P((hypre_CSRMatrix *matrix , int owns_data ));
hypre_CSRMatrix *hypre_CSRMatrixRead P((char *file_name ));
int hypre_CSRMatrixPrint P((hypre_CSRMatrix *matrix , char *file_name ));
int hypre_CSRMatrixCopy P((hypre_CSRMatrix *A , hypre_CSRMatrix *B , int copy_data ));

/* csr_matvec.c */
int hypre_CSRMatrixMatvec P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));
int hypre_CSRMatrixMatvecT P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

/* genpart.c */
int hypre_GeneratePartitioning P((int length , int num_procs , int **part_ptr ));

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_MappedMatrixCreate P((void ));
int hypre_MappedMatrixDestroy P((hypre_MappedMatrix *matrix ));
int hypre_MappedMatrixLimitedDestroy P((hypre_MappedMatrix *matrix ));
int hypre_MappedMatrixInitialize P((hypre_MappedMatrix *matrix ));
int hypre_MappedMatrixAssemble P((hypre_MappedMatrix *matrix ));
void hypre_MappedMatrixPrint P((hypre_MappedMatrix *matrix ));
int hypre_MappedMatrixGetColIndex P((hypre_MappedMatrix *matrix , int j ));
void *hypre_MappedMatrixGetMatrix P((hypre_MappedMatrix *matrix ));
int hypre_MappedMatrixSetMatrix P((hypre_MappedMatrix *matrix , void *matrix_data ));
int hypre_MappedMatrixSetColMap P((hypre_MappedMatrix *matrix , int (*ColMap )(int ,void *)));
int hypre_MappedMatrixSetMapData P((hypre_MappedMatrix *matrix , void *map_data ));

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate P((void ));
int hypre_MultiblockMatrixDestroy P((hypre_MultiblockMatrix *matrix ));
int hypre_MultiblockMatrixLimitedDestroy P((hypre_MultiblockMatrix *matrix ));
int hypre_MultiblockMatrixInitialize P((hypre_MultiblockMatrix *matrix ));
int hypre_MultiblockMatrixAssemble P((hypre_MultiblockMatrix *matrix ));
void hypre_MultiblockMatrixPrint P((hypre_MultiblockMatrix *matrix ));
int hypre_MultiblockMatrixSetNumSubmatrices P((hypre_MultiblockMatrix *matrix , int n ));
int hypre_MultiblockMatrixSetSubmatrixType P((hypre_MultiblockMatrix *matrix , int j , int type ));
int hypre_MultiblockMatrixSetSubmatrix P((hypre_MultiblockMatrix *matrix , int j , void *submatrix ));

/* vector.c */
hypre_Vector *hypre_VectorCreate P((int size ));
int hypre_VectorDestroy P((hypre_Vector *vector ));
int hypre_VectorInitialize P((hypre_Vector *vector ));
int hypre_VectorSetDataOwner P((hypre_Vector *vector , int owns_data ));
hypre_Vector *hypre_VectorRead P((char *file_name ));
int hypre_VectorPrint P((hypre_Vector *vector , char *file_name ));
int hypre_VectorSetConstantValues P((hypre_Vector *v , double value ));
int hypre_VectorSetRandomValues P((hypre_Vector *v , int seed ));
int hypre_VectorCopy P((hypre_Vector *x , hypre_Vector *y ));
int hypre_VectorScale P((double alpha , hypre_Vector *y ));
int hypre_VectorAxpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_VectorInnerProd P((hypre_Vector *x , hypre_Vector *y ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

