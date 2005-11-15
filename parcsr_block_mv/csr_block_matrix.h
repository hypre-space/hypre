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
 * Header info for CSR Block Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 * Note: everything is in terms of blocks (ie. num_rows is the number
 *       of block rows)
 *
 *****************************************************************************/


#ifndef hypre_CSR_BLOCK_MATRIX_HEADER
#define hypre_CSR_BLOCK_MATRIX_HEADER

#include "seq_mv.h"
#include "utilities.h"
                                                                                                               
#ifdef __cplusplus
extern "C" {
#endif
                                                                                                               

/*--------------------------------------------------------------------------
 * CSR Block Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
  double	        *data;
  int                   *i;
  int                   *j;
  int                   block_size;
  int     		num_rows;
  int     		num_cols;
  int                   num_nonzeros;

  int                   owns_data;

} hypre_CSRBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBlockMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRBlockMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRBlockMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRBlockMatrixBlockSize(matrix)    ((matrix) -> block_size)
#define hypre_CSRBlockMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRBlockMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRBlockMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRBlockMatrixOwnsData(matrix)     ((matrix) -> owns_data)

/*--------------------------------------------------------------------------
 * other functions for the CSR Block Matrix structure
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix 
      *hypre_CSRBlockMatrixCreate(int, int, int, int);
int hypre_CSRBlockMatrixDestroy(hypre_CSRBlockMatrix *);
int hypre_CSRBlockMatrixInitialize(hypre_CSRBlockMatrix *);
int hypre_CSRBlockMatrixSetDataOwner(hypre_CSRBlockMatrix *, int);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixCompress(hypre_CSRBlockMatrix *);
hypre_CSRMatrix 
      *hypre_CSRBlockMatrixConvertToCSRMatrix(hypre_CSRBlockMatrix *);
hypre_CSRBlockMatrix
      *hypre_CSRBlockMatrixConvertFromCSRMatrix(hypre_CSRMatrix *, int);
int hypre_CSRBlockMatrixBlockAdd(double *, double *, double*, int);
int hypre_CSRBlockMatrixBlockMultAdd(double *, double *, double, double *, int);
int hypre_CSRBlockMatrixBlockInvMult(double *, double *, double *, int);
int hypre_CSRBlockMatrixBlockMultInv(double *, double *, double *, int);
int hypre_CSRBlockMatrixBlockTranspose(double *, double *, int);

int hypre_CSRBlockMatrixTranspose(hypre_CSRBlockMatrix *A,
                                  hypre_CSRBlockMatrix **AT, int data);

int hypre_CSRBlockMatrixBlockCopyData(double*, double*, double, int);
   
int hypre_CSRBlockMatrixBlockAddAccumulate(double*, double*, int);
   


int
hypre_CSRBlockMatrixMatvec(double alpha, hypre_CSRBlockMatrix *A,
                           hypre_Vector *x, double beta, hypre_Vector *y);
   

int
hypre_CSRBlockMatrixMatvecT( double alpha, hypre_CSRBlockMatrix *A, hypre_Vector  *x,
                             double beta, hypre_Vector *y );

int
hypre_CSRBlockMatrixBlockInvMatvec(double* mat, double* v, 
                                   double* ov, int block_size);
   
int 
hypre_CSRBlockMatrixBlockMatvec(double alpha, double* mat, double* v, double beta, 
                                double* ov, int block_size);
   

int hypre_CSRBlockMatrixBlockNorm(int norm_type, double* data, double* out, int block_size);
   
int hypre_CSRBlockMatrixBlockSetScalar(double* o, double beta, int block_size);
   

#ifdef __cplusplus
}
#endif
#endif
