/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_CSRBlockMatrix class.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixCreate( int block_size,
				int num_rows,
				int num_cols,
				int num_nonzeros)
{
   hypre_CSRBlockMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_CSRBlockMatrix, 1);

   hypre_CSRBlockMatrixData(matrix) = NULL;
   hypre_CSRBlockMatrixI(matrix)    = NULL;
   hypre_CSRBlockMatrixJ(matrix)    = NULL;
   hypre_CSRBlockMatrixBlockSize(matrix) = block_size;
   hypre_CSRBlockMatrixNumRows(matrix) = num_rows;
   hypre_CSRBlockMatrixNumCols(matrix) = num_cols;
   hypre_CSRBlockMatrixNumNonzeros(matrix) = num_nonzeros;

   /* set defaults */
   hypre_CSRMatrixOwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_CSRBlockMatrixDestroy( hypre_CSRBlockMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_CSRBlockMatrixI(matrix));
      if ( hypre_CSRBlockMatrixOwnsData(matrix) )
      {
         hypre_TFree(hypre_CSRBlockMatrixData(matrix));
         hypre_TFree(hypre_CSRBlockMatrixJ(matrix));
      }
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_CSRBlockMatrixInitialize( hypre_CSRBlockMatrix *matrix )
{
  int block_size = hypre_CSRBlockMatrixBlockSize(matrix);
   int  num_rows     = hypre_CSRBlockMatrixNumRows(matrix);
   int  num_nonzeros = hypre_CSRBlockMatrixNumNonzeros(matrix);

   int  ierr=0;

   if ( ! hypre_CSRBlockMatrixData(matrix) && num_nonzeros )
      hypre_CSRBlockMatrixData(matrix) = hypre_CTAlloc(double, num_nonzeros*block_size*block_size);
   if ( ! hypre_CSRBlockMatrixI(matrix) )
      hypre_CSRBlockMatrixI(matrix)    = hypre_CTAlloc(int, num_rows + 1);
   if ( ! hypre_CSRBlockMatrixJ(matrix) && num_nonzeros )
      hypre_CSRBlockMatrixJ(matrix)    = hypre_CTAlloc(int, num_nonzeros);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_CSRBlockMatrixSetDataOwner( hypre_CSRBlockMatrix *matrix,
				     int              owns_data )
{
   int    ierr=0;

   hypre_CSRBlockMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixCompress
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRBlockMatrixCompress( hypre_CSRBlockMatrix *matrix )
{
  int block_size = hypre_CSRBlockMatrixBlockSize(matrix);
  int num_rows = hypre_CSRBlockMatrixNumRows(matrix);
  int num_cols = hypre_CSRBlockMatrixNumCols(matrix);
  int num_nonzeros = hypre_CSRBlockMatrixNumNonzeros(matrix);
  int* matrix_i = hypre_CSRBlockMatrixI(matrix);
  int* matrix_j = hypre_CSRBlockMatrixJ(matrix);
  double* matrix_data = hypre_CSRBlockMatrixData(matrix);

  hypre_CSRMatrix* matrix_C;
  int* matrix_C_i;
  int* matrix_C_j;
  double* matrix_C_data;

  int i, j;

  matrix_C = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

  hypre_CSRMatrixInitialize(matrix_C);

  matrix_C_i = hypre_CSRMatrixI(matrix_C);
  matrix_C_j = hypre_CSRMatrixJ(matrix_C);
  matrix_C_data = hypre_CSRMatrixData(matrix_C);

  for(i = 0; i < num_rows + 1; i++) {
    matrix_C_i[i] = matrix_i[i];
  }
  for(i = 0; i < num_nonzeros; i++) {
    matrix_C_j[i] = matrix_j[i];

    matrix_C_data[i] = 0;
    for(j = 0; j < block_size*block_size; j++) {
      matrix_C_data[i] += matrix_data[i*block_size*block_size + j]*matrix_data[i*block_size*block_size + j];
    }
    matrix_C_data[i] = sqrt(matrix_C_data[i]);
  }

  return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixConvertToCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRBlockMatrixConvertToCSRMatrix( hypre_CSRBlockMatrix *matrix )
{
  int block_size = hypre_CSRBlockMatrixBlockSize(matrix);
  int num_rows = hypre_CSRBlockMatrixNumRows(matrix);
  int num_cols = hypre_CSRBlockMatrixNumCols(matrix);
  int num_nonzeros = hypre_CSRBlockMatrixNumNonzeros(matrix);
  int* matrix_i = hypre_CSRBlockMatrixI(matrix);
  int* matrix_j = hypre_CSRBlockMatrixJ(matrix);
  double* matrix_data = hypre_CSRBlockMatrixData(matrix);

  hypre_CSRMatrix* matrix_C;
  int* matrix_C_i;
  int* matrix_C_j;
  double* matrix_C_data;

  int i, j, k, ii, C_ii;

  matrix_C = hypre_CSRMatrixCreate(num_rows*block_size, num_cols*block_size, num_nonzeros*block_size*block_size);
  hypre_CSRMatrixInitialize(matrix_C);

  matrix_C_i = hypre_CSRMatrixI(matrix_C);
  matrix_C_j = hypre_CSRMatrixJ(matrix_C);
  matrix_C_data = hypre_CSRMatrixData(matrix_C);

  for(i = 0; i < num_rows; i++) {
    for(j = 0; j < block_size; j++) {
      matrix_C_i[i*block_size + j] = matrix_i[i]*block_size*block_size + j*(matrix_i[i + 1] - matrix_i[i])*block_size;
    }
  }
  matrix_C_i[num_rows*block_size] = matrix_i[num_rows]*block_size*block_size;

  C_ii = 0;
  for(i = 0; i < num_rows; i++) {
    for(j = 0; j < block_size; j++) {
      for(ii = matrix_i[i]; ii < matrix_i[i + 1]; ii++) {
	k = j;
	matrix_C_j[C_ii] = matrix_j[ii]*block_size + k;
	matrix_C_data[C_ii] = matrix_data[ii*block_size*block_size + j*block_size + k];
	C_ii++;
	for(k = 0; k < block_size; k++) {
	  if(j != k) {
	    matrix_C_j[C_ii] = matrix_j[ii]*block_size + k;
	    matrix_C_data[C_ii] = matrix_data[ii*block_size*block_size + j*block_size + k];
	    C_ii++;
	  }
	} 
      }   
    }
  }

  return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixConvertFromCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixConvertFromCSRMatrix( hypre_CSRMatrix *matrix, int matrix_C_block_size )
{
  int num_rows = hypre_CSRMatrixNumRows(matrix);
  int num_cols = hypre_CSRMatrixNumCols(matrix);
  int* matrix_i = hypre_CSRMatrixI(matrix);
  int* matrix_j = hypre_CSRMatrixJ(matrix);
  double* matrix_data = hypre_CSRMatrixData(matrix);

  hypre_CSRBlockMatrix* matrix_C;
  int* matrix_C_i;
  int* matrix_C_j;
  double* matrix_C_data;
  int matrix_C_num_rows;
  int matrix_C_num_cols;
  int matrix_C_num_nonzeros;

  int* counter;
  int i, j, ii, jj, s_jj;

  matrix_C_num_rows = num_rows/matrix_C_block_size;
  matrix_C_num_cols = num_cols/matrix_C_block_size;

  counter = hypre_CTAlloc(int, matrix_C_num_cols);
  for(i = 0; i < matrix_C_num_cols; i++) counter[i] = -1;
  matrix_C_num_nonzeros = 0;
  for(i = 0; i < matrix_C_num_rows; i++) {
    for(j = 0; j < matrix_C_block_size; j++) {
      for(ii = matrix_i[i*matrix_C_block_size + j]; ii < matrix_i[i*matrix_C_block_size + j + 1]; ii++) {
	if(counter[matrix_j[ii]/matrix_C_block_size] < i) {
	  counter[matrix_j[ii]/matrix_C_block_size] = i;
	  matrix_C_num_nonzeros++;
	}
      }
    }
  }

  matrix_C = hypre_CSRBlockMatrixCreate(matrix_C_block_size, matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros);
  hypre_CSRBlockMatrixInitialize(matrix_C);
  matrix_C_i = hypre_CSRBlockMatrixI(matrix_C);
  matrix_C_j = hypre_CSRBlockMatrixJ(matrix_C);
  matrix_C_data = hypre_CSRBlockMatrixData(matrix_C);

  for(i = 0; i < matrix_C_num_cols; i++) counter[i] = -1;
  jj = s_jj = 0;
  for(i = 0; i < matrix_C_num_rows; i++) {
    matrix_C_i[i] = jj;
    for(j = 0; j < matrix_C_block_size; j++) {
      for(ii = matrix_i[i*matrix_C_block_size + j]; ii < matrix_i[i*matrix_C_block_size + j + 1]; ii++) {
	if(counter[matrix_j[ii]/matrix_C_block_size] < s_jj) {
	  counter[matrix_j[ii]/matrix_C_block_size] = jj;
	  matrix_C_j[jj] = matrix_j[ii]/matrix_C_block_size;
	  jj++;
	}
	matrix_C_data[counter[matrix_j[ii]/matrix_C_block_size]*matrix_C_block_size*matrix_C_block_size + j*matrix_C_block_size + matrix_j[ii]%matrix_C_block_size] = matrix_data[ii];
      }
    }
    s_jj = jj;
  }
  matrix_C_i[matrix_C_num_rows] = matrix_C_num_nonzeros;

  return matrix_C;
}

/* o = i1 + i2 */
int
hypre_CSRBlockMatrixBlockAdd(double* i1, double* i2, double* o, int block_size)
{
  int i, j;

  for(i = 0; i < block_size; i++) {
    for(j = 0; j < block_size; j++) {
      o[i*block_size + j] = i1[i*block_size + j] + i2[i*block_size + j];
    }
  }

  return 0;
}

/* o = i1*i2 */
int
hypre_CSRBlockMatrixBlockMult(double* i1, double* i2, double* o, int block_size)
{
  int i, j, k;

  for(i = 0; i < block_size; i++) {
    for(j = 0; j < block_size; j++) {
      o[i*block_size + j] = 0;
      for(k = 0; k < block_size; k++) {
	o[i*block_size + j] += i1[i*block_size + k] * i2[k*block_size + j];
      }
    }
  }

  return 0;
}

/* o = i1^{-1}*i2 */
int
hypre_CSRBlockMatrixBlockInvMult(double* i1, double* i2, double* o, int block_size)
{
  int i, j, k;
  double* t;

  t = hypre_CTAlloc(block_size*block_size, double);

  

  return 0;
}
