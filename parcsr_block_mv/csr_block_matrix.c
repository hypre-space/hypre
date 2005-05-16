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

#include "csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixCreate(int block_size, int num_rows, int num_cols,
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
   hypre_CSRBlockMatrixOwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_CSRBlockMatrixDestroy(hypre_CSRBlockMatrix *matrix)
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
hypre_CSRBlockMatrixInitialize(hypre_CSRBlockMatrix *matrix)
{
   int block_size   = hypre_CSRBlockMatrixBlockSize(matrix);
   int num_rows     = hypre_CSRBlockMatrixNumRows(matrix);
   int num_nonzeros = hypre_CSRBlockMatrixNumNonzeros(matrix);
   int ierr=0, nnz;

   if ( ! hypre_CSRBlockMatrixI(matrix) )
      hypre_TFree(hypre_CSRBlockMatrixI(matrix));
   if ( ! hypre_CSRBlockMatrixJ(matrix) )
      hypre_TFree(hypre_CSRBlockMatrixJ(matrix));
   if ( ! hypre_CSRBlockMatrixData(matrix) )
      hypre_TFree(hypre_CSRBlockMatrixData(matrix));

   nnz = num_nonzeros * block_size * block_size;
   hypre_CSRBlockMatrixI(matrix) = hypre_CTAlloc(int, num_rows + 1);
   if (nnz) hypre_CSRBlockMatrixData(matrix) = hypre_CTAlloc(double, nnz);
   else     hypre_CSRBlockMatrixData(matrix) = NULL;
   if (nnz) hypre_CSRBlockMatrixJ(matrix) = hypre_CTAlloc(int,num_nonzeros);
   else     hypre_CSRBlockMatrixJ(matrix) = NULL;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_CSRBlockMatrixSetDataOwner(hypre_CSRBlockMatrix *matrix, int owns_data)
{
   int    ierr=0;

   hypre_CSRBlockMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixCompress
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRBlockMatrixCompress(hypre_CSRBlockMatrix *matrix)
{
   int    block_size = hypre_CSRBlockMatrixBlockSize(matrix);
   int    num_rows = hypre_CSRBlockMatrixNumRows(matrix);
   int    num_cols = hypre_CSRBlockMatrixNumCols(matrix);
   int    num_nonzeros = hypre_CSRBlockMatrixNumNonzeros(matrix);
   int    *matrix_i = hypre_CSRBlockMatrixI(matrix);
   int    *matrix_j = hypre_CSRBlockMatrixJ(matrix);
   double *matrix_data = hypre_CSRBlockMatrixData(matrix);
   hypre_CSRMatrix* matrix_C;
   int    *matrix_C_i, *matrix_C_j, i, j, bnnz;
   double *matrix_C_data, ddata;

   matrix_C = hypre_CSRMatrixCreate(num_rows,num_cols,num_nonzeros);
   hypre_CSRMatrixInitialize(matrix_C);
   matrix_C_i = hypre_CSRMatrixI(matrix_C);
   matrix_C_j = hypre_CSRMatrixJ(matrix_C);
   matrix_C_data = hypre_CSRMatrixData(matrix_C);

   bnnz = block_size * block_size;
   for(i = 0; i < num_rows + 1; i++) matrix_C_i[i] = matrix_i[i];
   for(i = 0; i < num_nonzeros; i++)
   {
      matrix_C_j[i] = matrix_j[i];
      ddata = 0.0;
      for(j = 0; j < bnnz; j++)
         ddata += matrix_data[i*bnnz+j] * matrix_data[i*bnnz+j];
      matrix_C_data[i] = sqrt(ddata);
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
   int *matrix_i = hypre_CSRBlockMatrixI(matrix);
   int *matrix_j = hypre_CSRBlockMatrixJ(matrix);
   double* matrix_data = hypre_CSRBlockMatrixData(matrix);

   hypre_CSRMatrix* matrix_C;
   int    i, j, k, ii, C_ii, bnnz, new_nrows, new_ncols, new_num_nonzeros;
   int    *matrix_C_i, *matrix_C_j;
   double *matrix_C_data;

   bnnz      = block_size * block_size;
   new_nrows = num_rows * block_size;
   new_ncols = num_cols * block_size;
   new_num_nonzeros = block_size * block_size * num_nonzeros;
   matrix_C = hypre_CSRMatrixCreate(new_nrows,new_ncols,new_num_nonzeros);
   hypre_CSRMatrixInitialize(matrix_C);
   matrix_C_i    = hypre_CSRMatrixI(matrix_C);
   matrix_C_j    = hypre_CSRMatrixJ(matrix_C);
   matrix_C_data = hypre_CSRMatrixData(matrix_C);
   for(i = 0; i < num_rows; i++)
   {
      for(j = 0; j < block_size; j++)
         matrix_C_i[i*block_size + j] = matrix_i[i]*bnnz + 
                               j * (matrix_i[i + 1] - matrix_i[i])*block_size;
   }
   matrix_C_i[new_nrows] = matrix_i[num_rows] * bnnz;

   C_ii = 0;
   for(i = 0; i < num_rows; i++)
   {
      for(j = 0; j < block_size; j++)
      {
         for(ii = matrix_i[i]; ii < matrix_i[i + 1]; ii++)
         {
	    k = j;
	    matrix_C_j[C_ii] = matrix_j[ii]*block_size + k;
	    matrix_C_data[C_ii] = matrix_data[ii*bnnz+j*block_size+k];
	    C_ii++;
	    for(k = 0; k < block_size; k++)
            {
	       if(j != k)
               {
	          matrix_C_j[C_ii] = matrix_j[ii]*block_size + k;
	          matrix_C_data[C_ii] = matrix_data[ii*bnnz+j*block_size+k];
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
hypre_CSRBlockMatrixConvertFromCSRMatrix(hypre_CSRMatrix *matrix, 
                                         int matrix_C_block_size )
{
   int num_rows = hypre_CSRMatrixNumRows(matrix);
   int num_cols = hypre_CSRMatrixNumCols(matrix);
   int *matrix_i = hypre_CSRMatrixI(matrix);
   int *matrix_j = hypre_CSRMatrixJ(matrix);
   double* matrix_data = hypre_CSRMatrixData(matrix);

   hypre_CSRBlockMatrix* matrix_C;
   int    *matrix_C_i, *matrix_C_j;
   double *matrix_C_data;
   int    matrix_C_num_rows, matrix_C_num_cols, matrix_C_num_nonzeros;
   int    i, j, ii, jj, s_jj, index, *counter;

   matrix_C_num_rows = num_rows/matrix_C_block_size;
   matrix_C_num_cols = num_cols/matrix_C_block_size;

   counter = hypre_CTAlloc(int, matrix_C_num_cols);
   for(i = 0; i < matrix_C_num_cols; i++) counter[i] = -1;
   matrix_C_num_nonzeros = 0;
   for(i = 0; i < matrix_C_num_rows; i++)
   {
      for(j = 0; j < matrix_C_block_size; j++)
      {
         for(ii = matrix_i[i*matrix_C_block_size+j]; 
             ii < matrix_i[i*matrix_C_block_size+j+1]; ii++)
         {
	    if(counter[matrix_j[ii]/matrix_C_block_size] < i)
            {
	       counter[matrix_j[ii]/matrix_C_block_size] = i;
	       matrix_C_num_nonzeros++;
	    }
         }
      }
   }
   matrix_C = hypre_CSRBlockMatrixCreate(matrix_C_block_size, matrix_C_num_rows, 
                                       matrix_C_num_cols, matrix_C_num_nonzeros);
   hypre_CSRBlockMatrixInitialize(matrix_C);
   matrix_C_i = hypre_CSRBlockMatrixI(matrix_C);
   matrix_C_j = hypre_CSRBlockMatrixJ(matrix_C);
   matrix_C_data = hypre_CSRBlockMatrixData(matrix_C);
 
   for(i = 0; i < matrix_C_num_cols; i++) counter[i] = -1;
   jj = s_jj = 0;
   for (i = 0; i < matrix_C_num_rows; i++)
   {
      matrix_C_i[i] = jj;
      for(j = 0; j < matrix_C_block_size; j++)
      {
         for(ii = matrix_i[i*matrix_C_block_size+j];
             ii < matrix_i[i*matrix_C_block_size+j+1]; ii++)
         {
 	    if(counter[matrix_j[ii]/matrix_C_block_size] < s_jj)
            {
 	       counter[matrix_j[ii]/matrix_C_block_size] = jj;
 	       matrix_C_j[jj] = matrix_j[ii]/matrix_C_block_size;
 	       jj++;
 	    }
 	    index = counter[matrix_j[ii]/matrix_C_block_size] * matrix_C_block_size *
                    matrix_C_block_size + j * matrix_C_block_size + 
                    matrix_j[ii]%matrix_C_block_size;
 	    matrix_C_data[index] = matrix_data[ii];
         }
      }
      s_jj = jj;
   }
   matrix_C_i[matrix_C_num_rows] = matrix_C_num_nonzeros;
   return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixBlockAdd
 * (o = i1 + i2) 
 *--------------------------------------------------------------------------*/
int
hypre_CSRBlockMatrixBlockAdd(double* i1, double* i2, double* o, int block_size)
{
   int i, j;

   for (i = 0; i < block_size; i++)
      for (j = 0; j < block_size; j++)
         o[i*block_size+j] = i1[i*block_size+j] + i2[i*block_size+j];
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixBlockMultAdd
 * (o = i1 * i2 + beta * o) 
 *--------------------------------------------------------------------------*/
int
hypre_CSRBlockMatrixBlockMultAdd(double* i1, double* i2, double beta, 
                                 double* o, int block_size)
{
   int    i, j, k;
   double ddata;

   if (beta == 0.0)
   {
      for (i = 0; i < block_size; i++)
      {
         for (j = 0; j < block_size; j++)
         {
            ddata = 0.0;
            for (k = 0; k < block_size; k++)
            {
               ddata += i1[i*block_size + k] * i2[k*block_size + j];
            }
            o[i*block_size + j] = ddata;
         }
      }
   }
   else if (beta == 1.0)
   {
      for(i = 0; i < block_size; i++)
      {
         for(j = 0; j < block_size; j++)
         {
            ddata = o[i*block_size + j];
            for(k = 0; k < block_size; k++)
               ddata += i1[i*block_size + k] * i2[k*block_size + j];
            o[i*block_size + j] = ddata;
         }
      }
   }
   else
   {
      for(i = 0; i < block_size; i++)
      {
         for(j = 0; j < block_size; j++)
         {
            ddata = beta * o[i*block_size + j];
            for(k = 0; k < block_size; k++)
               ddata += i1[i*block_size + k] * i2[k*block_size + j];
            o[i*block_size + j] = ddata;
         }
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixBlockMult
 * (o = i1^{-1} * i2) 
 *--------------------------------------------------------------------------*/
int
hypre_CSRBlockMatrixBlockInvMult(double* i1, double* i2, double* o, int block_size)
{
   double *t;

   t = hypre_CTAlloc(double, block_size*block_size);
   printf("hypre_CSRblockMatrixblockInvMult : not implemented yet.\n");
   exit(1);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRBlockMatrixTranspose
 *--------------------------------------------------------------------------*/

int hypre_CSRBlockMatrixTranspose(hypre_CSRBlockMatrix *A, 
                                  hypre_CSRBlockMatrix **AT, int data)

{
   double       *A_data = hypre_CSRBlockMatrixData(A);
   int          *A_i = hypre_CSRBlockMatrixI(A);
   int          *A_j = hypre_CSRBlockMatrixJ(A);
   int           num_rowsA = hypre_CSRBlockMatrixNumRows(A);
   int           num_colsA = hypre_CSRBlockMatrixNumCols(A);
   int           num_nonzerosA = hypre_CSRBlockMatrixNumNonzeros(A);
   int           block_size = hypre_CSRBlockMatrixBlockSize(A);

   double       *AT_data;
   int          *AT_i;
   int          *AT_j;
   int           num_rowsAT;
   int           num_colsAT;
   int           num_nonzerosAT;

   int           max_col;
   int           i, j, k, m, offset, bnnz;

   /*-------------------------------------------------------------- 
    * First, ascertain that num_cols and num_nonzeros has been set. 
    * If not, set them.
    *--------------------------------------------------------------*/

   if (! num_nonzerosA) num_nonzerosA = A_i[num_rowsA];
   if (num_rowsA && ! num_colsA)
   {
      max_col = -1;
      for (i = 0; i < num_rowsA; ++i)
         for (j = A_i[i]; j < A_i[i+1]; j++)
            if (A_j[j] > max_col) max_col = A_j[j];
      num_colsA = max_col+1;
   }
   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;
   bnnz = block_size * block_size;

   *AT = hypre_CSRBlockMatrixCreate(block_size, num_rowsAT, num_colsAT, 
                                    num_nonzerosAT);

   AT_i = hypre_CTAlloc(int, num_rowsAT+1);
   AT_j = hypre_CTAlloc(int, num_nonzerosAT);
   hypre_CSRBlockMatrixI(*AT) = AT_i;
   hypre_CSRBlockMatrixJ(*AT) = AT_j;
   if (data) 
   {
      AT_data = hypre_CTAlloc(double, num_nonzerosAT*bnnz);
      hypre_CSRBlockMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Count the number of entries in each column of A (row of AT)
    * and fill the AT_i array.
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_nonzerosA; i++) ++AT_i[A_j[i]+1];
   for (i = 2; i <= num_rowsAT; i++) AT_i[i] += AT_i[i-1];

   /*----------------------------------------------------------------
    * Load the data and column numbers of AT
    *----------------------------------------------------------------*/

   for (i = 0; i < num_rowsA; i++)
   {
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         AT_j[AT_i[A_j[j]]] = i;
         if (data)
         {
            offset = AT_i[A_j[j]] * bnnz;
            for (k = 0; k < block_size; k++)
               for (m = 0; m < block_size; m++)
                  AT_data[offset+k*block_size+m] = 
                       A_data[j*bnnz+m*block_size+k];
         }
         AT_i[A_j[j]]++;
      }
   }

   /*------------------------------------------------------------
    * AT_i[j] now points to the *end* of the jth row of entries
    * instead of the beginning.  Restore AT_i to front of row.
    *------------------------------------------------------------*/

   for (i = num_rowsAT; i > 0; i--) AT_i[i] = AT_i[i-1];
   AT_i[0] = 0;

   return(0);
}

