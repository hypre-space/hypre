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
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixCreate( int num_rows,
                       int num_cols,
                       int num_nonzeros )
{
   hypre_CSRMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_CSRMatrix, 1);

   hypre_CSRMatrixData(matrix) = NULL;
   hypre_CSRMatrixI(matrix)    = NULL;
   hypre_CSRMatrixJ(matrix)    = NULL;
   hypre_CSRMatrixRownnz(matrix) = NULL;
   hypre_CSRMatrixNumRows(matrix) = num_rows;
   hypre_CSRMatrixNumCols(matrix) = num_cols;
   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;

   /* set defaults */
   hypre_CSRMatrixOwnsData(matrix) = 1;
   hypre_CSRMatrixNumRownnz(matrix) = num_rows;


   return matrix;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_CSRMatrixDestroy( hypre_CSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_CSRMatrixI(matrix));
      if (hypre_CSRMatrixRownnz(matrix))
         hypre_TFree(hypre_CSRMatrixRownnz(matrix));
      if ( hypre_CSRMatrixOwnsData(matrix) )
      {
         hypre_TFree(hypre_CSRMatrixData(matrix));
         hypre_TFree(hypre_CSRMatrixJ(matrix));
      }
      hypre_TFree(matrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_CSRMatrixInitialize( hypre_CSRMatrix *matrix )
{
   int  num_rows     = hypre_CSRMatrixNumRows(matrix);
   int  num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix);

   int  ierr=0;

   if ( ! hypre_CSRMatrixData(matrix) && num_nonzeros )
      hypre_CSRMatrixData(matrix) = hypre_CTAlloc(double, num_nonzeros);
   if ( ! hypre_CSRMatrixI(matrix) )
      hypre_CSRMatrixI(matrix)    = hypre_CTAlloc(int, num_rows + 1);
/*   if ( ! hypre_CSRMatrixRownnz(matrix) )
      hypre_CSRMatrixRownnz(matrix)    = hypre_CTAlloc(int, num_rownnz);*/
   if ( ! hypre_CSRMatrixJ(matrix) && num_nonzeros )
      hypre_CSRMatrixJ(matrix)    = hypre_CTAlloc(int, num_nonzeros);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

int 
hypre_CSRMatrixSetDataOwner( hypre_CSRMatrix *matrix,
                             int              owns_data )
{
   int    ierr=0;

   hypre_CSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixSetRownnz( hypre_CSRMatrix *matrix )
{
   int    ierr=0;
   int  num_rows = hypre_CSRMatrixNumRows(matrix);
   int  *A_i = hypre_CSRMatrixI(matrix);
   int  *Arownnz;

   int i, adiag;
   int irownnz=0;

   for (i=0; i < num_rows; i++)
   {
      adiag = (A_i[i+1] - A_i[i]);
      if(adiag > 0) irownnz++;
   }

   hypre_CSRMatrixNumRownnz(matrix) = irownnz;

   if ((irownnz == 0) || (irownnz == num_rows))
   {
       hypre_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
       Arownnz = hypre_CTAlloc(int, irownnz);
       irownnz = 0;
       for (i=0; i < num_rows; i++)
       {
          adiag = A_i[i+1]-A_i[i];
          if(adiag > 0) Arownnz[irownnz++] = i;
       }
        hypre_CSRMatrixRownnz(matrix) = Arownnz;
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixRead( char *file_name )
{
   hypre_CSRMatrix  *matrix;

   FILE    *fp;

   double  *matrix_data;
   int     *matrix_i;
   int     *matrix_j;
   int      num_rows;
   int      num_nonzeros;
   int      max_col = 0;

   int      file_base = 1;
   
   int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &num_rows);

   matrix_i = hypre_CTAlloc(int, num_rows + 1);
   for (j = 0; j < num_rows+1; j++)
   {
      fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize(matrix);

   matrix_j = hypre_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j++)
   {
      fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = hypre_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   hypre_CSRMatrixNumCols(matrix) = ++max_col;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_CSRMatrixPrint( hypre_CSRMatrix *matrix,
                      char            *file_name )
{
   FILE    *fp;

   double  *matrix_data;
   int     *matrix_i;
   int     *matrix_j;
   int      num_rows;
   
   int      file_base = 1;
   
   int      j;

   int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         fprintf(fp, "%le\n", matrix_data[j]);
      }
   }
   else
   {
      fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCopy:
 * copys A to B, 
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions of A and B match !!! 
 *--------------------------------------------------------------------------*/

int 
hypre_CSRMatrixCopy( hypre_CSRMatrix *A, hypre_CSRMatrix *B, int copy_data )
{
   int  ierr=0;
   int  num_rows = hypre_CSRMatrixNumRows(A);
   int *A_i = hypre_CSRMatrixI(A);
   int *A_j = hypre_CSRMatrixJ(A);
   double *A_data;
   int *B_i = hypre_CSRMatrixI(B);
   int *B_j = hypre_CSRMatrixJ(B);
   double *B_data;

   int i, j;

   for (i=0; i < num_rows; i++)
   {
	B_i[i] = A_i[i];
	for (j=A_i[i]; j < A_i[i+1]; j++)
	{
		B_j[j] = A_j[j];
	}
   }
   B_i[num_rows] = A_i[num_rows];
   if (copy_data)
   {
	A_data = hypre_CSRMatrixData(A);
	B_data = hypre_CSRMatrixData(B);
   	for (i=0; i < num_rows; i++)
   	{
	   for (j=A_i[i]; j < A_i[i+1]; j++)
	   {
		B_data[j] = A_data[j];
	   }
	}
   }
   return ierr;
}
