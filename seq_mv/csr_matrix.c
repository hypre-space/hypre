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
/*   int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix); */

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
         fprintf(fp, "%e\n", matrix_data[j]);
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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone
 * Creates and returns a new copy of the argument, A.
 * Data is not copied, only structural information is reproduced.
 * Copying is a deep copy in that no pointers are copied; new arrays are
 * created where necessary.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * hypre_CSRMatrixClone( hypre_CSRMatrix * A )
{
   int num_rows = hypre_CSRMatrixNumRows( A );
   int num_cols = hypre_CSRMatrixNumCols( A );
   int num_nonzeros = hypre_CSRMatrixNumNonzeros( A );
   hypre_CSRMatrix * B = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   int * A_i;
   int * A_j;
   int * B_i;
   int * B_j;
   int i, j;

   hypre_CSRMatrixInitialize( B );

   A_i = hypre_CSRMatrixI(A);
   A_j = hypre_CSRMatrixJ(A);
   B_i = hypre_CSRMatrixI(B);
   B_j = hypre_CSRMatrixJ(B);

   for ( i=0; i<num_rows+1; ++i )  B_i[i] = A_i[i];
   for ( j=0; j<num_nonzeros; ++j )  B_j[j] = A_j[j];
   hypre_CSRMatrixNumRownnz(B) =  hypre_CSRMatrixNumRownnz(A);
   if ( hypre_CSRMatrixRownnz(A) ) hypre_CSRMatrixSetRownnz( B );

   return B;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixUnion
 * Creates and returns a matrix whose elements are the union of those of A and B.
 * Data is not computed, only structural information is created.
 * A and B must have the same numbers of rows.
 * They need not have the same number of columns, if they were different I don't
 * know why anybody would want to call this function.
 * Nothing is done about Rownnz.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * hypre_CSRMatrixUnion( hypre_CSRMatrix * A, hypre_CSRMatrix * B )
{
   int num_rows = hypre_CSRMatrixNumRows( A );
   int num_cols_A = hypre_CSRMatrixNumCols( A );
   int num_cols_B = hypre_CSRMatrixNumCols( B );
   int num_nonzeros_A = hypre_CSRMatrixNumNonzeros( A );
   int num_nonzeros_B = hypre_CSRMatrixNumNonzeros( B );
   int num_cols = hypre_max( num_cols_A, num_cols_B );
   int num_nonzeros;
   int * A_i = hypre_CSRMatrixI(A);
   int * A_j = hypre_CSRMatrixJ(A);
   int * B_i = hypre_CSRMatrixI(B);
   int * B_j = hypre_CSRMatrixJ(B);
   int * C_i = hypre_CTAlloc( int, num_rows+1 );
   int * C_j;
   int * row_js = hypre_CTAlloc( int, num_cols );
   int i, j, ma, mb, mc;
   hypre_CSRMatrix * C;

   hypre_assert( num_rows == hypre_CSRMatrixNumRows(B) );

   /* The first run through A and B is to count the number of nonzero elements,
      without double-counting duplicates. */
   C_i[0] = 0;
   num_nonzeros = 0;
   for ( i=0; i<num_rows; ++i )
   {
      for ( j=0; j<num_cols; ++j )
         row_js[j] = 0;
      for ( ma=A_i[i]; ma<A_i[i+1]; ++ma )
         row_js[ A_j[ma] ] = 1;
      for ( mb=B_i[i]; mb<B_i[i+1]; ++mb )
         row_js[ B_j[mb] ] = 1;
      for ( j=0; j<num_cols; ++j )
         if ( row_js[j]>0 ) ++num_nonzeros;
   }

   C = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   hypre_CSRMatrixInitialize( C );

   /* The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
   C_i = hypre_CSRMatrixI(C);
   C_i[0] = 0;
   C_j = hypre_CSRMatrixJ(C);
   mc = 0;
   for ( i=0; i<num_rows; ++i )
   {
      for ( j=0; j<num_cols; ++j )
         row_js[j] = 0;
      for ( ma=A_i[i]; ma<A_i[i+1]; ++ma )
         row_js[ A_j[ma] ] = 1;
      for ( mb=B_i[i]; mb<B_i[i+1]; ++mb )
         row_js[ B_j[mb] ] = 1;
      for ( j=0; j<num_cols; ++j )
      {
         if ( row_js[j]>0 )
         {
            C_j[mc] = j;
            ++mc;
         }
      }
      C_i[i+1] = mc;
   }

   hypre_assert( mc == num_nonzeros );

   hypre_TFree( row_js );

   return C;
}
