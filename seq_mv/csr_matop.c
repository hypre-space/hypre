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
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_Matadd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 	 in A and B. To remove those, use hypre_DeleteZerosInMatrix 
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_Matadd( hypre_CSRMatrix *A,
              hypre_CSRMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         nrows_A  = hypre_CSRMatrixNumRows(A);
   int         ncols_A  = hypre_CSRMatrixNumCols(A);
   double     *B_data   = hypre_CSRMatrixData(B);
   int        *B_i      = hypre_CSRMatrixI(B);
   int        *B_j      = hypre_CSRMatrixJ(B);
   int         nrows_B  = hypre_CSRMatrixNumRows(B);
   int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *C;
   double     *C_data;
   int	      *C_i;
   int        *C_j;

   int         ia, ib, ic, jcol, num_nonzeros;
   int	       pos;
   int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
              printf("Warning! incompatible matrix dimensions!\n");
	      return NULL;
   }


   marker = hypre_CTAlloc(int, ncols_A);
   C_i = hypre_CTAlloc(int, nrows_A+1);

   for (ia = 0; ia < ncols_A; ia++)
	marker[ia] = -1;

   num_nonzeros = 0;
   C_i[0] = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		jcol = A_j[ia];
		marker[jcol] = ic;
		num_nonzeros++;
	}
	for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
	{
		jcol = B_j[ib];
		if (marker[jcol] != ic)
		{
			marker[jcol] = ic;
			num_nonzeros++;
		}
   	}
	C_i[ic+1] = num_nonzeros;
   }

   C = hypre_CreateCSRMatrix(nrows_A, ncols_A, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_InitializeCSRMatrix(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
	marker[ia] = -1;

   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		jcol = A_j[ia];
		C_j[pos] = jcol;
		C_data[pos] = A_data[ia];
		marker[jcol] = pos;
		pos++;
	}
	for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
	{
		jcol = B_j[ib];
		if (marker[jcol] < C_i[ic])
		{
			C_j[pos] = jcol;
			C_data[pos] = B_data[ib];
			marker[jcol] = pos;
			pos++;
		}
		else
		{
			C_data[marker[jcol]] += B_data[ib];
		}
   	}
   }

   hypre_TFree(marker);
   return C;
}	

/*--------------------------------------------------------------------------
 * hypre_Matmul
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 	 in A and B. To remove those, use hypre_DeleteZerosInMatrix 
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_Matmul( hypre_CSRMatrix *A,
              hypre_CSRMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         nrows_A  = hypre_CSRMatrixNumRows(A);
   int         ncols_A  = hypre_CSRMatrixNumCols(A);
   double     *B_data   = hypre_CSRMatrixData(B);
   int        *B_i      = hypre_CSRMatrixI(B);
   int        *B_j      = hypre_CSRMatrixJ(B);
   int         nrows_B  = hypre_CSRMatrixNumRows(B);
   int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *C;
   double     *C_data;
   int	      *C_i;
   int        *C_j;

   int         ia, ib, ic, ja, jb, num_nonzeros=0;
   int	       row_start, counter;
   double      a_entry, b_entry;
   int         *B_marker;

   if (ncols_A != nrows_B)
   {
              printf("Warning! incompatible matrix dimensions!\n");
	      return NULL;
   }


   B_marker = hypre_CTAlloc(int, ncols_B);
   C_i = hypre_CTAlloc(int, nrows_A+1);

   for (ib = 0; ib < ncols_B; ib++)
	B_marker[ib] = -1;

   for (ic = 0; ic < nrows_A; ic++)
   {
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		ja = A_j[ia];
		for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
		{
			jb = B_j[ib];
			if (B_marker[jb] != ic)
			{
				B_marker[jb] = ic;
				num_nonzeros++;
			}
		}
		C_i[ic+1] = num_nonzeros;
   	}
   }

   C = hypre_CreateCSRMatrix(nrows_A, ncols_B, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_InitializeCSRMatrix(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ib = 0; ib < ncols_B; ib++)
	B_marker[ib] = -1;

   counter = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
	row_start = C_i[ic];
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
		ja = A_j[ia];
		a_entry = A_data[ia];
		for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
		{
			jb = B_j[ib];
			b_entry = B_data[ib];
			if (B_marker[jb] < row_start)
			{
				B_marker[jb] = counter;
				C_j[B_marker[jb]] = jb;
				C_data[B_marker[jb]] = a_entry*b_entry;
				counter++;
			}
			else
				C_data[B_marker[jb]] += a_entry*b_entry;
				 
		}
	}
   }
   hypre_TFree(B_marker);
   return C;
}	

hypre_CSRMatrix *
hypre_DeleteZerosInMatrix( hypre_CSRMatrix *A, double tol)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         nrows_A  = hypre_CSRMatrixNumRows(A);
   int         ncols_A  = hypre_CSRMatrixNumCols(A);
   int         num_nonzeros  = hypre_CSRMatrixNumNonzeros(A);

   hypre_CSRMatrix *B;
   double     *B_data; 
   int        *B_i;
   int        *B_j;

   int zeros;
   int i, j;
   int pos_A, pos_B;

   zeros = 0;
   for (i=0; i < num_nonzeros; i++)
	if ( abs(A_data[i]) <= tol)
		zeros++;

   if (zeros)
   {
	B = hypre_CreateCSRMatrix(nrows_A,ncols_A,num_nonzeros-zeros);
	hypre_InitializeCSRMatrix(B);
	B_i = hypre_CSRMatrixI(B);
	B_j = hypre_CSRMatrixJ(B);
	B_data = hypre_CSRMatrixData(B);
	B_i[0] = 0;
	pos_A = 0;
	pos_B = 0;
	for (i=0; i < nrows_A; i++)
	{
		for (j = A_i[i]; j < A_i[i+1]; j++)
		{
		   if (abs(A_data[j]) <= tol)
		   {
			pos_A++;
		   }
		   else
		   {
			B_data[pos_B] = A_data[pos_A];
			B_j[pos_B] = A_j[pos_A];
			pos_B++;
			pos_A++;
		   }
		}
		B_i[i+1] = pos_B;
	}
	return B;
   }
   else
	return NULL;
}	
