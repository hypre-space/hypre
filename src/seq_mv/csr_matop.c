/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 	 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros 
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixAdd( hypre_CSRMatrix *A,
              hypre_CSRMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   double     *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *C;
   double     *C_data;
   HYPRE_Int	      *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, jcol, num_nonzeros;
   HYPRE_Int	       pos;
   HYPRE_Int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
              hypre_printf("Warning! incompatible matrix dimensions!\n");
	      return NULL;
   }


   marker = hypre_CTAlloc(HYPRE_Int, ncols_A);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1);

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

   C = hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
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
 * hypre_CSRMatrixMultiply
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 	 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros 
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixMultiply( hypre_CSRMatrix *A,
              hypre_CSRMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   double     *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *C;
   double     *C_data;
   HYPRE_Int	      *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, ja, jb, num_nonzeros=0;
   HYPRE_Int	       row_start, counter;
   double      a_entry, b_entry;
   HYPRE_Int         *B_marker;

   if (ncols_A != nrows_B)
   {
              hypre_printf("Warning! incompatible matrix dimensions!\n");
	      return NULL;
   }


   B_marker = hypre_CTAlloc(HYPRE_Int, ncols_B);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1);

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
   	}
	C_i[ic+1] = num_nonzeros;
   }

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
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
hypre_CSRMatrixDeleteZeros( hypre_CSRMatrix *A, double tol)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         num_nonzeros  = hypre_CSRMatrixNumNonzeros(A);

   hypre_CSRMatrix *B;
   double     *B_data; 
   HYPRE_Int        *B_i;
   HYPRE_Int        *B_j;

   HYPRE_Int zeros;
   HYPRE_Int i, j;
   HYPRE_Int pos_A, pos_B;

   zeros = 0;
   for (i=0; i < num_nonzeros; i++)
	if (fabs(A_data[i]) <= tol)
		zeros++;

   if (zeros)
   {
	B = hypre_CSRMatrixCreate(nrows_A,ncols_A,num_nonzeros-zeros);
	hypre_CSRMatrixInitialize(B);
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
		   if (fabs(A_data[j]) <= tol)
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


/******************************************************************************
 *
 * Finds transpose of a hypre_CSRMatrix
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/


HYPRE_Int hypre_CSRMatrixTranspose(hypre_CSRMatrix   *A, hypre_CSRMatrix   **AT,
				HYPRE_Int data)

{
   double       *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int          *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int          *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int           num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int           num_colsA = hypre_CSRMatrixNumCols(A);
   HYPRE_Int           num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);

   double       *AT_data;
   HYPRE_Int          *AT_i;
   HYPRE_Int          *AT_j;
   HYPRE_Int           num_rowsAT;
   HYPRE_Int           num_colsAT;
   HYPRE_Int           num_nonzerosAT;

   HYPRE_Int           max_col;
   HYPRE_Int           i, j;

   /*-------------------------------------------------------------- 
    * First, ascertain that num_cols and num_nonzeros has been set. 
    * If not, set them.
    *--------------------------------------------------------------*/

   if (! num_nonzerosA)
   {
      num_nonzerosA = A_i[num_rowsA];
   }

   if (num_rowsA && ! num_colsA)
   {
      max_col = -1;
      for (i = 0; i < num_rowsA; ++i)
      {
          for (j = A_i[i]; j < A_i[i+1]; j++)
          {
              if (A_j[j] > max_col)
                 max_col = A_j[j];
          }
      }
      num_colsA = max_col+1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = hypre_CSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT);

   AT_i = hypre_CTAlloc(HYPRE_Int, num_rowsAT+1);
   AT_j = hypre_CTAlloc(HYPRE_Int, num_nonzerosAT);
   hypre_CSRMatrixI(*AT) = AT_i;
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data) 
   {
      AT_data = hypre_CTAlloc(double, num_nonzerosAT);
      hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Count the number of entries in each column of A (row of AT)
    * and fill the AT_i array.
    *-----------------------------------------------------------------*/

   for (i = 0; i < num_nonzerosA; i++)
   {
       ++AT_i[A_j[i]+1];
   }

   for (i = 2; i <= num_rowsAT; i++)
   {
       AT_i[i] += AT_i[i-1];
   }

   /*----------------------------------------------------------------
    * Load the data and column numbers of AT
    *----------------------------------------------------------------*/

   for (i = 0; i < num_rowsA; i++)
   {
      for (j = A_i[i]; j < A_i[i+1]; j++)
      {
         hypre_assert( AT_i[A_j[j]] >= 0 );
         hypre_assert( AT_i[A_j[j]] < num_nonzerosAT );
         AT_j[AT_i[A_j[j]]] = i;
         if (data) AT_data[AT_i[A_j[j]]] = A_data[j];
         AT_i[A_j[j]]++;
      }
   }

   /*------------------------------------------------------------
    * AT_i[j] now points to the *end* of the jth row of entries
    * instead of the beginning.  Restore AT_i to front of row.
    *------------------------------------------------------------*/

   for (i = num_rowsAT; i > 0; i--)
   {
         AT_i[i] = AT_i[i-1];
   }

   AT_i[0] = 0;

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_CSRMatrixReorder:
 * Reorders the column and data arrays of a square CSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_CSRMatrixReorder(hypre_CSRMatrix *A)
{
   HYPRE_Int i, j, tempi, row_size;
   double tempd;

   double *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int    *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int    *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int     num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int     num_colsA = hypre_CSRMatrixNumCols(A);

   /* the matrix should be square */
   if (num_rowsA != num_colsA)
      return -1;

   for (i = 0; i < num_rowsA; i++)
   {
      row_size = A_i[i+1]-A_i[i];

      for (j = 0; j < row_size; j++)
      {
         if (A_j[j] == i)
         {
            if (j != 0)
            {
               tempi = A_j[0];
               A_j[0] = A_j[j];
               A_j[j] = tempi;

               tempd = A_data[0];
               A_data[0] = A_data[j];
               A_data[j] = tempd;
            }
            break;
         }

         /* diagonal element is missing */
         if (j == row_size-1)
            return -2;
      }

      A_j    += row_size;
      A_data += row_size;
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSumElts:
 * Returns the sum of all matrix elements.
 *--------------------------------------------------------------------------*/

double hypre_CSRMatrixSumElts( hypre_CSRMatrix *A )
{
   double sum = 0;
   double * data = hypre_CSRMatrixData( A );
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int i;

   for ( i=0; i<num_nonzeros; ++i ) sum += data[i];

   return sum;
}
