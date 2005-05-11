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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 	 in A and B. To remove those, use hypre_CSRMatrixDeleteZeros 
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixAdd(hypre_CSRBlockMatrix *A, hypre_CSRBlockMatrix *B)
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

   int         block_size  = hypre_CSRBlockMatrixBlockSize(A); 
   int         block_sizeB = hypre_CSRBlockMatrixBlockSize(B); 
   int         ia, ib, ic, ii, jcol, num_nonzeros, bnnz;
   int	       pos;
   int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   marker = hypre_CTAlloc(int, ncols_A);
   C_i = hypre_CTAlloc(int, nrows_A+1);

   for (ia = 0; ia < ncols_A; ia++) marker[ia] = -1;

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

   C = hypre_CSRBlockMatrixCreate(block_size,nrows_A,ncols_A,num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++) marker[ia] = -1;

   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         for (ii = 0; ii < bnnz; ii++)
            C_data[pos*bnnz+ii] = A_data[ia*bnnz+ii];
         marker[jcol] = pos;
         pos++;
      }
      for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] < C_i[ic])
         {
            C_j[pos] = jcol;
            for (ii = 0; ii < bnnz; ii++)
               C_data[pos*bnnz+ii] = B_data[ib*bnnz+ii];
            marker[jcol] = pos;
            pos++;
         }
	 else 
         {
            for (ii = 0; ii < bnnz; ii++)
               C_data[marker[jcol]*bnnz+ii] = B_data[ib*bnnz+ii];
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

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixMultiply(hypre_CSRBlockMatrix *A, hypre_CSRBlockMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   int        *A_i      = hypre_CSRMatrixI(A);
   int        *A_j      = hypre_CSRMatrixJ(A);
   int         nrows_A  = hypre_CSRMatrixNumRows(A);
   int         ncols_A  = hypre_CSRMatrixNumCols(A);
   int         block_size  = hypre_CSRBlockMatrixBlockSize(A); 
   double     *B_data   = hypre_CSRMatrixData(B);
   int        *B_i      = hypre_CSRMatrixI(B);
   int        *B_j      = hypre_CSRMatrixJ(B);
   int         nrows_B  = hypre_CSRMatrixNumRows(B);
   int         ncols_B  = hypre_CSRMatrixNumCols(B);
   int         block_sizeB = hypre_CSRBlockMatrixBlockSize(B); 
   hypre_CSRMatrix *C;
   double     *C_data;
   int	      *C_i;
   int        *C_j;

   int         ia, ib, ic, ja, jb, num_nonzeros=0, bnnz;
   int	       row_start, counter;
   double      *a_entries, *b_entries, *c_entries, dzero=0.0, done=1.0;
   int         *B_marker;

   if (ncols_A != nrows_B)
   {
      printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   B_marker = hypre_CTAlloc(int, ncols_B);
   C_i = hypre_CTAlloc(int, nrows_A+1);

   for (ib = 0; ib < ncols_B; ib++) B_marker[ib] = -1;

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

   C = hypre_CSRBlockMatrixCreate(block_size,nrows_A,ncols_B,num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ib = 0; ib < ncols_B; ib++) B_marker[ib] = -1;

   counter = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      row_start = C_i[ic];
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         ja = A_j[ia];
         a_entries = &(A_data[ia*bnnz]);
         for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
         {
            jb = B_j[ib];
            b_entries = &(B_data[ib*bnnz]);
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[B_marker[jb]] = jb;
               c_entries = &(C_data[B_marker[jb]*bnnz]);
               hypre_CSRBlockMatrixBlockMultAdd(a_entries,b_entries,dzero,
                                                c_entries, block_size);
               counter++;
            }
            else
            {
               c_entries = &(C_data[B_marker[jb]*bnnz]);
               hypre_CSRBlockMatrixBlockMultAdd(a_entries,b_entries,done,
                                                c_entries, block_size);
            }
         }
      }
   }
   hypre_TFree(B_marker);
   return C;
}	

/******************************************************************************
 * Finds transpose of a hypre_CSRMatrix
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/

int hypre_CSRMatrixTranspose(hypre_CSRMatrix   *A, hypre_CSRMatrix   **AT,
				int data)

{
   double       *A_data = hypre_CSRMatrixData(A);
   int          *A_i = hypre_CSRMatrixI(A);
   int          *A_j = hypre_CSRMatrixJ(A);
   int           num_rowsA = hypre_CSRMatrixNumRows(A);
   int           num_colsA = hypre_CSRMatrixNumCols(A);
   int           num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);
   int           block_size  = hypre_CSRBlockMatrixBlockSize(A); 

   double       *AT_data;
   int          *AT_i;
   int          *AT_j;
   int           num_rowsAT;
   int           num_colsAT;
   int           num_nonzerosAT;

   int           max_col;
   int           i, j, k, ind, bnnz;

   /*-------------------------------------------------------------- 
    * First, ascertain that num_cols and num_nonzeros has been set. 
    * If not, set them.
    *--------------------------------------------------------------*/

   if (! num_nonzerosA) num_nonzerosA = A_i[num_rowsA];

   if (num_rowsA && ! num_colsA)
   {
      max_col = -1;
      for (i = 0; i < num_rowsA; ++i)
      {
         for (j = A_i[i]; j < A_i[i+1]; j++)
            if (A_j[j] > max_col) max_col = A_j[j];
      }
      num_colsA = max_col+1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;
   bnnz = block_size * block_size;

   *AT = hypre_CSRBlockMatrixCreate(block_size, num_rowsAT, 
                                    num_colsAT, num_nonzerosAT);

   AT_i = hypre_CTAlloc(int, num_rowsAT+1);
   AT_j = hypre_CTAlloc(int, num_nonzerosAT);
   hypre_CSRMatrixI(*AT) = AT_i;
   hypre_CSRMatrixJ(*AT) = AT_j;
   if (data) 
   {
      AT_data = hypre_CTAlloc(double, num_nonzerosAT*bnnz);
      hypre_CSRMatrixData(*AT) = AT_data;
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
            ind = AT_i[A_j[j]];
            for (k = 0; k < bnnz; k++)
               AT_data[ind*bnnz+k] = A_data[j*bnnz+k];
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

