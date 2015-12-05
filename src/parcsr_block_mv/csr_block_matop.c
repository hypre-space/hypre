/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 1.4 $
 ***********************************************************************EHEADER*/



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

