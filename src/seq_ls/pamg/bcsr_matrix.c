/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/

/*****************************************************************************
 *
 * This code implements a class for block compressed sparse row matrices.
 *
 *****************************************************************************/

#include "headers.h"
#include "bcsr_matrix.h"

/*****************************************************************************
 *
 * Creates a BCSR Matrix.  Use Initialise to allocate all necessary memory.
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixCreate(HYPRE_Int num_block_rows, HYPRE_Int num_block_cols,
		       HYPRE_Int num_nonzero_blocks,
		       HYPRE_Int num_rows_per_block, HYPRE_Int num_cols_per_block) {
  hypre_BCSRMatrix* A;

  A = hypre_CTAlloc(hypre_BCSRMatrix, 1);

  hypre_BCSRMatrixBlocks(A) = NULL;
  hypre_BCSRMatrixI(A) = NULL;
  hypre_BCSRMatrixJ(A) = NULL;
  hypre_BCSRMatrixNumBlockRows(A) = num_block_rows;
  hypre_BCSRMatrixNumBlockCols(A) = num_block_cols;
  hypre_BCSRMatrixNumNonzeroBlocks(A) = num_nonzero_blocks;
  hypre_BCSRMatrixNumRowsPerBlock(A) = num_rows_per_block;
  hypre_BCSRMatrixNumColsPerBlock(A) = num_cols_per_block;

  return A;
}

/*****************************************************************************
 *
 * Destroy a BCSR Matrix.
 *
 *****************************************************************************/

HYPRE_Int
hypre_BCSRMatrixDestroy(hypre_BCSRMatrix* A) {
  hypre_BCSRMatrixBlock** blocks;
  HYPRE_Int i;

  if(A) {
    hypre_TFree(hypre_BCSRMatrixI(A));
    hypre_TFree(hypre_BCSRMatrixJ(A));
    blocks = hypre_BCSRMatrixBlocks(A);
    if(blocks) {
      for(i = 0; i < hypre_BCSRMatrixNumNonzeroBlocks(A); i++) {
	hypre_BCSRMatrixBlockDestroy(blocks[i]);
      }
      hypre_TFree(blocks);
    }
    hypre_TFree(A);
  }

  return 0;
}

/*****************************************************************************
 *
 * Initialise a BCSR Matrix.  Allocates all necessary memory.
 *
 *****************************************************************************/

HYPRE_Int
hypre_BCSRMatrixInitialise(hypre_BCSRMatrix* A) {
  HYPRE_Int i;

  if(!hypre_BCSRMatrixBlocks(A) && hypre_BCSRMatrixNumNonzeroBlocks(A)) {
    hypre_BCSRMatrixBlocks(A) =
      hypre_CTAlloc(hypre_BCSRMatrixBlock*,
		    hypre_BCSRMatrixNumNonzeroBlocks(A));
    for(i = 0; i < hypre_BCSRMatrixNumNonzeroBlocks(A); i++) {
      hypre_BCSRMatrixBlocks(A)[i] = NULL;
    }
  }
  if(!hypre_BCSRMatrixI(A)) {
    hypre_BCSRMatrixI(A) =
      hypre_CTAlloc(HYPRE_Int, hypre_BCSRMatrixNumBlockRows(A) + 1);
  }
  if(!hypre_BCSRMatrixJ(A) && hypre_BCSRMatrixNumNonzeroBlocks(A)) {
    hypre_BCSRMatrixJ(A) =
      hypre_CTAlloc(HYPRE_Int, hypre_BCSRMatrixNumNonzeroBlocks(A));
  }

  return 0;
}

/*****************************************************************************
 *
 * Print a BCSR matrix to a file of the given name.
 *
 *****************************************************************************/

HYPRE_Int
hypre_BCSRMatrixPrint(hypre_BCSRMatrix* A, char* file_name) {
  FILE* out_file = fopen(file_name, "w");
  HYPRE_Int file_base = 1;
  HYPRE_Int i;

  hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixNumBlockRows(A));
  hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixNumBlockCols(A));
  hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixNumNonzeroBlocks(A));
  hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixNumRowsPerBlock(A));
  hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixNumColsPerBlock(A));

  for(i = 0; i < hypre_BCSRMatrixNumBlockRows(A) + 1; i++) {
    hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixI(A)[i] + file_base);
  }

  for(i = 0; i < hypre_BCSRMatrixNumNonzeroBlocks(A); i++) {
    hypre_fprintf(out_file, "%d\n", hypre_BCSRMatrixJ(A)[i] + file_base);
  }

  for(i = 0; i < hypre_BCSRMatrixNumNonzeroBlocks(A); i++) {
    hypre_BCSRMatrixBlockPrint(hypre_BCSRMatrixBlocks(A)[i], out_file);
  }


  return 0;
  

}

/*****************************************************************************
 *
 * Transpose a BCSR Matrix.
 *
 *****************************************************************************/

HYPRE_Int
hypre_BCSRMatrixTranspose(hypre_BCSRMatrix* A, hypre_BCSRMatrix** AT) {
   hypre_BCSRMatrixBlock*       *A_blocks = hypre_BCSRMatrixBlocks(A);
   HYPRE_Int          *A_i = hypre_BCSRMatrixI(A);
   HYPRE_Int          *A_j = hypre_BCSRMatrixJ(A);
   HYPRE_Int           num_rowsA = hypre_BCSRMatrixNumBlockRows(A);
   HYPRE_Int           num_colsA = hypre_BCSRMatrixNumBlockCols(A);
   HYPRE_Int           num_nonzerosA = hypre_BCSRMatrixNumNonzeroBlocks(A);
   HYPRE_Int num_rows_per_block = hypre_BCSRMatrixNumRowsPerBlock(A);
   HYPRE_Int num_cols_per_block = hypre_BCSRMatrixNumColsPerBlock(A);

   hypre_BCSRMatrixBlock*       *AT_blocks;
   HYPRE_Int          *AT_i;
   HYPRE_Int          *AT_j;
   HYPRE_Int           num_rowsAT;
   HYPRE_Int           num_colsAT;
   HYPRE_Int           num_nonzerosAT;

/*    HYPRE_Int           max_col; */
   HYPRE_Int           i, j;

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = hypre_BCSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT,
				num_cols_per_block, num_rows_per_block);
   hypre_BCSRMatrixInitialise(*AT);
   AT_i = hypre_BCSRMatrixI(*AT);
   AT_j = hypre_BCSRMatrixJ(*AT);
   AT_blocks = hypre_BCSRMatrixBlocks(*AT);

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
         AT_j[AT_i[A_j[j]]] = i;
	 AT_blocks[AT_i[A_j[j]]] = hypre_BCSRMatrixBlockCopy(A_blocks[j]);
	 hypre_BCSRMatrixBlockTranspose(AT_blocks[AT_i[A_j[j]]]);
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

/*****************************************************************************
 *
 * Creates a BCSR matrix from a CSR matrix assuming the given block size.
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixFromCSRMatrix(hypre_CSRMatrix* A,
			      HYPRE_Int num_rows_per_block, HYPRE_Int num_cols_per_block) {
  hypre_BCSRMatrix* B;
  HYPRE_Int* B_i;
  HYPRE_Int* B_j;
  hypre_BCSRMatrixBlock** B_blocks;
  HYPRE_Int* A_i = hypre_CSRMatrixI(A);
  HYPRE_Int* A_j = hypre_CSRMatrixJ(A);
  double* A_data = hypre_CSRMatrixData(A);
  HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
  HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
  HYPRE_Int num_block_rows = num_rows/num_rows_per_block;
  HYPRE_Int num_block_cols = num_cols/num_cols_per_block;
  HYPRE_Int num_nonzero_blocks;
  HYPRE_Int i, j, jA, t, d, i_block, j_block, jB;
  HYPRE_Int* block_flag;
  HYPRE_Int* block_number;
  double** blocks;

  /*--------------------------------------------------------------------------
   *
   * First pass: count number of nonzero blocks in A and assign the
   * block number for each element of A
   *
   *-------------------------------------------------------------------------*/

  block_flag = hypre_CTAlloc(HYPRE_Int, num_block_cols);
  block_number = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(A));

  num_nonzero_blocks = 0;
  for(i = 0; i < num_block_rows; i++) {
    memset(block_flag, -1, sizeof(HYPRE_Int)*num_block_cols);
    for(d = 0; d < num_rows_per_block; d++) {
      t = i*num_rows_per_block + d;
      for(jA = A_i[t]; jA < A_i[t + 1]; jA++) {
	j = A_j[jA];
	if(block_flag[j/num_cols_per_block] < 0) {
	  block_number[jA] = num_nonzero_blocks;
	  block_flag[j/num_cols_per_block] = block_number[jA];
	  num_nonzero_blocks++;
	}
	else {
	  block_number[jA] = block_flag[j/num_cols_per_block];
	}
      }
    }
  }

  B = hypre_BCSRMatrixCreate(num_block_rows, num_block_cols,
			     num_nonzero_blocks,
			     num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixInitialise(B);
  blocks = hypre_CTAlloc(double*, num_nonzero_blocks);
  for(t = 0; t < num_nonzero_blocks; t++) {
    blocks[t] = hypre_CTAlloc(double, num_rows_per_block*num_cols_per_block);
    memset(blocks[t], 0, sizeof(double)*num_rows_per_block*num_cols_per_block);
  }

  /*--------------------------------------------------------------------------
   *
   * Second pass: fill in blocks array
   *
   *-------------------------------------------------------------------------*/

  for(i = 0; i < num_rows; i++) {
    for(jA = A_i[i]; jA < A_i[i + 1]; jA++) {
      j = A_j[jA];
      i_block = i%num_rows_per_block;
      j_block = j%num_cols_per_block;
      blocks[block_number[jA]][i_block*num_cols_per_block + j_block] =
	A_data[jA];
    }
  }

  /*--------------------------------------------------------------------------
   *
   * Third pass: fill in i, j, and blocks for B
   *
   *-------------------------------------------------------------------------*/

  B_i = hypre_BCSRMatrixI(B);
  B_j = hypre_BCSRMatrixJ(B);
  B_blocks = hypre_BCSRMatrixBlocks(B);

  for(i = 0; i < num_rows; i++) {
    if(i%num_rows_per_block == 0) {
      B_i[i/num_rows_per_block] = block_number[A_i[i]];
    }
    for(jA = A_i[i]; jA < A_i[i + 1]; jA++) {
      j = A_j[jA];
      jB = block_number[jA];
      if(B_blocks[jB] == NULL) {
	B_j[jB] = j/num_cols_per_block;
	B_blocks[jB] = hypre_BCSRMatrixBlockCreate(num_rows_per_block,
						   num_cols_per_block);
	hypre_BCSRMatrixBlockInitialise(B_blocks[jB]);
	hypre_BCSRMatrixBlockFillData(B_blocks[jB], blocks[jB]);
      }
    }
  }
  B_i[num_block_rows] = num_nonzero_blocks;

  return B;
}

/*****************************************************************************
 *
 * Creates a CSR matrix from a BCSR.
 *
 *****************************************************************************/

hypre_CSRMatrix*
hypre_BCSRMatrixToCSRMatrix(hypre_BCSRMatrix* B) {
  hypre_CSRMatrix* A;
  HYPRE_Int* A_i;
  HYPRE_Int* A_j;
  double* A_data;
  hypre_CSRMatrix* A_no_zeros;
  HYPRE_Int* B_i = hypre_BCSRMatrixI(B);
  HYPRE_Int* B_j = hypre_BCSRMatrixJ(B);
  hypre_BCSRMatrixBlock** B_blocks = hypre_BCSRMatrixBlocks(B);
  HYPRE_Int num_block_rows = hypre_BCSRMatrixNumBlockRows(B);
  HYPRE_Int num_block_cols = hypre_BCSRMatrixNumBlockCols(B);
  HYPRE_Int num_rows_per_block = hypre_BCSRMatrixNumRowsPerBlock(B);
  HYPRE_Int num_cols_per_block = hypre_BCSRMatrixNumColsPerBlock(B);
  HYPRE_Int num_rows = num_rows_per_block*num_block_rows;
  HYPRE_Int num_cols = num_cols_per_block*num_block_cols;
  HYPRE_Int num_nonzero_blocks = hypre_BCSRMatrixNumNonzeroBlocks(B);
  HYPRE_Int num_nonzeros = num_nonzero_blocks*num_rows_per_block*num_cols_per_block;
  HYPRE_Int i, j, k, l, d, jB;
  double* block = hypre_CTAlloc(double, num_rows_per_block*num_cols_per_block);

  A = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
  hypre_CSRMatrixInitialize(A);
  A_i = hypre_CSRMatrixI(A);
  A_j = hypre_CSRMatrixJ(A);
  A_data = hypre_CSRMatrixData(A);

  /* set A_i */
  A_i[0] = 0;
  for(i = 0; i < num_block_rows; i++) {
    for(k = 0; k < num_rows_per_block; k++) {
      A_i[i*num_rows_per_block + k + 1] = A_i[i*num_rows_per_block + k]
	+ (B_i[i + 1] - B_i[i])*num_cols_per_block;
    }
  }

  /* fill in A_j and A_data */
  for(i = 0; i < num_block_rows; i++) {
    for(jB = B_i[i]; jB < B_i[i + 1]; jB++) {
      j = B_j[jB];
      hypre_BCSRMatrixBlockGetData(B_blocks[jB], block);
      for(k = 0; k < num_rows_per_block; k++) {
	d = A_i[i*num_rows_per_block + k] + (jB - B_i[i])*num_cols_per_block;

	/* do diagonal element first */
	A_j[d] = j*num_cols_per_block + k;
	A_data[d] = block[k*num_cols_per_block + k];
	d++;
	for(l = 0; l < num_cols_per_block; l++) {
	  if(l != k) {
	    A_j[d] = j*num_cols_per_block + l;
	    A_data[d] = block[k*num_cols_per_block + l];
	    d++;
	  }
	}
      }
    }
  }


/*  A_no_zeros = hypre_CSRMatrixDeleteZeros(A, 0.0);*/
    A_no_zeros = hypre_CSRMatrixDeleteZeros(A, 1e-14);
  


  if(A_no_zeros) {
    hypre_CSRMatrixDestroy(A);

    return A_no_zeros;
  }

  return A;
}

/*****************************************************************************
 *
 * Compresses a BCSR matrix into a CSR matrix by replacing each block by its
 * (Froebenius) norm.
 *
 *****************************************************************************/

hypre_CSRMatrix*
hypre_BCSRMatrixCompress(hypre_BCSRMatrix* A) {
  HYPRE_Int num_block_rows = hypre_BCSRMatrixNumBlockRows(A);
  HYPRE_Int num_block_cols = hypre_BCSRMatrixNumBlockCols(A);
  HYPRE_Int num_nonzero_blocks = hypre_BCSRMatrixNumNonzeroBlocks(A);
  hypre_CSRMatrix* B = hypre_CSRMatrixCreate(num_block_rows, num_block_cols,
					     num_nonzero_blocks);
  HYPRE_Int i;

  hypre_CSRMatrixInitialize(B);
  for(i = 0; i < num_block_rows + 1; i++) {
    hypre_CSRMatrixI(B)[i] = hypre_BCSRMatrixI(A)[i];
  }
  for(i = 0; i < num_nonzero_blocks; i++) {
    hypre_CSRMatrixJ(B)[i] = hypre_BCSRMatrixJ(A)[i];
  }
  for(i = 0; i < num_nonzero_blocks; i++) {
    hypre_CSRMatrixData(B)[i] =
      hypre_BCSRMatrixBlockNorm(hypre_BCSRMatrixBlocks(A)[i], "froeb");
  /* "one", "inf" */
  }
  /* make all diagonal elements negative */
  for(i = 0; i < num_block_rows; i++) {
    hypre_CSRMatrixData(B)[hypre_CSRMatrixI(B)[i]] =
      -hypre_CSRMatrixData(B)[hypre_CSRMatrixI(B)[i]];
  }

  return B;
}
