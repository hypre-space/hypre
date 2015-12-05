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

#include "headers.h"
#include "bcsr_matrix.h"


#define ORDER_TEST 1

/*****************************************************************************
 *
 * Builds an interpolation operator based on all given information.
 * Full block version.
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixBuildInterp(hypre_BCSRMatrix* A, HYPRE_Int* CF_marker,
			    hypre_CSRMatrix* S, HYPRE_Int coarse_size) {
  hypre_BCSRMatrixBlock** A_blocks;
  HYPRE_Int* A_i;
  HYPRE_Int* A_j;

  HYPRE_Int* S_i;
  HYPRE_Int* S_j;

  hypre_BCSRMatrix* P; 

  hypre_BCSRMatrixBlock** P_blocks;
  HYPRE_Int* P_i;
  HYPRE_Int* P_j;

  HYPRE_Int P_size;
   
  HYPRE_Int* P_marker;

  HYPRE_Int jj_counter;
  HYPRE_Int jj_begin_row;
  HYPRE_Int jj_end_row;

  HYPRE_Int start_indexing = 0; /* start indexing for P_blocks at 0 */

  HYPRE_Int n_fine;
  HYPRE_Int n_coarse;

  HYPRE_Int strong_f_marker;

  HYPRE_Int *fine_to_coarse;
  HYPRE_Int coarse_counter;

  HYPRE_Int i, i1, i2;
  HYPRE_Int jj, jj1;
/*   HYPRE_Int sgn; */

  HYPRE_Int num_rows_per_block = hypre_BCSRMatrixNumRowsPerBlock(A);
  HYPRE_Int num_cols_per_block = hypre_BCSRMatrixNumColsPerBlock(A);
   
  hypre_BCSRMatrixBlock* diagonal;
  hypre_BCSRMatrixBlock* sum;
  hypre_BCSRMatrixBlock* distribute;          
   
  hypre_BCSRMatrixBlock* zero;
  hypre_BCSRMatrixBlock* one;

  hypre_BCSRMatrixBlock* temp;

  double* data;
   
  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for A and S. Also get size of fine grid.
   *-----------------------------------------------------------------------*/

  A_blocks = hypre_BCSRMatrixBlocks(A);
  A_i = hypre_BCSRMatrixI(A);
  A_j = hypre_BCSRMatrixJ(A);

  S_i = hypre_CSRMatrixI(S);
  S_j = hypre_CSRMatrixJ(S);

  n_fine = hypre_BCSRMatrixNumBlockRows(A);

  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

  coarse_counter = 0;

  fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);

  jj_counter = start_indexing;
      
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/
    
  for (i = 0; i < n_fine; i++) {
      
    /*--------------------------------------------------------------------
     *  If i is a c-point, interpolation is the identity. Also set up
     *  mapping vector.
     *--------------------------------------------------------------------*/

    if (CF_marker[i] >= 0) {
      jj_counter++;
      fine_to_coarse[i] = coarse_counter;
      coarse_counter++;
    }
      
    /*--------------------------------------------------------------------
     *  If i is a f-point, interpolation is from the C-points that
     *  strongly influence i.
     *--------------------------------------------------------------------*/

    else {
      for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	i1 = S_j[jj];           
	if (CF_marker[i1] >= 0)
	  {
	    jj_counter++;
	  }
      }
    }
  }
   
  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/

  n_coarse = coarse_counter;

  P_size = jj_counter;

  P = hypre_BCSRMatrixCreate(n_fine, n_coarse, P_size,
			     num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixInitialise(P);
  P_i = hypre_BCSRMatrixI(P);
  P_j = hypre_BCSRMatrixJ(P);
  P_blocks = hypre_BCSRMatrixBlocks(P);

  P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);

  /*-----------------------------------------------------------------------
   *  Second Pass: Define interpolation and fill in P_blocks, P_i, and P_j.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

  data = hypre_CTAlloc(double, num_rows_per_block*num_cols_per_block);
  zero = hypre_BCSRMatrixBlockCreate(num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixBlockInitialise(zero);
  hypre_BCSRMatrixBlockFillData(zero, data);
  for(i = 0; i < num_rows_per_block; i++) {
    data[i*num_cols_per_block + i] = 1.0;
  }
  one = hypre_BCSRMatrixBlockCreate(num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixBlockInitialise(one);
  hypre_BCSRMatrixBlockFillData(one, data);
  hypre_TFree(data);

  for (i = 0; i < n_fine; i++) {      
    P_marker[i] = -1;
  }
   
  strong_f_marker = -2;

  jj_counter = start_indexing;

  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
    
  for (i = 0; i  < n_fine  ; i ++) {

    /*--------------------------------------------------------------------
     *  If i is a c-point, interpolation is the identity.
     *--------------------------------------------------------------------*/
      
    if (CF_marker[i] > 0) {
      P_i[i] = jj_counter;
      P_j[jj_counter]    = fine_to_coarse[i];
      P_blocks[jj_counter] = hypre_BCSRMatrixBlockCopy(one);
      jj_counter++;
    }
      
      
    /*--------------------------------------------------------------------
     *  If i is a f-point, build interpolation.
     *--------------------------------------------------------------------*/

    else {
      P_i[i] = jj_counter;
      jj_begin_row = jj_counter;

      for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	i1 = S_j[jj];   
	
	/*--------------------------------------------------------------
	 * If nieghbor i1 is a c-point, set column number in P_j and
	 * initialize interpolation weight to zero.
	 *--------------------------------------------------------------*/
	
	if (CF_marker[i1] >= 0) {
	  P_marker[i1] = jj_counter;
	  P_j[jj_counter]    = fine_to_coarse[i1];
	  P_blocks[jj_counter] = hypre_BCSRMatrixBlockCopy(zero);
	  jj_counter++;
	}

	/*--------------------------------------------------------------
	 * If nieghbor i1 is a f-point, mark it as a strong f-point
	 * whose connection needs to be distributed.
	 *--------------------------------------------------------------*/

	else {
	  P_marker[i1] = strong_f_marker;
	}            
      }

      jj_end_row = jj_counter;
         
      diagonal = hypre_BCSRMatrixBlockCopy(A_blocks[A_i[i]]);
         
      for (jj = A_i[i]+1; jj < A_i[i+1]; jj++) {
	i1 = A_j[jj];

	/*--------------------------------------------------------------
	 * Case 1: nieghbor i1 is a c-point and strongly influences i,
	 * accumulate a_{i,i1} into the interpolation weight.
	 *--------------------------------------------------------------*/

	if (P_marker[i1] >= jj_begin_row) {
	  hypre_BCSRMatrixBlockAdd(P_blocks[P_marker[i1]], A_blocks[jj]);
	}
 
	/*--------------------------------------------------------------
	 * Case 2: nieghbor i1 is a f-point and strongly influences i,
	 * distribute a_{i,i1} to c-points that strongly infuence i.
	 * Note: currently no distribution to the diagonal in this case.
	 *--------------------------------------------------------------*/
            
	else if (P_marker[i1] == strong_f_marker) {
	  sum = hypre_BCSRMatrixBlockCopy(zero);
               
	  /*-----------------------------------------------------------
	   * Loop over row of A for point i1 and calculate the sum
	   * of the connections to c-points that strongly influence i.
	   *-----------------------------------------------------------*/

	  for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++) {
	    i2 = A_j[jj1];
	    if (P_marker[i2] >= jj_begin_row) {
	      hypre_BCSRMatrixBlockAdd(sum, A_blocks[jj1]);
	    }
	  }
               
#if ORDER_TEST
          /* distribute = A_blocks * (sum)^-1 - this is how it should be - code
            has not been written well, though...yet...*/  
          distribute = hypre_BCSRMatrixBlockCopy(A_blocks[jj]);
          hypre_BCSRMatrixBlockMultiplyInverse2(sum, distribute);
          distribute = hypre_BCSRMatrixBlockCopy(sum);
#else         
          /* distribute = sum^(-1)*A_blocks */             
	  distribute = hypre_BCSRMatrixBlockCopy(A_blocks[jj]);
	  hypre_BCSRMatrixBlockMulInv(distribute, sum);
#endif
          
	  /*-----------------------------------------------------------
	   * Loop over row of A for point i1 and do the distribution.
	   *-----------------------------------------------------------*/

	  for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++) {
	    i2 = A_j[jj1];
	    if (P_marker[i2] >= jj_begin_row) {
	      temp = hypre_BCSRMatrixBlockCopy(distribute);
	      hypre_BCSRMatrixBlockMultiply(temp, A_blocks[jj1]);
	      hypre_BCSRMatrixBlockAdd(P_blocks[P_marker[i2]], temp);
	      hypre_BCSRMatrixBlockDestroy(temp);
	    }
	  }
	  hypre_BCSRMatrixBlockDestroy(distribute);
	}
   
	/*--------------------------------------------------------------
	 * Case 3: nieghbor i1 weakly influences i, accumulate a_{i,i1}
	 * into the diagonal. This is done only if i and i1 are of the
	 * same function type.
	 *--------------------------------------------------------------*/

	else {
	  hypre_BCSRMatrixBlockAdd(diagonal, A_blocks[jj]);
	}            
      }

      /*-----------------------------------------------------------------
       * Set interpolation weight by dividing by the diagonal.
       *-----------------------------------------------------------------*/

      for (jj = jj_begin_row; jj < jj_end_row; jj++) {
	hypre_BCSRMatrixBlockMulInv(P_blocks[jj], diagonal);
	hypre_BCSRMatrixBlockNeg(P_blocks[jj]);
      }
      hypre_BCSRMatrixBlockDestroy(diagonal);
    }
   
    /*--------------------------------------------------------------------
     * Interpolation formula for i is done, update marker for strong
     * f connections for next i.
     *--------------------------------------------------------------------*/
   
    strong_f_marker--;
  }
  
  P_i[n_fine] = jj_counter;
  
  /*-----------------------------------------------------------------------
   *  Free mapping vector and marker array.
   *-----------------------------------------------------------------------*/

  hypre_TFree(P_marker);   
  hypre_TFree(fine_to_coarse);   
 
  return P;
}

/*****************************************************************************
 *
 * Builds an interpolation operator based on all given information.
 * Diagonal version.
 *
 *****************************************************************************/

hypre_BCSRMatrix*
hypre_BCSRMatrixBuildInterpD(hypre_BCSRMatrix* A, HYPRE_Int* CF_marker,
			    hypre_CSRMatrix* S, HYPRE_Int coarse_size) {
  hypre_BCSRMatrixBlock** A_blocks;
  HYPRE_Int* A_i;
  HYPRE_Int* A_j;

  HYPRE_Int* S_i;
  HYPRE_Int* S_j;

  hypre_BCSRMatrix* P; 

  hypre_BCSRMatrixBlock** P_blocks;
  HYPRE_Int* P_i;
  HYPRE_Int* P_j;

  HYPRE_Int P_size;
   
  HYPRE_Int* P_marker;

  HYPRE_Int jj_counter;
  HYPRE_Int jj_begin_row;
  HYPRE_Int jj_end_row;

  HYPRE_Int start_indexing = 0; /* start indexing for P_blocks at 0 */

  HYPRE_Int n_fine;
  HYPRE_Int n_coarse;

  HYPRE_Int strong_f_marker;

  HYPRE_Int *fine_to_coarse;
  HYPRE_Int coarse_counter;

  HYPRE_Int i, i1, i2;
  HYPRE_Int jj, jj1;
/*   HYPRE_Int sgn; */

  HYPRE_Int num_rows_per_block = hypre_BCSRMatrixNumRowsPerBlock(A);
  HYPRE_Int num_cols_per_block = hypre_BCSRMatrixNumColsPerBlock(A);
   
  hypre_BCSRMatrixBlock* diagonal;
  hypre_BCSRMatrixBlock* sum;
  hypre_BCSRMatrixBlock* distribute;          
   
  hypre_BCSRMatrixBlock* zero;
  hypre_BCSRMatrixBlock* one;

  hypre_BCSRMatrixBlock* temp;
  hypre_BCSRMatrixBlock* temp2;

  double* data;
   
  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for A and S. Also get size of fine grid.
   *-----------------------------------------------------------------------*/

  A_blocks = hypre_BCSRMatrixBlocks(A);
  A_i = hypre_BCSRMatrixI(A);
  A_j = hypre_BCSRMatrixJ(A);

  S_i = hypre_CSRMatrixI(S);
  S_j = hypre_CSRMatrixJ(S);

  n_fine = hypre_BCSRMatrixNumBlockRows(A);

  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

  coarse_counter = 0;

  fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine);

  jj_counter = start_indexing;
      
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/
    
  for (i = 0; i < n_fine; i++) {
      
    /*--------------------------------------------------------------------
     *  If i is a c-point, interpolation is the identity. Also set up
     *  mapping vector.
     *--------------------------------------------------------------------*/

    if (CF_marker[i] >= 0) {
      jj_counter++;
      fine_to_coarse[i] = coarse_counter;
      coarse_counter++;
    }
      
    /*--------------------------------------------------------------------
     *  If i is a f-point, interpolation is from the C-points that
     *  strongly influence i.
     *--------------------------------------------------------------------*/

    else {
      for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	i1 = S_j[jj];           
	if (CF_marker[i1] >= 0)
	  {
	    jj_counter++;
	  }
      }
    }
  }
   
  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/

  n_coarse = coarse_counter;

  P_size = jj_counter;

  P = hypre_BCSRMatrixCreate(n_fine, n_coarse, P_size,
			     num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixInitialise(P);
  P_i = hypre_BCSRMatrixI(P);
  P_j = hypre_BCSRMatrixJ(P);
  P_blocks = hypre_BCSRMatrixBlocks(P);

  P_marker = hypre_CTAlloc(HYPRE_Int, n_fine);

  /*-----------------------------------------------------------------------
   *  Second Pass: Define interpolation and fill in P_blocks, P_i, and P_j.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

  data = hypre_CTAlloc(double, num_rows_per_block*num_cols_per_block);
  zero = hypre_BCSRMatrixBlockCreate(num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixBlockInitialise(zero);
  hypre_BCSRMatrixBlockFillData(zero, data);
  for(i = 0; i < num_rows_per_block; i++) {
    data[i*num_cols_per_block + i] = 1.0;
  }
  one = hypre_BCSRMatrixBlockCreate(num_rows_per_block, num_cols_per_block);
  hypre_BCSRMatrixBlockInitialise(one);
  hypre_BCSRMatrixBlockFillData(one, data);
  hypre_TFree(data);

  for (i = 0; i < n_fine; i++) {      
    P_marker[i] = -1;
  }
   
  strong_f_marker = -2;

  jj_counter = start_indexing;

  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
    
  for (i = 0; i  < n_fine  ; i ++) {

    /*--------------------------------------------------------------------
     *  If i is a c-point, interpolation is the identity.
     *--------------------------------------------------------------------*/
      
    if (CF_marker[i] > 0) {
      P_i[i] = jj_counter;
      P_j[jj_counter]    = fine_to_coarse[i];
      P_blocks[jj_counter] = hypre_BCSRMatrixBlockCopy(one);
      jj_counter++;
    }
      
      
    /*--------------------------------------------------------------------
     *  If i is a f-point, build interpolation.
     *--------------------------------------------------------------------*/

    else {
      P_i[i] = jj_counter;
      jj_begin_row = jj_counter;

      for (jj = S_i[i]; jj < S_i[i+1]; jj++) {
	i1 = S_j[jj];   
	
	/*--------------------------------------------------------------
	 * If nieghbor i1 is a c-point, set column number in P_j and
	 * initialize interpolation weight to zero.
	 *--------------------------------------------------------------*/
	
	if (CF_marker[i1] >= 0) {
	  P_marker[i1] = jj_counter;
	  P_j[jj_counter]    = fine_to_coarse[i1];
	  P_blocks[jj_counter] = hypre_BCSRMatrixBlockCopy(zero);
	  jj_counter++;
	}

	/*--------------------------------------------------------------
	 * If nieghbor i1 is a f-point, mark it as a strong f-point
	 * whose connection needs to be distributed.
	 *--------------------------------------------------------------*/

	else {
	  P_marker[i1] = strong_f_marker;
	}            
      }

      jj_end_row = jj_counter;
         
      diagonal = hypre_BCSRMatrixBlockDiag(A_blocks[A_i[i]]);
         
      for (jj = A_i[i]+1; jj < A_i[i+1]; jj++) {
	i1 = A_j[jj];

	/*--------------------------------------------------------------
	 * Case 1: nieghbor i1 is a c-point and strongly influences i,
	 * accumulate a_{i,i1} into the interpolation weight.
	 *--------------------------------------------------------------*/

	if (P_marker[i1] >= jj_begin_row) {
	  temp = hypre_BCSRMatrixBlockDiag(A_blocks[jj]);
	  hypre_BCSRMatrixBlockAdd(P_blocks[P_marker[i1]], temp);
	  hypre_BCSRMatrixBlockDestroy(temp);
	}
 
	/*--------------------------------------------------------------
	 * Case 2: nieghbor i1 is a f-point and strongly influences i,
	 * distribute a_{i,i1} to c-points that strongly infuence i.
	 * Note: currently no distribution to the diagonal in this case.
	 *--------------------------------------------------------------*/
            
	else if (P_marker[i1] == strong_f_marker) {
	  sum = hypre_BCSRMatrixBlockCopy(zero);
               
	  /*-----------------------------------------------------------
	   * Loop over row of A for point i1 and calculate the sum
	   * of the connections to c-points that strongly influence i.
	   *-----------------------------------------------------------*/

	  for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++) {
	    i2 = A_j[jj1];
	    if (P_marker[i2] >= jj_begin_row) {
	      temp = hypre_BCSRMatrixBlockDiag(A_blocks[jj1]);
	      hypre_BCSRMatrixBlockAdd(sum, temp);
	      hypre_BCSRMatrixBlockDestroy(temp);
	    }
	  }
               
	  distribute = hypre_BCSRMatrixBlockDiag(A_blocks[jj]);
	  hypre_BCSRMatrixBlockMulInv(distribute, sum);
               
	  /*-----------------------------------------------------------
	   * Loop over row of A for point i1 and do the distribution.
	   *-----------------------------------------------------------*/

	  for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1++) {
	    i2 = A_j[jj1];
	    if (P_marker[i2] >= jj_begin_row) {
	      temp = hypre_BCSRMatrixBlockCopy(distribute);
	      temp2 = hypre_BCSRMatrixBlockDiag(A_blocks[jj1]);
	      hypre_BCSRMatrixBlockMultiply(temp, temp2);
	      hypre_BCSRMatrixBlockAdd(P_blocks[P_marker[i2]], temp);
	      hypre_BCSRMatrixBlockDestroy(temp);
	      hypre_BCSRMatrixBlockDestroy(temp2);
	    }
	  }
	  hypre_BCSRMatrixBlockDestroy(distribute);
	}
   
	/*--------------------------------------------------------------
	 * Case 3: nieghbor i1 weakly influences i, accumulate a_{i,i1}
	 * into the diagonal. This is done only if i and i1 are of the
	 * same function type.
	 *--------------------------------------------------------------*/

	else {
	  temp = hypre_BCSRMatrixBlockDiag(A_blocks[jj]);
	  hypre_BCSRMatrixBlockAdd(diagonal, temp);
	  hypre_BCSRMatrixBlockDestroy(temp);
	}            
      }

      /*-----------------------------------------------------------------
       * Set interpolation weight by dividing by the diagonal.
       *-----------------------------------------------------------------*/

      for (jj = jj_begin_row; jj < jj_end_row; jj++) {
	hypre_BCSRMatrixBlockMulInv(P_blocks[jj], diagonal);
	hypre_BCSRMatrixBlockNeg(P_blocks[jj]);
      }
      hypre_BCSRMatrixBlockDestroy(diagonal);
    }
   
    /*--------------------------------------------------------------------
     * Interpolation formula for i is done, update marker for strong
     * f connections for next i.
     *--------------------------------------------------------------------*/
   
    strong_f_marker--;
  }
  
  P_i[n_fine] = jj_counter;
  
  /*-----------------------------------------------------------------------
   *  Free mapping vector and marker array.
   *-----------------------------------------------------------------------*/

  hypre_TFree(P_marker);   
  hypre_TFree(fine_to_coarse);   
 
  return P;
}
