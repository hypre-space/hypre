/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/

/*****************************************************************************
 *
 * This code implements a class for a dense block of a compressed sparse row
 * matrix.
 *
 *****************************************************************************/

#include "headers.h"
#include "bcsr_matrix_dense_block.h"

/*****************************************************************************
 *
 * Functions
 *
 *****************************************************************************/

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCreate(HYPRE_Int num_rows, HYPRE_Int num_cols) {
  hypre_BCSRMatrixDenseBlock* A;

  A = hypre_CTAlloc(hypre_BCSRMatrixDenseBlock, 1);

  A->data = NULL;
  A->num_rows = num_rows;
  A->num_cols = num_cols;

  return A;
  

}

HYPRE_Int
hypre_BCSRMatrixDenseBlockDestroy(hypre_BCSRMatrixDenseBlock* A) {
  if(A) {
    hypre_TFree(A->data);
    hypre_TFree(A);
  }

  return 0;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockInitialise(hypre_BCSRMatrixDenseBlock* A) {
  if(!A->data) {
    A->data = hypre_CTAlloc(double, A->num_rows*A->num_cols);
  }

  return 0;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockFillData(hypre_BCSRMatrixDenseBlock* A,
				   double* data) {
  HYPRE_Int i;

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    A->data[i] = data[i];
  }

  return 0;
  

}

HYPRE_Int
hypre_BCSRMatrixDenseBlockGetData(hypre_BCSRMatrixDenseBlock* A,
				   double* data) {
  HYPRE_Int i;

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    data[i] = A->data[i];
  }

  return 0;
  

}

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockCopy(hypre_BCSRMatrixDenseBlock* A) {
  hypre_BCSRMatrixDenseBlock* B;

  B = hypre_BCSRMatrixDenseBlockCreate(A->num_rows, A->num_cols);
  hypre_BCSRMatrixDenseBlockInitialise(B);
  hypre_BCSRMatrixDenseBlockFillData(B, A->data);

  return B;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockAdd(hypre_BCSRMatrixDenseBlock* A,
			      hypre_BCSRMatrixDenseBlock* B) {
  HYPRE_Int i;

  if(A->num_rows != B->num_rows || A->num_cols != B->num_cols) {
    return -1;
  }

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    A->data[i] += B->data[i];
  }
  return 0;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiply(hypre_BCSRMatrixDenseBlock* A,
				   hypre_BCSRMatrixDenseBlock* B) {
  HYPRE_Int i, j, k;
  double t[A->num_rows*A->num_cols];

  if(A->num_rows != A->num_cols || A->num_rows != B->num_rows
     || A->num_rows != B->num_cols) {
    return -1;
  }

  for(i = 0; i < A->num_rows; i++) {
    for(j = 0; j < A->num_cols; j++) {
      t[i*A->num_cols + j] = 0.0;
      for(k = 0; k < A->num_rows; k++) {
	t[i*A->num_cols + j] +=
	  A->data[i*A->num_cols + k]*B->data[k*A->num_cols + j];
      }
    }
  }

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    A->data[i] = t[i];
  }

  return 0;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockNeg(hypre_BCSRMatrixDenseBlock* A) {
  HYPRE_Int i;

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    A->data[i] = -A->data[i];
  }

  return 0;
}

hypre_BCSRMatrixDenseBlock*
hypre_BCSRMatrixDenseBlockDiag(hypre_BCSRMatrixDenseBlock* A) {
  hypre_BCSRMatrixDenseBlock* B;
  HYPRE_Int i;

  B = hypre_BCSRMatrixDenseBlockCreate(A->num_rows, A->num_cols);
  hypre_BCSRMatrixDenseBlockInitialise(B);

  for(i = 0; i < A->num_rows; i++) {
    B->data[i*A->num_cols + i] = A->data[i*A->num_cols + i];
  }

  return B;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockMulInv(hypre_BCSRMatrixDenseBlock* A,
				 hypre_BCSRMatrixDenseBlock* B) {

 /* AHB 9/05: this is actually returning B{^-1}A in A*.  If B cannot be 
    inverted, returns A = A */

  HYPRE_Int i, j, k;
  HYPRE_Int num_rows = A->num_rows, num_cols = A->num_cols;
  double T[A->num_rows*A->num_cols];
  double Bi[A->num_rows*A->num_cols];
  double d;

  if(A->num_rows != A->num_cols || A->num_rows != B->num_rows
     || A->num_rows != B->num_cols) {
    return -1;
  }

  for(i = 0; i < num_rows*num_cols; i++) {
    T[i] = B->data[i];
    Bi[i] = 0.0;
  }
  for(i = 0; i < num_rows; i++) {
    Bi[i*num_cols + i] = 1.0;
  }

  for(i = 0; i < num_rows; i++) {
    d = T[i*num_cols + i];
    if(fabs(d) < 1.0e-6) {
       /* pivoting needed? */
       for(j = i + 1; j < num_rows; j++) {
	if(fabs(T[j*num_cols + i]) >= 1.0e-12) {
	  for(k = 0; k < num_cols; k++) {
	    d = T[j*num_cols + k];
	    T[j*num_cols + k] = T[i*num_cols + k];
	    T[i*num_cols + k] = d;
	    d = Bi[j*num_cols + k];
	    Bi[j*num_cols + k] = Bi[i*num_cols + k];
	    Bi[i*num_cols + k] = d;
	  }
	  break;
	}
      }
      d = T[i*num_cols + i];
      if(fabs(d) < 1.0e-6) {
	hypre_printf("Singular matrix block being inverted?!\n");
	hypre_BCSRMatrixDenseBlockPrint(B, stdout);
	return -2;
      }
    }
   /* gauss-jordan method */
    for(j = 0; j < num_cols; j++) {
      T[i*num_cols + j] = T[i*num_cols + j]/d;
      Bi[i*num_cols + j] = Bi[i*num_cols + j]/d;
    }
    for(k = i + 1; k < num_rows; k++) {
      d = -T[k*num_cols + i];
      for(j = 0; j < num_cols; j++) {
	T[k*num_cols + j] += d*T[i*num_cols + j];
	Bi[k*num_cols + j] += d*Bi[i*num_cols + j];
      }
    }
    for(k = 0; k < i; k++) {
      d = -T[k*num_cols + i];
      for(j = 0; j < num_cols; j++) {
	T[k*num_cols + j] += d*T[i*num_cols + j];
	Bi[k*num_cols + j] += d*Bi[i*num_cols + j];
      }
    }
  }


  /* inv(B)*A */

  for(i = 0; i < A->num_rows; i++) {
    for(j = 0; j < A->num_cols; j++) {
      T[i*A->num_cols + j] = 0.0;
      for(k = 0; k < A->num_rows; k++) {
	T[i*A->num_cols + j] +=
	  Bi[i*A->num_cols + k]*A->data[k*A->num_cols + j];
      }
    }
  }
  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    A->data[i] = T[i];
  }

  return 0;
}





HYPRE_Int
hypre_BCSRMatrixDenseBlockMultiplyInverse2(hypre_BCSRMatrixDenseBlock* A,
				 hypre_BCSRMatrixDenseBlock* B) 
{

  /* AHB:  here we return BA^{-1} in A.  If A cannot be inverted, returns
     A = B*/

   hypre_BCSRMatrixBlock *A_t, *B_t;
   double *out_data;
   HYPRE_Int ierr = 0;
   
   
   /* this is the lazy way - for testing */
   A_t = hypre_BCSRMatrixBlockCopy(A);
   hypre_BCSRMatrixDenseBlockTranspose(A_t);
   
   B_t = hypre_BCSRMatrixBlockCopy(B);
   hypre_BCSRMatrixBlockTranspose(B_t);

   /* now B_t = inv(A_t)*B_t */
   ierr = hypre_BCSRMatrixDenseBlockMulInv(B_t, A_t);
   
   /*result in B_t -copy to A and then take transpose */
   out_data = hypre_CTAlloc(double, A->num_rows*A->num_cols);

   hypre_BCSRMatrixDenseBlockGetData(B_t, out_data);
   hypre_BCSRMatrixDenseBlockFillData(A, out_data);   
   hypre_BCSRMatrixDenseBlockTranspose(A);
   
   hypre_BCSRMatrixDenseBlockDestroy(A_t);
   hypre_BCSRMatrixDenseBlockDestroy(B_t);
   
   hypre_TFree(out_data);
   

   return ierr;
   
   
}


HYPRE_Int
hypre_BCSRMatrixDenseBlockTranspose(hypre_BCSRMatrixDenseBlock* A) {
  HYPRE_Int num_rows = A->num_rows;
  HYPRE_Int num_cols = A->num_cols;
  double t[num_rows*num_cols];
  HYPRE_Int i, j;

  for(i = 0; i < num_rows; i++) {
    for(j = 0; j < num_cols; j++) {
      t[j*num_rows + i] = A->data[i*num_cols + j];
    }
  }
  for(i = 0; i < num_rows*num_cols; i++) {
    A->data[i] = t[i];
  }
  A->num_rows = num_cols;
  A->num_cols = num_rows;

  return 0;
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockMatvec(double alpha, hypre_BCSRMatrixBlock* A,
				 double* x_data, double beta, double* y_data) {
  HYPRE_Int num_rows = A->num_rows;
  HYPRE_Int num_cols = A->num_cols;
  HYPRE_Int i, j;
  double temp;
  HYPRE_Int ierr = 0;
  double* A_data = A->data;

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0) {
    for (i = 0; i < num_rows; i++) y_data[i] *= beta;

    return ierr;
  }


  /*-----------------------------------------------------------------------
   * y = (beta/alpha)*y
   *-----------------------------------------------------------------------*/
   
  temp = beta / alpha;
   
  if (temp != 1.0) {
    if (temp == 0.0) {
      for (i = 0; i < num_rows; i++) y_data[i] = 0.0;
    }
    else {
      for (i = 0; i < num_rows; i++) y_data[i] *= temp;
    }
  }

  /*-----------------------------------------------------------------
   * y += A*x
   *-----------------------------------------------------------------*/

  for (i = 0; i < num_rows; i++) {
    for (j = 0; j < num_cols; j++)
      y_data[i] += A_data[i*num_cols + j] * x_data[j];
  }

  /*-----------------------------------------------------------------
   * y = alpha*y
   *-----------------------------------------------------------------*/

  if (alpha != 1.0) {
    for (i = 0; i < num_rows; i++) y_data[i] *= alpha;
  }

  return 0;
  

}

HYPRE_Int
hypre_BCSRMatrixDenseBlockMatvecT(double alpha, hypre_BCSRMatrixBlock* A,
				  double* x_data, double beta,
				  double* y_data) {
  HYPRE_Int num_rows = A->num_rows;
  HYPRE_Int num_cols = A->num_cols;
  HYPRE_Int i, j;
  double temp;
  HYPRE_Int ierr = 0;
  double* A_data = A->data;

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0) {
    for (i = 0; i < num_cols; i++) y_data[i] *= beta;

    return ierr;
  }


  /*-----------------------------------------------------------------------
   * y = (beta/alpha)*y
   *-----------------------------------------------------------------------*/
   
  temp = beta / alpha;
   
  if (temp != 1.0) {
    if (temp == 0.0) {
      for (i = 0; i < num_cols; i++) y_data[i] = 0.0;
    }
    else {
      for (i = 0; i < num_cols; i++) y_data[i] *= temp;
    }
  }

  /*-----------------------------------------------------------------
   * y += A^T*x
   *-----------------------------------------------------------------*/

  for (i = 0; i < num_rows; i++) {
    for (j = 0; j < num_cols; j++)
      y_data[j] += A_data[i*num_cols + j] * x_data[i];
  }

  /*-----------------------------------------------------------------
   * y = alpha*y
   *-----------------------------------------------------------------*/

  if (alpha != 1.0) {
    for (i = 0; i < num_cols; i++) y_data[i] *= alpha;
  }

  return 0;
  

}

double
hypre_BCSRMatrixDenseBlockNorm(hypre_BCSRMatrixDenseBlock* A,
			       const char* norm) {
  HYPRE_Int num_rows = A->num_rows;
  HYPRE_Int num_cols = A->num_cols;
  double* data = A->data;
  HYPRE_Int i, j;

  if(!strcmp(norm, "one")) {
    double col_sums[num_cols];
    double max_col_sum;

    memset(col_sums, 0, sizeof(double)*num_cols);

    for(i = 0; i < num_rows; i++) {
      for(j = 0; j < num_cols; j++) {
	col_sums[j] += fabs(data[i*num_cols + j]);
      }
    }

    max_col_sum = col_sums[0];
    for(i = 1; i < num_cols; i++) {
      if(col_sums[i] > max_col_sum) max_col_sum = col_sums[i];
    }

    return max_col_sum;
  }
  else if(!strcmp(norm, "inf")) {
    double row_sums[num_rows];
    double max_row_sum;

    memset(row_sums, 0, sizeof(double)*num_rows);

    for(i = 0; i < num_rows; i++) {
      for(j = 0; j < num_cols; j++) {
	row_sums[i] += fabs(data[i*num_cols + j]);
      }
    }

    max_row_sum = row_sums[0];
    for(i = 1; i < num_rows; i++) {
      if(row_sums[i] > max_row_sum) max_row_sum = row_sums[i];
    }

    return max_row_sum;
  }
  else {
    /* Froebenius is the default */
    double sum = 0;

    for(i = 0; i < num_rows; i++) {
      for(j = 0; j < num_cols; j++) {
	sum += data[i*num_cols + j]*data[i*num_cols + j];
      }
    }

    sum = sqrt(sum);


    return sum;
  }
}

HYPRE_Int
hypre_BCSRMatrixDenseBlockPrint(hypre_BCSRMatrixDenseBlock* A,
				FILE* out_file) {
  HYPRE_Int i;

  for(i = 0; i < A->num_rows*A->num_cols; i++) {
    hypre_fprintf(out_file, "%f ", A->data[i]);
  }
  hypre_fprintf(out_file, "\n");

  return 0;
  

}
