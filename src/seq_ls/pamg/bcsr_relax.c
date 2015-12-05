/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * BCSR Matrix relaxation scheme
 *
 *****************************************************************************/

#include "headers.h"
#include "bcsr_matrix.h"


/*--------------------------------------------------------------------------
 *
 * hypre_BCSRMatrixRelax
 * (symmetric Gauss Seidel)
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BCSRMatrixRelax(hypre_BCSRMatrix *A, hypre_Vector *f, HYPRE_Int *cf_marker,
			  HYPRE_Int relax_points, hypre_Vector *u) {
   hypre_BCSRMatrixBlock** A_blocks = hypre_BCSRMatrixBlocks(A);
   HYPRE_Int* A_i = hypre_BCSRMatrixI(A);
   HYPRE_Int* A_j = hypre_BCSRMatrixJ(A);

   HYPRE_Int n = hypre_BCSRMatrixNumBlockRows(A);
   HYPRE_Int num_rows_per_block = hypre_BCSRMatrixNumRowsPerBlock(A);
   
   double *u_data  = hypre_VectorData(u);
   double *f_data  = hypre_VectorData(f);

   hypre_BCSRMatrixBlock* inv
     = hypre_BCSRMatrixBlockCreate(num_rows_per_block,
				   num_rows_per_block);

   double* res = hypre_CTAlloc(double, num_rows_per_block);

   double* eye;

   HYPRE_Int i, ii, jj;

   HYPRE_Int relax_error = 0;

   hypre_BCSRMatrixBlockInitialise(inv);
   eye = hypre_CTAlloc(double, num_rows_per_block*num_rows_per_block);
   for(i = 0; i < num_rows_per_block; i++) {
     eye[i*num_rows_per_block + i] = 1.0;
   }

   /*-----------------------------------------------------------------
    * Relax all points.
    *-----------------------------------------------------------------*/

   if (relax_points == 0) {
     for (i = 0; i < n; i++) {
       /*-----------------------------------------------------------
	* relax point i... maybe should check for singular diagonal?
	*-----------------------------------------------------------*/
       for(jj = 0; jj < num_rows_per_block; jj++) {
	 res[jj] = f_data[i*num_rows_per_block + jj];
       }
       for (jj = A_i[i] + 1; jj < A_i[i + 1]; jj++) {
	 ii = A_j[jj];
	 hypre_BCSRMatrixBlockMatvec(-1.0, A_blocks[jj],
				     &(u_data[ii*num_rows_per_block]),
				     1.0, res);
       }
       hypre_BCSRMatrixBlockFillData(inv, eye);
       hypre_BCSRMatrixBlockMulInv(inv, A_blocks[A_i[i]]);
       hypre_BCSRMatrixBlockMatvec(1.0, inv, res,
				   0.0, &(u_data[i*num_rows_per_block]));
     }
     for (i = n - 1; i > -1; i--) {
       /*-----------------------------------------------------------
	* relax point i... maybe should check for singular diagonal?
	*-----------------------------------------------------------*/
       for(jj = 0; jj < num_rows_per_block; jj++) {
	 res[jj] = f_data[i*num_rows_per_block + jj];
       }
       for (jj = A_i[i] + 1; jj < A_i[i + 1]; jj++) {
	 ii = A_j[jj];
	 hypre_BCSRMatrixBlockMatvec(-1.0, A_blocks[jj],
				     &(u_data[ii*num_rows_per_block]),
				     1.0, res);
       }
       hypre_BCSRMatrixBlockFillData(inv, eye);
       hypre_BCSRMatrixBlockMulInv(inv, A_blocks[A_i[i]]);
       hypre_BCSRMatrixBlockMatvec(1.0, inv, res,
				   0.0, &(u_data[i*num_rows_per_block]));
     }
   }

   /*-----------------------------------------------------------------
    * Relax only C or F points as determined by relax_points.
    *-----------------------------------------------------------------*/
   else {
     for (i = 0; i < n; i++) {
       /*-----------------------------------------------------------
	* If i is of the right type ( C or F )
	* should also check for nonsingular diagonal?
	*-----------------------------------------------------------*/
       if (cf_marker[i] == relax_points) {
	 for(jj = 0; jj < num_rows_per_block; jj++) {
	   res[jj] = f_data[i*num_rows_per_block + jj];
	 }
	 for (jj = A_i[i] + 1; jj < A_i[i + 1]; jj++) {
	   ii = A_j[jj];
	   hypre_BCSRMatrixBlockMatvec(-1.0, A_blocks[jj],
				       &(u_data[ii*num_rows_per_block]),
				       1.0, res);
	 }
	 hypre_BCSRMatrixBlockFillData(inv, eye);
	 hypre_BCSRMatrixBlockMulInv(inv, A_blocks[A_i[i]]);
	 hypre_BCSRMatrixBlockMatvec(1.0, inv, res,
				     0.0, &(u_data[i*num_rows_per_block]));
       }
     }
     for (i = n - 1; i > -1; i--) {
       /*-----------------------------------------------------------
	* If i is of the right type ( C or F )
	* should also check for nonsingular diagonal?
	*-----------------------------------------------------------*/
       if (cf_marker[i] == relax_points) {
	 for(jj = 0; jj < num_rows_per_block; jj++) {
	   res[jj] = f_data[i*num_rows_per_block + jj];
	 }
	 for (jj = A_i[i] + 1; jj < A_i[i + 1]; jj++) {
	   ii = A_j[jj];
	   hypre_BCSRMatrixBlockMatvec(-1.0, A_blocks[jj],
				       &(u_data[ii*num_rows_per_block]),
				       1.0, res);
	 }
	 hypre_BCSRMatrixBlockFillData(inv, eye);
	 hypre_BCSRMatrixBlockMulInv(inv, A_blocks[A_i[i]]);
	 hypre_BCSRMatrixBlockMatvec(1.0, inv, res,
				     0.0, &(u_data[i*num_rows_per_block]));
       }
     }     
   }

   return(relax_error); 
}
