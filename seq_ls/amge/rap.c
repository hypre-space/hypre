/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/



/*****************************************************************************
 *    creates coarse matrix P^TAP from P (interpolation) and A (fine matrix)
 *    all in hypre_CSRMatrix format;
 ****************************************************************************/

#include "headers.h"  

int hypre_AMGeRAP(hypre_CSRMatrix **A_crs_pointer,
		  hypre_CSRMatrix *A, 
		  hypre_CSRMatrix *P)

{
  int ierr = 0;

  int i,j,k,m,n,p,q; 
  hypre_CSRMatrix *A_crs;

  int *i_dof_dof_c, *j_dof_dof_c;
  int *i_dof_dof_c_t, *j_dof_dof_c_t;
  int *i_dof_dof_b, *j_dof_dof_b;

  int *i_dof_dof[2], *j_dof_dof[2];

  double *c_dof_dof, *sparse_matrix[2];

  int Ndofs[2];

  Ndofs[0] =  hypre_CSRMatrixNumRows(P);
  Ndofs[1] =  hypre_CSRMatrixNumCols(P);


  i_dof_dof_c = hypre_CSRMatrixI(P);
  j_dof_dof_c = hypre_CSRMatrixJ(P);
  c_dof_dof = hypre_CSRMatrixData(P);

  i_dof_dof[0] = hypre_CSRMatrixI(A);
  j_dof_dof[0] = hypre_CSRMatrixJ(A);
  sparse_matrix[0] = hypre_CSRMatrixData(A);



  ierr = matrix_matrix_product(&i_dof_dof_b, &j_dof_dof_b,
			       i_dof_dof[0], j_dof_dof[0], 
			       i_dof_dof_c, j_dof_dof_c,

			       Ndofs[0], Ndofs[0],
			       Ndofs[1]);

  ierr = transpose_matrix_create(&i_dof_dof_c_t, &j_dof_dof_c_t,
				 i_dof_dof_c, j_dof_dof_c,
				 Ndofs[0], Ndofs[1]);


  ierr = matrix_matrix_product(&i_dof_dof[1], &j_dof_dof[1],
			       i_dof_dof_c_t, j_dof_dof_c_t, 
			       i_dof_dof_b, j_dof_dof_b,

			       Ndofs[1],  Ndofs[0], 
			       Ndofs[1]);	

  free(i_dof_dof_c_t);
  free(j_dof_dof_c_t);
      
  free(i_dof_dof_b);
  free(j_dof_dof_b);

  sparse_matrix[1] = hypre_CTAlloc(double, i_dof_dof[1][Ndofs[1]]);

  for (i=0; i < Ndofs[1]; i++)
    for (j=i_dof_dof[1][i]; j<i_dof_dof[1][i+1]; j++)
      sparse_matrix[1][j] = 0.e0;



  for (k=0; k < Ndofs[0]; k++)                          /* k is a finedof; */
    {
      for (n=i_dof_dof_c[k]; n < i_dof_dof_c[k+1]; n++)
	{
	  i = j_dof_dof_c[n];                           /* i is a coarsedof; */

	  for (m=i_dof_dof[0][k]; m < i_dof_dof[0][k+1]; m++)
	    {
	      j=j_dof_dof[0][m];                        /* j is a finedof; */

	      for (p=i_dof_dof_c[j]; p < i_dof_dof_c[j+1]; p++)
		{
		  for (q=i_dof_dof[1][i]; q<i_dof_dof[1][i+1]; q++)
		    {
		      if (j_dof_dof[1][q] == j_dof_dof_c[p])
			{
			  sparse_matrix[1][q] += c_dof_dof[n] * 
			    sparse_matrix[0][m] * c_dof_dof[p];
			  break;
			}
		    }
		}

	    }
	}
    }

  A_crs = hypre_CSRMatrixCreate(Ndofs[1], Ndofs[1],
			    i_dof_dof[1][Ndofs[1]]);

  /* printf("coarse matrix nnz: %d\n", i_dof_dof[1][Ndofs[1]]); */

  hypre_CSRMatrixData(A_crs) = sparse_matrix[1];
  hypre_CSRMatrixI(A_crs) = i_dof_dof[1];
  hypre_CSRMatrixJ(A_crs) = j_dof_dof[1];

  *A_crs_pointer = A_crs;


  return ierr;

}
