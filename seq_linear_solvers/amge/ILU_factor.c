/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h"  

/*****************************************************************************
 *           ILU(1) factorization;
 *****************************************************************************
 * factorizes a: i_a, j_a, a_data, using change of variables (reordering);
 *     P^{-1} = ILUdof_to_dof from ILU-dof to original dof-ordering
 *     determined by a graph block_dof = (block_node x node_dof);
 * the factoziation is LD * U, where 
 *     LD = (L + D^{-1}): lower triangular + diagonal^{-1} part;
 *      U: unit upper triangular part;
 *  sparsity pattern of LD, U is based on that of a^2; 
 * returns -1 if factorization fails (i.e., 0 or negative diagonal pivot);
 ****************************************************************************/

int hypre_ILUfactor(int **i_ILUdof_to_dof_pointer,
		    /* int **i_dof_to_ILUdof_pointer, */

		    int **i_ILUdof_ILUdof_pointer,
		    int **j_ILUdof_ILUdof_pointer,
		    double **LD_data,

		    int **i_ILUdof_ILUdof_t_pointer,
		    int **j_ILUdof_ILUdof_t_pointer,
		    double **U_data,

		    hypre_CSRMatrix *A,

		    int *i_node_dof, int *j_node_dof,

		    int *i_block_node, int *j_block_node,
		    int num_blocks, 
				    
		    int num_dofs,
		    int num_nodes)

{

  int ierr = 0;
  int i, j, k, l;
  int i_dof;

  int j1, j2;

  int *i_block_dof, *j_block_dof;
  int *i_ILUdof_ILUdof, *j_ILUdof_ILUdof;

  int *i_dof_to_ILUdof, *i_ILUdof_to_dof;

  int *i_dof_dof_0, *j_dof_dof_0;
  int *i_dof_dof, *j_dof_dof;
  double *a_dof_dof;

  int *i_dof_index;

  int *i_a = hypre_CSRMatrixI(A);
  int *j_a = hypre_CSRMatrixJ(A);
  double *a_data = hypre_CSRMatrixData(A);

  double *b_dof_dof;
  double *b_dof_dof_t;

  int *i_ILUdof_ILUdof_t, *j_ILUdof_ILUdof_t;

  int ILUdof_ILUdof_counter = 0, ILUdof_ILUdof_t_counter=0;

  int i_dof_to_ILUdof_counter;

  double diag, diagonal, entry, row_sum;

  i_dof_to_ILUdof = hypre_CTAlloc(int, num_dofs);
  i_ILUdof_to_dof = hypre_CTAlloc(int, num_dofs);

  ierr = matrix_matrix_product(&i_block_dof, &j_block_dof,

			       i_block_node, j_block_node,
			       i_node_dof, j_node_dof,

			       num_blocks, num_nodes, num_dofs);

  /* -----------------------------------------------------------------
     i_ILUdof_to_dof gives dofs reordering from the ILU one
     to the original one. Similarly, i_dof_to_ILUdof gives dofs 
     reordering from the original to the ILU one;
     ----------------------------------------------------------------- */

  i_dof_to_ILUdof_counter = 0;
  for (l=0;  l < num_blocks; l++)
    for (j=i_block_dof[l]; j < i_block_dof[l+1]; j++)
      {
	i_dof_to_ILUdof[j_block_dof[j]] = i_dof_to_ILUdof_counter;


	i_ILUdof_to_dof[i_dof_to_ILUdof_counter] = j_block_dof[j];
      
	i_dof_to_ILUdof_counter++;
      }


  hypre_TFree(i_block_dof);
  hypre_TFree(j_block_dof);

  /* create sparsity pattern of ILU matrix based on a^2; -------------- */

  ierr = matrix_matrix_product(&i_dof_dof, &j_dof_dof,

			       i_a, j_a, 
			       i_a, j_a,

			       num_dofs, num_dofs, num_dofs);


  i_dof_index = hypre_CTAlloc(int, num_dofs);

  for (i=0; i< num_dofs; i++)
    i_dof_index[i] = -1;

  a_dof_dof =  hypre_CTAlloc(double, i_dof_dof[num_dofs]);

  /* fill-in a_dof_dof with a_data: ---------------------------------- */

  for (i=0; i < num_dofs; i++)
    {
      for (j=i_a[i]; j<i_a[i+1]; j++)
	i_dof_index[j_a[j]] = j;

      for (j=i_dof_dof[i]; j < i_dof_dof[i+1]; j++)
	{
	  if (i_dof_index[j_dof_dof[j]] > -1)
	    a_dof_dof[j] = a_data[i_dof_index[j_dof_dof[j]]];
	  else
	    a_dof_dof[j] = 0.e0;
	}

      for (j=i_a[i]; j<i_a[i+1]; j++)
	i_dof_index[j_a[j]] = -1;
    }

  hypre_TFree(i_dof_index);

  /* ==================================================================
     factorize the matrix: --------------------------------------------

     d_i = 1.e0 / a_{ii};
     
     a_{k,i} = a_{k,i} * d_i; k > i;

     a_{k,j}-= a_{k,i] * a_{i,i} * a_{i,j}, k,j > i;

     ================================================================== */

  i_ILUdof_ILUdof = hypre_CTAlloc(int, num_dofs+1);
  j_ILUdof_ILUdof = hypre_CTAlloc(int, (i_dof_dof[num_dofs]+num_dofs)/2);
  b_dof_dof = hypre_CTAlloc(double, (i_dof_dof[num_dofs]+num_dofs)/2);


  i_ILUdof_ILUdof_t =  hypre_CTAlloc(int, num_dofs+1);
  j_ILUdof_ILUdof_t = hypre_CTAlloc(int, (i_dof_dof[num_dofs]-num_dofs)/2);
  b_dof_dof_t = hypre_CTAlloc(double, (i_dof_dof[num_dofs]-num_dofs)/2);


  ILUdof_ILUdof_counter = 0;
  ILUdof_ILUdof_t_counter = 0;
  for (i=0; i < num_dofs; i++)
    {
      i_ILUdof_ILUdof_t[i] = ILUdof_ILUdof_t_counter;

      i_ILUdof_ILUdof[i] = ILUdof_ILUdof_counter;
      j_ILUdof_ILUdof[i_ILUdof_ILUdof[i]]=i;
      ILUdof_ILUdof_counter++;

      b_dof_dof[i_ILUdof_ILUdof[i]] = 0.e0;
      i_dof = i_ILUdof_to_dof[i];
      for (j=i_dof_dof[i_dof]; j < i_dof_dof[i_dof+1]; j++)
	{
	  if (j_dof_dof[j] == i_dof)
	    b_dof_dof[i_ILUdof_ILUdof[i]] += a_dof_dof[j];
	  else
	    if (i_dof_to_ILUdof[j_dof_dof[j]] < i)
	      {
		j_ILUdof_ILUdof[ILUdof_ILUdof_counter] = 
		  i_dof_to_ILUdof[j_dof_dof[j]];
		/* if (a_dof_dof[j] <= 0.e0) */
		  b_dof_dof[ILUdof_ILUdof_counter] = a_dof_dof[j];
		  /* else
		  {
		    b_dof_dof[ILUdof_ILUdof_counter] = 0.e0;
		    b_dof_dof[i_ILUdof_ILUdof[i]] += a_dof_dof[j];
		  } */
		ILUdof_ILUdof_counter++;
	      }
	    else
	      {
		j_ILUdof_ILUdof_t[ILUdof_ILUdof_t_counter] = 
		  i_dof_to_ILUdof[j_dof_dof[j]];
		/* if (a_dof_dof[j] <= 0.e0) */
		  b_dof_dof_t[ILUdof_ILUdof_t_counter] = a_dof_dof[j];
		  /* else
		  {
		    b_dof_dof_t[ILUdof_ILUdof_t_counter] = 0.e0;
		    b_dof_dof[i_ILUdof_ILUdof[i]] += a_dof_dof[j];
		  } */
		ILUdof_ILUdof_t_counter++;
	      }
	}
    }

  hypre_TFree(a_dof_dof);

  i_ILUdof_ILUdof[num_dofs] = ILUdof_ILUdof_counter;
  i_ILUdof_ILUdof_t[num_dofs] = ILUdof_ILUdof_t_counter;


  if (ILUdof_ILUdof_counter+ILUdof_ILUdof_t_counter != i_dof_dof[num_dofs])
    {
      printf("ERROR in extracting lower/upper triangular parts: *******\n");
      printf("lower_triang nnz: %d, upper_triang nnz: %d, total nnz: %d\n",
	     ILUdof_ILUdof_counter, ILUdof_ILUdof_t_counter, 
	     i_dof_dof[num_dofs]);
    }

  hypre_TFree(i_dof_dof);
  hypre_TFree(j_dof_dof);

  hypre_TFree(i_dof_to_ILUdof);

  /* FACTORIZATION PART: ===========================================*/

  for (i=0; i < num_dofs; i++)
    {
      diagonal = b_dof_dof[i_ILUdof_ILUdof[i]];

      if (diagonal <= 0.e0)
	{
	  printf("failure of ILU: non--positive diagonal entry: %e-----\n",
		 diagonal);
	  return -1;
	}
      else
	diag = 1.e0/diagonal;

      b_dof_dof[i_ILUdof_ILUdof[i]] = diag;

      for (j=i_ILUdof_ILUdof_t[i]; j < i_ILUdof_ILUdof_t[i+1]; j++)
	b_dof_dof_t[j] *= diag;

      /* form Schur complement: ------------------------------------*/

      for (j=i_ILUdof_ILUdof_t[i]; j < i_ILUdof_ILUdof_t[i+1]; j++)
	{
	  j1 = j_ILUdof_ILUdof_t[j];
	  for (k=i_ILUdof_ILUdof[j1]; k<i_ILUdof_ILUdof[j1+1]; k++)
	    if (j_ILUdof_ILUdof[k] == i)
	      {
		b_dof_dof[k]*=diag;
		entry = b_dof_dof[k];
		break;
	      }

	  /* row_sum = 0.e0;  */
	  for (k=i_ILUdof_ILUdof_t[i]; k<i_ILUdof_ILUdof_t[i+1]; k++)
	    {
	      j2 = j_ILUdof_ILUdof_t[k];
	      if (j1 >= j2)
		for (l=i_ILUdof_ILUdof[j1]; l<i_ILUdof_ILUdof[j1+1]; l++)
		  {
		    if (j_ILUdof_ILUdof[l] == j2)
		      {
			b_dof_dof[l] -= entry * diagonal * b_dof_dof_t[k];
			break;
		      }
		    /* else
		      row_sum-=entry * diagonal * b_dof_dof_t[k]; */
		  }
	      else
		for (l=i_ILUdof_ILUdof_t[j1]; l<i_ILUdof_ILUdof_t[j1+1]; l++)
		  if (j_ILUdof_ILUdof_t[l] == j2)
		    {
		      b_dof_dof_t[l] -= entry * diagonal * b_dof_dof_t[k];
		      break;
		    }
	      /* else
		    row_sum-= entry * diagonal * b_dof_dof_t[k]; */
	    }

	  /* b_dof_dof[i_ILUdof_ILUdof[j1]]+= row_sum;  */
	}
    }




  *i_ILUdof_to_dof_pointer = i_ILUdof_to_dof;
  /*  *i_dof_to_ILUdof_pointer = i_dof_to_ILUdof; */

  *i_ILUdof_ILUdof_pointer = i_ILUdof_ILUdof;
  *j_ILUdof_ILUdof_pointer = j_ILUdof_ILUdof;
  *LD_data = b_dof_dof;


  *i_ILUdof_ILUdof_t_pointer = i_ILUdof_ILUdof_t;
  *j_ILUdof_ILUdof_t_pointer = j_ILUdof_ILUdof_t;
  *U_data = b_dof_dof_t;

  /*
  printf("\n\n=======================================================\n\n");
  printf("                 E N D  ILU(1) FACTORIZATION:                \n");
  printf("\n\n=======================================================\n\n");
  */

  return ierr;

}
