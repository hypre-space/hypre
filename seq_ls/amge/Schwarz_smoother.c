/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h" 

int hypre_ComputeSchwarzSmoother(int *i_domain_dof,
				 int *j_domain_dof,
				 int num_domains,

				 hypre_CSRMatrix *A,

				 double **domain_matrixinverse_pointer)

{
  int ierr = 0;

  int i,j,k, j_loc, k_loc;

  double *domain_matrixinverse;

  int *i_dof_dof =  hypre_CSRMatrixI(A);
  int *j_dof_dof =  hypre_CSRMatrixJ(A);
  double *a_dof_dof =  hypre_CSRMatrixData(A);
  
  int num_dofs = hypre_CSRMatrixNumRows(A);
	
  double *matrix, *matrixinverse;

  int max_matrix_size, matrix_size, matrix_size_counter = 0;

  int *i_global_to_local;

  int local_dof_counter;


  max_matrix_size = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1]-i_domain_dof[i];
      if (max_matrix_size < matrix_size)
	max_matrix_size = matrix_size;

      matrix_size_counter+= matrix_size * matrix_size;
    }


  matrix        = hypre_CTAlloc(double, max_matrix_size* max_matrix_size);
  matrixinverse = hypre_CTAlloc(double, max_matrix_size* max_matrix_size);

  domain_matrixinverse = hypre_CTAlloc(double, matrix_size_counter);


  i_global_to_local = hypre_CTAlloc(int, num_dofs);
  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;


  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1]-i_domain_dof[i];
      local_dof_counter = 0;
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  i_global_to_local[j_domain_dof[j]] = local_dof_counter;
	  local_dof_counter++;
	}

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  /* j_loc = i_global_to_local[j_domain_dof[j]]; */
	  j_loc = j - i_domain_dof[i];
	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      /* k_loc = i_global_to_local[j_domain_dof[k]]; */
	      k_loc = k - i_domain_dof[i];
	      
	      matrix[j_loc +  matrix_size* k_loc] = 0.e0;
	    }
	}
      
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  /* j_loc = i_global_to_local[j_domain_dof[j]]; */
	  j_loc = j - i_domain_dof[i];

	  for (k = i_dof_dof[j_domain_dof[j]];
	       k < i_dof_dof[j_domain_dof[j]+1]; k++)
	    {
	      k_loc = i_global_to_local[j_dof_dof[k]];
	      if (k_loc > -1)
		matrix[j_loc +  matrix_size* k_loc] = a_dof_dof[k];
	    }
	}

      ierr = mat_inv(matrixinverse, matrix, matrix_size);
      if (ierr < 0)
	{
	  printf("ERROR: non--positive definite local (subdomain) matrix; \n");
	  return ierr;
	}

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  /* j_loc = i_global_to_local[j_domain_dof[j]]; */
	  j_loc = j - i_domain_dof[i];

	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      /* k_loc = i_global_to_local[j_domain_dof[k]]; */
	      k_loc = k - i_domain_dof[i];
	      
	      domain_matrixinverse[matrix_size_counter+
				  j_loc +  matrix_size* k_loc] = 
	      matrixinverse[j_loc +  matrix_size* k_loc];
	    }
	}

      matrix_size_counter+=matrix_size*matrix_size;

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_global_to_local[j_domain_dof[j]] = -1;

    }

  *domain_matrixinverse_pointer = domain_matrixinverse;

  hypre_TFree(matrix);
  hypre_TFree(matrixinverse);
  hypre_TFree(i_global_to_local);

  return ierr;
}
  
int hypre_SchwarzSolve(double *x,
		       double *rhs,
		       double *aux,

		       int *i_dof_dof,
		       int *j_dof_dof,
		       double *a_dof_dof,

		       int *i_domain_dof,
		       int *j_domain_dof,
		       int num_domains,

		       double *domain_matrixinverse,

		       int num_dofs)

{
  int ierr = 0;

  int i,j,k, j_loc, k_loc;


  int matrix_size, matrix_size_counter = 0;


  /* initiate:      ----------------------------------------------- */
  for (i=0; i < num_dofs; i++)
    x[i] = 0.e0;
  
  /* forward solve: ----------------------------------------------- */

  matrix_size_counter = 0;
  for (i=0; i < num_domains; i++)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];

      /* compute residual: ---------------------------------------- */

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[j_domain_dof[j]] = rhs[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[j_domain_dof[j]] -= a_dof_dof[k] * x[j_dof_dof[k]];
	}

      /* solve for correction: ------------------------------------- */
      
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j-i_domain_dof[i];

	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      k_loc = k-i_domain_dof[i];
	      x[j_domain_dof[j]]+= 
		domain_matrixinverse[matrix_size_counter 
				    + j_loc + k_loc * matrix_size]
		* aux[j_domain_dof[k]];
	    }
	}

      matrix_size_counter += matrix_size * matrix_size;  

    }

  /* backward solve: ------------------------------------------------ */
  for (i=num_domains-1; i > -1; i--)
    {
      matrix_size = i_domain_dof[i+1] - i_domain_dof[i];
      matrix_size_counter -= matrix_size * matrix_size;
      
      /* compute residual: ---------------------------------------- */

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux[j_domain_dof[j]] = rhs[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    aux[j_domain_dof[j]] -= a_dof_dof[k] * x[j_dof_dof[k]];
	}


      /* solve for correction: ------------------------------------- */
      
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  j_loc = j-i_domain_dof[i];

	  for (k=i_domain_dof[i]; k < i_domain_dof[i+1]; k++)
	    {
	      k_loc = k-i_domain_dof[i];
	      x[j_domain_dof[j]]+= 
		domain_matrixinverse[matrix_size_counter 
				    + j_loc + k_loc * matrix_size]
		* aux[j_domain_dof[k]];
	    }
	}
      
    }			      

  return ierr;

}
