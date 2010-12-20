/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/






#include "headers.h" 

HYPRE_Int hypre_ComputeSchwarzSmoother(HYPRE_Int *i_domain_dof,
				 HYPRE_Int *j_domain_dof,
				 HYPRE_Int num_domains,

				 hypre_CSRMatrix *A,

				 double **domain_matrixinverse_pointer)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int i,j,k, j_loc, k_loc;

  double *domain_matrixinverse;

  HYPRE_Int *i_dof_dof =  hypre_CSRMatrixI(A);
  HYPRE_Int *j_dof_dof =  hypre_CSRMatrixJ(A);
  double *a_dof_dof =  hypre_CSRMatrixData(A);
  
  HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A);
	
  double *matrix, *matrixinverse;

  HYPRE_Int max_matrix_size, matrix_size, matrix_size_counter = 0;

  HYPRE_Int *i_global_to_local;

  HYPRE_Int local_dof_counter;


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


  i_global_to_local = hypre_CTAlloc(HYPRE_Int, num_dofs);
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
	  hypre_printf("ERROR: non--positive definite local (subdomain) matrix; \n");
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
  
HYPRE_Int hypre_SchwarzSolve(double *x,
		       double *rhs,
		       double *aux,

		       HYPRE_Int *i_dof_dof,
		       HYPRE_Int *j_dof_dof,
		       double *a_dof_dof,

		       HYPRE_Int *i_domain_dof,
		       HYPRE_Int *j_domain_dof,
		       HYPRE_Int num_domains,

		       double *domain_matrixinverse,

		       HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int i,j,k, j_loc, k_loc;


  HYPRE_Int matrix_size, matrix_size_counter = 0;


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
