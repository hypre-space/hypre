
/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


/*****************************************************************************
 *
 * PCG: with ILU(1) preconditioner;
 *
 * initial iterate x provided as input;
 *
 ****************************************************************************/


#include "headers.h"  

int hypre_ILUpcg(double *x, double *rhs,
		 double *sparse_matrix, 

		 int *i_dof_dof, int *j_dof_dof,

		 int *i_ILUdof_to_dof,
		 int *i_ILUdof_ILUdof, int *j_ILUdof_ILUdof,
		 double *LD_data,
		 int *i_ILUdof_ILUdof_t, int *j_ILUdof_ILUdof_t,
		 double *U_data,

		 double *v, double *w, double *d,
		 int max_iter, 

		 int num_dofs)

{

  int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  int iter=0;
  float tau, alpha, beta;
  float delta_x;

  if (max_iter) max_iter = 1000;

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= rhs[i] * rhs[i];

  if (delta0 < eps*eps)
    {
      ierr = hypre_ILUsolve(x,

			    i_ILUdof_to_dof,

			    i_ILUdof_ILUdof,
			    j_ILUdof_ILUdof,
			    LD_data,

			    i_ILUdof_ILUdof_t,
			    j_ILUdof_ILUdof_t,
			    U_data,

			    rhs, 

			    num_dofs); 

      return ierr;
    }


  /*  initial iterate is input: ---------------------------

  ierr = hypre_ILUsolve(x,

			i_ILUdof_to_dof,

			i_ILUdof_ILUdof,
			j_ILUdof_ILUdof,
			LD_data,

			i_ILUdof_ILUdof_t,
			j_ILUdof_ILUdof_t,
			U_data,

			rhs, 

			num_dofs);



			*/
 
  /* sparse-matrix vector product: --------------------------*/

  ierr = sparse_matrix_vector_product(v,
				      sparse_matrix, 
				      x, 
				      i_dof_dof, j_dof_dof,
				      num_dofs);

  /* compute residual: w <-- rhs - A *x = rhs - v;          */
  for (i=0; i < num_dofs; i++)
    w[i] = rhs[i] - v[i];


  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= w[i] * w[i];

  if (delta0 < eps*eps)
    return ierr;


  ierr = hypre_ILUsolve(d,

			i_ILUdof_to_dof,
			
			i_ILUdof_ILUdof,
			j_ILUdof_ILUdof,
			LD_data,

			i_ILUdof_ILUdof_t,
			j_ILUdof_ILUdof_t,
			U_data,

			w, 

			num_dofs);

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= w[i] * d[i];

  if (max_iter > 999)
    printf("Hypre_ILUpcg_delta0: %e\n", delta0); 

  delta_old = delta0;

  /* for (i=0; i < num_dofs; i++)
     d[i] = w[i]; */

loop:

  /* sparse-matrix vector product: --------------------------*/

  ierr = sparse_matrix_vector_product(v,
				      sparse_matrix, 
				      d, 
				      i_dof_dof, j_dof_dof,
				      num_dofs);

  tau = 0.e0;
  for (i=0; i<num_dofs; i++)
    tau += d[i] * v[i];

      if (tau <= 0.e0) 
	{
	  printf("indefinite matrix: %e\n", tau);
	  /*	  return -1;                               */
	}

  alpha = delta_old/tau;
  for (i=0; i<num_dofs; i++)
    x[i] += alpha * d[i];

  for (i=0; i<num_dofs; i++)
    w[i] -= alpha * v[i];


  ierr = hypre_ILUsolve(v,

			i_ILUdof_to_dof,

			i_ILUdof_ILUdof,
			j_ILUdof_ILUdof,
			LD_data,

			i_ILUdof_ILUdof_t,
			j_ILUdof_ILUdof_t,
			U_data,

			w, 

			num_dofs);

  delta = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta += v[i] * w[i];

  beta = delta /delta_old;
  iter++;

  if (max_iter > 999)
    printf("              hypre_ILUpcg_iteration: %d;  residual_delta: %e,   arfac: %e\n", iter, sqrt(delta), sqrt(beta));	 

  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  if (max_iter > 999)
    {
      /*==================================================================*/
      printf("hypre_ILUpcg: delta0: %e; delta: %e\n", delta0, delta);
      printf("hypre_ILUpcg: iterations: %d; reduction factors: %e, %e\n", iter, asfac, arfac);
      /*==================================================================*/
    }
 

  return ierr;

}
int sparse_matrix_vector_product(double *v,
				 double *sparse_matrix, 
				 double *w, 
				 int *i_dof_dof, int *j_dof_dof,
				 int num_dofs)
{
  int ierr =0, i,k;

  for (i=0; i < num_dofs; i++)
    v[i] = 0.e0;

  for (i=0; i < num_dofs; i++)
    for (k=i_dof_dof[i]; k < i_dof_dof[i+1]; k++)
      v[j_dof_dof[k]] += sparse_matrix[k] * w[i];


  return ierr;

}	  
