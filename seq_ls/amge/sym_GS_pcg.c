#include "headers.h" 

int 
hypre_sym_GS_pcg(double *x, double *rhs,
		 double *sparse_matrix, 

		 int *i_dof_dof, int *j_dof_dof,
		
		 double *v, double *w, double *d, 
		
		 int max_iter, 
		
		 int num_dofs)

{

  int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  int iter=0;
  float tau, alpha, beta;
  float delta_x;

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= rhs[i] * rhs[i];

  if (delta0 < eps*eps)
    {
      for (i=0; i<num_dofs; i++)
	x[i] = 0.e0;

      ierr = GS_forw(x, rhs,
		     sparse_matrix, i_dof_dof, j_dof_dof,
		     num_dofs);

      ierr = GS_back(x, rhs,
		     sparse_matrix, i_dof_dof, j_dof_dof,
		     num_dofs); 
      return ierr;
    }


  /* sparse-matrix vector product: --------------------------*/

  ierr = sparse_matrix_vector_product(v,
				      sparse_matrix, 
				      x, 
				      i_dof_dof, j_dof_dof,
				      num_dofs);

  /* compute residual: w <-- rhs - A *x = rhs - v;          */
  for (i=0; i < num_dofs; i++)
    w[i] = rhs[i] - v[i];
  for (i=0; i<num_dofs; i++)
    d[i] = 0.e0;

  ierr = GS_forw(d, w,
		 sparse_matrix, i_dof_dof, j_dof_dof,
		 num_dofs);

  ierr = GS_back(d, w,
		 sparse_matrix, i_dof_dof, j_dof_dof,
		 num_dofs); 


  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= w[i] * d[i];

  if (max_iter > 999) 
    printf("cg_coarse_delta0: %e\n", delta0); 

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
	  printf("indefinite matrix in cg_iter_coarse: %e\n", tau);
	  /*	  return -1;                               */
	}

  alpha = delta_old/tau;
  for (i=0; i<num_dofs; i++)
    x[i] += alpha * d[i];

  iter++;
  if (max_iter > 999) 
    printf("sym_GS_pcg_iteration: %d;  residual_delta: %e\n", iter, delta_old);

  for (i=0; i<num_dofs; i++)
    w[i] -= alpha * v[i];


  for (i=0; i<num_dofs; i++)
    v[i] = 0.e0;

  ierr = GS_forw(v, w,
		 sparse_matrix, i_dof_dof, j_dof_dof,
		 num_dofs);

  ierr = GS_back(v, w,
		 sparse_matrix, i_dof_dof, j_dof_dof,
		 num_dofs); 


  delta = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta += v[i] * w[i];

  beta = delta /delta_old;
  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  if (max_iter > 999) 
    {
      asfac = sqrt(beta);
      arfac = exp(log(delta/delta0)/(2*iter));

      printf("sym_GS_pcg_arfac: %e; sym_GS_pcg_asfac : %e\n", 
	     arfac, asfac); 
    }

  return ierr;

}
int GS_forw(double *x, double *rhs,
	    double *sparse_matrix, int *i_dof_dof, int *j_dof_dof,
	    int num_dofs)

{
  int ierr = 0, i,j;
  double aux;
  double diag;

  for (i=0; i < num_dofs; i++)
    {
      aux = rhs[i];
      for (j=i_dof_dof[i]; j < i_dof_dof[i+1]; j++)
	{
	  if (j_dof_dof[j] != i ) 
	    aux -= sparse_matrix[j] * x[j_dof_dof[j]];
	  else diag = sparse_matrix[j];
	}
      x[i] = aux / diag; 
    }

  return ierr;

}     

int GS_back(double *x, double *rhs,
	    double *sparse_matrix, int *i_dof_dof, int *j_dof_dof,
	    int num_dofs)

{
  int ierr = 0,i,j;
  double aux, diag;

  for (i=num_dofs-1; i >= 0; i--)
    {
      aux = rhs[i];
      for (j=i_dof_dof[i+1]-1; j >= i_dof_dof[i]; j--)
	{
	  if (j_dof_dof[j] !=i ) 
	    aux -= sparse_matrix[j] * x[j_dof_dof[j]];
	  else diag = sparse_matrix[j];
	}
      x[i] = aux / diag;
    }

  return ierr;

} 
