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
 *
 * PCG method with one V-cycle iteration; x = 0 -- initial iterate;
 *                 as preconditioner;
 *
 * GS -- smoother (LD-solve: pre--smoother; U-solve: post-smoother);
 *
 ****************************************************************************/


int hypre_VcycleGSpcg(double *x, double *rhs,
		      double **v_cycle, double **w_cycle,

		      double *reduction_factor_pointer,

		      hypre_CSRMatrix **Matrix,

			     
		      hypre_CSRMatrix **P,

		      double *v_fine, double *w_fine, 
		      double *v, double *w, double *d, int max_iter, 
		       
		      double *v_coarse, double *w_coarse, double *d_coarse, 

		      int nu, 
		      int level, int coarse_level, 
		      int *Ndofs)

{
  int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  int iter=0;
  float tau, alpha, beta;
  float delta_x;

  int num_dofs = Ndofs[0];
  int *i_dof_dof = hypre_CSRMatrixI(Matrix[0]);
  int *j_dof_dof = hypre_CSRMatrixJ(Matrix[0]);
  double *sparse_matrix = hypre_CSRMatrixData(Matrix[0]);


  if (max_iter) max_iter = 1000;

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= rhs[i] * rhs[i];

  if (delta0 < eps*eps)
    {
      ierr = hypre_sym_GS_solve(x,
				i_dof_dof, j_dof_dof,
				sparse_matrix,


				rhs, 

				num_dofs); 

      return ierr;
    }



  for (i=0; i<num_dofs; i++)
    {
      v_cycle[0][i] = 0.e0; 
      w_cycle[0][i] = rhs[i];
    }

  ierr = hypre_VcycleGSsmoothing(v_cycle, w_cycle,

				 Matrix,

				 P,
				 v_fine, w_fine, 
				 v_coarse, w_coarse, d_coarse, 

				 nu, 
				 level, coarse_level, 
				 Ndofs);



  for (i=0; i < num_dofs; i++)
    x[i] = v_cycle[0][i];



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



  for (i=0; i<num_dofs; i++)
    {
      v_cycle[0][i] = 0.e0; 
      w_cycle[0][i] = w[i];
    }

  ierr = hypre_VcycleGSsmoothing(v_cycle, w_cycle,

				  Matrix,


				  P,
				  v_fine, w_fine, 
				  v_coarse, w_coarse, d_coarse, 

				  nu, 
				  level, coarse_level, 
				  Ndofs);



  for (i=0; i < num_dofs; i++)
    d[i] = v_cycle[0][i];

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= w[i] * d[i];

  if (max_iter > 999)
    printf("hypre_VcycleGSpcg: delta0: %e\n", delta0); 

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



  for (i=0; i<num_dofs; i++)
    {
      v_cycle[0][i] = 0.e0; 
      w_cycle[0][i] = w[i];
    }

  ierr = hypre_VcycleGSsmoothing(v_cycle, w_cycle,

				  Matrix,

				  P,
				  v_fine, w_fine, 
				  v_coarse, w_coarse, d_coarse, 

				  nu, 
				  level, coarse_level, 
				  Ndofs);


  for (i=0; i < num_dofs; i++)
    v[i] = v_cycle[0][i];


  delta = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta += v[i] * w[i];

  beta = delta /delta_old;
  iter++;

  if (max_iter > 999)
    printf("              hypre_VcycleGSpcg_iteration: %d;  residual_delta: %e,   arfac: %e\n", iter, sqrt(delta), sqrt(beta));	 

  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  if (max_iter > 999)
    {
      /*==================================================================*/
      printf("hypre_VcycleGSpcg: delta0: %e; delta: %e\n", delta0, delta);
      printf("hypre_VcycleGSpcg: iterations: %d; reduction factors: %e, %e\n", iter, asfac, arfac);
      /*==================================================================*/
    }
 
  *reduction_factor_pointer =  arfac;

  return ierr;
}
