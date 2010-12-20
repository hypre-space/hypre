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

/*****************************************************************************
 *
 * PCG method with one V-cycle iteration; x = 0 -- initial iterate;
 *                 as preconditioner;
 *
 * ILU(1) -- smoother (LD-solve: pre--smoother; U-solve: post-smoother);
 *
 *     LD = (L + D^{-1}): lower triangular + diagonal^{-1} part;
 *      U: unit upper triangular part;
 ****************************************************************************/


HYPRE_Int hypre_VcycleILUpcg(double *x, double *rhs,
		       double **v_cycle, double **w_cycle,

		       double *reduction_factor_pointer,

		       hypre_CSRMatrix **Matrix,

		       HYPRE_Int **i_ILUdof_to_dof,
			  
		       HYPRE_Int **i_ILUdof_ILUdof, 
		       HYPRE_Int **j_ILUdof_ILUdof,
		       double **LD_data,

		       HYPRE_Int **i_ILUdof_ILUdof_t, 
		       HYPRE_Int **j_ILUdof_ILUdof_t,
		       double **U_data,

			     
		       hypre_CSRMatrix **P,

		       double *v_fine, double *w_fine, 
		       double *v, double *w, double *d, HYPRE_Int max_iter, 
		       
		       double *v_coarse, double *w_coarse, double *d_coarse, 

		       HYPRE_Int nu, 
		       HYPRE_Int level, HYPRE_Int coarse_level, 
		       HYPRE_Int *Ndofs)

{
  HYPRE_Int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  HYPRE_Int iter=0;
  float tau, alpha, beta;
  float delta_x;

  HYPRE_Int num_dofs = Ndofs[0];
  HYPRE_Int *i_dof_dof = hypre_CSRMatrixI(Matrix[0]);
  HYPRE_Int *j_dof_dof = hypre_CSRMatrixJ(Matrix[0]);
  double *sparse_matrix = hypre_CSRMatrixData(Matrix[0]);


  if (max_iter) max_iter = 1000;

  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= rhs[i] * rhs[i];

  if (delta0 < eps*eps)
    {
      ierr = hypre_ILUsolve(x,

			    i_ILUdof_to_dof[0],

			    i_ILUdof_ILUdof[0],
			    j_ILUdof_ILUdof[0],
			    LD_data[0],

			    i_ILUdof_ILUdof_t[0],
			    j_ILUdof_ILUdof_t[0],
			    U_data[0],

			    rhs, 

			    num_dofs); 

      return ierr;
    }



  for (i=0; i<num_dofs; i++)
    {
      v_cycle[0][i] = 0.e0; 
      w_cycle[0][i] = rhs[i];
    }

  ierr = hypre_VcycleILUsmoothing(v_cycle, w_cycle,

				  Matrix,
				  i_ILUdof_to_dof,

				  i_ILUdof_ILUdof, 
				  j_ILUdof_ILUdof,
				  LD_data,

				  i_ILUdof_ILUdof_t, 
				  j_ILUdof_ILUdof_t,
				  U_data,

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

  ierr = hypre_VcycleILUsmoothing(v_cycle, w_cycle,

				  Matrix,
				  i_ILUdof_to_dof,

				  i_ILUdof_ILUdof, 
				  j_ILUdof_ILUdof,
				  LD_data,

				  i_ILUdof_ILUdof_t, 
				  j_ILUdof_ILUdof_t,
				  U_data,

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
    hypre_printf("hypre_VcycleILUpcg: delta0: %e\n", delta0); 

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
	  hypre_printf("indefinite matrix: %e\n", tau);
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

  ierr = hypre_VcycleILUsmoothing(v_cycle, w_cycle,

				  Matrix,
				  i_ILUdof_to_dof,

				  i_ILUdof_ILUdof, 
				  j_ILUdof_ILUdof,
				  LD_data,

				  i_ILUdof_ILUdof_t, 
				  j_ILUdof_ILUdof_t,
				  U_data,

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
    hypre_printf("              hypre_VcycleILUpcg_iteration: %d;  residual_delta: %e,   arfac: %e\n", iter, sqrt(delta), sqrt(beta));	 

  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  if (max_iter > 999)
    {
      /*==================================================================*/
      hypre_printf("hypre_VcycleILUpcg: delta0: %e; delta: %e\n", delta0, delta);
      hypre_printf("hypre_VcycleILUpcg: iterations: %d; reduction factors: %e, %e\n", iter, asfac, arfac);
      /*==================================================================*/
    }
 
  *reduction_factor_pointer =  arfac;

  return ierr;
}
