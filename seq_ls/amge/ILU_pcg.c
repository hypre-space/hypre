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







/*****************************************************************************
 *
 * PCG: with ILU(1) preconditioner;
 *
 * initial iterate x provided as input;
 *
 ****************************************************************************/


#include "headers.h"  

HYPRE_Int hypre_ILUpcg(double *x, double *rhs,
		 double *sparse_matrix, 

		 HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,

		 HYPRE_Int *i_ILUdof_to_dof,
		 HYPRE_Int *i_ILUdof_ILUdof, HYPRE_Int *j_ILUdof_ILUdof,
		 double *LD_data,
		 HYPRE_Int *i_ILUdof_ILUdof_t, HYPRE_Int *j_ILUdof_ILUdof_t,
		 double *U_data,

		 double *v, double *w, double *d,
		 HYPRE_Int max_iter, 

		 HYPRE_Int num_dofs)

{

  HYPRE_Int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  HYPRE_Int iter=0;
  float tau, alpha, beta;
  float delta_x;

  if (max_iter< 0) max_iter = 1000;

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
    hypre_printf("hypre_ILUpcg_delta0: %e\n", delta0); 

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
    hypre_printf("              hypre_ILUpcg_iteration: %d;  residual_delta: %e,   arfac: %e\n", iter, sqrt(delta), sqrt(beta));	 

  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  if (max_iter > 999)
    {
      /*==================================================================*/
      hypre_printf("hypre_ILUpcg: delta0: %e; delta: %e\n", delta0, delta);
      hypre_printf("hypre_ILUpcg: iterations: %d; reduction factors: %e, %e\n", iter, asfac, arfac);
      /*==================================================================*/
    }
 

  return ierr;

}
HYPRE_Int sparse_matrix_vector_product(double *v,
				 double *sparse_matrix, 
				 double *w, 
				 HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,
				 HYPRE_Int num_dofs)
{
  HYPRE_Int ierr =0, i,k;

  for (i=0; i < num_dofs; i++)
    v[i] = 0.e0;

  for (i=0; i < num_dofs; i++)
    for (k=i_dof_dof[i]; k < i_dof_dof[i+1]; k++)
      v[j_dof_dof[k]] += sparse_matrix[k] * w[i];


  return ierr;

}	  
