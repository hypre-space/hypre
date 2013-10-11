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
 * PCG: with multiplicative Schwarz preconditioner;
 *
 * initial iterate x provided as input;
 *
 ****************************************************************************/


#include "headers.h"  

HYPRE_Int hypre_Schwarzpcg(HYPRE_Real *x, HYPRE_Real *rhs, 
		     HYPRE_Real *sparse_matrix, 

		     HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,

		     HYPRE_Int *i_domain_dof, HYPRE_Int *j_domain_dof,
		     HYPRE_Real *domain_matrixinverse,
		     HYPRE_Int num_domains,

		     HYPRE_Real *v, HYPRE_Real *w, HYPRE_Real *d, HYPRE_Real *aux,
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
      ierr = hypre_SchwarzSolve(x, rhs, aux,
				
				i_dof_dof, j_dof_dof, sparse_matrix, 

				i_domain_dof, j_domain_dof, num_domains,

				domain_matrixinverse,
				

				num_dofs); 

      return ierr;
    }


  /*  initial iterate is input: 

  ierr = hypre_SchwarzSolve(x, rhs, aux,

			    i_dof_dof, j_dof_dof, sparse_matrix, 

			    i_domain_dof, j_domain_dof, num_domains,

			    domain_matrixinverse,

			    num_dofs);

			    ------------------------------------------ */


 
  /* sparse-matrix vector product: ----------------------------------- */

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

  ierr = hypre_SchwarzSolve(d, w, v, 

			    i_dof_dof, j_dof_dof, sparse_matrix, 

			    i_domain_dof, j_domain_dof, num_domains,

			    domain_matrixinverse,

			    num_dofs);


  delta0 = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta0+= w[i] * d[i];

  if (max_iter > 999)
    hypre_printf("hypre_Schwarzpcg_delta0: %e\n", delta0); 

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


  ierr = hypre_SchwarzSolve(v, w, aux,

			    i_dof_dof, j_dof_dof, sparse_matrix, 

			    i_domain_dof, j_domain_dof, num_domains,

			    domain_matrixinverse,

			    num_dofs);


  delta = 0.e0;
  for (i=0; i<num_dofs; i++)
    delta += v[i] * w[i];

  beta = delta /delta_old;
  iter++;

  if (max_iter > 999)
    hypre_printf("              hypre_Schwarzpcg_iteration: %d;  residual_delta: %e,   arfac: %e\n", iter, sqrt(delta), sqrt(beta));	 

  delta_old = delta;

  for (i=0; i<num_dofs; i++)
    d[i] = v[i] + beta * d[i];

  if (delta > eps * delta0 && iter < max_iter) goto loop;

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  if (max_iter > 999)
    {
      /*==================================================================*/
      hypre_printf("hypre_Schwarzpcg: delta0: %e; delta: %e\n", delta0, delta);
      hypre_printf("hypre_Schwarzpcg: iterations: %d; reduction factors: %e, %e\n", iter, asfac, arfac);
      /*==================================================================*/
    }
 

  return ierr;

}

HYPRE_Int sparse_matrix_vector_product(HYPRE_Real *v,
				 HYPRE_Real *sparse_matrix, 
				 HYPRE_Real *w, 
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

