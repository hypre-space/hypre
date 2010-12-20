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
 * Performes one V-cycle iteration; x = 0 -- initial iterate;
 * ILU(1) -- smoother (LD-solve: pre--smoother; U-solve: post-smoother);
 *
 *     LD = (L + D^{-1}): lower triangular + diagonal^{-1} part;
 *      U: unit upper triangular part;
 ****************************************************************************/


HYPRE_Int hypre_VcycleILUsmoothing(double **x, double **rhs,



			     hypre_CSRMatrix **Matrix,

			     HYPRE_Int **i_ILUdof_to_dof,
			  
			     HYPRE_Int **i_ILUdof_ILUdof, 
			     HYPRE_Int **j_ILUdof_ILUdof,
			     double **LD_data,

			     HYPRE_Int **i_ILUdof_ILUdof_t, 
			     HYPRE_Int **j_ILUdof_ILUdof_t,
			     double **U_data,

			     
			     hypre_CSRMatrix **P,


			     double *v, double *w, 
			     double *v_coarse,
			     double *w_coarse,
			     double *d_coarse, 

			     HYPRE_Int nu, 
			     HYPRE_Int level, HYPRE_Int coarse_level, 
			     HYPRE_Int *num_dofs)

{

  HYPRE_Int ierr=0, i, j, k, l, l_coarse;
  HYPRE_Int nu_new = nu;
  HYPRE_Int num_V_cycles;

  HYPRE_Int max_CG_iter=999; 
  HYPRE_Int num_coarsedofs;

  double eps = 1.e-12;
  double res_norm=0.e0;
  double rhs_norm;

  double *aux_v, *aux_w, *aux_d;

  double **sparse_matrix;
  HYPRE_Int **i_dof_dof;
  HYPRE_Int **j_dof_dof;

  double **Prolong_coeff;
  HYPRE_Int **i_dof_neighbor_coarsedof;
  HYPRE_Int **j_dof_neighbor_coarsedof;



  sparse_matrix = hypre_CTAlloc(double* , coarse_level+1);
  i_dof_dof =  hypre_CTAlloc(HYPRE_Int* , coarse_level+1);
  j_dof_dof =  hypre_CTAlloc(HYPRE_Int* , coarse_level+1);


  Prolong_coeff = hypre_CTAlloc(double* , coarse_level);
  i_dof_neighbor_coarsedof =  hypre_CTAlloc(HYPRE_Int* , coarse_level);
  j_dof_neighbor_coarsedof =  hypre_CTAlloc(HYPRE_Int* , coarse_level);

  for (l=0; l < coarse_level+1 && num_dofs[l] > 0; l++)
    {
      i_dof_dof[l] = hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof[l]= hypre_CSRMatrixJ(Matrix[l]);
      sparse_matrix[l]   = hypre_CSRMatrixData(Matrix[l]);
    }

  for (l=0; l < coarse_level && num_dofs[l] > 0; l++)
    {
      i_dof_neighbor_coarsedof[l] = hypre_CSRMatrixI(P[l]);
      j_dof_neighbor_coarsedof[l]= hypre_CSRMatrixJ(P[l]);
      Prolong_coeff[l]   = hypre_CSRMatrixData(P[l]);
    }


  /*
  aux_v = (double *) malloc(num_dofs[0] * sizeof(double));
  aux_w = (double *) malloc(num_dofs[0] * sizeof(double));
  aux_d = (double *) malloc(num_dofs[0] * sizeof(double));


  for (i=0; i <num_dofs[0]; i++)
    res_norm+=rhs[0][i]*rhs[0][i];
  
  res_norm = sqrt(res_norm);
  eps *= res_norm;

  /* hypre_printf("\n\n                      initial RHS-norm: %e\n", res_norm); */

  num_V_cycles = 0;

  l = 0;

  /*   nu_new = nu;  */

  /* initial iterate x[0] is input/output: ------------------- */

loop_20:

  /* smoothing step:  ----------------------------------------*/


  ierr = hypre_ILUsolve(x[l],
			i_ILUdof_to_dof[l],
					 
			i_ILUdof_ILUdof[l],
			j_ILUdof_ILUdof[l],
			LD_data[l],

			i_ILUdof_ILUdof_t[l], 
			j_ILUdof_ILUdof_t[l],
			U_data[l],


			rhs[l], 

			num_dofs[l]);

  /*-------------------------------------------------------------


  ierr = hypre_ILUpcg(x[l], rhs[l],
		      sparse_matrix[l],

		      i_dof_dof[l], j_dof_dof[l],

		      i_ILUdof_to_dof[l],
				   
		      i_ILUdof_ILUdof[l], 
		      j_ILUdof_ILUdof[l],
		      LD_data[l],

		      i_ILUdof_ILUdof_t[l], 
		      j_ILUdof_ILUdof_t[l],
		      U_data[l],


		      aux_v, aux_w, aux_d, nu_new,

		      num_dofs[l]);

		      ---------------------------------------------------*

  /* sparse-matrix vector product: --------------------------*/

  ierr = sparse_matrix_vector_product(v,
				      sparse_matrix[l], 
				      x[l], 
				      i_dof_dof[l], j_dof_dof[l],
				      num_dofs[l]);


  /* compute residual: w <-- rhs - v; */
  for (i=0; i < num_dofs[l]; i++)
    w[i] = rhs[l][i] - v[i];

  /* restrict defect: --------------------------------------*/


  ierr = fine_to_coarse_restriction(rhs[l+1], 
				    Prolong_coeff[l],
				    i_dof_neighbor_coarsedof[l],
				    j_dof_neighbor_coarsedof[l],

				    w,
				    num_dofs[l], num_dofs[l+1]);


  /*
  rhs_norm = 0.e0;
  for (i=0; i < num_coarsedofs; i++)
    rhs_norm += rhs[l+1][i]*rhs[l+1][i];

  hypre_printf("rhs_norm[%d]: %e\n", l+1, sqrt(rhs_norm));
  */


  if (l == coarse_level-1)
    {

      /* solve on coarse grid: +++++++++++++++++++++++++++++++++++++*/

      num_coarsedofs = num_dofs[l+1];
	  
      for (i=0; i < num_coarsedofs; i++)
	/* x[l+1][i] = rand();  */
	x[l+1][i] = 1.e0;


      /* solve on coarse grid: +++++++++++++++++++++++++++++++++++++*/

      ierr = cg_iter_coarse(x[l+1], rhs[l+1],
			    sparse_matrix[l+1], 

			    i_dof_dof[l+1], j_dof_dof[l+1],
			    v_coarse, w_coarse, d_coarse, max_CG_iter, 
			    num_coarsedofs);



      l++;
      goto loop_10;
    }
  else
    {
      l++;
      goto loop_20;
    }


loop_10:

  l--;

  /* interpolate coarse--grid solution: -------------------*/
      
  ierr = coarse_to_fine_interpolation(v,
				      Prolong_coeff[l],
				      i_dof_neighbor_coarsedof[l],
				      j_dof_neighbor_coarsedof[l],

				      x[l+1],
				      num_dofs[l], num_dofs[l+1]);

  /* update iterate: ----------------------------------------*/

  for (i=0; i < num_dofs[l]; i++)
    x[l][i] += v[i];
 

  /* post--smoothing: ---------------------------------------*/   


  ierr = sparse_matrix_vector_product(w,
				      sparse_matrix[l], 
				      x[l], 
				      i_dof_dof[l], j_dof_dof[l],
				      num_dofs[l]);




  for (i=0; i < num_dofs[l]; i++)
    w[i] = rhs[l][i] - w[i];


  ierr = hypre_ILUsolve(v,
			i_ILUdof_to_dof[l],
					 

			i_ILUdof_ILUdof[l], 
			j_ILUdof_ILUdof[l],
			LD_data[l],


			i_ILUdof_ILUdof_t[l], 
			j_ILUdof_ILUdof_t[l],
			U_data[l],

			w, 

			num_dofs[l]);

  for (i=0; i < num_dofs[l]; i++)
    x[l][i] += v[i];
 



  /* =======================================================  
  
  ierr = hypre_ILUpcg(x[l], rhs[l],
		      sparse_matrix[l],

		      i_dof_dof[l], j_dof_dof[l],

		      i_ILUdof_to_dof[l],
				   
		      i_ILUdof_ILUdof[l], 
		      j_ILUdof_ILUdof[l],
		      LD_data[l],

		      i_ILUdof_ILUdof_t[l], 
		      j_ILUdof_ILUdof_t[l],
		      U_data[l],

		      aux_v, aux_w, aux_d, nu_new,

		      num_dofs[l]);

   ================================================================== */
  if ( l > 0) goto loop_10;


  hypre_TFree(sparse_matrix);
  hypre_TFree(i_dof_dof);
  hypre_TFree(j_dof_dof);


  hypre_TFree(Prolong_coeff);
  hypre_TFree(i_dof_neighbor_coarsedof);
  hypre_TFree(j_dof_neighbor_coarsedof);


  return ierr;

}

HYPRE_Int coarse_to_fine_interpolation(double *v,
				 double *Prolong_coeff,
				 HYPRE_Int *i_dof_neighbor_coarsedof,
				 HYPRE_Int *j_dof_neighbor_coarsedof,

				 double *w_coarse,
				 HYPRE_Int num_dofs, HYPRE_Int num_coarsedofs)
{
  HYPRE_Int ierr=0, i,j;

  for (i=0; i < num_dofs; i++)
    v[i] = 0.e0;

  for (i=0; i < num_dofs; i++)
    for (j=i_dof_neighbor_coarsedof[i];
	 j<i_dof_neighbor_coarsedof[i+1]; j++)
      v[i] += Prolong_coeff[j] * w_coarse[j_dof_neighbor_coarsedof[j]];


  return ierr;

}

HYPRE_Int fine_to_coarse_restriction(double *v_coarse,
			       double *Prolong_coeff,
			       HYPRE_Int *i_dof_neighbor_coarsedof,
			       HYPRE_Int *j_dof_neighbor_coarsedof,

			       double *w,
			       HYPRE_Int num_dofs, HYPRE_Int num_coarsedofs)
{
  HYPRE_Int ierr=0, i,j;

  for (i=0; i < num_coarsedofs; i++)
    v_coarse[i] = 0.e0;

  for (i=0; i < num_dofs; i++)
    for (j=i_dof_neighbor_coarsedof[i];
	 j<i_dof_neighbor_coarsedof[i+1]; j++)
      v_coarse[j_dof_neighbor_coarsedof[j]] += Prolong_coeff[j] * w[i];        

  return ierr;


}

HYPRE_Int cg_iter_coarse(double *x, double *rhs,
		   double *sparse_matrix, 

		   HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,
		   double *v, double *w, double *d, HYPRE_Int max_iter, 
		   HYPRE_Int num_dofs)

{

  HYPRE_Int ierr=0, i, j;
  float delta0, delta_old, delta, asfac, arfac, eps = 1.e-12;
  
  HYPRE_Int iter=0;
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
    hypre_printf("cg_coarse_delta0: %e\n", delta0); 

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
	  hypre_printf("indefinite matrix in cg_iter_coarse: %e\n", tau);
	  /*	  return -1;                               */
	}

  alpha = delta_old/tau;
  for (i=0; i<num_dofs; i++)
    x[i] += alpha * d[i];

  iter++;
  if (max_iter > 999) 
    hypre_printf("cg_coarse_iteration: %d;  residual_delta: %e\n", iter, delta_old);

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

  asfac = sqrt(beta);
  arfac = exp(log(delta/delta0)/(2*iter));

  return ierr;

}
HYPRE_Int GS_forw(double *x, double *rhs,
	    double *sparse_matrix, HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,
	    HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0, i,j;
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

HYPRE_Int GS_back(double *x, double *rhs,
	    double *sparse_matrix, HYPRE_Int *i_dof_dof, HYPRE_Int *j_dof_dof,
	    HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0,i,j;
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
