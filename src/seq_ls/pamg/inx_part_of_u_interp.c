/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/

#include "headers.h"
#include "amg.h"

HYPRE_Int
hypre_CreateDomain
(HYPRE_Int *CF_marker, hypre_CSRMatrix *A, HYPRE_Int num_coarse,
HYPRE_Int *dof_func, HYPRE_Int **coarse_dof_ptr,
HYPRE_Int **domain_i_ptr, HYPRE_Int **domain_j_ptr)

{
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);    
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);    
   HYPRE_Int num_vars = hypre_CSRMatrixNumRows(A);
   HYPRE_Int i, j, cnt, domain, j_indx;
   HYPRE_Int *domain_i;
   HYPRE_Int *domain_j;
   HYPRE_Int *coarse_dof;
   HYPRE_Int num_pts = 0;

   domain_i = hypre_CTAlloc(HYPRE_Int, num_coarse+1);
   coarse_dof = hypre_CTAlloc(HYPRE_Int, num_coarse);

   cnt = 0;
   for (i=0; i < num_vars; i++)
   {
      if (CF_marker[i] > 0)
      {
         num_pts += A_i[i+1]-A_i[i];
         coarse_dof[cnt++] = dof_func[i];
      }
   }
   domain_j = hypre_CTAlloc(HYPRE_Int, num_pts);

   cnt = 0;
   domain = 0;
   domain_i[0] = 0;
   for (i=0; i < num_vars; i++)
   {
       if (CF_marker[i] > 0)
       {
          domain_j[cnt++] = i;
          for (j=A_i[i]; j < A_i[i+1]; j++)
	  {
	    j_indx = A_j[j];
	    if (CF_marker[j_indx]<1)
	    {
	       domain_j[cnt++] = j_indx;
            }
          }
          domain++;
          domain_i[domain] = cnt;
       }         
   } 

   *domain_i_ptr = domain_i;
   *domain_j_ptr = domain_j;
   *coarse_dof_ptr = coarse_dof;

   return 0;
}

HYPRE_Int
hypre_InexactPartitionOfUnityInterpolation
(hypre_CSRMatrix **P_pointer,

 HYPRE_Int    *i_dof_dof,
 HYPRE_Int    *j_dof_dof,
 double *a_dof_dof,


 double *unit_vector,


 HYPRE_Int *i_domain_dof,
 HYPRE_Int *j_domain_dof,

 HYPRE_Int num_domains, /* == num-coarsedofs */

 HYPRE_Int num_dofs)

{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k;

  HYPRE_Int ind = 1;
  HYPRE_Int nu, nu_max = 1;

  double  eps = 1.e-24;
  HYPRE_Int max_iter = 1000;
  HYPRE_Int iter;
  double delta0, delta_old, delta, alpha, tau, beta;
  double aux, diag;

  double *P_t_coeff;
  hypre_CSRMatrix *P_t, *P;

  double *x,*r,*d,*g,*h;
  double *row_sum;


  HYPRE_Int *i_global_to_local;
  HYPRE_Int local_dof_counter;


  double *diag_dof_dof;
  /* ------------------------------------------------------------------

     domain_dof relation should satisfy the following property:

     num_domains == num_coarsedofs;

     each domain contains only one coarse dof;

     ------------------------------------------------------------------ */

  
  i_global_to_local = hypre_CTAlloc(HYPRE_Int, num_dofs);

  for (i=0; i < num_dofs; i++)
    i_global_to_local[i] = -1;

  local_dof_counter = 0;
  for (i=0; i < num_domains; i++)
    if (local_dof_counter < i_domain_dof[i+1]-i_domain_dof[i])
      local_dof_counter = i_domain_dof[i+1]-i_domain_dof[i];
  /* solve T x = unit_vector; --------------------------------------- */

  /* cg loop: ------------------------------------------------------- */
  hypre_printf("\n---------------------- num_domains: %d, nnz: %d;\n", 
	 num_domains, i_domain_dof[num_domains]);

  x = hypre_CTAlloc(double, num_dofs);
  d = hypre_CTAlloc(double, num_dofs);
  g = hypre_CTAlloc(double, num_dofs);
  r = hypre_CTAlloc(double, num_dofs);

  h = hypre_CTAlloc(double, local_dof_counter);
  diag_dof_dof = hypre_CTAlloc(double, i_dof_dof[num_dofs]);
  for (i=0; i<num_dofs; i++)
    for (j=i_dof_dof[i]; j<i_dof_dof[i+1]; j++)
      if (i!=j_dof_dof[j])
	diag_dof_dof[j] = 0.e0;
      else
	diag_dof_dof[j] = a_dof_dof[j];	

  delta0 = 0.e0;
  for (i=0; i < num_dofs; i++)
    {
      x[i] = 0.e0;
      r[i] = unit_vector[i];
      delta0+=r[i]*r[i];
    }
  /* compute initial iterate:  

  ierr =
    compute_sum_A_i_action(x,
			   r, 
		       
			   i_domain_dof,
			   j_domain_dof,


			   i_dof_dof,
			   j_dof_dof,
			   a_dof_dof,

			   i_global_to_local,

			   num_domains,
			   num_dofs);
			   ------------------------------------- */	  


  /* matrix vector product: g < -- T x; ------------------------------ */

  ierr= 
    compute_sym_GS_T_action(g,
			    x,
			    h,

			    i_domain_dof,
			    j_domain_dof,
			    nu_max,
		     
			    i_dof_dof,
			    j_dof_dof,
			    a_dof_dof,

			    i_global_to_local,

			    num_domains,
			    num_dofs);

  delta = 0;
  for (i=0; i < num_dofs; i++)
    {
      r[i] -= g[i];
      delta+=r[i]*r[i];
    }

  if (delta < eps * delta0)
    goto end_cg;

  ierr= 
    compute_sym_GS_T_action(g,
			    unit_vector,
			    h,

			    i_domain_dof,
			    j_domain_dof,
			    1,
		     
			    i_dof_dof,
			    j_dof_dof,
			    diag_dof_dof,

			    i_global_to_local,

			    num_domains,
			    num_dofs);

  /* 
  ierr =
    compute_sum_A_i_action(d,
			   r, 
		       
			   i_domain_dof,
			   j_domain_dof,


			   i_dof_dof,
			   j_dof_dof,
			   a_dof_dof,

			   i_global_to_local,

			   num_domains,
			   num_dofs);
			   */

  for (i=0; i < num_dofs; i++)
    d[i]=r[i]/g[i];

  /* d contains precondtitioned residual: ------------------------ */
  delta = 0.e0;
  for (i=0; i < num_dofs; i++)
    delta+=d[i]*r[i];

  delta0 = delta;

  eps = 1.e-12;
  iter = 0;
loop:
  /* matrix vector product: -------------------------------------- */

  ierr= 
    compute_sym_GS_T_action(g,
			    d,
			    h,

			    i_domain_dof,
			    j_domain_dof,
			    nu_max,

			    i_dof_dof,
			    j_dof_dof,
			    a_dof_dof,

			    i_global_to_local,

			    num_domains,
			    num_dofs);

  tau = 0.e0;
  for (i=0; i < num_dofs; i++)
    tau += d[i]*g[i];

  alpha = delta/tau;

  for (i=0; i < num_dofs; i++)
    {
      x[i] += alpha * d[i];
      r[i] -= alpha * g[i];
    }

  iter++;
  delta_old = delta;
  /*
  ierr =
    compute_sum_A_i_action(g,
			   r, 
		       
			   i_domain_dof,
			   j_domain_dof,


			   i_dof_dof,
			   j_dof_dof,
			   a_dof_dof,

			   i_global_to_local,

			   num_domains,
			   num_dofs);

			   */
  ierr= 
    compute_sym_GS_T_action(g,
			    unit_vector,
			    h,

			    i_domain_dof,
			    j_domain_dof,
			    1,
		     
			    i_dof_dof,
			    j_dof_dof,
			    diag_dof_dof,

			    i_global_to_local,

			    num_domains,
			    num_dofs);

  for (i=0; i < num_dofs; i++)
    g[i] = r[i]/g[i];

  delta = 0.e0;
  for (i=0; i < num_dofs; i++)
    delta  += g[i] * r[i];

  hypre_printf("\n---------------------- iter: %d, delta: %le\n",
	 iter, delta);
  if (delta < eps * delta0 || iter > max_iter)
    goto end_cg;
  
  beta = delta/delta_old;

  for (i=0; i < num_dofs; i++)
    d[i] = g[i] + beta * d[i];

  goto loop;


 
end_cg:
  hypre_printf("\n END CG in partition of unity interpolation; num_iters: %d\n",
	 iter);

  hypre_TFree(r);
  hypre_TFree(g);
  hypre_TFree(d);

  /* ith column of P is T_i x; ----------------------------------- */

  P_t_coeff = hypre_CTAlloc(double, i_domain_dof[num_domains]);

  for (i=0; i < num_domains; i++)
    {
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  i_global_to_local[j_domain_dof[j]] = j-i_domain_dof[i];
	  h[j-i_domain_dof[i]] = 0.e0;
	}
      nu = 0;
    loop_nu:
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{

	  aux = x[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
             if (i_global_to_local[j_dof_dof[k]] > -1)
             {
	      /* this is a_{i_loc, j_loc} --------------------------------- */
               if (j_dof_dof[k] != j_domain_dof[j])
               {
                  aux -= a_dof_dof[k] * h[i_global_to_local[j_dof_dof[k]]];
               }
               else
               {
                  diag = a_dof_dof[k];
               }
             }
          

	  h[i_global_to_local[j_domain_dof[j]]] = aux/diag;
	}

      for (j=i_domain_dof[i+1]-1; j >= i_domain_dof[i]; j--)
	{
	  aux = x[j_domain_dof[j]];
	  for (k =i_dof_dof[j_domain_dof[j]+1]-1;
	       k>=i_dof_dof[j_domain_dof[j]]; k--)
             if (i_global_to_local[j_dof_dof[k]] > -1)
             {
	      /* this is a_{i_loc, j_loc} --------------------------------- */
               if (j_dof_dof[k] != j_domain_dof[j])
               {
                  aux -= a_dof_dof[k] * h[i_global_to_local[j_dof_dof[k]]];
               }
               else
               {
                  diag = a_dof_dof[k];
               }
             }
          
	  h[i_global_to_local[j_domain_dof[j]]] = aux/diag;
	}
      nu++;
      if (nu < nu_max)
	goto loop_nu;

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  P_t_coeff[j]= h[i_global_to_local[j_domain_dof[j]]];
	  i_global_to_local[j_domain_dof[j]] = -1;
	}

    }

  
  hypre_TFree(diag_dof_dof);


  hypre_TFree(x);
  hypre_TFree(h);

  hypre_TFree(i_global_to_local);

	  
  P_t = hypre_CSRMatrixCreate(num_domains, num_dofs,
			      i_domain_dof[num_domains]);


  hypre_CSRMatrixData(P_t) = P_t_coeff;
  hypre_CSRMatrixI(P_t) = i_domain_dof;
  hypre_CSRMatrixJ(P_t) = j_domain_dof;

  row_sum = hypre_CTAlloc(double, num_dofs);
  for (i=0; i < num_dofs; i++)
    row_sum[i] = 0.e0;
  for (i=0; i < num_domains; i++)
    for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
      row_sum[j_domain_dof[j]]+=P_t_coeff[j];

  delta = 0.e0;
  for (i=0; i < num_dofs; i++)
    delta+= (row_sum[i] - 1.e0)*(row_sum[i] - 1.e0);

  hypre_printf("\n unit row_sum deviation in seq_PU_interpolation: %le\n", 
	 sqrt(delta/num_dofs));

  hypre_TFree(row_sum);
    
  ind = 1;
  ierr =
    hypre_CSRMatrixTranspose(P_t, &P, ind);

  *P_pointer = P;

  hypre_CSRMatrixI(P_t) = NULL;
  hypre_CSRMatrixJ(P_t) = NULL;

  hypre_CSRMatrixDestroy(P_t);


  return ierr;

}
/* computes: x = T *v; -------------------------------------------- */
HYPRE_Int
compute_sym_GS_T_action(double *x,
			double *v,
			double *w,

			HYPRE_Int *i_domain_dof,
			HYPRE_Int *j_domain_dof,

			HYPRE_Int nu_max,

			HYPRE_Int    *i_dof_dof,
			HYPRE_Int    *j_dof_dof,
			double *a_dof_dof,

			HYPRE_Int *i_global_to_local,

			HYPRE_Int num_domains,
			HYPRE_Int num_dofs)
{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k;
  HYPRE_Int nu;

  double aux, diag;


  /* one sym_GS based loop: ------------------------------------------- */
  for (i=0; i < num_dofs; i++)
    {
      x[i] = 0.e0;

      i_global_to_local[i] = -1;
    }

  for (i=0; i < num_domains; i++)
    {
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  i_global_to_local[j_domain_dof[j]] = j-i_domain_dof[i];
	  w[j-i_domain_dof[i]] = 0.e0;
	}
      nu = 0;
    loop_nu:
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  aux = v[j_domain_dof[j]];
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    if (i_global_to_local[j_dof_dof[k]] > -1)
	    {  /* this is a_{i_loc, j_loc} --------------------------------- */
	      if (j_dof_dof[k] != j_domain_dof[j])
              {
                 aux -= a_dof_dof[k] * w[i_global_to_local[j_dof_dof[k]]];
              }
	      else
              {
                 diag = a_dof_dof[k];
              }
              
            }
          
	  w[i_global_to_local[j_domain_dof[j]]] = aux/diag;
	}

      for (j=i_domain_dof[i+1]-1; j >= i_domain_dof[i]; j--)
	{
	  aux = v[j_domain_dof[j]];
	  for (k =i_dof_dof[j_domain_dof[j]+1]-1;
	       k>=i_dof_dof[j_domain_dof[j]]; k--)
             if (i_global_to_local[j_dof_dof[k]] > -1)
             {
	      /* this is a_{i_loc, j_loc} --------------------------------- */
                if (j_dof_dof[k] != j_domain_dof[j])
                {
                   aux -= a_dof_dof[k] * w[i_global_to_local[j_dof_dof[k]]];
                }
                else
                {
                   diag = a_dof_dof[k];
                }
             }
          
	  w[i_global_to_local[j_domain_dof[j]]] = aux/diag;
	}
      nu++;
      if (nu < nu_max)
	goto loop_nu;
       

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  x[j_domain_dof[j]] += w[i_global_to_local[j_domain_dof[j]]];
	  w[i_global_to_local[j_domain_dof[j]]] = 0.e0;
	  i_global_to_local[j_domain_dof[j]] = -1;
	}


    }

  return ierr;

}
/* computes: x = \sum A_i *v; -------------------------------------------- */
HYPRE_Int
compute_sum_A_i_action(double *w,
		       double *v,
		       
		       HYPRE_Int *i_domain_dof,
		       HYPRE_Int *j_domain_dof,


		       HYPRE_Int    *i_dof_dof,
		       HYPRE_Int    *j_dof_dof,
		       double *a_dof_dof,

		       HYPRE_Int *i_global_to_local,

		       HYPRE_Int num_domains,
		       HYPRE_Int num_dofs)
{
  HYPRE_Int ierr = 0;
  HYPRE_Int i,j,k;


  for (i=0; i < num_dofs; i++)
    {
      w[i] = 0.e0;
      i_global_to_local[i] = -1;
    }

  for (i=0; i < num_domains; i++)
    {
      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_global_to_local[j_domain_dof[j]] = j-i_domain_dof[i];

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	{
	  for (k=i_dof_dof[j_domain_dof[j]];
	       k<i_dof_dof[j_domain_dof[j]+1]; k++)
	    if (i_global_to_local[j_dof_dof[k]] > -1)
	      /* this is a_{i_loc, j_loc} --------------------------------- */
		w[j_domain_dof[j]]+= a_dof_dof[k] * v[j_dof_dof[k]];
	}

      for (j=i_domain_dof[i]; j < i_domain_dof[i+1]; j++)
	i_global_to_local[j_domain_dof[j]] = -1;
    }

  return ierr;

}


