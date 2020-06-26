/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"


/******************************************************************************

Chebyshev relaxation

 
Can specify order 1-4 (this is the order of the resid polynomial)- here we
explicitly code the coefficients (instead of
iteratively determining)


variant 0: standard chebyshev
this is rlx 11 if scale = 0, and 16 if scale == 1

variant 1: modified cheby: T(t)* f(t) where f(t) = (1-b/t)
this is rlx 15 if scale = 0, and 17 if scale == 1

ratio indicates the percentage of the whole spectrum to use (so .5
means half, and .1 means 10percent)


*******************************************************************************/

HYPRE_Int hypre_ParCSRRelax_Cheby_Setup(hypre_ParCSRMatrix *A, /* matrix to relax with */
                            HYPRE_Real max_eig,      
                            HYPRE_Real min_eig,     
                            HYPRE_Real fraction,   
                            HYPRE_Int order,            /* polynomial order */
                            HYPRE_Int scale,            /* scale by diagonal?*/
                            HYPRE_Int variant,           
                            HYPRE_Real **coefs_ptr,
                            HYPRE_Real **ds_ptr)   /* initial/updated approximation */
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);

   HYPRE_Real theta, delta;
   
   HYPRE_Real den;
   HYPRE_Real upper_bound, lower_bound;
   
   HYPRE_Int j;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
 
   HYPRE_Real *coefs = NULL;
   
   HYPRE_Int cheby_order;

   HYPRE_Real *ds_data = NULL;
   HYPRE_Real  diag;

   /* u = u + p(A)r */

   if (order > 4)
      order = 4;
   if (order < 1)
      order = 1;

   coefs = hypre_CTAlloc(HYPRE_Real,  order+1, HYPRE_MEMORY_HOST);
   /* we are using the order of p(A) */
   cheby_order = order -1;
   
    /* make sure we are large enough -  Adams et al. 2003 */
   upper_bound = max_eig * 1.1;
   /* lower_bound = max_eig/fraction; */
   lower_bound = (upper_bound - min_eig)* fraction + min_eig; 


   /* theta and delta */
   theta = (upper_bound + lower_bound)/2;
   delta = (upper_bound - lower_bound)/2;

   if (variant == 1 )
   {
      switch ( cheby_order ) /* these are the corresponding cheby polynomials: u = u_o + s(A)r_0  - so order is
                                one less that  resid poly: r(t) = 1 - t*s(t) */ 
      {
         case 0: 
            coefs[0] = 1.0/theta;     
            
            break;
            
         case 1:  /* (del - t + 2*th)/(th^2 + del*th) */
            den = (theta*theta + delta*theta);
            
            coefs[0] = (delta + 2*theta)/den;     
            coefs[1] = -1.0/den;
            
            break;
            
         case 2:  /* (4*del*th - del^2 - t*(2*del + 6*th) + 2*t^2 + 6*th^2)/(2*del*th^2 - del^2*th - del^3 + 2*th^3)*/
            den = 2*delta*theta*theta - delta*delta*theta - pow(delta,3) + 2*pow(theta,3);
            
            coefs[0] = (4*delta*theta - pow(delta,2) +  6*pow(theta,2))/den;
            coefs[1] = -(2*delta + 6*theta)/den;
            coefs[2] =  2/den;
            
            break;
            
         case 3: /* -(6*del^2*th - 12*del*th^2 - t^2*(4*del + 16*th) + t*(12*del*th - 3*del^2 + 24*th^2) + 3*del^3 + 4*t^3 - 16*th^3)/(4*del*th^3 - 3*del^2*th^2 - 3*del^3*th + 4*th^4)*/
            den = - (4*delta*pow(theta,3) - 3*pow(delta,2)*pow(theta,2) - 3*pow(delta,3)*theta + 4*pow(theta,4) );
            
            coefs[0] = (6*pow(delta,2)*theta - 12*delta*pow(theta,2) + 3*pow(delta,3) - 16*pow(theta,3)   )/den;
            coefs[1] = (12*delta*theta - 3*pow(delta,2) + 24*pow(theta,2))/den;
            coefs[2] =  -( 4*delta + 16*theta)/den;
            coefs[3] = 4/den;
            
            break;
      }
   }
   
   else /* standard chebyshev */
   {
   
      switch ( cheby_order ) /* these are the corresponding cheby polynomials: u = u_o + s(A)r_0  - so order is
                                one less thatn resid poly: r(t) = 1 - t*s(t) */ 
      {
         case 0: 
            coefs[0] = 1.0/theta;     
            break;
            
         case 1:  /* (  2*t - 4*th)/(del^2 - 2*th^2) */
            den = delta*delta - 2*theta*theta;
            
            coefs[0] = -4*theta/den;     
            coefs[1] = 2/den;   
            
            break;
            
         case 2: /* (3*del^2 - 4*t^2 + 12*t*th - 12*th^2)/(3*del^2*th - 4*th^3)*/
            den = 3*(delta*delta)*theta - 4*(theta*theta*theta);
            
            coefs[0] = (3*delta*delta - 12 *theta*theta)/den;
            coefs[1] = 12*theta/den;
            coefs[2] = -4/den; 
            
            break;
            
         case 3: /*(t*(8*del^2 - 48*th^2) - 16*del^2*th + 32*t^2*th - 8*t^3 + 32*th^3)/(del^4 - 8*del^2*th^2 + 8*th^4)*/
            den = pow(delta,4) - 8*delta*delta*theta*theta + 8*pow(theta,4);
            
            coefs[0] = (32*pow(theta,3)- 16*delta*delta*theta)/den;
            coefs[1] = (8*delta*delta - 48*theta*theta)/den;
            coefs[2] = 32*theta/den;
            coefs[3] = -8/den;
            
            break;
      }
   }
   *coefs_ptr = coefs;

   if (scale)
   {
      /*grab 1/sqrt(diagonal) */
      ds_data = hypre_CTAlloc(HYPRE_Real,  num_rows, HYPRE_MEMORY_HOST);
      
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j,diag) HYPRE_SMP_SCHEDULE 
#endif
      for (j = 0; j < num_rows; j++)
      {
         diag = A_diag_data[A_diag_i[j]];
         ds_data[j] = 1/sqrt(diag);
      }

   }/* end of scaling code */
   *ds_ptr = ds_data;

   return hypre_error_flag;
}

HYPRE_Int hypre_ParCSRRelax_Cheby_Solve(hypre_ParCSRMatrix *A, /* matrix to relax with */
                            hypre_ParVector *f,    /* right-hand side */
                            HYPRE_Real *ds_data,
                            HYPRE_Real *coefs,
                            HYPRE_Int order,            /* polynomial order */
                            HYPRE_Int scale,            /* scale by diagonal?*/
                            HYPRE_Int variant,           
                            hypre_ParVector *u,   /* initial/updated approximation */
                            hypre_ParVector *v    /* temporary vector */,
                            hypre_ParVector *r    /*another temp vector */  )
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   HYPRE_Real  *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   HYPRE_Int i, j;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
 
   HYPRE_Real mult;
   HYPRE_Real *orig_u;
   
   HYPRE_Int cheby_order;

   HYPRE_Real  *tmp_data;

   hypre_ParVector    *tmp_vec;

   /* u = u + p(A)r */

   if (order > 4)
      order = 4;
   if (order < 1)
      order = 1;

   /* we are using the order of p(A) */
   cheby_order = order -1;
   
   orig_u = hypre_CTAlloc(HYPRE_Real,  num_rows, HYPRE_MEMORY_HOST);

   if (!scale)
   {
      /* get residual: r = f - A*u */
      hypre_ParVectorCopy(f, r); 
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      for ( i = 0; i < num_rows; i++ ) 
      {
         orig_u[i] = u_data[i];
         u_data[i] = r_data[i] * coefs[cheby_order]; 
      }
      for (i = cheby_order - 1; i >= 0; i-- ) 
      {
         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
         mult = coefs[i];
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            u_data[j] = mult * r_data[j] + v_data[j];
         }
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE 
#endif
      for ( i = 0; i < num_rows; i++ ) 
      {
         u_data[i] = orig_u[i] + u_data[i];
      }
   }
   else /* scaling! */
   {
      
      /*grab 1/sqrt(diagonal) */
      
      tmp_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(tmp_vec);
      hypre_ParVectorSetPartitioningOwner(tmp_vec,0);
      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

    /* get ds_data and get scaled residual: r = D^(-1/2)f -
       * D^(-1/2)A*u */

      hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
      for ( j = 0; j < num_rows; j++ ) 
      {
         r_data[j] = ds_data[j] * (f_data[j] + tmp_data[j]);
      }

      /* save original u, then start 
         the iteration by multiplying r by the cheby coef.*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
      for ( j = 0; j < num_rows; j++ ) 
      {
         orig_u[j] = u_data[j]; /* orig, unscaled u */

         u_data[j] = r_data[j] * coefs[cheby_order]; 
      }

      /* now do the other coefficients */   
      for (i = cheby_order - 1; i >= 0; i-- ) 
      {
         /* v = D^(-1/2)AD^(-1/2)u */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            tmp_data[j]  =  ds_data[j] * u_data[j];
         }
         hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

         /* u_new = coef*r + v*/
         mult = coefs[i];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            u_data[j] = mult * r_data[j] + ds_data[j]*v_data[j];
         }
         
      } /* end of cheby_order loop */

      /* now we have to scale u_data before adding it to u_orig*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) HYPRE_SMP_SCHEDULE 
#endif
      for ( j = 0; j < num_rows; j++ ) 
      {
         u_data[j] = orig_u[j] + ds_data[j]*u_data[j];
      }
   
      hypre_ParVectorDestroy(tmp_vec);  

   }/* end of scaling code */

   hypre_TFree(orig_u, HYPRE_MEMORY_HOST);
  
   return hypre_error_flag;
}

