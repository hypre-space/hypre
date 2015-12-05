

/******************************************************************************
 *
 * a few more relaxation schemes: Chebychev, FCF-Jacobi, CG  - 
 * these do not go through the CF interface (hypre_BoomerAMGRelaxIF)
 *
 *****************************************************************************/

#include "headers.h"
#include "float.h"

HYPRE_Int hypre_LINPACKcgtql1(HYPRE_Int*,double *,double *,HYPRE_Int *);

/******************************************************************************
 *
 *use max norm to estimate largest eigenvalue
 *
 *****************************************************************************/


HYPRE_Int hypre_ParCSRMaxEigEstimate(hypre_ParCSRMatrix *A, /* matrix to relax with */
                               HYPRE_Int scale, /* scale by diagonal?*/
                               double *max_eig)
{
                              
   double e_max;
   double row_sum, max_norm;
   double *col_val;
   double temp;
   double diag_value;

   HYPRE_Int   pos_diag, neg_diag;
   HYPRE_Int   start_row, end_row;
   HYPRE_Int   row_length;
   HYPRE_Int *col_ind;
   HYPRE_Int   j;
   HYPRE_Int i;
   

   /* estimate with the inf-norm of A - should be ok for SPD matrices */

   start_row  = hypre_ParCSRMatrixFirstRowIndex(A);
   end_row    =  hypre_ParCSRMatrixLastRowIndex(A);
    
   max_norm = 0.0;

   pos_diag = neg_diag = 0;
 
   for ( i = start_row; i <= end_row; i++ )
   {
      HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) A, i, &row_length, &col_ind, &col_val);

      row_sum = 0.0;

      for (j = 0; j < row_length; j++)
      {
         if (j==0) diag_value = fabs(col_val[j]);
     
         row_sum += fabs(col_val[j]);

         if ( col_ind[j] == i && col_val[j] > 0.0 ) pos_diag++;
         if ( col_ind[j] == i && col_val[j] < 0.0 ) neg_diag++;
      }
      if (scale)
      {
         if (diag_value != 0.0)
            row_sum = row_sum/diag_value;
      }
      

      if ( row_sum > max_norm ) max_norm = row_sum;

      HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) A, i, &row_length, &col_ind, &col_val);
   }

   /* get max across procs */
   hypre_MPI_Allreduce(&max_norm, &temp, 1, hypre_MPI_DOUBLE, hypre_MPI_MAX, hypre_ParCSRMatrixComm(A)); 
   max_norm = temp;

   /* from Charles */
   if ( pos_diag == 0 && neg_diag > 0 ) max_norm = - max_norm;
   
   /* eig estimates */
   e_max = max_norm;
      
   /* return */
   *max_eig = e_max;
   
   return hypre_error_flag;

}

/******************************************************************************
   use CG to get the eigenvalue estimate 
  scale means get eig est of  (D^{-1/2} A D^{-1/2} 
******************************************************************************/

HYPRE_Int hypre_ParCSRMaxEigEstimateCG(hypre_ParCSRMatrix *A, /* matrix to relax with */
                                 HYPRE_Int scale, /* scale by diagonal?*/
                                 HYPRE_Int max_iter,
                                 double *max_eig, 
                                 double *min_eig)
{

   HYPRE_Int i, j, err;
  
   hypre_ParVector    *p;
   hypre_ParVector    *s;
   hypre_ParVector    *r;
   hypre_ParVector    *ds;
   hypre_ParVector    *u;


   double   *tridiag;
   double   *trioffd;

   double lambda_max , max_row_sum;
   
   double beta, gamma = 0.0, alpha, sdotp, gamma_old, alphainv;
  
   double diag;
 
   double lambda_min;
   
   double *s_data, *p_data, *ds_data, *u_data;

   HYPRE_Int local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double         *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);


   /* check the size of A - don't iterate more than the size */
   HYPRE_Int size = hypre_ParCSRMatrixGlobalNumRows(A);
   
   if (size < max_iter)
      max_iter = size;

   /* create some temp vectors: p, s, r , ds, u*/

   r = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(r);
   hypre_ParVectorSetPartitioningOwner(r,0);

   p = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(p);
   hypre_ParVectorSetPartitioningOwner(p,0);


   s = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(s);
   hypre_ParVectorSetPartitioningOwner(s,0);


   ds = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(ds);
   hypre_ParVectorSetPartitioningOwner(ds,0);

   u = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(u);
   hypre_ParVectorSetPartitioningOwner(u,0);


   /* point to local data */
   s_data = hypre_VectorData(hypre_ParVectorLocalVector(s));
   p_data = hypre_VectorData(hypre_ParVectorLocalVector(p));
   ds_data = hypre_VectorData(hypre_ParVectorLocalVector(ds));
   u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

   /* make room for tri-diag matrix */
    tridiag  = hypre_CTAlloc(double, max_iter+1);
    trioffd  = hypre_CTAlloc(double, max_iter+1);
    for (i=0; i < max_iter + 1; i++)
    {
       tridiag[i] = 0;
       trioffd[i] = 0;
    }


    /* set residual to random */
    hypre_ParVectorSetRandomValues(r,1);
    
    if (scale)
    {
       for (i = 0; i < local_size; i++)
       {
          diag = A_diag_data[A_diag_i[i]];
          ds_data[i] = 1/sqrt(diag);
       }
       
    }
    else
    {
       /* set ds to 1 */
       hypre_ParVectorSetConstantValues(ds,1.0);
    }
    
 
    /* gamma = <r,Cr> */
    gamma = hypre_ParVectorInnerProd(r,p);

    /* for the initial filling of the tridiag matrix */
    beta = 1.0;
    max_row_sum = 0.0;
    
    i = 0;
    while (i < max_iter)
    {
       
        /* s = C*r */
       /* TO DO:  C = diag scale */
       hypre_ParVectorCopy(r, s);
       
       /*gamma = <r,Cr> */
       gamma_old = gamma;
       gamma = hypre_ParVectorInnerProd(r,s);

       if (i==0)
       {
          beta = 1.0;
          /* p_0 = C*r */
          hypre_ParVectorCopy(s, p);
       }
       else
       {
          /* beta = gamma / gamma_old */
          beta = gamma / gamma_old;
          
          /* p = s + beta p */
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
          for (j=0; j < local_size; j++)
          {
             p_data[j] = s_data[j] + beta*p_data[j];
          }
       }
       
       if (scale)
       {
           /* s = D^{-1/2}A*D^{-1/2}*p */
          for (j = 0; j < local_size; j++)
          {
             u_data[j] = ds_data[j] * p_data[j];
          }
          hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);
          for (j = 0; j < local_size; j++)
          {
             s_data[j] = ds_data[j] * s_data[j];
          }


       }
       else
       {
          /* s = A*p */
          hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, s);
       }
       
       /* <s,p> */
       sdotp =  hypre_ParVectorInnerProd(s,p);

       /* alpha = gamma / <s,p> */
       alpha = gamma/sdotp;
      
       /* get tridiagonal matrix */
       alphainv = 1.0/alpha;

       tridiag[i+1] = alphainv;
       tridiag[i] *= beta;
       tridiag[i] += alphainv;

       trioffd[i+1] = alphainv;
       trioffd[i] *= sqrt(beta);  

       /* x = x + alpha*p */
       /* don't need */

       /* r = r - alpha*s */
       hypre_ParVectorAxpy( -alpha, s, r);
              
       i++;
       
    }

    /* eispack routine - eigenvalues return in tridiag and ordered*/
    hypre_LINPACKcgtql1(&i,tridiag,trioffd,&err);
    
    lambda_max = tridiag[i-1];
    lambda_min = tridiag[0];
    /* hypre_printf("linpack max eig est = %g\n", lambda_max);*/
    /* hypre_printf("linpack min eig est = %g\n", lambda_min);*/
  

    hypre_ParVectorDestroy(r);
    hypre_ParVectorDestroy(s);
    hypre_ParVectorDestroy(p);
    hypre_ParVectorDestroy(ds);
    hypre_ParVectorDestroy(u);

   /* return */
   *max_eig = lambda_max;
   *min_eig = lambda_min;

   return hypre_error_flag;

}



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

HYPRE_Int hypre_ParCSRRelax_Cheby(hypre_ParCSRMatrix *A, /* matrix to relax with */
                            hypre_ParVector *f,    /* right-hand side */
                            double max_eig,      
                            double min_eig,     
                            double fraction,   
                            HYPRE_Int order,            /* polynomial order */
                            HYPRE_Int scale,            /* scale by diagonal?*/
                            HYPRE_Int variant,           
                            hypre_ParVector *u,   /* initial/updated approximation */
                            hypre_ParVector *v    /* temporary vector */,
                            hypre_ParVector *r    /*another temp vector */  )
{
   
   
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double         *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   double *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   double *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   double  *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   double theta, delta;
   
   double den;
   double upper_bound, lower_bound;
   
   HYPRE_Int i, j;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
 
   double coefs[5];
   double mult;
   double *orig_u;
   
   double tmp_d;

   HYPRE_Int cheby_order;

   double *ds_data, *tmp_data;
   double  diag;

   hypre_ParVector    *ds;
   hypre_ParVector    *tmp_vec;

   /* u = u + p(A)r */

   if (order > 4)
      order = 4;
   if (order < 1)
      order = 1;

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

   orig_u = hypre_CTAlloc(double, num_rows);

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
#pragma omp parallel for private(j) schedule(static) 
#endif

         for ( j = 0; j < num_rows; j++ )
         {
            u_data[j] = mult * r_data[j] + v_data[j];
         }
         
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) schedule(static) 
#endif
      for ( i = 0; i < num_rows; i++ ) 
      {
         u_data[i] = orig_u[i] + u_data[i];
      }
   
   
   }
   else /* scaling! */
   {
      
      /*grab 1/sqrt(diagonal) */
      
      ds = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(ds);
      hypre_ParVectorSetPartitioningOwner(ds,0);
      ds_data = hypre_VectorData(hypre_ParVectorLocalVector(ds));
      
      tmp_vec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(tmp_vec);
      hypre_ParVectorSetPartitioningOwner(tmp_vec,0);
      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

    /* get ds_data and get scaled residual: r = D^(-1/2)f -
       * D^(-1/2)A*u */


#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j,diag) schedule(static) 
#endif
      for (j = 0; j < num_rows; j++)
      {
         diag = A_diag_data[A_diag_i[j]];
         ds_data[j] = 1/sqrt(diag);

         r_data[j] = ds_data[j] * f_data[j];
      }

      hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) schedule(static) 
#endif
      for ( j = 0; j < num_rows; j++ ) 
      {
         r_data[j] += ds_data[j] * tmp_data[j];
      }

      /* save original u, then start 
         the iteration by multiplying r by the cheby coef.*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) schedule(static) 
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
#pragma omp parallel for private(j) schedule(static) 
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            tmp_data[j]  =  ds_data[j] * u_data[j];
         }
         hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

         /* u_new = coef*r + v*/
         mult = coefs[i];

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j,tmp_d) schedule(static) 
#endif
         for ( j = 0; j < num_rows; j++ )
         {
            tmp_d = ds_data[j]* v_data[j];
            u_data[j] = mult * r_data[j] + tmp_d;
         }
         
      } /* end of cheby_order loop */


      /* now we have to scale u_data before adding it to u_orig*/


#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j) schedule(static) 
#endif
      for ( j = 0; j < num_rows; j++ ) 
      {
         u_data[j] = orig_u[j] + ds_data[j]*u_data[j];
      }
   
      hypre_ParVectorDestroy(ds);
      hypre_ParVectorDestroy(tmp_vec);  


   }/* end of scaling code */



   hypre_TFree(orig_u);
  

   

   return hypre_error_flag;

   
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGRelax_FCFJacobi
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi( hypre_ParCSRMatrix *A,
                                    hypre_ParVector    *f,
                                    HYPRE_Int                *cf_marker,
                                    double              relax_weight,
                                    hypre_ParVector    *u,
                                    hypre_ParVector    *Vtemp)
{
   
   HYPRE_Int i;
   HYPRE_Int relax_points[3];
   HYPRE_Int relax_type = 0;
 
   hypre_ParVector    *Ztemp = NULL;
   

   relax_points[0] = -1; /*F */
   relax_points[1] =  1; /*C */
   relax_points[2] = -1; /*F */

   /* if we are on the coarsest level ,the cf_marker will be null
      and we just do one sweep regular jacobi */
   if (cf_marker == NULL)
   {
       hypre_BoomerAMGRelax(A,
                            f,
                            cf_marker,
                            relax_type,
                            0,
                            relax_weight,
                            0.0,
                            NULL,
                            u,
                            Vtemp, Ztemp); 
   }
   else   
   {
      for (i=0; i < 3; i++)
         hypre_BoomerAMGRelax(A,
                              f,
                              cf_marker,
                              relax_type,
                              relax_points[i],
                              relax_weight,
                              0.0,
                              NULL,
                              u,
                              Vtemp, Ztemp); 
   }
   

   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
 * CG Smoother - 
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRRelax_CG( HYPRE_Solver solver,
                             hypre_ParCSRMatrix *A,
                             hypre_ParVector    *f,
                             hypre_ParVector    *u,
                             HYPRE_Int num_its)
{
  
   HYPRE_PCGSetMaxIter(solver, num_its); /* max iterations */
   HYPRE_ParCSRPCGSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)f, (HYPRE_ParVector)u);

#if 0   
   {
      HYPRE_Int myid;
      HYPRE_Int num_iterations;
      double final_res_norm;

      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid ==0)
      {
         hypre_printf("            -----CG PCG Iterations = %d\n", num_iterations);
         hypre_printf("            -----CG PCG Final Relative Residual Norm = %e\n", final_res_norm);
      }
    }
#endif   
   
   return hypre_error_flag;

}





/* tql1.f --
  
  this is the eispack translation - from Barry Smith in Petsc

  Note that this routine always uses real numbers (not complex) even
  if the underlying matrix is Hermitian. This is because the Lanczos
  process applied to Hermitian matrices always produces a real,
  symmetric tridiagonal matrix.
*/

double hypre_LINPACKcgpthy(double*,double*);


HYPRE_Int hypre_LINPACKcgtql1(HYPRE_Int *n,double *d,double *e,HYPRE_Int *ierr)
{
    /* System generated locals */
    HYPRE_Int  i__1,i__2;
    double d__1,d__2,c_b10 = 1.0;

    /* Local variables */
     double c,f,g,h;
     HYPRE_Int  i,j,l,m;
     double p,r,s,c2,c3 = 0.0;
     HYPRE_Int  l1,l2;
     double s2 = 0.0;
     HYPRE_Int  ii;
     double dl1,el1;
     HYPRE_Int  mml;
     double tst1,tst2;

/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL1, */
/*     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND */
/*     WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971). */

/*     THIS SUBROUTINE FINDS THE EIGENVALUES OF A SYMMETRIC */
/*     TRIDIAGONAL MATRIX BY THE QL METHOD. */

/*     ON INPUT */

/*        N IS THE ORDER OF THE MATRIX. */

/*        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX. */

/*        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX */
/*          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY. */

/*      ON OUTPUT */

/*        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN */
/*          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT AND */
/*          ORDERED FOR INDICES 1,2,...IERR-1, BUT MAY NOT BE */
/*          THE SMALLEST EIGENVALUES. */

/*        E HAS BEEN DESTROYED. */

/*        IERR IS SET TO */
/*          ZERO       FOR NORMAL RETURN, */
/*          J          IF THE J-TH EIGENVALUE HAS NOT BEEN */
/*                     DETERMINED AFTER 30 ITERATIONS. */

/*     CALLS CGPTHY FOR  DSQRT(A*A + B*B) . */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------ 
*/
    double ds;

    --e;
    --d;

    *ierr = 0;
    if (*n == 1) {
        goto L1001;
    }

    i__1 = *n;
    for (i = 2; i <= i__1; ++i) {
        e[i - 1] = e[i];
    }

    f = 0.;
    tst1 = 0.;
    e[*n] = 0.;

    i__1 = *n;
    for (l = 1; l <= i__1; ++l) {
        j = 0;
        h = (d__1 = d[l],fabs(d__1)) + (d__2 = e[l],fabs(d__2));
        if (tst1 < h) {
            tst1 = h;
        }
/*     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT .......... */
        i__2 = *n;
        for (m = l; m <= i__2; ++m) {
            tst2 = tst1 + (d__1 = e[m],fabs(d__1));
            if (tst2 == tst1) {
                goto L120;
            }
/*     .......... E(N) IS ALWAYS ZERO,SO THERE IS NO EXIT */
/*                THROUGH THE BOTTOM OF THE LOOP .......... */
        }
L120:
        if (m == l) {
            goto L210;
        }
L130:
        if (j == 30) {
            goto L1000;
        }
        ++j;
/*     .......... FORM SHIFT .......... */
        l1 = l + 1;
        l2 = l1 + 1;
        g = d[l];
        p = (d[l1] - g) / (e[l] * 2.);
        r = hypre_LINPACKcgpthy(&p,&c_b10);
        ds = 1.0; if (p < 0.0) ds = -1.0;
        d[l] = e[l] / (p + ds*r);
        d[l1] = e[l] * (p + ds*r);
        dl1 = d[l1];
        h = g - d[l];
        if (l2 > *n) {
            goto L145;
        }

        i__2 = *n;
        for (i = l2; i <= i__2; ++i) {
            d[i] -= h;
        }

L145:
        f += h;
/*     .......... QL TRANSFORMATION .......... */
        p = d[m];
        c = 1.;
        c2 = c;
        el1 = e[l1];
        s = 0.;
        mml = m - l;
/*     .......... FOR I=M-1 STEP -1 UNTIL L DO -- .......... */
        i__2 = mml;
        for (ii = 1; ii <= i__2; ++ii) {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * e[i];
            h = c * p;
            r = hypre_LINPACKcgpthy(&p,&e[i]);
            e[i + 1] = s * r;
            s = e[i] / r;
            c = p / r;
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);
        }
 
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        tst2 = tst1 + (d__1 = e[l],fabs(d__1));
        if (tst2 > tst1) {
            goto L130;
        }
L210:
        p = d[l] + f;
/*     .......... ORDER EIGENVALUES .......... */
        if (l == 1) {
            goto L250;
        }
/*     .......... FOR I=L STEP -1 UNTIL 2 DO -- .......... */
        i__2 = l;
        for (ii = 2; ii <= i__2; ++ii) {
            i = l + 2 - ii;
            if (p >= d[i - 1]) {
                goto L270;
            }
            d[i] = d[i - 1];
        }

L250:
        i = 1;
L270:
        d[i] = p;
    }

    goto L1001;
/*     .......... SET ERROR -- NO CONVERGENCE TO AN */
/*                EIGENVALUE AFTER 30 ITERATIONS .......... */
L1000:
    *ierr = l;
L1001:
    return 0;
    
} /* cgtql1_ */


double hypre_LINPACKcgpthy(double *a,double *b)
{
    /* System generated locals */
    double ret_val,d__1,d__2,d__3;
 
    /* Local variables */
    double p,r,s,t,u;


/*     FINDS DSQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW */


/* Computing MAX */
    d__1 = fabs(*a),d__2 = fabs(*b);
    p = hypre_max(d__1,d__2);
    if (!p) {
        goto L20;
    }
/* Computing MIN */
    d__2 = fabs(*a),d__3 = fabs(*b);
/* Computing 2nd power */
    d__1 = hypre_min(d__2,d__3) / p;
    r = d__1 * d__1;
L10:
    t = r + 4.;
    if (t == 4.) {
        goto L20;
    }
    s = r / t;
    u = s * 2. + 1.;
    p = u * p;
/* Computing 2nd power */
    d__1 = s / u;
    r = d__1 * d__1 * r;
    goto L10;
L20:
    ret_val = p;

    return ret_val;
} /* cgpthy_ */


/*--------------------------------------------------------------------------
 * hypre_ParCSRRelax_L1_Jacobi (same as the one in AMS, but this allows CF)
  
  u += w D^{-1}(f - A u), where D_ii = ||A(i,:)||_1 
 *--------------------------------------------------------------------------*/

HYPRE_Int  hypre_ParCSRRelax_L1_Jacobi( hypre_ParCSRMatrix *A,
                                  hypre_ParVector    *f,
                                  HYPRE_Int                *cf_marker,
                                  HYPRE_Int                 relax_points,
                                  double              relax_weight,
                                  double             *l1_norms,
                                  hypre_ParVector    *u,
                                  hypre_ParVector    *Vtemp )

{

    
    MPI_Comm	   comm = hypre_ParCSRMatrixComm(A);
    hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
    double         *A_diag_data  = hypre_CSRMatrixData(A_diag);
    HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
    HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
    hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
    HYPRE_Int            *A_offd_i     = hypre_CSRMatrixI(A_offd);
    double         *A_offd_data  = hypre_CSRMatrixData(A_offd);
    HYPRE_Int            *A_offd_j     = hypre_CSRMatrixJ(A_offd);
    hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    hypre_ParCSRCommHandle *comm_handle;
    
    HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
    HYPRE_Int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
    
    hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
    double         *u_data  = hypre_VectorData(u_local);
    
    hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
    double         *f_data  = hypre_VectorData(f_local);
    
    hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
    double         *Vtemp_data = hypre_VectorData(Vtemp_local);
    double 	   *Vext_data;
    double 	   *v_buf_data;
    
    HYPRE_Int            i, j;
    HYPRE_Int            ii, jj;
    HYPRE_Int		   num_sends;
    HYPRE_Int		   index, start;
    HYPRE_Int		   num_procs, my_id ;
    
    double         zero = 0.0;
    double	   res;


    hypre_MPI_Comm_size(comm,&num_procs);  
    hypre_MPI_Comm_rank(comm,&my_id);  

    if (num_procs > 1)
    {
       num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
       
       v_buf_data = hypre_CTAlloc(double, 
                                  hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
       
       Vext_data = hypre_CTAlloc(double,num_cols_offd);
       
       if (num_cols_offd)
       {
          A_offd_j = hypre_CSRMatrixJ(A_offd);
          A_offd_data = hypre_CSRMatrixData(A_offd);
       }
       
       index = 0;
       for (i = 0; i < num_sends; i++)
       {
          start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
          for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
             v_buf_data[index++] 
                = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
       }
       
       comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, 
                                                   Vext_data);
    }

   /*-----------------------------------------------------------------
    * Copy current approximation into temporary vector.
    *-----------------------------------------------------------------*/
    
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
    for (i = 0; i < n; i++)
    {
       Vtemp_data[i] = u_data[i];
    }
    
    if (num_procs > 1)
    { 
       hypre_ParCSRCommHandleDestroy(comm_handle);
       comm_handle = NULL;
    } 
    
    /*-----------------------------------------------------------------
     * Relax all points.
     *-----------------------------------------------------------------*/
    
    if (relax_points == 0)
    {
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
       for (i = 0; i < n; i++)
       {
          
          /*-----------------------------------------------------------
           * If diagonal is nonzero, relax point i; otherwise, skip it.
           *-----------------------------------------------------------*/
          if (A_diag_data[A_diag_i[i]] != zero)
          {
             res = f_data[i];
             for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
             {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * Vtemp_data[ii];
             }
             for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
             {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
             }
             u_data[i] += (relax_weight*res)/l1_norms[i];
          }
       }
    }
    
    /*-----------------------------------------------------------------
     * Relax only C or F points as determined by relax_points.
     *-----------------------------------------------------------------*/
    else
    {
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
       for (i = 0; i < n; i++)
       {
          
          /*-----------------------------------------------------------
           * If i is of the right type ( C or F ) and diagonal is
           * nonzero, relax point i; otherwise, skip it.
           *-----------------------------------------------------------*/
          
          if (cf_marker[i] == relax_points 
              && A_diag_data[A_diag_i[i]] != zero)
          {
             res = f_data[i];
             for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
             {
                ii = A_diag_j[jj];
                res -= A_diag_data[jj] * Vtemp_data[ii];
             }
             for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
             {
                ii = A_offd_j[jj];
                res -= A_offd_data[jj] * Vext_data[ii];
             }
             u_data[i] += (relax_weight * res)/l1_norms[i];
          }
       }     
    }
    if (num_procs > 1)
    {
       hypre_TFree(Vext_data);
       hypre_TFree(v_buf_data);
    }

    return 0;

}


