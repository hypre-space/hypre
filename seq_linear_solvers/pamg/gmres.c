/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * GMR gmres
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_GMRData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      k_dim;
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  **p;

   void  *matvec_data;

   int    (*precond)();
   int    (*precond_setup)();
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} hypre_GMRData;

/*--------------------------------------------------------------------------
 * hypre_GMRCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_GMRCreate( )
{
   hypre_GMRData *gmres_data;
 
   gmres_data = hypre_CTAlloc(hypre_GMRData, 1);
 
   /* set defaults */
   (gmres_data -> k_dim)          = 5;
   (gmres_data -> tol)            = 1.0e-06;
   (gmres_data -> min_iter)       = 0;
   (gmres_data -> max_iter)       = 1000;
   (gmres_data -> stop_crit)      = 0; /* rel. residual norm */
   (gmres_data -> precond)        = hypre_CGIdentity;
   (gmres_data -> precond_setup)  = hypre_CGIdentitySetup;
   (gmres_data -> precond_data)   = NULL;
   (gmres_data -> logging)        = 0;
   (gmres_data -> p)              = NULL;
   (gmres_data -> r)              = NULL;
   (gmres_data -> w)              = NULL;
   (gmres_data -> matvec_data)    = NULL;
   (gmres_data -> norms)          = NULL;
   (gmres_data -> log_file_name)  = NULL;
 
   return (void *) gmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_GMRDestroy
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRDestroy( void *gmres_vdata )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int i, ierr = 0;
 
   if (gmres_data)
   {
      if ((gmres_data -> logging) > 0)
      {
         hypre_TFree(gmres_data -> norms);
      }
 
      hypre_CGMatvecDestroy(gmres_data -> matvec_data);
 
      hypre_CGDestroyVector(gmres_data -> r);
      hypre_CGDestroyVector(gmres_data -> w);
      for (i = 0; i < (gmres_data -> k_dim+1); i++)
      {
	 hypre_CGDestroyVector( (gmres_data -> p) [i]);
      }
      hypre_TFree(gmres_data -> p);
 
      hypre_TFree(gmres_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetup
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetup( void *gmres_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_GMRData *gmres_data     = gmres_vdata;
   int            k_dim            = (gmres_data -> k_dim);
   int            max_iter         = (gmres_data -> max_iter);
   int          (*precond_setup)() = (gmres_data -> precond_setup);
   void          *precond_data     = (gmres_data -> precond_data);
   int            ierr = 0;
 
   (gmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((gmres_data -> p) == NULL)
      (gmres_data -> p) = hypre_CGCreateVectorArray(k_dim+1,x);
   if ((gmres_data -> r) == NULL)
      (gmres_data -> r) = hypre_CGCreateVector(b);
   if ((gmres_data -> w) == NULL)
      (gmres_data -> w) = hypre_CGCreateVector(b);
 
   if ((gmres_data -> matvec_data) == NULL)
      (gmres_data -> matvec_data) = hypre_CGMatvecCreate(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((gmres_data -> logging) > 0)
   {
      if ((gmres_data -> norms) == NULL)
         (gmres_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((gmres_data -> log_file_name) == NULL)
         (gmres_data -> log_file_name) = "gmres.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRSolve
 *-------------------------------------------------------------------------*/

int
hypre_GMRSolve(void  *gmres_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_GMRData  *gmres_data   = gmres_vdata;
   int 		     k_dim        = (gmres_data -> k_dim);
   int               min_iter     = (gmres_data -> min_iter);
   int 		     max_iter     = (gmres_data -> max_iter);
   int 		     stop_crit    = (gmres_data -> stop_crit);
   double 	     accuracy     = (gmres_data -> tol);
   void             *matvec_data  = (gmres_data -> matvec_data);

   void             *r            = (gmres_data -> r);
   void             *w            = (gmres_data -> w);
   void            **p            = (gmres_data -> p);

   int 	           (*precond)()   = (gmres_data -> precond);
   int 	            *precond_data = (gmres_data -> precond_data);

   /* logging variables */
   int             logging        = (gmres_data -> logging);
   double         *norms          = (gmres_data -> norms);
/*   char           *log_file_name  = (gmres_data -> log_file_name);
     FILE           *fp; */
   
   int        ierr = 0;
   int	      i, j, k;
   double     *rs, **hh, *c, *s;
   int        iter; 
   int        my_id;
   double     epsilon, gamma, t, r_norm, b_norm;
   double     epsmac = 1.e-16; 

   /* hypre_CGCommInfo(A,&my_id,&num_procs); */
   my_id = 0;
   if (logging > 0)
   {
      norms          = (gmres_data -> norms);
      /* log_file_name  = (gmres_data -> log_file_name);
         fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   rs = hypre_CTAlloc(double,k_dim+1); 
   c = hypre_CTAlloc(double,k_dim); 
   s = hypre_CTAlloc(double,k_dim); 

   hh = hypre_CTAlloc(double*,k_dim+1); 
   for (i=0; i < k_dim+1; i++)
   {	
   	hh[i] = hypre_CTAlloc(double,k_dim); 
   }

   hypre_CGCopyVector(b,p[0]);

/* compute initial residual */

   hypre_CGMatvec(matvec_data,-1.0, A, x, 1.0, p[0]);
   r_norm = sqrt(hypre_CGInnerProd(p[0],p[0]));
   b_norm = sqrt(hypre_CGInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("Initial L2 norm of residual: %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
/* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
     epsilon = accuracy * b_norm;
   }
   else
   {
/* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
     epsilon = accuracy * r_norm;
   };

/* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit)
      epsilon = accuracy;

   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
	   ierr = 0;
	   return ierr;
	}

	if (r_norm <= epsilon && iter >= min_iter) 
        {
		hypre_CGCopyVector(b,r);
          	hypre_CGMatvec(matvec_data,-1.0,A,x,1.0,r);
		r_norm = sqrt(hypre_CGInnerProd(r,r));
		if (r_norm <= epsilon)
                {
                  if (logging > 0 && my_id == 0)
                     printf("Final L2 norm of residual: %e\n\n", r_norm);
                  break;
                }
	}

      	t = 1.0 / r_norm;
	hypre_CGScaleVector(t,p[0]);
	i = 0;
	while (i < k_dim && (r_norm > epsilon || iter < min_iter)
                         && iter < max_iter)
	{
		i++;
		iter++;
		hypre_CGClearVector(r);
		precond(precond_data, A, p[i-1], r);
		hypre_CGMatvec(matvec_data, 1.0, A, r, 0.0, p[i]);
		/* modified Gram_Schmidt */
		for (j=0; j < i; j++)
		{
			hh[j][i-1] = hypre_CGInnerProd(p[j],p[i]);
			hypre_CGAxpy(-hh[j][i-1],p[j],p[i]);
		}
		t = sqrt(hypre_CGInnerProd(p[i],p[i]));
		hh[i][i-1] = t;	
		if (t != 0.0)
		{
			t = 1.0/t;
			hypre_CGScaleVector(t,p[i]);
		}
		/* done with modified Gram_schmidt and Arnoldi step.
		   update factorization of hh */
		for (j = 1; j < i; j++)
		{
			t = hh[j-1][i-1];
			hh[j-1][i-1] = c[j-1]*t + s[j-1]*hh[j][i-1];		
			hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
		}
		gamma = sqrt(hh[i-1][i-1]*hh[i-1][i-1] + hh[i][i-1]*hh[i][i-1]);
		if (gamma == 0.0) gamma = epsmac;
		c[i-1] = hh[i-1][i-1]/gamma;
		s[i-1] = hh[i][i-1]/gamma;
		rs[i] = -s[i-1]*rs[i-1];
		rs[i-1] = c[i-1]*rs[i-1];
		/* determine residual norm */
		hh[i-1][i-1] = c[i-1]*hh[i-1][i-1] + s[i-1]*hh[i][i-1];
		r_norm = fabs(rs[i]);
		if (logging > 0)
		{
		   norms[iter] = r_norm;
		}
	}
	/* now compute solution, first solve upper triangular system */
	
	rs[i-1] = rs[i-1]/hh[i-1][i-1];
	for (k = i-2; k >= 0; k--)
	{
		t = rs[k];
		for (j = k+1; j < i; j++)
		{
			t -= hh[k][j]*rs[j];
		}
		rs[k] = t/hh[k][k];
	}
	/* form linear combination of p's to get solution */
	
	hypre_CGCopyVector(p[0],w);
	hypre_CGScaleVector(rs[0],w);
	for (j = 1; j < i; j++)
		hypre_CGAxpy(rs[j], p[j], w);

	hypre_CGClearVector(r);
	precond(precond_data, A, w, r);

	hypre_CGAxpy(1.0,r,x);

/* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
		hypre_CGCopyVector(b,r);
          	hypre_CGMatvec(matvec_data,-1.0,A,x,1.0,r);
		r_norm = sqrt(hypre_CGInnerProd(r,r));
		if (logging > 0)
		{
		   norms[iter] = r_norm;
		}
		if (r_norm <= epsilon)
                {
                  if (logging > 0 && my_id == 0)
                     printf("Final L2 norm of residual: %e\n\n", r_norm);
                  break;
                }
		else
		{
		   hypre_CGCopyVector(r,p[0]);
		   i = 0;
		}
	}

/* compute residual vector and continue loop */

	for (j=i ; j > 0; j--)
	{
		rs[j-1] = -s[j-1]*rs[j];
		rs[j] = c[j-1]*rs[j];
	}

	if (i) hypre_CGAxpy(rs[0]-1.0,p[0],p[0]);
	for (j=1; j < i+1; j++)
		hypre_CGAxpy(rs[j],p[j],p[0]);	
   }

   if (logging > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      
          for (j = 1; j <= iter; j++)
          {
             printf("% 5d    %e    %f   %e\n", j, norms[j],norms[j]/norms[j-1],
 	             norms[j]/b_norm);
          }
          printf("\n\n"); }

      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
          for (j = 1; j <= iter; j++)
          {
             printf("% 5d    %e    %f\n", j, norms[j],norms[j]/norms[j-1]);
          }
          printf("\n\n"); };
   }

   (gmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (gmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (gmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   hypre_TFree(c); 
   hypre_TFree(s); 
   hypre_TFree(rs);
 
   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFree(hh[i]);
   }
   hypre_TFree(hh); 

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetKDim
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetKDim( void   *gmres_vdata,
                    int   k_dim )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> k_dim) = k_dim;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetTol
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetTol( void   *gmres_vdata,
                   double  tol       )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetMinIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetMinIter( void *gmres_vdata,
                       int   min_iter  )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> min_iter) = min_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetMaxIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetMaxIter( void *gmres_vdata,
                       int   max_iter  )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetStopCrit( void   *gmres_vdata,
                        double  stop_crit       )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRSetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetPrecond( void  *gmres_vdata,
                       int  (*precond)(),
                       int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> precond)        = precond;
   (gmres_data -> precond_setup)  = precond_setup;
   (gmres_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRGetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRGetPrecond( void         *gmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *precond_data_ptr = (HYPRE_Solver)(gmres_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRSetLogging
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRSetLogging( void *gmres_vdata,
                       int   logging)
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRGetNumIterations
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRGetNumIterations( void *gmres_vdata,
                             int  *num_iterations )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *num_iterations = (gmres_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRGetFinalRelativeResidualNorm( void   *gmres_vdata,
                                         double *relative_residual_norm )
{
   hypre_GMRData *gmres_data = gmres_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (gmres_data -> rel_residual_norm);
   
   return ierr;
} 
