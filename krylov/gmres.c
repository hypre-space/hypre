/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#include "krylov.h"

/*--------------------------------------------------------------------------
 * hypre_GMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_GMRESFunctions *
hypre_GMRESFunctionsCreate(
   char * (*CAlloc)        ( int count, int elt_size ),
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( int size, void *vectors ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_GMRESFunctions * gmres_functions;
   gmres_functions = (hypre_GMRESFunctions *)
      CAlloc( 1, sizeof(hypre_GMRESFunctions) );

   gmres_functions->CAlloc = CAlloc;
   gmres_functions->Free = Free;
   gmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   gmres_functions->CreateVector = CreateVector;
   gmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   gmres_functions->DestroyVector = DestroyVector;
   gmres_functions->MatvecCreate = MatvecCreate;
   gmres_functions->Matvec = Matvec;
   gmres_functions->MatvecDestroy = MatvecDestroy;
   gmres_functions->InnerProd = InnerProd;
   gmres_functions->CopyVector = CopyVector;
   gmres_functions->ClearVector = ClearVector;
   gmres_functions->ScaleVector = ScaleVector;
   gmres_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   gmres_functions->precond_setup = PrecondSetup;
   gmres_functions->precond       = Precond;

   return gmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions )
{
   hypre_GMRESData *gmres_data;
 
   gmres_data = hypre_CTAllocF(hypre_GMRESData, 1, gmres_functions);

   gmres_data->functions = gmres_functions;
 
   /* set defaults */
   (gmres_data -> k_dim)          = 5;
   (gmres_data -> tol)            = 1.0e-06;
   (gmres_data -> min_iter)       = 0;
   (gmres_data -> max_iter)       = 1000;
   (gmres_data -> rel_change)     = 0;
   (gmres_data -> stop_crit)      = 0; /* rel. residual norm */
   (gmres_data -> precond_data)   = NULL;
   (gmres_data -> printlevel)     = 0;
   (gmres_data -> log_level)      = 0;
   (gmres_data -> p)              = NULL;
   (gmres_data -> r)              = NULL;
   (gmres_data -> w)              = NULL;
   (gmres_data -> matvec_data)    = NULL;
   (gmres_data -> norms)          = NULL;
   (gmres_data -> log_file_name)  = NULL;
 
   return (void *) gmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESDestroy
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESDestroy( void *gmres_vdata )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int i, ierr = 0;
 
   if (gmres_data)
   {
      if ( (gmres_data->log_level>0) || (gmres_data->printlevel) > 0 )
      {
         hypre_TFreeF( gmres_data -> norms, gmres_functions );
      }
 
      (*(gmres_functions->MatvecDestroy))(gmres_data -> matvec_data);
 
      (*(gmres_functions->DestroyVector))(gmres_data -> r);
      (*(gmres_functions->DestroyVector))(gmres_data -> w);
      for (i = 0; i < (gmres_data -> k_dim+1); i++)
      {
	 (*(gmres_functions->DestroyVector))( (gmres_data -> p) [i]);
      }
      hypre_TFreeF( gmres_data->p, gmres_functions );
      hypre_TFreeF( gmres_data, gmres_functions );
      hypre_TFreeF( gmres_functions, gmres_functions );
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetResidual
 *--------------------------------------------------------------------------*/

int hypre_GMRESGetResidual( void *gmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */
   int ierr = 0;
   hypre_GMRESData  *gmres_data     = gmres_vdata;
   *residual = gmres_data->r;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetup
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetup( void *gmres_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_GMRESData *gmres_data     = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;

   int            k_dim            = (gmres_data -> k_dim);
   int            max_iter         = (gmres_data -> max_iter);
   int          (*precond_setup)() = (gmres_functions->precond_setup);
   void          *precond_data     = (gmres_data -> precond_data);
   int            ierr = 0;
 
   (gmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((gmres_data -> p) == NULL)
      (gmres_data -> p) = (*(gmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((gmres_data -> r) == NULL)
      (gmres_data -> r) = (*(gmres_functions->CreateVector))(b);
   if ((gmres_data -> w) == NULL)
      (gmres_data -> w) = (*(gmres_functions->CreateVector))(b);
 
   if ((gmres_data -> matvec_data) == NULL)
      (gmres_data -> matvec_data) = (*(gmres_functions->MatvecCreate))(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ( (gmres_data->log_level)>0 || (gmres_data->printlevel) > 0 )
   {
      if ((gmres_data -> norms) == NULL)
         (gmres_data -> norms) = hypre_CTAllocF(double, max_iter + 1,gmres_functions);
   }
   if ( (gmres_data->printlevel) > 0 ) {
      if ((gmres_data -> log_file_name) == NULL)
         (gmres_data -> log_file_name) = "gmres.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESSolve
 *-------------------------------------------------------------------------*/

int
hypre_GMRESSolve(void  *gmres_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_GMRESData  *gmres_data   = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int 		     k_dim        = (gmres_data -> k_dim);
   int               min_iter     = (gmres_data -> min_iter);
   int 		     max_iter     = (gmres_data -> max_iter);
   int               rel_change   = (gmres_data -> rel_change);
   int 		     stop_crit    = (gmres_data -> stop_crit);
   double 	     accuracy     = (gmres_data -> tol);
   void             *matvec_data  = (gmres_data -> matvec_data);

   void             *r            = (gmres_data -> r);
   void             *w            = (gmres_data -> w);
   void            **p            = (gmres_data -> p);

   int 	           (*precond)()   = (gmres_functions -> precond);
   int 	            *precond_data = (gmres_data -> precond_data);

   int             printlevel     = (gmres_data -> printlevel);
   int             log_level      = (gmres_data -> log_level);

   double         *norms          = (gmres_data -> norms);
/* not used yet   char           *log_file_name  = (gmres_data -> log_file_name);*/
/*   FILE           *fp; */
   
   int        ierr = 0;
   int	      i, j, k;
   double     *rs, **hh, *c, *s;
   int        iter; 
   int        my_id, num_procs;
   double     epsilon, gamma, t, r_norm, b_norm, x_norm;
   double     epsmac = 1.e-16; 

   double          guard_zero_residual; 

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( log_level>0 || printlevel>0 )
   {
      norms          = (gmres_data -> norms);
      /* not used yet      log_file_name  = (gmres_data -> log_file_name);*/
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(double,k_dim+1,gmres_functions); 
   c = hypre_CTAllocF(double,k_dim,gmres_functions); 
   s = hypre_CTAllocF(double,k_dim,gmres_functions); 

   hh = hypre_CTAllocF(double*,k_dim+1,gmres_functions); 
   for (i=0; i < k_dim+1; i++)
   {	
   	hh[i] = hypre_CTAllocF(double,k_dim,gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);
   r_norm = sqrt((*(gmres_functions->InnerProd))(p[0],p[0]));
   if ( r_norm!=r_norm ) {
      /* ...NaN's in input will generally make r_norm a NaN.  This test
         for  rnorm==NaN  works on all IEEE-compliant compilers/machines,
         c.f. page 8 of "Lecture Notes on the Status of IEEE 754" by W. Kahan,
         May 31, 1996.  Currently (July 2002) this paper may be found at
         http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      ierr += 101;
      return ierr;
   }
   b_norm = sqrt((*(gmres_functions->InnerProd))(b,b));
   if ( log_level>0 || printlevel>0 )
   {
      norms[0] = r_norm;
      if ( printlevel>1 && my_id == 0 )
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
   if ( stop_crit && !rel_change )
      epsilon = accuracy;

   if ( printlevel>1 && my_id == 0 )
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      
          }

      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
          };
   }

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
		(*(gmres_functions->CopyVector))(b,r);
          	(*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
		r_norm = sqrt((*(gmres_functions->InnerProd))(r,r));
		if (r_norm <= epsilon)
                {
                  if ( printlevel>1 && my_id == 0)
                  {
                     printf("\n\n");
                     printf("Final L2 norm of residual: %e\n\n", r_norm);
                  }
                  break;
                }
	}

      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t,p[0]);
	i = 0;
	while (i < k_dim && (r_norm > epsilon || iter < min_iter)
                         && iter < max_iter)
	{
		i++;
		iter++;
		(*(gmres_functions->ClearVector))(r);
		precond(precond_data, A, p[i-1], r);
		(*(gmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
		/* modified Gram_Schmidt */
		for (j=0; j < i; j++)
		{
			hh[j][i-1] = (*(gmres_functions->InnerProd))(p[j],p[i]);
			(*(gmres_functions->Axpy))(-hh[j][i-1],p[j],p[i]);
		}
		t = sqrt((*(gmres_functions->InnerProd))(p[i],p[i]));
		hh[i][i-1] = t;	
		if (t != 0.0)
		{
			t = 1.0/t;
			(*(gmres_functions->ScaleVector))(t,p[i]);
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
		if ( printlevel>0 )
		{
		   norms[iter] = r_norm;
                   if ( printlevel>1 && my_id == 0 )
   		   {
      		      if (b_norm > 0.0)
             	         printf("% 5d    %e    %f   %e\n", iter, 
				norms[iter],norms[iter]/norms[iter-1],
 	             		norms[iter]/b_norm);
      		      else
             	         printf("% 5d    %e    %f\n", iter, norms[iter],
				norms[iter]/norms[iter-1]);
   		   }
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
	
	(*(gmres_functions->CopyVector))(p[0],w);
	(*(gmres_functions->ScaleVector))(rs[0],w);
	for (j = 1; j < i; j++)
		(*(gmres_functions->Axpy))(rs[j], p[j], w);

	(*(gmres_functions->ClearVector))(r);
	precond(precond_data, A, w, r);

	(*(gmres_functions->Axpy))(1.0,r,x);

/* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
		(*(gmres_functions->CopyVector))(b,r);
          	(*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
		r_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
		if (r_norm <= epsilon)
                {
                   if ( printlevel>1 && my_id == 0 )
                   {
                      printf("\n\n");
                      printf("Final L2 norm of residual: %e\n\n", r_norm);
                   }
                   if (rel_change && r_norm > guard_zero_residual)
                      /* Also test on relative change of iterates, x_i - x_(i-1) */
                   {  /* At this point r = x_i - x_(i-1) */
                      x_norm = sqrt( (*(gmres_functions->InnerProd))(x,x) );
                      if ( x_norm<=guard_zero_residual ) break; /* don't divide by 0 */
                      if ( r_norm/x_norm < epsilon )
                         break;
                   }
                   else
                   {
                      break;
                   }
                }
		else
		{
		   (*(gmres_functions->CopyVector))(r,p[0]);
		   i = 0;
		}
	}

/* compute residual vector and continue loop */

	for (j=i ; j > 0; j--)
	{
		rs[j-1] = -s[j-1]*rs[j];
		rs[j] = c[j-1]*rs[j];
	}

	if (i) (*(gmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
	for (j=1; j < i+1; j++)
		(*(gmres_functions->Axpy))(rs[j],p[j],p[0]);	
   }

   if ( printlevel>1 && my_id == 0 )
          printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (gmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (gmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   hypre_TFreeF(c,gmres_functions); 
   hypre_TFreeF(s,gmres_functions); 
   hypre_TFreeF(rs,gmres_functions);
 
   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFreeF(hh[i],gmres_functions);
   }
   hypre_TFreeF(hh,gmres_functions); 

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetKDim
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetKDim( void   *gmres_vdata,
                    int   k_dim )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> k_dim) = k_dim;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetTol
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetTol( void   *gmres_vdata,
                   double  tol       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMinIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetMinIter( void *gmres_vdata,
                       int   min_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> min_iter) = min_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMaxIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetMaxIter( void *gmres_vdata,
                       int   max_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetRelChange( void *gmres_vdata,
                         int   rel_change  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetStopCrit( void   *gmres_vdata,
                        double  stop_crit       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetPrecond( void  *gmres_vdata,
                       int  (*precond)(),
                       int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int              ierr = 0;
 
   (gmres_functions -> precond)        = precond;
   (gmres_functions -> precond_setup)  = precond_setup;
   (gmres_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetPrecond( void         *gmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *precond_data_ptr = (HYPRE_Solver)(gmres_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_PCGSetPrintLevel
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetPrintLevel( void *gmres_vdata,
                        int   level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> printlevel) = level;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetLogLevel
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetLogLevel( void *gmres_vdata,
                      int   level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> log_level) = level;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetNumIterations( void *gmres_vdata,
                             int  *num_iterations )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *num_iterations = (gmres_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetFinalRelativeResidualNorm( void   *gmres_vdata,
                                         double *relative_residual_norm )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (gmres_data -> rel_residual_norm);
   
   return ierr;
} 
