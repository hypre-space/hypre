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
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      min_iter;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *r0;
   void  *s;
   void  *v;
   void  *p;
   void  *q;

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

} hypre_BiCGSTABData;

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_BiCGSTABCreate( )
{
   hypre_BiCGSTABData *bicgstab_data;
 
   bicgstab_data = hypre_CTAlloc(hypre_BiCGSTABData, 1);
 
   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> min_iter)       = 0;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond)        = hypre_KrylovIdentity;
   (bicgstab_data -> precond_setup)  = hypre_KrylovIdentitySetup;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> p)              = NULL;
   (bicgstab_data -> q)              = NULL;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> r0)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> v)              = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABDestroy( void *bicgstab_vdata )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int i, ierr = 0;
 
   if (bicgstab_data)
   {
      if ((bicgstab_data -> logging) > 0)
      {
         hypre_TFree(bicgstab_data -> norms);
      }
 
      hypre_KrylovMatvecDestroy(bicgstab_data -> matvec_data);
 
      hypre_KrylovDestroyVector(bicgstab_data -> r);
      hypre_KrylovDestroyVector(bicgstab_data -> r0);
      hypre_KrylovDestroyVector(bicgstab_data -> s);
      hypre_KrylovDestroyVector(bicgstab_data -> v);
      hypre_KrylovDestroyVector(bicgstab_data -> p);
      hypre_KrylovDestroyVector(bicgstab_data -> q);
 
      hypre_TFree(bicgstab_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetup
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetup( void *bicgstab_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_BiCGSTABData *bicgstab_data     = bicgstab_vdata;
   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)() = (bicgstab_data -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> p) == NULL)
      (bicgstab_data -> p) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> q) == NULL)
      (bicgstab_data -> q) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> r0) == NULL)
      (bicgstab_data -> r0) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> s) == NULL)
      (bicgstab_data -> s) = hypre_KrylovCreateVector(b);
   if ((bicgstab_data -> v) == NULL)
      (bicgstab_data -> v) = hypre_KrylovCreateVector(b);
 
   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) = hypre_KrylovMatvecCreate(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data -> logging) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((bicgstab_data -> log_file_name) == NULL)
         (bicgstab_data -> log_file_name) = "bicgstab.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSolve
 *-------------------------------------------------------------------------*/

int
hypre_BiCGSTABSolve(void  *bicgstab_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_BiCGSTABData  *bicgstab_data   = bicgstab_vdata;
   int               min_iter     = (bicgstab_data -> min_iter);
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   void             *matvec_data  = (bicgstab_data -> matvec_data);

   void             *r            = (bicgstab_data -> r);
   void             *r0           = (bicgstab_data -> r0);
   void             *s            = (bicgstab_data -> s);
   void             *v            = (bicgstab_data -> v);
   void             *p            = (bicgstab_data -> p);
   void             *q            = (bicgstab_data -> q);

   int 	           (*precond)()   = (bicgstab_data -> precond);
   int 	            *precond_data = (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   double         *norms          = (bicgstab_data -> norms);
   char           *log_file_name  = (bicgstab_data -> log_file_name);
/*   FILE           *fp; */
   
   int        ierr = 0;
   int        iter; 
   int        j; 
   int        my_id, num_procs;
   double     alpha, beta, gamma, epsilon, res, r_norm, b_norm;
   double     epsmac = 1.e-16; 

   hypre_KrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgstab_data -> norms);
      log_file_name  = (bicgstab_data -> log_file_name);
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   hypre_KrylovCopyVector(b,r0);

/* compute initial residual */

   hypre_KrylovMatvec(matvec_data,-1.0, A, x, 1.0, r0);
   hypre_KrylovCopyVector(r0,r);
   hypre_KrylovCopyVector(r0,p);
   r_norm = sqrt(hypre_KrylovInnerProd(r0,r0));
   b_norm = sqrt(hypre_KrylovInnerProd(b,b));
   res = r_norm;
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

        if (r_norm == 0.0)
        {
	   ierr = 0;
	   return ierr;
	}

/* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
	   hypre_KrylovCopyVector(b,r);
           hypre_KrylovMatvec(matvec_data,-1.0,A,x,1.0,r);
	   r_norm = sqrt(hypre_KrylovInnerProd(r,r));
	   if (r_norm <= epsilon)
           {
              if (logging > 0 && my_id == 0)
                 printf("Final L2 norm of residual: %e\n\n", r_norm);
              break;
           }
	   else
	   {
	      hypre_KrylovCopyVector(r,p);
	   }
	}

        iter++;

        precond(precond_data, A, p, v);
        hypre_KrylovMatvec(matvec_data,1.0,A,v,0.0,q);
      	alpha = res/hypre_KrylovInnerProd(r0,q);
	hypre_KrylovAxpy(alpha,v,x);
	hypre_KrylovAxpy(-alpha,q,r);
        precond(precond_data, A, r, v);
        hypre_KrylovMatvec(matvec_data,1.0,A,v,0.0,s);
      	gamma = hypre_KrylovInnerProd(r,s)/hypre_KrylovInnerProd(s,s);
	hypre_KrylovAxpy(gamma,v,x);
	hypre_KrylovAxpy(-gamma,s,r);
        beta = 1.0/res;
        res = hypre_KrylovInnerProd(r0,r);
        beta *= res;    
	hypre_KrylovAxpy(-gamma,q,p);
        hypre_KrylovScaleVector((beta*alpha/gamma),p);
	hypre_KrylovAxpy(1.0,r,p);

	r_norm = sqrt(hypre_KrylovInnerProd(r,r));
	if (logging > 0)
	{
	   norms[iter] = r_norm;
	}
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

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetTol( void   *bicgstab_vdata,
                   double  tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetMinIter( void *bicgstab_vdata,
                       int   min_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> min_iter) = min_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata,
                       int   max_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetStopCrit( void   *bicgstab_vdata,
                        double  stop_crit       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetPrecond( void  *bicgstab_vdata,
                       int  (*precond)(),
                       int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> precond)        = precond;
   (bicgstab_data -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetPrecond( void         *bicgstab_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *precond_data_ptr = (HYPRE_Solver)(bicgstab_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetLogging( void *bicgstab_vdata,
                       int   logging)
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata,
                             int  *num_iterations )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 
