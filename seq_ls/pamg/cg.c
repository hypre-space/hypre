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
 * Preconditioned conjugate gradient (Omin) functions
 *
 *****************************************************************************/

/* include files for standalone cg.c */
#if 0
#include <stdlib.h>
#include <stdio.h>
#include <math.h> 
#endif

#include "headers.h"

/*--------------------------------------------------------------------------
 * Prototypes:
 *   These functions must be defined somewhere else.
 *--------------------------------------------------------------------------*/

char  *hypre_CGCAlloc( int count, int elt_size );
int    hypre_CGFree( char *ptr ); 
void  *hypre_CGCreateVector( void *vector );
int    hypre_CGDestroyVector( void *vector );
void  *hypre_CGMatvecCreate( void *A, void *x );
int    hypre_CGMatvec( void *matvec_data,
                        double alpha, void *A, void *x, double beta, void *y );
int    hypre_CGMatvecDestroy( void *matvec_data );
double hypre_CGInnerProd( void *x, void *y );
int    hypre_CGCopyVector( void *x, void *y );
int    hypre_CGClearVector( void *x );
int    hypre_CGScaleVector( double alpha, void *x );
int    hypre_CGAxpy( double alpha, void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_CGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   tol;
   double   rel_residual_norm;
   int      max_iter;
   int      two_norm;
   int      rel_change;
   int      stop_crit;

   void    *A;
   void    *p;
   void    *s;
   void    *r;

   void    *matvec_data;

   int    (*precond)();
   int    (*precond_setup)();
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   double  *rel_norms;
   char    *log_file_name;

} hypre_CGData;

/* memory macros for standalone pcg.c */
#if 0
#define hypre_CTAlloc(type, count) \
( (type *)hypre_CGCAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )
#define hypre_TFree(ptr) \
( hypre_CGFree((char *)ptr), ptr = NULL )
#endif

/*--------------------------------------------------------------------------
 * hypre_CGIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_CGIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CGIdentity
 *--------------------------------------------------------------------------*/

int
hypre_CGIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_CGCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 * hypre_CGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CGCreate( )
{
   hypre_CGData *cg_data;

   cg_data = hypre_CTAlloc(hypre_CGData, 1);

   /* set defaults */
   (cg_data -> tol)          = 1.0e-06;
   (cg_data -> max_iter)     = 1000;
   (cg_data -> two_norm)     = 0;
   (cg_data -> rel_change)   = 0;
   (cg_data -> stop_crit)    = 0; /* relative norm */
   (cg_data -> matvec_data)  = NULL;
   (cg_data -> precond)       = hypre_CGIdentity;
   (cg_data -> precond_setup) = hypre_CGIdentitySetup;
   (cg_data -> precond_data)  = NULL;
   (cg_data -> logging)      = 0;
   (cg_data -> norms)        = NULL;
   (cg_data -> rel_norms)    = NULL;

   return (void *) cg_data;
}

/*--------------------------------------------------------------------------
 * hypre_CGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_CGDestroy( void *pcg_vdata )
{
   hypre_CGData *cg_data = pcg_vdata;
   int ierr = 0;

   if (cg_data)
   {
      if ((cg_data -> logging) > 0)
      {
         hypre_TFree(cg_data -> norms);
         hypre_TFree(cg_data -> rel_norms);
      }

      hypre_CGMatvecDestroy(cg_data -> matvec_data);

      hypre_CGDestroyVector(cg_data -> p);
      hypre_CGDestroyVector(cg_data -> s);
      hypre_CGDestroyVector(cg_data -> r);

      hypre_TFree(cg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_CGSetup
 *--------------------------------------------------------------------------*/

int
hypre_CGSetup( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            max_iter         = (cg_data -> max_iter);
   int          (*precond_setup)() = (cg_data -> precond_setup);
   void          *precond_data     = (cg_data -> precond_data);
   int            ierr = 0;

   (cg_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   (cg_data -> p) = hypre_CGCreateVector(x);
   (cg_data -> s) = hypre_CGCreateVector(x);
   (cg_data -> r) = hypre_CGCreateVector(b);

   (cg_data -> matvec_data) = hypre_CGMatvecCreate(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((cg_data -> logging) > 0)
   {
      (cg_data -> norms)     = hypre_CTAlloc(double, max_iter + 1);
      (cg_data -> rel_norms) = hypre_CTAlloc(double, max_iter + 1);
      (cg_data -> log_file_name) = "pcg.out.log";
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSolve
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>  <  (tol^2)*<C*b,b> = eps
 *
 *--------------------------------------------------------------------------*/

int
hypre_CGSolve( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGData  *cg_data     = pcg_vdata;

   double          tol          = (cg_data -> tol);
   int             max_iter     = (cg_data -> max_iter);
   int             two_norm     = (cg_data -> two_norm);
   int             rel_change   = (cg_data -> rel_change);
   int             stop_crit    = (cg_data -> stop_crit);
   void           *p            = (cg_data -> p);
   void           *s            = (cg_data -> s);
   void           *r            = (cg_data -> r);
   void           *matvec_data  = (cg_data -> matvec_data);
   int           (*precond)()   = (cg_data -> precond);
   void           *precond_data = (cg_data -> precond_data);
   int             logging      = (cg_data -> logging);
   double         *norms        = (cg_data -> norms);
   double         *rel_norms    = (cg_data -> rel_norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps, tmp_norm;
   double          pi_prod, xi_prod;
                
   int             i = 0, j;
   int             ierr = 0;
/*   int             my_id, num_procs; */
/*   char		  *log_file_name; */
/*   FILE		  *fp; */

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/
   
/*   hypre_CGCommInfo(A,&my_id,&num_procs); */
   if (logging > 0)
   {
/*      log_file_name = (cg_data -> log_file_name); */
/*      fp
 = fopen(log_file_name,"w"); */
   }

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = hypre_CGInnerProd(b, b);
      if (logging > 0 )
          printf("<b,b>: %e\n",bi_prod);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      hypre_CGClearVector(p);
      precond(precond_data, A, b, p);
      bi_prod = hypre_CGInnerProd(p, b);
      if (logging > 0)
          printf("<C*b,b>: %e\n",bi_prod);
   }

#if 0
   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      hypre_CGCopyVector(b, x);
      if (logging > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      ierr = 0;
      return ierr;
   }
#endif

   /* r = b - Ax */
   hypre_CGCopyVector(b, r);
   hypre_CGMatvec(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   tmp_norm = sqrt(hypre_CGInnerProd(r,r));
   if (logging > 0)
   {
      norms[0] = tmp_norm;
   }

   /* p = C*r */
   hypre_CGClearVector(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> */
   gamma = hypre_CGInnerProd(r,p);

   if (bi_prod > 0.0)
   {
       if ( stop_crit && !rel_change ) eps = tol*tol;
       else                            eps = (tol*tol)*bi_prod;
   }
   else
   {
      if (two_norm)
      {
         if ( stop_crit && !rel_change ) 
         {
            eps = tol*tol;
            if (logging > 0 )
            {
               printf("Exiting when ||r||_2 < tol \n");
               printf("Initial ||r0||_2: %e\n",norms[0]);
            }
         }
         else
         {
            eps = (tol*tol)*hypre_CGInnerProd(r,r);
            if (logging > 0)
            {
               printf("Exiting when ||r||_2 < tol * ||r0||_2\n");
               printf("Initial ||r0||_2: %e\n",norms[0]);
            }
         }
      }
      else
      {
         eps = (tol*tol)*gamma;
         if (logging > 0)
         {
            printf("Exiting when ||r||_C < tol * ||r0||_C\n");
            printf("Initial ||r0||_C: %e\n",sqrt(gamma));
         }
      }
   }

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      hypre_CGMatvec(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / hypre_CGInnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_CGAxpy(alpha, p, x);

      /* r = r - alpha*s */
      hypre_CGAxpy(-alpha, s, r);
	 
      /* s = C*r */
      hypre_CGClearVector(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = hypre_CGInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = hypre_CGInnerProd(r,r);
      else
	 i_prod = gamma;

#if 0
      if (two_norm)
	 printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
		i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
      else
	 printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
		i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif
 
      /* log norm info */
      if (logging > 0)
      {
         norms[i]     = sqrt(i_prod);
         rel_norms[i] = sqrt(tol*tol*(i_prod/eps));
      }

      /* check for convergence */
      if (i_prod < eps)
      {
         if (rel_change)
         {
            pi_prod = hypre_CGInnerProd(p,p);
            xi_prod = hypre_CGInnerProd(x,x);
            if ((alpha*alpha*pi_prod/xi_prod) < tol*tol)
               break;
         }
         else
         {
            break;
         }
      }

      beta = gamma / gamma_old;

      /* p = s + beta p */
      hypre_CGScaleVector(beta, p);   
      hypre_CGAxpy(1.0, s, p);
   }

#if 0
   if (two_norm)
      printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
   else
      printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
	     i, sqrt(i_prod), (bi_prod ? sqrt(i_prod/bi_prod) : 0));
#endif

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   if (logging > 0)
   {
      if (bi_prod > 0.0)
         {if (two_norm)
             {
              if ( stop_crit && !rel_change )
              {
                printf("\n\n");
                printf("Iters       ||r||_2      conv.rate\n");
                printf("-----    ------------    ---------\n");
                for (j = 1; j <= i; j++)
                {
                   printf("% 5d    %e    %f\n", j, norms[j], norms[j]/ 
		       norms[j-1]);
                }
              }
              else
              {
                printf("\n\n");
                printf("Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
                printf("-----    ------------    ---------  ------------ \n");
                for (j = 1; j <= i; j++)
                {
                   printf("% 5d    %e    %f   %e\n", j, norms[j], norms[j]/ 
		       norms[j-1], rel_norms[j]);
                }
              }
             }
          else
             {
              printf("\n\n");
              printf("Iters       ||r||_C      conv.rate  ||r||_C/||b||_C\n");
              printf("-----    ------------    ---------  ------------ \n");
              for (j = 1; j <= i; j++)
              {
                 printf("% 5d    %e    %f   %e\n", j, norms[j], norms[j]/ 
		     norms[j-1], rel_norms[j]);
              }
             }
         printf("\n\n");}
      /* fclose(fp);}; */
      else
         {if (two_norm)
             {
              printf("\n\n");
              printf("Iters       ||r||_2      conv.rate\n");
              printf("-----    ------------    ---------\n");
             }
          else
             {
              printf("\n\n");
              printf("Iters       ||r||_C      conv.rate\n");
              printf("-----    ------------    ---------\n");
         }
         for (j = 1; j <= i; j++)
         {
            printf("% 5d    %e    %f \n",
                   j, norms[j], norms[j]/norms[j-1]);
         }
         printf("\n\n");};
      /* fclose(fp);}; */
   }

   (cg_data -> num_iterations) = i;
   if (logging > 0)
      (cg_data -> rel_residual_norm) = rel_norms[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_CGSetTol( void   *pcg_vdata,
                 double  tol       )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_CGSetMaxIter( void *pcg_vdata,
                     int   max_iter  )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_CGSetStopCrit( void  *pcg_vdata,
                         int   stop_crit       )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
hypre_CGSetTwoNorm( void *pcg_vdata,
                     int   two_norm  )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> two_norm) = two_norm;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_CGSetRelChange( void *pcg_vdata,
                       int   rel_change  )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_CGSetPrecond( void  *pcg_vdata,
                        int  (*precond)(),
                        int  (*precond_setup)(),
                        void  *precond_data )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> precond)       = precond;
   (cg_data -> precond_setup) = precond_setup;
   (cg_data -> precond_data)  = precond_data;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGGetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_CGGetPrecond( void         *pcg_vdata,
                        HYPRE_Solver *precond_data_ptr )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;

   *precond_data_ptr = (HYPRE_Solver)(cg_data -> precond_data);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_CGSetLogging( void *pcg_vdata,
                     int   logging)
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;
 
   (cg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_CGGetNumIterations( void *pcg_vdata,
                           int  *num_iterations )
{
   hypre_CGData *cg_data = pcg_vdata;
   int            ierr = 0;

   *num_iterations = (cg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_CGPrintLogging( void *pcg_vdata,
                       int   myid)
{
   hypre_CGData *cg_data = pcg_vdata;

   int            num_iterations  = (cg_data -> num_iterations);
   int            logging         = (cg_data -> logging);
   double        *norms           = (cg_data -> norms);
   double        *rel_norms       = (cg_data -> rel_norms);

   int            i;
   int            ierr = 0;

   if (myid == 0)
   {
      if (logging > 0)
      {
         for (i = 0; i < num_iterations; i++)
         {
            printf("Residual norm[%d] = %e   ", i, norms[i]);
            printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
         }
      }
   }
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_CGGetFinalRelativeResidualNorm( void   *pcg_vdata,
                                       double *relative_residual_norm )
{
   hypre_CGData *cg_data = pcg_vdata;

   int            num_iterations  = (cg_data -> num_iterations);
   int            logging         = (cg_data -> logging);
   double        *rel_norms       = (cg_data -> rel_norms);

   int            ierr = -1;
   
   if (logging > 0)
   {
      *relative_residual_norm = rel_norms[num_iterations];
      ierr = 0;
   }
   
   return ierr;
}

