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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif

/*--------------------------------------------------------------------------
 * Prototypes:
 *   These functions must be defined somewhere else.
 *--------------------------------------------------------------------------*/

char  *hypre_PCGCAlloc( int count, int elt_size );
int    hypre_PCGFree( char *ptr );
void  *hypre_PCGCreateVector( void *vector );
int    hypre_PCGDestroyVector( void *vector );
void  *hypre_PCGMatvecCreate( void *A, void *x );
int    hypre_PCGMatvec( void *matvec_data,
                        double alpha, void *A, void *x, double beta, void *y );
int    hypre_PCGMatvecDestroy( void *matvec_data );
double hypre_PCGInnerProd( void *x, void *y );
int    hypre_PCGCopyVector( void *x, void *y );
int    hypre_PCGClearVector( void *x );
int    hypre_PCGScaleVector( double alpha, void *x );
int    hypre_PCGAxpy( double alpha, void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   tol;
   double   cf_tol;
   int      max_iter;
   int      two_norm;
   int      rel_change;

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

} hypre_PCGData;

#define hypre_CTAlloc(type, count) \
( (type *)hypre_PCGCAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TFree(ptr) \
( hypre_PCGFree((char *)ptr), ptr = NULL )

/*--------------------------------------------------------------------------
 * hypre_PCGIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_PCGIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_PCGIdentity
 *--------------------------------------------------------------------------*/

int
hypre_PCGIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_PCGCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 * hypre_PCGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PCGCreate( )
{
   hypre_PCGData *pcg_data;

   pcg_data = hypre_CTAlloc(hypre_PCGData, 1);

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> cf_tol)      = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> matvec_data)  = NULL;
   (pcg_data -> precond)       = hypre_PCGIdentity;
   (pcg_data -> precond_setup) = hypre_PCGIdentitySetup;
   (pcg_data -> precond_data)  = NULL;
   (pcg_data -> logging)      = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;

   return (void *) pcg_data;
}

/*--------------------------------------------------------------------------
 * hypre_PCGDestroy
 *--------------------------------------------------------------------------*/

int
hypre_PCGDestroy( void *pcg_vdata )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int ierr = 0;

   if (pcg_data)
   {
      if ((pcg_data -> logging) > 0)
      {
         hypre_TFree(pcg_data -> norms);
         hypre_TFree(pcg_data -> rel_norms);
      }

      hypre_PCGMatvecDestroy(pcg_data -> matvec_data);

      hypre_PCGDestroyVector(pcg_data -> p);
      hypre_PCGDestroyVector(pcg_data -> s);
      hypre_PCGDestroyVector(pcg_data -> r);

      hypre_TFree(pcg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetup
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetup( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            max_iter         = (pcg_data -> max_iter);
   int          (*precond_setup)() = (pcg_data -> precond_setup);
   void          *precond_data     = (pcg_data -> precond_data);
   int            ierr = 0;

   (pcg_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   (pcg_data -> p) = hypre_PCGCreateVector(x);
   (pcg_data -> s) = hypre_PCGCreateVector(x);
   (pcg_data -> r) = hypre_PCGCreateVector(b);

   (pcg_data -> matvec_data) = hypre_PCGMatvecCreate(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((pcg_data -> logging) > 0)
   {
      (pcg_data -> norms)     = hypre_CTAlloc(double, max_iter + 1);
      (pcg_data -> rel_norms) = hypre_CTAlloc(double, max_iter + 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSolve
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
 *       gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 *--------------------------------------------------------------------------*/

int
hypre_PCGSolve( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_PCGData  *pcg_data     = pcg_vdata;

   double          tol          = (pcg_data -> tol);
   double          cf_tol       = (pcg_data -> cf_tol);
   int             max_iter     = (pcg_data -> max_iter);
   int             two_norm     = (pcg_data -> two_norm);
   int             rel_change   = (pcg_data -> rel_change);
   void           *p            = (pcg_data -> p);
   void           *s            = (pcg_data -> s);
   void           *r            = (pcg_data -> r);
   void           *matvec_data  = (pcg_data -> matvec_data);
   int           (*precond)()   = (pcg_data -> precond);
   void           *precond_data = (pcg_data -> precond_data);
   int             logging      = (pcg_data -> logging);
   double         *norms        = (pcg_data -> norms);
   double         *rel_norms    = (pcg_data -> rel_norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
   double          pi_prod, xi_prod;
                
   double          i_prod_0;
   double          cf_ave_0 = 0.0;
   double          cf_ave_1 = 0.0;
   double          weight;

   double          guard_zero_residual; 

   int             i = 0;
   int             ierr = 0;

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = hypre_PCGInnerProd(b, b);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      hypre_PCGClearVector(p);
      precond(precond_data, A, b, p);
      bi_prod = hypre_PCGInnerProd(p, b);
   }
   eps = tol*tol;

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      hypre_PCGCopyVector(b, x);
      if (logging > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      ierr = 0;
      return ierr;
   }

   /* r = b - Ax */
   hypre_PCGCopyVector(b, r);
   hypre_PCGMatvec(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   if (logging > 0 || cf_tol > 0.0)
   {
      i_prod_0   = hypre_PCGInnerProd(r,r);
      if (logging > 0) norms[0] = sqrt(i_prod_0);
   }

   /* p = C*r */
   hypre_PCGClearVector(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> */
   gamma = hypre_PCGInnerProd(r,p);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      hypre_PCGMatvec(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / hypre_PCGInnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_PCGAxpy(alpha, p, x);

      /* r = r - alpha*s */
      hypre_PCGAxpy(-alpha, s, r);
         
      /* s = C*r */
      hypre_PCGClearVector(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = hypre_PCGInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
         i_prod = hypre_PCGInnerProd(r,r);
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
         rel_norms[i] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }

      /* check for convergence */
      if (i_prod / bi_prod < eps)
      {
         if (rel_change && i_prod > guard_zero_residual)
         {
            pi_prod = hypre_PCGInnerProd(p,p);
            xi_prod = hypre_PCGInnerProd(x,x);
            if ((alpha*alpha*pi_prod/xi_prod) < eps)
               break;
         }
         else
         {
            break;
         }
      }

      /*--------------------------------------------------------------------
       * Optional test to see if adequate progress is being made.
       * The average convergence factor is recorded and compared
       * against the tolerance 'cf_tol'. The weighting factor is  
       * intended to pay more attention to the test when an accurate
       * estimate for average convergence factor is available.  
       *--------------------------------------------------------------------*/

      if (cf_tol > 0.0)
      {
         cf_ave_0 = cf_ave_1;
         cf_ave_1 = pow( i_prod / i_prod_0, 1.0/(2.0*i)); 

         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
#if 0
         printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                i, cf_ave_1, cf_ave_0, weight );
#endif
         if (weight * cf_ave_1 > cf_tol) break;
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      hypre_PCGScaleVector(beta, p);   
      hypre_PCGAxpy(1.0, s, p);
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

#if 0
   if (logging > 0)
   {
      if (two_norm)
      {
         printf("Iters       ||r||_2    ||r||_2/||b||_2\n");
         printf("-----    ------------    ------------ \n");
      }
      else
      {
         printf("Iters       ||r||_C    ||r||_C/||b||_C\n");
         printf("-----    ------------    ------------ \n");
      }
      for (j = 1; j <= i; j++)
      {
         printf("% 5d    %e    %e\n", j, norms[j], rel_norms[j]);
      }
   }
#endif

   (pcg_data -> num_iterations) = i;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetTol
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetTol( void   *pcg_vdata,
                 double  tol       )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetConvergenceTol
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetConvergenceFactorTol( void   *pcg_vdata,
                                  double  cf_tol   )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> cf_tol) = cf_tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetMaxIter( void *pcg_vdata,
                     int   max_iter  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetTwoNorm( void *pcg_vdata,
                     int   two_norm  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> two_norm) = two_norm;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetRelChange( void *pcg_vdata,
                       int   rel_change  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetPrecond( void  *pcg_vdata,
                     int  (*precond)(),
                     int  (*precond_setup)(),
                     void  *precond_data )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> precond)       = precond;
   (pcg_data -> precond_setup) = precond_setup;
   (pcg_data -> precond_data)  = precond_data;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetLogging( void *pcg_vdata,
                     int   logging)
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_PCGGetNumIterations( void *pcg_vdata,
                           int  *num_iterations )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;

   *num_iterations = (pcg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_PCGPrintLogging( void *pcg_vdata,
                       int   myid)
{
   hypre_PCGData *pcg_data = pcg_vdata;

   int            num_iterations  = (pcg_data -> num_iterations);
   int            logging         = (pcg_data -> logging);
   double        *norms           = (pcg_data -> norms);
   double        *rel_norms       = (pcg_data -> rel_norms);

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
 * hypre_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_PCGGetFinalRelativeResidualNorm( void   *pcg_vdata,
                                       double *relative_residual_norm )
{
   hypre_PCGData *pcg_data = pcg_vdata;

   int            num_iterations  = (pcg_data -> num_iterations);
   int            logging         = (pcg_data -> logging);
   double        *rel_norms       = (pcg_data -> rel_norms);

   int            ierr = -1;
   
   if (logging > 0)
   {
      *relative_residual_norm = rel_norms[num_iterations];
      ierr = 0;
   }
   
   return ierr;
}

