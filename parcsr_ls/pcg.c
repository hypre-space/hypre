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

/*--------------------------------------------------------------------------
 * Prototypes:
 *   These functions must be defined somewhere else.
 *--------------------------------------------------------------------------*/

char  *hypre_KrylovCAlloc( int count, int elt_size );
int    hypre_KrylovFree( char *ptr ); 
void  *hypre_KrylovCreateVector( void *vector );
int    hypre_KrylovDestroyVector( void *vector );
void  *hypre_KrylovMatvecCreate( void *A, void *x );
int    hypre_KrylovMatvec( void *matvec_data,
                        double alpha, void *A, void *x, double beta, void *y );
int    hypre_KrylovMatvecDestroy( void *matvec_data );
double hypre_KrylovInnerProd( void *x, void *y );
int    hypre_KrylovCopyVector( void *x, void *y );
int    hypre_KrylovClearVector( void *x );
int    hypre_KrylovScaleVector( double alpha, void *x );
int    hypre_KrylovAxpy( double alpha, void *x, void *y );

/*--------------------------------------------------------------------------
 * hypre_PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   tol;
   double   rel_residual_norm;
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
   char    *log_file_name;

} hypre_PCGData;

#define hypre_CTAlloc(type, count) \
( (type *)hypre_KrylovCAlloc((unsigned int)(count), (unsigned int)sizeof(type)) )

#define hypre_TFree(ptr) \
( hypre_KrylovFree((char *)ptr), ptr = NULL )

/*--------------------------------------------------------------------------
 * hypre_KrylovIdentitySetup
 *--------------------------------------------------------------------------*/

int
hypre_KrylovIdentitySetup( void *vdata,
                        void *A,
                        void *b,
                        void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovIdentity
 *--------------------------------------------------------------------------*/

int
hypre_KrylovIdentity( void *vdata,
                   void *A,
                   void *b,
                   void *x     )

{
   return( hypre_KrylovCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 * hypre_KrylovCreate
 *--------------------------------------------------------------------------*/

void *
hypre_KrylovCreate( )
{
   hypre_PCGData *pcg_data;

   pcg_data = hypre_CTAlloc(hypre_PCGData, 1);

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> matvec_data)  = NULL;
   (pcg_data -> precond)       = hypre_KrylovIdentity;
   (pcg_data -> precond_setup) = hypre_KrylovIdentitySetup;
   (pcg_data -> precond_data)  = NULL;
   (pcg_data -> logging)      = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;

   return (void *) pcg_data;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovDestroy
 *--------------------------------------------------------------------------*/

int
hypre_KrylovDestroy( void *pcg_vdata )
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

      hypre_KrylovMatvecDestroy(pcg_data -> matvec_data);

      hypre_KrylovDestroyVector(pcg_data -> p);
      hypre_KrylovDestroyVector(pcg_data -> s);
      hypre_KrylovDestroyVector(pcg_data -> r);

      hypre_TFree(pcg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetup
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetup( void *pcg_vdata,
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

   (pcg_data -> p) = hypre_KrylovCreateVector(x);
   (pcg_data -> s) = hypre_KrylovCreateVector(x);
   (pcg_data -> r) = hypre_KrylovCreateVector(b);

   (pcg_data -> matvec_data) = hypre_KrylovMatvecCreate(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((pcg_data -> logging) > 0)
   {
      (pcg_data -> norms)     = hypre_CTAlloc(double, max_iter + 1);
      (pcg_data -> rel_norms) = hypre_CTAlloc(double, max_iter + 1);
      (pcg_data -> log_file_name) = "pcg.out.log";
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSolve
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
hypre_KrylovSolve( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_PCGData  *pcg_data     = pcg_vdata;

   double          tol          = (pcg_data -> tol);
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
   double          bi_prod, i_prod, eps, tmp_norm;
   double          pi_prod, xi_prod;
                
   int             i = 0, j;
   int             ierr = 0;
   int             my_id, num_procs;
   char		  *log_file_name;
/*   FILE		  *fp; */

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/
   
   hypre_KrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      log_file_name = (pcg_data -> log_file_name);
/*      fp
 = fopen(log_file_name,"w"); */
   }

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = hypre_KrylovInnerProd(b, b);
      if (logging > 0 && my_id == 0)
          printf("<b,b>: %e\n",bi_prod);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      hypre_KrylovClearVector(p);
      precond(precond_data, A, b, p);
      bi_prod = hypre_KrylovInnerProd(p, b);
      if (logging > 0 && my_id == 0)
          printf("<C*b,b>: %e\n",bi_prod);
   }

#if 0
   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      hypre_KrylovCopyVector(b, x);
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
   hypre_KrylovCopyVector(b, r);
   hypre_KrylovMatvec(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   tmp_norm = sqrt(hypre_KrylovInnerProd(r,r));
   if (logging > 0)
   {
      norms[0] = tmp_norm;
   }

   /* p = C*r */
   hypre_KrylovClearVector(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> */
   gamma = hypre_KrylovInnerProd(r,p);

   if (bi_prod > 0.0)
       eps = (tol*tol)*bi_prod;
   else
      {if (two_norm)
          {eps = (tol*tol)*hypre_KrylovInnerProd(r,r);
           if (logging > 0 && my_id == 0)
              {printf("Exiting when ||r||_2 < tol * ||r0||_2\n");
               printf("Initial ||r0||_2: %e\n",norms[0]);};}
       else
          {eps = (tol*tol)*gamma;
           if (logging > 0 && my_id == 0)
              {printf("Exiting when ||r||_C < tol * ||r0||_C\n");
               printf("Initial ||r0||_C: %e\n",sqrt(gamma));};};};

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      hypre_KrylovMatvec(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / hypre_KrylovInnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_KrylovAxpy(alpha, p, x);

      /* r = r - alpha*s */
      hypre_KrylovAxpy(-alpha, s, r);
	 
      /* s = C*r */
      hypre_KrylovClearVector(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = hypre_KrylovInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
	 i_prod = hypre_KrylovInnerProd(r,r);
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
            pi_prod = hypre_KrylovInnerProd(p,p);
            xi_prod = hypre_KrylovInnerProd(x,x);
            if ((alpha*alpha*pi_prod/xi_prod) < tol*tol)
               break;
         }
         else
         {
            break;
         }
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      hypre_KrylovScaleVector(beta, p);   
      hypre_KrylovAxpy(1.0, s, p);
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

   if (logging > 0 && my_id == 0)
   {
      if (bi_prod > 0.0)
         {if (two_norm)
             {
              printf("\n\n");
              printf("Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
              printf("-----    ------------    ---------  ------------ \n");
             }
          else
             {
              printf("\n\n");
              printf("Iters       ||r||_C      conv.rate  ||r||_C/||b||_C\n");
              printf("-----    ------------    ---------  ------------ \n");
         }
         for (j = 1; j <= i; j++)
         {
            printf("% 5d    %e    %f   %e\n", j, norms[j], norms[j]/ 
		norms[j-1], rel_norms[j]);
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
            printf("% 5d    %e    %f   %e\n", j, norms[j], norms[j]/ 
		norms[j-1]);
         }
         printf("\n\n");};
      /* fclose(fp);}; */
   }

   (pcg_data -> num_iterations) = i;
   if (logging > 0)
      (pcg_data -> rel_residual_norm) = rel_norms[i];

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetTol
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetTol( void   *pcg_vdata,
                 double  tol       )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetMaxIter( void *pcg_vdata,
                     int   max_iter  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetTwoNorm
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetTwoNorm( void *pcg_vdata,
                     int   two_norm  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> two_norm) = two_norm;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetRelChange( void *pcg_vdata,
                       int   rel_change  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> rel_change) = rel_change;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetPrecond( void  *pcg_vdata,
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
 * hypre_KrylovSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_KrylovSetLogging( void *pcg_vdata,
                     int   logging)
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_KrylovGetNumIterations( void *pcg_vdata,
                           int  *num_iterations )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;

   *num_iterations = (pcg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_KrylovPrintLogging
 *--------------------------------------------------------------------------*/

int
hypre_KrylovPrintLogging( void *pcg_vdata,
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
 * hypre_KrylovGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_KrylovGetFinalRelativeResidualNorm( void   *pcg_vdata,
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

