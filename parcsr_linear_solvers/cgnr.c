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
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_CGNRData
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   tol;
   double   rel_residual_norm;
   int      max_iter;

   void    *A;
   void    *p;
   void    *q;
   void    *r;
   void    *t;

   void    *matvec_data;

   int    (*precond)();
   int    (*precondT)();
   int    (*precond_setup)();
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} hypre_CGNRData;

/*--------------------------------------------------------------------------
 * hypre_CGNRInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_CGNRInitialize( )
{
   hypre_CGNRData *cgnr_data;

   cgnr_data = hypre_CTAlloc(hypre_CGNRData, 1);

   /* set defaults */
   (cgnr_data -> tol)          = 1.0e-06;
   (cgnr_data -> max_iter)     = 1000;
   (cgnr_data -> matvec_data)  = NULL;
   (cgnr_data -> precond)       = hypre_PCGIdentity;
   (cgnr_data -> precondT)      = hypre_PCGIdentity;
   (cgnr_data -> precond_setup) = hypre_PCGIdentitySetup;
   (cgnr_data -> precond_data)  = NULL;
   (cgnr_data -> logging)      = 0;
   (cgnr_data -> norms)        = NULL;

   return (void *) cgnr_data;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRFinalize
 *--------------------------------------------------------------------------*/

int
hypre_CGNRFinalize( void *cgnr_vdata )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int ierr = 0;

   if (cgnr_data)
   {
      if ((cgnr_data -> logging) > 0)
      {
         hypre_TFree(cgnr_data -> norms);
      }

      hypre_PCGMatvecFinalize(cgnr_data -> matvec_data);

      hypre_PCGFreeVector(cgnr_data -> p);
      hypre_PCGFreeVector(cgnr_data -> q);
      hypre_PCGFreeVector(cgnr_data -> r);
      hypre_PCGFreeVector(cgnr_data -> t);

      hypre_TFree(cgnr_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetup
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetup(void *cgnr_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            max_iter         = (cgnr_data -> max_iter);
   int          (*precond_setup)() = (cgnr_data -> precond_setup);
   void          *precond_data     = (cgnr_data -> precond_data);
   int            ierr = 0;

   (cgnr_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   (cgnr_data -> p) = hypre_PCGNewVector(x);
   (cgnr_data -> q) = hypre_PCGNewVector(x);
   (cgnr_data -> r) = hypre_PCGNewVector(b);
   (cgnr_data -> t) = hypre_PCGNewVector(b);

   (cgnr_data -> matvec_data) = hypre_PCGMatvecInitialize(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((cgnr_data -> logging) > 0)
   {
      (cgnr_data -> norms)     = hypre_CTAlloc(double, max_iter + 1);
      (cgnr_data -> log_file_name) = "cgnr.out.log";
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSolve: apply CG to (AC)^TACy = (AC)^Tb, x = Cy
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSolve(void *cgnr_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGNRData  *cgnr_data   = cgnr_vdata;

   double          tol          = (cgnr_data -> tol);
   int             max_iter     = (cgnr_data -> max_iter);
   void           *p            = (cgnr_data -> p);
   void           *q            = (cgnr_data -> q);
   void           *r            = (cgnr_data -> r);
   void           *t            = (cgnr_data -> t);
   void           *matvec_data  = (cgnr_data -> matvec_data);
   int           (*precond)()   = (cgnr_data -> precond);
   int           (*precondT)()  = (cgnr_data -> precondT);
   void           *precond_data = (cgnr_data -> precond_data);
   int             logging      = (cgnr_data -> logging);
   double         *norms        = (cgnr_data -> norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
                
   int             i = 0, j;
   int             ierr = 0;
   int             x_not_set = 1;
   char		  *log_file_name;
   FILE		  *fp;

   /*-----------------------------------------------------------------------
    * Start cgnr solve
    *-----------------------------------------------------------------------*/

   if (logging > 0)
   {
      log_file_name = (cgnr_data -> log_file_name);
      fp = fopen(log_file_name,"w");
   }

   /* compute eps */
   bi_prod = hypre_PCGInnerProd(b, b);
   eps = (tol*tol)*bi_prod;

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      hypre_PCGCopyVector(b, x);
      if (logging > 0)
      {
         norms[0]     = 0.0;
      }
      ierr = 0;
      return ierr;
   }

   /* r = b - Ax */
   hypre_PCGCopyVector(b, r);
   hypre_PCGMatvec(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   if (logging > 0)
   {
      norms[0] = sqrt(hypre_PCGInnerProd(r,r));
   }

   /* t = C^T*A^T*r */
   hypre_PCGMatvecT(matvec_data, 1.0, A, r, 0.0, q);
   hypre_PCGClearVector(t);
   precondT(precond_data, A, q, t);

   /* p = r */
   hypre_PCGCopyVector(r, p);

   /* gamma = <t,t> */
   gamma = hypre_PCGInnerProd(t,t);

   while ((i+1) <= max_iter)
   {
      i++;

      /* q = A*C*p */
      hypre_PCGClearVector(t);
      precond(precond_data, A, p, t);
      hypre_PCGMatvec(matvec_data, 1.0, A, t, 0.0, q);

      /* alpha = gamma / <q,q> */
      alpha = gamma / hypre_PCGInnerProd(q, q);

      gamma_old = gamma;

      /* x = x + alpha*p */
      hypre_PCGAxpy(alpha, p, x);

      /* r = r - alpha*q */
      hypre_PCGAxpy(-alpha, q, r);
	 
      /* t = C^T*A^T*r */
      hypre_PCGMatvecT(matvec_data, 1.0, A, r, 0.0, q);
      hypre_PCGClearVector(t);
      precondT(precond_data, A, q, t);

      /* gamma = <t,t> */
      gamma = hypre_PCGInnerProd(t, t);

      /* set i_prod for convergence test */
      i_prod = hypre_PCGInnerProd(r,r);

      /* log norm info */
      if (logging > 0)
      {
         norms[i]     = sqrt(i_prod);
      }

      /* check for convergence */
      if (i_prod < eps)
      {
         /*-----------------------------------------------------------------
          * Generate solution q = Cx
          *-----------------------------------------------------------------*/
         hypre_PCGClearVector(q);
         precond(precond_data, A, x, q);
         /* r = b - Aq */
         hypre_PCGCopyVector(b, r);
         hypre_PCGMatvec(matvec_data, -1.0, A, q, 1.0, r);
         i_prod = hypre_PCGInnerProd(r,r);
         if (i_prod < eps) 
         {
            hypre_PCGCopyVector(q,x);
	    x_not_set = 0;
	    break;
         }
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = t + beta p */
      hypre_PCGScaleVector(beta, p);   
      hypre_PCGAxpy(1.0, t, p);
   }

  /*-----------------------------------------------------------------
   * Generate solution x = Cx
   *-----------------------------------------------------------------*/
   if (x_not_set)
   {
      hypre_PCGCopyVector(x,q);
      hypre_PCGClearVector(x);
      precond(precond_data, A, q, x);
   }

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   bi_prod = sqrt(bi_prod);

   if (logging > 0)
   {
      fprintf(fp,"Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
      fprintf(fp,"-----    ------------    ---------  ------------ \n");
      for (j = 1; j <= i; j++)
      {
         fprintf(fp,"% 5d    %e    %f   %e\n", j, norms[j], norms[j]/ 
		norms[j-1], norms[j]/bi_prod);
      }
      fclose(fp);
   }

   (cgnr_data -> num_iterations) = i;
   (cgnr_data -> rel_residual_norm) = norms[i]/bi_prod;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetTol
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetTol(void   *cgnr_vdata,
                 double  tol       )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetMaxIter( void *cgnr_vdata,
                     int   max_iter  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetPrecond(void  *cgnr_vdata,
                     int  (*precond)(),
                     int  (*precondT)(),
                     int  (*precond_setup)(),
                     void  *precond_data )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> precond)       = precond;
   (cgnr_data -> precondT)      = precondT;
   (cgnr_data -> precond_setup) = precond_setup;
   (cgnr_data -> precond_data)  = precond_data;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetLogging
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetLogging( void *cgnr_vdata,
                     int   logging)
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_CGNRGetNumIterations( void *cgnr_vdata,
                           int  *num_iterations )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;

   *num_iterations = (cgnr_data -> num_iterations);

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_CGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
hypre_CGNRGetFinalRelativeResidualNorm( void   *cgnr_vdata,
                                       double *relative_residual_norm )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int ierr = 0;

   *relative_residual_norm = (cgnr_data -> rel_residual_norm);
   
   return ierr;
}

