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
 * Preconditioned conjugate gradient (Omin) functions
 *
 *****************************************************************************/

/* This was based on the pcg.c formerly in struct_ls, with
   changes (GetPrecond and stop_crit) for compatibility with the pcg.c
   in parcsr_ls and elsewhere.  Incompatibilities with the
   parcsr_ls version:
   - logging is different; no attempt has been made to be the same
   - treatment of b=0 in Ax=b is different: this returns x=0; the parcsr
   version iterates with a special stopping criterion
*/

#include "krylov.h"

/*--------------------------------------------------------------------------
 * hypre_PCGFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_PCGFunctions *
hypre_PCGFunctionsCreate(
   char * (*CAlloc)        ( int count, int elt_size ),
   int    (*Free)          ( char *ptr ),
   void * (*CreateVector)  ( void *vector ),
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
   hypre_PCGFunctions * pcg_functions;
   pcg_functions = (hypre_PCGFunctions *)
      CAlloc( 1, sizeof(hypre_PCGFunctions) );

   pcg_functions->CAlloc = CAlloc;
   pcg_functions->Free = Free;
   pcg_functions->CreateVector = CreateVector;
   pcg_functions->DestroyVector = DestroyVector;
   pcg_functions->MatvecCreate = MatvecCreate;
   pcg_functions->Matvec = Matvec;
   pcg_functions->MatvecDestroy = MatvecDestroy;
   pcg_functions->InnerProd = InnerProd;
   pcg_functions->CopyVector = CopyVector;
   pcg_functions->ClearVector = ClearVector;
   pcg_functions->ScaleVector = ScaleVector;
   pcg_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   pcg_functions->precond_setup = PrecondSetup;
   pcg_functions->precond       = Precond;

   return pcg_functions;
}

/*--------------------------------------------------------------------------
 * hypre_PCGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PCGCreate( hypre_PCGFunctions *pcg_functions )
{
   hypre_PCGData *pcg_data;

   pcg_data = hypre_CTAllocF(hypre_PCGData, 1, pcg_functions);

   pcg_data -> functions = pcg_functions;

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> cf_tol)      = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> stop_crit)    = 0;
   (pcg_data -> matvec_data)  = NULL;
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
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;
   int ierr = 0;

   if (pcg_data)
   {
      if ((pcg_data -> logging) > 0)
      {
         hypre_TFreeF( pcg_data -> norms, pcg_functions );
         hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
      }

      (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);

      (*(pcg_functions->DestroyVector))(pcg_data -> p);
      (*(pcg_functions->DestroyVector))(pcg_data -> s);
      (*(pcg_functions->DestroyVector))(pcg_data -> r);

      hypre_TFreeF( pcg_data, pcg_functions );
      hypre_TFreeF( pcg_functions, pcg_functions );
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
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;
   int            max_iter         = (pcg_data -> max_iter);
   int          (*precond_setup)() = (pcg_functions -> precond_setup);
   void          *precond_data     = (pcg_data -> precond_data);
   int            ierr = 0;

   (pcg_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   (pcg_data -> p) = (*(pcg_functions->CreateVector))(x);
   (pcg_data -> s) = (*(pcg_functions->CreateVector))(x);
   (pcg_data -> r) = (*(pcg_functions->CreateVector))(b);

   (pcg_data -> matvec_data) = (*(pcg_functions->MatvecCreate))(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((pcg_data -> logging) > 0)
   {
      (pcg_data -> norms)     = hypre_CTAllocF( double, max_iter + 1,
                                               pcg_functions);
      (pcg_data -> rel_norms) = hypre_CTAllocF( double, max_iter + 1,
                                               pcg_functions );
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
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;

   double          tol          = (pcg_data -> tol);
   double          cf_tol       = (pcg_data -> cf_tol);
   int             max_iter     = (pcg_data -> max_iter);
   int             two_norm     = (pcg_data -> two_norm);
   int             rel_change   = (pcg_data -> rel_change);
   int             stop_crit    = (pcg_data -> stop_crit);
   void           *p            = (pcg_data -> p);
   void           *s            = (pcg_data -> s);
   void           *r            = (pcg_data -> r);
   void           *matvec_data  = (pcg_data -> matvec_data);
   int           (*precond)()   = (pcg_functions -> precond);
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
      bi_prod = (*(pcg_functions->InnerProd))(b, b);
   }
   else
   {
      /* bi_prod = <C*b,b> */
      (*(pcg_functions->ClearVector))(p);
      precond(precond_data, A, b, p);
      bi_prod = (*(pcg_functions->InnerProd))(p, b);
   };

   eps = tol*tol;
   if ( bi_prod > 0.0 ) {
      if ( stop_crit && !rel_change ) {  /* absolute tolerance */
         eps = eps / bi_prod;
      }
   }
   else    /* bi_prod==0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      (*(pcg_functions->CopyVector))(b, x);
      if (logging > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      ierr = 0;
      return ierr;
   };

   /* r = b - Ax */
   (*(pcg_functions->CopyVector))(b, r);
   (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   if (logging > 0 || cf_tol > 0.0)
   {
      i_prod_0   = (*(pcg_functions->InnerProd))(r,r);
      if (logging > 0) norms[0] = sqrt(i_prod_0);
   }

   /* p = C*r */
   (*(pcg_functions->ClearVector))(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> */
   gamma = (*(pcg_functions->InnerProd))(r,p);

   while ((i+1) <= max_iter)
   {
      i++;

      /* s = A*p */
      (*(pcg_functions->Matvec))(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / (*(pcg_functions->InnerProd))(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      (*(pcg_functions->Axpy))(alpha, p, x);

      /* r = r - alpha*s */
      (*(pcg_functions->Axpy))(-alpha, s, r);
         
      /* s = C*r */
      (*(pcg_functions->ClearVector))(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = (*(pcg_functions->InnerProd))(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
         i_prod = (*(pcg_functions->InnerProd))(r,r);
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
            pi_prod = (*(pcg_functions->InnerProd))(p,p);
            xi_prod = (*(pcg_functions->InnerProd))(x,x);
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
      (*(pcg_functions->ScaleVector))(beta, p);   
      (*(pcg_functions->Axpy))(1.0, s, p);
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
 * hypre_PCGSetStopCrit
 *--------------------------------------------------------------------------*/

int
hypre_PCGSetStopCrit( void *pcg_vdata,
                       int   stop_crit  )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;
 
   (pcg_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_PCGGetPrecond( void         *pcg_vdata,
                     HYPRE_Solver *precond_data_ptr )
{
   hypre_PCGData *pcg_data = pcg_vdata;
   int            ierr = 0;

   *precond_data_ptr = (HYPRE_Solver)(pcg_data -> precond_data);

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
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;
   int            ierr = 0;
 
   (pcg_functions -> precond)       = precond;
   (pcg_functions -> precond_setup) = precond_setup;
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

