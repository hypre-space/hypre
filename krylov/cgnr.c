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
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#include "krylov.h"
#include "utilities.h"

/*--------------------------------------------------------------------------
 * hypre_CGNRFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_CGNRFunctions *
hypre_CGNRFunctionsCreate(
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecT)       ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
   int    (*PrecondT)       ( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_CGNRFunctions * cgnr_functions;
   cgnr_functions = (hypre_CGNRFunctions *)
      hypre_CTAlloc( hypre_CGNRFunctions, 1 );

   cgnr_functions->CommInfo = CommInfo;
   cgnr_functions->CreateVector = CreateVector;
   cgnr_functions->DestroyVector = DestroyVector;
   cgnr_functions->MatvecCreate = MatvecCreate;
   cgnr_functions->Matvec = Matvec;
   cgnr_functions->MatvecT = MatvecT;
   cgnr_functions->MatvecDestroy = MatvecDestroy;
   cgnr_functions->InnerProd = InnerProd;
   cgnr_functions->CopyVector = CopyVector;
   cgnr_functions->ClearVector = ClearVector;
   cgnr_functions->ScaleVector = ScaleVector;
   cgnr_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   cgnr_functions->precond_setup = PrecondSetup;
   cgnr_functions->precond       = Precond;
   cgnr_functions->precondT       = Precond;

   return cgnr_functions;
}


/*--------------------------------------------------------------------------
 * hypre_CGNRCreate
 *--------------------------------------------------------------------------*/

void *
hypre_CGNRCreate( hypre_CGNRFunctions *cgnr_functions )
{
   hypre_CGNRData *cgnr_data;

   cgnr_data = hypre_CTAlloc( hypre_CGNRData, 1);
   cgnr_data->functions = cgnr_functions;

   /* set defaults */
   (cgnr_data -> tol)          = 1.0e-06;
   (cgnr_data -> min_iter)     = 0;
   (cgnr_data -> max_iter)     = 1000;
   (cgnr_data -> stop_crit)    = 0;
   (cgnr_data -> matvec_data)  = NULL;
   (cgnr_data -> precond_data)  = NULL;
   (cgnr_data -> logging)      = 0;
   (cgnr_data -> norms)        = NULL;

   return (void *) cgnr_data;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRDestroy
 *--------------------------------------------------------------------------*/

int
hypre_CGNRDestroy( void *cgnr_vdata )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   int ierr = 0;

   if (cgnr_data)
   {
      if ((cgnr_data -> logging) > 0)
      {
         hypre_TFree(cgnr_data -> norms);
      }

      (*(cgnr_functions->MatvecDestroy))(cgnr_data -> matvec_data);

      (*(cgnr_functions->DestroyVector))(cgnr_data -> p);
      (*(cgnr_functions->DestroyVector))(cgnr_data -> q);
      (*(cgnr_functions->DestroyVector))(cgnr_data -> r);
      (*(cgnr_functions->DestroyVector))(cgnr_data -> t);

      hypre_TFree(cgnr_data);
      hypre_TFree(cgnr_functions);
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
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   int            max_iter         = (cgnr_data -> max_iter);
   int          (*precond_setup)() = (cgnr_functions -> precond_setup);
   void          *precond_data     = (cgnr_data -> precond_data);
   int            ierr = 0;

   (cgnr_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   (cgnr_data -> p) = (*(cgnr_functions->CreateVector))(x);
   (cgnr_data -> q) = (*(cgnr_functions->CreateVector))(x);
   (cgnr_data -> r) = (*(cgnr_functions->CreateVector))(b);
   (cgnr_data -> t) = (*(cgnr_functions->CreateVector))(b);

   (cgnr_data -> matvec_data) = (*(cgnr_functions->MatvecCreate))(A, x);

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
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   double          tol          = (cgnr_data -> tol);
   int             max_iter     = (cgnr_data -> max_iter);
   int             stop_crit    = (cgnr_data -> stop_crit);
   void           *p            = (cgnr_data -> p);
   void           *q            = (cgnr_data -> q);
   void           *r            = (cgnr_data -> r);
   void           *t            = (cgnr_data -> t);
   void           *matvec_data  = (cgnr_data -> matvec_data);
   int           (*precond)()   = (cgnr_functions -> precond);
   int           (*precondT)()  = (cgnr_functions -> precondT);
   void           *precond_data = (cgnr_data -> precond_data);
   int             logging      = (cgnr_data -> logging);
   double         *norms        = (cgnr_data -> norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
                
   int             i = 0;
   int             ierr = 0;
   int             my_id, num_procs;
   int             x_not_set = 1;
   char		  *log_file_name;

   /*-----------------------------------------------------------------------
    * Start cgnr solve
    *-----------------------------------------------------------------------*/
   (*(cgnr_functions->CommInfo))(A,&my_id,&num_procs);
   if (logging > 1 && my_id == 0)
   {
/* not used yet      log_file_name = (cgnr_data -> log_file_name); */
      printf("Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
      printf("-----    ------------    ---------  ------------ \n");
   }


   /* compute eps */
   bi_prod = (*(cgnr_functions->InnerProd))(b, b);
   if (stop_crit) 
      eps = tol*tol; /* absolute residual norm */
   else
      eps = (tol*tol)*bi_prod; /* relative residual norm */

   /* Check to see if the rhs vector b is zero */
   if (bi_prod == 0.0)
   {
      /* Set x equal to zero and return */
      (*(cgnr_functions->CopyVector))(b, x);
      if (logging > 0)
      {
         norms[0]     = 0.0;
      }
      ierr = 0;
      return ierr;
   }

   /* r = b - Ax */
   (*(cgnr_functions->CopyVector))(b, r);
   (*(cgnr_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
 
   /* Set initial residual norm */
   if (logging > 0)
   {
      norms[0] = sqrt((*(cgnr_functions->InnerProd))(r,r));
   }

   /* t = C^T*A^T*r */
   (*(cgnr_functions->MatvecT))(matvec_data, 1.0, A, r, 0.0, q);
   (*(cgnr_functions->ClearVector))(t);
   precondT(precond_data, A, q, t);

   /* p = r */
   (*(cgnr_functions->CopyVector))(r, p);

   /* gamma = <t,t> */
   gamma = (*(cgnr_functions->InnerProd))(t,t);

   while ((i+1) <= max_iter)
   {
      i++;

      /* q = A*C*p */
      (*(cgnr_functions->ClearVector))(t);
      precond(precond_data, A, p, t);
      (*(cgnr_functions->Matvec))(matvec_data, 1.0, A, t, 0.0, q);

      /* alpha = gamma / <q,q> */
      alpha = gamma / (*(cgnr_functions->InnerProd))(q, q);

      gamma_old = gamma;

      /* x = x + alpha*p */
      (*(cgnr_functions->Axpy))(alpha, p, x);

      /* r = r - alpha*q */
      (*(cgnr_functions->Axpy))(-alpha, q, r);
	 
      /* t = C^T*A^T*r */
      (*(cgnr_functions->MatvecT))(matvec_data, 1.0, A, r, 0.0, q);
      (*(cgnr_functions->ClearVector))(t);
      precondT(precond_data, A, q, t);

      /* gamma = <t,t> */
      gamma = (*(cgnr_functions->InnerProd))(t, t);

      /* set i_prod for convergence test */
      i_prod = (*(cgnr_functions->InnerProd))(r,r);

      /* log norm info */
      if (logging > 0)
      {
         norms[i]     = sqrt(i_prod);
         if (logging > 1 && my_id == 0)
         {
            printf("% 5d    %e    %f   %e\n", i, norms[i], norms[i]/ 
		norms[i-1], norms[i]/bi_prod);
         }
      }

      /* check for convergence */
      if (i_prod < eps)
      {
         /*-----------------------------------------------------------------
          * Generate solution q = Cx
          *-----------------------------------------------------------------*/
         (*(cgnr_functions->ClearVector))(q);
         precond(precond_data, A, x, q);
         /* r = b - Aq */
         (*(cgnr_functions->CopyVector))(b, r);
         (*(cgnr_functions->Matvec))(matvec_data, -1.0, A, q, 1.0, r);
         i_prod = (*(cgnr_functions->InnerProd))(r,r);
         if (i_prod < eps) 
         {
            (*(cgnr_functions->CopyVector))(q,x);
	    x_not_set = 0;
	    break;
         }
      }

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = t + beta p */
      (*(cgnr_functions->ScaleVector))(beta, p);   
      (*(cgnr_functions->Axpy))(1.0, t, p);
   }

  /*-----------------------------------------------------------------
   * Generate solution x = Cx
   *-----------------------------------------------------------------*/
   if (x_not_set)
   {
      (*(cgnr_functions->CopyVector))(x,q);
      (*(cgnr_functions->ClearVector))(x);
      precond(precond_data, A, q, x);
   }

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   bi_prod = sqrt(bi_prod);

   if (logging > 1 && my_id == 0)
   {
      printf("\n\n");
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
 * hypre_CGNRSetMinIter
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetMinIter( void *cgnr_vdata,
                     int   min_iter  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> min_iter) = min_iter;
 
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
 * hypre_CGNRSetStopCrit
 *--------------------------------------------------------------------------*/

int
hypre_CGNRSetStopCrit( void *cgnr_vdata,
                     int   stop_crit  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int            ierr = 0;
 
   (cgnr_data -> stop_crit) = stop_crit;
 
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
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;
   int            ierr = 0;
 
   (cgnr_functions -> precond)       = precond;
   (cgnr_functions -> precondT)      = precondT;
   (cgnr_functions -> precond_setup) = precond_setup;
   (cgnr_data -> precond_data)  = precond_data;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRGetPrecond
 *--------------------------------------------------------------------------*/

int
hypre_CGNRGetPrecond( void         *cgnr_vdata,
                      HYPRE_Solver *precond_data_ptr )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   int             ierr = 0;

   *precond_data_ptr = (HYPRE_Solver)(cgnr_data -> precond_data);

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

