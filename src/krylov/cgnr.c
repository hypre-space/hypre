/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_CGNRFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_CGNRFunctions *
hypre_CGNRFunctionsCreate(
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecT)       ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*PrecondT)       ( void *vdata, void *A, void *b, void *x )
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

HYPRE_Int
hypre_CGNRDestroy( void *cgnr_vdata )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   HYPRE_Int ierr = 0;

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

HYPRE_Int
hypre_CGNRSetup(void *cgnr_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   HYPRE_Int            max_iter         = (cgnr_data -> max_iter);
   HYPRE_Int          (*precond_setup)() = (cgnr_functions -> precond_setup);
   void          *precond_data     = (cgnr_data -> precond_data);
   HYPRE_Int            ierr = 0;

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

   ierr = precond_setup(precond_data, A, b, x);

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

HYPRE_Int
hypre_CGNRSolve(void *cgnr_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_CGNRData  *cgnr_data   = cgnr_vdata;
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;

   double          tol          = (cgnr_data -> tol);
   HYPRE_Int             max_iter     = (cgnr_data -> max_iter);
   HYPRE_Int             stop_crit    = (cgnr_data -> stop_crit);
   void           *p            = (cgnr_data -> p);
   void           *q            = (cgnr_data -> q);
   void           *r            = (cgnr_data -> r);
   void           *t            = (cgnr_data -> t);
   void           *matvec_data  = (cgnr_data -> matvec_data);
   HYPRE_Int           (*precond)()   = (cgnr_functions -> precond);
   HYPRE_Int           (*precondT)()  = (cgnr_functions -> precondT);
   void           *precond_data = (cgnr_data -> precond_data);
   HYPRE_Int             logging      = (cgnr_data -> logging);
   double         *norms        = (cgnr_data -> norms);
                
   double          alpha, beta;
   double          gamma, gamma_old;
   double          bi_prod, i_prod, eps;
   double          ieee_check = 0.;
                
   HYPRE_Int             i = 0;
   HYPRE_Int             ierr = 0;
   HYPRE_Int             my_id, num_procs;
   HYPRE_Int             x_not_set = 1;
   /* char		  *log_file_name; */

   /*-----------------------------------------------------------------------
    * Start cgnr solve
    *-----------------------------------------------------------------------*/
   (*(cgnr_functions->CommInfo))(A,&my_id,&num_procs);
   if (logging > 1 && my_id == 0)
   {
/* not used yet      log_file_name = (cgnr_data -> log_file_name); */
      hypre_printf("Iters       ||r||_2      conv.rate  ||r||_2/||b||_2\n");
      hypre_printf("-----    ------------    ---------  ------------ \n");
   }


   /* compute eps */
   bi_prod = (*(cgnr_functions->InnerProd))(b, b);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) ieee_check = bi_prod/bi_prod; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0)
      {
        hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        hypre_printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied b.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

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

      /* Since it is does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (norms[0] != 0.) ieee_check = norms[0]/norms[0]; /* INF -> NaN conversion */
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (logging > 0)
         {
           hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
           hypre_printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
           hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
           hypre_printf("Returning error flag += 101.  Program not terminated.\n");
           hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         ierr += 101;
         return ierr;
      }
   }

   /* t = C^T*A^T*r */
   (*(cgnr_functions->MatvecT))(matvec_data, 1.0, A, r, 0.0, q);
   (*(cgnr_functions->ClearVector))(t);
   precondT(precond_data, A, q, t);

   /* p = r */
   (*(cgnr_functions->CopyVector))(r, p);

   /* gamma = <t,t> */
   gamma = (*(cgnr_functions->InnerProd))(t,t);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) ieee_check = gamma/gamma; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0)
      {
        hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        hypre_printf("ERROR -- hypre_CGNRSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

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
            hypre_printf("% 5d    %e    %f   %e\n", i, norms[i], norms[i]/ 
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
      hypre_printf("\n\n");
   }

   (cgnr_data -> num_iterations) = i;
   (cgnr_data -> rel_residual_norm) = norms[i]/bi_prod;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetTol(void   *cgnr_vdata,
                 double  tol       )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;
 
   (cgnr_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetMinIter( void *cgnr_vdata,
                     HYPRE_Int   min_iter  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;
 
   (cgnr_data -> min_iter) = min_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetMaxIter( void *cgnr_vdata,
                     HYPRE_Int   max_iter  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;
 
   (cgnr_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetStopCrit( void *cgnr_vdata,
                     HYPRE_Int   stop_crit  )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;
 
   (cgnr_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetPrecond(void  *cgnr_vdata,
                     HYPRE_Int  (*precond)(),
                     HYPRE_Int  (*precondT)(),
                     HYPRE_Int  (*precond_setup)(),
                     void  *precond_data )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   hypre_CGNRFunctions *cgnr_functions = cgnr_data->functions;
   HYPRE_Int            ierr = 0;
 
   (cgnr_functions -> precond)       = precond;
   (cgnr_functions -> precondT)      = precondT;
   (cgnr_functions -> precond_setup) = precond_setup;
   (cgnr_data -> precond_data)  = precond_data;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRGetPrecond( void         *cgnr_vdata,
                      HYPRE_Solver *precond_data_ptr )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int             ierr = 0;

   *precond_data_ptr = (HYPRE_Solver)(cgnr_data -> precond_data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRSetLogging( void *cgnr_vdata,
                     HYPRE_Int   logging)
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;
 
   (cgnr_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CGNRGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRGetNumIterations( void *cgnr_vdata,
                           HYPRE_Int  *num_iterations )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int            ierr = 0;

   *num_iterations = (cgnr_data -> num_iterations);

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_CGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CGNRGetFinalRelativeResidualNorm( void   *cgnr_vdata,
                                       double *relative_residual_norm )
{
   hypre_CGNRData *cgnr_data = cgnr_vdata;
   HYPRE_Int ierr = 0;

   *relative_residual_norm = (cgnr_data -> rel_residual_norm);
   
   return ierr;
}

