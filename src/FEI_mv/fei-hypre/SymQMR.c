/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Symmetric QMR 
 *
 *****************************************************************************/

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SymQMRData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *q;
   void  *u;
   void  *d;
   void  *t;
   void  *rq;

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

} hypre_SymQMRData;

/*--------------------------------------------------------------------------
 * hypre_SymQMRCreate
 *--------------------------------------------------------------------------*/
 
void * hypre_SymQMRCreate( )
{
   hypre_SymQMRData *symqmr_data;
 
   symqmr_data = hypre_CTAlloc(hypre_SymQMRData, 1);
 
   /* set defaults */
   (symqmr_data -> tol)            = 1.0e-06;
   (symqmr_data -> max_iter)       = 1000;
   (symqmr_data -> stop_crit)      = 0; /* rel. residual norm */
   (symqmr_data -> precond)        = hypre_ParKrylovIdentity;
   (symqmr_data -> precond_setup)  = hypre_ParKrylovIdentitySetup;
   (symqmr_data -> precond_data)   = NULL;
   (symqmr_data -> logging)        = 0;
   (symqmr_data -> r)              = NULL;
   (symqmr_data -> q)              = NULL;
   (symqmr_data -> u)              = NULL;
   (symqmr_data -> d)              = NULL;
   (symqmr_data -> t)              = NULL;
   (symqmr_data -> rq)             = NULL;
   (symqmr_data -> matvec_data)    = NULL;
   (symqmr_data -> norms)          = NULL;
   (symqmr_data -> log_file_name)  = NULL;
 
   return (void *) symqmr_data;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRDestroy( void *symqmr_vdata )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int ierr = 0;
 
   if (symqmr_data)
   {
      if ((symqmr_data -> logging) > 0)
      {
         hypre_TFree(symqmr_data -> norms);
      }
 
      hypre_ParKrylovMatvecDestroy(symqmr_data -> matvec_data);
 
      hypre_ParKrylovDestroyVector(symqmr_data -> r);
      hypre_ParKrylovDestroyVector(symqmr_data -> q);
      hypre_ParKrylovDestroyVector(symqmr_data -> u);
      hypre_ParKrylovDestroyVector(symqmr_data -> d);
      hypre_ParKrylovDestroyVector(symqmr_data -> t);
      hypre_ParKrylovDestroyVector(symqmr_data -> rq);
 
      hypre_TFree(symqmr_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRSetup
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetup( void *symqmr_vdata, void *A, void *b, void *x         )
{
   hypre_SymQMRData *symqmr_data   = symqmr_vdata;
   int            max_iter         = (symqmr_data -> max_iter);
   int          (*precond_setup)() = (symqmr_data -> precond_setup);
   void          *precond_data     = (symqmr_data -> precond_data);
   int            ierr = 0;
 
   (symqmr_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((symqmr_data -> r) == NULL)
      (symqmr_data -> r) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> q) == NULL)
      (symqmr_data -> q) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> u) == NULL)
      (symqmr_data -> u) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> d) == NULL)
      (symqmr_data -> d) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> t) == NULL)
      (symqmr_data -> t) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> rq) == NULL)
      (symqmr_data -> rq) = hypre_ParKrylovCreateVector(b);
   if ((symqmr_data -> matvec_data) == NULL)
      (symqmr_data -> matvec_data) = hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((symqmr_data -> logging) > 0)
   {
      if ((symqmr_data -> norms) == NULL)
         (symqmr_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((symqmr_data -> log_file_name) == NULL)
         (symqmr_data -> log_file_name) = "symqmr.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_SymQMRSolve
 *-------------------------------------------------------------------------*/

int hypre_SymQMRSolve(void  *symqmr_vdata, void  *A, void  *b, void  *x)
{
   hypre_SymQMRData  *symqmr_data    = symqmr_vdata;
   int 		     max_iter      = (symqmr_data -> max_iter);
   int 		     stop_crit     = (symqmr_data -> stop_crit);
   double 	     accuracy      = (symqmr_data -> tol);
   void             *matvec_data   = (symqmr_data -> matvec_data);
 
   void             *r             = (symqmr_data -> r);
   void             *q             = (symqmr_data -> q);
   void             *u             = (symqmr_data -> u);
   void             *d             = (symqmr_data -> d);
   void             *t             = (symqmr_data -> t);
   void             *rq            = (symqmr_data -> rq);
   int 	           (*precond)()    = (symqmr_data -> precond);
   int 	            *precond_data  = (symqmr_data -> precond_data);

   /* logging variables */
   int               logging       = (symqmr_data -> logging);
   double           *norms         = (symqmr_data -> norms);
   
   int               ierr=0, my_id, num_procs, iter;
   double            theta, tau, rhom1, rho, dtmp, r_norm;
   double            thetam1, c, epsilon; 
   double            sigma, alpha, beta;

   hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (symqmr_data -> norms);
   }

   /* initialize work arrays */

   hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
         printf("SymQMR : Initial L2 norm of residual = %e\n", r_norm);
   }
   iter = 0;
   epsilon = accuracy * r_norm;

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   while ( iter < max_iter && r_norm > epsilon )
   {
      if ( my_id == 0 && iter > 0 && logging ) printf("SymQMR restart... \n");

      tau = r_norm;
      precond(precond_data, A, r, q);
      rho = hypre_ParKrylovInnerProd(r,q);
      theta = 0.0;
      hypre_ParKrylovClearVector(d);
      hypre_ParKrylovCopyVector(r,rq);

      while ( iter < max_iter && r_norm > epsilon )
      {
         iter++;

         hypre_ParKrylovMatvec(matvec_data,1.0,A,q,0.0,t);
         sigma = hypre_ParKrylovInnerProd(q,t);
         if ( sigma == 0.0 )
         {
            printf("SymQMR ERROR : sigma = 0.0\n");
            exit(1);
         }
         alpha = rho / sigma;
         dtmp = - alpha;
         hypre_ParKrylovAxpy(dtmp,t,r);
         thetam1 = theta;
         theta = sqrt(hypre_ParKrylovInnerProd(r,r)) / tau;
         c = 1.0 / sqrt(1.0 + theta * theta );
         tau = tau * theta * c;
         dtmp = c * c * thetam1 * thetam1;
         hypre_ParKrylovScaleVector(dtmp,d);
         dtmp = c * c * alpha;
         hypre_ParKrylovAxpy(dtmp,q,d);
         dtmp = 1.0;
         hypre_ParKrylovAxpy(dtmp,d,x);

         precond(precond_data, A, r, u);
         rhom1 = rho;
         rho = hypre_ParKrylovInnerProd(r,u);
         beta = rho / rhom1;
         hypre_ParKrylovScaleVector(beta,q);
         dtmp = 1.0;
         hypre_ParKrylovAxpy(dtmp,u,q);

         dtmp = 1.0 - c * c;
         hypre_ParKrylovScaleVector(dtmp,rq);
         dtmp = c * c;
         hypre_ParKrylovAxpy(dtmp,r,rq);
         r_norm = sqrt(hypre_ParKrylovInnerProd(rq,rq));
         norms[iter] = r_norm;

         if ( my_id == 0 && logging )
            printf(" SymQMR : iteration %4d - residual norm = %e \n", 
                   iter, r_norm);
      }

      /* compute true residual */

      hypre_ParKrylovCopyVector(b,r);
      hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
      r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
   }

   (symqmr_data -> num_iterations)    = iter;
   (symqmr_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetTol( void *symqmr_vdata, double tol )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int            ierr = 0;
 
   (symqmr_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetMaxIter( void *symqmr_vdata, int max_iter )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetStopCrit( void *symqmr_vdata, double stop_crit )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int            ierr = 0;
 
   (symqmr_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetPrecond( void  *symqmr_vdata, int  (*precond)(),
                       int  (*precond_setup)(), void  *precond_data )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> precond)        = precond;
   (symqmr_data -> precond_setup)  = precond_setup;
   (symqmr_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_SymQMRSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRSetLogging( void *symqmr_vdata, int logging)
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int              ierr = 0;
 
   (symqmr_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SymQMRGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRGetNumIterations(void *symqmr_vdata,int  *num_iterations)
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int              ierr = 0;
 
   *num_iterations = (symqmr_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_SymQMRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_SymQMRGetFinalRelativeResidualNorm( void   *symqmr_vdata,
                                         double *relative_residual_norm )
{
   hypre_SymQMRData *symqmr_data = symqmr_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (symqmr_data -> rel_residual_norm);
   
   return ierr;
} 

