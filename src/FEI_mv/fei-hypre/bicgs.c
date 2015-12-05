/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * BiCGS 
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
 * hypre_BiCGSData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *p;
   void  *v;
   void  *q;
   void  *rh;
   void  *u;
   void  *t1;
   void  *t2;

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

} hypre_BiCGSData;

/*--------------------------------------------------------------------------
 * hypre_BiCGSCreate
 *--------------------------------------------------------------------------*/
 
void * hypre_BiCGSCreate( )
{
   hypre_BiCGSData *bicgs_data;
 
   bicgs_data = hypre_CTAlloc(hypre_BiCGSData, 1);
 
   /* set defaults */
   (bicgs_data -> tol)            = 1.0e-06;
   (bicgs_data -> max_iter)       = 1000;
   (bicgs_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgs_data -> precond)        = hypre_ParKrylovIdentity;
   (bicgs_data -> precond_setup)  = hypre_ParKrylovIdentitySetup;
   (bicgs_data -> precond_data)   = NULL;
   (bicgs_data -> logging)        = 0;
   (bicgs_data -> r)              = NULL;
   (bicgs_data -> rh)             = NULL;
   (bicgs_data -> p)              = NULL;
   (bicgs_data -> v)              = NULL;
   (bicgs_data -> q)              = NULL;
   (bicgs_data -> u)              = NULL;
   (bicgs_data -> t1)             = NULL;
   (bicgs_data -> t2)             = NULL;
   (bicgs_data -> matvec_data)    = NULL;
   (bicgs_data -> norms)          = NULL;
   (bicgs_data -> log_file_name)  = NULL;
 
   return (void *) bicgs_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSDestroy( void *bicgs_vdata )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int ierr = 0;
 
   if (bicgs_data)
   {
      if ((bicgs_data -> logging) > 0)
      {
         hypre_TFree(bicgs_data -> norms);
      }
 
      hypre_ParKrylovMatvecDestroy(bicgs_data -> matvec_data);
 
      hypre_ParKrylovDestroyVector(bicgs_data -> r);
      hypre_ParKrylovDestroyVector(bicgs_data -> rh);
      hypre_ParKrylovDestroyVector(bicgs_data -> v);
      hypre_ParKrylovDestroyVector(bicgs_data -> p);
      hypre_ParKrylovDestroyVector(bicgs_data -> q);
      hypre_ParKrylovDestroyVector(bicgs_data -> u);
      hypre_ParKrylovDestroyVector(bicgs_data -> t1);
      hypre_ParKrylovDestroyVector(bicgs_data -> t2);
 
      hypre_TFree(bicgs_data);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSSetup
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetup( void *bicgs_vdata, void *A, void *b, void *x         )
{
   hypre_BiCGSData *bicgs_data     = bicgs_vdata;
   int            max_iter         = (bicgs_data -> max_iter);
   int          (*precond_setup)() = (bicgs_data -> precond_setup);
   void          *precond_data     = (bicgs_data -> precond_data);
   int            ierr = 0;
 
   (bicgs_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgs_data -> r) == NULL)
      (bicgs_data -> r) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> rh) == NULL)
      (bicgs_data -> rh) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> v) == NULL)
      (bicgs_data -> v) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> p) == NULL)
      (bicgs_data -> p) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> q) == NULL)
      (bicgs_data -> q) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> u) == NULL)
      (bicgs_data -> u) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> t1) == NULL)
      (bicgs_data -> t1) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> t2) == NULL)
      (bicgs_data -> t2) = hypre_ParKrylovCreateVector(b);
   if ((bicgs_data -> matvec_data) == NULL)
      (bicgs_data -> matvec_data) = hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgs_data -> logging) > 0)
   {
      if ((bicgs_data -> norms) == NULL)
         (bicgs_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((bicgs_data -> log_file_name) == NULL)
         (bicgs_data -> log_file_name) = "bicgs.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSSolve
 *-------------------------------------------------------------------------*/

int hypre_BiCGSSolve(void  *bicgs_vdata, void  *A, void  *b, void  *x)
{
   hypre_BiCGSData  *bicgs_data    = bicgs_vdata;
   int 		     max_iter      = (bicgs_data -> max_iter);
   int 		     stop_crit     = (bicgs_data -> stop_crit);
   double 	     accuracy      = (bicgs_data -> tol);
   void             *matvec_data   = (bicgs_data -> matvec_data);
 
   void             *r             = (bicgs_data -> r);
   void             *rh            = (bicgs_data -> rh);
   void             *v             = (bicgs_data -> v);
   void             *p             = (bicgs_data -> p);
   void             *q             = (bicgs_data -> q);
   void             *u             = (bicgs_data -> u);
   void             *t1            = (bicgs_data -> t1);
   void             *t2            = (bicgs_data -> t2);
   int 	           (*precond)()    = (bicgs_data -> precond);
   int 	            *precond_data  = (bicgs_data -> precond_data);

   /* logging variables */
   int               logging       = (bicgs_data -> logging);
   double           *norms         = (bicgs_data -> norms);
   
   int               ierr, my_id, num_procs, iter;
   double            rho1, rho2, sigma, alpha, dtmp, r_norm, b_norm;
   double            beta, epsilon; 

   hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgs_data -> norms);
   }

   /* initialize work arrays */

   hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("BiCGS : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGS : Initial L2 norm of residual = %e\n", r_norm);
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
   if (stop_crit) epsilon = accuracy;

   hypre_ParKrylovCopyVector(r,rh);
   hypre_ParKrylovClearVector(p);
   hypre_ParKrylovClearVector(q);
   rho2 = r_norm * r_norm;
   beta = rho2;

   while ( iter < max_iter && r_norm > epsilon )
   {
      iter++;

      rho1 = rho2;
      hypre_ParKrylovCopyVector(r,u);
      hypre_ParKrylovAxpy(beta,q,u);

      hypre_ParKrylovCopyVector(q,t1);
      hypre_ParKrylovAxpy(beta,p,t1);
      hypre_ParKrylovCopyVector(u,p);
      hypre_ParKrylovAxpy(beta,t1,p);

      precond(precond_data, A, p, t1);
      hypre_ParKrylovMatvec(matvec_data,1.0,A,t1,0.0,v);

      sigma = hypre_ParKrylovInnerProd(rh,v);
      alpha = rho1 / sigma;

      hypre_ParKrylovCopyVector(u,q);
      dtmp = - alpha;
      hypre_ParKrylovAxpy(dtmp,v,q);

      dtmp = 1.0;
      hypre_ParKrylovAxpy(dtmp,q,u);

      precond(precond_data, A, u, t1);
      hypre_ParKrylovAxpy(alpha,t1,x);

      hypre_ParKrylovMatvec(matvec_data,1.0,A,t1,0.0,t2);

      dtmp = - alpha;
      hypre_ParKrylovAxpy(dtmp,t2,r);

      rho2 = hypre_ParKrylovInnerProd(r,rh);
      beta = rho2 / rho1;

      r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));

      if ( my_id == 0 && logging )
         printf(" BiCGS : iter %4d - res. norm = %e \n", iter, r_norm);
   }

   (bicgs_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgs_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgs_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetTol( void *bicgs_vdata, double tol )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int            ierr = 0;
 
   (bicgs_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetMaxIter( void *bicgs_vdata, int max_iter )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetStopCrit( void *bicgs_vdata, double stop_crit )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int            ierr = 0;
 
   (bicgs_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetPrecond( void  *bicgs_vdata, int  (*precond)(),
                       int  (*precond_setup)(), void  *precond_data )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> precond)        = precond;
   (bicgs_data -> precond_setup)  = precond_setup;
   (bicgs_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSSetLogging( void *bicgs_vdata, int logging)
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int              ierr = 0;
 
   (bicgs_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSGetNumIterations(void *bicgs_vdata,int  *num_iterations)
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgs_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_BiCGSGetFinalRelativeResidualNorm( void   *bicgs_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSData *bicgs_data = bicgs_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgs_data -> rel_residual_norm);
   
   return ierr;
} 

