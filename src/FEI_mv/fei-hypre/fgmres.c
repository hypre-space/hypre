/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * FGMRES - flexible gmres
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
 * hypre_FGMRESData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      stop_crit;
   int      k_dim;
   double   tol;
   double   rel_residual_norm;
   void     *A;
   void     *w;
   void     **p;
   void     **z;
   void     *r;
   void     *matvec_data;
   int     (*precond)();
   int     (*precond_setup)();
   void     *precond_data;
   int      num_iterations;
   int      logging;
   double  *norms;
   char    *log_file_name;
   int     precond_tol_update;
   int     (*precond_update_tol)();

} hypre_FGMRESData;

/*--------------------------------------------------------------------------
 * hypre_FGMRESCreate
 *--------------------------------------------------------------------------*/
 
void *hypre_FGMRESCreate()
{
   hypre_FGMRESData *fgmres_data;
 
   fgmres_data = hypre_CTAlloc(hypre_FGMRESData, 1);
 
   /* set defaults */

   (fgmres_data -> k_dim)              = 5;
   (fgmres_data -> tol)                = 1.0e-06;
   (fgmres_data -> max_iter)           = 1000;
   (fgmres_data -> stop_crit)          = 0; /* rel. residual norm */
   (fgmres_data -> precond)            = hypre_ParKrylovIdentity;
   (fgmres_data -> precond_setup)      = hypre_ParKrylovIdentitySetup;
   (fgmres_data -> precond_data)       = NULL;
   (fgmres_data -> logging)            = 0;
   (fgmres_data -> p)                  = NULL;
   (fgmres_data -> z)                  = NULL;
   (fgmres_data -> r)                  = NULL;
   (fgmres_data -> w)                  = NULL;
   (fgmres_data -> matvec_data)        = NULL;
   (fgmres_data -> norms)              = NULL;
   (fgmres_data -> log_file_name)      = NULL;
   (fgmres_data -> logging)            = 0;
   (fgmres_data -> precond_tol_update) = 0;
   (fgmres_data -> precond_update_tol) = NULL;
   return (void *) fgmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESDestroy( void *fgmres_vdata )
{
   int              i, ierr=0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   if (fgmres_data)
   {
      if ( (fgmres_data->logging) > 0 && (fgmres_data->norms != NULL) )
         hypre_TFree( fgmres_data -> norms );
      if ( (fgmres_data->matvec_data) != NULL )
         hypre_ParKrylovMatvecDestroy(fgmres_data -> matvec_data);
      if ( (fgmres_data-> r) != NULL )
         hypre_ParKrylovDestroyVector(fgmres_data -> r);
      if ( (fgmres_data-> w) != NULL )
         hypre_ParKrylovDestroyVector(fgmres_data -> w);
      if ( (fgmres_data-> p) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim+1); i++)
            hypre_ParKrylovDestroyVector((fgmres_data -> p)[i]);
         hypre_TFree( fgmres_data -> p );
      }
      if ( (fgmres_data-> z) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim+1); i++)
            hypre_ParKrylovDestroyVector((fgmres_data -> z)[i]);
         hypre_TFree( fgmres_data -> z );
      }
      hypre_TFree( fgmres_data );
   }
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetup
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetup( void *fgmres_vdata, void *A, void *b, void *x )
{
   hypre_FGMRESData *fgmres_data     = fgmres_vdata;
   int              k_dim            = (fgmres_data -> k_dim);
   int              max_iter         = (fgmres_data -> max_iter);
   int            (*precond_setup)() = (fgmres_data -> precond_setup);
   void            *precond_data     = (fgmres_data -> precond_data);
   int              ierr = 0;
 
   (fgmres_data -> A) = A;
 
   if ((fgmres_data -> r) == NULL)
      (fgmres_data -> r) = hypre_ParKrylovCreateVector(b);
   if ((fgmres_data -> w) == NULL)
      (fgmres_data -> w) = hypre_ParKrylovCreateVector(b);
   if ((fgmres_data -> p) == NULL)
      (fgmres_data -> p) = hypre_ParKrylovCreateVectorArray(k_dim+1,b);
   if ((fgmres_data -> z) == NULL)
      (fgmres_data -> z) = hypre_ParKrylovCreateVectorArray(k_dim+1,b);

   if ((fgmres_data -> matvec_data) == NULL)
      (fgmres_data -> matvec_data) = hypre_ParKrylovMatvecCreate(A, x);

   ierr = precond_setup(precond_data, A, b, x);
 
   if ((fgmres_data -> logging) > 0)
   {
      if ((fgmres_data -> norms) == NULL)
         (fgmres_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
      if ((fgmres_data -> log_file_name) == NULL)
         (fgmres_data -> log_file_name) = "fgmres.out.log";
   }
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_FGMRESSolve
 *-------------------------------------------------------------------------*/

int hypre_FGMRESSolve(void  *fgmres_vdata, void  *A, void  *b, void  *x)
{
   hypre_FGMRESData *fgmres_data  = fgmres_vdata;
   int 		     k_dim        = (fgmres_data -> k_dim);
   int 		     max_iter     = (fgmres_data -> max_iter);
   int 		     stop_crit    = (fgmres_data -> stop_crit);
   double 	     accuracy     = (fgmres_data -> tol);
   void             *matvec_data  = (fgmres_data -> matvec_data);

   void             *r            = (fgmres_data -> r);
   void            **p            = (fgmres_data -> p);
   void            **z            = (fgmres_data -> z);

   int 	           (*precond)()   = (fgmres_data -> precond);
   int 	            *precond_data = (fgmres_data -> precond_data);

   int             logging        = (fgmres_data -> logging);
   double         *norms          = (fgmres_data -> norms);
   
   int 	           tol_update     = (fgmres_data -> precond_tol_update);
   int 	           (*update_tol)()= (fgmres_data -> precond_update_tol);

   int	           i, j, k, ierr = 0, iter, my_id, num_procs;
   double          *rs, **hh, *c, *s, t;
   double          epsilon, gamma, r_norm, b_norm, epsmac = 1.e-16; 

   hypre_ParKrylovCommInfo(A,&my_id,&num_procs);

   /* initialize work arrays */

   if (logging > 0) norms = (fgmres_data -> norms);
   rs = hypre_CTAlloc(double, k_dim+1); 
   c  = hypre_CTAlloc(double, k_dim); 
   s  = hypre_CTAlloc(double, k_dim); 
   hh = hypre_CTAlloc(double*, k_dim+1); 
   for (i=0; i < k_dim+1; i++) hh[i] = hypre_CTAlloc(double, k_dim); 
   hypre_ParKrylovCopyVector(b,p[0]);

   /* compute initial residual */

   hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, p[0]);
   r_norm = sqrt(hypre_ParKrylovInnerProd(p[0],p[0]));
   b_norm = sqrt(hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("FGMRES : L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("FGMRES : Initial L2 norm of residual: %e\n", r_norm);
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

   if ( stop_crit ) epsilon = accuracy;

   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */

      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         ierr = 0;
         return ierr;
      }

      if (r_norm <= epsilon && iter > 0) 
      {
         hypre_ParKrylovCopyVector(b,r);
         hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
         r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
         if (r_norm <= epsilon)
         {
            if (logging > 0 && my_id == 0)
               printf("Final L2 norm of residual: %e\n\n", r_norm);
            break;
         }
      }

      t = 1.0 / r_norm;
      hypre_ParKrylovScaleVector(t,p[0]);
      i = 0;
      while (i < k_dim && r_norm > epsilon && iter < max_iter)
      {
         i++;
         iter++;
         hypre_ParKrylovClearVector(z[i-1]);

         if ( tol_update != 0 && update_tol != NULL ) 
            update_tol(precond_data,r_norm/b_norm);

         precond(precond_data, A, p[i-1], z[i-1]);
         hypre_ParKrylovMatvec(matvec_data, 1.0, A, z[i-1], 0.0, p[i]);

         /* modified Gram_Schmidt */

         for (j=0; j < i; j++)
         {
            hh[j][i-1] = hypre_ParKrylovInnerProd(p[j],p[i]);
            hypre_ParKrylovAxpy(-hh[j][i-1],p[j],p[i]);
         }
         t = sqrt(hypre_ParKrylovInnerProd(p[i],p[i]));
         hh[i][i-1] = t;	
         if (t != 0.0)
         {
            t = 1.0/t;
            hypre_ParKrylovScaleVector(t, p[i]);
         }

         /* done with modified Gram_schmidt. update factorization of hh */

         for (j = 1; j < i; j++)
         {
            t = hh[j-1][i-1];
            hh[j-1][i-1] = c[j-1]*t + s[j-1]*hh[j][i-1];		
            hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
         }
         gamma = sqrt(hh[i-1][i-1]*hh[i-1][i-1] + hh[i][i-1]*hh[i][i-1]);
         if (gamma == 0.0) gamma = epsmac;
         c[i-1] = hh[i-1][i-1]/gamma;
         s[i-1] = hh[i][i-1]/gamma;
         rs[i] = -s[i-1]*rs[i-1];
         rs[i-1] = c[i-1]*rs[i-1];

         /* determine residual norm */

         hh[i-1][i-1] = c[i-1]*hh[i-1][i-1] + s[i-1]*hh[i][i-1];
         r_norm = fabs(rs[i]);
         if (logging > 0)
         {
            norms[iter] = r_norm;
            if (my_id == 0)
               printf("FGMRES : iteration = %6d, norm of r = %e\n", iter,
                      r_norm);
         }
      }

      /* now compute solution, first solve upper triangular system */
	
      rs[i-1] = rs[i-1]/hh[i-1][i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = rs[k];
         for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];
         rs[k] = t/hh[k][k];
      }

	
      for (j = 0; j < i; j++) hypre_ParKrylovAxpy(rs[j], z[j], x);

      /* check for convergence, evaluate actual residual */

      hypre_ParKrylovCopyVector(b,p[0]);
      hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, p[0]);
      r_norm = sqrt(hypre_ParKrylovInnerProd(p[0],p[0]));
      if (r_norm <= epsilon) 
      {
         if (logging > 0 && my_id == 0)
            printf("FGMRES Final L2 norm of residual: %e\n\n", r_norm);
         break;
      }
   }

   (fgmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   hypre_TFree(c); 
   hypre_TFree(s); 
   hypre_TFree(rs);
 
   for (i=0; i < k_dim+1; i++) hypre_TFree(hh[i]);
   hypre_TFree(hh); 

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetKDim
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetKDim( void *fgmres_vdata, int k_dim )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> k_dim) = k_dim;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetTol( void *fgmres_vdata, double tol )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetMaxIter( void *fgmres_vdata, int max_iter )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetStopCrit( void *fgmres_vdata, double stop_crit )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetPrecond( void *fgmres_vdata, int (*precond)(),
                            int  (*precond_setup)(), void  *precond_data )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> precond)        = precond;
   (fgmres_data -> precond_setup)  = precond_setup;
   (fgmres_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_FGMRESGetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESGetPrecond(void *fgmres_vdata, HYPRE_Solver *precond_data_ptr)
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   *precond_data_ptr = (HYPRE_Solver)(fgmres_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_FGMRESSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESSetLogging( void *fgmres_vdata, int logging )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FGMRESGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESGetNumIterations( void *fgmres_vdata, int *num_iterations )
{
   int              ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   *num_iterations = (fgmres_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_FGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESGetFinalRelativeResidualNorm(void *fgmres_vdata,
                                             double *relative_residual_norm )
{
   int 		    ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   *relative_residual_norm = (fgmres_data -> rel_residual_norm);
   
   return ierr;
} 

/*--------------------------------------------------------------------------
 * hypre_FGMRESUpdatePrecondTolerance
 *--------------------------------------------------------------------------*/
 
int hypre_FGMRESUpdatePrecondTolerance(void *fgmres_vdata, int (*update_tol)())
{
   int 		    ierr = 0;
   hypre_FGMRESData *fgmres_data = fgmres_vdata;
 
   (fgmres_data -> precond_tol_update) = 1;
   (fgmres_data -> precond_update_tol) = update_tol;
   return ierr;
} 

