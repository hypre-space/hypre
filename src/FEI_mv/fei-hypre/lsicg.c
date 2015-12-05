/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * LSICG 
 *
 *****************************************************************************/

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "seq_mv/seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_LSICGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    max_iter;
   int    stop_crit;
   double tol;
   double rel_residual_norm;

   void   *A;
   void   *r;
   void   *ap;
   void   *p;
   void   *z;

   void   *matvec_data;

   int    (*precond)();
   int    (*precond_setup)();
   void   *precond_data;

   int     num_iterations;
 
   int     logging;

} hypre_LSICGData;

/*--------------------------------------------------------------------------
 * hypre_LSICGCreate
 *--------------------------------------------------------------------------*/
 
void *hypre_LSICGCreate( )
{
   hypre_LSICGData *lsicg_data;
 
   lsicg_data = hypre_CTAlloc(hypre_LSICGData, 1);
 
   /* set defaults */
   (lsicg_data -> tol)            = 1.0e-06;
   (lsicg_data -> max_iter)       = 1000;
   (lsicg_data -> stop_crit)      = 0; /* rel. residual norm */
   (lsicg_data -> precond)        = hypre_ParKrylovIdentity;
   (lsicg_data -> precond_setup)  = hypre_ParKrylovIdentitySetup;
   (lsicg_data -> precond_data)   = NULL;
   (lsicg_data -> logging)        = 0;
   (lsicg_data -> r)              = NULL;
   (lsicg_data -> p)              = NULL;
   (lsicg_data -> ap)             = NULL;
   (lsicg_data -> z)              = NULL;
   (lsicg_data -> matvec_data)    = NULL;
 
   return (void *) lsicg_data;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGDestroy
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGDestroy( void *lsicg_vdata )
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   int             ierr = 0;
 
   if (lsicg_data)
   {
      hypre_ParKrylovMatvecDestroy(lsicg_data -> matvec_data);
      hypre_ParKrylovDestroyVector(lsicg_data -> r);
      hypre_ParKrylovDestroyVector(lsicg_data -> p);
      hypre_ParKrylovDestroyVector(lsicg_data -> ap);
      hypre_ParKrylovDestroyVector(lsicg_data -> z);
      hypre_TFree(lsicg_data);
   }
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_LSICGSetup
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetup( void *lsicg_vdata, void *A, void *b, void *x         )
{
   hypre_LSICGData *lsicg_data       = lsicg_vdata;
   int            (*precond_setup)() = (lsicg_data -> precond_setup);
   void           *precond_data      = (lsicg_data -> precond_data);
   int            ierr = 0;
 
   (lsicg_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((lsicg_data -> r) == NULL)
      (lsicg_data -> r) = hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> p) == NULL)
      (lsicg_data -> p) = hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> z) == NULL)
      (lsicg_data -> z) = hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> ap) == NULL)
      (lsicg_data -> ap) = hypre_ParKrylovCreateVector(b);
   if ((lsicg_data -> matvec_data) == NULL)
      (lsicg_data -> matvec_data) = hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_LSICGSolve
 *-------------------------------------------------------------------------*/

int hypre_LSICGSolve(void  *lsicg_vdata, void  *A, void  *b, void  *x)
{
   int               ierr=0, mypid, nprocs, iter, converged=0;
   double            rhom1, rho, r_norm, b_norm, epsilon;
   double            sigma, alpha, beta, dArray[2], dArray2[2];
   hypre_Vector     *r_local, *z_local;
   MPI_Comm          comm;

   hypre_LSICGData  *lsicg_data    = lsicg_vdata;
   int 		     max_iter      = (lsicg_data -> max_iter);
   int 		     stop_crit     = (lsicg_data -> stop_crit);
   double 	     accuracy      = (lsicg_data -> tol);
   void             *matvec_data   = (lsicg_data -> matvec_data);
   void             *r             = (lsicg_data -> r);
   void             *p             = (lsicg_data -> p);
   void             *z             = (lsicg_data -> z);
   void             *ap            = (lsicg_data -> ap);
   int 	           (*precond)()    = (lsicg_data -> precond);
   int 	            *precond_data  = (lsicg_data -> precond_data);
   int               logging       = (lsicg_data -> logging);

   /* compute initial residual */

   r_local = hypre_ParVectorLocalVector((hypre_ParVector *) r);
   z_local = hypre_ParVectorLocalVector((hypre_ParVector *) z);
   comm    = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *) A);
   hypre_ParKrylovCommInfo(A,&mypid,&nprocs);
   hypre_ParKrylovCopyVector(b,r);
   hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      if (mypid == 0)
      {
  	 printf("LSICG : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("LSICG : Initial L2 norm of residual = %e\n", r_norm);
      }
   }

   /* set convergence criterion */

   if (b_norm > 0.0) epsilon = accuracy * b_norm;
   else              epsilon = accuracy * r_norm;
   if ( stop_crit )  epsilon = accuracy;

   iter = 0;
   hypre_ParKrylovClearVector(p);

   while ( converged == 0 )
   {
      while ( r_norm > epsilon && iter < max_iter )
      {
         iter++;
         if ( iter == 1 )
         {
            precond(precond_data, A, r, z);
            rhom1 = rho;
            rho   = hypre_ParKrylovInnerProd(r,z);
            beta = 0.0;
         }
         else beta = rho / rhom1;
         hypre_ParKrylovScaleVector( beta, p );
         hypre_ParKrylovAxpy(1.0e0, z, p);
         hypre_ParKrylovMatvec(matvec_data,1.0e0,A,p,0.0,ap);
         sigma = hypre_ParKrylovInnerProd(p,ap);
         alpha  = rho / sigma;
         if ( sigma == 0.0 )
         {
            printf("HYPRE::LSICG ERROR - sigma = 0.0.\n");
            ierr = 2;
            return ierr;
         }
         hypre_ParKrylovAxpy(alpha, p, x);
         hypre_ParKrylovAxpy(-alpha, ap, r);
         dArray[0] = hypre_SeqVectorInnerProd( r_local, r_local );
         precond(precond_data, A, r, z);
         rhom1 = rho;
         dArray[1] = hypre_SeqVectorInnerProd( r_local, z_local );
         MPI_Allreduce(dArray, dArray2, 2, MPI_DOUBLE, MPI_SUM, comm);
         rho = dArray2[1];
         r_norm = sqrt( dArray2[0] );
         if ( iter % 1 == 0 && mypid == 0 )
            printf("LSICG : iteration %d - residual norm = %e (%e)\n",
                   iter, r_norm, epsilon);
      }
      hypre_ParKrylovCopyVector(b,r);
      hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
      r_norm = sqrt(hypre_ParKrylovInnerProd(r,r));
      if ( logging >= 1 && mypid == 0 )
         printf("LSICG actual residual norm = %e \n",r_norm);
      if ( r_norm < epsilon || iter >= max_iter ) converged = 1;
   }
   if ( iter >= max_iter ) ierr = 1;
   lsicg_data->rel_residual_norm = r_norm;
   lsicg_data->num_iterations    = iter;
   if ( logging >= 1 && mypid == 0 )
      printf("LSICG : total number of iterations = %d \n",iter);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGSetTol
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetTol( void *lsicg_vdata, double tol )
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   (lsicg_data -> tol) = tol;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGSetMaxIter
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetMaxIter( void *lsicg_vdata, int max_iter )
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   (lsicg_data -> max_iter) = max_iter;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGSetStopCrit
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetStopCrit( void *lsicg_vdata, double stop_crit )
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   (lsicg_data -> stop_crit) = stop_crit;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGSetPrecond
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetPrecond( void  *lsicg_vdata, int  (*precond)(),
                       int  (*precond_setup)(), void  *precond_data )
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   (lsicg_data -> precond)        = precond;
   (lsicg_data -> precond_setup)  = precond_setup;
   (lsicg_data -> precond_data)   = precond_data;
   return 0;
}
 
/*--------------------------------------------------------------------------
 * hypre_LSICGSetLogging
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGSetLogging( void *lsicg_vdata, int logging)
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   (lsicg_data -> logging) = logging;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_LSICGGetNumIterations
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGGetNumIterations(void *lsicg_vdata,int  *num_iterations)
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   *num_iterations = (lsicg_data -> num_iterations);
   return 0;
}
 
/*--------------------------------------------------------------------------
 * hypre_LSICGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int hypre_LSICGGetFinalRelativeResidualNorm(void *lsicg_vdata,
                                            double *relative_residual_norm)
{
   hypre_LSICGData *lsicg_data = lsicg_vdata;
   *relative_residual_norm = (lsicg_data -> rel_residual_norm);
   return 0;
} 

