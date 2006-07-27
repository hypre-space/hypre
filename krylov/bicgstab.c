/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#include "krylov.h"
#include "utilities.h"


/*--------------------------------------------------------------------------
 * hypre_BiCGSTABFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_BiCGSTABFunctions *
hypre_BiCGSTABFunctionsCreate(
   void *(*CreateVector)( void *vvector ),
   int (*DestroyVector)( void *vvector ),
   void *(*MatvecCreate)( void *A , void *x ),
   int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y ),
   int (*MatvecDestroy)( void *matvec_data ),
   double (*InnerProd)( void *x , void *y ),
   int (*CopyVector)( void *x , void *y ),
   int (*ScaleVector)( double alpha , void *x ),
   int (*Axpy)( double alpha , void *x , void *y ),
   int (*CommInfo)( void *A , int *my_id , int *num_procs ),
   int (*PrecondSetup)(  void *vdata, void *A, void *b, void *x ),
   int (*Precond)( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_BiCGSTABFunctions * bicgstab_functions;
   bicgstab_functions = (hypre_BiCGSTABFunctions *)
      hypre_CTAlloc( hypre_BiCGSTABFunctions, 1 );

   bicgstab_functions->CreateVector = CreateVector;
   bicgstab_functions->DestroyVector = DestroyVector;
   bicgstab_functions->MatvecCreate = MatvecCreate;
   bicgstab_functions->Matvec = Matvec;
   bicgstab_functions->MatvecDestroy = MatvecDestroy;
   bicgstab_functions->InnerProd = InnerProd;
   bicgstab_functions->CopyVector = CopyVector;
   bicgstab_functions->ScaleVector = ScaleVector;
   bicgstab_functions->Axpy = Axpy;
   bicgstab_functions->CommInfo = CommInfo;
   bicgstab_functions->precond_setup = PrecondSetup;
   bicgstab_functions->precond = Precond;

   return bicgstab_functions;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions * bicgstab_functions )
{
   hypre_BiCGSTABData *bicgstab_data;
 
   bicgstab_data = hypre_CTAlloc( hypre_BiCGSTABData, 1);
 
   bicgstab_data->functions = bicgstab_functions;

   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> min_iter)       = 0;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> print_level)    = 0;
   (bicgstab_data -> p)              = NULL;
   (bicgstab_data -> q)              = NULL;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> r0)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> v)              = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABDestroy( void *bicgstab_vdata )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   int ierr = 0;
 
   if (bicgstab_data)
   {
      if ( (bicgstab_data -> norms) != NULL )
            hypre_TFree(bicgstab_data -> norms);
 
      (*(bicgstab_functions->MatvecDestroy))(bicgstab_data -> matvec_data);
 
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r0);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> s);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> v);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> p);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> q);
 
      hypre_TFree(bicgstab_data);
      hypre_TFree(bicgstab_functions);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetup
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetup( void *bicgstab_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_BiCGSTABData *bicgstab_data     = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)() = (bicgstab_functions -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> p) == NULL)
      (bicgstab_data -> p) = (*(bicgstab_functions->CreateVector))(b);
   if ((bicgstab_data -> q) == NULL)
      (bicgstab_data -> q) = (*(bicgstab_functions->CreateVector))(b);
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = (*(bicgstab_functions->CreateVector))(b);
   if ((bicgstab_data -> r0) == NULL)
      (bicgstab_data -> r0) = (*(bicgstab_functions->CreateVector))(b);
   if ((bicgstab_data -> s) == NULL)
      (bicgstab_data -> s) = (*(bicgstab_functions->CreateVector))(b);
   if ((bicgstab_data -> v) == NULL)
      (bicgstab_data -> v) = (*(bicgstab_functions->CreateVector))(b);
 
   if ((bicgstab_data -> matvec_data) == NULL) 
      (bicgstab_data -> matvec_data) =
         (*(bicgstab_functions->MatvecCreate))(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data->logging)>0 || (bicgstab_data->print_level) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = hypre_CTAlloc(double, max_iter + 1);
   }
   if ((bicgstab_data -> print_level) > 0)
   {
      if ((bicgstab_data -> log_file_name) == NULL)
         (bicgstab_data -> log_file_name) = "bicgstab.out.log";
   }
 
   return ierr;
}
 
/*-------------------------------------------------------------------------- 
 * hypre_BiCGSTABSolve
 *-------------------------------------------------------------------------*/

int
hypre_BiCGSTABSolve(void  *bicgstab_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_BiCGSTABData  *bicgstab_data   = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   int               min_iter     = (bicgstab_data -> min_iter);
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   double 	     cf_tol       = (bicgstab_data -> cf_tol);
   void             *matvec_data  = (bicgstab_data -> matvec_data);

   void             *r            = (bicgstab_data -> r);
   void             *r0           = (bicgstab_data -> r0);
   void             *s            = (bicgstab_data -> s);
   void             *v            = (bicgstab_data -> v);
   void             *p            = (bicgstab_data -> p);
   void             *q            = (bicgstab_data -> q);

   int 	           (*precond)()   = (bicgstab_functions -> precond);
   int 	            *precond_data = (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   int             print_level    = (bicgstab_data -> print_level);
   double         *norms          = (bicgstab_data -> norms);
/*   char           *log_file_name  = (bicgstab_data -> log_file_name);
     FILE           *fp; */
   
   int        ierr = 0;
   int        iter; 
   int        my_id, num_procs;
   double     alpha, beta, gamma, epsilon, temp, res, r_norm, b_norm;
   double     epsmac = 1.e-128; 
   double     ieee_check = 0.;
   double     cf_ave_0 = 0.0;
   double     cf_ave_1 = 0.0;
   double     weight;
   double     r_norm_0;

   (bicgstab_data -> converged) = 0;

   (*(bicgstab_functions->CommInfo))(A,&my_id,&num_procs);
   if (logging > 0 || print_level > 0)
   {
      norms          = (bicgstab_data -> norms);
      /* log_file_name  = (bicgstab_data -> log_file_name);
         fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   (*(bicgstab_functions->CopyVector))(b,r0);

/* compute initial residual */

   (*(bicgstab_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, r0);
   (*(bicgstab_functions->CopyVector))(r0,r);
   (*(bicgstab_functions->CopyVector))(r0,p);

   b_norm = sqrt((*(bicgstab_functions->InnerProd))(b,b));

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied b.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   res = (*(bicgstab_functions->InnerProd))(r0,r0);
   r_norm = sqrt(res);
   r_norm_0 = r_norm;
 
   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied A or x_0.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   if (logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0 && my_id == 0)
      {
  	 printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("Initial L2 norm of residual: %e\n", r_norm);
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
   if (stop_crit)
      epsilon = accuracy;

   if (print_level > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      }
      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
      }
   }

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   while (iter < max_iter)
   {

        if (r_norm == 0.0)
        {
	   ierr = 0;
	   return ierr;
	}

/* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
	   (*(bicgstab_functions->CopyVector))(b,r);
           (*(bicgstab_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
	   r_norm = sqrt((*(bicgstab_functions->InnerProd))(r,r));
	   if (r_norm <= epsilon)
           {
              if (print_level > 0 && my_id == 0)
              {
                 printf("\n\n");
                 printf("Final L2 norm of residual: %e\n\n", r_norm);
              }
              (bicgstab_data -> converged) = 1;
              break;
           }
	   else
	   {
	      (*(bicgstab_functions->CopyVector))(r,p);
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
           cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

           weight   = fabs(cf_ave_1 - cf_ave_0);
           weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
           weight   = 1.0 - weight;
#if 0
           printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                i, cf_ave_1, cf_ave_0, weight );
#endif
           if (weight * cf_ave_1 > cf_tol) break;
        }

        iter++;

        precond(precond_data, A, p, v);
        (*(bicgstab_functions->Matvec))(matvec_data,1.0,A,v,0.0,q);
      	temp = (*(bicgstab_functions->InnerProd))(r0,q);
      	if (fabs(temp) >= epsmac)
	   alpha = res/temp;
	else
	{
	   printf("BiCGSTAB broke down!! divide by near zero\n");
	   return(1);
	}
	(*(bicgstab_functions->Axpy))(alpha,v,x);
	(*(bicgstab_functions->Axpy))(-alpha,q,r);
        precond(precond_data, A, r, v);
        (*(bicgstab_functions->Matvec))(matvec_data,1.0,A,v,0.0,s);
      	gamma = (*(bicgstab_functions->InnerProd))(r,s) /
           (*(bicgstab_functions->InnerProd))(s,s);
	(*(bicgstab_functions->Axpy))(gamma,v,x);
	(*(bicgstab_functions->Axpy))(-gamma,s,r);
      	if (fabs(res) >= epsmac)
           beta = 1.0/res;
	else
	{
	   printf("BiCGSTAB broke down!! res=0 \n");
	   return(2);
	}
        res = (*(bicgstab_functions->InnerProd))(r0,r);
        beta *= res;    
	(*(bicgstab_functions->Axpy))(-gamma,q,p);
      	if (fabs(gamma) >= epsmac)
           (*(bicgstab_functions->ScaleVector))((beta*alpha/gamma),p);
	else
	{
	   printf("BiCGSTAB broke down!! gamma=0 \n");
	   return(3);
	}
	(*(bicgstab_functions->Axpy))(1.0,r,p);

	r_norm = sqrt((*(bicgstab_functions->InnerProd))(r,r));
	if (logging > 0 || print_level > 0)
	{
	   norms[iter] = r_norm;
	}

        if (print_level > 0 && my_id == 0)
	{
           if (b_norm > 0.0)
              printf("% 5d    %e    %f   %e\n", iter, norms[iter],
			norms[iter]/norms[iter-1], norms[iter]/b_norm);
           else
              printf("% 5d    %e    %f\n", iter, norms[iter],
		norms[iter]/norms[iter-1]);
	}
   }

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetTol( void   *bicgstab_vdata,
                   double  tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetConvergenceFactorTol( void   *bicgstab_vdata,
                   double  cf_tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> cf_tol) = cf_tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetMinIter( void *bicgstab_vdata,
                       int   min_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> min_iter) = min_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata,
                       int   max_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetStopCrit( void   *bicgstab_vdata,
                        int  stop_crit       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetPrecond( void  *bicgstab_vdata,
                       int  (*precond)(),
                       int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   int              ierr = 0;
 
   (bicgstab_functions -> precond)        = precond;
   (bicgstab_functions -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetPrecond( void         *bicgstab_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *precond_data_ptr = (HYPRE_Solver)(bicgstab_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetLogging( void *bicgstab_vdata,
                       int   logging)
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABSetPrintLevel( void *bicgstab_vdata,
                       int   print_level)
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> print_level) = print_level;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetConverged
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetConverged( void *bicgstab_vdata,
                             int  *converged )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *converged = (bicgstab_data -> converged);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata,
                             int  *num_iterations )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/
 
int
hypre_BiCGSTABGetResidual( void   *bicgstab_vdata,
                           void **residual )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   int 		ierr = 0;
 
   *residual = (bicgstab_data -> r);
   
   return ierr;
} 
