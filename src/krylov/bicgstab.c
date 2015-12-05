/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.23 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"


/*--------------------------------------------------------------------------
 * hypre_BiCGSTABFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_BiCGSTABFunctions *
hypre_BiCGSTABFunctionsCreate(
   void *(*CreateVector)( void *vvector ),
   HYPRE_Int (*DestroyVector)( void *vvector ),
   void *(*MatvecCreate)( void *A , void *x ),
   HYPRE_Int (*Matvec)( void *matvec_data , double alpha , void *A , void *x , double beta , void *y ),
   HYPRE_Int (*MatvecDestroy)( void *matvec_data ),
   double (*InnerProd)( void *x , void *y ),
   HYPRE_Int (*CopyVector)( void *x , void *y ),
   HYPRE_Int (*ClearVector)( void *x ),
   HYPRE_Int (*ScaleVector)( double alpha , void *x ),
   HYPRE_Int (*Axpy)( double alpha , void *x , void *y ),
   HYPRE_Int (*CommInfo)( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs ),
   HYPRE_Int (*PrecondSetup)(  void *vdata, void *A, void *b, void *x ),
   HYPRE_Int (*Precond)( void *vdata, void *A, void *b, void *x )
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
   bicgstab_functions->ClearVector = ClearVector;
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
   (bicgstab_data -> a_tol)          = 0.0;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> print_level)    = 0;
   (bicgstab_data -> p)              = NULL;
   (bicgstab_data -> q)              = NULL;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> r0)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> v)             = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABDestroy( void *bicgstab_vdata )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

 
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
 
   return(hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetup
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetup( void *bicgstab_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_BiCGSTABData *bicgstab_data     = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   HYPRE_Int            max_iter         = (bicgstab_data -> max_iter);
   HYPRE_Int          (*precond_setup)() = (bicgstab_functions -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
 
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
 
   precond_setup(precond_data, A, b, x);
 
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
 
   return hypre_error_flag;
}
 
/*-------------------------------------------------------------------------- 
 * hypre_BiCGSTABSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_BiCGSTABSolve(void  *bicgstab_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_BiCGSTABData  *bicgstab_data   = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   HYPRE_Int               min_iter     = (bicgstab_data -> min_iter);
   HYPRE_Int 		     max_iter     = (bicgstab_data -> max_iter);
   HYPRE_Int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     r_tol     = (bicgstab_data -> tol);
   double 	     cf_tol       = (bicgstab_data -> cf_tol);
   void             *matvec_data  = (bicgstab_data -> matvec_data);
   double            a_tol        = (bicgstab_data -> a_tol);
  
   

   void             *r            = (bicgstab_data -> r);
   void             *r0           = (bicgstab_data -> r0);
   void             *s            = (bicgstab_data -> s);
   void             *v           = (bicgstab_data -> v);
   void             *p            = (bicgstab_data -> p);
   void             *q            = (bicgstab_data -> q);

   HYPRE_Int 	           (*precond)()   = (bicgstab_functions -> precond);
   HYPRE_Int 	            *precond_data = (bicgstab_data -> precond_data);

   /* logging variables */
   HYPRE_Int             logging        = (bicgstab_data -> logging);
   HYPRE_Int             print_level    = (bicgstab_data -> print_level);
   double         *norms          = (bicgstab_data -> norms);
   /*   char           *log_file_name  = (bicgstab_data -> log_file_name);
     FILE           *fp; */
   
   HYPRE_Int        ierr = 0;
   HYPRE_Int        iter; 
   HYPRE_Int        my_id, num_procs;
   double     alpha, beta, gamma, epsilon, temp, res, r_norm, b_norm;
   double     epsmac = 1.e-128; 
   double     ieee_check = 0.;
   double     cf_ave_0 = 0.0;
   double     cf_ave_1 = 0.0;
   double     weight;
   double     r_norm_0;
   double     den_norm;
   double     gamma_numer;
   double     gamma_denom;

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
        hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        hypre_printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied b.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
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
        hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
        hypre_printf("ERROR -- hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }

      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   if (logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0 && my_id == 0)
      {
   	     hypre_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= r_tol*|b| if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= r_tol*|r0| if |b| = 0 */
      den_norm = r_norm;
   };

   /* convergence criterion |r_i| <= r_tol/a_tol , absolute residual norm*/
   if (stop_crit)
   {
      if (a_tol == 0.0) /* this is for backwards compatibility
                           (accomodating setting stop_crit to 1, but not setting a_tol) -
                           eventually we will get rid of the stop_crit flag as with GMRES */
         epsilon = r_tol;
      else
         epsilon = a_tol; /* this means new interface fcn called */
      
   }
   else /* default convergence test (stop_crit = 0)*/
   {
      
      /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user also specifies a_tol or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
      epsilon = hypre_max(a_tol, r_tol*den_norm);
   
   }
   
   
   if (print_level > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
         {hypre_printf("=============================================\n\n");
          hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          hypre_printf("-----    ------------    ---------- ------------\n");
      }
      else
         {hypre_printf("=============================================\n\n");
          hypre_printf("Iters     resid.norm     conv.rate\n");
          hypre_printf("-----    ------------    ----------\n");
      
      }
   }

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   /* check for convergence before starting */
   if (r_norm == 0.0)
   {
	   ierr = 0;
	   return hypre_error_flag;
   }
   else if (r_norm <= epsilon && iter >= min_iter) 
   {
       if (print_level > 0 && my_id == 0)
       {
          hypre_printf("\n\n");
          hypre_printf("Tolerance and min_iter requirements satisfied by initial data.\n");
          hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
       }
       (bicgstab_data -> converged) = 1;
       return hypre_error_flag;
   }
   /* Start BiCGStab iterations */
   while (iter < max_iter)
   {
        iter++;

	(*(bicgstab_functions->ClearVector))(v);
        precond(precond_data, A, p, v);
        (*(bicgstab_functions->Matvec))(matvec_data,1.0,A,v,0.0,q);
      	temp = (*(bicgstab_functions->InnerProd))(r0,q);
      	if (fabs(temp) >= epsmac)
	   alpha = res/temp;
	else
	{
	   hypre_printf("BiCGSTAB broke down!! divide by near zero\n");
	   return(1);
	}
	(*(bicgstab_functions->Axpy))(alpha,v,x);
	(*(bicgstab_functions->Axpy))(-alpha,q,r);
	(*(bicgstab_functions->ClearVector))(v);
        precond(precond_data, A, r, v);
        (*(bicgstab_functions->Matvec))(matvec_data,1.0,A,v,0.0,s);
      	/* Handle case when gamma = 0.0/0.0 as 0.0 and not NAN */
        gamma_numer = (*(bicgstab_functions->InnerProd))(r,s);
        gamma_denom = (*(bicgstab_functions->InnerProd))(s,s);
        if ((gamma_numer == 0.0) && (gamma_denom == 0.0))
            gamma = 0.0;
        else
            gamma= gamma_numer/gamma_denom;
	(*(bicgstab_functions->Axpy))(gamma,v,x);
	(*(bicgstab_functions->Axpy))(-gamma,s,r);
    /* residual is now updated, must immediately check for convergence */
	r_norm = sqrt((*(bicgstab_functions->InnerProd))(r,r));
	if (logging > 0 || print_level > 0)
	{
	   norms[iter] = r_norm;
	}
    if (print_level > 0 && my_id == 0)
    {
        if (b_norm > 0.0)
           hypre_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                      norms[iter]/norms[iter-1], norms[iter]/b_norm);
        else
           hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
		                             norms[iter]/norms[iter-1]);
	}
    /* Is this extra check for r_norm exactly 0.0 necessary ?*/
    if (r_norm == 0.0)
    {
	   ierr = 0;
	   return hypre_error_flag;
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
              hypre_printf("\n\n");
              hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
           }
           (bicgstab_data -> converged) = 1;
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
       cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

       weight   = fabs(cf_ave_1 - cf_ave_0);
       weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
       weight   = 1.0 - weight;
       if (weight * cf_ave_1 > cf_tol) break;
    }

      	if (fabs(res) >= epsmac)
           beta = 1.0/res;
	else
	{
	   hypre_printf("BiCGSTAB broke down!! res=0 \n");
	   return(2);
	}
        res = (*(bicgstab_functions->InnerProd))(r0,r);
        beta *= res;    
	(*(bicgstab_functions->Axpy))(-gamma,q,p);
      	if (fabs(gamma) >= epsmac)
           (*(bicgstab_functions->ScaleVector))((beta*alpha/gamma),p);
	else
	{
	   hypre_printf("BiCGSTAB broke down!! gamma=0 \n");
	   return(3);
	}
	(*(bicgstab_functions->Axpy))(1.0,r,p);
   } /* end while loop */
    
   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) hypre_error(HYPRE_ERROR_CONV);


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetTol( void   *bicgstab_vdata,
                   double  tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> tol) = tol;
 
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetAbsoluteTol( void   *bicgstab_vdata,
                   double  a_tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> a_tol) = a_tol;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetConvergenceFactorTol( void   *bicgstab_vdata,
                   double  cf_tol       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> cf_tol) = cf_tol;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetMinIter( void *bicgstab_vdata,
                       HYPRE_Int   min_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> min_iter) = min_iter;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata,
                       HYPRE_Int   max_iter  )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetStopCrit( void   *bicgstab_vdata,
                        HYPRE_Int  stop_crit       )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetPrecond( void  *bicgstab_vdata,
                       HYPRE_Int  (*precond)(),
                       HYPRE_Int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

 
   (bicgstab_functions -> precond)        = precond;
   (bicgstab_functions -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABGetPrecond( void         *bicgstab_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   *precond_data_ptr = (HYPRE_Solver)(bicgstab_data -> precond_data);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetLogging( void *bicgstab_vdata,
                       HYPRE_Int   logging)
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> logging) = logging;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABSetPrintLevel( void *bicgstab_vdata,
                       HYPRE_Int   print_level)
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   (bicgstab_data -> print_level) = print_level;
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetConverged
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABGetConverged( void *bicgstab_vdata,
                             HYPRE_Int  *converged )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   *converged = (bicgstab_data -> converged);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata,
                             HYPRE_Int  *num_iterations )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return hypre_error_flag;
} 

/*--------------------------------------------------------------------------
 * hypre_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BiCGSTABGetResidual( void   *bicgstab_vdata,
                           void **residual )
{
   hypre_BiCGSTABData *bicgstab_data = bicgstab_vdata;
 
   *residual = (bicgstab_data -> r);
   
   return hypre_error_flag;
} 
