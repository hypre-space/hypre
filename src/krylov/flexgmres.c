/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * FlexGMRES flexgmres 
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"



/*--------------------------------------------------------------------------
 * hypre_FlexGMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_FlexGMRESFunctions *
hypre_FlexGMRESFunctionsCreate(
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   HYPRE_Int    (*Free)          ( char *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id, HYPRE_Int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( double alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   )
{
   hypre_FlexGMRESFunctions * fgmres_functions;
   fgmres_functions = (hypre_FlexGMRESFunctions *)
      CAlloc( 1, sizeof(hypre_FlexGMRESFunctions) );

   fgmres_functions->CAlloc = CAlloc;
   fgmres_functions->Free = Free;
   fgmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   fgmres_functions->CreateVector = CreateVector;
   fgmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   fgmres_functions->DestroyVector = DestroyVector;
   fgmres_functions->MatvecCreate = MatvecCreate;
   fgmres_functions->Matvec = Matvec;
   fgmres_functions->MatvecDestroy = MatvecDestroy;
   fgmres_functions->InnerProd = InnerProd;
   fgmres_functions->CopyVector = CopyVector;
   fgmres_functions->ClearVector = ClearVector;
   fgmres_functions->ScaleVector = ScaleVector;
   fgmres_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   fgmres_functions->precond_setup = PrecondSetup;
   fgmres_functions->precond       = Precond;

   fgmres_functions->modify_pc     = hypre_FlexGMRESModifyPCDefault;


   return fgmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_FlexGMRESCreate( hypre_FlexGMRESFunctions *fgmres_functions )
{
   hypre_FlexGMRESData *fgmres_data;
 
   fgmres_data = hypre_CTAllocF(hypre_FlexGMRESData, 1, fgmres_functions);

   fgmres_data->functions = fgmres_functions;
 
   /* set defaults */
   (fgmres_data -> k_dim)          = 20;
   (fgmres_data -> tol)            = 1.0e-06;
   (fgmres_data -> cf_tol)         = 0.0;
   (fgmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (fgmres_data -> min_iter)       = 0;
   (fgmres_data -> max_iter)       = 1000;
   (fgmres_data -> rel_change)     = 0;
   (fgmres_data -> stop_crit)      = 0; /* rel. residual norm */
   (fgmres_data -> converged)      = 0;
   (fgmres_data -> precond_data)   = NULL;
   (fgmres_data -> print_level)    = 0;
   (fgmres_data -> logging)        = 0;
   (fgmres_data -> p)              = NULL;
   (fgmres_data -> r)              = NULL;
   (fgmres_data -> w)              = NULL;
   (fgmres_data -> w_2)            = NULL;
   (fgmres_data -> matvec_data)    = NULL;
   (fgmres_data -> norms)          = NULL;
   (fgmres_data -> log_file_name)  = NULL;
 

   return (void *) fgmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESDestroy
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESDestroy( void *fgmres_vdata )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;
   hypre_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
   HYPRE_Int i;
 
   if (fgmres_data)
   {
      if ( (fgmres_data->logging>0) || (fgmres_data->print_level) > 0 )
      {
         if ( (fgmres_data -> norms) != NULL )
            hypre_TFreeF( fgmres_data -> norms, fgmres_functions );
      }
 
      if ( (fgmres_data -> matvec_data) != NULL )
         (*(fgmres_functions->MatvecDestroy))(fgmres_data -> matvec_data);
 
      if ( (fgmres_data -> r) != NULL )
         (*(fgmres_functions->DestroyVector))(fgmres_data -> r);
      if ( (fgmres_data -> w) != NULL )
         (*(fgmres_functions->DestroyVector))(fgmres_data -> w);
      if ( (fgmres_data -> w_2) != NULL )
         (*(fgmres_functions->DestroyVector))(fgmres_data -> w_2);


      if ( (fgmres_data -> p) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim+1); i++)
         {
            if ( (fgmres_data -> p)[i] != NULL )
	       (*(fgmres_functions->DestroyVector))( (fgmres_data -> p) [i]);
         }
         hypre_TFreeF( fgmres_data->p, fgmres_functions );
      }

      /* fgmres mod  - space for precond. vectors*/
      if ( (fgmres_data -> pre_vecs) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim + 1); i++)
         {
            if ( (fgmres_data -> pre_vecs)[i] != NULL )
	       (*(fgmres_functions->DestroyVector))( (fgmres_data -> pre_vecs) [i]);
         }
         hypre_TFreeF( fgmres_data->pre_vecs, fgmres_functions );
      }
      /*---*/



      hypre_TFreeF( fgmres_data, fgmres_functions );
      hypre_TFreeF( fgmres_functions, fgmres_functions );
   }
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_FlexGMRESGetResidual( void *fgmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */

   hypre_FlexGMRESData  *fgmres_data     = fgmres_vdata;
   *residual = fgmres_data->r;
   return hypre_error_flag;
   
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetup
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetup( void *fgmres_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_FlexGMRESData *fgmres_data     = fgmres_vdata;
   hypre_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;

   HYPRE_Int            k_dim            = (fgmres_data -> k_dim);
   HYPRE_Int            max_iter         = (fgmres_data -> max_iter);
   HYPRE_Int          (*precond_setup)() = (fgmres_functions->precond_setup);
   void          *precond_data     = (fgmres_data -> precond_data);

   HYPRE_Int            rel_change       = (fgmres_data -> rel_change);
   
 
   (fgmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((fgmres_data -> p) == NULL)
      (fgmres_data -> p) = (*(fgmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((fgmres_data -> r) == NULL)
      (fgmres_data -> r) = (*(fgmres_functions->CreateVector))(b);
   if ((fgmres_data -> w) == NULL)
      (fgmres_data -> w) = (*(fgmres_functions->CreateVector))(b);
 
   if (rel_change)
   {  
      if ((fgmres_data -> w_2) == NULL)
         (fgmres_data -> w_2) = (*(fgmres_functions->CreateVector))(b);
   }
 
   /* fgmres mod */
   (fgmres_data -> pre_vecs) = (*(fgmres_functions->CreateVectorArray))(k_dim+1,x); 
   /*---*/


   if ((fgmres_data -> matvec_data) == NULL)
      (fgmres_data -> matvec_data) = (*(fgmres_functions->MatvecCreate))(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ( (fgmres_data->logging)>0 || (fgmres_data->print_level) > 0 )
   {
      if ((fgmres_data -> norms) == NULL)
         (fgmres_data -> norms) = hypre_CTAllocF(double, max_iter + 1,fgmres_functions);
   }
   if ( (fgmres_data->print_level) > 0 ) {
      if ((fgmres_data -> log_file_name) == NULL)
         (fgmres_data -> log_file_name) = "gmres.out.log";
   }
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_FlexGMRESSolve(void  *fgmres_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_FlexGMRESData  *fgmres_data   = fgmres_vdata;
   hypre_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
   HYPRE_Int 		     k_dim        = (fgmres_data -> k_dim);
   HYPRE_Int               min_iter     = (fgmres_data -> min_iter);
   HYPRE_Int 		     max_iter     = (fgmres_data -> max_iter);
   HYPRE_Int               rel_change   = (fgmres_data -> rel_change);
   double 	     r_tol        = (fgmres_data -> tol);
   double 	     cf_tol       = (fgmres_data -> cf_tol);
   double            a_tol        = (fgmres_data -> a_tol);
   void             *matvec_data  = (fgmres_data -> matvec_data);

   void             *r            = (fgmres_data -> r);
   void             *w            = (fgmres_data -> w);
   
   void            **p            = (fgmres_data -> p);

   /* fgmres  mod*/
   void          **pre_vecs       = (fgmres_data ->pre_vecs);
   /*---*/

   HYPRE_Int 	           (*precond)()   = (fgmres_functions -> precond);
   HYPRE_Int 	            *precond_data = (fgmres_data -> precond_data);

   HYPRE_Int             print_level    = (fgmres_data -> print_level);
   HYPRE_Int             logging        = (fgmres_data -> logging);

   double         *norms          = (fgmres_data -> norms);
   
   HYPRE_Int        break_value = 0;
   HYPRE_Int	      i, j, k;
   double     *rs, **hh, *c, *s; 
   HYPRE_Int        iter; 
   HYPRE_Int        my_id, num_procs;
   double     epsilon, gamma, t, r_norm, b_norm, den_norm;
   
   double     epsmac = 1.e-16; 
   double     ieee_check = 0.;

   double     guard_zero_residual; 
   double     cf_ave_0 = 0.0;
   double     cf_ave_1 = 0.0;
   double     weight;
   double     r_norm_0;

   HYPRE_Int 	      (*modify_pc)()   = (fgmres_functions -> modify_pc);

   /* We are not checking rel. change for now... */
   rel_change = 0;
   


   (fgmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(fgmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (fgmres_data -> norms);
      /* not used yet      log_file_name  = (fgmres_data -> log_file_name);*/
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays  */
   rs = hypre_CTAllocF(double,k_dim+1,fgmres_functions); 
   c = hypre_CTAllocF(double,k_dim,fgmres_functions); 
   s = hypre_CTAllocF(double,k_dim,fgmres_functions); 


  /* fgmres mod. - need non-modified hessenberg ???? */
   hh = hypre_CTAllocF(double*,k_dim+1,fgmres_functions); 
   for (i=0; i < k_dim+1; i++)
   {	
      hh[i] = hypre_CTAllocF(double,k_dim,fgmres_functions); 
   }
   
   (*(fgmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(fgmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(fgmres_functions->InnerProd))(b,b));

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
        hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        hypre_printf("ERROR -- hypre_FlexGMRESSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied b.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   r_norm = sqrt((*(fgmres_functions->InnerProd))(p[0],p[0]));
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
        hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        hypre_printf("ERROR -- hypre_FlexGMRESSolve: INFs and/or NaNs detected in input.\n");
        hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
        hypre_printf("Returning error flag += 101.  Program not terminated.\n");
        hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   if ( logging>0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level>1 && my_id == 0 )
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
/* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
     den_norm= b_norm;
   }
   else
   {
/* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
     den_norm= r_norm;
   };

/* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = hypre_max(a_tol,r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */


   if ( print_level>1 && my_id == 0 )
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
      
          };
   }



   /* outer iteration cycle */
   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           hypre_TFreeF(c,fgmres_functions); 
           hypre_TFreeF(s,fgmres_functions); 
           hypre_TFreeF(rs,fgmres_functions);

           for (i=0; i < k_dim+1; i++) {
              hypre_TFreeF(hh[i],fgmres_functions);
           }

           hypre_TFreeF(hh,fgmres_functions); 
	   return hypre_error_flag;
           
	}

        /* see if we are already converged and 
           should print the final norm and exit */
	if (r_norm  <= epsilon && iter >= min_iter) 
        {
           
           (*(fgmres_functions->CopyVector))(b,r);
           (*(fgmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
           r_norm = sqrt((*(fgmres_functions->InnerProd))(r,r));
           if (r_norm <= epsilon)
           {
              if ( print_level>1 && my_id == 0)
              {
                 hypre_printf("\n\n");
                 hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
              }
              break;
           }
           else
              if ( print_level>0 && my_id == 0)
                 hypre_printf("false convergence 1\n");
           
	}
        
      	t = 1.0 / r_norm;

	
        (*(fgmres_functions->ScaleVector))(t,p[0]);
	i = 0;


        /***RESTART CYCLE (right-preconditioning) ***/
        while (i < k_dim  && iter < max_iter)
	{
           i++;
           iter++;
           
           (*(fgmres_functions->ClearVector))(pre_vecs[i-1]);
           
           /* allow some user function here (to change
            * prec. attributes, i.e.tolerances, etc. ? */
           modify_pc(precond_data, iter, r_norm/den_norm );

           /*apply preconditioner and store in pre_vecs */
           precond(precond_data, A, p[i-1], pre_vecs[i-1]);
           /*apply operator and store in p */
           (*(fgmres_functions->Matvec))(matvec_data, 1.0, A, pre_vecs[i-1], 0.0, p[i]);
           

           /* modified Gram_Schmidt */
           for (j=0; j < i; j++)
           {
              hh[j][i-1] = (*(fgmres_functions->InnerProd))(p[j],p[i]);
              (*(fgmres_functions->Axpy))(-hh[j][i-1],p[j],p[i]);
           }
           t = sqrt((*(fgmres_functions->InnerProd))(p[i],p[i]));
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0/t;
              (*(fgmres_functions->ScaleVector))(t,p[i]);
           }


           /* done with modified Gram_schmidt and Arnoldi step.
              update factorization of hh */
           for (j = 1; j < i; j++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t= hh[i][i-1]*hh[i][i-1];
           t+= hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i]/=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if ( print_level>0 )
           {
              norms[iter] = r_norm;
              if ( print_level>1 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                    hypre_printf("% 5d    %e    %f   %e\n", iter, 
                           norms[iter],norms[iter]/norms[iter-1],
                           norms[iter]/b_norm);
                 else
                    hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                           norms[iter]/norms[iter-1]);
              }
           }
           /*convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));
              
              weight   = fabs(cf_ave_1 - cf_ave_0);
              weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
              weight   = 1.0 - weight;
#if 0
              hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                     i, cf_ave_1, cf_ave_0, weight );
#endif
              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm  <= epsilon && iter >= min_iter)
           {
              /* no relative change */
           
              break;
           
           }
           

	} /*** end of restart cycle ***/

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1]/hh[i-1][i-1];
	for (k = i-2; k >= 0; k--)
	{
           t = 0.0;
           for (j = k+1; j < i; j++)
           {
              t -= hh[k][j]*rs[j];
           }
           t+= rs[k];
           rs[k] = t/hh[k][k];
	}
        /* form linear combination of pre_vecs's to get solution */
      
        (*(fgmres_functions->CopyVector))(pre_vecs[i-1],w);
        (*(fgmres_functions->ScaleVector))(rs[i-1],w);
        for (j = i-2; j >=0; j--)
           (*(fgmres_functions->Axpy))(rs[j], pre_vecs[j], w);
        

        /* don't need to un-wind precond... - so now the correction is
         * in w */


        /* update current solution x (in x) */
	(*(fgmres_functions->Axpy))(1.0,w,x);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm*/
           (*(fgmres_functions->CopyVector))(b,r);
           (*(fgmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
           r_norm = sqrt( (*(fgmres_functions->InnerProd))(r,r) );
           
           if (r_norm <= epsilon)
           {
              if ( print_level>1 && my_id == 0 )
              {
                 hypre_printf("\n\n");
                 hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
              }
              (fgmres_data -> converged) = 1;
              break;
              
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if ( print_level>0 && my_id == 0)
                 hypre_printf("false convergence 2\n");
              (*(fgmres_functions->CopyVector))(r,p[0]);
              i = 0;
           }
	} /* end of convergence check */
        
        /* compute residual vector and continue loop */
	for (j=i ; j > 0; j--)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) (*(fgmres_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
        for (j=i-1 ; j > 0; j--)
           (*(fgmres_functions->Axpy))(rs[j],p[j],p[i]);
        
        if (i)
        {
           (*(fgmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
           (*(fgmres_functions->Axpy))(1.0,p[i],p[0]);
        }

   } /* END of iteration while loop */
        

   if ( print_level>1 && my_id == 0 )
          hypre_printf("\n\n"); 

   (fgmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) hypre_error(HYPRE_ERROR_CONV);
   

   hypre_TFreeF(c,fgmres_functions); 
   hypre_TFreeF(s,fgmres_functions); 
   hypre_TFreeF(rs,fgmres_functions);

   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFreeF(hh[i],fgmres_functions);
   }
   hypre_TFreeF(hh,fgmres_functions); 

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetKDim, hypre_FlexGMRESGetKDim
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetKDim( void   *fgmres_vdata,
                    HYPRE_Int   k_dim )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

   
   (fgmres_data -> k_dim) = k_dim;
 
   return hypre_error_flag;
   
}

HYPRE_Int
hypre_FlexGMRESGetKDim( void   *fgmres_vdata,
                    HYPRE_Int * k_dim )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *k_dim = (fgmres_data -> k_dim);
 
   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetTol, hypre_FlexGMRESGetTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetTol( void   *fgmres_vdata,
                   double  tol       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> tol) = tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetTol( void   *fgmres_vdata,
                   double  * tol      )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *tol = (fgmres_data -> tol);
 
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetAbsoluteTol, hypre_FlexGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetAbsoluteTol( void   *fgmres_vdata,
                   double  a_tol       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> a_tol) = a_tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetAbsoluteTol( void   *fgmres_vdata,
                   double  * a_tol      )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *a_tol = (fgmres_data -> a_tol);
 
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetConvergenceFactorTol, hypre_FlexGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetConvergenceFactorTol( void   *fgmres_vdata,
                   double  cf_tol       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> cf_tol) = cf_tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetConvergenceFactorTol( void   *fgmres_vdata,
                   double * cf_tol       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *cf_tol = (fgmres_data -> cf_tol);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetMinIter, hypre_FlexGMRESGetMinIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetMinIter( void *fgmres_vdata,
                       HYPRE_Int   min_iter  )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> min_iter) = min_iter;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetMinIter( void *fgmres_vdata,
                       HYPRE_Int * min_iter  )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *min_iter = (fgmres_data -> min_iter);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetMaxIter, hypre_FlexGMRESGetMaxIter
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetMaxIter( void *fgmres_vdata,
                       HYPRE_Int   max_iter  )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> max_iter) = max_iter;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetMaxIter( void *fgmres_vdata,
                       HYPRE_Int * max_iter  )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *max_iter = (fgmres_data -> max_iter);
 
   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetStopCrit, hypre_FlexGMRESGetStopCrit
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetStopCrit( void   *fgmres_vdata,
                        HYPRE_Int  stop_crit       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> stop_crit) = stop_crit;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetStopCrit( void   *fgmres_vdata,
                        HYPRE_Int * stop_crit       )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *stop_crit = (fgmres_data -> stop_crit);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESSetPrecond( void  *fgmres_vdata,
                       HYPRE_Int  (*precond)(),
                       HYPRE_Int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;
   hypre_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;

 
   (fgmres_functions -> precond)        = precond;
   (fgmres_functions -> precond_setup)  = precond_setup;
   (fgmres_data -> precond_data)   = precond_data;
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESGetPrecond( void         *fgmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *precond_data_ptr = (HYPRE_Solver)(fgmres_data -> precond_data);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetPrintLevel, hypre_FlexGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FlexGMRESSetPrintLevel( void *fgmres_vdata,
                        HYPRE_Int   level)
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> print_level) = level;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetPrintLevel( void *fgmres_vdata,
                        HYPRE_Int * level)
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *level = (fgmres_data -> print_level);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetLogging, hypre_FlexGMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FlexGMRESSetLogging( void *fgmres_vdata,
                      HYPRE_Int   level)
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   (fgmres_data -> logging) = level;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_FlexGMRESGetLogging( void *fgmres_vdata,
                      HYPRE_Int * level)
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *level = (fgmres_data -> logging);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESGetNumIterations( void *fgmres_vdata,
                             HYPRE_Int  *num_iterations )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *num_iterations = (fgmres_data -> num_iterations);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESGetConverged
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESGetConverged( void *fgmres_vdata,
                             HYPRE_Int  *converged )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *converged = (fgmres_data -> converged);
 
   return hypre_error_flag;
}
 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_FlexGMRESGetFinalRelativeResidualNorm( void   *fgmres_vdata,
                                         double *relative_residual_norm )
{
   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;

 
   *relative_residual_norm = (fgmres_data -> rel_residual_norm);
   
   return hypre_error_flag;
} 
/*--------------------------------------------------------------------------
 * hypre_FlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 
HYPRE_Int hypre_FlexGMRESSetModifyPC(void *fgmres_vdata, 
                               HYPRE_Int (*modify_pc)())
{

   hypre_FlexGMRESData *fgmres_data = fgmres_vdata;
   hypre_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;

   (fgmres_functions -> modify_pc)        = modify_pc;

   return hypre_error_flag;
} 


/*--------------------------------------------------------------------------
 * hypre_FlexGMRESModifyPCDefault - if the user does not specify a function
 *--------------------------------------------------------------------------*/
 
HYPRE_Int hypre_FlexGMRESModifyPCDefault(void *precond_data, HYPRE_Int iteration, 
                                double rel_residual_norm)
{


   /* Here would could check the number of its and the current
      residual and make some changes to the preconditioner.  There is
      an example in ex5.c.*/


   return 0;
} 







