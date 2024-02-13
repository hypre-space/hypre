/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * COGMRES cogmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_COGMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_COGMRESFunctions *
hypre_COGMRESFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
   HYPRE_Int    (*Free)          ( void *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*MassInnerProd) (void *x, void **y, HYPRE_Int k, HYPRE_Int unroll, void *result),
   HYPRE_Int    (*MassDotpTwo)   (void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll,
                                  void *result_x, void *result_y),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   hypre_COGMRESFunctions * cogmres_functions;
   cogmres_functions = (hypre_COGMRESFunctions *)
                       CAlloc( 1, sizeof(hypre_COGMRESFunctions), HYPRE_MEMORY_HOST );

   cogmres_functions->CAlloc            = CAlloc;
   cogmres_functions->Free              = Free;
   cogmres_functions->CommInfo          = CommInfo; /* not in PCGFunctionsCreate */
   cogmres_functions->CreateVector      = CreateVector;
   cogmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   cogmres_functions->DestroyVector     = DestroyVector;
   cogmres_functions->MatvecCreate      = MatvecCreate;
   cogmres_functions->Matvec            = Matvec;
   cogmres_functions->MatvecDestroy     = MatvecDestroy;
   cogmres_functions->InnerProd         = InnerProd;
   cogmres_functions->MassInnerProd     = MassInnerProd;
   cogmres_functions->MassDotpTwo       = MassDotpTwo;
   cogmres_functions->CopyVector        = CopyVector;
   cogmres_functions->ClearVector       = ClearVector;
   cogmres_functions->ScaleVector       = ScaleVector;
   cogmres_functions->Axpy              = Axpy;
   cogmres_functions->MassAxpy          = MassAxpy;
   /* default preconditioner must be set here but can be changed later... */
   cogmres_functions->precond_setup     = PrecondSetup;
   cogmres_functions->precond           = Precond;

   return cogmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESCreate
 *--------------------------------------------------------------------------*/

void *
hypre_COGMRESCreate( hypre_COGMRESFunctions *cogmres_functions )
{
   hypre_COGMRESData *cogmres_data;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   cogmres_data = hypre_CTAllocF(hypre_COGMRESData, 1, cogmres_functions, HYPRE_MEMORY_HOST);
   cogmres_data->functions = cogmres_functions;

   /* set defaults */
   (cogmres_data -> k_dim)          = 5;
   (cogmres_data -> cgs)            = 1; /* if 2 performs reorthogonalization */
   (cogmres_data -> tol)            = 1.0e-06; /* relative residual tol */
   (cogmres_data -> cf_tol)         = 0.0;
   (cogmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (cogmres_data -> min_iter)       = 0;
   (cogmres_data -> max_iter)       = 1000;
   (cogmres_data -> rel_change)     = 0;
   (cogmres_data -> skip_real_r_check) = 0;
   (cogmres_data -> converged)      = 0;
   (cogmres_data -> precond_data)   = NULL;
   (cogmres_data -> print_level)    = 0;
   (cogmres_data -> logging)        = 0;
   (cogmres_data -> p)              = NULL;
   (cogmres_data -> r)              = NULL;
   (cogmres_data -> w)              = NULL;
   (cogmres_data -> w_2)            = NULL;
   (cogmres_data -> matvec_data)    = NULL;
   (cogmres_data -> norms)          = NULL;
   (cogmres_data -> log_file_name)  = NULL;
   (cogmres_data -> unroll)         = 0;

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) cogmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESDestroy( void *cogmres_vdata )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   HYPRE_Int i;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (cogmres_data)
   {
      hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
      if ( (cogmres_data->logging > 0) || (cogmres_data->print_level) > 0 )
      {
         if ( (cogmres_data -> norms) != NULL )
         {
            hypre_TFreeF( cogmres_data -> norms, cogmres_functions );
         }
      }

      if ( (cogmres_data -> matvec_data) != NULL )
      {
         (*(cogmres_functions->MatvecDestroy))(cogmres_data -> matvec_data);
      }

      if ( (cogmres_data -> r) != NULL )
      {
         (*(cogmres_functions->DestroyVector))(cogmres_data -> r);
      }
      if ( (cogmres_data -> w) != NULL )
      {
         (*(cogmres_functions->DestroyVector))(cogmres_data -> w);
      }
      if ( (cogmres_data -> w_2) != NULL )
      {
         (*(cogmres_functions->DestroyVector))(cogmres_data -> w_2);
      }


      if ( (cogmres_data -> p) != NULL )
      {
         for (i = 0; i < (cogmres_data -> k_dim + 1); i++)
         {
            if ( (cogmres_data -> p)[i] != NULL )
            {
               (*(cogmres_functions->DestroyVector))( (cogmres_data -> p) [i]);
            }
         }
         hypre_TFreeF( cogmres_data->p, cogmres_functions );
      }
      hypre_TFreeF( cogmres_data, cogmres_functions );
      hypre_TFreeF( cogmres_functions, cogmres_functions );
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_COGMRESGetResidual( void *cogmres_vdata, void **residual )
{
   hypre_COGMRESData  *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *residual = cogmres_data->r;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetup( void *cogmres_vdata,
                    void *A,
                    void *b,
                    void *x         )
{
   hypre_COGMRESData *cogmres_data     = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;

   HYPRE_Int k_dim            = (cogmres_data -> k_dim);
   HYPRE_Int max_iter         = (cogmres_data -> max_iter);
   HYPRE_Int (*precond_setup)(void*, void*, void*, void*) = (cogmres_functions->precond_setup);
   void       *precond_data   = (cogmres_data -> precond_data);
   HYPRE_Int rel_change       = (cogmres_data -> rel_change);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (cogmres_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((cogmres_data -> p) == NULL)
   {
      (cogmres_data -> p) = (void**)(*(cogmres_functions->CreateVectorArray))(k_dim + 1, x);
   }
   if ((cogmres_data -> r) == NULL)
   {
      (cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);
   }
   if ((cogmres_data -> w) == NULL)
   {
      (cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
   }

   if (rel_change)
   {
      if ((cogmres_data -> w_2) == NULL)
      {
         (cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
      }
   }


   if ((cogmres_data -> matvec_data) == NULL)
   {
      (cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);
   }

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (cogmres_data->logging) > 0 || (cogmres_data->print_level) > 0 )
   {
      if ((cogmres_data -> norms) == NULL)
      {
         (cogmres_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1, cogmres_functions,
                                                  HYPRE_MEMORY_HOST);
      }
   }
   if ( (cogmres_data->print_level) > 0 )
   {
      if ((cogmres_data -> log_file_name) == NULL)
      {
         (cogmres_data -> log_file_name) = (char*)"cogmres.out.log";
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSolve(void  *cogmres_vdata,
                   void  *A,
                   void  *b,
                   void  *x)
{

   hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   HYPRE_Int     k_dim             = (cogmres_data -> k_dim);
   HYPRE_Int     unroll            = (cogmres_data -> unroll);
   HYPRE_Int     cgs               = (cogmres_data -> cgs);
   HYPRE_Int     min_iter          = (cogmres_data -> min_iter);
   HYPRE_Int     max_iter          = (cogmres_data -> max_iter);
   HYPRE_Int     rel_change        = (cogmres_data -> rel_change);
   HYPRE_Int     skip_real_r_check = (cogmres_data -> skip_real_r_check);
   HYPRE_Real    r_tol             = (cogmres_data -> tol);
   HYPRE_Real    cf_tol            = (cogmres_data -> cf_tol);
   HYPRE_Real    a_tol             = (cogmres_data -> a_tol);
   void         *matvec_data       = (cogmres_data -> matvec_data);

   void         *r                 = (cogmres_data -> r);
   void         *w                 = (cogmres_data -> w);
   /* note: w_2 is only allocated if rel_change = 1 */
   void         *w_2               = (cogmres_data -> w_2);

   void        **p                 = (cogmres_data -> p);

   HYPRE_Int (*precond)(void*, void*, void*, void*) = (cogmres_functions -> precond);
   HYPRE_Int  *precond_data       = (HYPRE_Int*)(cogmres_data -> precond_data);

   HYPRE_Int print_level = (cogmres_data -> print_level);
   HYPRE_Int logging     = (cogmres_data -> logging);

   HYPRE_Real     *norms          = (cogmres_data -> norms);
   /* not used yet   char           *log_file_name  = (cogmres_data -> log_file_name);*/
   /*   FILE           *fp; */

   HYPRE_Int  break_value = 0;
   HYPRE_Int  i, j, k;
   /*KS: rv is the norm history */
   HYPRE_Real *rs, *hh, *uu, *c, *s, *rs_2 = NULL, *rv;
   //, *tmp;
   HYPRE_Int  iter;
   HYPRE_Int  my_id, num_procs;
   HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   HYPRE_Real w_norm;

   HYPRE_Real epsmac = 1.e-16;
   HYPRE_Real ieee_check = 0.;

   HYPRE_Real guard_zero_residual;
   HYPRE_Real cf_ave_0 = 0.0;
   HYPRE_Real cf_ave_1 = 0.0;
   HYPRE_Real weight;
   HYPRE_Real r_norm_0;
   HYPRE_Real relative_error = 1.0;

   HYPRE_Int        rel_change_passed = 0, num_rel_change_check = 0;
   HYPRE_Int    itmp = 0;

   HYPRE_Real real_r_norm_old, real_r_norm_new;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (cogmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(cogmres_functions->CommInfo))(A, &my_id, &num_procs);
   if ( logging > 0 || print_level > 0 )
   {
      norms = (cogmres_data -> norms);
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(HYPRE_Real, k_dim + 1, cogmres_functions, HYPRE_MEMORY_HOST);
   c  = hypre_CTAllocF(HYPRE_Real, k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
   s  = hypre_CTAllocF(HYPRE_Real, k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
   if (rel_change) { rs_2 = hypre_CTAllocF(HYPRE_Real, k_dim + 1, cogmres_functions, HYPRE_MEMORY_HOST); }

   rv = hypre_CTAllocF(HYPRE_Real, k_dim + 1, cogmres_functions, HYPRE_MEMORY_HOST);

   hh = hypre_CTAllocF(HYPRE_Real, (k_dim + 1) * k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
   uu = hypre_CTAllocF(HYPRE_Real, (k_dim + 1) * k_dim, cogmres_functions, HYPRE_MEMORY_HOST);

   (*(cogmres_functions->CopyVector))(b, p[0]);

   /* compute initial residual */
   (*(cogmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, p[0]);

   b_norm = hypre_sqrt((*(cogmres_functions->InnerProd))(b, b));
   real_r_norm_old = b_norm;

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) { ieee_check = b_norm / b_norm; } /* INF -> NaN conversion */
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
         hypre_printf("ERROR -- hypre_COGMRESSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied b.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   r_norm   = hypre_sqrt((*(cogmres_functions->InnerProd))(p[0], p[0]));
   r_norm_0 = r_norm;

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) { ieee_check = r_norm / r_norm; } /* INF -> NaN conversion */
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
         hypre_printf("ERROR -- hypre_COGMRESSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if ( logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level > 1 && my_id == 0 )
      {
         hypre_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   };

   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
      user specifies a_tol, or sets r_tol = 0.0, which means absolute
      tol only is checked  */

   epsilon = hypre_max(a_tol, r_tol * den_norm);

   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level > 1 && my_id == 0 )
   {
      if (b_norm > 0.0)
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
         hypre_printf("-----    ------------    ---------- ------------\n");

      }
      else
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate\n");
         hypre_printf("-----    ------------    ----------\n");
      };
   }


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */
      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         hypre_TFreeF(c, cogmres_functions);
         hypre_TFreeF(s, cogmres_functions);
         hypre_TFreeF(rs, cogmres_functions);
         hypre_TFreeF(rv, cogmres_functions);
         if (rel_change) { hypre_TFreeF(rs_2, cogmres_functions); }
         hypre_TFreeF(hh, cogmres_functions);
         hypre_TFreeF(uu, cogmres_functions);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      /* see if we are already converged and
         should print the final norm and exit */

      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (!rel_change) /* shouldn't exit after no iterations if
                           * relative change is on*/
         {
            (*(cogmres_functions->CopyVector))(b, r);
            (*(cogmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
            r_norm = hypre_sqrt((*(cogmres_functions->InnerProd))(r, r));
            if (r_norm  <= epsilon)
            {
               if ( print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               break;
            }
            else if ( print_level > 0 && my_id == 0)
            {
               hypre_printf("false convergence 1\n");
            }
         }
      }



      t = 1.0 / r_norm;
      (*(cogmres_functions->ScaleVector))(t, p[0]);
      i = 0;
      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim && iter < max_iter)
      {
         i++;
         iter++;
         itmp = (i - 1) * (k_dim + 1);

         (*(cogmres_functions->ClearVector))(r);

         precond(precond_data, A, p[i - 1], r);
         (*(cogmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
         for (j = 0; j < i; j++)
         {
            rv[j]  = 0;
         }

         if (cgs > 1)
         {
            (*(cogmres_functions->MassDotpTwo))((void *) p[i], p[i - 1], p, i, unroll, &hh[itmp], &uu[itmp]);
            for (j = 0; j < i - 1; j++) { uu[j * (k_dim + 1) + i - 1] = uu[itmp + j]; }
            for (j = 0; j < i; j++) { rv[j] = hh[itmp + j]; }
            for (k = 0; k < i; k++)
            {
               for (j = 0; j < i; j++)
               {
                  hh[itmp + j] -= (uu[k * (k_dim + 1) + j] * rv[j]);
               }
            }
            for (j = 0; j < i; j++)
            {
               hh[itmp + j]  = -rv[j] - hh[itmp + j];
            }
         }
         else
         {
            (*(cogmres_functions->MassInnerProd))((void *) p[i], p, i, unroll, &hh[itmp]);
            for (j = 0; j < i; j++)
            {
               hh[itmp + j]  = -hh[itmp + j];
            }
         }

         (*(cogmres_functions->MassAxpy))(&hh[itmp], p, p[i], i, unroll);
         for (j = 0; j < i; j++)
         {
            hh[itmp + j]  = -hh[itmp + j];
         }
         t = hypre_sqrt( (*(cogmres_functions->InnerProd))(p[i], p[i]) );
         hh[itmp + i] = t;

         if (hh[itmp + i] != 0.0)
         {
            t = 1.0 / t;
            (*(cogmres_functions->ScaleVector))(t, p[i]);
         }
         for (j = 1; j < i; j++)
         {
            t = hh[itmp + j - 1];
            hh[itmp + j - 1] = s[j - 1] * hh[itmp + j] + c[j - 1] * t;
            hh[itmp + j] = -s[j - 1] * t + c[j - 1] * hh[itmp + j];
         }
         t = hh[itmp + i] * hh[itmp + i];
         t += hh[itmp + i - 1] * hh[itmp + i - 1];
         gamma = hypre_sqrt(t);
         if (gamma == 0.0) { gamma = epsmac; }
         c[i - 1] = hh[itmp + i - 1] / gamma;
         s[i - 1] = hh[itmp + i] / gamma;
         rs[i] = -hh[itmp + i] * rs[i - 1];
         rs[i] /=  gamma;
         rs[i - 1] = c[i - 1] * rs[i - 1];
         // determine residual norm
         hh[itmp + i - 1] = s[i - 1] * hh[itmp + i] + c[i - 1] * hh[itmp + i - 1];
         r_norm = hypre_abs(rs[i]);
         if ( print_level > 0 )
         {
            norms[iter] = r_norm;
            if ( print_level > 1 && my_id == 0 )
            {
               if (b_norm > 0.0)
                  hypre_printf("% 5d    %e    %f   %e\n", iter,
                               norms[iter], norms[iter] / norms[iter - 1],
                               norms[iter] / b_norm);
               else
                  hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                               norms[iter] / norms[iter - 1]);
            }
         }
         /*convergence factor tolerance */
         if (cf_tol > 0.0)
         {
            cf_ave_0 = cf_ave_1;
            cf_ave_1 = hypre_pow( r_norm / r_norm_0, 1.0 / (2.0 * iter));

            weight = hypre_abs(cf_ave_1 - cf_ave_0);
            weight = weight / hypre_max(cf_ave_1, cf_ave_0);

            weight = 1.0 - weight;
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
         if (r_norm <= epsilon && iter >= min_iter)
         {
            if (rel_change && !rel_change_passed)
            {
               /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)
                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
               for (k = 0; k < i; k++) /* extra copy of rs so we don't need
                                   to change the later solve */
               {
                  rs_2[k] = rs[k];
               }

               /* solve tri. system*/
               rs_2[i - 1] = rs_2[i - 1] / hh[itmp + i - 1];
               for (k = i - 2; k >= 0; k--)
               {
                  t = 0.0;
                  for (j = k + 1; j < i; j++)
                  {
                     t -= hh[j * (k_dim + 1) + k] * rs_2[j];
                  }
                  t += rs_2[k];
                  rs_2[k] = t / hh[k * (k_dim + 1) + k];
               }
               (*(cogmres_functions->CopyVector))(p[i - 1], w);
               (*(cogmres_functions->ScaleVector))(rs_2[i - 1], w);
               for (j = i - 2; j >= 0; j--)
               {
                  (*(cogmres_functions->Axpy))(rs_2[j], p[j], w);
               }

               (*(cogmres_functions->ClearVector))(r);
               /* find correction (in r) */
               precond(precond_data, A, w, r);
               /* copy current solution (x) to w (don't want to over-write x)*/
               (*(cogmres_functions->CopyVector))(x, w);

               /* add the correction */
               (*(cogmres_functions->Axpy))(1.0, r, w);

               /* now w is the approx solution  - get the norm*/
               x_norm = hypre_sqrt( (*(cogmres_functions->InnerProd))(w, w) );

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {
                  /* now get  x_i - x_i-1 */
                  if (num_rel_change_check)
                  {
                     /* have already checked once so we can avoid another precond.
                        solve */
                     (*(cogmres_functions->CopyVector))(w, r);
                     (*(cogmres_functions->Axpy))(-1.0, w_2, r);
                     /* now r contains x_i - x_i-1*/

                     /* save current soln w in w_2 for next time */
                     (*(cogmres_functions->CopyVector))(w, w_2);
                  }
                  else
                  {
                     /* first time to check rel change*/
                     /* first save current soln w in w_2 for next time */
                     (*(cogmres_functions->CopyVector))(w, w_2);

                     (*(cogmres_functions->ClearVector))(w);
                     (*(cogmres_functions->Axpy))(rs_2[i - 1], p[i - 1], w);
                     (*(cogmres_functions->ClearVector))(r);
                     /* apply the preconditioner */
                     precond(precond_data, A, w, r);
                     /* now r contains x_i - x_i-1 */
                  }
                  /* find the norm of x_i - x_i-1 */
                  w_norm = hypre_sqrt( (*(cogmres_functions->InnerProd))(r, r) );
                  relative_error = w_norm / x_norm;
                  if (relative_error <= r_tol)
                  {
                     rel_change_passed = 1;
                     break;
                  }
               }
               else
               {
                  rel_change_passed = 1;
                  break;
               }
               num_rel_change_check++;
            }
            else /* no relative change */
            {
               break;
            }
         }
      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */
      if (break_value) { break; }

      rs[i - 1] = rs[i - 1] / hh[itmp + i - 1];
      for (k = i - 2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k + 1; j < i; j++)
         {
            t -= hh[j * (k_dim + 1) + k] * rs[j];
         }
         t += rs[k];
         rs[k] = t / hh[k * (k_dim + 1) + k];
      }

      (*(cogmres_functions->CopyVector))(p[i - 1], w);
      (*(cogmres_functions->ScaleVector))(rs[i - 1], w);
      for (j = i - 2; j >= 0; j--)
      {
         (*(cogmres_functions->Axpy))(rs[j], p[j], w);
      }

      (*(cogmres_functions->ClearVector))(r);
      /* find correction (in r) */
      precond(precond_data, A, w, r);

      /* update current solution x (in x) */
      (*(cogmres_functions->Axpy))(1.0, r, x);


      /* check for convergence by evaluating the actual residual */
      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (skip_real_r_check)
         {
            (cogmres_data -> converged) = 1;
            break;
         }

         /* calculate actual residual norm*/
         (*(cogmres_functions->CopyVector))(b, r);
         (*(cogmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         real_r_norm_new = r_norm = hypre_sqrt( (*(cogmres_functions->InnerProd))(r, r) );

         if (r_norm <= epsilon)
         {
            if (rel_change && !rel_change_passed) /* calculate the relative change */
            {
               /* calculate the norm of the solution */
               x_norm = hypre_sqrt( (*(cogmres_functions->InnerProd))(x, x) );

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {
                  (*(cogmres_functions->ClearVector))(w);
                  (*(cogmres_functions->Axpy))(rs[i - 1], p[i - 1], w);
                  (*(cogmres_functions->ClearVector))(r);
                  /* apply the preconditioner */
                  precond(precond_data, A, w, r);
                  /* find the norm of x_i - x_i-1 */
                  w_norm = hypre_sqrt( (*(cogmres_functions->InnerProd))(r, r) );
                  relative_error = w_norm / x_norm;
                  if ( relative_error < r_tol )
                  {
                     (cogmres_data -> converged) = 1;
                     if ( print_level > 1 && my_id == 0 )
                     {
                        hypre_printf("\n\n");
                        hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                     }
                     break;
                  }
               }
               else
               {
                  (cogmres_data -> converged) = 1;
                  if ( print_level > 1 && my_id == 0 )
                  {
                     hypre_printf("\n\n");
                     hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                  }
                  break;
               }
            }
            else /* don't need to check rel. change */
            {
               if ( print_level > 1 && my_id == 0 )
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_data -> converged) = 1;
               break;
            }
         }
         else /* conv. has not occurred, according to true residual */
         {
            /* exit if the real residual norm has not decreased */
            if (real_r_norm_new >= real_r_norm_old)
            {
               if (print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_data -> converged) = 1;
               break;
            }
            /* report discrepancy between real/COGMRES residuals and restart */
            if ( print_level > 0 && my_id == 0)
            {
               hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
            }
            (*(cogmres_functions->CopyVector))(r, p[0]);
            i = 0;
            real_r_norm_old = real_r_norm_new;
         }
      } /* end of convergence check */

      /* compute residual vector and continue loop */
      for (j = i ; j > 0; j--)
      {
         rs[j - 1] = -s[j - 1] * rs[j];
         rs[j] = c[j - 1] * rs[j];
      }

      if (i) { (*(cogmres_functions->Axpy))(rs[i] - 1.0, p[i], p[i]); }
      for (j = i - 1 ; j > 0; j--)
      {
         (*(cogmres_functions->Axpy))(rs[j], p[j], p[i]);
      }

      if (i)
      {
         (*(cogmres_functions->Axpy))(rs[0] - 1.0, p[0], p[0]);
         (*(cogmres_functions->Axpy))(1.0, p[i], p[0]);
      }

   } /* END of iteration while loop */


   (cogmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
   {
      (cogmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (cogmres_data -> rel_residual_norm) = r_norm;
   }

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) { hypre_error(HYPRE_ERROR_CONV); }

   hypre_TFreeF(c, cogmres_functions);
   hypre_TFreeF(s, cogmres_functions);
   hypre_TFreeF(rs, cogmres_functions);
   hypre_TFreeF(rv, cogmres_functions);
   if (rel_change) { hypre_TFreeF(rs_2, cogmres_functions); }

   /*for (i=0; i < k_dim+1; i++)
   {
      hypre_TFreeF(hh[i],cogmres_functions);
      hypre_TFreeF(uu[i],cogmres_functions);
   }*/
   hypre_TFreeF(hh, cogmres_functions);
   hypre_TFreeF(uu, cogmres_functions);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetKDim, hypre_COGMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetKDim( void   *cogmres_vdata,
                      HYPRE_Int   k_dim )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *) cogmres_vdata;
   (cogmres_data -> k_dim) = k_dim;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetKDim( void   *cogmres_vdata,
                      HYPRE_Int * k_dim )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *k_dim = (cogmres_data -> k_dim);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetUnroll, hypre_COGMRESGetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetUnroll( void   *cogmres_vdata,
                        HYPRE_Int   unroll )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *) cogmres_vdata;
   (cogmres_data -> unroll) = unroll;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetUnroll( void   *cogmres_vdata,
                        HYPRE_Int * unroll )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *unroll = (cogmres_data -> unroll);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetCGS, hypre_COGMRESGetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetCGS( void   *cogmres_vdata,
                     HYPRE_Int   cgs )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *) cogmres_vdata;
   (cogmres_data -> cgs) = cgs;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetCGS( void   *cogmres_vdata,
                     HYPRE_Int * cgs )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *cgs = (cogmres_data -> cgs);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetTol, hypre_COGMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetTol( void   *cogmres_vdata,
                     HYPRE_Real  tol       )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> tol) = tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetTol( void   *cogmres_vdata,
                     HYPRE_Real  * tol      )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *tol = (cogmres_data -> tol);
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSetAbsoluteTol, hypre_COGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetAbsoluteTol( void   *cogmres_vdata,
                             HYPRE_Real  a_tol       )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> a_tol) = a_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetAbsoluteTol( void   *cogmres_vdata,
                             HYPRE_Real  * a_tol      )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *a_tol = (cogmres_data -> a_tol);
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSetConvergenceFactorTol, hypre_COGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetConvergenceFactorTol( void   *cogmres_vdata,
                                      HYPRE_Real  cf_tol       )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> cf_tol) = cf_tol;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetConvergenceFactorTol( void   *cogmres_vdata,
                                      HYPRE_Real * cf_tol       )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *cf_tol = (cogmres_data -> cf_tol);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetMinIter, hypre_COGMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetMinIter( void *cogmres_vdata,
                         HYPRE_Int   min_iter  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> min_iter) = min_iter;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetMinIter( void *cogmres_vdata,
                         HYPRE_Int * min_iter  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *min_iter = (cogmres_data -> min_iter);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetMaxIter, hypre_COGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetMaxIter( void *cogmres_vdata,
                         HYPRE_Int   max_iter  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> max_iter) = max_iter;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetMaxIter( void *cogmres_vdata,
                         HYPRE_Int * max_iter  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *max_iter = (cogmres_data -> max_iter);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetRelChange, hypre_COGMRESGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetRelChange( void *cogmres_vdata,
                           HYPRE_Int   rel_change  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> rel_change) = rel_change;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetRelChange( void *cogmres_vdata,
                           HYPRE_Int * rel_change  )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *rel_change = (cogmres_data -> rel_change);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetSkipRealResidualCheck, hypre_COGMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetSkipRealResidualCheck( void *cogmres_vdata,
                                       HYPRE_Int skip_real_r_check )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> skip_real_r_check) = skip_real_r_check;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetSkipRealResidualCheck( void *cogmres_vdata,
                                       HYPRE_Int *skip_real_r_check)
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *skip_real_r_check = (cogmres_data -> skip_real_r_check);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetPrecond( void  *cogmres_vdata,
                         HYPRE_Int  (*precond)(void*, void*, void*, void*),
                         HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                         void  *precond_data )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   (cogmres_functions -> precond)        = precond;
   (cogmres_functions -> precond_setup)  = precond_setup;
   (cogmres_data -> precond_data)   = precond_data;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESGetPrecond( void         *cogmres_vdata,
                         HYPRE_Solver *precond_data_ptr )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *precond_data_ptr = (HYPRE_Solver)(cogmres_data -> precond_data);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetPrintLevel, hypre_COGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetPrintLevel( void *cogmres_vdata,
                            HYPRE_Int   level)
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> print_level) = level;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetPrintLevel( void *cogmres_vdata,
                            HYPRE_Int * level)
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *level = (cogmres_data -> print_level);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetLogging, hypre_COGMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSetLogging( void *cogmres_vdata,
                         HYPRE_Int   level)
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   (cogmres_data -> logging) = level;
   return hypre_error_flag;
}

HYPRE_Int
hypre_COGMRESGetLogging( void *cogmres_vdata,
                         HYPRE_Int * level)
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *level = (cogmres_data -> logging);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESGetNumIterations( void *cogmres_vdata,
                               HYPRE_Int  *num_iterations )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *num_iterations = (cogmres_data -> num_iterations);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESGetConverged( void *cogmres_vdata,
                           HYPRE_Int  *converged )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *converged = (cogmres_data -> converged);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESGetFinalRelativeResidualNorm( void   *cogmres_vdata,
                                           HYPRE_Real *relative_residual_norm )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   *relative_residual_norm = (cogmres_data -> rel_residual_norm);
   return hypre_error_flag;
}


HYPRE_Int
hypre_COGMRESSetModifyPC(void *cogmres_vdata,
                         HYPRE_Int (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm))
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   (cogmres_functions -> modify_pc)        = modify_pc;
   return hypre_error_flag;
}
