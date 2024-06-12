/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Preconditioned conjugate gradient (Omin) functions
 *
 *****************************************************************************/

/* This was based on the pcg.c formerly in struct_ls, with
   changes (GetPrecond and stop_crit) for compatibility with the pcg.c
   in parcsr_ls and elsewhere.  Incompatibilities with the
   parcsr_ls version:
   - logging is different; no attempt has been made to be the same
   - treatment of b=0 in Ax=b is different: this returns x=0; the parcsr
   version iterates with a special stopping criterion
*/

#include "krylov.h"
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_PCGFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_PCGFunctions *
hypre_PCGFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
   HYPRE_Int    (*Free)          ( void *ptr ),
   HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                   HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                   void *x, HYPRE_Complex beta, void *y ),
   HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   hypre_PCGFunctions * pcg_functions;
   pcg_functions = (hypre_PCGFunctions *)
                   CAlloc( 1, sizeof(hypre_PCGFunctions), HYPRE_MEMORY_HOST );

   pcg_functions->CAlloc = CAlloc;
   pcg_functions->Free = Free;
   pcg_functions->CommInfo = CommInfo;
   pcg_functions->CreateVector = CreateVector;
   pcg_functions->DestroyVector = DestroyVector;
   pcg_functions->MatvecCreate = MatvecCreate;
   pcg_functions->Matvec = Matvec;
   pcg_functions->MatvecDestroy = MatvecDestroy;
   pcg_functions->InnerProd = InnerProd;
   pcg_functions->CopyVector = CopyVector;
   pcg_functions->ClearVector = ClearVector;
   pcg_functions->ScaleVector = ScaleVector;
   pcg_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   pcg_functions->precond_setup = PrecondSetup;
   pcg_functions->precond       = Precond;

   return pcg_functions;
}

/*--------------------------------------------------------------------------
 * hypre_PCGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PCGCreate( hypre_PCGFunctions *pcg_functions )
{
   hypre_PCGData *pcg_data;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   pcg_data = hypre_CTAllocF(hypre_PCGData, 1, pcg_functions, HYPRE_MEMORY_HOST);

   pcg_data -> functions = pcg_functions;

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> atolf)        = 0.0;
   (pcg_data -> cf_tol)       = 0.0;
   (pcg_data -> a_tol)        = 0.0;
   (pcg_data -> rtol)         = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> recompute_residual) = 0;
   (pcg_data -> recompute_residual_p) = 0;
   (pcg_data -> stop_crit)    = 0;
   (pcg_data -> skip_break)   = 0;
   (pcg_data -> converged)    = 0;
   (pcg_data -> hybrid)       = 0;
   (pcg_data -> owns_matvec_data ) = 1;
   (pcg_data -> matvec_data)  = NULL;
   (pcg_data -> precond_data) = NULL;
   (pcg_data -> print_level)  = 0;
   (pcg_data -> logging)      = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;
   (pcg_data -> p)            = NULL;
   (pcg_data -> s)            = NULL;
   (pcg_data -> r)            = NULL;
   (pcg_data -> r_old)        = NULL;
   (pcg_data -> v)            = NULL;

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) pcg_data;
}

/*--------------------------------------------------------------------------
 * hypre_PCGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGDestroy( void *pcg_vdata )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (pcg_data)
   {
      hypre_PCGFunctions *pcg_functions = pcg_data->functions;
      if ( (pcg_data -> norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> norms, pcg_functions );
         pcg_data -> norms = NULL;
      }
      if ( (pcg_data -> rel_norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
         pcg_data -> rel_norms = NULL;
      }
      if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
      {
         (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
         pcg_data -> matvec_data = NULL;
      }
      if ( pcg_data -> p != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> p);
         pcg_data -> p = NULL;
      }
      if ( pcg_data -> s != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> s);
         pcg_data -> s = NULL;
      }
      if ( pcg_data -> r != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r);
         pcg_data -> r = NULL;
      }
      if ( pcg_data -> r_old != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r_old);
         pcg_data -> r_old = NULL;
      }
      if ( pcg_data -> v != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> v);
         pcg_data -> v = NULL;
      }
      hypre_TFreeF( pcg_data, pcg_functions );
      hypre_TFreeF( pcg_functions, pcg_functions );
   }

   HYPRE_ANNOTATE_FUNC_END;

   return (hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_PCGGetResidual( void *pcg_vdata, void **residual )
{
   /* returns a pointer to the residual vector */

   hypre_PCGData  *pcg_data     =  (hypre_PCGData *)pcg_vdata;
   *residual = pcg_data->r;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetup( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_PCGData *pcg_data =  (hypre_PCGData *)pcg_vdata;
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;
   HYPRE_Int            max_iter         = (pcg_data -> max_iter);
   HYPRE_Int            recompute_residual_p = (pcg_data -> recompute_residual_p);
   HYPRE_Real           rtol = (pcg_data -> rtol);
   HYPRE_Int            two_norm = (pcg_data -> two_norm);
   HYPRE_Int            flex = (pcg_data -> flex);
   HYPRE_Int          (*precond_setup)(void*, void*, void*, void*) = (pcg_functions -> precond_setup);
   void          *precond_data     = (pcg_data -> precond_data);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (pcg_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ( pcg_data -> p != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> p);
   }
   (pcg_data -> p) = (*(pcg_functions->CreateVector))(x);

   if ( pcg_data -> s != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> s);
   }
   (pcg_data -> s) = (*(pcg_functions->CreateVector))(x);

   if ( pcg_data -> r != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> r);
   }
   (pcg_data -> r) = (*(pcg_functions->CreateVector))(b);

   if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
   {
      (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
   }
   (pcg_data -> matvec_data) = (*(pcg_functions->MatvecCreate))(A, x);

   if (flex)
   {
      if ( pcg_data -> v != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r_old);
      }
      (pcg_data -> r_old) = (*(pcg_functions->CreateVector))(b);
   }

   if (rtol && recompute_residual_p && (!two_norm))
   {
      if ( pcg_data -> v != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> v);
      }
      (pcg_data -> v) = (*(pcg_functions->CreateVector))(b);
   }

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (pcg_data->logging) > 0  || (pcg_data->print_level) > 0 )
   {
      if ( (pcg_data -> norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> norms, pcg_functions );
      }
      (pcg_data -> norms)     = hypre_CTAllocF( HYPRE_Real, max_iter + 1,
                                                pcg_functions, HYPRE_MEMORY_HOST);

      if ( (pcg_data -> rel_norms) != NULL )
      {
         hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
      }
      (pcg_data -> rel_norms) = hypre_CTAllocF( HYPRE_Real, max_iter + 1,
                                                pcg_functions, HYPRE_MEMORY_HOST );
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSolve
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSolve( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   hypre_PCGData  *pcg_data     =  (hypre_PCGData *)pcg_vdata;
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;

   HYPRE_Real      r_tol        = (pcg_data -> tol);
   HYPRE_Real      a_tol        = (pcg_data -> a_tol);
   HYPRE_Real      atolf        = (pcg_data -> atolf);
   HYPRE_Real      cf_tol       = (pcg_data -> cf_tol);
   HYPRE_Real      rtol         = (pcg_data -> rtol);
   HYPRE_Int       max_iter     = (pcg_data -> max_iter);
   HYPRE_Int       two_norm     = (pcg_data -> two_norm);
   HYPRE_Int       rel_change   = (pcg_data -> rel_change);
   HYPRE_Int       recompute_residual   = (pcg_data -> recompute_residual);
   HYPRE_Int       recompute_residual_p = (pcg_data -> recompute_residual_p);
   HYPRE_Int       stop_crit    = (pcg_data -> stop_crit);
   HYPRE_Int       hybrid       = (pcg_data -> hybrid);
   HYPRE_Int       skip_break   = (pcg_data -> skip_break);
   HYPRE_Int       flex    = (pcg_data -> flex);
   /* flex replaces the generally used Fletcher-Reeves method for the
    * parameter used to update the direction vector p, beta=<r,Cr>/<r_old,Cr_old>
    * with the Polak-Ribiere method, which is more flexible, can be more stable,
    * and llows varying preconditioners, but requires an extra dot product
    * beta = <r-r_old, Cr>/<r_old, Cr_old> */
   /*
      HYPRE_Int             converged    = (pcg_data -> converged);
   */
   void           *p            = (pcg_data -> p);
   void           *s            = (pcg_data -> s);
   void           *r            = (pcg_data -> r);
   void           *r_old        = (pcg_data -> r_old);
   void           *v            = (pcg_data -> v);
   void           *matvec_data  = (pcg_data -> matvec_data);
   HYPRE_Int     (*precond)(void*, void*, void*, void*)   = (pcg_functions -> precond);
   void           *precond_data = (pcg_data -> precond_data);
   HYPRE_Int       print_level  = (pcg_data -> print_level);
   HYPRE_Int       logging      = (pcg_data -> logging);
   HYPRE_Real     *norms        = (pcg_data -> norms);
   HYPRE_Real     *rel_norms    = (pcg_data -> rel_norms);

   HYPRE_Real      alpha, beta;
   HYPRE_Real      delta = 0.0;
   HYPRE_Real      gamma, gamma_old;
   HYPRE_Real      bi_prod, eps;
   HYPRE_Real      pi_prod, xi_prod;
   HYPRE_Real      ieee_check = 0.;

   HYPRE_Real      i_prod = 0.0;
   HYPRE_Real      i_prod_0 = 0.0;
   HYPRE_Real      cf_ave_0 = 0.0;
   HYPRE_Real      cf_ave_1 = 0.0;
   HYPRE_Real      weight;
   HYPRE_Real      ratio;

   HYPRE_Real      guard_zero_residual, sdotp;
   HYPRE_Int       tentatively_converged = 0;
   HYPRE_Int       recompute_true_residual = 0;

   HYPRE_Int       i = 0;
   HYPRE_Int       my_id, num_procs;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (pcg_data -> converged) = 0;

   (*(pcg_functions->CommInfo))(A, &my_id, &num_procs);

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = (*(pcg_functions->InnerProd))(b, b);
      if (print_level > 1 && my_id == 0)
      {
         hypre_printf("<b,b>: %e\n", bi_prod);
      }
   }
   else
   {
      /* bi_prod = <C*b,b> */
      (*(pcg_functions->ClearVector))(p);
      precond(precond_data, A, b, p);
      bi_prod = (*(pcg_functions->InnerProd))(p, b);
      if (print_level > 1 && my_id == 0)
      {
         hypre_printf("<C*b,b>: %e\n", bi_prod);
      }
   };

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) { ieee_check = bi_prod / bi_prod; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         hypre_printf("ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied b.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   eps = r_tol * r_tol; /* note: this may be re-assigned below */
   if ( bi_prod > 0.0 )
   {
      if ( stop_crit && !rel_change && atolf <= 0 ) /* pure absolute tolerance */
      {
         eps = eps / bi_prod;
         /* Note: this section is obsolete.  Aside from backwards comatability
            concerns, we could delete the stop_crit parameter and related code,
            using tol & atolf instead. */
      }
      else if ( atolf > 0 ) /* mixed relative and absolute tolerance */
      {
         bi_prod += atolf;
      }
      else /* DEFAULT (stop_crit and atolf exist for backwards compatibilty
              and are not in the reference manual) */
      {
         /* convergence criteria:  <C*r,r>  <= max( a_tol^2, r_tol^2 * <C*b,b> )
             note: default for a_tol is 0.0, so relative residual criteria is used unless
             user specifies a_tol, or sets r_tol = 0.0, which means absolute
             tol only is checked  */
         eps = hypre_max(r_tol * r_tol, a_tol * a_tol / bi_prod);

      }
   }
   else    /* bi_prod==0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      (*(pcg_functions->CopyVector))(b, x);
      if (logging > 0 || print_level > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
      /* In this case, for the original parcsr pcg, the code would take special
         action to force iterations even though the exact value was known. */
   };

   /* r = b - Ax */
   (*(pcg_functions->CopyVector))(b, r);

   (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);

   //hypre_ParVectorUpdateHost(r);
   /* p = C*r */
   (*(pcg_functions->ClearVector))(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> = <r,Cr> */
   gamma = (*(pcg_functions->InnerProd))(r, p);

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) { ieee_check = gamma / gamma; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         hypre_printf("ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* Set initial residual norm */
   if ( logging > 0 || print_level > 0 || cf_tol > 0.0 )
   {
      if (two_norm)
      {
         i_prod_0 = (*(pcg_functions->InnerProd))(r, r);
      }
      else
      {
         i_prod_0 = gamma;
      }

      if ( logging > 0 || print_level > 0 ) { norms[0] = hypre_sqrt(i_prod_0); }
   }
   if ( print_level > 1 && my_id == 0 )
   {
      hypre_printf("\n\n");
      if (two_norm)
      {
         if ( stop_crit && !rel_change && atolf == 0 ) /* pure absolute tolerance */
         {
            hypre_printf("Iters       ||r||_2     conv.rate\n");
            hypre_printf("-----    ------------   ---------\n");
         }
         else
         {
            hypre_printf("Iters       ||r||_2     conv.rate  ||r||_2/||b||_2\n");
            hypre_printf("-----    ------------   ---------  ------------ \n");
         }
      }
      else  /* !two_norm */
      {
         hypre_printf("Iters       ||r||_C     conv.rate  ||r||_C/||b||_C\n");
         hypre_printf("-----    ------------    ---------  ------------ \n");
      }
      /* hypre_printf("% 5d    %e\n", i, norms[i]); */
   }

   while ((i + 1) <= max_iter)
   {
      /*--------------------------------------------------------------------
       * the core CG calculations...
       *--------------------------------------------------------------------*/
      i++;

      /* At user request, periodically recompute the residual from the formula
         r = b - A x (instead of using the recursive definition). Note that this
         is potentially expensive and can lead to degraded convergence (since it
         essentially a "restarted CG"). */
      recompute_true_residual = recompute_residual_p && !(i % recompute_residual_p);

      /* s = A*p */
      (*(pcg_functions->Matvec))(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      sdotp = (*(pcg_functions->InnerProd))(s, p);
      if ( sdotp == 0.0 )
      {
         hypre_error_w_msg(HYPRE_ERROR_CONV, "Zero sdotp value in PCG");
         if (i == 1) { i_prod = i_prod_0; }
         break;
      }
      alpha = gamma / sdotp;
      /* alpha should always be greater zero for spd A, spd precond. and nonzero p, r*/
      /* alpha = <r, Cr> / <p, Ap> */
      /* for alpha close to 0 x and r will not change much unless x and/or r are close to 0 */
      if (alpha <= 0.0)
      {
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("alpha %e", alpha);
         }
         hypre_error_w_msg(HYPRE_ERROR_CONV, "Negative or zero alpha value in PCG");
         if (skip_break < 3)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }
      else if (! (alpha  >= HYPRE_REAL_TRUE_MIN))
      {
         hypre_error_w_msg(HYPRE_ERROR_CONV, "alpha value less than TRUE_MIN in PCG");
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("alpha %e", alpha);
         }
         if (skip_break < 2)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }
      else if (! (alpha  >= HYPRE_REAL_MIN))
      {
         hypre_error_w_msg(HYPRE_ERROR_CONV, "Subnormal alpha value in PCG");
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("alpha %e", alpha);
         }
         if (skip_break < 1)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }

      gamma_old = gamma;

      /* x = x + alpha*p */
      (*(pcg_functions->Axpy))(alpha, p, x);

      if (flex)
      {
         (*(pcg_functions->CopyVector))(r, r_old); /*save old residual */
      }

      /* r = r - alpha*s */
      if ( !recompute_true_residual )
      {
         (*(pcg_functions->Axpy))(-alpha, s, r);
      }
      else
      {
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("Recomputing the residual...\n");
         }
         (*(pcg_functions->CopyVector))(r, s); /*save old residual */
         if (flex)
         {
            (*(pcg_functions->CopyVector))(r, r_old); /*save old residual */
         }
         (*(pcg_functions->CopyVector))(b, r);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         if (rtol)
         {
            /* compute s = r_old-r_new */
            (*(pcg_functions->Axpy))(-1.0, s, r);
            if (two_norm)
               /* residual-based stopping criteria: ||r_new-r_old|| < rtol ||b|| */
            {
               HYPRE_Real drob2 = (*(pcg_functions->InnerProd))(s, s) / bi_prod;
               if ( drob2 < rtol * rtol )
               {
                  if (print_level > 1 && my_id == 0)
                  {
                     hypre_printf("\n\n||r_old-r_new||/||b||: %e\n", hypre_sqrt(drob2));
                  }
                  break;
               }
            }
            else
               /* residual-based stopping criteria: ||r_new-r_old||_C < rtol ||b||_C */
            {
               HYPRE_Real r2ob2;
               /* v = C*s = C*(r_old-r_new) */
               (*(pcg_functions->ClearVector))(v);
               precond(precond_data, A, s, v);
               /* <s,v> */
               r2ob2 = (*(pcg_functions->InnerProd))(s, v) / bi_prod;
               if ( r2ob2 < rtol * rtol )
               {
                  if (print_level > 1 && my_id == 0)
                  {
                     hypre_printf("\n\n||r_old-r_new||_C/||b||_C: %e\n", hypre_sqrt(r2ob2));
                  }
                  break;
               }
            }
         }
      }

      if (rtol && two_norm)
      {
         if (!recompute_true_residual)
         {
            /* use that r_new-r_old = alpha * s */
            HYPRE_Real drob2 = alpha * alpha * (*(pcg_functions->InnerProd))(s, s) / bi_prod;
            if ( drob2 < rtol * rtol )
            {
               if (print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n||r_old-r_new||/||b||: %e\n", hypre_sqrt(drob2));
               }
               break;
            }
         }
      }

      /* s = C*r */
      (*(pcg_functions->ClearVector))(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = (*(pcg_functions->InnerProd))(r, s);
      if (flex)
      {
         delta = gamma - (*(pcg_functions->InnerProd))(r_old, s);
      }

      /* residual-based stopping criteria: ||r_new-r_old||_C < rtol ||b||_C */
      if (rtol && !two_norm)
      {
         if (!recompute_true_residual)
         {
            /* The following assumes that residuals are C-orthogonal: */
            /* use that ||r_new-r_old||_C^2 = (r_new ,C r_new) + (r_old, C r_old) */
            HYPRE_Real r2ob2 = (gamma + gamma_old) / bi_prod;
            if ( r2ob2 < rtol * rtol)
            {
               if (print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n||r_old-r_new||_C/||b||_C: %e\n", hypre_sqrt(r2ob2));
               }
               break;
            }
         }
      }

      /* set i_prod for convergence test */
      if (two_norm)
      {
         i_prod = (*(pcg_functions->InnerProd))(r, r);
      }
      else
      {
         i_prod = gamma;
      }

      /*--------------------------------------------------------------------
       * optional output
       *--------------------------------------------------------------------*/
#if 0
      if (two_norm)
         hypre_printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                      i, hypre_sqrt(i_prod), (bi_prod ? hypre_sqrt(i_prod / bi_prod) : 0));
      else
         hypre_printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                      i, hypre_sqrt(i_prod), (bi_prod ? hypre_sqrt(i_prod / bi_prod) : 0));
#endif

      /* print norm info */
      if ( logging > 0 || print_level > 0 )
      {
         norms[i]     = hypre_sqrt(i_prod);
         rel_norms[i] = bi_prod ? hypre_sqrt(i_prod / bi_prod) : 0;
      }
      if ( print_level > 1 && my_id == 0 )
      {
         if (two_norm)
         {
            if ( stop_crit && !rel_change && atolf == 0 )  /* pure absolute tolerance */
            {
               hypre_printf("% 5d    %e    %f\n", i, norms[i],
                            norms[i] / norms[i - 1] );
            }
            else
            {
               hypre_printf("% 5d    %e    %f    %e\n", i, norms[i],
                            norms[i] / norms[i - 1], rel_norms[i] );
            }
         }
         else
         {
            hypre_printf("% 5d    %e    %f    %e\n", i, norms[i],
                         norms[i] / norms[i - 1], rel_norms[i] );
         }
      }


      /*--------------------------------------------------------------------
       * check for convergence
       *--------------------------------------------------------------------*/
      if (i_prod / bi_prod < eps)  /* the basic convergence test */
      {
         tentatively_converged = 1;
      }
      if ( tentatively_converged && recompute_residual )
         /* At user request, don't trust the convergence test until we've recomputed
            the residual from scratch.  This is expensive in the usual case where
            the norm is the energy norm.
            This calculation is coded on the assumption that r's accuracy is only a
            concern for problems where CG takes many iterations. */
      {
         /* r = b - Ax */
         (*(pcg_functions->CopyVector))(b, r);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);

         /* set i_prod for convergence test */
         if (two_norm)
         {
            i_prod = (*(pcg_functions->InnerProd))(r, r);
         }
         else
         {
            /* s = C*r */
            (*(pcg_functions->ClearVector))(s);
            precond(precond_data, A, r, s);
            /* iprod = gamma = <r,s> */
            i_prod = (*(pcg_functions->InnerProd))(r, s);
            gamma = i_prod;
         }
         if (i_prod / bi_prod >= eps) { tentatively_converged = 0; }
      }
      if ( tentatively_converged && rel_change && (i_prod > guard_zero_residual ))
         /* At user request, don't treat this as converged unless x didn't change
            much in the last iteration. */
      {
         pi_prod = (*(pcg_functions->InnerProd))(p, p);
         xi_prod = (*(pcg_functions->InnerProd))(x, x);
         ratio = alpha * alpha * pi_prod / xi_prod;
         if (ratio >= eps) { tentatively_converged = 0; }
      }
      if ( tentatively_converged )
         /* we've passed all the convergence tests, it's for real */
      {
         (pcg_data -> converged) = 1;
         break;
      }
      /* gamma should generally be greater than 0 for spd prec and nonzero r */
      if (gamma <= 0.0)
      {
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("gamma %e", gamma);
         }
         hypre_error_w_msg(HYPRE_ERROR_CONV, "Negative or zero gamma value in PCG");
         if (skip_break < 3)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }
      else if (! (gamma  >= HYPRE_REAL_TRUE_MIN))
      {
         hypre_error_w_msg(HYPRE_ERROR_CONV, "gamma value less than TRUE_MIN in PCG");
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("gamma %e", gamma);
         }
         if (skip_break < 2)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }
      else if (! (gamma  >= HYPRE_REAL_MIN))
      {
         hypre_error_w_msg(HYPRE_ERROR_CONV, "Subnormal gamma value in PCG");
         if (print_level > 1 && my_id == 0)
         {
            hypre_printf("gamma %e", gamma);
         }
         if (skip_break < 1)
         {
            if (i == 1) { i_prod = i_prod_0; }
            break;
         }
      }
      /* ... gamma should be >=0.  IEEE subnormal numbers are < 2**(-1022)=2.2e-308
         (and >= 2**(-1074)=4.9e-324).  So a gamma this small means we're getting
         dangerously close to subnormal or zero numbers (usually if gamma is small,
         so will be other variables).  Thus further calculations risk a crash.
         Such small gamma generally means no hope of progress anyway. */

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
         /* i_prod_0 is zero, or (almost) subnormal, yet i_prod wasn't small
            enough to pass the convergence test.  Therefore initial guess was good,
            and we're just calculating garbage - time to bail out before the
            next step, which will be a divide by zero (or close to it). */
         if (i_prod_0 <= 0.0)
         {
            if (print_level > 1 && my_id == 0)
            {
               hypre_printf("i_prod_0 %e", i_prod_0);
            }
            hypre_error_w_msg(HYPRE_ERROR_CONV, "Negative or zero i_prod_0 value in PCG");
            if (skip_break < 3)
            {
               break;
            }
         }
         else if (! (i_prod_0  >= HYPRE_REAL_TRUE_MIN))
         {
            hypre_error_w_msg(HYPRE_ERROR_CONV, "i_prod_0 value less than TRUE_MIN in PCG");
            if (print_level > 1 && my_id == 0)
            {
               hypre_printf("i_prod_0 %e", i_prod_0);
            }
            if (skip_break < 2)
            {
               break;
            }
         }
         else if (! (i_prod_0  >= HYPRE_REAL_MIN))
         {
            hypre_error_w_msg(HYPRE_ERROR_CONV, "Subnormal i_prod_0 value in PCG");
            if (print_level > 1 && my_id == 0)
            {
               hypre_printf("i_prod_0 %e", i_prod_0);
            }
            if (skip_break < 1)
            {
               break;
            }
         }
         cf_ave_1 = hypre_pow( i_prod / i_prod_0, 1.0 / (2.0 * i) );

         weight   = hypre_abs(cf_ave_1 - cf_ave_0);
         weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
#if 0
         hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                      i, cf_ave_1, cf_ave_0, weight );
#endif
         if (weight * cf_ave_1 > cf_tol) { break; }
      }

      /*--------------------------------------------------------------------
       * back to the core CG calculations
       *--------------------------------------------------------------------*/

      /* beta = gamma / gamma_old */
      if (!flex)
      {
         beta = gamma / gamma_old;
      }
      else
      {
         beta = delta / gamma_old;
      }

      /* p = s + beta p */
      if ( !recompute_true_residual )
      {
         (*(pcg_functions->ScaleVector))(beta, p);
         (*(pcg_functions->Axpy))(1.0, s, p);
      }
      else
      {
         (*(pcg_functions->CopyVector))(s, p);
      }
   }

   /*--------------------------------------------------------------------
    * Finish up with some outputs.
    *--------------------------------------------------------------------*/

   if ( print_level > 1 && my_id == 0 )
   {
      hypre_printf("\n\n");
   }

   if (i >= max_iter && (i_prod / bi_prod) >= eps && eps > 0 && hybrid != -1)
   {
      char msg[1024];
      hypre_sprintf(msg, "Reached max iterations %d in PCG before convergence", max_iter);
      hypre_error_w_msg(HYPRE_ERROR_CONV, msg);
   }

   (pcg_data -> num_iterations) = i;
   if (bi_prod > 0.0)
   {
      (pcg_data -> rel_residual_norm) = hypre_sqrt(i_prod / bi_prod);
   }
   else /* actually, we'll never get here... */
   {
      (pcg_data -> rel_residual_norm) = 0.0;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetTol, hypre_PCGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetTol( void   *pcg_vdata,
                 HYPRE_Real  tol       )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> tol) = tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetTol( void   *pcg_vdata,
                 HYPRE_Real * tol       )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *tol = (pcg_data -> tol);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_PCGSetAbsoluteTol, hypre_PCGGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetAbsoluteTol( void   *pcg_vdata,
                         HYPRE_Real  a_tol       )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> a_tol) = a_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetAbsoluteTol( void   *pcg_vdata,
                         HYPRE_Real * a_tol       )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *a_tol = (pcg_data -> a_tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetAbsoluteTolFactor, hypre_PCGGetAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetAbsoluteTolFactor( void   *pcg_vdata,
                               HYPRE_Real  atolf   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> atolf) = atolf;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetAbsoluteTolFactor( void   *pcg_vdata,
                               HYPRE_Real  * atolf   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *atolf = (pcg_data -> atolf);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetResidualTol, hypre_PCGGetResidualTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetResidualTol( void   *pcg_vdata,
                         HYPRE_Real  rtol   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> rtol) = rtol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetResidualTol( void   *pcg_vdata,
                         HYPRE_Real  * rtol   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *rtol = (pcg_data -> rtol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetConvergenceFactorTol, hypre_PCGGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetConvergenceFactorTol( void   *pcg_vdata,
                                  HYPRE_Real  cf_tol   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> cf_tol) = cf_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetConvergenceFactorTol( void   *pcg_vdata,
                                  HYPRE_Real * cf_tol   )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *cf_tol = (pcg_data -> cf_tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetMaxIter, hypre_PCGGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetMaxIter( void *pcg_vdata,
                     HYPRE_Int   max_iter  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetMaxIter( void *pcg_vdata,
                     HYPRE_Int * max_iter  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *max_iter = (pcg_data -> max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetTwoNorm, hypre_PCGGetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetTwoNorm( void *pcg_vdata,
                     HYPRE_Int   two_norm  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> two_norm) = two_norm;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetTwoNorm( void *pcg_vdata,
                     HYPRE_Int * two_norm  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *two_norm = (pcg_data -> two_norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetRelChange, hypre_PCGGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetRelChange( void *pcg_vdata,
                       HYPRE_Int   rel_change  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetRelChange( void *pcg_vdata,
                       HYPRE_Int * rel_change  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *rel_change = (pcg_data -> rel_change);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetRecomputeResidual, hypre_PCGGetRecomputeResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetRecomputeResidual( void *pcg_vdata,
                               HYPRE_Int   recompute_residual  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> recompute_residual) = recompute_residual;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetRecomputeResidual( void *pcg_vdata,
                               HYPRE_Int * recompute_residual  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *recompute_residual = (pcg_data -> recompute_residual);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetRecomputeResidualP, hypre_PCGGetRecomputeResidualP
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetRecomputeResidualP( void *pcg_vdata,
                                HYPRE_Int   recompute_residual_p  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> recompute_residual_p) = recompute_residual_p;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetRecomputeResidualP( void *pcg_vdata,
                                HYPRE_Int * recompute_residual_p  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *recompute_residual_p = (pcg_data -> recompute_residual_p);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetStopCrit, hypre_PCGGetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetStopCrit( void *pcg_vdata,
                      HYPRE_Int   stop_crit  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> stop_crit) = stop_crit;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetStopCrit( void *pcg_vdata,
                      HYPRE_Int * stop_crit  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *stop_crit = (pcg_data -> stop_crit);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetSkipBreak, hypre_PCGGetSkipBreak
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetSkipBreak( void *pcg_vdata,
                       HYPRE_Int   skip_break  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> skip_break) = skip_break;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetSkipBreak( void *pcg_vdata,
                       HYPRE_Int * skip_break  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *skip_break = (pcg_data -> skip_break);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetFlex, hypre_PCGGetFlex
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetFlex( void *pcg_vdata,
                  HYPRE_Int   flex  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> flex) = flex;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetFlex( void *pcg_vdata,
                  HYPRE_Int * flex  )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *flex = (pcg_data -> flex);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGGetPrecond( void         *pcg_vdata,
                     HYPRE_Solver *precond_data_ptr )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *precond_data_ptr = (HYPRE_Solver)(pcg_data -> precond_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetPrecond( void  *pcg_vdata,
                     HYPRE_Int  (*precond)(void*, void*, void*, void*),
                     HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                     void  *precond_data )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;

   (pcg_functions -> precond)       = precond;
   (pcg_functions -> precond_setup) = precond_setup;
   (pcg_data -> precond_data)  = precond_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetPreconditioner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetPreconditioner(void *pcg_vdata,
                           void *precond_data )
{
   hypre_PCGData      *pcg_data      = (hypre_PCGData*) pcg_vdata;
   hypre_Solver       *base          = (hypre_Solver*)  precond_data;
   hypre_PCGFunctions *pcg_functions = pcg_data->functions;

   (pcg_data -> precond_data)       = precond_data;

   (pcg_functions -> precond)       = (HYPRE_Int (*)(void*, void*, void*,
                                                     void*)) hypre_SolverSolve(base);
   (pcg_functions -> precond_setup) = (HYPRE_Int (*)(void*, void*, void*,
                                                     void*)) hypre_SolverSetup(base);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetPrintLevel, hypre_PCGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetPrintLevel( void *pcg_vdata,
                        HYPRE_Int   level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   (pcg_data -> print_level) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetPrintLevel( void *pcg_vdata,
                        HYPRE_Int * level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;


   *level = (pcg_data -> print_level);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGSetLogging, hypre_PCGGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGSetLogging( void *pcg_vdata,
                     HYPRE_Int   level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> logging) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGGetLogging( void *pcg_vdata,
                     HYPRE_Int * level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *level = (pcg_data -> logging);

   return hypre_error_flag;
}

HYPRE_Int
hypre_PCGSetHybrid( void *pcg_vdata,
                    HYPRE_Int   level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   (pcg_data -> hybrid) = level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGGetNumIterations( void *pcg_vdata,
                           HYPRE_Int  *num_iterations )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *num_iterations = (pcg_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGGetConverged( void *pcg_vdata,
                       HYPRE_Int  *converged)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   *converged = (pcg_data -> converged);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGPrintLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGPrintLogging( void *pcg_vdata,
                       HYPRE_Int   myid)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   HYPRE_Int            num_iterations  = (pcg_data -> num_iterations);
   HYPRE_Int            print_level     = (pcg_data -> print_level);
   HYPRE_Real    *norms           = (pcg_data -> norms);
   HYPRE_Real    *rel_norms       = (pcg_data -> rel_norms);

   HYPRE_Int            i;

   if (myid == 0)
   {
      if (print_level > 0)
      {
         for (i = 0; i < num_iterations; i++)
         {
            hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
            hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PCGGetFinalRelativeResidualNorm( void   *pcg_vdata,
                                       HYPRE_Real *relative_residual_norm )
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)pcg_vdata;

   HYPRE_Real     rel_residual_norm = (pcg_data -> rel_residual_norm);

   *relative_residual_norm = rel_residual_norm;

   return hypre_error_flag;
}
