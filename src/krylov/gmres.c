/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_GMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_GMRESFunctions *
hypre_GMRESFunctionsCreate(
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
   HYPRE_Int    (*InnerProd)     ( void *x, void *y, HYPRE_Int *num_tags_ptr,
                                   HYPRE_Complex **iprod_ptr ),
   HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   HYPRE_Int    (*ClearVector)   ( void *x ),
   HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
   HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
   HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   hypre_GMRESFunctions * gmres_functions;
   gmres_functions = (hypre_GMRESFunctions *)
                     CAlloc( 1, sizeof(hypre_GMRESFunctions), HYPRE_MEMORY_HOST );

   gmres_functions->CAlloc = CAlloc;
   gmres_functions->Free = Free;
   gmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   gmres_functions->CreateVector = CreateVector;
   gmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   gmres_functions->DestroyVector = DestroyVector;
   gmres_functions->MatvecCreate = MatvecCreate;
   gmres_functions->Matvec = Matvec;
   gmres_functions->MatvecDestroy = MatvecDestroy;
   gmres_functions->InnerProd = InnerProd;
   gmres_functions->CopyVector = CopyVector;
   gmres_functions->ClearVector = ClearVector;
   gmres_functions->ScaleVector = ScaleVector;
   gmres_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   gmres_functions->precond_setup = PrecondSetup;
   gmres_functions->precond       = Precond;

   return gmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESCreate
 *--------------------------------------------------------------------------*/

void *
hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions )
{
   hypre_GMRESData *gmres_data;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   gmres_data = hypre_CTAllocF(hypre_GMRESData, 1, gmres_functions, HYPRE_MEMORY_HOST);
   gmres_data->functions = gmres_functions;

   /* set defaults */
   (gmres_data -> k_dim)          = 5;
   (gmres_data -> tol)            = 1.0e-06; /* relative residual tol */
   (gmres_data -> cf_tol)         = 0.0;
   (gmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (gmres_data -> min_iter)       = 0;
   (gmres_data -> max_iter)       = 1000;
   (gmres_data -> rel_change)     = 0;
   (gmres_data -> skip_real_r_check) = 0;
   (gmres_data -> stop_crit)      = 0; /* rel. residual norm  - this is obsolete!*/
   (gmres_data -> converged)      = 0;
   (gmres_data -> hybrid)         = 0;
   (gmres_data -> precond_data)   = NULL;
   (gmres_data -> print_level)    = 0;
   (gmres_data -> logging)        = 0;
   (gmres_data -> p)              = NULL;
   (gmres_data -> r)              = NULL;
   (gmres_data -> w)              = NULL;
   (gmres_data -> w_2)            = NULL;
   (gmres_data -> w_3)            = NULL;
   (gmres_data -> xref)           = NULL;
   (gmres_data -> matvec_data)    = NULL;
   (gmres_data -> norms)          = NULL;
   (gmres_data -> log_file_name)  = NULL;

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) gmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESDestroy( void *gmres_vdata )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;
   HYPRE_Int i;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   if (gmres_data)
   {
      hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
      if ( (gmres_data -> logging > 0) || (gmres_data->print_level) > 0 )
      {
         if ( (gmres_data -> norms) != NULL )
         {
            hypre_TFreeF( gmres_data -> norms, gmres_functions );
         }
      }

      if ( (gmres_data -> matvec_data) != NULL )
      {
         (*(gmres_functions->MatvecDestroy))(gmres_data -> matvec_data);
      }

      /* Destroy work vectors */
      (*(gmres_functions->DestroyVector))(gmres_data -> r);
      (*(gmres_functions->DestroyVector))(gmres_data -> w);
      (*(gmres_functions->DestroyVector))(gmres_data -> w_2);
      (*(gmres_functions->DestroyVector))(gmres_data -> w_3);

      if ( (gmres_data -> p) != NULL )
      {
         for (i = 0; i < (gmres_data -> k_dim + 1); i++)
         {
            if ( (gmres_data -> p)[i] != NULL )
            {
               (*(gmres_functions->DestroyVector))( (gmres_data -> p) [i]);
            }
         }
         hypre_TFreeF( gmres_data->p, gmres_functions );
      }
      hypre_TFreeF( gmres_data, gmres_functions );
      hypre_TFreeF( gmres_functions, gmres_functions );
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_GMRESGetResidual( void *gmres_vdata, void **residual )
{

   hypre_GMRESData  *gmres_data = (hypre_GMRESData *)gmres_vdata;
   *residual = gmres_data->r;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetup( void *gmres_vdata,
                  void *A,
                  void *b,
                  void *x )
{
   hypre_GMRESData      *gmres_data      = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = (gmres_data -> functions);

   HYPRE_Int             k_dim           = (gmres_data -> k_dim);
   HYPRE_Int             max_iter        = (gmres_data -> max_iter);
   void                 *precond_data    = (gmres_data -> precond_data);
   HYPRE_Int             rel_change      = (gmres_data -> rel_change);

   HYPRE_Int (*precond_setup)(void*, void*, void*, void*) = (gmres_functions->precond_setup);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (gmres_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((gmres_data -> p) == NULL)
   {
      (gmres_data -> p) = (void**)(*(gmres_functions->CreateVectorArray))(k_dim + 1, x);
   }

   if ((gmres_data -> r) == NULL)
   {
      (gmres_data -> r) = (*(gmres_functions->CreateVector))(b);
   }

   if ((gmres_data -> w) == NULL)
   {
      (gmres_data -> w) = (*(gmres_functions->CreateVector))(b);
   }

   if (rel_change)
   {
      if ((gmres_data -> w_2) == NULL)
      {
         (gmres_data -> w_2) = (*(gmres_functions->CreateVector))(b);
      }
   }

   /* Additional work vector for computing individual function (DOF) residuals */
   if ((gmres_data -> print_level) > 2 || (gmres_data -> xref))
   {
      if ((gmres_data -> w_3) == NULL)
      {
         (gmres_data -> w_3) = (*(gmres_functions->CreateVector))(b);
      }
   }

   if ((gmres_data -> matvec_data) == NULL)
   {
      (gmres_data -> matvec_data) = (*(gmres_functions->MatvecCreate))(A, x);
   }

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (gmres_data->logging) > 0 || (gmres_data->print_level) > 0 )
   {
      if ((gmres_data -> norms) != NULL)
      {
         hypre_TFreeF(gmres_data -> norms, gmres_functions);
      }
      (gmres_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1, gmres_functions,
                                             HYPRE_MEMORY_HOST);
   }
   if ( (gmres_data->print_level) > 0 )
   {
      if ((gmres_data -> log_file_name) == NULL)
      {
         (gmres_data -> log_file_name) = (char*)"gmres.out.log";
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSolve(void  *gmres_vdata,
                 void  *A,
                 void  *b,
                 void  *x)
{
   hypre_GMRESData      *gmres_data         = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions    = (gmres_data -> functions);

   HYPRE_Int             k_dim              = (gmres_data -> k_dim);
   HYPRE_Int             min_iter           = (gmres_data -> min_iter);
   HYPRE_Int             max_iter           = (gmres_data -> max_iter);
   HYPRE_Int             rel_change         = (gmres_data -> rel_change);
   HYPRE_Int             skip_real_r_check  = (gmres_data -> skip_real_r_check);
   HYPRE_Int             hybrid             = (gmres_data -> hybrid);
   HYPRE_Real            r_tol              = (gmres_data -> tol);
   HYPRE_Real            cf_tol             = (gmres_data -> cf_tol);
   HYPRE_Real            a_tol              = (gmres_data -> a_tol);
   void                 *matvec_data        = (gmres_data -> matvec_data);
   void                 *r                  = (gmres_data -> r);
   void                 *w                  = (gmres_data -> w);
   void                 *xref               = (gmres_data -> xref);

   /* note:
    *   - w_2 is only allocated if rel_change = 1
    *   - w_3 is only allocated if print_level > 2 or when xref is provided
    */
   void                 *w_2                = (gmres_data -> w_2);
   void                 *w_3                = (gmres_data -> w_3);
   void                **p                  = (gmres_data -> p);

   HYPRE_Int           (*precond)(void*, void*, void*, void*) = (gmres_functions -> precond);
   HYPRE_Int            *precond_data = (HYPRE_Int*) (gmres_data -> precond_data);

   HYPRE_Int             print_level        = (gmres_data -> print_level);
   HYPRE_Int             logging            = (gmres_data -> logging);
   HYPRE_Real           *norms              = (gmres_data -> norms);
   /* not used yet   char           *log_file_name  = (gmres_data -> log_file_name);*/
   /*   FILE           *fp; */

   HYPRE_Int             break_value = 0;
   HYPRE_Int             i, j, k;
   HYPRE_Real           *rs, **hh, *c, *s, *rs_2 = NULL, *rs_3 = NULL;
   HYPRE_Int             tag, num_tags;
   HYPRE_Complex        *iprod = NULL;
   HYPRE_Complex        *biprod = NULL;
   HYPRE_Complex        *xiprod = NULL;
   HYPRE_Int             iter;
   HYPRE_Int             my_id, num_procs;
   HYPRE_Real            epsilon, gamma, t;
   HYPRE_Real            r_norm, b_norm, den_norm, x_norm, w_norm, e_norm = 1.0;

   HYPRE_Real            epsmac = 1.e-16;
   HYPRE_Real            ieee_check = 0.;

   HYPRE_Real            guard_zero_residual;
   HYPRE_Real            cf_ave_0 = 0.0;
   HYPRE_Real            cf_ave_1 = 0.0;
   HYPRE_Real            weight;
   HYPRE_Real            r_norm_0;
   HYPRE_Real            relative_error = 1.0;
   HYPRE_Int             rel_change_passed = 0, num_rel_change_check = 0;
   HYPRE_Real            real_r_norm_old, real_r_norm_new;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   (gmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(A, &my_id, &num_procs);
   if ( logging > 0 || print_level > 0 )
   {
      norms = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(HYPRE_Real, k_dim + 1, gmres_functions, HYPRE_MEMORY_HOST);
   c = hypre_CTAllocF(HYPRE_Real, k_dim, gmres_functions, HYPRE_MEMORY_HOST);
   s = hypre_CTAllocF(HYPRE_Real, k_dim, gmres_functions, HYPRE_MEMORY_HOST);
   if (rel_change)
   {
      rs_2 = hypre_CTAllocF(HYPRE_Real, k_dim + 1, gmres_functions, HYPRE_MEMORY_HOST);
   }
   if (print_level > 2)
   {
      rs_3 = hypre_CTAllocF(HYPRE_Real, k_dim + 1, gmres_functions, HYPRE_MEMORY_HOST);
   }
   hh = hypre_CTAllocF(HYPRE_Real*, k_dim + 1, gmres_functions, HYPRE_MEMORY_HOST);
   for (i = 0; i < k_dim + 1; i++)
   {
      hh[i] = hypre_CTAllocF(HYPRE_Real, k_dim, gmres_functions, HYPRE_MEMORY_HOST);
   }

   (*(gmres_functions->CopyVector))(b, p[0]);

   /* Compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, p[0]);

   (*(gmres_functions->InnerProd))(b, b, &num_tags, &biprod);
   b_norm = hypre_sqrt(biprod[0]);
   real_r_norm_old = b_norm;

   /* Compute initial error when a reference solution is provided */
   if (xref)
   {
      (*(gmres_functions->CopyVector))(x, w_3);
      (*(gmres_functions->Axpy))(-1.0, xref, w_3);
      (*(gmres_functions->InnerProd))(w_3, w_3, &num_tags, &xiprod);
      e_norm = hypre_sqrt(xiprod[0]);
   }

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.)
   {
      ieee_check = b_norm / b_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         hypre_printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied b.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   (*(gmres_functions->InnerProd))(p[0], p[0], &num_tags, &iprod);
   r_norm = hypre_sqrt(iprod[0]);
   r_norm_0 = r_norm;

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.)
   {
      ieee_check = r_norm / r_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         hypre_printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
         hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if (logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if (!my_id && print_level > 0)
      {
         hypre_printf("L2 norm of b: %e\n", b_norm);
         if (num_tags > 1)
         {
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("L2 norm of b%*d: %e\n", hypre_ndigits(num_tags),
                            tag, hypre_sqrt(biprod[tag + 1]));
            }
         }
         if (b_norm == 0.0)
         {
            hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         }

         hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
         if (num_tags > 1)
         {
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("Initial L2 norm of r%*d: %e\n", hypre_ndigits(num_tags),
                            tag, hypre_sqrt(iprod[tag + 1]));
            }
         }

         if (xref)
         {
            hypre_printf("Initial L2 norm of error: %e\n", e_norm);
            if (num_tags > 1)
            {
               for (tag = 0; tag < num_tags; tag++)
               {
                  hypre_printf("Initial L2 norm of e%*d: %e\n", hypre_ndigits(num_tags),
                               tag, hypre_sqrt(xiprod[tag + 1]));
               }
            }
         }
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
   }

   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */

   epsilon = hypre_max(a_tol, r_tol * den_norm);

   /* so now our stop criteria is |r_i| <= epsilon */

   /* Print header for tag-wise residuals if this is the first iteration */
   if (print_level > 1 && my_id == 0)
   {
      hypre_printf("=============================================\n\n");
      if (num_tags <= 1 || print_level == 2 || print_level > 9 || (!xref && print_level > 5))
      {
         if (b_norm > 0.0)
         {
            hypre_printf("Iters      resid.norm     conv.rate   rel.res.norm\n");
            hypre_printf("-----    ------------    ----------   ------------\n");
         }
         else
         {
            hypre_printf("Iters      resid.norm     conv.rate\n");
            hypre_printf("-----    ------------    ----------\n");
         }
      }
      else if (xref && (num_tags <= 1 || print_level == 6))
      {
         if (e_norm > 0.0)
         {
            hypre_printf("Iters      error.norm     conv.rate   rel.err.norm\n");
            hypre_printf("-----    ------------    ----------   ------------\n");
         }
         else
         {
            hypre_printf("Iters      error.norm     conv.rate\n");
            hypre_printf("-----    ------------    ----------\n");
         }
      }
      else if (num_tags > 1)
      {
         hypre_printf("  Iters ");
         if (print_level == 3)
         {
            hypre_printf("            |r|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("           |r%d|_2", tag);
            }
         }
         else if (print_level == 4)
         {
            hypre_printf("      |r|_2/|b|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("    |r%d|_2/|b%d|_2", tag, tag);
            }
         }
         else if (print_level == 5)
         {
            hypre_printf("      |r|_2/|b|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("     |r%d|_2/|b|_2", tag);
            }
         }
         else if (print_level == 7)
         {
            hypre_printf("            |e|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("           |e%d|_2", tag);
            }
         }
         else if (print_level == 8)
         {
            hypre_printf("     |e|_2/|eI|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("   |e%d|_2/|eI%d|_2", tag, tag);
            }
         }
         else if (print_level == 9)
         {
            hypre_printf("     |e|_2/|eI|_2");
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("    |e%d|_2/|eI|_2", tag);
            }
         }
         hypre_printf("\n ------  ");
         for (tag = 0; tag < num_tags + 1; tag++)
         {
            hypre_printf("   ------------- ");
         }
         hypre_printf("\n");
      }
   }

   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   /* outer iteration cycle */
   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */

      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         hypre_TFree(iprod, HYPRE_MEMORY_HOST);
         hypre_TFree(xiprod, HYPRE_MEMORY_HOST);
         hypre_TFree(biprod, HYPRE_MEMORY_HOST);
         hypre_TFreeF(c, gmres_functions);
         hypre_TFreeF(s, gmres_functions);
         hypre_TFreeF(rs, gmres_functions);
         if (rel_change) { hypre_TFreeF(rs_2, gmres_functions); }
         if (print_level > 2) { hypre_TFreeF(rs_3, gmres_functions); }
         for (i = 0; i < k_dim + 1; i++) { hypre_TFreeF(hh[i], gmres_functions); }
         hypre_TFreeF(hh, gmres_functions);

         (gmres_data -> num_iterations) = iter;
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
            (*(gmres_functions->CopyVector))(b, r);
            (*(gmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
            (*(gmres_functions->InnerProd))(r, r, &num_tags, &iprod);
            r_norm = hypre_sqrt(iprod[0]);
            if (r_norm <= epsilon)
            {
               break;
            }
            else
            {
               if (!my_id && print_level > 0)
               {
                  hypre_printf("false convergence 1\n");
               }
            }
         }
      }

      t = 1.0 / r_norm;
      (*(gmres_functions->ScaleVector))(t, p[0]);
      i = 0;

      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim && iter < max_iter)
      {
         i++;
         iter++;
         (*(gmres_functions->ClearVector))(r);
         precond(precond_data, A, p[i - 1], r);
         (*(gmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);

         /* modified Gram_Schmidt */
         for (j = 0; j < i; j++)
         {
            (*(gmres_functions->InnerProd))(p[j], p[i], &num_tags, &iprod);
            hh[j][i - 1] = iprod[0];
            (*(gmres_functions->Axpy))(-hh[j][i - 1], p[j], p[i]);
         }
         (*(gmres_functions->InnerProd))(p[i], p[i], &num_tags, &iprod);
         t = hypre_sqrt(iprod[0]);
         hh[i][i - 1] = t;
         if (t != 0.0)
         {
            t = 1.0 / t;
            (*(gmres_functions->ScaleVector))(t, p[i]);
         }

         /* done with modified Gram_schmidt and Arnoldi step.
            update factorization of hh */
         for (j = 1; j < i; j++)
         {
            t = hh[j - 1][i - 1];
            hh[j - 1][i - 1] = s[j - 1] * hh[j][i - 1] + c[j - 1] * t;
            hh[j][i - 1] = -s[j - 1] * t + c[j - 1] * hh[j][i - 1];
         }
         t = hh[i][i - 1] * hh[i][i - 1];
         t += hh[i - 1][i - 1] * hh[i - 1][i - 1];
         gamma = hypre_sqrt(t);
         if (gamma == 0.0)
         {
            gamma = epsmac;
         }
         c[i - 1] = hh[i - 1][i - 1] / gamma;
         s[i - 1] = hh[i][i - 1] / gamma;
         rs[i] = -hh[i][i - 1] * rs[i - 1];
         rs[i] /=  gamma;
         rs[i - 1] = c[i - 1] * rs[i - 1];

         /* determine residual norm */
         hh[i - 1][i - 1] = s[i - 1] * hh[i][i - 1] + c[i - 1] * hh[i - 1][i - 1];
         r_norm = hypre_abs(rs[i]);

         /* Print residual/error norms? */
         if (print_level > 0)
         {
            norms[iter] = r_norm;
            if ((num_tags <= 1 && print_level > 1) || (print_level == 2) ||
                (print_level > 9) || (!xref && print_level > 5))
            {
               if (!my_id)
               {
                  if (b_norm > 0.0)
                  {
                     hypre_printf("%5d    %e      %f   %e\n", iter,
                                  norms[iter], norms[iter] / norms[iter - 1],
                                  norms[iter] / b_norm);
                  }
                  else
                  {
                     hypre_printf("%5d    %e      %f\n", iter, norms[iter],
                                  norms[iter] / norms[iter - 1]);
                  }
               }
            }
            else if (num_tags > 1 && print_level > 2)
            {
               /* For each tag, we need to compute the residual norm based on the
                  current GMRES state. The residual vector in GMRES is represented
                  by rs[i] and the orthogonal basis vectors p[0...i] */

               /* Copy rs values so we don't modify the original solution */
               for (k = 0; k < i; k++)
               {
                  rs_3[k] = rs[k];
               }

               /* Solve the triangular system */
               for (k = i - 1; k >= 0; k--)
               {
                  rs_3[k] = rs_3[k] / hh[k][k];
                  for (j = 0; j < k; j++)
                  {
                     rs_3[j] -= hh[j][k] * rs_3[k];
                  }
               }

               /* Compute the solution update vector w_3 = sum_{j=0}^{i-1} rs_3[j] p[j] */
               (*(gmres_functions->CopyVector))(p[0], w_3);
               (*(gmres_functions->ScaleVector))(rs_3[0], w_3);
               for (j = 1; j < i; j++)
               {
                  (*(gmres_functions->Axpy))(rs_3[j], p[j], w_3);
               }

               /* Apply preconditioner to get the correction */
               (*(gmres_functions->ClearVector))(w);
               precond(precond_data, A, w_3, w);

               /* Compute current approximate solution x_i = x_0 + correction */
               (*(gmres_functions->CopyVector))(x, w_3);
               (*(gmres_functions->Axpy))(1.0, w, w_3);

               if (xref == NULL || print_level == 3 || print_level == 4 || print_level == 5)
               {
                  /* Now compute real residual r = b - A*x_i */
                  (*(gmres_functions->CopyVector))(b, r);
                  (*(gmres_functions->Matvec))(matvec_data, -1.0, A, w_3, 1.0, r);
               }
               else if (xref != NULL && (print_level == 6 || print_level == 7 || print_level == 8))
               {
                  /* Compute error e = x_ref - x_i */
                  (*(gmres_functions->CopyVector))(xref, r);
                  (*(gmres_functions->Axpy))(-1.0, w_3, r);
               }

               /* Compute tag-specific residual/error norms */
               (*(gmres_functions->InnerProd))(r, r, &num_tags, &iprod);

               /* Print tag-specific residual/error norms */
               if (!my_id && print_level != 6)
               {
                  hypre_printf(" %6d  ", iter);
                  for (tag = 0; tag < num_tags + 1; tag++)
                  {
                     hypre_printf("  %14.6e ",
                                  print_level == 3 || print_level == 7 ?
                                  hypre_sqrt(iprod[tag]) :
                                  print_level == 4 ?
                                  hypre_sqrt(iprod[tag]) / hypre_sqrt(biprod[tag]) :
                                  print_level == 5 ?
                                  hypre_sqrt(iprod[tag]) / hypre_sqrt(biprod[0]) :
                                  print_level == 8 ?
                                  hypre_sqrt(iprod[tag]) / hypre_sqrt(xiprod[tag]) :
                                  print_level == 9 ?
                                  hypre_sqrt(iprod[tag]) / hypre_sqrt(xiprod[0]) :
                                  hypre_sqrt(iprod[tag]));
                  }
                  hypre_printf("\n");
               }
               else if (!my_id && print_level == 6)
               {
                  norms[iter] = hypre_sqrt(iprod[0]);
                  if (e_norm > 0.0)
                  {
                     hypre_printf("%5d    %e      %f   %e\n", iter,
                                  norms[iter], norms[iter] / norms[iter - 1],
                                  norms[iter] / e_norm);
                  }
                  else
                  {
                     hypre_printf("%5d    %e    %f\n", iter, norms[iter],
                                  norms[iter] / norms[iter - 1]);
                  }
               }
            }
         }

         /* convergence factor tolerance */
         if (cf_tol > 0.0)
         {
            cf_ave_0 = cf_ave_1;
            cf_ave_1 = hypre_pow( r_norm / r_norm_0, 1.0 / (2.0 * iter));

            weight   = hypre_abs(cf_ave_1 - cf_ave_0);
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

               for (k = 0; k < i; k++)
               {
                  /* extra copy of rs so we don't need to change the later solve */
                  rs_2[k] = rs[k];
               }

               /* solve tri. system*/
               rs_2[i - 1] = rs_2[i - 1] / hh[i - 1][i - 1];
               for (k = i - 2; k >= 0; k--)
               {
                  t = 0.0;
                  for (j = k + 1; j < i; j++)
                  {
                     t -= hh[k][j] * rs_2[j];
                  }
                  t += rs_2[k];
                  rs_2[k] = t / hh[k][k];
               }

               (*(gmres_functions->CopyVector))(p[i - 1], w);
               (*(gmres_functions->ScaleVector))(rs_2[i - 1], w);
               for (j = i - 2; j >= 0; j--)
               {
                  (*(gmres_functions->Axpy))(rs_2[j], p[j], w);
               }
               (*(gmres_functions->ClearVector))(r);
               /* find correction (in r) */
               precond(precond_data, A, w, r);
               /* copy current solution (x) to w (don't want to over-write x)*/
               (*(gmres_functions->CopyVector))(x, w);

               /* add the correction */
               (*(gmres_functions->Axpy))(1.0, r, w);

               /* now w is the approx solution - get the norm */
               (*(gmres_functions->InnerProd))(w, w, &num_tags, &iprod);
               x_norm = hypre_sqrt(iprod[0]);

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {
                  /* now get x_i - x_i-1 */
                  if (num_rel_change_check)
                  {
                     /* have already checked once so we can avoid another precond.
                        solve */
                     (*(gmres_functions->CopyVector))(w, r);
                     (*(gmres_functions->Axpy))(-1.0, w_2, r);
                     /* now r contains x_i - x_i-1*/

                     /* save current soln w in w_2 for next time */
                     (*(gmres_functions->CopyVector))(w, w_2);
                  }
                  else
                  {
                     /* first time to check rel change*/

                     /* first save current soln w in w_2 for next time */
                     (*(gmres_functions->CopyVector))(w, w_2);

                     /* for relative change take x_(i-1) to be
                        x + M^{-1}[sum{j=0..i-2} rs_j p_j ].
                        Now
                        x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                        - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                        = M^{-1} rs_{i-1}{p_{i-1}} */

                     (*(gmres_functions->ClearVector))(w);
                     (*(gmres_functions->Axpy))(rs_2[i - 1], p[i - 1], w);
                     (*(gmres_functions->ClearVector))(r);
                     /* apply the preconditioner */
                     precond(precond_data, A, w, r);
                     /* now r contains x_i - x_i-1 */
                  }
                  /* find the norm of x_i - x_i-1 */
                  (*(gmres_functions->InnerProd))(r, r, &num_tags, &iprod);
                  w_norm = hypre_sqrt(iprod[0]);
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

      if (break_value)
      {
         break;
      }

      rs[i - 1] = rs[i - 1] / hh[i - 1][i - 1];
      for (k = i - 2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k + 1; j < i; j++)
         {
            t -= hh[k][j] * rs[j];
         }
         t += rs[k];
         rs[k] = t / hh[k][k];
      }

      (*(gmres_functions->CopyVector))(p[i - 1], w);
      (*(gmres_functions->ScaleVector))(rs[i - 1], w);
      for (j = i - 2; j >= 0; j--)
      {
         (*(gmres_functions->Axpy))(rs[j], p[j], w);
      }

      (*(gmres_functions->ClearVector))(r);
      /* find correction (in r) */
      precond(precond_data, A, w, r);

      /* update current solution x (in x) */
      (*(gmres_functions->Axpy))(1.0, r, x);

      /* check for convergence by evaluating the actual residual */
      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (skip_real_r_check)
         {
            (gmres_data -> converged) = 1;
            break;
         }

         /* calculate actual residual norm*/
         (*(gmres_functions->CopyVector))(b, r);
         (*(gmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         (*(gmres_functions->InnerProd))(r, r, &num_tags, &iprod);
         real_r_norm_new = r_norm = hypre_sqrt(iprod[0]);

         if (r_norm <= epsilon)
         {
            if (rel_change && !rel_change_passed) /* calculate the relative change */
            {
               /* calculate the norm of the solution */
               (*(gmres_functions->InnerProd))(x, x, &num_tags, &iprod);
               x_norm = hypre_sqrt(iprod[0]);

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {
                  /* for relative change take x_(i-1) to be
                     x + M^{-1}[sum{j=0..i-2} rs_j p_j ].
                     Now
                     x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                     - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                     = M^{-1} rs_{i-1}{p_{i-1}} */
                  (*(gmres_functions->ClearVector))(w);
                  (*(gmres_functions->Axpy))(rs[i - 1], p[i - 1], w);
                  (*(gmres_functions->ClearVector))(r);
                  /* apply the preconditioner */
                  precond(precond_data, A, w, r);
                  /* find the norm of x_i - x_i-1 */
                  (*(gmres_functions->InnerProd))(r, r, &num_tags, &iprod);
                  w_norm = hypre_sqrt(iprod[0]);

                  relative_error = w_norm / x_norm;
                  if (relative_error < r_tol)
                  {
                     (gmres_data -> converged) = 1;
                     break;
                  }
               }
               else
               {
                  (gmres_data -> converged) = 1;
                  break;
               }
            }
            else /* don't need to check rel. change */
            {
               (gmres_data -> converged) = 1;
               break;
            }
         }
         else /* conv. has not occurred, according to true residual */
         {
            /* exit if the real residual norm has not decreased */
            if (real_r_norm_new >= real_r_norm_old)
            {
               (gmres_data -> converged) = 1;
               break;
            }

            /* report discrepancy between real/GMRES residuals and restart */
            if (!my_id && print_level > 0)
            {
               hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
            }
            (*(gmres_functions->CopyVector))(r, p[0]);
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

      if (i) { (*(gmres_functions->Axpy))(rs[i] - 1.0, p[i], p[i]); }
      for (j = i - 1 ; j > 0; j--)
      {
         (*(gmres_functions->Axpy))(rs[j], p[j], p[i]);
      }

      if (i)
      {
         (*(gmres_functions->Axpy))(rs[0] - 1.0, p[0], p[0]);
         (*(gmres_functions->Axpy))(1.0, p[i], p[0]);
      }
   } /* END of iteration while loop */

   if (!my_id && print_level > 1)
   {
      hypre_printf("\n\n");
   }

   if (!my_id && print_level > 0)
   {
      hypre_printf("Final L2 norm of residual: %e\n", r_norm);
      if (num_tags > 1)
      {
         for (tag = 0; tag < num_tags; tag++)
         {
            hypre_printf("Final L2 norm of r%*d: %e\n", hypre_ndigits(num_tags),
                         tag, hypre_sqrt(iprod[tag + 1]));
         }
      }
   }

   if (xref && print_level > 0)
   {
      (*(gmres_functions->CopyVector))(x, w_3);
      (*(gmres_functions->Axpy))(-1.0, xref, w_3);
      (*(gmres_functions->InnerProd))(w_3, w_3, &num_tags, &iprod);

      if (!my_id)
      {
         hypre_printf("Final L2 norm of error: %e\n", hypre_sqrt(iprod[0]));
         if (num_tags > 1)
         {
            for (tag = 0; tag < num_tags; tag++)
            {
               hypre_printf("Final L2 norm of e%*d: %e\n", hypre_ndigits(num_tags),
                            tag, hypre_sqrt(iprod[tag + 1]));
            }
         }
      }
   }

   if (!my_id && print_level > 0)
   {
      hypre_printf("\n");
   }

   (gmres_data -> num_iterations) = iter;

   if (b_norm > 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / b_norm;
   }

   if (b_norm == 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm;
   }

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0 && hybrid != -1)
   {
      hypre_error(HYPRE_ERROR_CONV);
   }

   /* free up memory */
   hypre_TFree(iprod, HYPRE_MEMORY_HOST);
   hypre_TFree(xiprod, HYPRE_MEMORY_HOST);
   hypre_TFree(biprod, HYPRE_MEMORY_HOST);
   hypre_TFreeF(c, gmres_functions);
   hypre_TFreeF(s, gmres_functions);
   hypre_TFreeF(rs, gmres_functions);

   if (rel_change)
   {
      hypre_TFreeF(rs_2, gmres_functions);
   }

   if (print_level > 2)
   {
      hypre_TFreeF(rs_3, gmres_functions);
   }

   for (i = 0; i < k_dim + 1; i++)
   {
      hypre_TFreeF(hh[i], gmres_functions);
   }

   hypre_TFreeF(hh, gmres_functions);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetKDim, hypre_GMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetKDim( void     *gmres_vdata,
                    HYPRE_Int k_dim )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *) gmres_vdata;


   (gmres_data -> k_dim) = k_dim;

   return hypre_error_flag;

}

HYPRE_Int
hypre_GMRESGetKDim( void      *gmres_vdata,
                    HYPRE_Int *k_dim )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *k_dim = (gmres_data -> k_dim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetTol, hypre_GMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetTol( void      *gmres_vdata,
                   HYPRE_Real tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> tol) = tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetTol( void       *gmres_vdata,
                   HYPRE_Real *tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *tol = (gmres_data -> tol);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_GMRESSetAbsoluteTol, hypre_GMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetAbsoluteTol( void      *gmres_vdata,
                           HYPRE_Real a_tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> a_tol) = a_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetAbsoluteTol( void       *gmres_vdata,
                           HYPRE_Real *a_tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *a_tol = (gmres_data -> a_tol);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_GMRESSetConvergenceFactorTol, hypre_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetConvergenceFactorTol( void      *gmres_vdata,
                                    HYPRE_Real cf_tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> cf_tol) = cf_tol;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetConvergenceFactorTol( void       *gmres_vdata,
                                    HYPRE_Real *cf_tol )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *cf_tol = (gmres_data -> cf_tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMinIter, hypre_GMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetMinIter( void     *gmres_vdata,
                       HYPRE_Int min_iter )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> min_iter) = min_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetMinIter( void      *gmres_vdata,
                       HYPRE_Int *min_iter )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *min_iter = (gmres_data -> min_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMaxIter, hypre_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetMaxIter( void      *gmres_vdata,
                       HYPRE_Int  max_iter )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetMaxIter( void      *gmres_vdata,
                       HYPRE_Int *max_iter )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *max_iter = (gmres_data -> max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetRelChange, hypre_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetRelChange( void     *gmres_vdata,
                         HYPRE_Int rel_change )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetRelChange( void      *gmres_vdata,
                         HYPRE_Int *rel_change )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *rel_change = (gmres_data -> rel_change);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetSkipRealResidualCheck, hypre_GMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetSkipRealResidualCheck( void     *gmres_vdata,
                                     HYPRE_Int skip_real_r_check )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   (gmres_data -> skip_real_r_check) = skip_real_r_check;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetSkipRealResidualCheck( void      *gmres_vdata,
                                     HYPRE_Int *skip_real_r_check)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   *skip_real_r_check = (gmres_data -> skip_real_r_check);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetStopCrit, hypre_GMRESGetStopCrit
 *
 *  OBSOLETE
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetStopCrit( void      *gmres_vdata,
                        HYPRE_Int  stop_crit )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> stop_crit) = stop_crit;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetStopCrit( void      *gmres_vdata,
                        HYPRE_Int *stop_crit )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *stop_crit = (gmres_data -> stop_crit);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetPrecond( void  *gmres_vdata,
                       HYPRE_Int  (*precond)(void*, void*, void*, void*),
                       HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                       void  *precond_data )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;


   (gmres_functions -> precond)        = precond;
   (gmres_functions -> precond_setup)  = precond_setup;
   (gmres_data -> precond_data)        = precond_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESGetPrecond( void         *gmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *precond_data_ptr = (HYPRE_Solver)(gmres_data -> precond_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrintLevel, hypre_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetPrintLevel( void      *gmres_vdata,
                          HYPRE_Int  level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   (gmres_data -> print_level) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetPrintLevel( void      *gmres_vdata,
                          HYPRE_Int *level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *level = (gmres_data -> print_level);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetLogging, hypre_GMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetLogging( void     *gmres_vdata,
                       HYPRE_Int level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   (gmres_data -> logging) = level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESGetLogging( void      *gmres_vdata,
                       HYPRE_Int *level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   *level = (gmres_data -> logging);

   return hypre_error_flag;
}

HYPRE_Int
hypre_GMRESSetHybrid( void *gmres_vdata,
                      HYPRE_Int   level)
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   (gmres_data -> hybrid) = level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESGetNumIterations( void      *gmres_vdata,
                             HYPRE_Int *num_iterations )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *num_iterations = (gmres_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESGetConverged( void      *gmres_vdata,
                         HYPRE_Int *converged )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *converged = (gmres_data -> converged);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESGetFinalRelativeResidualNorm( void       *gmres_vdata,
                                         HYPRE_Real *relative_residual_norm )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;


   *relative_residual_norm = (gmres_data -> rel_residual_norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetRefSolution
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESSetRefSolution( void  *gmres_vdata,
                           void  *xref )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   (gmres_data -> xref) = xref;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetRefSolution
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GMRESGetRefSolution( void   *gmres_vdata,
                           void  **xref )
{
   hypre_GMRESData *gmres_data = (hypre_GMRESData *)gmres_vdata;

   *xref = (gmres_data -> xref);

   return hypre_error_flag;
}
