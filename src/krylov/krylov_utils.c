/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "krylov.h"

/*--------------------------------------------------------------------------
 * Print a standardized error message for NaN/INF detection in Krylov solvers.
 *  - solver_name: Name of the Krylov solver (e.g., "GMRES", "PCG").
 *  - context: Description of the input (e.g., "b", "A or x_0").
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_KrylovPrintErrorNaN(const char *solver_name,
                          const char *context)
{
   hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
   hypre_printf("ERROR -- %s: INFs and/or NaNs detected in input.\n", solver_name);
   hypre_printf("User probably placed non-numerics in supplied %s.\n", context);
   hypre_printf("Returning error flag += 101.  Program not terminated.\n");
   hypre_printf("ERROR detected by Hypre ... END\n\n\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print a standardized header for residual/error output in Krylov solvers.
 * The parameters are generic and can be adapted for each solver's needs.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_KrylovPrintHeader(HYPRE_Int   print_level,
                        HYPRE_Int   num_tags,
                        HYPRE_Real  b_norm,
                        HYPRE_Real  e_norm)
{
   HYPRE_Int tag;

   if (print_level > 1)
   {
      hypre_printf("=============================================\n\n");
      if (num_tags <= 1 || print_level == 2 || print_level > 9 ||
          ((e_norm == 0.0) && print_level > 5))
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
         hypre_printf("Iters      error.norm     conv.rate   rel.err.norm\n");
         hypre_printf("-----    ------------    ----------   ------------\n");
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print per-iteration residual/error norms (simple case).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_KrylovPrintIterSimple(HYPRE_Int   iter,
                            HYPRE_Real *norms,
                            HYPRE_Real  ref_norm)
{
   if (!my_id)
   {
      if (ref_norm > 0.0)
      {
         hypre_printf("%5d    %e      %f   %e\n",
                      iter, norms[iter],
                      norms[iter] / norms[iter - 1],
                      norms[iter] / ref_norm);
      }
      else
      {
         hypre_printf("%5d    %e      %f\n",
                      iter, norms[iter],
                      norms[iter] / norms[iter - 1]);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print per-iteration tag-wise residual/error norms (complex case).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_KrylovPrintIterTags(HYPRE_Int      print_level,
                          HYPRE_Int      iter,
                          HYPRE_Int      num_tags,
                          HYPRE_Complex *iprod,
                          HYPRE_Complex *biprod,
                          HYPRE_Complex *xiprod)
{
   HYPRE_Int tag;

   if (print_level != 6)
   {
      hypre_printf(" %6d  ", iter);
      for (tag = 0; tag < num_tags + 1; tag++)
      {
         hypre_printf("  %14.6e ",
                      print_level == 3 || print_level == 7 ? hypre_sqrt(iprod[tag]) :
                      print_level == 4 ? hypre_sqrt(iprod[tag]) / hypre_sqrt(biprod[tag]) :
                      print_level == 5 ? hypre_sqrt(iprod[tag]) / hypre_sqrt(biprod[0]) :
                      print_level == 8 ? hypre_sqrt(iprod[tag]) / hypre_sqrt(xiprod[tag]) :
                      print_level == 9 ? hypre_sqrt(iprod[tag]) / hypre_sqrt(xiprod[0]) :
                      hypre_sqrt(iprod[tag]));
      }
      hypre_printf("\n");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print final summary of residuals and errors.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_KrylovPrintFinalSummary(HYPRE_Int       num_tags,
                              HYPRE_Complex  *riprod,
                              HYPRE_Complex  *xiprod)
{
   HYPRE_Int tag;

   /* Print residual norms */
   hypre_printf("Final L2 norm of residual: %e\n", hypre_sqrt(riprod[0]));
   if (num_tags > 1)
   {
      for (tag = 0; tag < num_tags; tag++)
      {
         hypre_printf("Final L2 norm of r%*d: %e\n", hypre_ndigits(num_tags), tag,
                      hypre_sqrt(riprod[tag + 1]));
      }
   }

   /* Print error norms */
   if (xiprod)
   {
      hypre_printf("Final L2 norm of error: %e\n", hypre_sqrt(xiprod[0]));
      if (num_tags > 1)
      {
         for (tag = 0; tag < num_tags; tag++)
         {
            hypre_printf("Final L2 norm of e%*d: %e\n", hypre_ndigits(num_tags), tag,
                         hypre_sqrt(xiprod[tag + 1]));
         }
      }
   }
   hypre_printf("\n");

   return hypre_error_flag;
}
