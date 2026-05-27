/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_KRYLOV_RES_PRINT_HEADER
#define hypre_KRYLOV_RES_PRINT_HEADER

typedef enum
{
   hypre_KrylovResPrintNone = 0,
   hypre_KrylovResPrintScalarResidual,
   hypre_KrylovResPrintScalarError,
   hypre_KrylovResPrintTagged
} hypre_KrylovResPrintMode;

static inline hypre_KrylovResPrintMode
hypre_KrylovResPrintGetMode( HYPRE_Int print_level,
                             HYPRE_Int num_tags,
                             HYPRE_Int has_xref )
{
   if (print_level <= 0)
   {
      return hypre_KrylovResPrintNone;
   }

   if (num_tags <= 1 || print_level == 2 || print_level > 9 || (!has_xref && print_level > 5))
   {
      return hypre_KrylovResPrintScalarResidual;
   }

   if (has_xref && print_level == 6)
   {
      return hypre_KrylovResPrintScalarError;
   }

   if (num_tags > 1 && print_level > 2)
   {
      return hypre_KrylovResPrintTagged;
   }

   return hypre_KrylovResPrintNone;
}

static inline void
hypre_KrylovResPrintHeader( hypre_KrylovResPrintMode print_mode,
                            HYPRE_Int                     print_level,
                            HYPRE_Int                     num_tags,
                            HYPRE_Real                    b_norm,
                            HYPRE_Real                    e_norm )
{
   HYPRE_Int tag;

   hypre_printf("=============================================\n\n");

   if (print_mode == hypre_KrylovResPrintScalarResidual)
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

      return;
   }

   if (print_mode == hypre_KrylovResPrintScalarError)
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

      return;
   }

   if (print_mode == hypre_KrylovResPrintTagged && num_tags > 1)
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

static inline void
hypre_KrylovResPrintScalarRow( HYPRE_Int  iter,
                               HYPRE_Real norm,
                               HYPRE_Real prev_norm,
                               HYPRE_Real ref_norm )
{
   if (ref_norm > 0.0)
   {
      hypre_printf("%5d    %e      %f   %e\n", iter, norm, norm / prev_norm, norm / ref_norm);
   }
   else
   {
      hypre_printf("%5d    %e      %f\n", iter, norm, norm / prev_norm);
   }
}

static inline HYPRE_Real
hypre_KrylovResPrintTaggedValue( HYPRE_Int      print_level,
                                 HYPRE_Int      tag,
                                 HYPRE_Complex *iprod,
                                 HYPRE_Complex *biprod,
                                 HYPRE_Complex *xiprod )
{
   HYPRE_Real value = hypre_sqrt(iprod[tag]);

   if (print_level == 4)
   {
      value /= hypre_sqrt(biprod[tag]);
   }
   else if (print_level == 5)
   {
      value /= hypre_sqrt(biprod[0]);
   }
   else if (print_level == 8)
   {
      value /= hypre_sqrt(xiprod[tag]);
   }
   else if (print_level == 9)
   {
      value /= hypre_sqrt(xiprod[0]);
   }

   return value;
}

static inline void
hypre_KrylovResPrintTaggedRow( HYPRE_Int      iter,
                               HYPRE_Int      print_level,
                               HYPRE_Int      num_tags,
                               HYPRE_Complex *iprod,
                               HYPRE_Complex *biprod,
                               HYPRE_Complex *xiprod )
{
   HYPRE_Int tag;

   hypre_printf(" %6d  ", iter);
   for (tag = 0; tag < num_tags + 1; tag++)
   {
      hypre_printf("  %14.6e ",
                   hypre_KrylovResPrintTaggedValue(print_level, tag, iprod, biprod, xiprod));
   }
   hypre_printf("\n");
}

#endif
