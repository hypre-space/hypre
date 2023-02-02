/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_Log2:
 *   This routine returns the integer, floor(log_2(p)).
 *   If p <= 0, it returns a -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_Log2( HYPRE_Int p )
{
   HYPRE_Int  e;

   if (p <= 0)
   {
      return -1;
   }

   e = 0;
   while (p > 1)
   {
      e += 1;
      p /= 2;
   }

   return e;
}
