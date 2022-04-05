/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_GeneratePartitioning:
 * generates load balanced partitioning of a 1-d array
 *--------------------------------------------------------------------------*/
/* for multivectors, length should be the (global) length of a single vector.
 Thus each of the vectors of the multivector will get the same data distribution. */

HYPRE_Int
hypre_GeneratePartitioning(HYPRE_BigInt length, HYPRE_Int num_procs, HYPRE_BigInt **part_ptr)
{
   HYPRE_Int ierr = 0;
   HYPRE_BigInt *part;
   HYPRE_Int size, rest;
   HYPRE_Int i;

   part = hypre_CTAlloc(HYPRE_BigInt,  num_procs + 1, HYPRE_MEMORY_HOST);
   size = (HYPRE_Int)(length / (HYPRE_BigInt)num_procs);
   rest = (HYPRE_Int)(length - (HYPRE_BigInt)(size * num_procs));
   part[0] = 0;
   for (i = 0; i < num_procs; i++)
   {
      part[i + 1] = part[i] + (HYPRE_BigInt)size;
      if (i < rest) { part[i + 1]++; }
   }

   *part_ptr = part;
   return ierr;
}


/* This function differs from the above in that it only returns
   the portion of the partition belonging to the individual process -
   to do this it requires the processor id as well AHB 6/05.

   This functions assumes that part is on the stack memory
   and has size equal to 2.
*/

HYPRE_Int
hypre_GenerateLocalPartitioning(HYPRE_BigInt   length,
                                HYPRE_Int      num_procs,
                                HYPRE_Int      myid,
                                HYPRE_BigInt  *part)
{
   HYPRE_Int  size, rest;

   size = (HYPRE_Int)(length / (HYPRE_BigInt)num_procs);
   rest = (HYPRE_Int)(length - (HYPRE_BigInt)(size * num_procs));

   /* first row I own */
   part[0] = (HYPRE_BigInt)(size * myid);
   part[0] += (HYPRE_BigInt)(hypre_min(myid, rest));

   /* last row I own */
   part[1] =  (HYPRE_BigInt)(size * (myid + 1));
   part[1] += (HYPRE_BigInt)(hypre_min(myid + 1, rest));
   part[1] = part[1] - 1;

   /* add 1 to last row since this is for "starts" vector */
   part[1] = part[1] + 1;

   return hypre_error_flag;
}
