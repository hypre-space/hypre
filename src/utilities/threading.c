/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "_hypre_utilities.h"

#ifdef HYPRE_USING_OPENMP

HYPRE_Int
hypre_NumThreads( void )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_max_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

HYPRE_Int
hypre_NumActiveThreads( void )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_num_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

HYPRE_Int
hypre_GetThreadNum( void )
{
   HYPRE_Int my_thread_num;

   my_thread_num = omp_get_thread_num();

   return my_thread_num;
}

void
hypre_SetNumThreads( HYPRE_Int nt )
{
   omp_set_num_threads(nt);
}

#endif

/* This next function must be called from within a parallel region! */

void
hypre_GetSimpleThreadPartition( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n )
{
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   HYPRE_Int my_thread_num = hypre_GetThreadNum();

   HYPRE_Int n_per_thread = (n + num_threads - 1) / num_threads;

   *begin = hypre_min(n_per_thread * my_thread_num, n);
   *end = hypre_min(*begin + n_per_thread, n);
}
