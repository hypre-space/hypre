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

/*--------------------------------------------------------------------------
 * hypre_NumThreads
 *
 * Returns the maximum number of threads that can be used.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NumThreads( void )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_max_threads();

   return num_threads;
}

/*--------------------------------------------------------------------------
 * hypre_NumOptimalThreads
 *
 * Returns the optimal number of threads for the given problem size. Considers
 * the minimum work per thread and the maximum number of threads to avoid
 * thread creation overhead. Must be called from outside of a parallel region.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NumOptimalThreads(HYPRE_Int size)
{
   /* Minimum work per thread */
   const HYPRE_Int min_rows_per_thread = 500;

   HYPRE_Int optimal_threads = size / min_rows_per_thread;

   return hypre_max(1, hypre_min(optimal_threads, omp_get_max_threads()));
}

/*--------------------------------------------------------------------------
 * hypre_SetNumThreads
 *
 * Sets the number of threads to use. Must be called from outside of a
 * parallel region.
 *--------------------------------------------------------------------------*/

void
hypre_SetNumThreads( HYPRE_Int nt )
{
   omp_set_num_threads(nt);
}

/*--------------------------------------------------------------------------
 * hypre_NumActiveThreads
 *
 * Returns the number of threads currently active. Must be called from within
 * a parallel region.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NumActiveThreads( void )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_num_threads();

   return num_threads;
}

/*--------------------------------------------------------------------------
 * hypre_GetThreadNum
 *
 * Returns the thread ID of the calling thread. Must be called from within a
 * parallel region.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GetThreadNum( void )
{
   HYPRE_Int my_thread_num;

   my_thread_num = omp_get_thread_num();

   return my_thread_num;
}

#endif

/*--------------------------------------------------------------------------
 * hypre_GetSimpleThreadPartition
 *
 * Partitions the rows of a matrix into a simple thread partition. Must be
 * called from within a parallel region.
 *--------------------------------------------------------------------------*/

void
hypre_GetSimpleThreadPartition( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n )
{
   HYPRE_Int num_threads = hypre_NumActiveThreads();
   HYPRE_Int my_thread_num = hypre_GetThreadNum();

   HYPRE_Int n_per_thread = (n + num_threads - 1) / num_threads;

   *begin = hypre_min(n_per_thread * my_thread_num, n);
   *end = hypre_min(*begin + n_per_thread, n);
}
