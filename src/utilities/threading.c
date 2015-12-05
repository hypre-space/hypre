/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <stdlib.h>
#include <stdio.h>
#include "_hypre_utilities.h"

#ifdef HYPRE_USING_OPENMP

HYPRE_Int
hypre_NumThreads( )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_max_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

HYPRE_Int
hypre_NumActiveThreads( )
{
   HYPRE_Int num_threads;

   num_threads = omp_get_num_threads();

   return num_threads;
}

/* This next function must be called from within a parallel region! */

HYPRE_Int
hypre_GetThreadNum( )
{
   HYPRE_Int my_thread_num;

   my_thread_num = omp_get_thread_num();

   return my_thread_num;
}

#endif
