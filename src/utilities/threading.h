/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_THREADING_HEADER
#define hypre_THREADING_HEADER

#ifdef HYPRE_USING_OPENMP

HYPRE_Int hypre_NumThreads( void );
HYPRE_Int hypre_NumActiveThreads( void );
HYPRE_Int hypre_GetThreadNum( void );

#else

#define hypre_NumThreads() 1
#define hypre_NumActiveThreads() 1
#define hypre_GetThreadNum() 0

#endif

void hypre_GetSimpleThreadPartition( HYPRE_Int *begin, HYPRE_Int *end, HYPRE_Int n );

#endif

