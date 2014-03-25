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

#endif
