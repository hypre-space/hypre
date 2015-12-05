/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

/*****************************************************************************
 *	Wrapper code for SMP compiler directives.  Translates 
 *	hypre SMP directives into the appropriate OpenMP,
 *	IBM, SGI, or pgcc (Red) SMP compiler directives.
 ****************************************************************************/

#ifndef HYPRE_SMP_PRIVATE
#define HYPRE_SMP_PRIVATE
#endif

/* OpenMP */

#ifdef HYPRE_USING_OPENMP

#ifndef HYPRE_SMP_REDUCTION_OP
#ifndef HYPRE_SMP_PAR_REGION
#ifndef HYPRE_SMP_FOR
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) schedule(static)
#endif
#endif
#endif

#ifdef HYPRE_SMP_PAR_REGION
#pragma omp parallel private(HYPRE_SMP_PRIVATE)
#endif

#ifdef HYPRE_SMP_FOR
#pragma omp for schedule(static)
#endif

#ifdef HYPRE_SMP_REDUCTION_OP
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) \
reduction(HYPRE_SMP_REDUCTION_OP: HYPRE_SMP_REDUCTION_VARS) \
schedule(static)
#endif

#endif

/* SGI */

#ifdef HYPRE_USING_SGI_SMP
#pragma parallel
#pragma pfor
#pragma schedtype(gss)
#pragma chunksize(10)
#endif

/* IBM */

#ifdef HYPRE_USING_IBM_SMP
#pragma parallel_loop
#pragma schedule (guided,10)
#endif

/* PGCC */

#ifdef HYPRE_USING_PGCC_SMP
#ifndef HYPRE_SMP_REDUCTION_OP
#pragma parallel local(HYPRE_SMP_PRIVATE) pfor
#endif
#ifdef HYPRE_SMP_REDUCTION_OP
#endif
#endif

#undef HYPRE_SMP_PRIVATE
#undef HYPRE_SMP_REDUCTION_OP
#undef HYPRE_SMP_REDUCTION_VARS
#undef HYPRE_SMP_PAR_REGION
#undef HYPRE_SMP_FOR
