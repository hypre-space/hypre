/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/*****************************************************************************
 *	Wrapper code for SMP compiler directives.  Translates 
 *	hypre SMP directives into the appropriate Open MP,
 *	IBM, SGI, or pgcc (Red) SMP compiler directives.
 ****************************************************************************/

#ifndef HYPRE_SMP_PRIVATE
#define HYPRE_SMP_PRIVATE
#endif

#ifdef HYPRE_USING_OPENMP
#ifndef HYPRE_SMP_REDUCTION_OP
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) schedule(static)
#endif
#ifdef HYPRE_SMP_REDUCTION_OP
#pragma omp parallel for private(HYPRE_SMP_PRIVATE) \
reduction(HYPRE_SMP_REDUCTION_OP: HYPRE_SMP_REDUCTION_VARS) \
schedule(static)
#endif
#endif

#ifdef HYPRE_USING_SGI_SMP
#pragma parallel
#pragma pfor
#pragma schedtype(gss)
#pragma chunksize(10)
#endif

#ifdef HYPRE_USING_IBM_SMP
#pragma parallel_loop
#pragma schedule (guided,10)
#endif

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
