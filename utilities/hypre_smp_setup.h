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
#ifdef HYPRE_USING_OPENMP
omp_set_num_threads(4);
#endif

#ifdef HYPRE_USING_IBM_SMP

#endif



