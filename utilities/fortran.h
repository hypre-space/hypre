/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Fortran <-> C interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_FORT_HEADER
#define HYPRE_FORT_HEADER

#if defined(HYPRE_SOLARIS) || defined(HYPRE_ALPHA)

#  define hypre_NAME_C_CALLING_FORT(name) name##_
#  define hypre_NAME_FORT_CALLING_C(name) name##_

#elif defined(HYPRE_RS6000)

#  define hypre_NAME_C_CALLING_FORT(name) name
#  define hypre_NAME_FORT_CALLING_C(name) name

#else

#  define hypre_NAME_C_CALLING_FORT(name) name##_
#  define hypre_NAME_FORT_CALLING_C(name) name##_

#endif

#if defined(HYPRE_BLAS)
#  define hypre_NAME_C_CALLING_FORT(name) name##_
#  define hypre_NAME_HYPRE_CALLING_BLAS(name) name##_
#endif

#define hypre_F90_IFACE(iface_name) hypre_NAME_FORT_CALLING_C(iface_name)
#define hypre_F90_NAME(iface_name)  hypre_NAME_C_CALLING_FORT(iface_name)
#define hypre_BLAS_NAME(iface_name)  hypre_NAME_HYPRE_CALLING_BLAS(iface_name)

#endif
