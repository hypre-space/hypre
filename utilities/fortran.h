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

#if defined(HYPRE_RS6000)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name

#elif defined(HYPRE_CRAY)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) NAME
#  define hypre_NAME_FORT_CALLING_C(name,NAME) NAME

#elif defined(HYPRE_LINUX)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name##__
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name##__

#elif defined(HYPRE_HP)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name

#else

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name##_
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name##_

#endif

#define hypre_F90_IFACE(name,NAME) hypre_NAME_FORT_CALLING_C(name,NAME)
#define hypre_F90_NAME(name,NAME)  hypre_NAME_C_CALLING_FORT(name,NAME)

#ifdef HYPRE_USING_HYPRE_BLAS
#define hypre_F90_NAME_BLAS(name,NAME)  name##_
#else
#define hypre_F90_NAME_BLAS(name,NAME)  hypre_NAME_C_CALLING_FORT(name,NAME)
#endif

#endif
