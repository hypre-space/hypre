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



/******************************************************************************
 *
 * Fortran <-> C interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_FORT_HEADER
#define HYPRE_FORT_HEADER

#include "HYPRE_config.h"

#ifdef WIN32
#include "mkl.h"
#endif

#if defined(F77_FUNC_)
/* F77_FUNC_ macro assumes underscores exist in name */
#  define hypre_NAME_C_CALLING_FORT(name,NAME) F77_FUNC_(name,NAME)
#  define hypre_NAME_FORT_CALLING_C(name,NAME) F77_FUNC_(name,NAME)

#elif defined(HYPRE_RS6000)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name

#elif defined(HYPRE_CRAY)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) NAME
#  define hypre_NAME_FORT_CALLING_C(name,NAME) NAME

#elif defined(HYPRE_LINUX)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name##__
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name##__

#elif defined(HYPRE_LINUX_CHAOS)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name##_
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name##_

#elif defined(HYPRE_HPPA)

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name

#elif defined(WIN32)
#  define hypre_NAME_C_CALLING_FORT(name,NAME) NAME
#  define hypre_NAME_FORT_CALLING_C(name,NAME) NAME

#else

#  define hypre_NAME_C_CALLING_FORT(name,NAME) name##_
#  define hypre_NAME_FORT_CALLING_C(name,NAME) name##_

#endif

#define hypre_F90_IFACE(name,NAME) hypre_NAME_FORT_CALLING_C(name,NAME)
#define hypre_F90_NAME(name,NAME)  hypre_NAME_C_CALLING_FORT(name,NAME)

#ifdef WIN32
#define hypre_F90_NAME_BLAS(name,NAME) NAME
#else

#ifdef HYPRE_USING_HYPRE_BLAS
#define hypre_F90_NAME_BLAS(name,NAME)  hypre_##name
#else
#if defined(F77_FUNC)
/* F77_FUNC macro assumes NO underscores exist in name */
#define hypre_F90_NAME_BLAS(name,NAME) F77_FUNC(name,NAME)
#else
#define hypre_F90_NAME_BLAS(name,NAME) hypre_NAME_C_CALLING_FORT(name,NAME)
#endif
#endif
#endif

#ifdef HYPRE_USING_HYPRE_LAPACK
#define hypre_F90_NAME_LAPACK(name,NAME)  hypre_##name
#else
#if defined(F77_FUNC)
/* F77_FUNC macro assumes NO underscores exist in name */
#define hypre_F90_NAME_LAPACK(name,NAME) F77_FUNC(name,NAME)
#else
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_NAME_C_CALLING_FORT(name,NAME)
#endif
#endif

#endif
