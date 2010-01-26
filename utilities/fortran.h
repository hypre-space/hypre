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
 * Developers should only use the following in hypre:
 *   hypre_F90_NAME() or hypre_F90_IFACE()
 *   hypre_F90_NAME_BLAS()
 *   hypre_F90_NAME_LAPACK()
 *
 *****************************************************************************/

#ifndef HYPRE_FORT_HEADER
#define HYPRE_FORT_HEADER

#include "HYPRE_config.h"

/*-------------------------------------------------------
 * Define specific name mangling macros to be used below
 *-------------------------------------------------------*/

#define hypre_F90_NAME_1(name,NAME) name
#define hypre_F90_NAME_2(name,NAME) name##_
#define hypre_F90_NAME_3(name,NAME) name##__
#define hypre_F90_NAME_4(name,NAME) NAME

/*-------------------------------------------------------
 * Define hypre_F90_NAME and hypre_F90_IFACE
 *-------------------------------------------------------*/

#if   (HYPRE_FMANGLE == 1)
#define hypre_F90_NAME(name,NAME) hypre_F90_NAME_1(name,NAME)
#elif (HYPRE_FMANGLE == 2)
#define hypre_F90_NAME(name,NAME) hypre_F90_NAME_2(name,NAME)
#elif (HYPRE_FMANGLE == 3)
#define hypre_F90_NAME(name,NAME) hypre_F90_NAME_3(name,NAME)
#elif (HYPRE_FMANGLE == 4)
#define hypre_F90_NAME(name,NAME) hypre_F90_NAME_4(name,NAME)

#elif defined(HYPRE_F77_FUNC_)
/* HYPRE_F77_FUNC_ macro assumes underscores exist in name */
#define hypre_F90_NAME(name,NAME) HYPRE_F77_FUNC_(name,NAME)

#else
#define hypre_F90_NAME(name,NAME) hypre_F90_NAME_2(name,NAME)

#endif

#define hypre_F90_IFACE(name,NAME) hypre_F90_NAME(name,NAME)

/*-------------------------------------------------------
 * Define hypre_F90_NAME_BLAS
 *-------------------------------------------------------*/

#ifdef HYPRE_USING_HYPRE_BLAS
#define hypre_F90_NAME_BLAS(name,NAME)  hypre_##name

#elif (HYPRE_FMANGLE_BLAS == 1)
#define hypre_F90_NAME_BLAS(name,NAME) hypre_F90_NAME_1(name,NAME)
#elif (HYPRE_FMANGLE_BLAS == 2)
#define hypre_F90_NAME_BLAS(name,NAME) hypre_F90_NAME_2(name,NAME)
#elif (HYPRE_FMANGLE_BLAS == 3)
#define hypre_F90_NAME_BLAS(name,NAME) hypre_F90_NAME_3(name,NAME)
#elif (HYPRE_FMANGLE_BLAS == 4)
#define hypre_F90_NAME_BLAS(name,NAME) hypre_F90_NAME_4(name,NAME)

#elif defined(HYPRE_F77_FUNC)
/* HYPRE_F77_FUNC macro assumes NO underscores exist in name */
#define hypre_F90_NAME_BLAS(name,NAME) HYPRE_F77_FUNC(name,NAME)

#else
#define hypre_F90_NAME_BLAS(name,NAME) hypre_F90_NAME_2(name,NAME)

#endif

/*-------------------------------------------------------
 * Define hypre_F90_NAME_LAPACK
 *-------------------------------------------------------*/

#ifdef HYPRE_USING_HYPRE_LAPACK
#define hypre_F90_NAME_LAPACK(name,NAME)  hypre_##name

#elif (HYPRE_FMANGLE_LAPACK == 1)
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_F90_NAME_1(name,NAME)
#elif (HYPRE_FMANGLE_LAPACK == 2)
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_F90_NAME_2(name,NAME)
#elif (HYPRE_FMANGLE_LAPACK == 3)
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_F90_NAME_3(name,NAME)
#elif (HYPRE_FMANGLE_LAPACK == 4)
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_F90_NAME_4(name,NAME)

#elif defined(HYPRE_F77_FUNC)
/* HYPRE_F77_FUNC macro assumes NO underscores exist in name */
#define hypre_F90_NAME_LAPACK(name,NAME) HYPRE_F77_FUNC(name,NAME)

#else
#define hypre_F90_NAME_LAPACK(name,NAME) hypre_F90_NAME_2(name,NAME)

#endif

#endif
