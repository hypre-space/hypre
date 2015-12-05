/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.22 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Fortran <-> C interface macros
 *
 * Developers should only use the following in hypre:
 *   hypre_F90_NAME() or hypre_F90_IFACE()
 *   hypre_F90_NAME_BLAS()
 *   hypre_F90_NAME_LAPACK()
 *   any of the interface argument macros at the bottom of this file
 *
 *****************************************************************************/

#ifndef HYPRE_FORT_HEADER
#define HYPRE_FORT_HEADER

#include "_hypre_utilities.h"

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

/*-------------------------------------------------------
 * Define interface argument types and macros
 *
 * A Fortran communicator is always the size of an integer
 * and hence usually the size of hypre_int.
 *-------------------------------------------------------*/

typedef hypre_int  hypre_F90_Comm;
typedef HYPRE_Int  hypre_F90_Int;
typedef HYPRE_Int  hypre_F90_IntArray;
typedef double     hypre_F90_Dbl;
typedef double     hypre_F90_DblArray;
typedef HYPRE_Int *hypre_F90_Obj;
typedef HYPRE_Int *hypre_F90_ObjRef;

#define hypre_F90_PassComm(arg)       (hypre_MPI_Comm_f2c(*arg))
#define hypre_F90_PassInt(arg)        ((HYPRE_Int) *arg)
#define hypre_F90_PassIntRef(arg)     ((HYPRE_Int *) arg)
#define hypre_F90_PassIntArray(arg)   ((HYPRE_Int *) arg)
#define hypre_F90_PassDbl(arg)        ((double) *arg)
#define hypre_F90_PassDblRef(arg)     ((double *) arg)
#define hypre_F90_PassDblArray(arg)   ((double *) arg)
#define hypre_F90_PassObj(obj,arg)    ((obj) *arg)
#define hypre_F90_PassObjRef(obj,arg) ((obj *) arg)

#endif
