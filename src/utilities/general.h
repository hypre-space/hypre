/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef hypre_GENERAL_HEADER
#define hypre_GENERAL_HEADER

#include <math.h>

/* This allows us to consistently avoid 'int' throughout hypre */
typedef int                    hypre_int;
typedef long int               hypre_longint;
typedef unsigned int           hypre_uint;
typedef unsigned long int      hypre_ulongint;
typedef unsigned long long int hypre_ulonglongint;

/* This allows us to consistently avoid 'double' throughout hypre */
typedef double                 hypre_double;

/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef hypre_max
#define hypre_max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef hypre_min
#define hypre_min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef hypre_abs
#define hypre_abs(a)  (((a)>0) ? (a) : -(a))
#endif

#ifndef hypre_round
#define hypre_round(x)  ( ((x) < 0.0) ? ((HYPRE_Int)(x - 0.5)) : ((HYPRE_Int)(x + 0.5)) )
#endif

#ifndef hypre_pow2
#define hypre_pow2(i)  ( 1 << (i) )
#endif

#ifndef hypre_sqrt
#if defined(HYPRE_SINGLE)
#define hypre_sqrt sqrtf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_sqrt sqrtl
#else
#define hypre_sqrt sqrt
#endif
#endif

#ifndef hypre_pow
#if defined(HYPRE_SINGLE)
#define hypre_pow powf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_pow powl
#else
#define hypre_pow pow
#endif
#endif

#ifndef hypre_ceil
#if defined(HYPRE_SINGLE)
#define hypre_ceil ceilf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_ceil ceill
#else
#define hypre_ceil ceil
#endif
#endif

#ifndef hypre_floor
#if defined(HYPRE_SINGLE)
#define hypre_floor floorf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_floor floorl
#else
#define hypre_floor floor
#endif
#endif

#ifndef hypre_log
#if defined(HYPRE_SINGLE)
#define hypre_log logf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_log logl
#else
#define hypre_log log
#endif
#endif

#ifndef hypre_exp
#if defined(HYPRE_SINGLE)
#define hypre_exp expf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_exp expl
#else
#define hypre_exp exp
#endif
#endif

#ifndef hypre_sin
#if defined(HYPRE_SINGLE)
#define hypre_sin sinf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_sin sinl
#else
#define hypre_sin sin
#endif
#endif

#ifndef hypre_cos
#if defined(HYPRE_SINGLE)
#define hypre_cos cosf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_cos cosl
#else
#define hypre_cos cos
#endif
#endif

#ifndef hypre_atan
#if defined(HYPRE_SINGLE)
#define hypre_atan atanf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_atan atanl
#else
#define hypre_atan atan
#endif
#endif

#ifndef hypre_fmod
#if defined(HYPRE_SINGLE)
#define hypre_fmod fmodf
#elif defined(HYPRE_LONG_DOUBLE)
#define hypre_fmod fmodl
#else
#define hypre_fmod fmod
#endif
#endif

#endif /* hypre_GENERAL_HEADER */

