/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#endif /* hypre_GENERAL_HEADER */

