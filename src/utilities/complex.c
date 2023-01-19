/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_COMPLEX

#include <complex.h>

HYPRE_Complex
hypre_conj( HYPRE_Complex value )
{
#if defined(HYPRE_SINGLE)
   return conjf(value);
#elif defined(HYPRE_LONG_DOUBLE)
   return conjl(value);
#else
   return conj(value);
#endif
}

HYPRE_Real
hypre_cabs( HYPRE_Complex value )
{
#if defined(HYPRE_SINGLE)
   return cabsf(value);
#elif defined(HYPRE_LONG_DOUBLE)
   return cabsl(value);
#else
   return cabs(value);
#endif
}

HYPRE_Real
hypre_creal( HYPRE_Complex value )
{
#if defined(HYPRE_SINGLE)
   return crealf(value);
#elif defined(HYPRE_LONG_DOUBLE)
   return creall(value);
#else
   return creal(value);
#endif
}

HYPRE_Real
hypre_cimag( HYPRE_Complex value )
{
#if defined(HYPRE_SINGLE)
   return cimagf(value);
#elif defined(HYPRE_LONG_DOUBLE)
   return cimagl(value);
#else
   return cimag(value);
#endif
}

HYPRE_Complex
hypre_csqrt( HYPRE_Complex value )
{
#if defined(HYPRE_SINGLE)
   return csqrtf(value);
#elif defined(HYPRE_LONG_DOUBLE)
   return csqrtl(value);
#else
   return csqrt(value);
#endif
}

#endif
