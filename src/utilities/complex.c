/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_COMPLEX

HYPRE_Complex
hypre_conj( HYPRE_Complex value )
{
#ifdef HYPRE_USING_SYCL
   return std::conj(value);
#elif defined(HYPRE_USING_GPU)
   return thrust::conj(value);
#else
   return conj(value);
#endif
}

HYPRE_Real
hypre_cabs( HYPRE_Complex value )
{
#ifdef HYPRE_USING_SYCL
   return std::abs(value);
#elif defined(HYPRE_USING_GPU)
   return thrust::abs(value);
#else
   return cabs(value);
#endif
}

HYPRE_Real
hypre_creal( HYPRE_Complex value )
{
#ifdef HYPRE_USING_SYCL
   return std::real(value);
#elif defined(HYPRE_USING_GPU)
   return thrust::real(value);
#else
   return creal(value);
#endif
}

HYPRE_Real
hypre_cimag( HYPRE_Complex value )
{
#ifdef HYPRE_USING_SYCL
   return std::imag(value);
#elif defined(HYPRE_USING_GPU)
   return thrust::imag(value);
#else
   return cimag(value);
#endif
}

#endif // HYPRE_COMPLEX
