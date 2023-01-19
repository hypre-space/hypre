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
   return hypre_conjugate(value);
}

HYPRE_Real
hypre_cabs( HYPRE_Complex value )
{
   return hypre_complex_abs(value);
}

HYPRE_Real
hypre_creal( HYPRE_Complex value )
{
   return hypre_complex_real(value);
}

HYPRE_Real
hypre_cimag( HYPRE_Complex value )
{
   return hypre_complex_imag(value);
}

HYPRE_Complex
hypre_csqrt( HYPRE_Complex value )
{
   return hypre_complex_sqrt(value);
}

#endif
