/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
   return conj(value);
}

HYPRE_Real
hypre_cabs( HYPRE_Complex value )
{
   return cabs(value);
}

HYPRE_Real
hypre_creal( HYPRE_Complex value )
{
   return creal(value);
}

HYPRE_Real
hypre_cimag( HYPRE_Complex value )
{
   return cimag(value);
}

#endif
