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
 * Routines for generating random numbers.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/

static HYPRE_Int Seed = 13579;

#define M 1048576
#define L 1027

/*--------------------------------------------------------------------------
 * hypre_SeedRand:
 *   The seed must always be positive.
 *
 *   Note: the internal seed must be positive and odd, so it is set
 *   to (2*input_seed - 1);
 *--------------------------------------------------------------------------*/

void  hypre_SeedRand(seed)
HYPRE_Int   seed;
{
   Seed = (2*seed - 1) % M;
}

/*--------------------------------------------------------------------------
 * hypre_Rand
 *--------------------------------------------------------------------------*/

HYPRE_Real  hypre_Rand()
{
   Seed = (L * Seed) % M;

   return ( ((HYPRE_Real) Seed) / ((HYPRE_Real) M) );
}
