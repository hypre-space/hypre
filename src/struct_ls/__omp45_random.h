#ifndef OMP45_RANDOM_H
#define OMP45_RANDOM_H

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
 * This file contains routines that implement a pseudo-random number generator
 * detailed in the following paper.
 *
 * @article{RNG_Park_Miller,
 *   author = {S. K. Park and K. W. Miller},
 *   title = {Random number generators: good ones are hard to find},
 *   journal = {Commun. ACM},
 *   volume = {31},
 *   number = {10},
 *   year = {1988},
 *   pages = {1192--1201},
 * }
 *
 * This RNG has been shown to appear fairly random, it is a full period
 * generating function (the sequence uses all of the values available to it up
 * to 2147483647), and can be implemented on any architecture using 32-bit
 * integers. The implementation in this file will not overflow for 32-bit
 * arithmetic, which all modern computers should support.
 *
 * @author David Alber
 * @date March 2005
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/
#pragma omp declare target
static HYPRE_Int Seed = 13579;
#pragma omp end declare target

#define a  16807
#define m  2147483647
#define q  127773
#define r  2836

/*--------------------------------------------------------------------------
 * Initializes the pseudo-random number generator to a place in the sequence.
 *
 * @param seed an HYPRE_Int containing the seed for the RNG.
 *--------------------------------------------------------------------------*/
#pragma omp declare target
static inline void  hypre_SeedRand_inline( HYPRE_Int seed )
{
   if (seed < 1)
   {
      seed = 1;
   }
   else if (seed >= m)
   {
      seed = m - 1;
   }

   Seed = seed;
}
#pragma omp end declare target

/*--------------------------------------------------------------------------
 * Computes the next pseudo-random number in the sequence using the global
 * variable Seed.
 *
 * @return a HYPRE_Int between (0, 2147483647]
 *--------------------------------------------------------------------------*/
#pragma omp declare target
static inline HYPRE_Int  hypre_RandI_inline()
{
   HYPRE_Int  low, high, test;

   high = Seed / q;
   low = Seed % q;
   test = a * low - r * high;
   if(test > 0)
   {
      Seed = test;
   }
   else
   {
      Seed = test + m;
   }

   return Seed;
}
#pragma omp end declare target

/*--------------------------------------------------------------------------
 * Computes the next pseudo-random number in the sequence using the global
 * variable Seed.
 *
 * @return a HYPRE_Real containing the next number in the sequence divided by
 * 2147483647 so that the numbers are in (0, 1].
 *--------------------------------------------------------------------------*/
#pragma omp declare target
static inline HYPRE_Real hypre_Rand_inline()
{
   return ((HYPRE_Real)(hypre_RandI_inline()) / m);
}
#pragma omp end declare target

#define hypre_SeedRand hypre_SeedRand_inline
#define hypre_RandI    hypre_RandI_inline
#define hypre_Rand     hypre_Rand_inline
#endif
