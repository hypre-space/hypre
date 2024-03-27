/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

#include "_hypre_utilities.h"

static HYPRE_Int Seed = 13579;

/*-------------------------------------------------------------------------------
 * Static global variable: Seed
 * ``... all initial seeds between 1 and 2147483646 (2^31-2) are equally valid''
 *-------------------------------------------------------------------------------*/

#define a  16807      /* 7^5 */
#define m  2147483647 /* 2*31 - 1 */
#define q  127773     /* m div a */
#define r  2836       /* m mod a */

/*--------------------------------------------------------------------------
 * Initializes the pseudo-random number generator to a place in the sequence.
 *
 * @param seed an HYPRE_Int containing the seed for the RNG.
 *--------------------------------------------------------------------------*/
void hypre_SeedRand( HYPRE_Int seed )
{
   /* RL: seed must be between 1 and 2^31-2 */
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

/*--------------------------------------------------------------------------
 * Computes the next pseudo-random number in the sequence using the global
 * variable Seed.
 *
 * @return a HYPRE_Int between (0, 2147483647]
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_RandI( void )
{
   HYPRE_Int  low, high, test;
   high = Seed / q;
   low = Seed % q;
   test = a * low - r * high;
   if (test > 0)
   {
      Seed = test;
   }
   else
   {
      Seed = test + m;
   }

   return Seed;
}

/*--------------------------------------------------------------------------
 * Computes the next pseudo-random number in the sequence using the global
 * variable Seed.
 *
 * @return a HYPRE_Real containing the next number in the sequence divided by
 * 2147483647 so that the numbers are in (0, 1].
 *--------------------------------------------------------------------------*/
HYPRE_Real hypre_Rand( void )
{
   return ((HYPRE_Real)(hypre_RandI()) / (HYPRE_Real)m);
}

