/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Routines for generating random numbers.
 *
 *****************************************************************************/

#include "amg.h"


/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/

static int Seed = 13579;

#define M 1048576
#define L 1027

/*--------------------------------------------------------------------------
 * SeedRand:
 *   The seed must always be positive.
 *
 *   Note: the internal seed must be positive and odd, so it is set
 *   to (2*input_seed - 1);
 *--------------------------------------------------------------------------*/

void  SeedRand(seed)
int   seed;
{
   Seed = (2*seed - 1) % M;
}

/*--------------------------------------------------------------------------
 * Rand
 *--------------------------------------------------------------------------*/

double  Rand()
{
   Seed = (L * Seed) % M;

   return ( ((double) Seed) / ((double) M) );
}
