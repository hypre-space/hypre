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


/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/

static int Seed = 13579;

#define L 1664525
#define M 1024

/*--------------------------------------------------------------------------
 * hypre_SeedRand:
 *   The seed must always be positive.
 *
 *   Note: the internal seed must be positive and odd, so it is set
 *   to (2*input_seed - 1);
 *--------------------------------------------------------------------------*/

void  hypre_SeedRand(seed)
int   seed;
{
   Seed = (2*seed - 1) % M;
}

/*--------------------------------------------------------------------------
 * hypre_Rand
 *--------------------------------------------------------------------------*/

double  hypre_Rand()
{
   Seed = (L * Seed) % M;

   return ( ((double) Seed) / ((double) M) );
}
