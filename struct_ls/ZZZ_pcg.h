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
 * Header for ZZZ_PCG
 *
 *****************************************************************************/

#ifndef _ZZZ_PCG_HEADER
#define _ZZZ_PCG_HEADER


/*--------------------------------------------------------------------------
 * ZZZ_PCGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      max_iter;
   int      two_norm;

   Matrix  *A;
   Vector  *p;
   Vector  *s;
   Vector  *r;

   int    (*precond)();
   void    *precond_data;

   int      num_iterations;
   double   norm;
   double   rel_norm;

} ZZZ_PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ZZZ_PCGData structure
 *--------------------------------------------------------------------------*/

#define ZZZ_PCGDataMaxIter(pcg_data)       ((pcg_data) -> max_iter)
#define ZZZ_PCGDataTwoNorm(pcg_data)       ((pcg_data) -> two_norm)

#define ZZZ_PCGDataA(pcg_data)             ((pcg_data) -> A)
#define ZZZ_PCGDataP(pcg_data)             ((pcg_data) -> p)
#define ZZZ_PCGDataS(pcg_data)             ((pcg_data) -> s)
#define ZZZ_PCGDataR(pcg_data)             ((pcg_data) -> r)

#define ZZZ_PCGDataPrecond(pcg_data)       ((pcg_data) -> precond)
#define ZZZ_PCGDataPrecondData(pcg_data)   ((pcg_data) -> precond_data)

#define ZZZ_PCGDataNumIterations(pcg_data) ((pcg_data) -> num_iterations)
#define ZZZ_PCGDataNorm(pcg_data)          ((pcg_data) -> norm)
#define ZZZ_PCGDataRelNorm(pcg_data)       ((pcg_data) -> rel_norm)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

void ZZZ_PCG( Vector *x, Vector *b, double  tol, void *data );
