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
 * Header for HYPRE_PCG
 *
 *****************************************************************************/

#ifndef HYPRE_PCG_HEADER
#define HYPRE_PCG_HEADER

/*--------------------------------------------------------------------------
 * HYPRE_PCGData
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

} HYPRE_PCGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the HYPRE_PCGData structure
 *--------------------------------------------------------------------------*/

#define HYPRE_PCGDataMaxIter(pcg_data)       ((pcg_data) -> max_iter)
#define HYPRE_PCGDataTwoNorm(pcg_data)       ((pcg_data) -> two_norm)

#define HYPRE_PCGDataA(pcg_data)             ((pcg_data) -> A)
#define HYPRE_PCGDataP(pcg_data)             ((pcg_data) -> p)
#define HYPRE_PCGDataS(pcg_data)             ((pcg_data) -> s)
#define HYPRE_PCGDataR(pcg_data)             ((pcg_data) -> r)

#define HYPRE_PCGDataPrecond(pcg_data)       ((pcg_data) -> precond)
#define HYPRE_PCGDataPrecondData(pcg_data)   ((pcg_data) -> precond_data)

#define HYPRE_PCGDataNumIterations(pcg_data) ((pcg_data) -> num_iterations)
#define HYPRE_PCGDataNorm(pcg_data)          ((pcg_data) -> norm)
#define HYPRE_PCGDataRelNorm(pcg_data)       ((pcg_data) -> rel_norm)

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* HYPRE_pcg.c */
void HYPRE_PCG P((Vector *x , Vector *b , double tol , void *data ));
 
#undef P

#endif

