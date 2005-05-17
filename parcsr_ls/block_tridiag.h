/*BHEADER**********************************************************************
 * (c) 2005   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef hypre_BLOCKTRIDIAG_HEADER
#define hypre_BLOCKTRIDIAG_HEADER

#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    num_sweeps;  
   int    relax_type;   
   int    *index_set1, *index_set2;
   int    print_level;
   double threshold;
   hypre_ParCSRMatrix *A11, *A21, *A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   HYPRE_Solver       precon1, precon2;

} hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate();
int  hypre_BlockTridiagDestroy(void *);
int  hypre_BlockTridiagSetup(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
int  hypre_BlockTridiagSolve(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
int  hypre_BlockTridiagSetIndexSet(void *, int, int *);
int  hypre_BlockTridiagSetAMGStrengthThreshold(void *, double);
int  hypre_BlockTridiagSetAMGNumSweeps(void *, int);
int  hypre_BlockTridiagSetAMGRelaxType(void *, int);
int  hypre_BlockTridiagSetPrintLevel(void *, int);

#endif

