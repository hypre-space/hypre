/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_BLOCKTRIDIAG_HEADER
#define hypre_BLOCKTRIDIAG_HEADER

#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    num_sweeps;
   HYPRE_Int    relax_type;
   HYPRE_Int    *index_set1, *index_set2;
   HYPRE_Int    print_level;
   HYPRE_Real threshold;
   hypre_ParCSRMatrix *A11, *A21, *A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   HYPRE_Solver       precon1, precon2;

} hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate(void);
HYPRE_Int  hypre_BlockTridiagDestroy(void *);
HYPRE_Int  hypre_BlockTridiagSetup(void *, hypre_ParCSRMatrix *,
                                   hypre_ParVector *, hypre_ParVector *);
HYPRE_Int  hypre_BlockTridiagSolve(void *, hypre_ParCSRMatrix *,
                                   hypre_ParVector *, hypre_ParVector *);
HYPRE_Int  hypre_BlockTridiagSetIndexSet(void *, HYPRE_Int, HYPRE_Int *);
HYPRE_Int  hypre_BlockTridiagSetAMGStrengthThreshold(void *, HYPRE_Real);
HYPRE_Int  hypre_BlockTridiagSetAMGNumSweeps(void *, HYPRE_Int);
HYPRE_Int  hypre_BlockTridiagSetAMGRelaxType(void *, HYPRE_Int);
HYPRE_Int  hypre_BlockTridiagSetPrintLevel(void *, HYPRE_Int);

#endif

