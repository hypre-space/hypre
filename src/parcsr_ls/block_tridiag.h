/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





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
   double threshold;
   hypre_ParCSRMatrix *A11, *A21, *A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   HYPRE_Solver       precon1, precon2;

} hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate();
HYPRE_Int  hypre_BlockTridiagDestroy(void *);
HYPRE_Int  hypre_BlockTridiagSetup(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
HYPRE_Int  hypre_BlockTridiagSolve(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
HYPRE_Int  hypre_BlockTridiagSetIndexSet(void *, HYPRE_Int, HYPRE_Int *);
HYPRE_Int  hypre_BlockTridiagSetAMGStrengthThreshold(void *, double);
HYPRE_Int  hypre_BlockTridiagSetAMGNumSweeps(void *, HYPRE_Int);
HYPRE_Int  hypre_BlockTridiagSetAMGRelaxType(void *, HYPRE_Int);
HYPRE_Int  hypre_BlockTridiagSetPrintLevel(void *, HYPRE_Int);

#endif

