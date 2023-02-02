/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

HYPRE_Int
hypre_ParVectorZeroBCValues(hypre_ParVector *v,
                            HYPRE_Int       *rows,
                            HYPRE_Int        nrows)
{
   HYPRE_Int   ierr = 0;

   hypre_Vector *v_local = hypre_ParVectorLocalVector(v);

   hypre_SeqVectorZeroBCValues(v_local, rows, nrows);

   return ierr;
}

HYPRE_Int
hypre_SeqVectorZeroBCValues(hypre_Vector *v,
                            HYPRE_Int    *rows,
                            HYPRE_Int     nrows)
{
   HYPRE_Real  *vector_data = hypre_VectorData(v);
   HYPRE_Int      i;
   HYPRE_Int      ierr  = 0;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nrows; i++)
   {
      vector_data[rows[i]] = 0.0;
   }

   return ierr;
}

