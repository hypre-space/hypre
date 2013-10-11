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

#include "_hypre_sstruct_ls.h"

HYPRE_Int
hypre_ParVectorZeroBCValues(hypre_ParVector *v,
                            HYPRE_Int       *rows,
                            HYPRE_Int        nrows)
{
   HYPRE_Int   ierr= 0;

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

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nrows; i++)
      vector_data[rows[i]]= 0.0;

   return ierr;
}

