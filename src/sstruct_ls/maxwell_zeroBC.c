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

#include "headers.h"

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
   double  *vector_data = hypre_VectorData(v);
   HYPRE_Int      i;
   HYPRE_Int      ierr  = 0;

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
   for (i = 0; i < nrows; i++)
      vector_data[rows[i]]= 0.0;

   return ierr;
}

