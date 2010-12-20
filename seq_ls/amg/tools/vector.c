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





/******************************************************************************
 *
 * Constructors and destructors for vector structure.
 *
 *****************************************************************************/

#include "general.h"
#include "vector.h"


/*--------------------------------------------------------------------------
 * hypre_NewVector
 *--------------------------------------------------------------------------*/

hypre_Vector  *hypre_NewVector(data, size)
double  *data;
HYPRE_Int      size;
{
   hypre_Vector     *new;


   new = hypre_TAlloc(hypre_Vector, 1);

   hypre_VectorData(new) = data;
   hypre_VectorSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * hypre_FreeVector
 *--------------------------------------------------------------------------*/

void     hypre_FreeVector(vector)
hypre_Vector  *vector;
{
   if (vector)
   {
      hypre_TFree(hypre_VectorData(vector));
      hypre_TFree(vector);
   }
}

