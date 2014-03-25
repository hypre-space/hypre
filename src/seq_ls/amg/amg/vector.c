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

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_NewVector
 *--------------------------------------------------------------------------*/

hypre_Vector  *hypre_NewVector(data, size)
HYPRE_Real  *data;
HYPRE_Int      size;
{
   hypre_Vector     *new;


   new = hypre_CTAlloc(hypre_Vector, 1);

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

/*---------------------------------------------------------------------------
 *    hypre_NewVectorInt
 *--------------------------------------------------------------------------*/

hypre_VectorInt  *hypre_NewVectorInt(data, size)
HYPRE_Int     *data;
HYPRE_Int      size;
{
   hypre_VectorInt     *new;

   new = hypre_CTAlloc(hypre_VectorInt, 1);

   hypre_VectorIntData(new) = data;
   hypre_VectorIntSize(new) = size;

   return new;

}


/*--------------------------------------------------------------------------
 * hypre_FreeVectorInt
 *--------------------------------------------------------------------------*/

void     hypre_FreeVectorInt(vector)
hypre_VectorInt  *vector;
{
   if (vector)
   {
      hypre_TFree(hypre_VectorIntData(vector));
      hypre_TFree(vector);
   }
}

/*--------------------------------------------------------------------------
 * hypre_InitVector
 *--------------------------------------------------------------------------*/

void    hypre_InitVector(v, value)
hypre_Vector *v;
HYPRE_Real  value;
{
   HYPRE_Real *vp = hypre_VectorData(v);
   HYPRE_Int         n  = hypre_VectorSize(v);

   HYPRE_Int         i;


   for (i = 0; i < n; i++)
      vp[i] = value;
}

/*--------------------------------------------------------------------------
 * hypre_InitVectorRandom
 *--------------------------------------------------------------------------*/

void    hypre_InitVectorRandom(v)
hypre_Vector *v;
{
   HYPRE_Real *vp = hypre_VectorData(v);
   HYPRE_Int         n  = hypre_VectorSize(v);

   HYPRE_Int         i;


   for (i = 0; i < n; i++)
      vp[i] = hypre_Rand();
}

/*--------------------------------------------------------------------------
 * hypre_CopyVector
 *--------------------------------------------------------------------------*/

void     hypre_CopyVector(x, y)
hypre_Vector  *x;
hypre_Vector  *y;
{
   HYPRE_Real *xp = hypre_VectorData(x);
   HYPRE_Real *yp = hypre_VectorData(y);
   HYPRE_Int         n  = hypre_VectorSize(x);

   HYPRE_Int         i;


   for (i = 0; i < n; i++)
      yp[i] = xp[i];
}

/*--------------------------------------------------------------------------
 * hypre_ScaleVector
 *--------------------------------------------------------------------------*/

void     hypre_ScaleVector(alpha, y)
HYPRE_Real   alpha;
hypre_Vector  *y;
{
   HYPRE_Real *yp = hypre_VectorData(y);
   HYPRE_Int         n  = hypre_VectorSize(y);

   HYPRE_Int         i;


   for (i = 0; i < n; i++)
      yp[i] *= alpha;
}

/*--------------------------------------------------------------------------
 * hypre_Axpy
 *--------------------------------------------------------------------------*/

void     hypre_Axpy(alpha, x, y)
HYPRE_Real   alpha;
hypre_Vector  *x;
hypre_Vector  *y;
{
   HYPRE_Real *xp = hypre_VectorData(x);
   HYPRE_Real *yp = hypre_VectorData(y);
   HYPRE_Int         n  = hypre_VectorSize(x);

   HYPRE_Int         i;


   for (i = 0; i < n; i++)
      yp[i] += alpha * xp[i];
}

/*--------------------------------------------------------------------------
 * hypre_InnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real   hypre_InnerProd(x, y)
hypre_Vector  *x;
hypre_Vector  *y;
{
   HYPRE_Real *xp = hypre_VectorData(x);
   HYPRE_Real *yp = hypre_VectorData(y);
   HYPRE_Int         n  = hypre_VectorSize(x);

   HYPRE_Int         i;

   HYPRE_Real  result = 0.0;


   for (i = 0; i < n; i++)
      result += yp[i] * xp[i];
  
   return result;
}

