/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
double  *data;
int      size;
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
int     *data;
int      size;
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
double  value;
{
   double     *vp = hypre_VectorData(v);
   int         n  = hypre_VectorSize(v);

   int         i;


   for (i = 0; i < n; i++)
      vp[i] = value;
}

/*--------------------------------------------------------------------------
 * hypre_InitVectorRandom
 *--------------------------------------------------------------------------*/

void    hypre_InitVectorRandom(v)
hypre_Vector *v;
{
   double     *vp = hypre_VectorData(v);
   int         n  = hypre_VectorSize(v);

   int         i;


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
   double     *xp = hypre_VectorData(x);
   double     *yp = hypre_VectorData(y);
   int         n  = hypre_VectorSize(x);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] = xp[i];
}

/*--------------------------------------------------------------------------
 * hypre_ScaleVector
 *--------------------------------------------------------------------------*/

void     hypre_ScaleVector(alpha, y)
double   alpha;
hypre_Vector  *y;
{
   double     *yp = hypre_VectorData(y);
   int         n  = hypre_VectorSize(y);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] *= alpha;
}

/*--------------------------------------------------------------------------
 * hypre_Axpy
 *--------------------------------------------------------------------------*/

void     hypre_Axpy(alpha, x, y)
double   alpha;
hypre_Vector  *x;
hypre_Vector  *y;
{
   double     *xp = hypre_VectorData(x);
   double     *yp = hypre_VectorData(y);
   int         n  = hypre_VectorSize(x);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] += alpha * xp[i];
}

/*--------------------------------------------------------------------------
 * hypre_InnerProd
 *--------------------------------------------------------------------------*/

double   hypre_InnerProd(x, y)
hypre_Vector  *x;
hypre_Vector  *y;
{
   double     *xp = hypre_VectorData(x);
   double     *yp = hypre_VectorData(y);
   int         n  = hypre_VectorSize(x);

   int         i;

   double      result = 0.0;


   for (i = 0; i < n; i++)
      result += yp[i] * xp[i];
  
   return result;
}

