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
 * Constructors and destructors for vector structure.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * NewVector
 *--------------------------------------------------------------------------*/

Vector  *NewVector(data, size)
double  *data;
int      size;
{
   Vector     *new;


   new = ctalloc(Vector, 1);

   VectorData(new) = data;
   VectorSize(new) = size;

   return new;
}

/*--------------------------------------------------------------------------
 * FreeVector
 *--------------------------------------------------------------------------*/

void     FreeVector(vector)
Vector  *vector;
{
   if (vector)
   {
      tfree(VectorData(vector));
      tfree(vector);
   }
}

/*--------------------------------------------------------------------------
 * InitVector
 *--------------------------------------------------------------------------*/

void    InitVector(v, value)
Vector *v;
double  value;
{
   double     *vp = VectorData(v);
   int         n  = VectorSize(v);

   int         i;


   for (i = 0; i < n; i++)
      vp[i] = value;
}

/*--------------------------------------------------------------------------
 * InitVectorRandom
 *--------------------------------------------------------------------------*/

void    InitVectorRandom(v)
Vector *v;
{
   double     *vp = VectorData(v);
   int         n  = VectorSize(v);

   int         i;


   for (i = 0; i < n; i++)
      vp[i] = Rand();
}

/*--------------------------------------------------------------------------
 * CopyVector
 *--------------------------------------------------------------------------*/

void     CopyVector(x, y)
Vector  *x;
Vector  *y;
{
   double     *xp = VectorData(x);
   double     *yp = VectorData(y);
   int         n  = VectorSize(x);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] = xp[i];
}

/*--------------------------------------------------------------------------
 * ScaleVector
 *--------------------------------------------------------------------------*/

void     ScaleVector(alpha, y)
double   alpha;
Vector  *y;
{
   double     *yp = VectorData(y);
   int         n  = VectorSize(y);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] *= alpha;
}

/*--------------------------------------------------------------------------
 * Axpy
 *--------------------------------------------------------------------------*/

void     Axpy(alpha, x, y)
double   alpha;
Vector  *x;
Vector  *y;
{
   double     *xp = VectorData(x);
   double     *yp = VectorData(y);
   int         n  = VectorSize(x);

   int         i;


   for (i = 0; i < n; i++)
      yp[i] += alpha * xp[i];
}

/*--------------------------------------------------------------------------
 * InnerProd
 *--------------------------------------------------------------------------*/

double   InnerProd(x, y)
Vector  *x;
Vector  *y;
{
   double     *xp = VectorData(x);
   double     *yp = VectorData(y);
   int         n  = VectorSize(x);

   int         i;

   double      result = 0.0;


   for (i = 0; i < n; i++)
      result += yp[i] * xp[i];
  
   return result;
}

