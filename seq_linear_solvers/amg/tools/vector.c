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

#include "general.h"
#include "vector.h"


/*--------------------------------------------------------------------------
 * NewVector
 *--------------------------------------------------------------------------*/

Vector  *NewVector(data, size)
double  *data;
int      size;
{
   Vector     *new;


   new = talloc(Vector, 1);

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

