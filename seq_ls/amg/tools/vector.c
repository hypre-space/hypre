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
 * hypre_NewVector
 *--------------------------------------------------------------------------*/

hypre_Vector  *hypre_NewVector(data, size)
double  *data;
int      size;
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

