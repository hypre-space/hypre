/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * SStruct scale routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPScale
 *--------------------------------------------------------------------------*/

int
hypre_SStructPScale( double                alpha,
                     hypre_SStructPVector *py )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(py);
   int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructScale(alpha, hypre_SStructPVectorSVector(py, var));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructScale
 *--------------------------------------------------------------------------*/

int
hypre_SStructScale( double               alpha,
                    hypre_SStructVector *y )
{
   int ierr = 0;
   int nparts = hypre_SStructVectorNParts(y);
   int part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPScale(alpha, hypre_SStructVectorPVector(y, part));
   }

   return ierr;
}
