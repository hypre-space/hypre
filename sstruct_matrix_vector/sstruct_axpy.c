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
 * SStruct axpy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPAxpy( double                alpha,
                    hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(px);
   int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructAxpy(alpha,
                       hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructAxpy
 *--------------------------------------------------------------------------*/

int
hypre_SStructAxpy( double               alpha,
                   hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   int ierr = 0;
   int nparts = hypre_SStructVectorNParts(x);
   int part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPAxpy(alpha,
                         hypre_SStructVectorPVector(x, part),
                         hypre_SStructVectorPVector(y, part));
   }

   return ierr;
}
