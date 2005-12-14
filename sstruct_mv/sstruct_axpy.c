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

   int    x_object_type= hypre_SStructVectorObjectType(x);
   int    y_object_type= hypre_SStructVectorObjectType(y);
                                                                                                                     
   if (x_object_type != y_object_type)
   {
       printf("vector object types different- cannot compute Axpy\n");
       return ierr;
   }

   if (x_object_type == HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPAxpy(alpha,
                            hypre_SStructVectorPVector(x, part),
                            hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      hypre_ParVectorAxpy(alpha, x_par, y_par);
   }

   return ierr;
}
