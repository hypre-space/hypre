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
 * SStruct inner product routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructPInnerProd( hypre_SStructPVector *px,
                         hypre_SStructPVector *py,
                         double               *presult_ptr )
{
   int    ierr = 0;
   int    nvars = hypre_SStructPVectorNVars(px);
   double presult;
   double sresult;
   int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = hypre_StructInnerProd(hypre_SStructPVectorSVector(px, var),
                                      hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructInnerProd( hypre_SStructVector *x,
                        hypre_SStructVector *y,
                        double              *result_ptr )
{
   int    ierr = 0;
   int    nparts = hypre_SStructVectorNParts(x);
   double result;
   double presult;
   int    part;

   result = 0.0;
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPInnerProd(hypre_SStructVectorPVector(x, part),
                              hypre_SStructVectorPVector(y, part), &presult);
      result += presult;
   }

   *result_ptr = result;

   return ierr;
}
