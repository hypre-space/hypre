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
 * SStruct copy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPCopy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPCopy( hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(px);
   int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructCopy(hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPartialPCopy: Copy the components on only a subset of the
 * pgrid. For each box of an sgrid, an array of subboxes are copied.
 *--------------------------------------------------------------------------*/

int
hypre_SStructPartialPCopy( hypre_SStructPVector *px,
                           hypre_SStructPVector *py,
                           hypre_BoxArrayArray **array_boxes )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(px);
   hypre_BoxArrayArray  *boxes;
   int var;

   for (var = 0; var < nvars; var++)
   {
      boxes= array_boxes[var];
      hypre_StructPartialCopy(hypre_SStructPVectorSVector(px, var),
                              hypre_SStructPVectorSVector(py, var),
                              boxes);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructCopy
 *--------------------------------------------------------------------------*/

int
hypre_SStructCopy( hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   int ierr = 0;

   int nparts = hypre_SStructVectorNParts(x);
   int part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPCopy(hypre_SStructVectorPVector(x, part),
                         hypre_SStructVectorPVector(y, part));
   }

   return ierr;
}
