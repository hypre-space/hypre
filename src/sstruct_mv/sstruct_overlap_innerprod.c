/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * SStruct inner product routine for overlapped grids.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPOverlapInnerProd( hypre_SStructPVector *px,
                                hypre_SStructPVector *py,
                                double               *presult_ptr )
{
   HYPRE_Int    nvars = hypre_SStructPVectorNVars(px);
   double presult;
   double sresult;
   HYPRE_Int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = hypre_StructOverlapInnerProd(hypre_SStructPVectorSVector(px, var),
                                             hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructOverlapInnerProd( hypre_SStructVector *x,
                               hypre_SStructVector *y,
                               double              *result_ptr )
{
   HYPRE_Int    nparts = hypre_SStructVectorNParts(x);
   double result;
   double presult;
   HYPRE_Int    part;

   result = 0.0;
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPOverlapInnerProd(hypre_SStructVectorPVector(x, part),
                                     hypre_SStructVectorPVector(y, part), &presult);
      result += presult;
   }

   *result_ptr = result;

   return hypre_error_flag;
}
