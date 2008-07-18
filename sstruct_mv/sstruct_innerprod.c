/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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

   int    x_object_type= hypre_SStructVectorObjectType(x);
   int    y_object_type= hypre_SStructVectorObjectType(y);
   
   if (x_object_type != y_object_type)
   {
       printf("vector object types different- cannot compute inner product\n");
       return ierr;
   }

   result = 0.0;

   if ( (x_object_type == HYPRE_SSTRUCT) || (x_object_type == HYPRE_STRUCT) )
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPInnerProd(hypre_SStructVectorPVector(x, part),
                                 hypre_SStructVectorPVector(y, part), &presult);
         result += presult;
      }
   }

   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      result= hypre_ParVectorInnerProd(x_par, y_par);
   }
                                                                                                                
   *result_ptr = result;

   return ierr;
}
