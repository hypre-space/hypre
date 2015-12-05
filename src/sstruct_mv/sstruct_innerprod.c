/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
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

HYPRE_Int
hypre_SStructPInnerProd( hypre_SStructPVector *px,
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
      sresult = hypre_StructInnerProd(hypre_SStructPVectorSVector(px, var),
                                      hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructInnerProd( hypre_SStructVector *x,
                        hypre_SStructVector *y,
                        double              *result_ptr )
{
   HYPRE_Int    nparts = hypre_SStructVectorNParts(x);
   double result;
   double presult;
   HYPRE_Int    part;

   HYPRE_Int    x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int    y_object_type= hypre_SStructVectorObjectType(y);
   
   if (x_object_type != y_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
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

   return hypre_error_flag;
}
