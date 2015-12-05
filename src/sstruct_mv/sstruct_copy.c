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
 * SStruct copy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPCopy( hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   HYPRE_Int nvars = hypre_SStructPVectorNVars(px);
   HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructCopy(hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPartialPCopy: Copy the components on only a subset of the
 * pgrid. For each box of an sgrid, an array of subboxes are copied.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPartialPCopy( hypre_SStructPVector *px,
                           hypre_SStructPVector *py,
                           hypre_BoxArrayArray **array_boxes )
{
   HYPRE_Int nvars = hypre_SStructPVectorNVars(px);
   hypre_BoxArrayArray  *boxes;
   HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      boxes= array_boxes[var];
      hypre_StructPartialCopy(hypre_SStructPVectorSVector(px, var),
                              hypre_SStructPVectorSVector(py, var),
                              boxes);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructCopy( hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   HYPRE_Int nparts = hypre_SStructVectorNParts(x);
   HYPRE_Int part;

   HYPRE_Int x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int y_object_type= hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }


   if (x_object_type == HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPCopy(hypre_SStructVectorPVector(x, part),
                            hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      hypre_ParVectorCopy(x_par, y_par);
   }

   return hypre_error_flag;
}
