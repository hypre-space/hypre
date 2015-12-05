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
 * SStruct scale routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPScale( double                alpha,
                     hypre_SStructPVector *py )
{
   HYPRE_Int nvars = hypre_SStructPVectorNVars(py);
   HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructScale(alpha, hypre_SStructPVectorSVector(py, var));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructScale( double               alpha,
                    hypre_SStructVector *y )
{
   HYPRE_Int nparts = hypre_SStructVectorNParts(y);
   HYPRE_Int part;
   HYPRE_Int y_object_type= hypre_SStructVectorObjectType(y);

   if (y_object_type == HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPScale(alpha, hypre_SStructVectorPVector(y, part));
      }
   }

   else if (y_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *y_par;
  
      hypre_SStructVectorConvert(y, &y_par);
      hypre_ParVectorScale(alpha, y_par);
   }

   return hypre_error_flag;
}
