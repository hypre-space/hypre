/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct scale routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPScale( HYPRE_Complex         alpha,
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
hypre_SStructScale( HYPRE_Complex        alpha,
                    hypre_SStructVector *y )
{
   HYPRE_Int nparts = hypre_SStructVectorNParts(y);
   HYPRE_Int part;
   HYPRE_Int y_object_type = hypre_SStructVectorObjectType(y);

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
