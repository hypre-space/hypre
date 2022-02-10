/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct inner product routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPInnerProd( hypre_SStructPVector *px,
                         hypre_SStructPVector *py,
                         HYPRE_Real           *presult_ptr )
{
   HYPRE_Real   local_result;
   HYPRE_Real   global_result;

   hypre_SStructPInnerProdLocal(px, py, &local_result);
   hypre_MPI_Allreduce(&local_result, &global_result, 1,
                       HYPRE_MPI_REAL, hypre_MPI_SUM,
                       hypre_SStructPVectorComm(px));

   *presult_ptr = global_result;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPInnerProdLocal
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPInnerProdLocal( hypre_SStructPVector *px,
                              hypre_SStructPVector *py,
                              HYPRE_Real           *presult_ptr )
{
   HYPRE_Int    nvars = hypre_SStructPVectorNVars(px);
   HYPRE_Real   presult;
   HYPRE_Real   sresult;
   HYPRE_Int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = hypre_StructInnerProdLocal(hypre_SStructPVectorSVector(px, var),
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
                        HYPRE_Real          *result_ptr )
{
   HYPRE_Int    nparts = hypre_SStructVectorNParts(x);
   HYPRE_Real   local_result;
   HYPRE_Real   result;
   HYPRE_Real   presult;
   HYPRE_Int    part;

   HYPRE_Int    x_object_type = hypre_SStructVectorObjectType(x);
   HYPRE_Int    y_object_type = hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if ( (x_object_type == HYPRE_SSTRUCT) || (x_object_type == HYPRE_STRUCT) )
   {
      local_result = 0.0;
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPInnerProdLocal(hypre_SStructVectorPVector(x, part),
                                      hypre_SStructVectorPVector(y, part),
                                      &presult);
         local_result += presult;
      }

      hypre_MPI_Allreduce(&local_result, &result, 1,
                          HYPRE_MPI_REAL, hypre_MPI_SUM,
                          hypre_SStructVectorComm(x));
   }

   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      result = hypre_ParVectorInnerProd(x_par, y_par);
   }

   *result_ptr = result;

   return hypre_error_flag;
}
