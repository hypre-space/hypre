/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct axpy routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPAxpy( HYPRE_Complex         alpha,
                    hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   HYPRE_Int nvars = hypre_SStructPVectorNVars(px);
   HYPRE_Int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructAxpy(alpha,
                       hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_SStructPVectorElmdivpy
 *
 * This function computes
 *
 *   y = alpha*x./z + beta*y
 *
 * for a part if it is active.
 *----------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPVectorElmdivpy( HYPRE_Complex         alpha,
                              hypre_SStructPVector *px,
                              hypre_SStructPVector *pz,
                              HYPRE_Complex         beta,
                              hypre_SStructPVector *py )
{
   HYPRE_Int   nvars = hypre_SStructPVectorNVars(px);
   HYPRE_Int   var, active;

   hypre_SStructPGrid *pgrid;
   hypre_StructVector *sx;
   hypre_StructVector *sy;
   hypre_StructVector *sz;

   for (var = 0; var < nvars; var++)
   {
      pgrid  = hypre_SStructPVectorPGrid(px);
      active = hypre_SStructPGridActive(pgrid, var);

      if (active)
      {
         sx = hypre_SStructPVectorSVector(px, var);
         sy = hypre_SStructPVectorSVector(py, var);
         sz = hypre_SStructPVectorSVector(pz, var);

         hypre_StructVectorElmdivpy(alpha, sx, sz, beta, sy);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructAxpy( HYPRE_Complex        alpha,
                   hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   HYPRE_Int nparts = hypre_SStructVectorNParts(x);
   HYPRE_Int part;

   HYPRE_Int x_object_type = hypre_SStructVectorObjectType(x);
   HYPRE_Int y_object_type = hypre_SStructVectorObjectType(y);

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

   return hypre_error_flag;
}

/*------------------------------------------------------------------
 * hypre_SStructVectorElmdivpy
 *
 * y = alpha[part]*x./z + beta[part]*y
 *----------------------------------------------------------------*/

HYPRE_Int
hypre_SStructVectorElmdivpy( HYPRE_Complex       *alpha,
                             hypre_SStructVector *x,
                             hypre_SStructVector *z,
                             HYPRE_Complex       *beta,
                             hypre_SStructVector *y )
{
   HYPRE_Int  nparts = hypre_SStructVectorNParts(x);
   HYPRE_Int  part;

   HYPRE_Int  x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int  y_object_type= hypre_SStructVectorObjectType(y);
   HYPRE_Int  z_object_type= hypre_SStructVectorObjectType(z);

   hypre_ParVector *x_par;
   hypre_ParVector *y_par;
   hypre_ParVector *z_par;

   hypre_SStructPVector *px;
   hypre_SStructPVector *py;
   hypre_SStructPVector *pz;

   /* Sanity check */
   if ((x_object_type != y_object_type) || (x_object_type != z_object_type))
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (x_object_type == HYPRE_SSTRUCT || x_object_type == HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);
         pz = hypre_SStructVectorPVector(z, part);

         hypre_SStructPVectorElmdivpy(alpha[part], px, pz, beta[part], py);
      }
   }
   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);
      hypre_SStructVectorConvert(z, &z_par);

      if ((alpha[0] != 1.0) || (beta[0] != 1.0))
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"{alpha, beta} != 1.0 not implemented!");
         return hypre_error_flag;
      }

      hypre_ParVectorElmdivpy(x_par, z_par, y_par);
   }

   return hypre_error_flag;
}
