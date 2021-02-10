/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*==========================================================================
 * PMatvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructPMatvecData_struct
{
   HYPRE_Int    nvars;
   HYPRE_Int    transpose;
   void      ***smatvec_data;

} hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = hypre_CTAlloc(hypre_SStructPMatvecData, 1, HYPRE_MEMORY_HOST);
   (pmatvec_data -> nvars)     = 0;
   (pmatvec_data -> transpose) = 0;

   *pmatvec_vdata_ptr = (void *) pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecSetTranspose
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecSetTranspose( void      *pmatvec_vdata,
                                  HYPRE_Int  transpose )
{
   hypre_SStructPMatvecData  *pmatvec_data = (hypre_SStructPMatvecData *) pmatvec_vdata;

   (pmatvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecSetSkipDiag
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecSetSkipDiag( void      *pmatvec_vdata,
                                 HYPRE_Int  skip_diag )
{
   hypre_SStructPMatvecData *pmatvec_data = (hypre_SStructPMatvecData *) pmatvec_vdata;
   HYPRE_Int                 nvars        = (pmatvec_data -> nvars);
   HYPRE_Int                 var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructMatvecSetSkipDiag(pmatvec_data -> smatvec_data[var][var],
                                    skip_diag);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   HYPRE_Int                   vi, vj;

   if (pmatvec_data)
   {
      nvars        = (pmatvec_data -> nvars);
      smatvec_data = (pmatvec_data -> smatvec_data);
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            if (smatvec_data[vi][vj] != NULL)
            {
               hypre_StructMatvecDestroy(smatvec_data[vi][vj]);
            }
         }
         hypre_TFree(smatvec_data[vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(smatvec_data, HYPRE_MEMORY_HOST);
      hypre_TFree(pmatvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecSetup( void                 *pmatvec_vdata,
                           hypre_SStructPMatrix *pA,
                           hypre_SStructPVector *px )
{
   hypre_SStructPMatvecData    *pmatvec_data = (hypre_SStructPMatvecData *) pmatvec_vdata;
   HYPRE_Int                    transpose = (pmatvec_data -> transpose);
   HYPRE_Int                    nvars;
   void                      ***smatvec_data;
   hypre_StructMatrix          *sA;
   hypre_StructVector          *sx;
   HYPRE_Int                    vi, vj;

   nvars = hypre_SStructPMatrixNVars(pA);
   smatvec_data = hypre_TAlloc(void **, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatvec_data[vi] = hypre_TAlloc(void *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
         sx = hypre_SStructPVectorSVector(px, vj);
         smatvec_data[vi][vj] = NULL;
         if (sA != NULL)
         {
            smatvec_data[vi][vj] = hypre_StructMatvecCreate();

            hypre_StructMatvecSetTranspose(smatvec_data[vi][vj], transpose);
            hypre_StructMatvecSetup(smatvec_data[vi][vj], sA, sx);
         }
      }
   }
   (pmatvec_data -> nvars)        = nvars;
   (pmatvec_data -> smatvec_data) = smatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecCompute( void                 *pmatvec_vdata,
                             HYPRE_Complex         alpha,
                             hypre_SStructPMatrix *pA,
                             hypre_SStructPVector *px,
                             HYPRE_Complex         beta,
                             hypre_SStructPVector *py )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   HYPRE_Int                   nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                       *sdata;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   hypre_StructVector         *sy;
   HYPRE_Int                   vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      sy = hypre_SStructPVectorSVector(py, vi);

      /* diagonal block computation */
      if (smatvec_data[vi][vi] != NULL)
      {
         sdata = smatvec_data[vi][vi];
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         hypre_StructMatvecCompute(sdata, alpha, sA, sx, beta, sy);
      }
      else
      {
         hypre_StructScale(beta, sy);
      }

      /* off-diagonal block computation */
      for (vj = 0; vj < nvars; vj++)
      {
         if ((smatvec_data[vi][vj] != NULL) && (vj != vi))
         {
            sdata = smatvec_data[vi][vj];
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sx = hypre_SStructPVectorSVector(px, vj);
            hypre_StructMatvecCompute(sdata, alpha, sA, sx, 1.0, sy);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecDiagScale
 *
 * y = alpha*inv(A_D)*x + beta*y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecDiagScale( HYPRE_Complex         alpha,
                               hypre_SStructPMatrix *A,
                               hypre_SStructPVector *x,
                               HYPRE_Complex         beta,
                               hypre_SStructPVector *y )
{
   hypre_StructMatrix  *sA;
   hypre_StructVector  *sx, *sy;
   HYPRE_Int            vi, nvars;

   nvars = hypre_SStructPMatrixNVars(A);
   for (vi = 0; vi < nvars; vi++)
   {
      sA = hypre_SStructPMatrixSMatrix(A, vi, vi);
      sx = hypre_SStructPVectorSVector(x, vi);
      sy = hypre_SStructPVectorSVector(y, vi);

      hypre_StructMatvecDiagScale(alpha, sA, sx, beta, sy);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvec( HYPRE_Complex         alpha,
                      hypre_SStructPMatrix *pA,
                      hypre_SStructPVector *px,
                      HYPRE_Complex         beta,
                      hypre_SStructPVector *py )
{
   void *pmatvec_data;

   hypre_SStructPMatvecCreate(&pmatvec_data);
   hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   hypre_SStructPMatvecCompute(pmatvec_data, alpha, pA, px, beta, py);
   hypre_SStructPMatvecDestroy(pmatvec_data);

   return hypre_error_flag;
}

/*==========================================================================
 * Matvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructMatvecData_struct
{
   HYPRE_Int    nparts;
   HYPRE_Int    transpose;
   HYPRE_Int   *active;
   void       **pmatvec_data;
} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   hypre_SStructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_SStructMatvecData, 1, HYPRE_MEMORY_HOST);
   (matvec_data -> nparts)    = 0;
   (matvec_data -> transpose) = 0;
   (matvec_data -> active)    = NULL;

   *matvec_vdata_ptr = (void *) matvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetTranspose
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatvecSetTranspose( void      *matvec_vdata,
                                 HYPRE_Int  transpose )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData *) matvec_vdata;

   (matvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetSkipDiag
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatvecSetSkipDiag( void      *matvec_vdata,
                                HYPRE_Int  skip_diag )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData *) matvec_vdata;
   HYPRE_Int                 nparts      = (matvec_data -> nparts);
   HYPRE_Int                 part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatvecSetSkipDiag(matvec_data -> pmatvec_data[part],
                                      skip_diag);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetActiveParts
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatvecSetActiveParts( void      *matvec_vdata,
                                   HYPRE_Int *active )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData *) matvec_vdata;
   HYPRE_Int                 nparts      = (matvec_data -> nparts);
   HYPRE_Int                 part;

   if (!(matvec_data -> active))
   {
      (matvec_data -> active) = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
   }

   for (part = 0; part < nparts; part++)
   {
      (matvec_data -> active[part]) = active[part];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetAllPartsActive
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatvecSetAllPartsActive( void *matvec_vdata )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData *) matvec_vdata;
   HYPRE_Int                 nparts      = (matvec_data -> nparts);
   HYPRE_Int                 part;

   if ((matvec_data -> active))
   {
      for (part = 0; part < nparts; part++)
      {
         (matvec_data -> active[part]) = 1;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecSetup( void                *matvec_vdata,
                          hypre_SStructMatrix *A,
                          hypre_SStructVector *x )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 transpose   = (matvec_data -> transpose);

   HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   HYPRE_Int                *active;
   HYPRE_Int                 part;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   nparts = hypre_SStructMatrixNParts(A);
   pmatvec_data = hypre_TAlloc(void *, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatvecCreate(&pmatvec_data[part]);
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);

      hypre_SStructPMatvecSetTranspose(pmatvec_data[part], transpose);
      hypre_SStructPMatvecSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;

   /* Set active parts */
   if (!(matvec_data -> active))
   {
      active = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         active[part] = 1;
      }
      (matvec_data -> active) = active;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCompute( void                *matvec_vdata,
                            HYPRE_Complex        alpha,
                            hypre_SStructMatrix *A,
                            hypre_SStructVector *x,
                            HYPRE_Complex        beta,
                            hypre_SStructVector *y )
{
   hypre_SStructMatvecData  *matvec_data  = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 nparts       = (matvec_data -> nparts);
   HYPRE_Int                 transpose    = (matvec_data -> transpose);
   HYPRE_Int                *active       = (matvec_data -> active);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   hypre_SStructPVector     *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx;
   hypre_ParVector          *pary;

   HYPRE_Int                 x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int                 A_object_type= hypre_SStructMatrixObjectType(A);
   HYPRE_Int                 part;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (x_object_type != A_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   if ( (x_object_type == HYPRE_SSTRUCT) || (x_object_type == HYPRE_STRUCT) )
   {
      /* do S-matrix computations */
      for (part = 0; part < nparts; part++)
      {
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);

         if (active[part])
         {
            pdata = pmatvec_data[part];

            pA = hypre_SStructMatrixPMatrix(A, part);
            hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, py);
         }
         else
         {
            hypre_SStructPCopy(px, py);
         }
      }

      if (x_object_type == HYPRE_SSTRUCT)
      {

         /* do U-matrix computations */

         /* GEC1002 the data chunk pointed by the local-parvectors
          *  inside the semistruct vectors x and y is now identical to the
          *  data chunk of the structure vectors x and y. The role of the function
          *  convert is to pass the addresses of the data chunk
          *  to the parx and pary. */

         hypre_SStructVectorConvert(x, &parx);
         hypre_SStructVectorConvert(y, &pary);

         if (transpose)
         {
            hypre_ParCSRMatrixMatvecT(alpha, parcsrA, parx, 1.0, pary);
         }
         else
         {
            hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);
         }

         /* dummy functions since there is nothing to restore  */

         hypre_SStructVectorRestore(x, NULL);
         hypre_SStructVectorRestore(y, pary);

         parx = NULL;
      }

  }

  else if (x_object_type == HYPRE_PARCSR)
  {
      hypre_SStructVectorConvert(x, &parx);
      hypre_SStructVectorConvert(y, &pary);

      hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, beta, pary);

      hypre_SStructVectorRestore(x, NULL);
      hypre_SStructVectorRestore(y, pary);

      parx = NULL;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecDiagScale
 *
 * y = alpha*inv(A_D)*x + beta*y
 *
 * TODO: Add UMatrix contribution to inv(A_D)
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatvecDiagScale( HYPRE_Complex       *alpha,
                              hypre_SStructMatrix *A,
                              hypre_SStructVector *x,
                              HYPRE_Complex       *beta,
                              hypre_SStructVector *y )
{
   HYPRE_Int                 nparts = hypre_SStructMatrixNParts(A);

   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px, *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx;
   hypre_ParVector          *pary;

   HYPRE_Int                 A_object_type= hypre_SStructMatrixObjectType(A);
   HYPRE_Int                 x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int                 y_object_type= hypre_SStructVectorObjectType(y);
   HYPRE_Int                 part;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Safety checks */
   if (x_object_type != A_object_type)
   {
      hypre_error_in_arg(1);
      hypre_error_in_arg(2);

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }
   if (x_object_type != y_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(4);

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   if (x_object_type == HYPRE_SSTRUCT || x_object_type == HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);

         if (beta == NULL)
         {
            hypre_SStructPMatvecDiagScale(alpha[part], pA, px, 0.0, py);
         }
         else
         {
            hypre_SStructPMatvecDiagScale(alpha[part], pA, px, beta[part], py);
         }
      } /* loop on parts */
   }
   else  /* x_object_type == HYPRE_PARCSR */
   {
      hypre_SStructVectorConvert(x, &parx);
      hypre_SStructVectorConvert(y, &pary);

      if (beta == NULL)
      {
         hypre_ParCSRMatrixMatvecDiagScale(alpha[0], parcsrA, parx, 0.0, pary);
      }
      else
      {
         hypre_ParCSRMatrixMatvecDiagScale(alpha[0], parcsrA, parx, beta[0], pary);
      }

      hypre_SStructVectorRestore(x, parx);
      hypre_SStructVectorRestore(y, pary);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   HYPRE_Int                 part;

   if (matvec_data)
   {
      nparts       = (matvec_data -> nparts);
      pmatvec_data = (matvec_data -> pmatvec_data);
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPMatvecDestroy(pmatvec_data[part]);
      }
      hypre_TFree(matvec_data -> active, HYPRE_MEMORY_HOST);
      hypre_TFree(pmatvec_data, HYPRE_MEMORY_HOST);
      hypre_TFree(matvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvec( HYPRE_Complex        alpha,
                     hypre_SStructMatrix *A,
                     hypre_SStructVector *x,
                     HYPRE_Complex        beta,
                     hypre_SStructVector *y )
{
   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecSetup(matvec_data, A, x);
   hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_SStructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
