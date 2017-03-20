/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/

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

typedef struct
{
   HYPRE_Int     nvars;
   void ***smatvec_data;

} hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecTSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecTSetup(
      void                 *pmatvec_vdata,
      hypre_SStructPMatrix *pA,
      hypre_SStructPVector *px
      )
{
   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
   HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   HYPRE_Int                   vi, vj;

   nvars = hypre_SStructPMatrixNVars(pA);

   smatvec_data = hypre_TAlloc(void **, nvars);

   for (vi = 0; vi < nvars; vi++)
   {
      smatvec_data[vi] = hypre_TAlloc(void *, nvars);

      for (vj = 0; vj < nvars; vj++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
         sx = hypre_SStructPVectorSVector(px, vj);
         smatvec_data[vi][vj] = NULL;

         if (sA != NULL)
         {
            smatvec_data[vi][vj] = hypre_StructMatvecCreate();
            hypre_StructMatvecTSetup(smatvec_data[vi][vj], sA, sx);
         }
      }
   }

   (pmatvec_data -> nvars)        = nvars;

   (pmatvec_data -> smatvec_data) = smatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecTCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecTCompute(
      void                 *pmatvec_vdata,
      HYPRE_Complex         alpha,
      hypre_SStructPMatrix *pA,
      hypre_SStructPVector *px,
      HYPRE_Complex         beta,
      hypre_SStructPVector *py
      )
{
   hypre_SStructPMatvecData *pmatvec_data = pmatvec_vdata;
   HYPRE_Int                 nvars        = (pmatvec_data -> nvars);
   void                   ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                     *sdata;
   hypre_StructMatrix       *sA;
   hypre_StructVector       *sx;
   hypre_StructVector       *sy;

   HYPRE_Int                 vi, vj;

   /* diagonal block computation (including initialization) */
   for (vi = 0; vi < nvars; vi++)
   {
      if (smatvec_data[vi][vi] != NULL)
      {
         sdata = smatvec_data[vi][vi];
         sy = hypre_SStructPVectorSVector(py, vi);
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         hypre_StructMatvecTCompute(sdata, alpha, sA, sx, beta, sy);
      }
      else
      {
         sy = hypre_SStructPVectorSVector(py, vi);
         hypre_StructScale(beta, sy);
      }
   }

   /* off-diagonal block computation */
   for ( vi = 0; vi < nvars; vi++ )
   {
      for (vj = 0; vj < nvars; vj++)
      {
         if ((smatvec_data[vi][vj] != NULL) && (vj != vi))
         {
            // swap blocks here (and swap block elements in StructMatvecT)
            sy = hypre_SStructPVectorSVector(py, vj);
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sdata = smatvec_data[vi][vj];
            sx = hypre_SStructPVectorSVector(px, vi);

            hypre_StructMatvecTCompute(sdata, alpha, sA, sx, 1.0, sy);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecT(
      HYPRE_Complex         alpha,
      hypre_SStructPMatrix *pA,
      hypre_SStructPVector *px,
      HYPRE_Complex         beta,
      hypre_SStructPVector *py
      )
{
   void *pmatvec_data;

   hypre_SStructPMatvecCreate(&pmatvec_data);
   hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   hypre_SStructPMatvecTCompute(pmatvec_data, alpha, pA, px, beta, py);
   hypre_SStructPMatvecDestroy(pmatvec_data);

   return hypre_error_flag;
}

/*==========================================================================
 * MatvecT routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    nparts;
   void **pmatvec_data;
} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecTSetup( 
      void                *matvec_vdata,
      hypre_SStructMatrix *A,
      hypre_SStructVector *x )
{
   hypre_SStructMatvecData  *matvec_data = matvec_vdata;
   HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   HYPRE_Int                 part;

   nparts = hypre_SStructMatrixNParts(A);
   pmatvec_data = hypre_TAlloc(void *, nparts);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatvecCreate(&pmatvec_data[part]);
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      hypre_SStructPMatvecTSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecTCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecTCompute( 
      void                *matvec_vdata,
      HYPRE_Complex        alpha,
      hypre_SStructMatrix *A,
      hypre_SStructVector *x,
      HYPRE_Complex        beta,
      hypre_SStructVector *y )
{
   hypre_SStructMatvecData  *matvec_data  = matvec_vdata;
   HYPRE_Int                 nparts       = (matvec_data -> nparts);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   hypre_SStructPVector     *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx;
   hypre_ParVector          *pary;

   HYPRE_Int                 part;
   HYPRE_Int                 x_object_type= hypre_SStructVectorObjectType(x);
   HYPRE_Int                 A_object_type= hypre_SStructMatrixObjectType(A);

   if (x_object_type != A_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if ( (x_object_type == HYPRE_SSTRUCT) || (x_object_type == HYPRE_STRUCT) )
   {
      /* do S-matrix computations */
      for (part = 0; part < nparts; part++)
      {
         pdata = pmatvec_data[part];
         pA = hypre_SStructMatrixPMatrix(A, part);
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);
         hypre_SStructPMatvecTCompute(pdata, alpha, pA, px, beta, py);
      }

      if ( (x_object_type == HYPRE_SSTRUCT) )
      {

         /* do U-matrix computations */

         /* GEC1002 the data chunk pointed by the local-parvectors 
          *  inside the semistruct vectors x and y is now identical to the
          *  data chunk of the structure vectors x and y. The role of the function
          *  convert is to pass the addresses of the data chunk
          *  to the parx and pary. */  

         hypre_SStructVectorConvert(x, &parx);
         hypre_SStructVectorConvert(y, &pary); 

         hypre_ParCSRMatrixMatvecT(alpha, parcsrA, parx, 1.0, pary); // XXX change from vec to vecT not tested

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

      hypre_ParCSRMatrixMatvecT(alpha, parcsrA, parx, beta, pary); // XXX change from vec to vecT not tested

      hypre_SStructVectorRestore(x, NULL);
      hypre_SStructVectorRestore(y, pary); 

      parx = NULL; 
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecT( 
      HYPRE_Complex        alpha,
      hypre_SStructMatrix *A,
      hypre_SStructVector *x,
      HYPRE_Complex        beta,
      hypre_SStructVector *y )
{
   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecTSetup(matvec_data, A, x);
   hypre_SStructMatvecTCompute(matvec_data, alpha, A, x, beta, y);
   hypre_SStructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

