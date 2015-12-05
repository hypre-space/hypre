/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * SStruct matrix-vector multiply routine
 *
 *****************************************************************************/

#include "headers.h"

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
 * hypre_SStructPMatvecCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = hypre_CTAlloc(hypre_SStructPMatvecData, 1);
   *pmatvec_vdata_ptr = (void *) pmatvec_data;

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
                             double                alpha,
                             hypre_SStructPMatrix *pA,
                             hypre_SStructPVector *px,
                             double                beta,
                             hypre_SStructPVector *py )
{
   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
   HYPRE_Int                   nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                       *sdata;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   hypre_StructVector         *sy;

   HYPRE_Int                  vi, vj;

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
 * hypre_SStructPMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
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
         hypre_TFree(smatvec_data[vi]);
      }
      hypre_TFree(smatvec_data);
      hypre_TFree(pmatvec_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvec( double                alpha,
                      hypre_SStructPMatrix *pA,
                      hypre_SStructPVector *px,
                      double                beta,
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

typedef struct
{
   HYPRE_Int    nparts;
   void **pmatvec_data;

} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   hypre_SStructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_SStructMatvecData, 1);
   *matvec_vdata_ptr = (void *) matvec_data;

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
      hypre_SStructPMatvecSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCompute( void                *matvec_vdata,
                            double               alpha,
                            hypre_SStructMatrix *A,
                            hypre_SStructVector *x,
                            double               beta,
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
         hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, py);
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

         hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);

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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   hypre_SStructMatvecData  *matvec_data = matvec_vdata;
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
      hypre_TFree(pmatvec_data);
      hypre_TFree(matvec_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvec( double               alpha,
                     hypre_SStructMatrix *A,
                     hypre_SStructVector *x,
                     double               beta,
                     hypre_SStructVector *y )
{
   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecSetup(matvec_data, A, x);
   hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_SStructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
