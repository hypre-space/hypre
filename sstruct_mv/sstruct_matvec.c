/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
   int     nvars;
   void ***smatvec_data;

} hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCreate
 *--------------------------------------------------------------------------*/


hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   int ierr = 0;
   hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = hypre_CTAlloc(hypre_SStructPMatvecData, 1);
   *pmatvec_vdata_ptr = (void *) pmatvec_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecSetup
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatvecSetup( void                 *pmatvec_vdata,
                           hypre_SStructPMatrix *pA,
                           hypre_SStructPVector *px )
{
   int ierr = 0;
   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
   int                         nvars;
   void                     ***smatvec_data;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   int                         vi, vj;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecCompute
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatvecCompute( void                 *pmatvec_vdata,
                             double                alpha,
                             hypre_SStructPMatrix *pA,
                             hypre_SStructPVector *px,
                             double                beta,
                             hypre_SStructPVector *py )
{
   int ierr = 0;

   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
   int                         nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   void                       *sdata;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx;
   hypre_StructVector         *sy;

   int                        vi, vj;

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
         hypre_StructAxpy(beta, sy, sy);
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   int ierr = 0;
   hypre_SStructPMatvecData   *pmatvec_data = pmatvec_vdata;
   int                         nvars;
   void                     ***smatvec_data;
   int                         vi, vj;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvec
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatvec( double                alpha,
                      hypre_SStructPMatrix *pA,
                      hypre_SStructPVector *px,
                      double                beta,
                      hypre_SStructPVector *py )
{
   int ierr = 0;

   void *pmatvec_data;

   hypre_SStructPMatvecCreate(&pmatvec_data);
   ierr = hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   ierr = hypre_SStructPMatvecCompute(pmatvec_data, alpha, pA, px, beta, py);
   ierr = hypre_SStructPMatvecDestroy(pmatvec_data);

   return ierr;
}

/*==========================================================================
 * Matvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    nparts;
   void **pmatvec_data;

} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCreate
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   int ierr = 0;
   hypre_SStructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_SStructMatvecData, 1);
   *matvec_vdata_ptr = (void *) matvec_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecSetup
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatvecSetup( void                *matvec_vdata,
                          hypre_SStructMatrix *A,
                          hypre_SStructVector *x )
{
   int ierr = 0;
   hypre_SStructMatvecData  *matvec_data = matvec_vdata;
   int                       nparts;
   void                    **pmatvec_data;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   int                       part;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecCompute
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatvecCompute( void                *matvec_vdata,
                            double               alpha,
                            hypre_SStructMatrix *A,
                            hypre_SStructVector *x,
                            double               beta,
                            hypre_SStructVector *y )
{
   int ierr = 0;

   hypre_SStructMatvecData  *matvec_data  = matvec_vdata;
   int                       nparts       = (matvec_data -> nparts);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;
   hypre_SStructPVector     *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx;
   hypre_ParVector          *pary;

   int                       part;

   /* do S-matrix computations */
   for (part = 0; part < nparts; part++)
   {
      pdata = pmatvec_data[part];
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, py);
   }

   /* do U-matrix computations */
   hypre_SStructVectorConvert(x, &parx);
   hypre_SStructVectorConvert(y, &pary);
   hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);
   hypre_SStructVectorRestore(x, NULL);
   hypre_SStructVectorRestore(y, pary);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   int ierr = 0;
   hypre_SStructMatvecData  *matvec_data = matvec_vdata;
   int                       nparts;
   void                    **pmatvec_data;
   int                       part;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatvec
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatvec( double               alpha,
                     hypre_SStructMatrix *A,
                     hypre_SStructVector *x,
                     double               beta,
                     hypre_SStructVector *y )
{
   int ierr = 0;

   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   ierr = hypre_SStructMatvecSetup(matvec_data, A, x);
   ierr = hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   ierr = hypre_SStructMatvecDestroy(matvec_data);

   return ierr;
}
