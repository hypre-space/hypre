/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_SysPFMGCreateInterpOp
 *--------------------------------------------------------------------------*/

hypre_SStructPMatrix *
hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A,
                             hypre_SStructPGrid   *cgrid,
                             int                   cdir  )
{
   hypre_SStructPMatrix  *P;

   hypre_Index           *stencil_shape;
   int                    stencil_size;
                       
   int                    ndim;

   int                    nvars;
   hypre_SStructStencil **P_stencils;

   int                    num_ghost[] = {1, 1, 1, 1, 1, 1};
                        
   int                    i,s;

   int                    ierr;

   /* set up stencil_shape */
   stencil_size = 2;
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;

   /* set up P_stencils */
   ndim = hypre_StructStencilDim(hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = hypre_SStructPMatrixNVars(A);
   P_stencils = hypre_CTAlloc(hypre_SStructStencil *, nvars);
   for (s = 0; s < nvars; s++)
   {
      HYPRE_SStructStencilCreate(ndim, stencil_size, &P_stencils[s]);
      for (i = 0; i < stencil_size; i++)
      {
         HYPRE_SStructStencilSetEntry(P_stencils[s], i,
                                      stencil_shape[i], s);
      }
   }

   /* create interpolation matrix */
   ierr = hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), cgrid,
                                    P_stencils, &P);

   hypre_TFree(stencil_shape);

 
   return P;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetupInterpOp
 *--------------------------------------------------------------------------*/

int
hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *A,
                            int                   cdir,
                            hypre_Index           findex,
                            hypre_Index           stride,
                            hypre_SStructPMatrix *P      )
{
   int                    nvars;
   hypre_StructMatrix    *A_s;
   hypre_StructMatrix    *P_s;
   int                    vi;

   int                    ierr;

   nvars = hypre_SStructPMatrixNVars(A);

   for (vi = 0; vi < nvars; vi++)
   {
      A_s = hypre_SStructPMatrixSMatrix(A, vi, vi);
      P_s = hypre_SStructPMatrixSMatrix(P, vi, vi);
      ierr = hypre_PFMGSetupInterpOp(A_s, cdir, findex, stride, P_s);
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

