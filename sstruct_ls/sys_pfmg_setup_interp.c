/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

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

   int                    i,s;

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
   hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), cgrid,
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
      ierr = hypre_PFMGSetupInterpOp(A_s, cdir, findex, stride, P_s, 0);
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

