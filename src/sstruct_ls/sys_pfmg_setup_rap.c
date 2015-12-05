/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SysPFMGCreateRAPOp
 *--------------------------------------------------------------------------*/
 
hypre_SStructPMatrix *
hypre_SysPFMGCreateRAPOp( hypre_SStructPMatrix *R,
                          hypre_SStructPMatrix *A,
                          hypre_SStructPMatrix *P,
                          hypre_SStructPGrid   *coarse_grid,
                          HYPRE_Int             cdir        )
{
   hypre_SStructPMatrix    *RAP;
   HYPRE_Int                ndim;
   HYPRE_Int                nvars;
   hypre_SStructVariable    vartype;

   hypre_SStructStencil **RAP_stencils;

   hypre_StructMatrix    *RAP_s;
   hypre_StructMatrix    *R_s;
   hypre_StructMatrix    *A_s;
   hypre_StructMatrix    *P_s;

   hypre_Index          **RAP_shapes;

   hypre_StructStencil   *sstencil;
   hypre_Index           *shape;
   HYPRE_Int              s;
   HYPRE_Int             *sstencil_sizes;

   HYPRE_Int              stencil_size;

   hypre_StructGrid      *cgrid;

   HYPRE_Int              vi,vj;

   HYPRE_Int              sten_cntr;

   HYPRE_Int              P_stored_as_transpose = 0;

   ndim = hypre_StructStencilDim(hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = hypre_SStructPMatrixNVars(A);

   vartype = hypre_SStructPGridVarType(coarse_grid, 0);
   cgrid = hypre_SStructPGridVTSGrid(coarse_grid, vartype);

   RAP_stencils = hypre_CTAlloc(hypre_SStructStencil *, nvars);

   RAP_shapes = hypre_CTAlloc(hypre_Index *, nvars);
   sstencil_sizes = hypre_CTAlloc(HYPRE_Int, nvars);

/*--------------------------------------------------------------------------
 * Symmetry within a block is exploited, but not symmetry of the form
 * A_{vi,vj} = A_{vj,vi}^T.
 *--------------------------------------------------------------------------*/

   for (vi = 0; vi < nvars; vi++)
   {
      R_s = hypre_SStructPMatrixSMatrix(R, vi, vi);
      stencil_size = 0;
      for (vj = 0; vj < nvars; vj++)
      {
         A_s = hypre_SStructPMatrixSMatrix(A, vi, vj);
         P_s = hypre_SStructPMatrixSMatrix(P, vj, vj);
         sstencil_sizes[vj] = 0;
         if (A_s != NULL)
         {         
            RAP_s = hypre_SemiCreateRAPOp(R_s, A_s, P_s,
                                          cgrid, cdir,
                                          P_stored_as_transpose);
            /* Just want stencil for RAP */
            hypre_StructMatrixInitializeShell(RAP_s);
            sstencil = hypre_StructMatrixStencil(RAP_s);
            shape = hypre_StructStencilShape(sstencil);
            sstencil_sizes[vj] = hypre_StructStencilSize(sstencil);
            stencil_size += sstencil_sizes[vj];
            RAP_shapes[vj] = hypre_CTAlloc(hypre_Index,
                                          sstencil_sizes[vj]);
            for (s = 0; s < sstencil_sizes[vj]; s++)
            {
               hypre_CopyIndex(shape[s],RAP_shapes[vj][s]);
            }
            hypre_StructMatrixDestroy(RAP_s);
         }
      }

      HYPRE_SStructStencilCreate(ndim, stencil_size, &RAP_stencils[vi]);
      sten_cntr = 0;
      for (vj = 0; vj < nvars; vj++)
      {
         if (sstencil_sizes[vj] > 0)
         {
            for (s = 0; s < sstencil_sizes[vj]; s++)
            {
               HYPRE_SStructStencilSetEntry(RAP_stencils[vi],
                                            sten_cntr, RAP_shapes[vj][s],
                                            vj);
               sten_cntr++;
            }
            hypre_TFree(RAP_shapes[vj]);
         }
      }
   }

   /* create RAP Pmatrix */
   hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), 
                                     coarse_grid, RAP_stencils, &RAP);

   hypre_TFree(RAP_shapes);
   hypre_TFree(sstencil_sizes);
   return RAP;
}

/*--------------------------------------------------------------------------
 * hypre_SysPFMGSetupRAPOp
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_SysPFMGSetupRAPOp( hypre_SStructPMatrix *R,
                         hypre_SStructPMatrix *A,
                         hypre_SStructPMatrix *P,
                         HYPRE_Int             cdir,
                         hypre_Index           cindex,
                         hypre_Index           cstride,
                         hypre_SStructPMatrix *Ac      )
{
   HYPRE_Int ierr = 0;
 
   HYPRE_Int               nvars;
   HYPRE_Int               vi,vj;

   hypre_StructMatrix    *R_s;
   hypre_StructMatrix    *A_s;
   hypre_StructMatrix    *P_s;

   hypre_StructMatrix    *Ac_s;

   HYPRE_Int              P_stored_as_transpose = 0;

   nvars = hypre_SStructPMatrixNVars(A);

/*--------------------------------------------------------------------------
 * Symmetry within a block is exploited, but not symmetry of the form
 * A_{vi,vj} = A_{vj,vi}^T.
 *--------------------------------------------------------------------------*/
   for (vi = 0; vi < nvars; vi++)
   {
      R_s = hypre_SStructPMatrixSMatrix(R, vi, vi);
      for (vj = 0; vj < nvars; vj++)
      {
         A_s  = hypre_SStructPMatrixSMatrix(A, vi, vj);
         Ac_s = hypre_SStructPMatrixSMatrix(Ac, vi, vj);
         P_s  = hypre_SStructPMatrixSMatrix(P, vj, vj);
         if (A_s != NULL)
         {
            ierr = hypre_SemiBuildRAP(A_s, P_s, R_s,
                              cdir, cindex, cstride,
                              P_stored_as_transpose,
                              Ac_s);
            /* Assemble here? */
            hypre_StructMatrixAssemble(Ac_s);
         }
      }
   }

   return ierr;
}

