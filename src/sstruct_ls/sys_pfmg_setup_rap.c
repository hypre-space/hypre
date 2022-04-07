/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
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

   HYPRE_Int              vi, vj;

   HYPRE_Int              sten_cntr;

   HYPRE_Int              P_stored_as_transpose = 0;

   ndim = hypre_StructStencilNDim(hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = hypre_SStructPMatrixNVars(A);

   vartype = hypre_SStructPGridVarType(coarse_grid, 0);
   cgrid = hypre_SStructPGridVTSGrid(coarse_grid, vartype);

   RAP_stencils = hypre_CTAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);

   RAP_shapes = hypre_CTAlloc(hypre_Index *,  nvars, HYPRE_MEMORY_HOST);
   sstencil_sizes = hypre_CTAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);

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
                                           sstencil_sizes[vj], HYPRE_MEMORY_HOST);
            for (s = 0; s < sstencil_sizes[vj]; s++)
            {
               hypre_CopyIndex(shape[s], RAP_shapes[vj][s]);
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
                                            sten_cntr, RAP_shapes[vj][s], vj);
               sten_cntr++;
            }
            hypre_TFree(RAP_shapes[vj], HYPRE_MEMORY_HOST);
         }
      }
   }

   /* create RAP Pmatrix */
   hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A),
                              coarse_grid, RAP_stencils, &RAP);

   hypre_TFree(RAP_shapes, HYPRE_MEMORY_HOST);
   hypre_TFree(sstencil_sizes, HYPRE_MEMORY_HOST);

   return RAP;
}

/*--------------------------------------------------------------------------
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
   HYPRE_Int               nvars;
   HYPRE_Int               vi, vj;

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
            hypre_SemiBuildRAP(A_s, P_s, R_s, cdir, cindex, cstride,
                               P_stored_as_transpose, Ac_s);
            /* Assemble here? */
            hypre_StructMatrixAssemble(Ac_s);
         }
      }
   }

   return hypre_error_flag;
}

