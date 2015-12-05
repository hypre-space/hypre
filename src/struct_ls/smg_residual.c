/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Routine for computing residuals in the SMG code
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SMGResidualData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index          base_index;
   hypre_Index          base_stride;

   hypre_StructMatrix  *A;
   hypre_StructVector  *x;
   hypre_StructVector  *b;
   hypre_StructVector  *r;
   hypre_BoxArray      *base_points;
   hypre_ComputePkg    *compute_pkg;

   HYPRE_Int            time_index;
   HYPRE_Int            flops;

} hypre_SMGResidualData;

/*--------------------------------------------------------------------------
 * hypre_SMGResidualCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SMGResidualCreate( )
{
   hypre_SMGResidualData *residual_data;

   residual_data = hypre_CTAlloc(hypre_SMGResidualData, 1);

   (residual_data -> time_index)  = hypre_InitializeTiming("SMGResidual");

   /* set defaults */
   hypre_SetIndex((residual_data -> base_index), 0, 0, 0);
   hypre_SetIndex((residual_data -> base_stride), 1, 1, 1);

   return (void *) residual_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGResidualSetup( void               *residual_vdata,
                        hypre_StructMatrix *A,
                        hypre_StructVector *x,
                        hypre_StructVector *b,
                        hypre_StructVector *r              )
{
   HYPRE_Int ierr = 0;

   hypre_SMGResidualData  *residual_data = residual_vdata;

   hypre_IndexRef          base_index  = (residual_data -> base_index);
   hypre_IndexRef          base_stride = (residual_data -> base_stride);

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_BoxArray         *base_points;
   hypre_ComputeInfo      *compute_info;
   hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up base points and the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   base_points = hypre_BoxArrayDuplicate(hypre_StructGridBoxes(grid));
   hypre_ProjectBoxArray(base_points, base_index, base_stride);

   hypre_CreateComputeInfo(grid, stencil, &compute_info);
   hypre_ComputeInfoProjectComp(compute_info, base_index, base_stride);
   hypre_ComputePkgCreate(compute_info, hypre_StructVectorDataSpace(x), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the residual data structure
    *----------------------------------------------------------*/

   (residual_data -> A)           = hypre_StructMatrixRef(A);
   (residual_data -> x)           = hypre_StructVectorRef(x);
   (residual_data -> b)           = hypre_StructVectorRef(b);
   (residual_data -> r)           = hypre_StructVectorRef(r);
   (residual_data -> base_points) = base_points;
   (residual_data -> compute_pkg) = compute_pkg;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   (residual_data -> flops) =
      (hypre_StructMatrixGlobalSize(A) + hypre_StructVectorGlobalSize(x)) /
      (hypre_IndexX(base_stride) *
       hypre_IndexY(base_stride) *
       hypre_IndexZ(base_stride)  );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGResidual( void               *residual_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *x,
                   hypre_StructVector *b,
                   hypre_StructVector *r              )
{
   HYPRE_Int ierr = 0;

   hypre_SMGResidualData  *residual_data = residual_vdata;

   hypre_IndexRef          base_stride = (residual_data -> base_stride);
   hypre_BoxArray         *base_points = (residual_data -> base_points);
   hypre_ComputePkg       *compute_pkg = (residual_data -> compute_pkg);

   hypre_CommHandle       *comm_handle;
                       
   hypre_BoxArrayArray    *compute_box_aa;
   hypre_BoxArray         *compute_box_a;
   hypre_Box              *compute_box;
                       
   hypre_Box              *A_data_box;
   hypre_Box              *x_data_box;
   hypre_Box              *b_data_box;
   hypre_Box              *r_data_box;
                       
   HYPRE_Int               Ai;
   HYPRE_Int               xi;
   HYPRE_Int               bi;
   HYPRE_Int               ri;
                         
   double                 *Ap;
   double                 *xp;
   double                 *bp;
   double                 *rp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;
   HYPRE_Int               stencil_size;

   HYPRE_Int               compute_i, i, j, si;
   HYPRE_Int               loopi, loopj, loopk;

   hypre_BeginTiming(residual_data -> time_index);

   /*-----------------------------------------------------------------------
    * Compute residual r = b - Ax
    *-----------------------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

            /*----------------------------------------
             * Copy b into r
             *----------------------------------------*/

            compute_box_a = base_points;
            hypre_ForBoxI(i, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, i);
                  start = hypre_BoxIMin(compute_box);

                  b_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
                  r_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);

                  bp = hypre_StructVectorBoxData(b, i);
                  rp = hypre_StructVectorBoxData(r, i);

                  hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                  hypre_BoxLoop2Begin(loop_size,
                                      b_data_box, start, base_stride, bi,
                                      r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,bi,ri
#include "hypre_box_smp_forloop.h"
                  hypre_BoxLoop2For(loopi, loopj, loopk, bi, ri)
                     {
                        rp[ri] = bp[bi];
                     }
                  hypre_BoxLoop2End(bi, ri);
               }
         }
         break;

         case 1:
         {
            hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * Compute r -= A*x
       *--------------------------------------------------------------------*/

      hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            r_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);

            rp = hypre_StructVectorBoxData(r, i);

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);

                  for (si = 0; si < stencil_size; si++)
                  {
                     Ap = hypre_StructMatrixBoxData(A, i, si);
                     xp = hypre_StructVectorBoxData(x, i) +
                        hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

                     hypre_BoxGetStrideSize(compute_box, base_stride,
                                            loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
                           rp[ri] -= Ap[Ai] * xp[xi];
                        }
                     hypre_BoxLoop3End(Ai, xi, ri);
                  }
               }
         }
   }
   
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(residual_data -> flops);
   hypre_EndTiming(residual_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualSetBase
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_SMGResidualSetBase( void        *residual_vdata,
                          hypre_Index  base_index,
                          hypre_Index  base_stride )
{
   hypre_SMGResidualData *residual_data = residual_vdata;
   HYPRE_Int              d;
   HYPRE_Int              ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((residual_data -> base_index),  d)
         = hypre_IndexD(base_index,  d);
      hypre_IndexD((residual_data -> base_stride), d)
         = hypre_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGResidualDestroy( void *residual_vdata )
{
   HYPRE_Int ierr = 0;

   hypre_SMGResidualData *residual_data = residual_vdata;

   if (residual_data)
   {
      hypre_StructMatrixDestroy(residual_data -> A);
      hypre_StructVectorDestroy(residual_data -> x);
      hypre_StructVectorDestroy(residual_data -> b);
      hypre_StructVectorDestroy(residual_data -> r);
      hypre_BoxArrayDestroy(residual_data -> base_points);
      hypre_ComputePkgDestroy(residual_data -> compute_pkg );
      hypre_FinalizeTiming(residual_data -> time_index);
      hypre_TFree(residual_data);
   }

   return ierr;
}

