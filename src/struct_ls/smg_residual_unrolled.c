/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
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
   HYPRE_Int ierr;

   hypre_SMGResidualData  *residual_data = residual_vdata;

   hypre_IndexRef          base_index  = (residual_data -> base_index);
   hypre_IndexRef          base_stride = (residual_data -> base_stride);
   hypre_Index             unit_stride;

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

   hypre_SetIndex(unit_stride, 1, 1, 1);

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
   HYPRE_Int ierr;

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
                         
   double                 *Ap0;
   double                 *xp0;
   double                 *bp;
   double                 *rp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;
   HYPRE_Int               stencil_size;

   HYPRE_Int               compute_i, i, j, si;
   HYPRE_Int               loopi, loopj, loopk;

   double            *Ap1, *Ap2;
   double            *Ap3, *Ap4;
   double            *Ap5, *Ap6;
   double            *Ap7, *Ap8, *Ap9;
   double            *Ap10, *Ap11, *Ap12, *Ap13, *Ap14;
   double            *Ap15, *Ap16, *Ap17, *Ap18;
   double            *Ap19, *Ap20, *Ap21, *Ap22, *Ap23, *Ap24, *Ap25, *Ap26;
   double            *xp1, *xp2;
   double            *xp3, *xp4;
   double            *xp5, *xp6;
   double            *xp7, *xp8, *xp9;
   double            *xp10, *xp11, *xp12, *xp13, *xp14;
   double            *xp15, *xp16, *xp17, *xp18;
   double            *xp19, *xp20, *xp21, *xp22, *xp23, *xp24, *xp25, *xp26;

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
            xp0 = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp0, &comm_handle);
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

            /*--------------------------------------------------------------
             * Switch statement to direct control (based on stencil size) to
             * code to get pointers and offsets fo A and x.
             *--------------------------------------------------------------*/

            switch (stencil_size)
            {
               case 1:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);

               break;

               case 3:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);

               break;

               case 5:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);

               break;

               case 7:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = hypre_StructMatrixBoxData(A, i, 6);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);

               break;

               case 9:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = hypre_StructMatrixBoxData(A, i, 8);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);

               break;

               case 15:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = hypre_StructMatrixBoxData(A, i, 14);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);

               break;

               case 19:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = hypre_StructMatrixBoxData(A, i, 14);
               Ap15 = hypre_StructMatrixBoxData(A, i, 15);
               Ap16 = hypre_StructMatrixBoxData(A, i, 16);
               Ap17 = hypre_StructMatrixBoxData(A, i, 17);
               Ap18 = hypre_StructMatrixBoxData(A, i, 18);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);
               xp15 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[15]);
               xp16 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[16]);
               xp17 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[17]);
               xp18 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[18]);

               break;

               case 27:

               Ap0 = hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = hypre_StructMatrixBoxData(A, i, 14);
               Ap15 = hypre_StructMatrixBoxData(A, i, 15);
               Ap16 = hypre_StructMatrixBoxData(A, i, 16);
               Ap17 = hypre_StructMatrixBoxData(A, i, 17);
               Ap18 = hypre_StructMatrixBoxData(A, i, 18);
               Ap19 = hypre_StructMatrixBoxData(A, i, 19);
               Ap20 = hypre_StructMatrixBoxData(A, i, 20);
               Ap21 = hypre_StructMatrixBoxData(A, i, 21);
               Ap22 = hypre_StructMatrixBoxData(A, i, 22);
               Ap23 = hypre_StructMatrixBoxData(A, i, 23);
               Ap24 = hypre_StructMatrixBoxData(A, i, 24);
               Ap25 = hypre_StructMatrixBoxData(A, i, 25);
               Ap26 = hypre_StructMatrixBoxData(A, i, 26);

               xp0 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);
               xp15 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[15]);
               xp16 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[16]);
               xp17 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[17]);
               xp18 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[18]);
               xp19 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[19]);
               xp20 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[20]);
               xp21 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[21]);
               xp22 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[22]);
               xp23 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[23]);
               xp24 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[24]);
               xp25 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[25]);
               xp26 = hypre_StructVectorBoxData(x, i) +
                  hypre_BoxOffsetDistance(x_data_box, stencil_shape[26]);

               break;

               default:
               ;
            }

            hypre_ForBoxI(j, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, j);

                  start  = hypre_BoxIMin(compute_box);

                  /*------------------------------------------------------
                   * Switch statement to direct control to appropriate
                   * box loop depending on stencil size
                   *------------------------------------------------------*/

                  switch (stencil_size)
                  {

                     case 1:
   
                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {

                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 3:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
 
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 5:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
 
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 7:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {

                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 9:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
   
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi];
   
                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 15:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
   
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);

                     break;

                     case 19:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
   
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi]
                              - Ap15[Ai] * xp15[xi]
                              - Ap16[Ai] * xp16[xi]
                              - Ap17[Ai] * xp17[xi]
                              - Ap18[Ai] * xp18[xi];
   
                        }
                     hypre_BoxLoop3End(Ai, xi, ri);
   
                     break;
   
                     case 27:

                     hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);
                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ri
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, xi, ri)
                        {
   
                           rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi]
                              - Ap15[Ai] * xp15[xi]
                              - Ap16[Ai] * xp16[xi]
                              - Ap17[Ai] * xp17[xi]
                              - Ap18[Ai] * xp18[xi]
                              - Ap19[Ai] * xp19[xi]
                              - Ap20[Ai] * xp20[xi]
                              - Ap21[Ai] * xp21[xi]
                              - Ap22[Ai] * xp22[xi]
                              - Ap23[Ai] * xp23[xi]
                              - Ap24[Ai] * xp24[xi]
                              - Ap25[Ai] * xp25[xi]
                              - Ap26[Ai] * xp26[xi];

                        }
                     hypre_BoxLoop3End(Ai, xi, ri);
   
                     break;

                     default:

                     for (si = 0; si < stencil_size; si++)
                     {
                        Ap0 = hypre_StructMatrixBoxData(A, i, si);
                        xp0 = hypre_StructVectorBoxData(x, i) +
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
                              rp[ri] -= Ap0[Ai] * xp0[xi];
                           }
                        hypre_BoxLoop3End(Ai, xi, ri);
                     }
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
   HYPRE_Int ierr;

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

