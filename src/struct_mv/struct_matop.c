/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"

#define UNROLL_MAXDEPTH 9

/*--------------------------------------------------------------------------
 * hypre_StructMatrixComputeRowSum
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixComputeRowSum( hypre_StructMatrix  *A,
                                 HYPRE_Int            type,
                                 hypre_StructVector  *rowsum )
{
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_StructGrid     *grid    = hypre_StructVectorGrid(A);
   hypre_BoxArray       *boxes   = hypre_StructGridBoxes(grid);
   HYPRE_Int             stencil_size = hypre_StructStencilSize(stencil);

   hypre_Box            *box;
   hypre_Box            *rdbox;
   hypre_Box            *Adbox;
   hypre_Index           loop_size;
   HYPRE_Int             k, i, si;
   HYPRE_Int             depth, cdepth, vdepth;
   HYPRE_Int             csi[UNROLL_MAXDEPTH], vsi[UNROLL_MAXDEPTH];

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      hypre_BoxGetSize(box, loop_size);

      Adbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      rdbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(rowsum), i);

      /* unroll up to depth UNROLL_MAXDEPTH */
      for (si = 0; si < stencil_size; si += UNROLL_MAXDEPTH)
      {
         depth = hypre_min(UNROLL_MAXDEPTH, (stencil_size - si));

         cdepth = vdepth = 0;
         for (k = 0; k < depth; k++)
         {
            if (hypre_StructMatrixConstEntry(A, si +k))
            {
               csi[cdepth++] = si + k;
            }
            else
            {
               vsi[vdepth++] = si + k;
            }
         }

         /* Operate on constant coefficients */
         hypre_StructMatrixComputeRowSum_core_CC(A, rowsum, i, cdepth, csi,
                                                 box, Adbox, rdbox, type);

         /* Operate on variable coefficients */
         hypre_StructMatrixComputeRowSum_core_VC(A, rowsum, i, vdepth, vsi,
                                                 box, Adbox, rdbox, type);
      } /* loop on stencil entries */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixComputeRowSum_core_CC
 *
 * Core function for computing rowsum for constant coeficients in A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixComputeRowSum_core_CC(hypre_StructMatrix  *A,
                                        hypre_StructVector  *rowsum,
                                        HYPRE_Int            box_id,
                                        HYPRE_Int            nentries,
                                        HYPRE_Int           *entries,
                                        hypre_Box           *box,
                                        hypre_Box           *Adbox,
                                        hypre_Box           *rdbox,
                                        HYPRE_Int            type)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *rp;

   start = hypre_BoxIMin(box);
   hypre_BoxGetSize(box, loop_size);
   hypre_SetIndex(ustride, 1);
   rp = hypre_StructVectorBoxData(rowsum, box_id);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, box_id, entries[8]);

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, box_id, entries[7]);

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, box_id, entries[6]);

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, box_id, entries[5]);

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, box_id, entries[4]);

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, box_id, entries[3]);

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, box_id, entries[2]);

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, box_id, entries[1]);

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, box_id, entries[0]);

      case 0:
         break;
   }

   if (type == 0)
   {
      /* Compute row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0] + Ap4[0] + Ap5[0] +
                         Ap6[0] + Ap7[0] + Ap8[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 8:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0] + Ap4[0] + Ap5[0] +
                         Ap6[0] + Ap7[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 7:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0] + Ap4[0] + Ap5[0] +
                         Ap6[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 6:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0] + Ap4[0] + Ap5[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 5:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0] + Ap4[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 4:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0] +
                         Ap3[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 3:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0] + Ap2[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 2:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] + Ap1[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 1:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }
   else if (type == 1)
   {
      /* Compute absolute row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]) + hypre_cabs(Ap4[0]) + hypre_cabs(Ap5[0]) +
                         hypre_cabs(Ap6[0]) + hypre_cabs(Ap7[0]) + hypre_cabs(Ap8[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 8:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]) + hypre_cabs(Ap4[0]) + hypre_cabs(Ap5[0]) +
                         hypre_cabs(Ap6[0]) + hypre_cabs(Ap7[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 7:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]) + hypre_cabs(Ap4[0]) + hypre_cabs(Ap5[0]) +
                         hypre_cabs(Ap6[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 6:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]) + hypre_cabs(Ap4[0]) + hypre_cabs(Ap5[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 5:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]) + hypre_cabs(Ap4[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 4:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]) +
                         hypre_cabs(Ap3[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 3:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]) + hypre_cabs(Ap2[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 2:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]) + hypre_cabs(Ap1[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 1:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[0]);
            }
            hypre_BoxLoop1End(ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }
   else if (type == 2)
   {
      /* Compute squared row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0] + Ap4[0]*Ap4[0] + Ap5[0]*Ap5[0] +
                         Ap6[0]*Ap6[0] + Ap7[0]*Ap7[0] + Ap8[0]*Ap8[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 8:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0] + Ap4[0]*Ap4[0] + Ap5[0]*Ap5[0] +
                         Ap6[0]*Ap6[0] + Ap7[0]*Ap7[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 7:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0] + Ap4[0]*Ap4[0] + Ap5[0]*Ap5[0] +
                         Ap6[0]*Ap6[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 6:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0] + Ap4[0]*Ap4[0] + Ap5[0]*Ap5[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 5:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0] + Ap4[0]*Ap4[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 4:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0] +
                         Ap3[0]*Ap3[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 3:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0] + Ap2[0]*Ap2[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 2:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0] + Ap1[0]*Ap1[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 1:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0]*Ap0[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixComputeRowSum_core_VC
 *
 * Core function for computing rowsum for variable coeficients in A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixComputeRowSum_core_VC(hypre_StructMatrix  *A,
                                        hypre_StructVector  *rowsum,
                                        HYPRE_Int            box_id,
                                        HYPRE_Int            nentries,
                                        HYPRE_Int           *entries,
                                        hypre_Box           *box,
                                        hypre_Box           *Adbox,
                                        hypre_Box           *rdbox,
                                        HYPRE_Int            type)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *rp;

   start = hypre_BoxIMin(box);
   hypre_BoxGetSize(box, loop_size);
   hypre_SetIndex(ustride, 1);
   rp = hypre_StructVectorBoxData(rowsum, box_id);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, box_id, entries[8]);

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, box_id, entries[7]);

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, box_id, entries[6]);

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, box_id, entries[5]);

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, box_id, entries[4]);

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, box_id, entries[3]);

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, box_id, entries[2]);

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, box_id, entries[1]);

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, box_id, entries[0]);

      case 0:
         break;
   }

   if (type == 0)
   {
      /* Compute row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                         Ap6[Ai] + Ap7[Ai] + Ap8[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                         Ap6[Ai] + Ap7[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                         Ap6[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai] + Ap4[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                         Ap3[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai] + Ap2[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] + Ap1[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }
   else if (type == 1)
   {
      /* Compute absolute row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]) + hypre_cabs(Ap4[Ai]) + hypre_cabs(Ap5[Ai]) +
                         hypre_cabs(Ap6[Ai]) + hypre_cabs(Ap7[Ai]) + hypre_cabs(Ap8[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]) + hypre_cabs(Ap4[Ai]) + hypre_cabs(Ap5[Ai]) +
                         hypre_cabs(Ap6[Ai]) + hypre_cabs(Ap7[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]) + hypre_cabs(Ap4[Ai]) + hypre_cabs(Ap5[Ai]) +
                         hypre_cabs(Ap6[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]) + hypre_cabs(Ap4[Ai]) + hypre_cabs(Ap5[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]) + hypre_cabs(Ap4[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]) +
                         hypre_cabs(Ap3[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]) + hypre_cabs(Ap2[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]) + hypre_cabs(Ap1[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += hypre_cabs(Ap0[Ai]);
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }
   else if (type == 2)
   {
      /* Compute squared row sums */
      switch (nentries)
      {
         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri)
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai] + Ap4[Ai]*Ap4[Ai] + Ap5[Ai]*Ap5[Ai] +
                         Ap6[Ai]*Ap6[Ai] + Ap7[Ai]*Ap7[Ai] + Ap8[Ai]*Ap8[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai] + Ap4[Ai]*Ap4[Ai] + Ap5[Ai]*Ap5[Ai] +
                         Ap6[Ai]*Ap6[Ai] + Ap7[Ai]*Ap7[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai] + Ap4[Ai]*Ap4[Ai] + Ap5[Ai]*Ap5[Ai] +
                         Ap6[Ai]*Ap6[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai] + Ap4[Ai]*Ap4[Ai] + Ap5[Ai]*Ap5[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai] + Ap4[Ai]*Ap4[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai] +
                         Ap3[Ai]*Ap3[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai] + Ap2[Ai]*Ap2[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai] + Ap1[Ai]*Ap1[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai]*Ap0[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }

   return hypre_error_flag;
}
