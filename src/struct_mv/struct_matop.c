/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#define UNROLL_MAXDEPTH 9

/*--------------------------------------------------------------------------
 * Returns 1 if there is a zero on the diagonal, otherwise returns 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixZeroDiagonal( hypre_StructMatrix *A )
{
   HYPRE_Int              ndim       = hypre_StructMatrixNDim(A);
   hypre_StructStencil   *stencil    = hypre_StructMatrixStencil(A);
   HYPRE_Int              diag_entry = hypre_StructStencilDiagEntry(stencil);
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Index            loop_size;
   hypre_IndexRef         start;
   hypre_Index            ustride;

   HYPRE_Complex         *Ap;
   hypre_Box             *A_dbox;
   HYPRE_Int              i;
   HYPRE_Real             diag_product = 0.0;
   HYPRE_Int              zero_diag = 0;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   hypre_SetIndex(ustride, 1);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      start  = hypre_BoxIMin(compute_box);
      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      hypre_BoxGetStrideSize(compute_box, ustride, loop_size);

      Ap = hypre_StructMatrixBoxData(A, i, diag_entry);
      if (hypre_StructMatrixConstEntry(A, diag_entry))
      {
         hypre_TMemcpy(&diag_product, Ap, HYPRE_Complex, 1,
                       HYPRE_MEMORY_HOST, memory_location);
         diag_product = diag_product == 0 ? 1 : 0;
      }
      else
      {
#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
         HYPRE_Real diag_product_local = diag_product;
#elif defined(HYPRE_USING_RAJA)
         ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> diag_product_local(diag_product);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         ReduceSum<HYPRE_Real> diag_product_local(diag_product);
#else
         HYPRE_Real diag_product_local = diag_product;
#endif

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:diag_product_local) reduction(+:diag_product_local)
#else
#define HYPRE_BOX_REDUCTION reduction(+:diag_product_local)
#endif

#define DEVICE_VAR is_device_ptr(Ap)
         hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox, start, ustride,
                                      Ai, diag_product_local);
         {
            HYPRE_Real one  = 1.0;
            HYPRE_Real zero = 0.0;
            if (Ap[Ai] == 0.0)
            {
               diag_product_local += one;
            }
            else
            {
               diag_product_local += zero;
            }
         }
         hypre_BoxLoop1ReductionEnd(Ai, diag_product_local);
#undef DEVICE_VAR
#undef HYPRE_BOX_REDUCTION
#define HYPRE_BOX_REDUCTION

         diag_product += (HYPRE_Real) diag_product_local;
      }
   }

   if (diag_product > 0)
   {
      zero_diag = 1;
   }

   return zero_diag;
}

/*--------------------------------------------------------------------------
 * Core function for computing rowsum for constant coeficients in A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixComputeRowSum_core_CC(hypre_StructMatrix  *A,
                                        hypre_StructVector  *rowsum,
                                        HYPRE_Int            box_id,
                                        HYPRE_Int            nentries,
                                        HYPRE_Int           *entries,
                                        hypre_Box           *box,
                                        hypre_Box           *rdbox,
                                        HYPRE_Int            type)
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL, *Ap7 = NULL, *Ap8 = NULL;
   HYPRE_Complex        *rp;

   start = hypre_BoxIMin(box);
   hypre_BoxGetSize(box, loop_size);
   hypre_SetIndex(ustride, 1);
   rp = hypre_StructVectorBoxData(rowsum, box_id);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, box_id, entries[8]);
         HYPRE_FALLTHROUGH;

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, box_id, entries[7]);
         HYPRE_FALLTHROUGH;

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, box_id, entries[6]);
         HYPRE_FALLTHROUGH;

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, box_id, entries[5]);
         HYPRE_FALLTHROUGH;

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, box_id, entries[4]);
         HYPRE_FALLTHROUGH;

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, box_id, entries[3]);
         HYPRE_FALLTHROUGH;

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, box_id, entries[2]);
         HYPRE_FALLTHROUGH;

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, box_id, entries[1]);
         HYPRE_FALLTHROUGH;

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, box_id, entries[0]);
         HYPRE_FALLTHROUGH;

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
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0] + Ap4[0] * Ap4[0] + Ap5[0] * Ap5[0] +
                         Ap6[0] * Ap6[0] + Ap7[0] * Ap7[0] + Ap8[0] * Ap8[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 8:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0] + Ap4[0] * Ap4[0] + Ap5[0] * Ap5[0] +
                         Ap6[0] * Ap6[0] + Ap7[0] * Ap7[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 7:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0] + Ap4[0] * Ap4[0] + Ap5[0] * Ap5[0] +
                         Ap6[0] * Ap6[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 6:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0] + Ap4[0] * Ap4[0] + Ap5[0] * Ap5[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 5:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0] + Ap4[0] * Ap4[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 4:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0] +
                         Ap3[0] * Ap3[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 3:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0] + Ap2[0] * Ap2[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 2:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0] + Ap1[0] * Ap1[0];
            }
            hypre_BoxLoop1End(ri);
            break;

         case 1:
            hypre_BoxLoop1Begin(ndim, loop_size,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[0] * Ap0[0];
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

   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL, *Ap7 = NULL, *Ap8 = NULL;
   HYPRE_Complex        *rp  = NULL;

   start = hypre_BoxIMin(box);
   hypre_BoxGetSize(box, loop_size);
   hypre_SetIndex(ustride, 1);
   rp = hypre_StructVectorBoxData(rowsum, box_id);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, box_id, entries[8]);
         HYPRE_FALLTHROUGH;

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, box_id, entries[7]);
         HYPRE_FALLTHROUGH;

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, box_id, entries[6]);
         HYPRE_FALLTHROUGH;

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, box_id, entries[5]);
         HYPRE_FALLTHROUGH;

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, box_id, entries[4]);
         HYPRE_FALLTHROUGH;

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, box_id, entries[3]);
         HYPRE_FALLTHROUGH;

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, box_id, entries[2]);
         HYPRE_FALLTHROUGH;

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, box_id, entries[1]);
         HYPRE_FALLTHROUGH;

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, box_id, entries[0]);
         HYPRE_FALLTHROUGH;

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
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai] + Ap4[Ai] * Ap4[Ai] + Ap5[Ai] * Ap5[Ai] +
                         Ap6[Ai] * Ap6[Ai] + Ap7[Ai] * Ap7[Ai] + Ap8[Ai] * Ap8[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);

            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai] + Ap4[Ai] * Ap4[Ai] + Ap5[Ai] * Ap5[Ai] +
                         Ap6[Ai] * Ap6[Ai] + Ap7[Ai] * Ap7[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai] + Ap4[Ai] * Ap4[Ai] + Ap5[Ai] * Ap5[Ai] +
                         Ap6[Ai] * Ap6[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai] + Ap4[Ai] * Ap4[Ai] + Ap5[Ai] * Ap5[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai] + Ap4[Ai] * Ap4[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai] +
                         Ap3[Ai] * Ap3[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai] + Ap2[Ai] * Ap2[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai] + Ap1[Ai] * Ap1[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                rdbox, start, ustride, ri);
            {
               rp[ri] += Ap0[Ai] * Ap0[Ai];
            }
            hypre_BoxLoop2End(Ai, ri);
            break;

         case 0:
            break;
      } /* switch (nentries) */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * RDF TODO: This routine should assume that the base grid for A and rowsum are
 * the same.  It should use the range boxes of A and work for general
 * rectangular matrices.
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

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("StructMatrixComputeRowSum");

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
            if (hypre_StructMatrixConstEntry(A, si + k))
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
                                                 box, rdbox, type);

         /* Operate on variable coefficients */
         hypre_StructMatrixComputeRowSum_core_VC(A, rowsum, i, vdepth, vsi,
                                                 box, Adbox, rdbox, type);
      } /* loop on stencil entries */
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
