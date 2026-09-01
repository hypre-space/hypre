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
      A_dbox = hypre_StructMatrixBoxDataBox(A, i);
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
                                        HYPRE_Int            boxnum,
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
   rp = hypre_StructVectorBoxData(rowsum, boxnum);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, boxnum, entries[8]);
         HYPRE_FALLTHROUGH;

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, boxnum, entries[7]);
         HYPRE_FALLTHROUGH;

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, boxnum, entries[6]);
         HYPRE_FALLTHROUGH;

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, boxnum, entries[5]);
         HYPRE_FALLTHROUGH;

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, boxnum, entries[4]);
         HYPRE_FALLTHROUGH;

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, boxnum, entries[3]);
         HYPRE_FALLTHROUGH;

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, boxnum, entries[2]);
         HYPRE_FALLTHROUGH;

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, boxnum, entries[1]);
         HYPRE_FALLTHROUGH;

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, boxnum, entries[0]);
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
                                        HYPRE_Int            boxnum,
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
   rp = hypre_StructVectorBoxData(rowsum, boxnum);

   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, boxnum, entries[8]);
         HYPRE_FALLTHROUGH;

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, boxnum, entries[7]);
         HYPRE_FALLTHROUGH;

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, boxnum, entries[6]);
         HYPRE_FALLTHROUGH;

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, boxnum, entries[5]);
         HYPRE_FALLTHROUGH;

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, boxnum, entries[4]);
         HYPRE_FALLTHROUGH;

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, boxnum, entries[3]);
         HYPRE_FALLTHROUGH;

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, boxnum, entries[2]);
         HYPRE_FALLTHROUGH;

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, boxnum, entries[1]);
         HYPRE_FALLTHROUGH;

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, boxnum, entries[0]);
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

      Adbox = hypre_StructMatrixBoxDataBox(A, i);
      rdbox = hypre_StructVectorBoxDataBox(rowsum, i);

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

/*--------------------------------------------------------------------------
 * hypre_StructMatrixScale
 *
 * Scales Struct matrix: A = scalar * A.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixScale( hypre_StructMatrix *A,
                         HYPRE_Complex       scalar)
{
   HYPRE_Complex *data = hypre_StructMatrixData(A);
   HYPRE_Int      i;
   HYPRE_Int      k = hypre_StructMatrixDataSize(A);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_StructMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ComplexScalenDevice(data, k, data, scalar);
   }
   else
#endif
   {
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < k; i++)
      {
         data[i] *= scalar;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Assumptions:
 * - The number of matrices to add is greater than zero, i.e. nmatrices > 0
 * - The matrices have the same stencil grid, range grid, and domain grid
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixAddInit( HYPRE_Int            nmatrices,
                           hypre_StructMatrix **matrices,
                           hypre_StructMatrix **A_ptr )
{
   hypre_StructMatrix  *A, *mat0;
   hypre_StructStencil *stencil;
   hypre_Index         *offsets;
   hypre_IndexRef       offset;
   HYPRE_Int           *isvar;
   HYPRE_Int            ndim, nconst, size, entry, m, i;

   if ( !(nmatrices > 0) )
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Require at least one matrix to add!");
      return hypre_error_flag;
   }

   mat0 = matrices[0];
   ndim = hypre_StructMatrixNDim(mat0);

   /* Compute an upper bound for the stencil size */
   size = 0;
   for (m = 0; m < nmatrices; m++)
   {
      stencil = hypre_StructMatrixStencil(matrices[m]);
      size += hypre_StructStencilSize(stencil);
   }
   offsets = hypre_CTAlloc(hypre_Index,  size, HYPRE_MEMORY_HOST);
   isvar   = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);

   /* Find the set of unique offsets in matrices - use them to define the stencil for A */
   size = 0;
   for (m = 0; m < nmatrices; m++)
   {
      stencil = hypre_StructMatrixStencil(matrices[m]);
      for (entry = 0; entry < hypre_StructStencilSize(stencil); entry++)
      {
         offset = hypre_StructStencilOffset(stencil, entry);
         for (i = 0; i < size; i++)
         {
            if (hypre_IndexesEqual(offset, offsets[i], ndim))
            {
               break;
            }
         }
         if ( !hypre_StructMatrixConstEntry(matrices[m], entry) )
         {
            /* This stencil entry of A must be variable (not constant) */
            isvar[i] = 1;
         }
         if (i == size)
         {
            /* This is a new offset */
            hypre_CopyIndex(offset, offsets[i]);
            size ++;
         }
      }
   }

   /* Create the stencil for A */
   HYPRE_StructStencilCreate(ndim, size, &stencil);
   nconst = 0;
   for (i = 0; i < size; i++)
   {
      HYPRE_StructStencilSetEntry(stencil, i, offsets[i]);
      if ( !isvar[i] )
      {
         isvar[nconst] = i;  /* Now use isvar to hold the constant entries */
         nconst++;
      }
   }

   /* Create A */
   HYPRE_StructMatrixCreate(hypre_StructMatrixComm(mat0), hypre_StructMatrixGrid(mat0), stencil, &A);
   HYPRE_StructMatrixSetRangeStride(A, hypre_StructMatrixRanStride(mat0));
   HYPRE_StructMatrixSetDomainStride(A, hypre_StructMatrixDomStride(mat0));
   HYPRE_StructMatrixSetSymmetric(A, hypre_StructMatrixSymmetric(mat0));
   HYPRE_StructMatrixSetConstantEntries(A, nconst, isvar);
   HYPRE_StructMatrixInitialize(A);

   hypre_TFree(offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(isvar, HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute A += beta * B
 *
 * Assumptions:
 * - A and B have the same stencil grid, range grid, and domain grid
 * - Every stencil entry of B is also present in A
 * - Constant entries in A map to constant entries in B (variable entries in A
 *   can map to either constant or variable entries in B)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixAddMat( hypre_StructMatrix *A,
                          HYPRE_Complex       beta,
                          hypre_StructMatrix *B )
{
   HYPRE_Int             ndim     = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *Astencil = hypre_StructMatrixStencil(A);
   hypre_StructStencil  *Bstencil = hypre_StructMatrixStencil(B);
   hypre_Box            *Adbox, *Bdbox;
   HYPRE_Complex        *Adata, *Bdata;
   HYPRE_Int             Aentry, Bentry;

   hypre_Box            *loop_box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           ustride;
   HYPRE_Int             i;

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   // RDF TODO: Optimize by fusing loops and separating the adds into groups:
   // type CC (constant A - constant B), VC, and VV

   for (Bentry = 0; Bentry < hypre_StructStencilSize(Bstencil); Bentry++)
   {
      hypre_IndexRef  Boffset = hypre_StructStencilOffset(Bstencil, Bentry);

      /* Find the entry in A that correspond to Bentry */
      Aentry = hypre_StructStencilOffsetEntry(Astencil, Boffset);
      if (Aentry < 0)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Stencil offset for B not present in A!");
         return hypre_error_flag;
      }

      if (hypre_StructMatrixConstEntry(A, Aentry))
      {
         Adata = hypre_StructMatrixConstData(A, Aentry);
         Bdata = hypre_StructMatrixConstData(B, Bentry);

         Adata[0] += beta * Bdata[0];
      }
      else
      {
         for (i = 0; i < hypre_StructMatrixRanNBoxes(A); i++)
         {
            Adbox = hypre_StructMatrixRanDataBox(A, i);
            Adata = hypre_StructMatrixRanData(A, i, Aentry);

            hypre_CopyBox(hypre_StructMatrixRanBox(A, i), loop_box);
            hypre_StructMatrixMapDataBox(A, loop_box);
            start = hypre_BoxIMin(loop_box);
            hypre_BoxGetSize(loop_box, loop_size);

            if (hypre_StructMatrixConstEntry(B, Bentry))
            {
               Bdata = hypre_StructMatrixConstData(B, Bentry);
               hypre_BoxLoop1Begin(ndim, loop_size,
                                   Adbox, start, ustride, Ai)
               {
                  Adata[Ai] += beta * Bdata[0];
               }
               hypre_BoxLoop1End(Ai);
            }
            else
            {
               Bdbox = hypre_StructMatrixRanDataBox(B, i);
               Bdata = hypre_StructMatrixRanData(B, i, Bentry);
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Adbox, start, ustride, Ai,
                                   Bdbox, start, ustride, Bi)
               {
                  Adata[Ai] += beta * Bdata[Bi];
               }
               hypre_BoxLoop2End(Ai, Bi);
            }
         }
      }
   }

   hypre_BoxDestroy(loop_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute C = alpha * A + beta * B
 * TODO
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixAdd( HYPRE_Complex        alpha,
                       hypre_StructMatrix  *A,
                       HYPRE_Complex        beta,
                       hypre_StructMatrix  *B,
                       hypre_StructMatrix **C_ptr )
{
   HYPRE_UNUSED_VAR(alpha);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(beta);
   HYPRE_UNUSED_VAR(B);
   HYPRE_UNUSED_VAR(C_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute the matrix polynomial: polyA = c0 I + c1 A + ... + cm A^m
 * Here, 'coeffs[i]' = ci and 'order' = m.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPoly( hypre_StructMatrix       *A,
                        HYPRE_Int                 order,
                        HYPRE_Complex            *coeffs,
                        hypre_StructMatrix      **polyA_ptr )
{
   HYPRE_Int            ndim    = hypre_StructMatrixNDim(A);
   hypre_StructGrid    *grid    = hypre_StructMatrixGrid(A);
   hypre_StructMatrix  *polyA   = NULL;
   hypre_StructMatrix  *T, *TA;
   hypre_StructStencil *stencil;
   HYPRE_Int            i;

   if (order == 0)
   {
      /* Treat (order = 0) as a special case: polyA = c0 I */
      polyA = hypre_StructMatrixDiagonal(grid, coeffs[0]);
   }
   else if (order > 0)
   {
      HYPRE_Int  nconst = 0, *const_entries = NULL;

      /* Compute the stencil for A^order */
      if (order == 1)
      {
         /* For (order = 1): use stencil for A, match the constant entry structure */
         HYPRE_Int  e, size;

         stencil = hypre_StructStencilRef( hypre_StructMatrixStencil(A) );
         size    = hypre_StructStencilSize(stencil);
         const_entries = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
         for (e = 0; e < size; e++)
         {
            if (hypre_StructMatrixConstEntry(A, e))
            {
               const_entries[nconst] = e;
               nconst++;
            }
         }
      }
      else
      {
         /* For (order > 1): compute stencil with StMatrixMatmult, use fully variable entries */
         hypre_StMatrix **st_matrices, *st_Aorder;
         HYPRE_Int       *transposes;

         st_matrices = hypre_CTAlloc(hypre_StMatrix *, order, HYPRE_MEMORY_HOST);
         transposes  = hypre_CTAlloc(HYPRE_Int,        order, HYPRE_MEMORY_HOST);
         hypre_StMatrixCreateFromStencil(hypre_StructMatrixStencil(A),
                                         hypre_StructMatrixRanStride(A),
                                         hypre_StructMatrixDomStride(A),
                                         0, &st_matrices[0]);
         for (i = 1; i < order; i++)
         {
            st_matrices[i] = st_matrices[0];
         }
         hypre_StMatrixMatmult(order, st_matrices, transposes, order, ndim, &st_Aorder);
         hypre_StMatrixDestroy(st_matrices[0]);
         hypre_TFree(st_matrices, HYPRE_MEMORY_HOST);
         hypre_StMatrixGetStencil(st_Aorder, ndim, &stencil);
         hypre_StMatrixDestroy(st_Aorder);
      }

      /* Initialize polyA = 0 */
      HYPRE_StructMatrixCreate(hypre_StructGridComm(grid), grid, stencil, &polyA);
      HYPRE_StructMatrixSetRangeStride(polyA, hypre_StructMatrixRanStride(A));
      HYPRE_StructMatrixSetDomainStride(polyA, hypre_StructMatrixDomStride(A));
      HYPRE_StructMatrixSetSymmetric(polyA, hypre_StructMatrixSymmetric(A));
      HYPRE_StructMatrixSetConstantEntries(polyA, nconst, const_entries);
      HYPRE_StructMatrixInitialize(polyA);

      /* Compute (order = 0) and (order = 1) components: polyA = c0 I + c1 A */
      T = hypre_StructMatrixDiagonal(grid, 1.0);
      hypre_StructMatrixAddMat(polyA, coeffs[0], T);
      hypre_StructMatrixAddMat(polyA, coeffs[1], A);
      hypre_StructMatrixDestroy(T);
      T = hypre_StructMatrixRef(A);

      /* Compute (order > 1) components: polyA += ci A^i */
      for (i = 2; i <= order; i++)
      {
         hypre_StructMatmat(T, A, &TA);
         hypre_StructMatrixAddMat(polyA, coeffs[i], TA);  // RDF write this
         hypre_StructMatrixDestroy(T);
         T = TA;
      }

      HYPRE_StructMatrixAssemble(polyA);

      /* Clean up */
      hypre_StructMatrixDestroy(T);
      hypre_StructStencilDestroy(stencil);
      hypre_TFree(const_entries, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "Polynomial order is negative");
   }

   *polyA_ptr = polyA;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return the diagonal matrix D defined as follows.
 *
 *   type == 0:  D = weight*diag(A)
 *   type == 1:  D = weight*diag(A)^-1
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetDiagMat( hypre_StructMatrix  *A,
                              HYPRE_Real           weight,
                              HYPRE_Int            type,
                              hypre_StructMatrix **D_ptr )
{
   HYPRE_Int             ndim   = hypre_StructMatrixNDim(A);
   hypre_StructGrid     *grid   = hypre_StructMatrixGrid(A);
   HYPRE_Int             Adiag  = hypre_StructStencilDiagEntry(hypre_StructMatrixStencil(A));
   HYPRE_Complex        *Adata;
   hypre_StructMatrix   *D;
   HYPRE_Complex        *Ddata;
   hypre_StructStencil  *stencil;
   hypre_Index           offset;

   hypre_SetIndex(offset, 0);
   HYPRE_StructStencilCreate(hypre_StructGridNDim(grid), 1, &stencil);
   HYPRE_StructStencilSetEntry(stencil, 0, offset);
   HYPRE_StructMatrixCreate(hypre_StructGridComm(grid), grid, stencil, &D);
   HYPRE_StructMatrixInitialize(D);

   if (hypre_StructMatrixConstEntry(A, Adiag))
   {
      Adata = hypre_StructMatrixConstData(A, Adiag);
      Ddata = hypre_StructMatrixConstData(D, 0);

      if (type == 0)
      {
         Ddata[0] = weight * Adata[0];
      }
      else if (type == 1)
      {
         Ddata[0] = weight / Adata[0];
      }
   }
   else
   {
      hypre_Box            *Adbox, *Ddbox;
      HYPRE_Complex        *Adata, *Ddata;
      hypre_Box            *loop_box;
      hypre_Index           loop_size;
      hypre_IndexRef        start;
      hypre_Index           ustride;
      HYPRE_Int             i;

      loop_box = hypre_BoxCreate(ndim);
      hypre_SetIndex(ustride, 1);

      for (i = 0; i < hypre_StructMatrixRanNBoxes(A); i++)
      {
         Adbox = hypre_StructMatrixRanDataBox(A, i);
         Ddbox = hypre_StructMatrixRanDataBox(D, i);
         Adata = hypre_StructMatrixRanData(A, i, Adiag);
         Ddata = hypre_StructMatrixRanData(D, i, 0);

         hypre_CopyBox(hypre_StructMatrixRanBox(A, i), loop_box);
         hypre_StructMatrixMapDataBox(A, loop_box);
         start = hypre_BoxIMin(loop_box);
         hypre_BoxGetSize(loop_box, loop_size);

         if (type == 0)
         {
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                Ddbox, start, ustride, Di);
            {
               Ddata[Di] = weight * Adata[Ai];
            }
            hypre_BoxLoop2End(Ai, Di);
         }
         else if (type == 1)
         {
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Adbox, start, ustride, Ai,
                                Ddbox, start, ustride, Di);
            {
               Ddata[Di] = weight / Adata[Ai];
            }
            hypre_BoxLoop2End(Ai, Di);
         }
      }

      hypre_BoxDestroy(loop_box);
   }

   HYPRE_StructMatrixAssemble(D);
   *D_ptr = D;

   return hypre_error_flag;
}
