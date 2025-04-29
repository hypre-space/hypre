/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

#ifdef HYPRE_UNROLL_MAXDEPTH
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 9

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz_core_VC
 *
 * Core function for computing stencil collapsing (cxyz and sqcxyz) for the
 * case of variable coefficients (VC)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz_core_VC(hypre_StructMatrix *A,
                              HYPRE_Int           Ab,
                              HYPRE_Int           diag_is_constant,
                              HYPRE_Int           diag_entry,
                              HYPRE_Int          *nentries,
                              HYPRE_Int         **entries,
                              hypre_IndexRef      start,
                              hypre_Index         loop_size,
                              hypre_Box          *A_dbox,
                              HYPRE_Complex      *A_const,
                              HYPRE_Real         *cxyz,
                              HYPRE_Real         *sqcxyz)
{
#define DEVICE_VAR is_device_ptr(A_diag,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8)

   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_Index           ustride;
   HYPRE_Int             d;

   HYPRE_Complex         A_diag_const;
   HYPRE_Complex        *A_diag = NULL;
   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL, *Ap7 = NULL, *Ap8 = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (diag_is_constant)
   {
      A_diag_const = A_const[diag_entry];
   }
   else
   {
      A_diag = hypre_StructMatrixBoxData(A, Ab, diag_entry);
   }

   hypre_SetIndex(ustride, 1);
   for (d = 0; d < ndim; d++)
   {
      switch (nentries[d])
      {
         case 9:
            Ap8 = hypre_StructMatrixBoxData(A, Ab, entries[d][8]);
            HYPRE_FALLTHROUGH;

         case 8:
            Ap7 = hypre_StructMatrixBoxData(A, Ab, entries[d][7]);
            HYPRE_FALLTHROUGH;

         case 7:
            Ap6 = hypre_StructMatrixBoxData(A, Ab, entries[d][6]);
            HYPRE_FALLTHROUGH;

         case 6:
            Ap5 = hypre_StructMatrixBoxData(A, Ab, entries[d][5]);
            HYPRE_FALLTHROUGH;

         case 5:
            Ap4 = hypre_StructMatrixBoxData(A, Ab, entries[d][4]);
            HYPRE_FALLTHROUGH;

         case 4:
            Ap3 = hypre_StructMatrixBoxData(A, Ab, entries[d][3]);
            HYPRE_FALLTHROUGH;

         case 3:
            Ap2 = hypre_StructMatrixBoxData(A, Ab, entries[d][2]);
            HYPRE_FALLTHROUGH;

         case 2:
            Ap1 = hypre_StructMatrixBoxData(A, Ab, entries[d][1]);
            HYPRE_FALLTHROUGH;

         case 1:
            Ap0 = hypre_StructMatrixBoxData(A, Ab, entries[d][0]);
            HYPRE_FALLTHROUGH;

         case 0:
            break;
      }

#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
      HYPRE_Real cdb   = cxyz[d];
      HYPRE_Real sqcdb = sqcxyz[d];

      switch (nentries[d])
      {
         case 9:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai] + Ap8[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai] + Ap8[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 8:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 7:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 6:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 5:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 4:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 3:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 2:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai]);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai]);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 1:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * Ap0[Ai];

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * Ap0[Ai];

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 0:
            break;
      }
#else
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Real2 d2(cxyz[d], sqcxyz[d]);
   ReduceSum<HYPRE_Real2> sum2(d2);
#else
   HYPRE_Real cdb   = cxyz[d];
   HYPRE_Real sqcdb = sqcxyz[d];

#if defined(HYPRE_BOX_REDUCTION)
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:cdb,sqcdb) reduction(+:cdb,sqcdb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cdb,sqcdb)
#endif

#endif
      switch (nentries[d])
      {
         case 9:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai] + Ap8[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 8:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai] + Ap7[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 7:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai] +
                                         Ap6[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 6:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai] + Ap5[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 5:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai] + Ap4[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 4:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai] +
                                         Ap3[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 3:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai] + Ap2[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 2:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * (Ap0[Ai] + Ap1[Ai]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 1:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = diag_is_constant ?
                                 (A_diag_const < 0.0 ? 1.0 : -1.0) :
                                 (A_diag[Ai]   < 0.0 ? 1.0 : -1.0);

               HYPRE_Real temp = sign * Ap0[Ai];

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 0:
            break;
      }
#endif

      if (nentries[d] > 0)
      {
#if !defined(HYPRE_USING_KOKKOS) &&\
  (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
         HYPRE_Real2 temp2 = (HYPRE_Real2) sum2;
         cxyz[d]   = temp2.x;
         sqcxyz[d] = temp2.y;
#else
         cxyz[d]   = (HYPRE_Real) cdb;
         sqcxyz[d] = (HYPRE_Real) sqcdb;
#endif
      }
   }
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz_core_CC
 *
 * Core function for computing stencil collapsing (cxyz and sqcxyz) for the
 * case of constant coefficients (CC)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz_core_CC(hypre_StructMatrix *A,
                              HYPRE_Int           Ab,
                              HYPRE_Int           diag_is_constant,
                              HYPRE_Int           diag_entry,
                              HYPRE_Int          *nentries,
                              HYPRE_Int         **entries,
                              hypre_IndexRef      start,
                              hypre_Index         loop_size,
                              hypre_Box          *A_dbox,
                              HYPRE_Complex      *A_const,
                              HYPRE_Real         *cxyz,
                              HYPRE_Real         *sqcxyz)
{
#define DEVICE_VAR is_device_ptr(A_diag)

   HYPRE_Int             ndim   = hypre_StructMatrixNDim(A);
   HYPRE_Int             volume = hypre_IndexElementMult(loop_size, ndim);
   hypre_Index           ustride;
   HYPRE_Int             d, all_zero;
   HYPRE_Complex         factor;
   HYPRE_Complex         A_diag_const;

   HYPRE_Complex        *A_diag;
   HYPRE_Complex         Ap0 = 0.0, Ap1 = 0.0, Ap2 = 0.0;
   HYPRE_Complex         Ap3 = 0.0, Ap4 = 0.0, Ap5 = 0.0;
   HYPRE_Complex         Ap6 = 0.0, Ap7 = 0.0, Ap8 = 0.0;

   /* Exit if there are no constant coefficients */
   all_zero = 1;
   for (d = 0; d < ndim; d++)
   {
      if (nentries[d] > 0)
      {
         all_zero = 0;
         break;
      }
   }
   if (all_zero)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (diag_is_constant)
   {
      A_diag_const = A_const[diag_entry];
   }
   else
   {
      A_diag = hypre_StructMatrixBoxData(A, Ab, diag_entry);
   }

   hypre_SetIndex(ustride, 1);
   for (d = 0; d < ndim; d++)
   {
      switch (nentries[d])
      {
         case 9:
            Ap8 = A_const[entries[d][8]];
            HYPRE_FALLTHROUGH;

         case 8:
            Ap7 = A_const[entries[d][7]];
            HYPRE_FALLTHROUGH;

         case 7:
            Ap6 = A_const[entries[d][6]];
            HYPRE_FALLTHROUGH;

         case 6:
            Ap5 = A_const[entries[d][5]];
            HYPRE_FALLTHROUGH;

         case 5:
            Ap4 = A_const[entries[d][4]];
            HYPRE_FALLTHROUGH;

         case 4:
            Ap3 = A_const[entries[d][3]];
            HYPRE_FALLTHROUGH;

         case 3:
            Ap2 = A_const[entries[d][2]];
            HYPRE_FALLTHROUGH;

         case 2:
            Ap1 = A_const[entries[d][1]];
            HYPRE_FALLTHROUGH;

         case 1:
            Ap0 = A_const[entries[d][0]];
            HYPRE_FALLTHROUGH;

         case 0:
            break;
      }

      /* Constant diagonal case */
      if (diag_is_constant)
      {
         factor = (A_diag_const < 0.0) ?
                  (HYPRE_Complex) volume : (HYPRE_Complex) -volume;

         switch (nentries[d])
         {
            case 9:
               cxyz[d]   += factor * Ap8;
               sqcxyz[d] += hypre_squared(factor * Ap8);
               HYPRE_FALLTHROUGH;

            case 8:
               cxyz[d]   += factor * Ap7;
               sqcxyz[d] += hypre_squared(factor * Ap7);
               HYPRE_FALLTHROUGH;

            case 7:
               cxyz[d]   += factor * Ap6;
               sqcxyz[d] += hypre_squared(factor * Ap6);
               HYPRE_FALLTHROUGH;

            case 6:
               cxyz[d]   += factor * Ap5;
               sqcxyz[d] += hypre_squared(factor * Ap5);
               HYPRE_FALLTHROUGH;

            case 5:
               cxyz[d]   += factor * Ap4;
               sqcxyz[d] += hypre_squared(factor * Ap4);
               HYPRE_FALLTHROUGH;

            case 4:
               cxyz[d]   += factor * Ap3;
               sqcxyz[d] += hypre_squared(factor * Ap3);
               HYPRE_FALLTHROUGH;

            case 3:
               cxyz[d]   += factor * Ap2;
               sqcxyz[d] += hypre_squared(factor * Ap2);
               HYPRE_FALLTHROUGH;

            case 2:
               cxyz[d]   += factor * Ap1;
               sqcxyz[d] += hypre_squared(factor * Ap1);
               HYPRE_FALLTHROUGH;

            case 1:
               cxyz[d]   += factor * Ap0;
               sqcxyz[d] += hypre_squared(factor * Ap0);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }

         /* Jump to next dimension */
         continue;
      }

#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
      HYPRE_Real cdb   = cxyz[d];
      HYPRE_Real sqcdb = sqcxyz[d];

      switch (nentries[d])
      {
         case 9:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7 + Ap8);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7 + Ap8);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 8:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 7:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 6:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 5:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 4:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 + Ap3);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 + Ap3);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 3:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 2:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1);

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1);

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 1:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, cdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * Ap0;

               cdb += temp;
            }
            hypre_BoxLoop1ReductionEnd(Ai, cdb)

            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sqcdb)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * Ap0;

               sqcdb += hypre_squared(temp);
            }
            hypre_BoxLoop1ReductionEnd(Ai, sqcdb)
            break;

         case 0:
            break;
      }
#else
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Real2 d2(cxyz[d], sqcxyz[d]);
   ReduceSum<HYPRE_Real2> sum2(d2);
#else
   HYPRE_Real cdb   = cxyz[d];
   HYPRE_Real sqcdb = sqcxyz[d];

#if defined(HYPRE_BOX_REDUCTION)
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom:cdb,sqcdb) reduction(+:cdb,sqcdb)
#else
#define HYPRE_BOX_REDUCTION reduction(+:cdb,sqcdb)
#endif

#endif
      switch (nentries[d])
      {
         case 9:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7 + Ap8);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 8:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6 + Ap7);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 7:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5 +
                                         Ap6);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 6:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 +
                                         Ap3 + Ap4 + Ap5);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 5:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 + Ap3 + Ap4);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 4:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2 + Ap3);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 3:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1 + Ap2);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 2:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * (Ap0 + Ap1);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 1:
            hypre_BoxLoop1ReductionBegin(ndim, loop_size, A_dbox,
                                         start, ustride, Ai, sum2)
            {
               HYPRE_Real sign = A_diag[Ai] < 0.0 ? 1.0 : -1.0;
               HYPRE_Real temp = sign * Ap0;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
               HYPRE_Real2 temp2(temp, hypre_squared(temp));
               sum2  += temp2;
#else
               cdb   += temp;
               sqcdb += hypre_squared(temp);
#endif
            }
            hypre_BoxLoop1ReductionEnd(Ai, sum2)
            break;

         case 0:
            break;
      }
#endif

      if (nentries[d] > 0)
      {
#if !defined(HYPRE_USING_KOKKOS) &&\
  (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
         HYPRE_Real2 temp2 = (HYPRE_Real2) sum2;
         cxyz[d]   = temp2.x;
         sqcxyz[d] = temp2.y;
#else
         cxyz[d]   = (HYPRE_Real) cdb;
         sqcxyz[d] = (HYPRE_Real) sqcdb;
#endif
      }
   }
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeCxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeCxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *cxyz,
                       HYPRE_Real         *sqcxyz)
{
   HYPRE_Int              ndim          = hypre_StructMatrixNDim(A);
   hypre_StructGrid      *grid          = hypre_StructMatrixGrid(A);
   hypre_StructStencil   *stencil       = hypre_StructMatrixStencil(A);
   HYPRE_Int             *constant      = hypre_StructMatrixConstant(A);
   HYPRE_Int             *const_indices = hypre_StructMatrixConstIndices(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size  = hypre_StructStencilSize(stencil);
   HYPRE_Int              diag_entry    = hypre_StructStencilDiagEntry(stencil);
   hypre_BoxArray        *compute_boxes = hypre_StructGridBoxes(grid);

   HYPRE_Complex         *A_const;
   hypre_Box             *A_dbox;
   hypre_Box             *compute_box;
   hypre_Index            loop_size;
   hypre_IndexRef         start;

   HYPRE_Int              d, i, k, si;
   HYPRE_Int              depth;
   HYPRE_Int              cdepth[HYPRE_MAXDIM];
   HYPRE_Int              vdepth[HYPRE_MAXDIM];
   HYPRE_Int              csi[HYPRE_MAXDIM][HYPRE_UNROLL_MAXDEPTH];
   HYPRE_Int              vsi[HYPRE_MAXDIM][HYPRE_UNROLL_MAXDEPTH];
   HYPRE_Int             *entries[HYPRE_MAXDIM];
   HYPRE_Int              diag_is_constant;

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy  exec_policy     = hypre_GetExecPolicy1(memory_location);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*----------------------------------------------------------
    * Initialize data
    *----------------------------------------------------------*/

   for (d = 0; d < ndim; d++)
   {
      cxyz[d] = 0.0;
      sqcxyz[d] = 0.0;
   }

#if defined(HYPRE_USING_GPU)
   if (exec_policy == HYPRE_EXEC_DEVICE)
   {
      /* Copy constant coefficients to host memory */
      A_const = hypre_TAlloc(HYPRE_Complex,
                             hypre_StructMatrixVDataOffset(A),
                             HYPRE_MEMORY_HOST);

      hypre_TMemcpy(A_const, hypre_StructMatrixData(A),
                    HYPRE_Complex, hypre_StructMatrixVDataOffset(A),
                    HYPRE_MEMORY_HOST, memory_location);
   }
   else
#endif
   {
      A_const = hypre_StructMatrixData(A);
   }

   /* Check if diagonal entry is constant (1) or variable (0) */
   diag_is_constant = constant[diag_entry] ? 1 : 0;

   /*----------------------------------------------------------
    * Compute cxyz (use arithmetic mean)
    *----------------------------------------------------------*/

   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);
      A_dbox  = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      start   = hypre_BoxIMin(compute_box);
      hypre_BoxGetSize(compute_box, loop_size);

      /* Operate on variable coefficients */
      for (si = 0; si < stencil_size; si += HYPRE_UNROLL_MAXDEPTH)
      {
         depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (stencil_size - si));
         for (d = 0; d < ndim; d++)
         {
            cdepth[d] = vdepth[d] = 0;
         }

         for (k = 0; k < depth; k++)
         {
            if (hypre_StructMatrixConstEntry(A, si + k))
            {
               if (hypre_IndexD(stencil_shape[si + k], 0) != 0)
               {
                  csi[0][cdepth[0]++] = const_indices[si + k];
               }
               else if (hypre_IndexD(stencil_shape[si + k], 1) != 0)
               {
                  csi[1][cdepth[1]++] = const_indices[si + k];
               }
               else if (hypre_IndexD(stencil_shape[si + k], 2) != 0)
               {
                  csi[2][cdepth[2]++] = const_indices[si + k];
               }
            }
            else
            {
               if (hypre_IndexD(stencil_shape[si + k], 0) != 0)
               {
                  vsi[0][vdepth[0]++] = si + k;
               }
               else if (hypre_IndexD(stencil_shape[si + k], 1) != 0)
               {
                  vsi[1][vdepth[1]++] = si + k;
               }
               else if (hypre_IndexD(stencil_shape[si + k], 2) != 0)
               {
                  vsi[2][vdepth[2]++] = si + k;
               }
            }
         }

         /* Collect pointers to variable stencil entries */
         for (d = 0; d < ndim; d++)
         {
            entries[d] = vsi[d];
         }

         /* Compute variable coefficient contributions */
         hypre_PFMGComputeCxyz_core_VC(A, i, diag_entry, diag_is_constant,
                                       vdepth, entries, start, loop_size,
                                       A_dbox, A_const, cxyz, sqcxyz);

         /* Collect pointers to constant stencil entries */
         for (d = 0; d < ndim; d++)
         {
            entries[d] = csi[d];
         }

         /* Compute constant coefficient contributions */
         hypre_PFMGComputeCxyz_core_CC(A, i, diag_entry, diag_is_constant,
                                       cdepth, entries, start, loop_size,
                                       A_dbox, A_const, cxyz, sqcxyz);
      }
   }

   /* Free memory */
   if (A_const != hypre_StructMatrixData(A))
   {
      hypre_TFree(A_const, HYPRE_MEMORY_HOST);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGComputeDxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGComputeDxyz( hypre_StructMatrix *A,
                       HYPRE_Real         *dxyz,
                       HYPRE_Int          *dxyz_flag )
{
   MPI_Comm           comm = hypre_StructMatrixComm(A);
   hypre_StructGrid  *grid = hypre_StructMatrixGrid(A);

   HYPRE_Int          cte_coeff;
   HYPRE_Real         cxyz_max;
   HYPRE_Real         cxyz[HYPRE_MAXDIM];
   HYPRE_Real         sqcxyz[HYPRE_MAXDIM];
   HYPRE_Real         tcxyz[HYPRE_MAXDIM];
   HYPRE_Real         mean[HYPRE_MAXDIM];
   HYPRE_Real         deviation[HYPRE_MAXDIM];

   HYPRE_Int          d, ndim;
   HYPRE_BigInt       global_size;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*----------------------------------------------------------
    * Exit if user gives dxyz different than zero
    *----------------------------------------------------------*/

   if ((dxyz[0] != 0) && (dxyz[1] != 0) && (dxyz[2] != 0))
   {
      *dxyz_flag = 0;

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   ndim        = hypre_StructMatrixNDim(A);
   cte_coeff   = hypre_StructMatrixConstantCoefficient(A);
   global_size = hypre_StructGridGlobalSize(grid);

   /* Compute cxyz and sqcxyz arrays */
   hypre_PFMGComputeCxyz(A, cxyz, sqcxyz);

   /*----------------------------------------------------------
    * Compute dxyz
    *----------------------------------------------------------*/

   if (cte_coeff)
   {
      /* all coefficients constant or variable diagonal */
      global_size = 1;
   }
   else
   {
      /* all coefficients vary with space */
      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = cxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, cxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

      for (d = 0; d < ndim; d++)
      {
         tcxyz[d] = sqcxyz[d];
      }
      hypre_MPI_Allreduce(tcxyz, sqcxyz, ndim, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   }

   for (d = 0; d < ndim; d++)
   {
      mean[d] = cxyz[d] / (HYPRE_Real) global_size;
      deviation[d] = sqcxyz[d] / (HYPRE_Real) global_size;
   }

   cxyz_max = 0.0;
   for (d = 0; d < ndim; d++)
   {
      cxyz_max = hypre_max(cxyz_max, cxyz[d]);
   }

   if (cxyz_max == 0.0)
   {
      /* Do isotropic coarsening */
      for (d = 0; d < ndim; d++)
      {
         cxyz[d] = 1.0;
      }
      cxyz_max = 1.0;
   }

   /* Set dxyz values that are scaled appropriately for the coarsening routine */
   for (d = 0; d < ndim; d++)
   {
      HYPRE_Real max_anisotropy = HYPRE_REAL_MAX / 1000;
      if (cxyz[d] > (cxyz_max / max_anisotropy))
      {
         cxyz[d] /= cxyz_max;
         dxyz[d] = hypre_sqrt(1.0 / cxyz[d]);
      }
      else
      {
         dxyz[d] = hypre_sqrt(max_anisotropy);
      }
   }

   /* Set 'dxyz_flag' if the matrix-coefficient variation is "too large".
    * This is used later to set relaxation weights for Jacobi.
    *
    * Use the "square of the coefficient of variation" = (sigma/mu)^2,
    * where sigma is the standard deviation and mu is the mean.  This is
    * equivalent to computing (d - mu^2)/mu^2 where d is the average of
    * the squares of the coefficients stored in 'deviation'.  Care is
    * taken to avoid dividing by zero when the mean is zero. */

   *dxyz_flag = 0;
   for (d = 0; d < ndim; d++)
   {
      deviation[d] -= mean[d] * mean[d];
      if ( deviation[d] > 0.1 * (mean[d]*mean[d]) )
      {
         *dxyz_flag = 1;
         break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGCoarsen( hypre_Box     *cbox,
                   hypre_Index    periodic,
                   HYPRE_Int      max_levels,
                   HYPRE_Int      dxyz_flag,
                   HYPRE_Real    *dxyz,
                   HYPRE_Int    **cdir_l_ptr,
                   HYPRE_Int    **active_l_ptr,
                   HYPRE_Real   **relax_weights_ptr,
                   HYPRE_Int     *num_levels )
{
   HYPRE_Int      ndim = hypre_BoxNDim(cbox);
   HYPRE_Int     *cdir_l;
   HYPRE_Int     *active_l;
   HYPRE_Real    *relax_weights;

   hypre_Index    coarsen;
   hypre_Index    cindex;
   hypre_Index    stride;

   HYPRE_Real     alpha, beta, min_dxyz;
   HYPRE_Int      d, l, cdir;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Allocate data */
   cdir_l        = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   active_l      = hypre_TAlloc(HYPRE_Int, max_levels, HYPRE_MEMORY_HOST);
   relax_weights = hypre_CTAlloc(HYPRE_Real, max_levels, HYPRE_MEMORY_HOST);

   /* Force relaxation on finest grid */
   hypre_SetIndex(coarsen, 1);
   for (l = 0; l < max_levels; l++)
   {
      /* Initialize min_dxyz */
      min_dxyz = 1;
      for (d = 0; d < ndim; d++)
      {
         min_dxyz += dxyz[d];
      }

      /* Determine cdir */
      cdir = -1;
      alpha = 0.0;
      for (d = 0; d < ndim; d++)
      {
         if ((hypre_BoxIMaxD(cbox, d) > hypre_BoxIMinD(cbox, d)) &&
             (dxyz[d] < min_dxyz))
         {
            min_dxyz = dxyz[d];
            cdir = d;
         }
         alpha += 1.0 / (dxyz[d] * dxyz[d]);
      }
      relax_weights[l] = 1.0;

      /* If it's possible to coarsen, change relax_weights */
      beta = 0.0;
      if (cdir != -1)
      {
         if (dxyz_flag || (ndim == 1))
         {
            relax_weights[l] = 2.0 / 3.0;
         }
         else
         {
            for (d = 0; d < ndim; d++)
            {
               if (d != cdir)
               {
                  beta += 1.0 / (dxyz[d] * dxyz[d]);
               }
            }

            /* determine level Jacobi weights */
            relax_weights[l] = 2.0 / (3.0 - beta / alpha);
         }

         /*    don't coarsen if a periodic direction and not divisible by 2
            or don't coarsen if we've reached max_levels*/
         if (((periodic[cdir]) && (periodic[cdir] % 2)) || l == (max_levels - 1))
         {
            cdir = -1;
         }
      }

      /* stop coarsening */
      if (cdir == -1)
      {
         active_l[l] = 1; /* forces relaxation on coarsest grid */
         break;
      }

      cdir_l[l] = cdir;

      if (hypre_IndexD(coarsen, cdir) != 0)
      {
         /* coarsened previously in this direction, relax level l */
         active_l[l] = 1;
         hypre_SetIndex(coarsen, 0);
      }
      else
      {
         active_l[l] = 0;
      }
      hypre_IndexD(coarsen, cdir) = 1;

      /* set cindex and stride */
      hypre_PFMGSetCIndex(cdir, cindex);
      hypre_PFMGSetStride(cdir, stride);

      /* update dxyz and coarsen cbox*/
      dxyz[cdir] *= 2;
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride, hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride, hypre_BoxIMax(cbox));

      /* update periodic */
      periodic[cdir] /= 2;
   }
   *num_levels = l + 1;

   *cdir_l_ptr        = cdir_l;
   *active_l_ptr      = active_l;
   *relax_weights_ptr = relax_weights;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
