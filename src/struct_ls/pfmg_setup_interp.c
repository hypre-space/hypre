/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_PFMGCreateInterpOp( hypre_StructMatrix *A,
                          HYPRE_Int           cdir,
                          hypre_Index         stride,
                          HYPRE_Int           rap_type )
{
   MPI_Comm              comm            = hypre_StructMatrixComm(A);
   HYPRE_Int             ndim            = hypre_StructMatrixNDim(A);
   hypre_StructGrid     *grid            = hypre_StructMatrixGrid(A);
   hypre_StructStencil  *stencil         = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape   = hypre_StructStencilShape(stencil);
   HYPRE_Int             stencil_size    = hypre_StructStencilSize(stencil);
   HYPRE_Int             diag_entry      = hypre_StructStencilDiagEntry(stencil);
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(A);

   hypre_StructMatrix   *P;
   HYPRE_Int             centries[3] = {0, 1, 2};
   HYPRE_Int             ncentries, i;

   /* Figure out which entries to make constant (ncentries) */
   ncentries = 3; /* Make all entries in P constant by default */
   for (i = 0; i < stencil_size; i++)
   {
      /* Check for entries in A in direction cdir that are variable */
      if (hypre_IndexD(stencil_shape[i], cdir) != 0)
      {
         if (!hypre_StructMatrixConstEntry(A, i))
         {
            ncentries = 1; /* Make only the diagonal of P constant */
            break;
         }
      }
   }

   /* If diagonal of A is not constant and using RAP, do variable interpolation.
    *
    * NOTE: This is important right now because of an issue with Matmult, where
    * it computes constant stencil entries that may not truly be constant along
    * grid boundaries.  RDF: Remove this?  I don't think it's needed anymore,
    * because Matmult uses a mask. */
   if (!hypre_StructMatrixConstEntry(A, diag_entry) && (rap_type == 0))
   {
      ncentries = 1; /* Make only the diagonal of P constant */
   }

   /* Set up the stencil for P */
   stencil_size = 3;
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0);
   }
   hypre_IndexD(stencil_shape[1], cdir) = -1;
   hypre_IndexD(stencil_shape[2], cdir) =  1;
   stencil = hypre_StructStencilCreate(ndim, stencil_size, stencil_shape);

   /* Set up the P matrix */
   P = hypre_StructMatrixCreate(comm, grid, stencil);
   hypre_StructMatrixSetDomainStride(P, stride);
   hypre_StructMatrixSetConstantEntries(P, ncentries, centries);
   hypre_StructMatrixSetMemoryLocation(P, memory_location);

   hypre_StructStencilDestroy(stencil);

   return P;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetupInterpOp_core_CC
 *
 * Core function to compute the constant part of the stencil collapse of the
 * prolongation matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetupInterpOp_core_CC( hypre_StructMatrix   *P,
                                 hypre_StructMatrix   *A,
                                 HYPRE_Int             cdir,
                                 HYPRE_Complex        *Pconst0_ptr,
                                 HYPRE_Complex        *Pconst1_ptr,
                                 HYPRE_Complex        *Pconst2_ptr)
{
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(P);
   HYPRE_ExecutionPolicy  exec            = hypre_GetExecPolicy1(memory_location);
   HYPRE_Complex         *A_data          = hypre_StructMatrixData(A);
   HYPRE_Int             *A_const_indices = hypre_StructMatrixConstIndices(A);
   hypre_StructStencil   *A_stencil       = hypre_StructMatrixStencil(A);
   HYPRE_Int              A_stencil_size  = hypre_StructStencilSize(A_stencil);
   hypre_Index           *A_stencil_shape = hypre_StructStencilShape(A_stencil);
   hypre_StructStencil   *P_stencil       = hypre_StructMatrixStencil(P);
   hypre_Index           *P_stencil_shape = hypre_StructStencilShape(P_stencil);
   HYPRE_Int              Pstenc1         = hypre_IndexD(P_stencil_shape[1], cdir);
   HYPRE_Int              Pstenc2         = hypre_IndexD(P_stencil_shape[2], cdir);

   HYPRE_Complex         *A_const_data_h;
   HYPRE_Complex          Pconst0;
   HYPRE_Complex          Pconst1;
   HYPRE_Complex          Pconst2;
   HYPRE_Int              Astenc, si;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CC");

   /* Set host pointer to constant data entries in A */
   if (exec == HYPRE_EXEC_DEVICE)
   {
      A_const_data_h = hypre_TAlloc(HYPRE_Complex, A_stencil_size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(A_const_data_h, A_data, HYPRE_Complex,
                    A_stencil_size, HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      A_const_data_h = A_data;
   }

   /* Compute the constant part of the stencil collapse (independent of boxes) */
   Pconst0 = 0.0;
   Pconst1 = 0.0;
   Pconst2 = 0.0;
   for (si = 0; si < A_stencil_size; si++)
   {
      if (hypre_StructMatrixConstEntry(A, si))
      {
         Astenc = hypre_IndexD(A_stencil_shape[si], cdir);

         if (Astenc == 0)
         {
            Pconst0 += A_const_data_h[A_const_indices[si]];
         }
         else if (Astenc == Pstenc1)
         {
            Pconst1 -= A_const_data_h[A_const_indices[si]];
         }
         else if (Astenc == Pstenc2)
         {
            Pconst2 -= A_const_data_h[A_const_indices[si]];
         }
      }
   }

   /* Free memory */
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_TFree(A_const_data_h, HYPRE_MEMORY_HOST);
   }

   /* Set output pointers */
   *Pconst0_ptr = Pconst0;
   *Pconst1_ptr = Pconst1;
   *Pconst2_ptr = Pconst2;

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetupInterpOp_core_VC
 *
 * Core function to compute the variable part of the stencil collapse of the
 * prolongation matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetupInterpOp_core_VC( hypre_StructMatrix *P,
                                 hypre_StructMatrix *A,
                                 HYPRE_Int           cdir,
                                 HYPRE_Complex       Pconst0,
                                 HYPRE_Complex       Pconst1,
                                 HYPRE_Complex       Pconst2 )
{
#if defined(HYPRE_UNROLL_MAXDEPTH)
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 9
#define HYPRE_DECLARE_2VARS(n)                             \
    HYPRE_Complex *Ap##n = NULL;                           \
    HYPRE_Int      As##n = 0
#define HYPRE_UPDATE_VALUES(n)                             \
    do {                                                   \
        if      ((As##n) == 0)       mid[Pi] += Ap##n[Ai]; \
        else if ((As##n) == Pstenc1) Pp1[Pi] -= Ap##n[Ai]; \
        else if ((As##n) == Pstenc2) Pp2[Pi] -= Ap##n[Ai]; \
    } while (0)

   HYPRE_Int              ndim            = hypre_StructMatrixNDim(A);
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(P);
   hypre_StructStencil   *A_stencil       = hypre_StructMatrixStencil(A);
   HYPRE_Int              A_stencil_size  = hypre_StructStencilSize(A_stencil);
   hypre_Index           *A_stencil_shape = hypre_StructStencilShape(A_stencil);
   hypre_StructStencil   *P_stencil       = hypre_StructMatrixStencil(P);
   hypre_Index           *P_stencil_shape = hypre_StructStencilShape(P_stencil);
   HYPRE_Int              Pstenc1         = hypre_IndexD(P_stencil_shape[1], cdir);
   HYPRE_Int              Pstenc2         = hypre_IndexD(P_stencil_shape[2], cdir);

   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
   hypre_Box             *A_dbox;
   hypre_Box             *P_dbox;

   HYPRE_Int              P_dbox_volume;
   HYPRE_Complex         *Pp1, *Pp2, *mid;

   HYPRE_Complex         *Ap_l0 = NULL, *Ap_r0 = NULL, *Ap_m0 = NULL;
   HYPRE_Complex         *Ap_l1 = NULL, *Ap_r1 = NULL, *Ap_m1 = NULL;
   HYPRE_Complex         *Ap_l2 = NULL, *Ap_r2 = NULL, *Ap_m2 = NULL;
   HYPRE_Complex         *Ap_l3 = NULL, *Ap_r3 = NULL, *Ap_m3 = NULL;
   HYPRE_Complex         *Ap_l4 = NULL, *Ap_r4 = NULL, *Ap_m4 = NULL;
   HYPRE_Complex         *Ap_l5 = NULL, *Ap_r5 = NULL, *Ap_m5 = NULL;
   HYPRE_Complex         *Ap_l6 = NULL, *Ap_r6 = NULL, *Ap_m6 = NULL;
   HYPRE_Complex         *Ap_l7 = NULL, *Ap_r7 = NULL, *Ap_m7 = NULL;
   HYPRE_Complex         *Ap_l8 = NULL, *Ap_r8 = NULL, *Ap_m8 = NULL;

   HYPRE_Int              i, l, r, m, k, si;
   HYPRE_Int              depth, vdepth, vsi[HYPRE_UNROLL_MAXDEPTH];
   HYPRE_Int              msi[HYPRE_MAX_MMTERMS];
   HYPRE_Int              lsi[HYPRE_MAX_MMTERMS];
   HYPRE_Int              rsi[HYPRE_MAX_MMTERMS];
   hypre_Index            Astart, Astride, Pstart, Pstride;
   hypre_Index            origin, stride, loop_size;
   HYPRE_DECLARE_2VARS(0);
   HYPRE_DECLARE_2VARS(1);
   HYPRE_DECLARE_2VARS(2);
   HYPRE_DECLARE_2VARS(3);
   HYPRE_DECLARE_2VARS(4);
   HYPRE_DECLARE_2VARS(5);
   HYPRE_DECLARE_2VARS(6);
   HYPRE_DECLARE_2VARS(7);
   HYPRE_DECLARE_2VARS(8);

   /* Sanity check */
   if (A_stencil_size >= HYPRE_MAX_MMTERMS)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Reached max. stencil size! Increase HYPRE_MAX_MMTERMS!");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VC");

   /* Gather stencil indices */
   for (i = 0, l = 0, r = 0, m = 0; i < A_stencil_size; i++)
   {
      if (!hypre_StructMatrixConstEntry(A, i))
      {
         if (hypre_IndexD(A_stencil_shape[i], cdir) == 0)
         {
            msi[m++] = i;
         }
         else if (hypre_IndexD(A_stencil_shape[i], cdir) == Pstenc1)
         {
            lsi[l++] = i;
         }
         else if (hypre_IndexD(A_stencil_shape[i], cdir) == Pstenc2)
         {
            rsi[r++] = i;
         }
      }
   }

#if defined(DEBUG_PFMG)
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== PFMG Interp Stencil ===\n");
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "Number of (left, right, middle) stencil. entries: (%d, %d, %d)\n", l ,r, m);
   hypre_ParPrintf(hypre_MPI_COMM_WORLD, "=== PFMG Interp Stencil ===\n");
#endif

   /* Off-diagonal entries are variable */
   compute_box = hypre_BoxCreate(ndim);

   /* Get the stencil space on the base grid for entry 1 of P (valid also for entry 2) */
   hypre_StructMatrixGetStencilSpace(P, 1, 0, origin, stride);

   hypre_CopyToIndex(stride, ndim, Astride);
   hypre_StructMatrixMapDataStride(A, Astride);
   hypre_CopyToIndex(stride, ndim, Pstride);
   hypre_StructMatrixMapDataStride(P, Pstride);

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));
   hypre_ForBoxI(i, compute_boxes)
   {
      hypre_CopyBox(hypre_BoxArrayBox(compute_boxes, i), compute_box);
      hypre_ProjectBox(compute_box, origin, stride);
      hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Astart);
      hypre_StructMatrixMapDataIndex(A, Astart);
      hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Pstart);
      hypre_StructMatrixMapDataIndex(P, Pstart);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

      Pp1 = hypre_StructMatrixBoxData(P, i, 1);
      Pp2 = hypre_StructMatrixBoxData(P, i, 2);

      hypre_BoxGetStrideSize(compute_box, stride, loop_size);

      if (1 && l == 1 && r == 1 && m == 5)
      {
         /* Load data pointers - Left: 1 | Right: 1 | Middle: 5 */
         Ap_l0 = hypre_StructMatrixBoxData(A, i, lsi[0]);
         Ap_r0 = hypre_StructMatrixBoxData(A, i, rsi[0]);
         Ap_m0 = hypre_StructMatrixBoxData(A, i, msi[0]);
         Ap_m1 = hypre_StructMatrixBoxData(A, i, msi[1]);
         Ap_m2 = hypre_StructMatrixBoxData(A, i, msi[2]);
         Ap_m3 = hypre_StructMatrixBoxData(A, i, msi[3]);
         Ap_m4 = hypre_StructMatrixBoxData(A, i, msi[4]);
         hypre_BoxLoop2Begin(ndim, loop_size,
                             A_dbox, Astart, Astride, Ai,
                             P_dbox, Pstart, Pstride, Pi);
         {
            HYPRE_Complex center = Pconst0 + Ap_m0[Ai] + Ap_m1[Ai] + Ap_m2[Ai] + Ap_m3[Ai] + Ap_m4[Ai];
            if (center)
            {
               Pp1[Pi] = (Pconst1 - Ap_l0[Ai]) / center;
               Pp2[Pi] = (Pconst2 - Ap_r0[Ai]) / center;
            }
            else
            {
               Pp1[Pi] = 0.0;
               Pp2[Pi] = 0.0;
            }
         }
         hypre_BoxLoop2End(Ai, Pi);
      }
      else if (1 && l == 3 && r == 3 && m == 9)
      {
         /* Load data pointers - Left: 3 | Right: 3 | Middle: 9 */
         Ap_l0 = hypre_StructMatrixBoxData(A, i, lsi[0]);
         Ap_l1 = hypre_StructMatrixBoxData(A, i, lsi[1]);
         Ap_l2 = hypre_StructMatrixBoxData(A, i, lsi[2]);
         Ap_r0 = hypre_StructMatrixBoxData(A, i, rsi[0]);
         Ap_r1 = hypre_StructMatrixBoxData(A, i, rsi[1]);
         Ap_r2 = hypre_StructMatrixBoxData(A, i, rsi[2]);
         Ap_m0 = hypre_StructMatrixBoxData(A, i, msi[0]);
         Ap_m1 = hypre_StructMatrixBoxData(A, i, msi[1]);
         Ap_m2 = hypre_StructMatrixBoxData(A, i, msi[2]);
         Ap_m3 = hypre_StructMatrixBoxData(A, i, msi[3]);
         Ap_m4 = hypre_StructMatrixBoxData(A, i, msi[4]);
         Ap_m5 = hypre_StructMatrixBoxData(A, i, msi[5]);
         Ap_m6 = hypre_StructMatrixBoxData(A, i, msi[6]);
         Ap_m7 = hypre_StructMatrixBoxData(A, i, msi[7]);
         Ap_m8 = hypre_StructMatrixBoxData(A, i, msi[8]);
         hypre_BoxLoop2Begin(ndim, loop_size,
                             A_dbox, Astart, Astride, Ai,
                             P_dbox, Pstart, Pstride, Pi);
         {
            HYPRE_Complex center = Pconst0   + Ap_m0[Ai] + Ap_m1[Ai] + Ap_m2[Ai] + Ap_m3[Ai] +
                                   Ap_m4[Ai] + Ap_m5[Ai] + Ap_m6[Ai] + Ap_m7[Ai] + Ap_m8[Ai];

            if (center)
            {
               Pp1[Pi] = (Pconst1 - Ap_l0[Ai] - Ap_l1[Ai] - Ap_l2[Ai]) / center;
               Pp2[Pi] = (Pconst2 - Ap_r0[Ai] - Ap_r1[Ai] - Ap_r2[Ai]) / center;
            }
            else
            {
               Pp1[Pi] = 0.0;
               Pp2[Pi] = 0.0;
            }
         }
         hypre_BoxLoop2End(Ai, Pi);
      }
      else if (1 && l == 9 && r == 9 && m == 9)
      {
         /* Load data pointers - Left: 9 | Right: 9 | Middle: 9 */
         Ap_l0 = hypre_StructMatrixBoxData(A, i, lsi[0]);
         Ap_l1 = hypre_StructMatrixBoxData(A, i, lsi[1]);
         Ap_l2 = hypre_StructMatrixBoxData(A, i, lsi[2]);
         Ap_l3 = hypre_StructMatrixBoxData(A, i, lsi[3]);
         Ap_l4 = hypre_StructMatrixBoxData(A, i, lsi[4]);
         Ap_l5 = hypre_StructMatrixBoxData(A, i, lsi[5]);
         Ap_l6 = hypre_StructMatrixBoxData(A, i, lsi[6]);
         Ap_l7 = hypre_StructMatrixBoxData(A, i, lsi[7]);
         Ap_l8 = hypre_StructMatrixBoxData(A, i, lsi[8]);
         Ap_r0 = hypre_StructMatrixBoxData(A, i, rsi[0]);
         Ap_r1 = hypre_StructMatrixBoxData(A, i, rsi[1]);
         Ap_r2 = hypre_StructMatrixBoxData(A, i, rsi[2]);
         Ap_r3 = hypre_StructMatrixBoxData(A, i, rsi[3]);
         Ap_r4 = hypre_StructMatrixBoxData(A, i, rsi[4]);
         Ap_r5 = hypre_StructMatrixBoxData(A, i, rsi[5]);
         Ap_r6 = hypre_StructMatrixBoxData(A, i, rsi[6]);
         Ap_r7 = hypre_StructMatrixBoxData(A, i, rsi[7]);
         Ap_r8 = hypre_StructMatrixBoxData(A, i, rsi[8]);
         Ap_m0 = hypre_StructMatrixBoxData(A, i, msi[0]);
         Ap_m1 = hypre_StructMatrixBoxData(A, i, msi[1]);
         Ap_m2 = hypre_StructMatrixBoxData(A, i, msi[2]);
         Ap_m3 = hypre_StructMatrixBoxData(A, i, msi[3]);
         Ap_m4 = hypre_StructMatrixBoxData(A, i, msi[4]);
         Ap_m5 = hypre_StructMatrixBoxData(A, i, msi[5]);
         Ap_m6 = hypre_StructMatrixBoxData(A, i, msi[6]);
         Ap_m7 = hypre_StructMatrixBoxData(A, i, msi[7]);
         Ap_m8 = hypre_StructMatrixBoxData(A, i, msi[8]);
         hypre_BoxLoop2Begin(ndim, loop_size,
                             A_dbox, Astart, Astride, Ai,
                             P_dbox, Pstart, Pstride, Pi);
         {
            HYPRE_Complex center = Pconst0   + Ap_m0[Ai] + Ap_m1[Ai] + Ap_m2[Ai] + Ap_m3[Ai] +
                                   Ap_m4[Ai] + Ap_m5[Ai] + Ap_m6[Ai] + Ap_m7[Ai] + Ap_m8[Ai];

            if (center)
            {
               Pp1[Pi] = (Pconst1   - Ap_l0[Ai] - Ap_l1[Ai] - Ap_l2[Ai] - Ap_l3[Ai] -
                          Ap_l4[Ai] - Ap_l5[Ai] - Ap_l6[Ai] - Ap_l7[Ai] - Ap_l8[Ai]) / center;
               Pp2[Pi] = (Pconst2   - Ap_r0[Ai] - Ap_r1[Ai] - Ap_r2[Ai] - Ap_r3[Ai] -
                          Ap_r4[Ai] - Ap_r5[Ai] - Ap_r6[Ai] - Ap_r7[Ai] - Ap_r8[Ai]) / center;
            }
            else
            {
               Pp1[Pi] = 0.0;
               Pp2[Pi] = 0.0;
            }
         }
         hypre_BoxLoop2End(Ai, Pi);
      }
      else
      {
         /* Allocate array for storing center coefficient */
         P_dbox_volume = hypre_BoxVolume(P_dbox);
         mid = hypre_TAlloc(HYPRE_Complex, P_dbox_volume, memory_location);

         /* Phase 1: Set initial coefficient values in P */
         hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
         {
            mid[Pi] = Pconst0;
            Pp1[Pi] = Pconst1;
            Pp2[Pi] = Pconst2;
         }
         hypre_BoxLoop1End(Pi);

         /* Phase 2: Update coefficients in P with variable coefficients from A */
         for (si = 0; si < A_stencil_size; si += HYPRE_UNROLL_MAXDEPTH)
         {
            depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (A_stencil_size - si));
            for (k = 0, vdepth = 0; k < depth; k++)
            {
               if (!hypre_StructMatrixConstEntry(A, si + k))
               {
                  vsi[vdepth++] = si + k;
               }
            }

            /* Exit loop if there are no variable stencil coefficients left */
            if (!vdepth)
            {
               continue;
            }

            switch (vdepth)
            {
               case 9:
                  Ap8 = hypre_StructMatrixBoxData(A, i, vsi[8]);
                  As8 = hypre_IndexD(A_stencil_shape[vsi[8]], cdir);
                  HYPRE_FALLTHROUGH;

               case 8:
                  Ap7 = hypre_StructMatrixBoxData(A, i, vsi[7]);
                  As7 = hypre_IndexD(A_stencil_shape[vsi[7]], cdir);
                  HYPRE_FALLTHROUGH;

               case 7:
                  Ap6 = hypre_StructMatrixBoxData(A, i, vsi[6]);
                  As6 = hypre_IndexD(A_stencil_shape[vsi[6]], cdir);
                  HYPRE_FALLTHROUGH;

               case 6:
                  Ap5 = hypre_StructMatrixBoxData(A, i, vsi[5]);
                  As5 = hypre_IndexD(A_stencil_shape[vsi[5]], cdir);
                  HYPRE_FALLTHROUGH;

               case 5:
                  Ap4 = hypre_StructMatrixBoxData(A, i, vsi[4]);
                  As4 = hypre_IndexD(A_stencil_shape[vsi[4]], cdir);
                  HYPRE_FALLTHROUGH;

               case 4:
                  Ap3 = hypre_StructMatrixBoxData(A, i, vsi[3]);
                  As3 = hypre_IndexD(A_stencil_shape[vsi[3]], cdir);
                  HYPRE_FALLTHROUGH;

               case 3:
                  Ap2 = hypre_StructMatrixBoxData(A, i, vsi[2]);
                  As2 = hypre_IndexD(A_stencil_shape[vsi[2]], cdir);
                  HYPRE_FALLTHROUGH;

               case 2:
                  Ap1 = hypre_StructMatrixBoxData(A, i, vsi[1]);
                  As1 = hypre_IndexD(A_stencil_shape[vsi[1]], cdir);
                  HYPRE_FALLTHROUGH;

               case 1:
                  Ap0 = hypre_StructMatrixBoxData(A, i, vsi[0]);
                  As0 = hypre_IndexD(A_stencil_shape[vsi[0]], cdir);
            }

            switch (vdepth)
            {
               case 9:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(8);
                     HYPRE_UPDATE_VALUES(7);
                     HYPRE_UPDATE_VALUES(6);
                     HYPRE_UPDATE_VALUES(5);
                     HYPRE_UPDATE_VALUES(4);
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 8:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(7);
                     HYPRE_UPDATE_VALUES(6);
                     HYPRE_UPDATE_VALUES(5);
                     HYPRE_UPDATE_VALUES(4);
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 7:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(6);
                     HYPRE_UPDATE_VALUES(5);
                     HYPRE_UPDATE_VALUES(4);
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 6:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(5);
                     HYPRE_UPDATE_VALUES(4);
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 5:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(4);
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 4:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(3);
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 3:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(2);
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 2:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(1);
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;

               case 1:
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      A_dbox, Astart, Astride, Ai,
                                      P_dbox, Pstart, Pstride, Pi);
                  {
                     HYPRE_UPDATE_VALUES(0);
                  }
                  hypre_BoxLoop2End(Ai, Pi);
                  break;
            }
         }

         /* Phase 3: set final coefficients */
         hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
         {
            if (mid[Pi])
            {
               /* Average out prolongation coefficients */
               Pp1[Pi] /= mid[Pi];
               Pp2[Pi] /= mid[Pi];
            }
            else
            {
               /* For some reason the interpolation coefficients sum to zero */
               Pp1[Pi] = 0.0;
               Pp2[Pi] = 0.0;
            }
         }
         hypre_BoxLoop1End(Pi);

         /* Free memory */
         hypre_TFree(mid, memory_location);
      }
   }

   /* Free memory */
   hypre_BoxDestroy(compute_box);

#undef HYPRE_UPDATE_VALUES
#undef HYPRE_UNROLL_MAXDEPTH

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetupInterpOp( hypre_StructMatrix *P,
                         hypre_StructMatrix *A,
                         HYPRE_Int           cdir )
{
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(P);
   HYPRE_Int              constant;

   HYPRE_Complex          Pconst0, Pconst1, Pconst2;
   HYPRE_Complex          one  = 1.0;
   HYPRE_Complex          half = 0.5;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("PFMGSetupInterp");

   /* 0: Only the diagonal is constant
      1: All entries are constant */
   constant = (hypre_StructMatrixConstEntry(P, 1)) ? 1 : 0;

   /*----------------------------------------------------------
    * Compute Prolongation Matrix
    *----------------------------------------------------------*/

   /* Set center (diagonal) coefficient to 1, since it is constant */
   hypre_TMemcpy(hypre_StructMatrixConstData(P, 0), &one, HYPRE_Complex, 1,
                 memory_location, HYPRE_MEMORY_HOST);

   if (constant)
   {
      /* Off-diagonal entries are constant */
      hypre_TMemcpy(hypre_StructMatrixConstData(P, 1), &half, HYPRE_Complex, 1,
                    memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_StructMatrixConstData(P, 2), &half, HYPRE_Complex, 1,
                    memory_location, HYPRE_MEMORY_HOST);
   }
   else
   {
      /* Set prolongation entries derived from constant coefficients in A */
      hypre_PFMGSetupInterpOp_core_CC(P, A, cdir, &Pconst0, &Pconst1, &Pconst2);

      /* Set prolongation entries derived from variable coefficients in A */
      hypre_PFMGSetupInterpOp_core_VC(P, A, cdir, Pconst0, Pconst1, Pconst2);
   }

   /* Assemble prolongation matrix */
   hypre_StructMatrixAssemble(P);

   /* The following call is needed to prevent cases where interpolation reaches
    * outside the boundary with nonzero coefficient */
   hypre_StructMatrixClearBoundary(P);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
