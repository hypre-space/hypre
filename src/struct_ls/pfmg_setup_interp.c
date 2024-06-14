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
hypre_zPFMGCreateInterpOp( hypre_StructMatrix *A,
                           HYPRE_Int           cdir,
                           hypre_Index         stride,
                           HYPRE_Int           rap_type )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructMatrix   *P;
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size, diag_entry;
   HYPRE_Int             centries[3] = {0, 1, 2};
   HYPRE_Int             ncentries, i;

   /* Figure out which entries to make constant (ncentries) */
   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   diag_entry    = hypre_StructStencilDiagEntry(stencil);
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
    * NOTE: This is important right now because of an issue with MatMult, where
    * it computes constant stencil entries that may not truly be constant along
    * grid boundaries. */
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
   P = hypre_StructMatrixCreate(hypre_StructMatrixComm(A), hypre_StructMatrixGrid(A), stencil);
   hypre_StructMatrixSetDomainStride(P, stride);
   hypre_StructMatrixSetConstantEntries(P, ncentries, centries);

   hypre_StructStencilDestroy(stencil);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_zPFMGSetupInterpOp( hypre_StructMatrix *P,
                          hypre_StructMatrix *A,
                          HYPRE_Int           cdir )
{
   HYPRE_Int              ndim = hypre_StructMatrixNDim(A);
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Box             *A_dbox;
   hypre_Box             *P_dbox;

   HYPRE_Real            *Ap, *Pp0, *Pp1, *Pp2;
   HYPRE_Real             Pconst0, Pconst1, Pconst2;
   HYPRE_Int              constant;

   hypre_StructStencil   *A_stencil;
   hypre_Index           *A_stencil_shape;
   HYPRE_Int              A_stencil_size;
   hypre_StructStencil   *P_stencil;
   hypre_Index           *P_stencil_shape;
   HYPRE_Int             *ventries, nventries;

   HYPRE_Int              Astenc, Pstenc1, Pstenc2;
   hypre_Index            Astart, Astride, Pstart, Pstride;
   hypre_Index            origin, stride, loop_size;

   HYPRE_Int              i, si;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   A_stencil       = hypre_StructMatrixStencil(A);
   A_stencil_shape = hypre_StructStencilShape(A_stencil);
   A_stencil_size  = hypre_StructStencilSize(A_stencil);

   P_stencil       = hypre_StructMatrixStencil(P);
   P_stencil_shape = hypre_StructStencilShape(P_stencil);

   constant = 0; /* Only the diagonal is constant */
   if (hypre_StructMatrixConstEntry(P, 1))
   {
      constant = 1; /* All entries are constant */
   }

   /*----------------------------------------------------------
    * Compute P
    *----------------------------------------------------------*/

   /* Set center coefficient to 1 */
   Pp0 = hypre_StructMatrixConstData(P, 0);
   Pp0[0] = 1;

   if (constant)
   {
      /* Off-diagonal entries are constant */

      Pp1 = hypre_StructMatrixConstData(P, 1);
      Pp2 = hypre_StructMatrixConstData(P, 2);

      Pp1[0] = 0.5;
      Pp2[0] = 0.5;
   }
   else
   {
      /* Off-diagonal entries are variable */

      compute_box = hypre_BoxCreate(ndim);

      Pstenc1 = hypre_IndexD(P_stencil_shape[1], cdir);
      Pstenc2 = hypre_IndexD(P_stencil_shape[2], cdir);

      /* Compute the constant part of the stencil collapse */
      ventries = hypre_TAlloc(HYPRE_Int, A_stencil_size, HYPRE_MEMORY_HOST);
      nventries = 0;
      Pconst0 = 0.0;
      Pconst1 = 0.0;
      Pconst2 = 0.0;
      for (si = 0; si < A_stencil_size; si++)
      {
         if (hypre_StructMatrixConstEntry(A, si))
         {
            Ap = hypre_StructMatrixConstData(A, si);
            Astenc = hypre_IndexD(A_stencil_shape[si], cdir);

            if (Astenc == 0)
            {
               Pconst0 += Ap[0];
            }
            else if (Astenc == Pstenc1)
            {
               Pconst1 -= Ap[0];
            }
            else if (Astenc == Pstenc2)
            {
               Pconst2 -= Ap[0];
            }
         }
         else
         {
            ventries[nventries++] = si;
         }
      }

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

         hypre_BoxLoop2Begin(ndim, loop_size,
                             A_dbox, Astart, Astride, Ai,
                             P_dbox, Pstart, Pstride, Pi);
         {
            HYPRE_Int    vi, si, Astenc;
            HYPRE_Real   center;
            HYPRE_Real  *Ap;

            center  = Pconst0;
            Pp1[Pi] = Pconst1;
            Pp2[Pi] = Pconst2;
            for (vi = 0; vi < nventries; vi++)
            {
               si = ventries[vi];
               Ap = hypre_StructMatrixBoxData(A, i, si);
               Astenc = hypre_IndexD(A_stencil_shape[si], cdir);

               if (Astenc == 0)
               {
                  center += Ap[Ai];
               }
               else if (Astenc == Pstenc1)
               {
                  Pp1[Pi] -= Ap[Ai];
               }
               else if (Astenc == Pstenc2)
               {
                  Pp2[Pi] -= Ap[Ai];
               }
            }

            if (center)
            {
               Pp1[Pi] /= center;
               Pp2[Pi] /= center;
            }
            else
            {
               /* For some reason the interpolation coefficients sum to zero */
               Pp1[Pi] = 0.0;
               Pp2[Pi] = 0.0;
            }
         }
         hypre_BoxLoop2End(Ai, Pi);
      }

      hypre_BoxDestroy(compute_box);
      hypre_TFree(ventries, HYPRE_MEMORY_HOST);
   }

   hypre_StructMatrixAssemble(P);
   /* The following call is needed to prevent cases where interpolation reaches
    * outside the boundary with nonzero coefficient */
   hypre_StructMatrixClearBoundary(P);

   return hypre_error_flag;
}


/* TODO (VPM): Incorporate the specialized code below for computing prolongation */
#if 0

/*--------------------------------------------------------------------------
 * RDF: OLD STUFF TO PHASE OUT
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_PFMGCreateInterpOp( hypre_StructMatrix *A,
                          hypre_StructGrid   *cgrid,
                          HYPRE_Int           cdir,
                          HYPRE_Int           rap_type )
{
   hypre_StructMatrix   *P;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   HYPRE_Int             stencil_dim;

   HYPRE_Int             num_ghost[] = {1, 1, 1, 1, 1, 1};

   HYPRE_Int             i;
   HYPRE_Int             constant_coefficient;

   /* set up stencil */
   stencil_size = 2;
   stencil_dim = hypre_StructStencilNDim(hypre_StructMatrixStencil(A));
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex3(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;
   stencil =
      hypre_StructStencilCreate(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   P = hypre_StructMatrixCreate(hypre_StructMatrixComm(A), cgrid, stencil);
   HYPRE_StructMatrixSetNumGhost(P, num_ghost);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ((constant_coefficient) && !(constant_coefficient == 2 && rap_type == 0))
   {
      HYPRE_Int *entries;

      entries = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < stencil_size; i++)
      {
         entries[i] = i;
      }
      hypre_StructMatrixSetConstantEntries(P, stencil_size, entries);
      hypre_TFree(entries, HYPRE_MEMORY_HOST);
   }

   hypre_StructStencilDestroy(stencil);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetupInterpOp( hypre_StructMatrix *A,
                         HYPRE_Int           cdir,
                         hypre_Index         findex,
                         hypre_Index         stride,
                         hypre_StructMatrix *P,
                         HYPRE_Int           rap_type )
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;

   hypre_Box             *A_dbox;
   hypre_Box             *P_dbox;

   HYPRE_Real            *Pp0, *Pp1;
   HYPRE_Int              constant_coefficient;

   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
   hypre_StructStencil   *P_stencil;
   hypre_Index           *P_stencil_shape;

   HYPRE_Int              Pstenc0, Pstenc1;

   hypre_Index            loop_size;
   hypre_Index            start;
   hypre_IndexRef         startc;
   hypre_Index            stridec;

   HYPRE_Int              i, si;

   HYPRE_Int              si0, si1;
   HYPRE_Int              mrk0, mrk1;
   HYPRE_Int              d;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   P_stencil       = hypre_StructMatrixStencil(P);
   P_stencil_shape = hypre_StructStencilShape(P_stencil);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   /*----------------------------------------------------------
    * Find stencil enties in A corresponding to P
    *----------------------------------------------------------*/

   si0 = -1;
   si1 = -1;
   for (si = 0; si < stencil_size; si++)
   {
      mrk0 = 0;
      mrk1 = 0;
      for (d = 0; d < hypre_StructStencilNDim(stencil); d++)
      {
         if (hypre_IndexD(stencil_shape[si], d) ==
             hypre_IndexD(P_stencil_shape[0], d))
         {
            mrk0++;
         }
         if (hypre_IndexD(stencil_shape[si], d) ==
             hypre_IndexD(P_stencil_shape[1], d))
         {
            mrk1++;
         }
      }
      if (mrk0 == hypre_StructStencilNDim(stencil))
      {
         si0 = si;
      }
      if (mrk1 == hypre_StructStencilNDim(stencil))
      {
         si1 = si;
      }
   }

   hypre_SetIndex3(stridec, 1, 1, 1);

   /*----------------------------------------------------------
    * Compute P
    *----------------------------------------------------------*/

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));
   hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

      Pp0 = hypre_StructMatrixBoxData(P, i, 0);
      Pp1 = hypre_StructMatrixBoxData(P, i, 1);

      Pstenc0 = hypre_IndexD(P_stencil_shape[0], cdir);
      Pstenc1 = hypre_IndexD(P_stencil_shape[1], cdir);

      startc  = hypre_BoxIMin(compute_box);
      hypre_StructMapCoarseToFine(startc, findex, stride, start);

      hypre_BoxGetStrideSize(compute_box, stridec, loop_size);

      if ( constant_coefficient == 1 )
         /* all coefficients are constant */
      {
         hypre_PFMGSetupInterpOp_CC1
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
      }

      else if ( constant_coefficient == 2 )
         /* all coefficients are constant except the diagonal is variable */
      {
         hypre_PFMGSetupInterpOp_CC2
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
      }

      else
         /* constant_coefficient == 0 , all coefficients in A vary */
      {
         hypre_PFMGSetupInterpOp_CC0
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
      }
   }

#if 0
   hypre_StructMatrixAssemble(P);
#else
   hypre_StructInterpAssemble(A, P, 0, cdir, findex, stride);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0( HYPRE_Int           i, /* box index */
                             hypre_StructMatrix *A,
                             hypre_Box          *A_dbox,
                             HYPRE_Int           cdir,
                             hypre_Index         stride,
                             hypre_Index         stridec,
                             hypre_Index         start,
                             hypre_IndexRef      startc,
                             hypre_Index         loop_size,
                             hypre_Box          *P_dbox,
                             HYPRE_Int           Pstenc0,
                             HYPRE_Int           Pstenc1,
                             HYPRE_Real         *Pp0,
                             HYPRE_Real         *Pp1,
                             HYPRE_Int           rap_type,
                             HYPRE_Int           si0,
                             HYPRE_Int           si1 )
{
   HYPRE_Int            **data_indices    = hypre_StructMatrixDataIndices(A);
   HYPRE_Complex         *matrixA_data    = hypre_StructMatrixData(A);
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);
   hypre_StructStencil   *stencil         = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape   = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size    = hypre_StructStencilSize(stencil);

   HYPRE_Int              warning_cnt     = 0;
   HYPRE_Int             *data_indices_boxi_d;
   hypre_Index           *stencil_shape_d;

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      data_indices_boxi_d = hypre_TAlloc(HYPRE_Int, stencil_size, memory_location);
      stencil_shape_d     = hypre_TAlloc(hypre_Index, stencil_size, memory_location);

      hypre_TMemcpy(data_indices_boxi_d, data_indices[i], HYPRE_Int, stencil_size,
                    memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(stencil_shape_d, stencil_shape, hypre_Index, stencil_size,
                    memory_location, HYPRE_MEMORY_HOST);
   }
   else
   {
      data_indices_boxi_d = data_indices[i];
      stencil_shape_d     = stencil_shape;
   }

#define DEVICE_VAR is_device_ptr(Pp0, Pp1, matrixA_data, stencil_shape_d, data_indices_boxi_d)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      HYPRE_Int      si, mrk0, mrk1;
      HYPRE_Int      Astenc;
      HYPRE_Real     center;
      HYPRE_Real    *Ap;

      center  = 0.0;
      Pp0[Pi] = 0.0;
      Pp1[Pi] = 0.0;
      mrk0 = 0;
      mrk1 = 0;

      for (si = 0; si < stencil_size; si++)
      {
         Ap = hypre_StructMatrixBoxData(A, i, si);
         Astenc = hypre_IndexD(stencil_shape[si], cdir);

         if (Astenc == 0)
         {
            center += Ap[Ai];
         }
         else if (Astenc == Pstenc0)
         {
            Pp0[Pi] -= Ap[Ai];
         }
         else if (Astenc == Pstenc1)
         {
            Pp1[Pi] -= Ap[Ai];
         }

         if (si == si0 && Ap[Ai] == 0.0)
         {
            mrk0++;
         }
         if (si == si1 && Ap[Ai] == 0.0)
         {
            mrk1++;
         }
      }

      if (!center)
      {
         warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         Pp0[Pi] /= center;
         Pp1[Pi] /= center;
      }

      /*----------------------------------------------
       * Set interpolation weight to zero, if stencil
       * entry in same direction is zero. Prevents
       * interpolation and operator stencils reaching
       * outside domain.
       *----------------------------------------------*/
      if (mrk0 != 0)
      {
         Pp0[Pi] = 0.0;
      }
      if (mrk1 != 0)
      {
         Pp1[Pi] = 0.0;
      }
   }
   hypre_BoxLoop2End(Ai, Pi);

   if (warning_cnt)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Warning 0 center in interpolation. Setting interp = 0.");
   }

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      hypre_TFree(data_indices_boxi_d, memory_location);
      hypre_TFree(stencil_shape_d, memory_location);
   }

   return hypre_error_flag;
}

#if CC0_IMPLEMENTATION == 1

HYPRE_Int
hypre_PFMGSetupInterpOp_CC0
( HYPRE_Int           i, /* box index */
  hypre_StructMatrix *A,
  hypre_Box          *A_dbox,
  HYPRE_Int           cdir,
  hypre_Index         stride,
  hypre_Index         stridec,
  hypre_Index         start,
  hypre_IndexRef      startc,
  hypre_Index         loop_size,
  hypre_Box          *P_dbox,
  HYPRE_Int           Pstenc0,
  HYPRE_Int           Pstenc1,
  HYPRE_Real         *Pp0,
  HYPRE_Real         *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int              warning_cnt = 0;
   HYPRE_Int              dim, si, loop_length = 1, Astenc;
   HYPRE_Real            *Ap, *center, *Ap0, *Ap1;
   HYPRE_MemoryLocation   memory_location = hypre_StructMatrixMemoryLocation(A);

   for (dim = 0; dim < hypre_StructMatrixNDim(A); dim++)
   {
      loop_length *= loop_size[dim];
   }
   center = hypre_CTAlloc(HYPRE_Real, loop_length, memory_location);

   for (si = 0; si < stencil_size; si++)
   {
      Ap     = hypre_StructMatrixBoxData(A, i, si);
      Astenc = hypre_IndexD(stencil_shape[si], cdir);

      if (Astenc == 0)
      {
#define DEVICE_VAR is_device_ptr(center, Ap)
         hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         center[idx] += Ap[Ai];
         hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
      else if (Astenc == Pstenc0)
      {
#define DEVICE_VAR is_device_ptr(Pp0, Ap)
         hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         Pp0[Pi] -= Ap[Ai];
         hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
      else if (Astenc == Pstenc1)
      {
#define DEVICE_VAR is_device_ptr(Pp1, Ap)
         hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         Pp1[Pi] -= Ap[Ai];
         hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
   }

   Ap0 = hypre_StructMatrixBoxData(A, i, si0);
   Ap1 = hypre_StructMatrixBoxData(A, i, si1);
#define DEVICE_VAR is_device_ptr(center, Pp0, Pp1, Ap0, Ap1)
   hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start,  stride,  Ai,
                       P_dbox, startc, stridec, Pi)
   HYPRE_Real cval = center[idx];
   if (Ap0[Ai] == 0.0 || cval == 0.0)
   {
      Pp0[Pi] = 0.0;
   }
   else
   {
      Pp0[Pi] /= cval;
   }

   if (Ap1[Ai] == 0.0 || cval == 0.0)
   {
      Pp1[Pi] = 0.0;
   }
   else
   {
      Pp1[Pi] /= cval;
   }
   hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR

   if (warning_cnt)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Warning 0 center in interpolation. Setting interp = 0.");
   }

   hypre_TFree(center, memory_location);

   return hypre_error_flag;
}

#endif /* #if CC0_IMPLEMENTATION == 1 */

HYPRE_Int
hypre_PFMGSetupInterpOp_CC1
( HYPRE_Int           i, /* box index, doesn't matter */
  hypre_StructMatrix *A,
  hypre_Box          *A_dbox,
  HYPRE_Int           cdir,
  hypre_Index         stride,
  hypre_Index         stridec,
  hypre_Index         start,
  hypre_IndexRef      startc,
  hypre_Index         loop_size,
  hypre_Box          *P_dbox,
  HYPRE_Int           Pstenc0,
  HYPRE_Int           Pstenc1,
  HYPRE_Real         *Pp0,
  HYPRE_Real         *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   HYPRE_UNUSED_VAR(A_dbox);
   HYPRE_UNUSED_VAR(stride);
   HYPRE_UNUSED_VAR(stridec);
   HYPRE_UNUSED_VAR(start);
   HYPRE_UNUSED_VAR(startc);
   HYPRE_UNUSED_VAR(loop_size);
   HYPRE_UNUSED_VAR(P_dbox);
   HYPRE_UNUSED_VAR(rap_type);

   HYPRE_Int              si;
   HYPRE_Int              Ai, Pi;
   HYPRE_Real            *Ap;
   HYPRE_Real             center;
   HYPRE_Int              Astenc;
   HYPRE_Int              mrk0, mrk1;
   hypre_StructStencil   *stencil = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int              warning_cnt = 0;

   Ai = hypre_CCBoxIndexRank(A_dbox, start );
   Pi = hypre_CCBoxIndexRank(P_dbox, startc);

   center  = 0.0;
   Pp0[Pi] = 0.0;
   Pp1[Pi] = 0.0;
   mrk0 = 0;
   mrk1 = 0;

   for (si = 0; si < stencil_size; si++)
   {
      Ap = hypre_StructMatrixBoxData(A, i, si);
      Astenc = hypre_IndexD(stencil_shape[si], cdir);

      if (Astenc == 0)
      {
         center += Ap[Ai];
      }
      else if (Astenc == Pstenc0)
      {
         Pp0[Pi] -= Ap[Ai];
      }
      else if (Astenc == Pstenc1)
      {
         Pp1[Pi] -= Ap[Ai];
      }

      if (si == si0 && Ap[Ai] == 0.0)
      {
         mrk0++;
      }
      if (si == si1 && Ap[Ai] == 0.0)
      {
         mrk1++;
      }
   }
   if (!center)
   {
      warning_cnt++;
      Pp0[Pi] = 0.0;
      Pp1[Pi] = 0.0;
   }
   else
   {
      Pp0[Pi] /= center;
      Pp1[Pi] /= center;
   }

   /*----------------------------------------------
    * Set interpolation weight to zero, if stencil
    * entry in same direction is zero.
    * For variable coefficients, this was meant to prevent
    * interpolation and operator stencils from reaching
    * outside the domain.
    * For constant coefficients it will hardly ever happen
    * (means the stencil point shouldn't have been defined there)
    * but it's possible and then it would still make sense to
    * do this.
    *----------------------------------------------*/
   if (mrk0 != 0)
   {
      Pp0[Pi] = 0.0;
   }
   if (mrk1 != 0)
   {
      Pp1[Pi] = 0.0;
   }

   if (warning_cnt)
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGSetupInterpOp_CC2
( HYPRE_Int           i, /* box index */
  hypre_StructMatrix *A,
  hypre_Box          *A_dbox,
  HYPRE_Int           cdir,
  hypre_Index         stride,
  hypre_Index         stridec,
  hypre_Index         start,
  hypre_IndexRef      startc,
  hypre_Index         loop_size,
  hypre_Box          *P_dbox,
  HYPRE_Int           Pstenc0,
  HYPRE_Int           Pstenc1,
  HYPRE_Real         *Pp0,
  HYPRE_Real         *Pp1,
  HYPRE_Int           rap_type,
  HYPRE_Int           si0,
  HYPRE_Int           si1 )
{
   hypre_StructStencil   *stencil            = hypre_StructMatrixStencil(A);
   hypre_Index           *stencil_shape      = hypre_StructStencilShape(stencil);
   HYPRE_Int              stencil_size       = hypre_StructStencilSize(stencil);
   HYPRE_Int              stencil_diag_entry = hypre_StructStencilDiagEntry(stencil);

   HYPRE_Int              si;
   HYPRE_Int              Ai;
   HYPRE_Int              Pi;
   HYPRE_Real            *Ap;
   HYPRE_Real             P0, P1;
   HYPRE_Real             center, center_offd;
   HYPRE_Int              Astenc;
   HYPRE_Int              mrk0, mrk1, mrk0_offd, mrk1_offd;
   HYPRE_Int              warning_cnt = 0;

   if ( rap_type != 0 )
   {
      /* simply force P to be constant coefficient, all 0.5's */
      Pi = hypre_CCBoxIndexRank(P_dbox, startc);
      Pp0[Pi] = 0.5;
      Pp1[Pi] = 0.5;
   }
   else
   {
      /* Most coeffients of A go into P like for constant_coefficient=1.
         But P is entirely variable coefficient, because the diagonal of A is
         variable, and hence "center" below is variable. So we use the constant
         coefficient calculation to initialize the diagonal's variable
         coefficient calculation (which is like constant_coefficient=0). */
      Ai = hypre_CCBoxIndexRank(A_dbox, start);

      center_offd  = 0.0;
      P0 = 0.0;
      P1 = 0.0;
      mrk0_offd = 0;
      mrk1_offd = 0;

      for (si = 0; si < stencil_size; si++)
      {
         if ( si != stencil_diag_entry )
         {
            Ap = hypre_StructMatrixBoxData(A, i, si);
            Astenc = hypre_IndexD(stencil_shape[si], cdir);

            if (Astenc == 0)
            {
               center_offd += Ap[Ai];
            }
            else if (Astenc == Pstenc0)
            {
               P0 -= Ap[Ai];
            }
            else if (Astenc == Pstenc1)
            {
               P1 -= Ap[Ai];
            }

            if (si == si0 && Ap[Ai] == 0.0)
            {
               mrk0_offd++;
            }
            if (si == si1 && Ap[Ai] == 0.0)
            {
               mrk1_offd++;
            }
         }
      }

      si = stencil_diag_entry;
      hypre_BoxLoop2Begin(hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         Pp0[Pi] = P0;
         Pp1[Pi] = P1;
         center = center_offd;
         mrk0 = mrk0_offd;
         mrk1 = mrk1_offd;

         Ap = hypre_StructMatrixBoxData(A, i, si);
         Astenc = hypre_IndexD(stencil_shape[si], cdir);
         hypre_assert( Astenc == 0 );
         center += Ap[Ai];

         if (si == si0 && Ap[Ai] == 0.0)
         {
            mrk0++;
         }
         if (si == si1 && Ap[Ai] == 0.0)
         {
            mrk1++;
         }

         if (!center)
         {
            warning_cnt++;
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            Pp0[Pi] /= center;
            Pp1[Pi] /= center;
         }

         /*----------------------------------------------
          * Set interpolation weight to zero, if stencil
          * entry in same direction is zero. Prevents
          * interpolation and operator stencils reaching
          * outside domain.
          *----------------------------------------------*/
         if (mrk0 != 0)
         {
            Pp0[Pi] = 0.0;
         }
         if (mrk1 != 0)
         {
            Pp1[Pi] = 0.0;
         }

      }
      hypre_BoxLoop2End(Ai, Pi);
   }

   if (warning_cnt)
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return hypre_error_flag;
}

#endif
